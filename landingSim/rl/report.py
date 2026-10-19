""" report.py
Writes the visual record of a training run into its folder: training curve, Monte Carlo comparison, dispersion, nominal flight.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

Everything here reads a saved model and the scenario config, so it can be re-run on any run folder.
"""

import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from landingSim.analysis.criteria import evaluate
from landingSim.analysis.monteCarlo import buildEpisode, missionStart, perturbConfig, runEpisode, runMonteCarlo, summarize
from landingSim.analysis.plots3d import animateLanding3d, plotDispersion3d, plotLanding3d, plotLandingMontage
from landingSim.core import attitude
from landingSim.gnc.controller import LandingController
from landingSim.rl.residualController import ResidualController
from landingSim.rl.runs import loadModel, modelFile
from landingSim.vehicle.mass import massProperties

outcomeColors = {"succeeded": "tab:green", "softFailed": "tab:orange", "hardFailed": "tab:red"}


class ResidualControllerFactory:
    """Picklable controller factory for worker pools: loads the model once per process."""

    def __init__(self, modelPath, runDir=None):
        self.modelPath = str(modelPath)
        self.runDir = None if runDir is None else str(runDir)
        self._model = None

    def __call__(self, config):
        if self._model is None:
            self._model = loadModel(self.modelPath, self.runDir)
        return ResidualController(config, self._model)

    def __getstate__(self):
        return {"modelPath": self.modelPath, "runDir": self.runDir, "_model": None}


def plotTrainingCurve(episodes, path, window=50):
    """episodes: list of (timestep, outcome, episodeReward)."""
    if not episodes:
        return None
    steps = np.array([e[0] for e in episodes], float)
    success = np.array([e[1] == "succeeded" for e in episodes], float)
    hard = np.array([e[1] == "hardFailed" for e in episodes], float)
    reward = np.array([e[2] for e in episodes], float)
    kernel = np.ones(min(window, len(episodes))) / min(window, len(episodes))
    roll = lambda x: np.convolve(x, kernel, mode="valid")
    xs = steps[len(steps) - len(roll(success)):]

    fig, (axRate, axReward) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    axRate.plot(xs, 100 * roll(success), color="tab:green", label=f"succeeded (rolling {len(kernel)})")
    axRate.plot(xs, 100 * roll(hard), color="tab:red", label="hard failed")
    axRate.axhline(95, color="k", linestyle="--", linewidth=1, label="95 % requirement")
    axRate.set_ylabel("rate (%)"); axRate.set_ylim(0, 102); axRate.grid(alpha=0.4); axRate.legend(fontsize=8, loc="lower right")
    axReward.plot(steps, reward, ".", color="0.7", markersize=3, label="episode reward")
    axReward.plot(xs, roll(reward), color="tab:blue", label="rolling mean")
    axReward.set_ylabel("episode reward (scaled)"); axReward.set_xlabel("timesteps"); axReward.grid(alpha=0.4); axReward.legend(fontsize=8)
    fig.suptitle("Training: exploration noise on, so rates sit below the deterministic policy's")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plotEvalHistory(runDir, path):
    """Held-out deterministic success and propellant against timesteps, from evalHistory.json."""
    history = json.loads((pathlib.Path(runDir) / "evalHistory.json").read_text(encoding="utf-8"))
    if not history:
        return None
    steps = [h["timesteps"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, [100 * h["successRate"] for h in history], "o-", color="tab:green", label="held-out succeeded (deterministic)")
    ax.plot(steps, [100 * h["meanPropellantUsed"] for h in history], "s-", color="tab:blue", label="propellant used")
    ax.axhline(95, color="k", linestyle="--", linewidth=1, label="95 % requirement")
    ax.set_xlabel("timesteps"); ax.set_ylabel("%"); ax.set_ylim(0, 102); ax.grid(alpha=0.4); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Held-out evaluation during training (best model is kept)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plotTrainingCurveFromLog(logPath, path):
    """Fallback for runs trained before per-episode rewards were recorded: parse the rolling-window lines of trainLog.txt."""
    import re
    rows = []
    for line in pathlib.Path(logPath).read_text(encoding="utf-8").splitlines():
        m = re.search(r"(\d+) steps\s+(\d+) episodes\s+last (\d+):\s+([\d.]+) % succeeded\s+([\d.]+) % hard failed", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(4)), float(m.group(5))))
    if not rows:
        return None
    steps, success, hard = (np.array(c) for c in zip(*rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, success, "o-", color="tab:green", label="succeeded (rolling 50)")
    ax.plot(steps, hard, "s-", color="tab:red", label="hard failed")
    ax.axhline(95, color="k", linestyle="--", linewidth=1, label="95 % requirement")
    ax.set_xlabel("timesteps"); ax.set_ylabel("rate (%)"); ax.set_ylim(0, 102); ax.grid(alpha=0.4); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Training (from trainLog.txt): exploration noise on, so rates sit below the deterministic policy's")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def timeHistory(trajectory, config):
    n = len(trajectory.time)
    cg = np.array([massProperties(trajectory.state(i), config.spacecraft).cgHeight for i in range(n)])
    tilt = np.degrees(np.arccos(np.clip([attitude.rotateBodyToWorld(trajectory.state(i).quaternion, np.array([0.0, 0.0, 1.0]))[2] for i in range(n)], -1, 1)))
    return {
        "t": trajectory.time, "feet": trajectory.states[:, 2] - cg, "vz": trajectory.states[:, 5],
        "offset": np.linalg.norm(trajectory.states[:, 0:2], axis=1), "throttle": trajectory.commands[:, 0],
        "tilt": tilt, "propellant": 100 * trajectory.states[:, 13] / config.spacecraft.propellant.initialMass,
    }


def plotNominalComparison(baselineTraj, residualTraj, config, path, baseResult, resResult):
    a, b = timeHistory(baselineTraj, config), timeHistory(residualTraj, config)
    panels = [("feet", "feet altitude (m)"), ("vz", "vertical speed (m/s)"), ("offset", "offset from pad (m)"),
              ("throttle", "throttle"), ("tilt", "tilt (deg)"), ("propellant", "propellant (%)")]
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 9))
    for ax, (key, label) in zip(axes.flat, panels):
        ax.plot(a["t"], a[key], color="0.4", label="baseline")
        ax.plot(b["t"], b[key], color="tab:blue", label="baseline + residual")
        ax.set_ylabel(label); ax.grid(alpha=0.4)
    axes[2, 0].set_xlabel("time (s)"); axes[2, 1].set_xlabel("time (s)"); axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Nominal mission. baseline: {baseResult.outcome}, {baseResult.finalOffset:.2f} m, {100 * baseResult.propellantFraction:.0f} % left   |   "
                 f"residual: {resResult.outcome}, {resResult.finalOffset:.2f} m, {100 * resResult.propellantFraction:.0f} % left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plotOutcomeScatter(results, path, title):
    fig, (axWind, axOffset) = plt.subplots(1, 2, figsize=(12, 5))
    for outcome, color in outcomeColors.items():
        picked = [(c, r) for c, r in results if r.outcome == outcome]
        if not picked:
            continue
        axWind.scatter([c.environment.wind.meanSpeed for c, _ in picked], [max(r.finalOffset, 0.01) for _, r in picked], color=color, label=outcome, alpha=0.8)
        axOffset.scatter([r.touchdownSpeed for _, r in picked], [100 * r.propellantFraction for _, r in picked], color=color, label=outcome, alpha=0.8)
    axWind.axhline(0.5, color="k", linestyle="--", linewidth=1, label="0.5 m criterion")
    axWind.set_xlabel("wind speed at 10 m (m/s)"); axWind.set_ylabel("final offset (m)"); axWind.set_yscale("log"); axWind.legend(fontsize=8)
    axOffset.axhline(5.0, color="k", linestyle="--", linewidth=1, label="5 % criterion")
    axOffset.set_xlabel("touchdown speed (m/s)"); axOffset.set_ylabel("propellant left (%)"); axOffset.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def summaryLines(name, s):
    return (f"{name:22s} succeeded {100 * s['successRate']:5.1f} %  soft {100 * s['softFailRate']:5.1f} %  hard {100 * s['hardFailRate']:5.1f} %  "
            f"propellant used {100 * s['meanPropellantUsed']:5.1f} %  mean offset {s['meanOffset']:5.2f} m  mission success {'yes' if s['missionSuccess'] else 'no'}")


def writeRunReport(runDir, config, evaluateEpisodes=50, seed=None, workers=1, dispersionEpisodes=24, log=print):
    """Monte Carlo comparison, dispersion figure, nominal comparison, and summary.txt into runDir."""
    runDir = pathlib.Path(runDir)
    modelPath = runDir / modelFile
    factory = ResidualControllerFactory(modelPath, runDir)
    seed = config.monteCarlo.seed if seed is None else seed
    if not (runDir / "trainingCurve.png").exists() and (runDir / "trainLog.txt").exists():
        plotTrainingCurveFromLog(runDir / "trainLog.txt", runDir / "trainingCurve.png")
    if (runDir / "evalHistory.json").exists():
        plotEvalHistory(runDir, runDir / "evalHistory.png")

    log(f"  Monte Carlo: baseline vs residual, {evaluateEpisodes} draws, seed {seed}")
    baseline = runMonteCarlo(config, LandingController, episodes=evaluateEpisodes, seed=seed, workers=workers)
    residual = runMonteCarlo(config, factory, episodes=evaluateEpisodes, seed=seed, workers=workers)
    baseSummary, resSummary = summarize(baseline), summarize(residual)
    plotOutcomeScatter(baseline, runDir / "monteCarloBaseline.png", f"Baseline, {evaluateEpisodes} draws, {100 * baseSummary['successRate']:.0f} % succeeded")
    plotOutcomeScatter(residual, runDir / "monteCarloResidual.png", f"Baseline + residual, {evaluateEpisodes} draws, {100 * resSummary['successRate']:.0f} % succeeded")

    log(f"  dispersion figure, {dispersionEpisodes} draws")
    rng = np.random.default_rng(seed)
    trajectories, results = [], []
    for _ in range(dispersionEpisodes):
        perturbed = perturbConfig(config, rng)
        traj, result = runEpisode(perturbed, factory(config))
        trajectories.append(traj); results.append(result)
    plotDispersion3d(trajectories, results, config, runDir / "dispersion3d.png", title="Monte Carlo landing dispersion, baseline + residual")
    plt.close("all")

    log("  nominal mission, baseline vs residual")
    baseTraj, baseResult = runEpisode(config, LandingController(config))
    resTraj, resResult = runEpisode(config, factory(config))
    resTraj.save(runDir / "landingResidual.npz")
    plotNominalComparison(baseTraj, resTraj, config, runDir / "nominalComparison.png", baseResult, resResult)
    plotLanding3d(resTraj, config, runDir / "landingResidual3d.png", title="Nominal mission, baseline + residual")
    plotLandingMontage(resTraj, config, runDir / "landingResidual3dMontage.png", title="Nominal mission, baseline + residual")
    plt.close("all")
    log("  video of the policy's nominal flight")
    animateLanding3d(resTraj, config, runDir / "landingResidual3d", fps=30, speed=1.5, dpi=120,
                     title="Nominal mission, baseline + residual")

    text = "\n".join([
        f"run {runDir.name}",
        f"Monte Carlo, {evaluateEpisodes} draws, seed {seed}:",
        "  " + summaryLines("baseline", baseSummary),
        "  " + summaryLines("baseline + residual", resSummary),
        f"nominal mission: baseline {baseResult.outcome}, offset {baseResult.finalOffset:.2f} m, propellant left {100 * baseResult.propellantFraction:.1f} %; "
        f"residual {resResult.outcome}, offset {resResult.finalOffset:.2f} m, propellant left {100 * resResult.propellantFraction:.1f} %",
        "",
    ])
    (runDir / "summary.txt").write_text(text, encoding="utf-8")
    (runDir / "evaluation.json").write_text(json.dumps({"episodes": evaluateEpisodes, "seed": seed, "baseline": baseSummary, "residual": resSummary}, indent=2), encoding="utf-8")
    log(text)
    return baseSummary, resSummary
