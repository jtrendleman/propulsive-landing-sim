""" sweepBraking.py
Sweeps the braking deceleration from gentle to hoverslam and plots propellant used and success rate against it.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt
import numpy as np

from landingSim.analysis.monteCarlo import missionSuccessRate
from landingSim.analysis.optimize import sweepReferenceDecel
from landingSim.config import loadConfig
from landingSim.outputs import outputDir


def main():
    parser = argparse.ArgumentParser(description="Fuel and success rate versus braking deceleration.")
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--stop", type=float, default=4.5)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    seed = cfg.monteCarlo.seed if args.seed is None else args.seed
    values = np.arange(args.start, args.stop + 1e-9, args.step)
    eng = cfg.spacecraft.engine
    available = eng.maxThrust * eng.count / (cfg.spacecraft.mass.dryMass + cfg.spacecraft.propellant.initialMass) - cfg.environment.gravity
    print(f"max available deceleration at full tank: {available:.2f} m/s^2")
    print(f"{'decel':>6} {'propellant used':>16} {'succeeded':>10} {'soft':>6} {'hard':>6} {'mission':>8}")

    def report(value, summary):
        print(f"{value:6.2f} {100 * summary['meanPropellantUsed']:15.1f} % {100 * summary['successRate']:9.0f} % "
              f"{100 * summary['softFailRate']:5.0f} % {100 * summary['hardFailRate']:5.0f} % {'yes' if summary['missionSuccess'] else 'no':>8}", flush=True)

    rows = sweepReferenceDecel(cfg, values, args.episodes, seed, args.workers, progress=report)
    decels = np.array([v for v, _ in rows])
    used = np.array([s["meanPropellantUsed"] for _, s in rows])
    success = np.array([s["successRate"] for _, s in rows])
    resultsDir = outputDir(repoRoot)
    np.savez(resultsDir / "brakingSweep.npz", referenceDecel=decels, propellantUsed=used, successRate=success,
             softFailRate=[s["softFailRate"] for _, s in rows], hardFailRate=[s["hardFailRate"] for _, s in rows])

    fig, axFuel = plt.subplots(figsize=(9, 5.5))
    axFuel.plot(decels, 100 * used, "o-", color="#0072bd", label="propellant used")
    axFuel.set_xlabel("braking deceleration (m/s$^2$)")
    axFuel.set_ylabel("propellant used (% of load)", color="#0072bd")
    axFuel.grid(True, alpha=0.4)
    axRate = axFuel.twinx()
    axRate.plot(decels, 100 * success, "s-", color="#d95319", label="success rate")
    axRate.axhline(100 * missionSuccessRate, color="k", linestyle="--", linewidth=1, label=f"{100 * missionSuccessRate:.0f} % mission requirement")
    axRate.set_ylabel("Monte Carlo success rate (%)", color="#d95319")
    axRate.set_ylim(0, 105)
    axFuel.axvline(available, color="0.5", linestyle=":", label="max available at full tank")
    lines = axFuel.get_legend_handles_labels()[0] + axRate.get_legend_handles_labels()[0]
    labels = axFuel.get_legend_handles_labels()[1] + axRate.get_legend_handles_labels()[1]
    axFuel.legend(lines, labels, loc="center left", fontsize=9)
    axFuel.set_title(f"Gentle braking to hoverslam: fuel against reliability ({args.episodes} draws per point)")
    fig.tight_layout()
    fig.savefig(resultsDir / "brakingSweep.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
