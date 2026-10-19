""" runMonteCarlo.py
Flies the baseline controller through perturbed scenarios, prints the pass rates, and plots the landing scatter.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt
import numpy as np

from landingSim.analysis.monteCarlo import runMonteCarlo, summarize
from landingSim.config import loadConfig
from landingSim.outputs import outputDir
from landingSim.gnc.controller import LandingController

outcomeColors = {"succeeded": "tab:green", "softFailed": "tab:orange", "hardFailed": "tab:red"}


def printProgress(i, n, result):
    print(f"  episode {i:3d}/{n}: {result.outcome:11s} offset {result.finalOffset:7.2f} m  "
          f"touchdown {result.touchdownSpeed:6.2f} m/s  propellant {100 * result.propellantFraction:5.1f} %  ({result.endReason})")


def printSummary(title, summary):
    print(f"\n{title}")
    print(f"  episodes            {summary['episodes']}")
    print(f"  succeeded           {100 * summary['successRate']:5.1f} %")
    print(f"  soft failed         {100 * summary['softFailRate']:5.1f} %")
    print(f"  hard failed         {100 * summary['hardFailRate']:5.1f} %")
    print(f"  strut pass          {100 * summary['strutPassRate']:5.1f} %")
    print(f"  propellant pass     {100 * summary['propellantPassRate']:5.1f} %")
    print(f"  offset pass         {100 * summary['offsetPassRate']:5.1f} %")
    print(f"  load factor pass    {100 * summary['loadFactorPassRate']:5.1f} %  (mean peak {summary['meanPeakLoadFactor']:.2f} g)")
    print(f"  mean offset         {summary['meanOffset']:6.2f} m (landed episodes)")
    print(f"  mean touchdown      {summary['meanTouchdownSpeed']:6.2f} m/s (landed episodes)")


def saveResults(results, path):
    rows = np.array([[r.finalOffset, r.touchdownSpeed, r.propellantFraction, r.peakStrutStress, r.flightTime,
                      c.environment.wind.meanSpeed, c.spacecraft.engine.maxThrust, c.spacecraft.mass.dryMass,
                      c.spacecraft.mass.cgHeightDry, c.mission.initialHorizontalOffset] for c, r in results])
    outcomes = np.array([r.outcome for _, r in results])
    np.savez(path, rows=rows, outcomes=outcomes,
             columns=np.array(["finalOffset", "touchdownSpeed", "propellantFraction", "peakStrutStress", "flightTime",
                               "windMeanSpeed", "maxThrust", "dryMass", "cgHeightDry", "initialHorizontalOffset"]))


def plotScatter(results, path, title):
    fig, (axWind, axOffset) = plt.subplots(1, 2, figsize=(12, 5))
    for outcome, color in outcomeColors.items():
        picked = [(c, r) for c, r in results if r.outcome == outcome]
        if not picked:
            continue
        axWind.scatter([c.environment.wind.meanSpeed for c, _ in picked], [max(r.finalOffset, 0.01) for _, r in picked],
                       color=color, label=outcome, alpha=0.8)
        axOffset.scatter([r.touchdownSpeed for _, r in picked], [100 * r.propellantFraction for _, r in picked],
                         color=color, label=outcome, alpha=0.8)
    axWind.axhline(0.5, color="k", linestyle="--", linewidth=1, label="0.5 m criterion")
    axWind.set_xlabel("wind mean speed [m/s]"); axWind.set_ylabel("final offset [m]"); axWind.set_yscale("log")
    axWind.legend()
    axOffset.axhline(5.0, color="k", linestyle="--", linewidth=1, label="5 % criterion")
    axOffset.set_xlabel("touchdown speed [m/s]"); axOffset.set_ylabel("propellant left [%]")
    axOffset.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Baseline Monte Carlo over perturbed scenarios.")
    parser.add_argument("--episodes", type=int, default=None, help="override [monteCarlo] episodes")
    parser.add_argument("--seed", type=int, default=None, help="override [monteCarlo] seed")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    results = runMonteCarlo(cfg, LandingController, episodes=args.episodes, seed=args.seed, dt=args.dt, progress=printProgress)
    summary = summarize(results)
    printSummary("Baseline controller", summary)

    resultsDir = outputDir(repoRoot)
    saveResults(results, resultsDir / "monteCarlo.npz")
    plotScatter(results, resultsDir / "monteCarlo.png", f"Baseline Monte Carlo, {summary['episodes']} episodes, "
                f"{100 * summary['successRate']:.0f} % succeeded")
    plt.show()


if __name__ == "__main__":
    main()
