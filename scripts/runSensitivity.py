""" runSensitivity.py
Which perturbation drives the failures: Monte Carlo separation plus a one-at-a-time sweep, as a tornado chart.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt

from landingSim.analysis.monteCarlo import runMonteCarlo, summarize
from landingSim.analysis.sensitivity import oneAtATime, plotTornado, separation
from landingSim.config import loadConfig
from landingSim.outputs import outputDir
from landingSim.gnc.controller import LandingController


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of the landing outcome to each perturbation.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sigmas", type=float, default=2.0)
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    results = runMonteCarlo(cfg, LandingController, episodes=args.episodes, seed=args.seed, workers=args.workers)
    summary = summarize(results)
    print(f"{args.episodes} draws: {100 * summary['successRate']:.0f} % succeeded\n")
    print("separation (Cohen's d, failed minus succeeded):")
    rows = separation(results)
    for name, d in rows:
        print(f"  {name:26s} {d:+.2f}")
    print(f"\none at a time at +-{args.sigmas} sigma:")
    oat = oneAtATime(cfg, sigmas=args.sigmas)
    for name, low, high in oat:
        print(f"  {name:40s} low: {low.outcome:10s} {low.finalOffset:6.2f} m {100 * low.propellantFraction:5.1f} %   high: {high.outcome:10s} {high.finalOffset:6.2f} m {100 * high.propellantFraction:5.1f} %")
    out = outputDir(repoRoot) / f"sensitivity{'' if args.config.stem == 'earthBooster' else args.config.stem.replace('earthBooster', '')}.png"
    plotTornado(rows, oat, out, title=f"Sensitivity, {args.config.stem}, {args.episodes} draws")
    print(f"\nwrote {out}")
    plt.show()


if __name__ == "__main__":
    main()
