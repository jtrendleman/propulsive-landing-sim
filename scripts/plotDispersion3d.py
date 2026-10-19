""" plotDispersion3d.py
Flies a Monte Carlo batch, keeps the trajectories, and draws the landing-dispersion surface with the trajectory bundle.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt
import numpy as np

from landingSim.analysis.monteCarlo import perturbConfig, runEpisode
from landingSim.analysis.plots3d import plotDispersion3d
from landingSim.config import loadConfig
from landingSim.outputs import outputDir
from landingSim.gnc.controller import LandingController


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo dispersion figure.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run", default=None, help="fly a trained run instead of the baseline: run folder, model zip, or 'latest'")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="default: outputs/<version folder>/dispersion3d.png, or the run folder with --run")
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    seed = cfg.monteCarlo.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    if args.run is not None:
        from landingSim.rl.residualController import ResidualController
        from landingSim.rl.runs import loadModel, resolveModel
        runDir, modelPath = resolveModel(repoRoot, None if args.run == "latest" else args.run)
        model = loadModel(modelPath, runDir)
        args.out = runDir / "dispersion3d.png" if args.out is None else args.out
        controllerFactory = lambda c: ResidualController(c, model)
        title = "Monte Carlo landing dispersion, baseline + learned residual"
    else:
        controllerFactory = LandingController
        title = "Monte Carlo landing dispersion, baseline controller"

    if args.out is None:
        args.out = outputDir(repoRoot) / "dispersion3d.png"
    trajectories, results = [], []
    for i in range(args.episodes):
        perturbed = perturbConfig(cfg, rng)
        traj, result = runEpisode(perturbed, controllerFactory(cfg))
        trajectories.append(traj)
        results.append(result)
        print(f"  episode {i + 1:3d}/{args.episodes}: {result.outcome:11s} offset {result.finalOffset:7.2f} m", flush=True)

    plotDispersion3d(trajectories, results, cfg, args.out, title=title)
    print(f"wrote {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
