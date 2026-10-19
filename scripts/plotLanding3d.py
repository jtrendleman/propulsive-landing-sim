""" plotLanding3d.py
Draws the MATLAB-style 3D trajectory figure and the four-window montage from a saved landing log.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt

from landingSim.analysis.plots3d import plotLanding3d, plotLandingMontage
from landingSim.config import loadConfig
from landingSim.outputs import findLatest
from landingSim.dynamics.trajectory import Trajectory


def main():
    parser = argparse.ArgumentParser(description="3D trajectory figures from a trajectory .npz file.")
    parser.add_argument("log", nargs="?", type=pathlib.Path, default=findLatest(repoRoot, "landing.npz"))
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml")
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    traj = Trajectory.load(args.log)
    stem = args.log.with_suffix("")
    plotLanding3d(traj, cfg, stem.with_name(stem.name + "3d.png"))
    plotLandingMontage(traj, cfg, stem.with_name(stem.name + "3dMontage.png"))
    print(f"wrote {stem.name}3d.png and {stem.name}3dMontage.png in {stem.parent}")
    plt.show()


if __name__ == "__main__":
    main()
