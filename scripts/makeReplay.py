""" makeReplay.py
Converts a saved trajectory log into a self-contained 3D replay HTML file.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

from landingSim.config import loadConfig
from landingSim.outputs import findLatest
from landingSim.dynamics.trajectory import Trajectory
from landingSim.viewer.export import writeReplay


def main():
    parser = argparse.ArgumentParser(description="Build the 3D replay page from a trajectory .npz file.")
    parser.add_argument("log", nargs="?", type=pathlib.Path, default=findLatest(repoRoot, "landing.npz"))
    parser.add_argument("--out", type=pathlib.Path, default=None, help="defaults to the log name with Replay.html")
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml")
    args = parser.parse_args()

    out = args.out if args.out is not None else args.log.with_name(args.log.stem + "Replay.html")
    path = writeReplay(Trajectory.load(args.log), loadConfig(args.config), out)
    print(f"wrote {path}\nopen it in a browser (double-click, or: open {path})")


if __name__ == "__main__":
    main()
