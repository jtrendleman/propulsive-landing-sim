""" animateLanding3d.py
Renders the 3D trajectory figure as a video (MP4 with ffmpeg, otherwise GIF) from a saved landing log.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

from landingSim.analysis.plots3d import animateLanding3d
from landingSim.config import loadConfig
from landingSim.outputs import findLatest
from landingSim.dynamics.trajectory import Trajectory


def main():
    parser = argparse.ArgumentParser(description="Animate a trajectory .npz file as a 3D video.")
    parser.add_argument("log", nargs="?", type=pathlib.Path, default=findLatest(repoRoot, "landing.npz"))
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=2.0, help="playback speed relative to real time")
    parser.add_argument("--no-orbit", dest="orbit", action="store_false", help="keep the camera fixed")
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    cfg = loadConfig(args.config)
    traj = Trajectory.load(args.log)
    out = animateLanding3d(traj, cfg, args.log.with_name(args.log.stem + "3d"), fps=args.fps, speed=args.speed, orbit=args.orbit, dpi=args.dpi)
    print(f"wrote {out}  ({'mp4' if out.suffix == '.mp4' else 'gif; install ffmpeg for mp4'})")


if __name__ == "__main__":
    main()
