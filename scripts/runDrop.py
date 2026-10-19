""" runDrop.py
Drops the vehicle with no thrust from the mission initial state, logs it, and plots altitude and speed.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt

from landingSim.config import loadConfig
from landingSim.outputs import outputDir
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import initialState


def main():
    cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
    sim = Simulation(cfg, producers=[], dt=0.01)
    traj = sim.run(initialState(cfg.mission, cfg.spacecraft))

    resultsDir = outputDir(repoRoot)
    traj.save(resultsDir / "drop.npz")

    altitude = traj.states[:, 2]
    verticalSpeed = traj.states[:, 5]

    fig, (axAlt, axSpeed) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axAlt.plot(traj.time, altitude)
    axAlt.set_ylabel("altitude [m]")
    axAlt.set_title(f"Free drop, ended by {traj.endReason} at t = {traj.time[-1]:.2f} s")
    axSpeed.plot(traj.time, verticalSpeed)
    axSpeed.set_ylabel("vertical speed [m/s]")
    axSpeed.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(resultsDir / "drop.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
