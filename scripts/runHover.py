""" runHover.py
Hovers the vehicle with the engine at the weight-matching throttle, logs it, and plots altitude and speed.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from landingSim.config import loadConfig
from landingSim.outputs import outputDir
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import Commands, initialState
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.mass import massProperties


def main():
    cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
    cfg = dataclasses.replace(cfg, mission=dataclasses.replace(cfg.mission, maxTime=20.0, initialVerticalSpeed=0.0))
    eng = cfg.spacecraft.engine
    maxTotalThrust = eng.maxThrust * eng.count

    def hoverController(state, t):
        total = massProperties(state, cfg.spacecraft).total
        return Commands(throttle=total * cfg.environment.gravity / maxTotalThrust, gimbal=np.zeros(2))

    sim = Simulation(cfg, producers=[Engine(cfg.spacecraft)], dt=0.01)
    traj = sim.run(initialState(cfg.mission, cfg.spacecraft), controller=hoverController)

    resultsDir = outputDir(repoRoot)
    traj.save(resultsDir / "hover.npz")

    fig, (axAlt, axSpeed, axProp) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axAlt.plot(traj.time, traj.states[:, 2])
    axAlt.set_ylabel("altitude [m]")
    axAlt.set_title(f"Hover, ended by {traj.endReason} at t = {traj.time[-1]:.2f} s")
    axSpeed.plot(traj.time, traj.states[:, 5])
    axSpeed.set_ylabel("vertical speed [m/s]")
    axProp.plot(traj.time, traj.states[:, 13])
    axProp.set_ylabel("propellant [kg]")
    axProp.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(resultsDir / "hover.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
