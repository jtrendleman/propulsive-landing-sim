""" runLanding.py
Runs the full closed-loop landing from the mission start, logs it, prints the outcome, and plots the descent.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import argparse
import pathlib
import sys

repoRoot = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repoRoot))

import matplotlib.pyplot as plt
import numpy as np

from landingSim.config import loadConfig
from landingSim.outputs import outputDir
from landingSim.core import attitude
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import State, initialState
from landingSim.dynamics.terminators import LandingTerminator
from landingSim.gnc.controller import LandingController
from landingSim.vehicle.aero import Drag
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties


def main():
    parser = argparse.ArgumentParser(description="One closed-loop landing from the mission start.")
    parser.add_argument("--config", type=pathlib.Path, default=repoRoot / "config" / "earthBooster.toml", help="scenario file")
    parser.add_argument("--tag", type=str, default="", help="suffix for the output names, e.g. KscExtreme -> landingKscExtreme.npz")
    args = parser.parse_args()
    cfg = loadConfig(args.config)

    # Mission initialAltitude is the feet altitude; State.position is the CG.
    start = initialState(cfg.mission, cfg.spacecraft)
    cgHeight0 = massProperties(start, cfg.spacecraft).cgHeight
    start = State(start.position + np.array([0.0, 0.0, cgHeight0]), start.velocity, start.quaternion,
                  start.omega, start.propellantMass)

    gear = LandingGear(cfg.spacecraft)
    producers = [Engine(cfg.spacecraft), gear, Drag(cfg.spacecraft, cfg.environment)]
    sim = Simulation(cfg, producers, dt=0.01, terminator=LandingTerminator(cfg.spacecraft, gear))
    traj = sim.run(start, controller=LandingController(cfg))

    resultsDir = outputDir(repoRoot)
    traj.save(resultsDir / f"landing{args.tag}.npz")

    n = len(traj.time)
    cgHeights = np.array([massProperties(traj.state(i), cfg.spacecraft).cgHeight for i in range(n)])
    feetHeight = traj.states[:, 2] - cgHeights
    tilt = np.array([np.degrees(np.arccos(np.clip(attitude.rotateBodyToWorld(traj.state(i).quaternion, np.array([0.0, 0.0, 1.0]))[2], -1, 1)))
                     for i in range(n)])

    touchdownRows = np.where(feetHeight <= 0.0)[0]
    touchdownSpeed = traj.states[touchdownRows[0], 5] if len(touchdownRows) else float("nan")
    final = traj.state(n - 1)
    offset = np.linalg.norm(final.position[0:2])
    propellantFraction = final.propellantMass / cfg.spacecraft.propellant.initialMass

    print(f"endReason          = {traj.endReason}")
    print(f"flight time        = {traj.time[-1]:.2f} s")
    print(f"touchdown speed    = {touchdownSpeed:.2f} m/s")
    print(f"final offset       = {offset:.3f} m")
    print(f"final tilt         = {tilt[-1]:.2f} deg")
    print(f"propellant left    = {100 * propellantFraction:.1f} %")

    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 9))
    axes[0, 0].plot(traj.time, feetHeight); axes[0, 0].set_ylabel("feet altitude [m]")
    axes[1, 0].plot(traj.time, traj.states[:, 5]); axes[1, 0].set_ylabel("vertical speed [m/s]")
    axes[2, 0].plot(traj.time, tilt); axes[2, 0].set_ylabel("tilt [deg]"); axes[2, 0].set_xlabel("time [s]")
    axes[0, 1].plot(traj.time, traj.commands[:, 0]); axes[0, 1].set_ylabel("throttle")
    axes[1, 1].plot(traj.time, np.degrees(traj.commands[:, 1]), label="gx")
    axes[1, 1].plot(traj.time, np.degrees(traj.commands[:, 2]), label="gy")
    axes[1, 1].set_ylabel("gimbal [deg]"); axes[1, 1].legend()
    axes[2, 1].plot(traj.time, traj.states[:, 13]); axes[2, 1].set_ylabel("propellant [kg]"); axes[2, 1].set_xlabel("time [s]")
    fig.suptitle(f"Closed-loop landing, ended by {traj.endReason} at t = {traj.time[-1]:.2f} s")
    fig.tight_layout()
    fig.savefig(resultsDir / f"landing{args.tag}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
