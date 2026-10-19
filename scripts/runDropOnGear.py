""" runDropOnGear.py
Drops the vehicle onto its landing gear from a few metres, logs it, and plots CG height and strut compression.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

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
from landingSim.dynamics.state import State
from landingSim.dynamics.terminators import LandingTerminator
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties


def main():
    cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
    dropHeight = 5.0

    probe = State(np.zeros(3), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    cgHeight = massProperties(probe, cfg.spacecraft).cgHeight
    start = State(np.array([0.0, 0.0, cgHeight + dropHeight]), np.zeros(3), attitude.identity.copy(), np.zeros(3),
                  cfg.spacecraft.propellant.initialMass)

    gear = LandingGear(cfg.spacecraft)
    sim = Simulation(cfg, producers=[gear], dt=0.001, terminator=LandingTerminator(cfg.spacecraft, gear))
    traj = sim.run(start)

    resultsDir = outputDir(repoRoot)
    traj.save(resultsDir / "dropOnGear.npz")

    compression = np.array([gear.maxPenetration(traj.state(i)) for i in range(len(traj.time))])
    print(f"endReason = {traj.endReason}, t = {traj.time[-1]:.2f} s, peak compression = {compression.max():.3f} m "
          f"(stroke {cfg.spacecraft.landingGear.strutStroke} m)")

    fig, (axCg, axComp) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axCg.plot(traj.time, traj.states[:, 2])
    axCg.set_ylabel("CG height [m]")
    axCg.set_title(f"Drop from {dropHeight} m onto gear, ended by {traj.endReason} at t = {traj.time[-1]:.2f} s")
    axComp.plot(traj.time, compression)
    axComp.axhline(cfg.spacecraft.landingGear.strutStroke, color="r", linestyle="--", label="strut stroke")
    axComp.set_ylabel("strut compression [m]")
    axComp.set_xlabel("time [s]")
    axComp.legend()
    fig.tight_layout()
    fig.savefig(resultsDir / "dropOnGear.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
