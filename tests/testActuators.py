""" testActuators.py
Engine actuator dynamics: ignition delay, throttle lag, gimbal rate limit, cutoff, and their effect on a closed-loop landing.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.dynamics.state import Commands
from landingSim.vehicle.actuators import Actuators

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
eng = cfg.spacecraft.engine


def full(gx=0.0):
    return Commands(throttle=1.0, gimbal=np.array([gx, 0.0]))


def testIgnitionDelayThenMinThrottle():
    act = Actuators(cfg.spacecraft)
    dt = 0.01
    steps = int(round(eng.ignitionDelay / dt))
    for _ in range(steps - 1):
        assert act.step(full(), dt).throttle == 0.0
    assert act.step(full(), dt).throttle == pytest.approx(eng.minThrottle)


def testThrottleRelaxesWithTimeConstant():
    act = Actuators(cfg.spacecraft)
    dt = 0.01
    for _ in range(int(eng.ignitionDelay / dt) + 1):
        act.step(full(), dt)
    for _ in range(int(eng.throttleTimeConstant / dt)):
        achieved = act.step(full(), dt)
    assert 0.6 < achieved.throttle < 0.95                      # one time constant: ~63 % of the way from min to full
    assert achieved.throttle > eng.minThrottle


def testCutoffIsImmediateAndNeedsReignition():
    act = Actuators(cfg.spacecraft)
    dt = 0.01
    for _ in range(100):
        act.step(full(), dt)
    off = act.step(Commands(throttle=0.0, gimbal=np.zeros(2)), dt)
    assert off.throttle == 0.0 and not act.lit
    assert act.step(full(), dt).throttle == 0.0                 # ignition delay starts over


def testGimbalIsRateLimited():
    act = Actuators(cfg.spacecraft)
    dt = 0.01
    achieved = act.step(full(gx=eng.maxGimbalAngle), dt)
    assert achieved.gimbal[0] == pytest.approx(eng.maxGimbalRate * dt)
    for _ in range(200):
        achieved = act.step(full(gx=eng.maxGimbalAngle), dt)
    assert achieved.gimbal[0] == pytest.approx(eng.maxGimbalAngle)


def testSimulationLogsAchievedSeparately():
    from landingSim.analysis.monteCarlo import runEpisode
    from landingSim.gnc.controller import LandingController
    traj, result = runEpisode(cfg, LandingController(cfg))
    assert result.endReason == "landed"
    assert traj.achieved.shape == traj.commands.shape
    assert np.max(traj.commands[:, 0] - traj.achieved[:, 0]) > 0.05     # the lag shows up somewhere in the flight
