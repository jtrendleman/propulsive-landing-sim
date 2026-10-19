""" testSimulation.py
Episode loop: termination on touchdown and timeout, trajectory logging, save/load.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.simulation import Simulation, zeroCommands
from landingSim.dynamics.state import Commands, State
from landingSim.dynamics.trajectory import Trajectory

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
dt = 0.01


def restState(altitude):
    return State(
        position=np.array([0.0, 0.0, altitude]),
        velocity=np.zeros(3),
        quaternion=attitude.identity.copy(),
        omega=np.zeros(3),
        propellantMass=cfg.spacecraft.propellant.initialMass,
    )


def testDropEndsOnTouchdownAtClosedFormTime():
    h0, g = 1000.0, cfg.environment.gravity
    traj = Simulation(cfg, producers=[], dt=dt).run(restState(h0))
    assert traj.endReason == "touchdown"
    assert traj.states[-1, 2] <= 0.0
    assert traj.states[-2, 2] > 0.0
    assert abs(traj.time[-1] - np.sqrt(2 * h0 / g)) <= dt


def testTrajectoryShapesAndTimeStep():
    traj = Simulation(cfg, producers=[], dt=dt).run(restState(100.0))
    n = len(traj.time)
    assert traj.states.shape == (n, 22)
    assert traj.commands.shape == (n, 8) and traj.achieved.shape == (n, 8)
    assert traj.time[0] == 0.0
    assert np.allclose(np.diff(traj.time), dt)


def testTimeoutWhenMaxTimeTooShort():
    shortMission = dataclasses.replace(cfg.mission, maxTime=0.5)
    shortCfg = dataclasses.replace(cfg, mission=shortMission)
    traj = Simulation(shortCfg, producers=[], dt=dt).run(restState(1000.0))
    assert traj.endReason == "timeout"
    assert traj.time[-1] == pytest.approx(0.5)
    assert traj.states[-1, 2] > 0.0


def testControllerCommandsAreRecorded():
    def halfThrottle(state, t):
        return Commands(throttle=0.5, gimbal=np.array([0.01, -0.02]))

    traj = Simulation(cfg, producers=[], dt=dt).run(restState(50.0), controller=halfThrottle)
    assert np.allclose(traj.commands[:, 0], 0.5)
    assert np.allclose(traj.commands[:, 1], 0.01)
    assert np.allclose(traj.commands[:, 2], -0.02)
    assert np.allclose(traj.commands[:, 3:8], 0.0)                 # rcs and fins default to zero
    zero = zeroCommands(restState(50.0), 0.0)
    assert zero.throttle == 0.0 and np.allclose(zero.gimbal, 0.0)


def testStateAccessorMatchesRow():
    traj = Simulation(cfg, producers=[], dt=dt).run(restState(50.0))
    mid = len(traj.time) // 2
    s = traj.state(mid)
    assert isinstance(s, State)
    assert np.allclose(s.position, traj.states[mid, 0:3])
    assert np.allclose(s.velocity, traj.states[mid, 3:6])


def testSaveLoadRoundTrip(tmp_path):
    traj = Simulation(cfg, producers=[], dt=dt).run(restState(50.0))
    path = tmp_path / "drop.npz"
    traj.save(path)
    back = Trajectory.load(path)
    assert np.allclose(back.time, traj.time)
    assert np.allclose(back.states, traj.states)
    assert np.allclose(back.commands, traj.commands)
    assert back.endReason == traj.endReason


def testContactSubsteppingKeepsLogSpacing():
    from landingSim.vehicle.landingGear import LandingGear
    from landingSim.dynamics.terminators import LandingTerminator
    gear = LandingGear(cfg.spacecraft)
    probe = restState(0.0)
    from landingSim.vehicle.mass import massProperties
    cgHeight = massProperties(probe, cfg.spacecraft).cgHeight
    start = State(np.array([0.0, 0.0, cgHeight + 0.5]), np.zeros(3), probe.quaternion, np.zeros(3), probe.propellantMass)
    traj = Simulation(cfg, producers=[gear], dt=dt, terminator=LandingTerminator(cfg.spacecraft, gear), contactSubsteps=4).run(start)
    assert traj.endReason == "landed"
    assert np.allclose(np.diff(traj.time), dt)
