""" testRcs.py
RCS producer torque scaling and the controller's hand-off of roll and leftover torque to the RCS.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.state import Commands, State
from landingSim.gnc.control import ThrustVectorController
from landingSim.vehicle.rcs import Rcs

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
rcsCfg = cfg.spacecraft.rcs
maxTorque = rcsCfg.thrustPerThruster * rcsCfg.momentArm


def state(omega=(0.0, 0.0, 0.0), quaternion=None):
    return State(np.array([0.0, 0.0, 100.0]), np.zeros(3), attitude.identity.copy() if quaternion is None else quaternion,
                 np.array(omega, float), cfg.spacecraft.propellant.initialMass)


def testFullCommandGivesMaxTorqueAndNoForce():
    force, torque, massFlow = Rcs(cfg.spacecraft)(state(), Commands(0.0, np.zeros(2), rcs=np.array([1.0, -0.5, 0.25])), 0.0)
    assert np.allclose(force, 0.0) and massFlow == 0.0
    assert np.allclose(torque, [maxTorque, -0.5 * maxTorque, 0.25 * maxTorque])


def testCommandIsClampedToUnitRange():
    _, torque, _ = Rcs(cfg.spacecraft)(state(), Commands(0.0, np.zeros(2), rcs=np.array([5.0, 0.0, -5.0])), 0.0)
    assert np.allclose(torque, [maxTorque, 0.0, -maxTorque])


def testDefaultCommandsHaveZeroRcs():
    _, torque, _ = Rcs(cfg.spacecraft)(state(), Commands(0.5, np.zeros(2)), 0.0)
    assert np.allclose(torque, 0.0)


def testControllerDampsRollWithRcsOnly():
    commands = ThrustVectorController(cfg.spacecraft)(state(omega=(0.0, 0.0, 0.5)), np.array([0.0, 0.0, 12.0]), 0.0)
    assert commands.rcs[2] < 0.0                               # opposes +z spin
    assert np.allclose(commands.gimbal, 0.0, atol=1e-9)        # gimbal cannot help with roll
    assert np.allclose(commands.rcs[0:2], 0.0, atol=1e-9)


def testControllerLeavesPitchToGimbalWhenUnsaturated():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.01)
    controller = ThrustVectorController(cfg.spacecraft, cfg.guidance.kp, cfg.guidance.kd)   # the gains the booster flies with
    commands = controller(state(quaternion=q), np.array([0.0, 0.0, 12.0]), 0.0)
    assert abs(commands.gimbal[0]) < cfg.spacecraft.engine.maxGimbalAngle
    assert np.allclose(commands.rcs, 0.0, atol=1e-9)


def testControllerHandsSaturatedPitchRemainderToRcs():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.5)   # big lean, gimbal will clamp
    commands = ThrustVectorController(cfg.spacecraft)(state(quaternion=q), np.array([0.0, 0.0, 12.0]), 0.0)
    assert commands.gimbal[0] == pytest.approx(cfg.spacecraft.engine.maxGimbalAngle)
    assert commands.rcs[1] < 0.0                               # same sign as the gimbal's pitch torque


def testEngineOffRoutesAllTorqueToRcs():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.1)
    commands = ThrustVectorController(cfg.spacecraft)(state(quaternion=q), np.array([0.0, 0.0, 1.0]), 0.0, engineOn=False)
    assert commands.throttle == 0.0 and np.allclose(commands.gimbal, 0.0)
    assert commands.rcs[1] < 0.0 and np.isclose(commands.rcs[0], 0.0)
