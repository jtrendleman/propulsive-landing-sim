""" testEngine.py
Engine producer: thrust magnitude, gimbal direction and torque sign, mass flow, cutoffs, hover.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude, constants
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import Commands, State
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
eng = cfg.spacecraft.engine
maxTotalThrust = eng.maxThrust * eng.count


def upright(altitude=500.0, propellant=None):
    return State(
        position=np.array([0.0, 0.0, altitude]),
        velocity=np.zeros(3),
        quaternion=attitude.identity.copy(),
        omega=np.zeros(3),
        propellantMass=cfg.spacecraft.propellant.initialMass if propellant is None else propellant,
    )


def cmd(throttle, gx=0.0, gy=0.0):
    return Commands(throttle=throttle, gimbal=np.array([gx, gy]))


def testFullThrottleUprightPushesStraightUp():
    force, torque, massFlow = Engine(cfg.spacecraft)(upright(), cmd(1.0), 0.0)
    assert np.allclose(force, [0.0, 0.0, maxTotalThrust])
    assert np.allclose(torque, 0.0)
    assert massFlow == pytest.approx(maxTotalThrust / (eng.isp * constants.g0))


def testBelowMinThrottleIsOff():
    force, torque, massFlow = Engine(cfg.spacecraft)(upright(), cmd(eng.minThrottle * 0.5), 0.0)
    assert np.allclose(force, 0.0) and np.allclose(torque, 0.0) and massFlow == 0.0


def testEmptyTankIsOff():
    force, torque, massFlow = Engine(cfg.spacecraft)(upright(propellant=0.0), cmd(1.0), 0.0)
    assert np.allclose(force, 0.0) and massFlow == 0.0


def testGimbalPlusXMakesNegativePitchTorqueOnly():
    force, torque, _ = Engine(cfg.spacecraft)(upright(), cmd(1.0, gx=0.05), 0.0)
    assert force[0] > 0.0                       # thrust tilts toward +x
    assert torque[1] < 0.0                      # lever arm below CG -> negative y torque
    assert np.isclose(torque[0], 0.0) and np.isclose(torque[2], 0.0)
    assert np.linalg.norm(force) == pytest.approx(maxTotalThrust)


def testGimbalIsClampedToMaxAngle():
    tooFar = eng.maxGimbalAngle * 3
    force, _, _ = Engine(cfg.spacecraft)(upright(), cmd(1.0, gx=tooFar), 0.0)
    angle = np.arctan2(force[0], force[2])
    assert angle == pytest.approx(eng.maxGimbalAngle)


def testForceRotatesWithVehicle():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), np.pi / 2)   # body +z now along world +x
    s = upright()
    tilted = State(s.position, s.velocity, q, s.omega, s.propellantMass)
    force, _, _ = Engine(cfg.spacecraft)(tilted, cmd(1.0), 0.0)
    assert np.allclose(force, [maxTotalThrust, 0.0, 0.0], atol=1e-6)


def testCannotHoverAtTheThrottleFloor():
    # The defining physics of a Falcon 9-class landing: even at minimum throttle the engine out-lifts the vehicle,
    # and the throttle that would match its weight is below the floor, where the engine is off.
    import dataclasses
    total = massProperties(upright(), cfg.spacecraft).total
    floorThrust = eng.minThrottle * maxTotalThrust
    assert floorThrust > total * cfg.environment.gravity
    oneSecond = dataclasses.replace(cfg, mission=dataclasses.replace(cfg.mission, maxTime=1.0))
    atFloor = Simulation(oneSecond, producers=[Engine(cfg.spacecraft)], dt=0.01, actuators=False).run(upright(), controller=lambda s, t: cmd(eng.minThrottle))
    assert atFloor.states[-1, 5] > 0.0                       # climbing
    weightMatching = total * cfg.environment.gravity / maxTotalThrust
    assert weightMatching < eng.minThrottle
    belowFloor = Simulation(oneSecond, producers=[Engine(cfg.spacecraft)], dt=0.01, actuators=False).run(upright(), controller=lambda s, t: cmd(weightMatching))
    assert belowFloor.states[-1, 5] == pytest.approx(-cfg.environment.gravity, rel=0.02)   # engine off: free fall


def testThrustGivesTheExpectedAcceleration():
    import dataclasses
    oneSecond = dataclasses.replace(cfg, mission=dataclasses.replace(cfg.mission, maxTime=1.0))
    traj = Simulation(oneSecond, producers=[Engine(cfg.spacecraft)], dt=0.01, actuators=False).run(upright(), controller=lambda s, t: cmd(0.8))
    total = massProperties(upright(), cfg.spacecraft).total
    expected = 0.8 * maxTotalThrust / total - cfg.environment.gravity     # mass falls ~0.8 % in a second, so allow 2 %
    assert traj.states[-1, 5] == pytest.approx(expected * traj.time[-1], rel=0.02)


def testPropellantDropsAtMassFlowRate():
    import dataclasses
    oneSecond = dataclasses.replace(cfg, mission=dataclasses.replace(cfg.mission, maxTime=1.0))
    traj = Simulation(oneSecond, producers=[Engine(cfg.spacecraft)], dt=0.01, actuators=False).run(upright(), controller=lambda s, t: cmd(1.0))
    expectedFlow = maxTotalThrust / (eng.isp * constants.g0)
    used = traj.states[0, 13] - traj.states[-1, 13]
    assert used == pytest.approx(expectedFlow * traj.time[-1], rel=1e-6)
