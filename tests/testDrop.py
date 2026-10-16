""" testDrop.py
Point-mass drop: state packing, gravity field, mass properties, RK4 against
closed-form free fall.
Author: Jack Rendleman Date: 10/15/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.integrator import rk4Step, stateDerivative, step
from landingSim.dynamics.state import Commands, State, initialState
from landingSim.environment.gravity import gravityAcceleration
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
noCommands = Commands(throttle=0.0, gimbal=np.zeros(2))


def restState(altitude):
    return State(
        position=np.array([0.0, 0.0, altitude]),
        velocity=np.zeros(3),
        quaternion=attitude.identity.copy(),
        omega=np.zeros(3),
        propellantMass=cfg.spacecraft.propellant.initialMass,
    )


def testArrayRoundTrip():
    s = restState(1000.0)
    back = State.fromArray(s.toArray())
    assert s.toArray().shape == (14,)
    assert np.allclose(back.position, s.position)
    assert np.allclose(back.velocity, s.velocity)
    assert np.allclose(back.quaternion, s.quaternion)
    assert np.allclose(back.omega, s.omega)
    assert back.propellantMass == s.propellantMass


def testFromArrayRenormalizesQuaternion():
    arr = restState(1000.0).toArray()
    arr[6:10] = [2.0, 0.0, 0.0, 0.0]
    assert np.allclose(State.fromArray(arr).quaternion, attitude.identity)


def testInitialStateFromConfig():
    s = initialState(cfg.mission, cfg.spacecraft)
    assert np.allclose(s.position, [cfg.mission.initialHorizontalOffset, 0.0, cfg.mission.initialAltitude])
    assert np.allclose(s.velocity, [0.0, 0.0, cfg.mission.initialVerticalSpeed])
    assert np.allclose(s.quaternion, attitude.identity)
    assert np.allclose(s.omega, 0.0)
    assert s.propellantMass == cfg.spacecraft.propellant.initialMass


def testGravityPointsDown():
    g = gravityAcceleration(cfg.environment, np.array([0.0, 0.0, 500.0]))
    assert np.allclose(g, [0.0, 0.0, -cfg.environment.gravity])


def testMassPropertiesTotal():
    props = massProperties(restState(1000.0), cfg.spacecraft)
    assert props.total == pytest.approx(cfg.spacecraft.mass.dryMass + cfg.spacecraft.propellant.initialMass)
    assert props.inertia.shape == (3,)


def testRk4OnExponential():
    # y' = y, y(0) = 1  =>  y(1) = e. RK4 at dt=0.01 should be good to ~1e-9.
    y, t, dt = np.array([1.0]), 0.0, 0.01
    for _ in range(100):
        y = rk4Step(lambda time, state: state, y, t, dt)
        t += dt
    assert np.isclose(y[0], np.e, atol=1e-8)


def testDerivativeOfRestStateIsPureGravity():
    d = stateDerivative(restState(1000.0), [], noCommands, 0.0, cfg.spacecraft, cfg.environment)
    assert d.shape == (14,)
    assert np.allclose(d[0:3], 0.0)                              # dPosition = velocity = 0
    assert np.allclose(d[3:6], [0.0, 0.0, -cfg.environment.gravity])  # dVelocity = g
    assert np.allclose(d[6:], 0.0)                               # no rotation, no mass flow


def testFreeFallMatchesClosedForm():
    g, h0, dt = cfg.environment.gravity, 1000.0, 0.01
    s, t = restState(h0), 0.0
    for _ in range(1000):  # 10 s
        s = step(s, [], noCommands, t, dt, cfg.spacecraft, cfg.environment)
        t += dt
    assert np.isclose(s.position[2], h0 - 0.5 * g * t**2, atol=1e-6)
    assert np.isclose(s.velocity[2], -g * t, atol=1e-6)
    assert np.allclose(s.quaternion, attitude.identity)
    assert s.propellantMass == pytest.approx(cfg.spacecraft.propellant.initialMass)


def testTouchdownTimeWithinOneMillisecond():
    g, h0, dt = cfg.environment.gravity, 1000.0, 0.01
    s, t = restState(h0), 0.0
    while s.position[2] > 0.0:
        previous, tPrevious = s, t
        s = step(s, [], noCommands, t, dt, cfg.spacecraft, cfg.environment)
        t += dt
    # Linear interpolation of the zero crossing inside the last step.
    fraction = previous.position[2] / (previous.position[2] - s.position[2])
    touchdown = tPrevious + fraction * dt
    assert abs(touchdown - np.sqrt(2 * h0 / g)) < 1e-3


def testCgEqualsDryCgWhenTankEmpty():
    s = restState(1000.0)
    empty = State(s.position, s.velocity, s.quaternion, s.omega, propellantMass=0.0)
    assert massProperties(empty, cfg.spacecraft).cgHeight == pytest.approx(cfg.spacecraft.mass.cgHeightDry)


def testCgAtInitialMassIsWeightedAverage():
    sc = cfg.spacecraft
    props = massProperties(restState(1000.0), sc)
    liquidHeight = sc.propellant.initialMass / (sc.propellant.density * np.pi * sc.propellant.tankRadius**2)
    cgPropellant = sc.propellant.tankBottomHeight + liquidHeight / 2
    expected = (sc.mass.dryMass * sc.mass.cgHeightDry + sc.propellant.initialMass * cgPropellant) / props.total
    assert props.cgHeight == pytest.approx(expected)



def testInertiaEqualsDryInertiaWhenTankEmpty():
    s = restState(1000.0)
    empty = State(s.position, s.velocity, s.quaternion, s.omega, propellantMass=0.0)
    assert np.allclose(massProperties(empty, cfg.spacecraft).inertia, cfg.spacecraft.mass.inertiaDry)


def testInertiaWithPropellantExceedsDryAndIsSymmetric():
    props = massProperties(restState(1000.0), cfg.spacecraft)
    dry = np.array(cfg.spacecraft.mass.inertiaDry)
    assert np.all(props.inertia > dry)            # added mass can only add inertia
    assert props.inertia[0] == pytest.approx(props.inertia[1])  # axisymmetric rocket
