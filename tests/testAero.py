""" testAero.py
Exponential atmosphere, wind field, and drag force/torque at the center of pressure.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import Commands, State
from landingSim.environment.atmosphere import airDensity
from landingSim.environment.wind import windVelocity
from landingSim.vehicle.aero import Drag
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
aero = cfg.spacecraft.aero
noCommands = Commands(throttle=0.0, gimbal=np.zeros(2))
stillAir = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=0.0))


def upright(altitude, velocity):
    return State(np.array([0.0, 0.0, altitude]), np.array(velocity, float), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)


def testIsaDensityPressureAndTemperature():
    from landingSim.environment.atmosphere import pressure, temperature
    assert airDensity(cfg.environment, 0.0) == pytest.approx(1.225, rel=1e-3)
    assert airDensity(cfg.environment, 1000.0) == pytest.approx(1.112, rel=2e-3)          # U.S. Standard Atmosphere 1976 at 1 km
    assert pressure(cfg.environment, 1000.0) == pytest.approx(89875.0, rel=2e-3)
    assert temperature(cfg.environment, 1000.0) == pytest.approx(281.65)
    assert airDensity(cfg.environment, -5.0) == pytest.approx(airDensity(cfg.environment, 0.0))


def testWindBlowsAlongXAtTheReferenceHeight():
    windy = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=7.0, direction=0.0, directionShearPerKm=0.0))
    assert np.allclose(windVelocity(windy, np.array([0.0, 0.0, windy.wind.referenceHeight]), 0.0), [7.0, 0.0, 0.0])


def testWindGrowsWithAltitudeByThePowerLaw():
    windy = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=5.0, direction=0.0, referenceHeight=10.0, profileExponent=0.14, directionShearPerKm=0.0))
    aloft = windVelocity(windy, np.array([0.0, 0.0, 1000.0]), 0.0)
    assert aloft[0] == pytest.approx(5.0 * 100.0**0.14)
    low = windVelocity(windy, np.array([0.0, 0.0, 0.0]), 0.0)
    assert low[0] == pytest.approx(5.0 * (1.0 / 10.0)**0.14)           # held at the 1 m value, never zero


def testWindDirectionRotatesTheVector():
    windy = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=3.0, direction=np.pi / 2, directionShearPerKm=0.0))
    v = windVelocity(windy, np.array([0.0, 0.0, windy.wind.referenceHeight]), 0.0)
    assert np.allclose(v, [0.0, 3.0, 0.0], atol=1e-12)


def testNoDragAtRestInStillAir():
    force, torque, _ = Drag(cfg.spacecraft, stillAir)(upright(100.0, [0, 0, 0]), noCommands, 0.0)
    assert np.allclose(force, 0.0) and np.allclose(torque, 0.0)


def testFallingUprightDragPointsUpWithTextbookMagnitude():
    s = upright(0.0, [0.0, 0.0, -80.0])
    force, torque, _ = Drag(cfg.spacecraft, stillAir)(s, noCommands, 0.0)
    area = aero.referenceArea + aero.gridFinArea
    expected = 0.5 * airDensity(cfg.environment, 0.0) * aero.dragCoefficient * area * 80.0**2
    assert np.allclose(force, [0.0, 0.0, expected])
    assert np.allclose(torque, 0.0, atol=1e-9)


def testSideWindPushesWithWindAndTorquesAboutCp():
    windy = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=5.0, direction=0.0, directionShearPerKm=0.0))
    s = upright(100.0, [0.0, 0.0, 0.0])
    force, torque, _ = Drag(cfg.spacecraft, windy)(s, noCommands, 0.0)
    assert force[0] > 0.0 and np.isclose(force[1], 0.0) and np.isclose(force[2], 0.0)
    cgHeight = massProperties(s, cfg.spacecraft).cgHeight
    # Pure crossflow: the body's drag acts at the crossflow centre of pressure, the fins' at the fins.
    bodyShare = aero.referenceArea * (1 + aero.dragAlphaFactor)
    pushHeight = (bodyShare * aero.cpHeightCrossflow + aero.gridFinArea * aero.finHeight) / (bodyShare + aero.gridFinArea)
    assert np.sign(torque[1]) == np.sign(pushHeight - cgHeight) and np.isclose(torque[0], 0.0)


def testTerminalVelocityIsReached():
    # A light test article of the booster's shape: the 33.6 t booster's ~190 m/s terminal speed needs far more than 3 km to settle.
    light = dataclasses.replace(cfg.spacecraft, mass=dataclasses.replace(cfg.spacecraft.mass, dryMass=1000.0))
    lightConfig = dataclasses.replace(cfg, spacecraft=light)
    start = upright(3000.0, [0, 0, 0])
    start = State(start.position, start.velocity, start.quaternion, start.omega, 0.0)
    traj = Simulation(lightConfig, producers=[Drag(light, stillAir)], dt=0.01).run(start)
    vz = traj.states[-100:, 5]
    assert traj.endReason == "touchdown"
    assert np.max(np.abs(vz)) / np.min(np.abs(vz)) < 1.01
    weight = massProperties(traj.state(0), light).total * cfg.environment.gravity
    area = aero.referenceArea + aero.gridFinArea
    vTerminal = np.sqrt(2 * weight / (airDensity(cfg.environment, 0.0) * aero.dragCoefficient * area))
    assert abs(vz[-1]) == pytest.approx(vTerminal, rel=0.02)
