""" testEngineRealism.py
Pressure-dependent thrust, thrust misalignment, and lateral CG effects on the engine, gear, and drag torques.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.state import Commands, State
from landingSim.environment.atmosphere import pressure
from landingSim.vehicle.aero import Drag
from landingSim.vehicle.engine import Engine
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
eng = cfg.spacecraft.engine


def upright(altitude=0.0):
    return State(np.array([0.0, 0.0, altitude]), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)


def testThrustGrowsWithAltitude():
    engine = Engine(cfg.spacecraft, cfg.environment)
    seaLevel = engine(upright(0.0), Commands(1.0, np.zeros(2)), 0.0)[0][2]
    aloft = engine(upright(1000.0), Commands(1.0, np.zeros(2)), 0.0)[0][2]
    expectedGain = eng.nozzleExitArea * (cfg.environment.seaLevelPressure - pressure(cfg.environment, 1000.0)) * eng.count
    assert seaLevel == pytest.approx(eng.maxThrust * eng.count)
    assert aloft - seaLevel == pytest.approx(expectedGain)
    assert Engine(cfg.spacecraft)(upright(1000.0), Commands(1.0, np.zeros(2)), 0.0)[0][2] == pytest.approx(seaLevel)


def testMassFlowDoesNotDependOnAltitude():
    engine = Engine(cfg.spacecraft, cfg.environment)
    assert engine(upright(0.0), Commands(1.0, np.zeros(2)), 0.0)[2] == pytest.approx(engine(upright(1000.0), Commands(1.0, np.zeros(2)), 0.0)[2])


def testThrustMisalignmentMakesATorqueWithCentredGimbal():
    tilted = dataclasses.replace(cfg.spacecraft, engine=dataclasses.replace(eng, thrustMisalignment=(0.01, 0.0)))
    _, torque, _ = Engine(tilted)(upright(), Commands(1.0, np.zeros(2)), 0.0)
    assert torque[1] < 0.0 and np.isclose(torque[0], 0.0)      # same sign convention as a +x gimbal


def testLateralCgMakesAnEngineTorqueAndShiftsTheFeet():
    offset = dataclasses.replace(cfg.spacecraft, mass=dataclasses.replace(cfg.spacecraft.mass, cgLateralOffset=(0.05, 0.0)))
    props = massProperties(upright(), offset)
    assert props.cgLateral[0] == pytest.approx(0.05 * cfg.spacecraft.mass.dryMass / props.total)
    _, torque, _ = Engine(offset)(upright(), Commands(1.0, np.zeros(2)), 0.0)
    assert torque[1] > 0.0                                     # thrust passes +x-ward of... the CG sits +x of the thrust line, nose tips toward +x
    feet = LandingGear(offset).footStates(upright(), props.cgHeight, props.cgLateral)
    assert feet[0][0][0] == pytest.approx(cfg.spacecraft.landingGear.footRadius - props.cgLateral[0])


def testLateralCgTorqueFromDragIsSmallButPresent():
    offset = dataclasses.replace(cfg.spacecraft, mass=dataclasses.replace(cfg.spacecraft.mass, cgLateralOffset=(0.05, 0.0)))
    still = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=0.0))
    falling = State(np.array([0.0, 0.0, 100.0]), np.array([0.0, 0.0, -50.0]), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    _, torque, _ = Drag(offset, still)(falling, Commands(0.0, np.zeros(2)), 0.0)
    assert abs(torque[1]) > 0.0 and np.isclose(torque[0], 0.0)
