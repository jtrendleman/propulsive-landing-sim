""" testLandingGear.py
Landing gear contact forces, restoring torque, and landed/crashed episode termination.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import Commands, State
from landingSim.dynamics.terminators import LandingTerminator, groundTerminator
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
gear = cfg.spacecraft.landingGear
noCommands = Commands(throttle=0.0, gimbal=np.zeros(2))


def stateWithFeetAt(footHeight, quaternion=None, velocity=None):
    """CG placed so the feet (on the base plane) sit at footHeight when upright."""
    probe = State(np.zeros(3), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    cgHeight = massProperties(probe, cfg.spacecraft).cgHeight
    return State(
        position=np.array([0.0, 0.0, cgHeight + footHeight]),
        velocity=np.zeros(3) if velocity is None else velocity,
        quaternion=attitude.identity.copy() if quaternion is None else quaternion,
        omega=np.zeros(3),
        propellantMass=cfg.spacecraft.propellant.initialMass,
    )


def testNoForceAboveGround():
    force, torque, _ = LandingGear(cfg.spacecraft)(stateWithFeetAt(1.0), noCommands, 0.0)
    assert np.allclose(force, 0.0) and np.allclose(torque, 0.0)
    assert LandingGear(cfg.spacecraft).maxPenetration(stateWithFeetAt(1.0)) == 0.0


def testUprightCompressionGivesPureSpringForce():
    penetration = 0.1
    s = stateWithFeetAt(-penetration)
    force, torque, _ = LandingGear(cfg.spacecraft)(s, noCommands, 0.0)
    assert np.allclose(force, [0.0, 0.0, gear.strutCount * gear.strutStiffness * penetration * (1 + gear.strutHardening * (penetration / gear.strutStroke) ** 2)])
    assert np.allclose(torque, 0.0, atol=1e-6)
    assert LandingGear(cfg.spacecraft).maxPenetration(s) == pytest.approx(penetration)


def testDownwardFootVelocityAddsDamping():
    s = stateWithFeetAt(-0.1, velocity=np.array([0.0, 0.0, -1.0]))
    force, _, _ = LandingGear(cfg.spacecraft)(s, noCommands, 0.0)
    expected = gear.strutCount * (gear.strutStiffness * 0.1 * (1 + gear.strutHardening * (0.1 / gear.strutStroke) ** 2) + gear.strutDamping * 1.0)
    assert force[2] == pytest.approx(expected)


def testTiltProducesRestoringTorque():
    tilt = 0.02                                                     # rad about +y, leans +z toward +x
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), tilt)
    s = stateWithFeetAt(-0.1, quaternion=q)
    _, torque, _ = LandingGear(cfg.spacecraft)(s, noCommands, 0.0)
    assert torque[1] < 0.0                                          # opposes the +y tilt
    assert abs(torque[0]) < 1e-6 and abs(torque[2]) < 1e-6


def testGentleDropLands():
    gearProducer = LandingGear(cfg.spacecraft)
    sim = Simulation(cfg, producers=[gearProducer], dt=0.001, terminator=LandingTerminator(cfg.spacecraft, gearProducer))
    traj = sim.run(stateWithFeetAt(1.0))
    assert traj.endReason == "landed"
    final = traj.state(len(traj.time) - 1)
    weight = massProperties(final, cfg.spacecraft).total * cfg.environment.gravity
    perStrut = weight / gear.strutCount
    x = perStrut / gear.strutStiffness
    for _ in range(50):                                             # fixed-point solve of the progressive spring
        x = perStrut / (gear.strutStiffness * (1 + gear.strutHardening * (x / gear.strutStroke) ** 2))
    assert gearProducer.maxPenetration(final) == pytest.approx(x, rel=0.05)


def testHardDropCrashesStrut():
    gearProducer = LandingGear(cfg.spacecraft)
    sim = Simulation(cfg, producers=[gearProducer], dt=0.001, terminator=LandingTerminator(cfg.spacecraft, gearProducer))
    traj = sim.run(stateWithFeetAt(50.0))
    assert traj.endReason == "crashed_strut"


def testGroundTerminatorStillWorksForPointMass():
    assert groundTerminator(stateWithFeetAt(1.0), 0.0) is None
    below = State(np.array([0.0, 0.0, -0.01]), np.zeros(3), attitude.identity.copy(), np.zeros(3), 0.0)
    assert groundTerminator(below, 0.0) == "touchdown"


def testTumblingVehicleIsACrash():
    gearProducer = LandingGear(cfg.spacecraft)
    terminator = LandingTerminator(cfg.spacecraft, gearProducer)
    spinning = State(np.array([0.0, 0.0, 50.0]), np.zeros(3), attitude.identity.copy(), np.array([0.0, 20.0, 0.0]),
                     cfg.spacecraft.propellant.initialMass)
    assert terminator(spinning, 0.0) == "crashed_tumbled"
    broken = State(np.array([np.nan, 0.0, 50.0]), np.zeros(3), attitude.identity.copy(), np.zeros(3), 0.0)
    assert terminator(broken, 0.0) == "crashed_tumbled"
