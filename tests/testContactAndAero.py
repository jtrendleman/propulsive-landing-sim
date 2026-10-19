""" testContactAndAero.py
Contact and aero physics: stick-slip friction, progressive strut, sloped pad, angle-of-attack drag and CP shift, grid fins and their allocation.
Author: Jack Rendleman Date: 10/18/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.analysis.monteCarlo import runEpisode
from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.state import Commands, State
from landingSim.environment.ground import heightAbovePad, padNormal
from landingSim.gnc.control import ThrustVectorController
from landingSim.gnc.controller import LandingController
from landingSim.vehicle.aero import Drag, GridFins, angleOfAttack
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
gear = cfg.spacecraft.landingGear
noCommands = Commands(0.0, np.zeros(2))
still = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=0.0, turbulenceIntensity=0.0))


def onPad(penetration=0.05, velocity=(0.0, 0.0, 0.0), environment=None):
    probe = State(np.zeros(3), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    cg = massProperties(probe, cfg.spacecraft).cgHeight
    return State(np.array([0.0, 0.0, cg - penetration]), np.array(velocity, float), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)


def testFrictionOpposesSlidingAndSaturatesAtCoulomb():
    g = LandingGear(cfg.spacecraft, still)
    slow, _, _ = g(onPad(velocity=(0.005, 0.0, 0.0)), noCommands, 0.0)
    fast, _, _ = g(onPad(velocity=(10.0, 0.0, 0.0)), noCommands, 0.0)
    assert slow[0] < 0.0 and fast[0] < 0.0
    coulombLimit = gear.frictionCoefficient * fast[2]
    assert abs(fast[0]) == pytest.approx(coulombLimit, rel=1e-6)      # a fast slide is capped at mu N
    assert abs(slow[0]) == pytest.approx(gear.strutCount * gear.footFriction * 0.005, rel=1e-6)   # below the cap: the damper


def testStuckFootPullsBackToItsAnchor():
    g = LandingGear(cfg.spacecraft, still)
    planted = onPad()
    g.commit(planted)                                                 # every foot pins its anchor
    shifted = State(planted.position + np.array([0.01, 0.0, 0.0]), planted.velocity, planted.quaternion, planted.omega, planted.propellantMass)
    force, _, _ = g(shifted, noCommands, 0.0)
    assert force[0] == pytest.approx(-gear.strutCount * gear.tangentialStiffness * 0.01, rel=1e-6)   # stiction: a spring, not a creep
    far = State(planted.position + np.array([1.0, 0.0, 0.0]), planted.velocity, planted.quaternion, planted.omega, planted.propellantMass)
    g.commit(far)                                                     # pushed past the friction limit: the anchors slide along
    assert abs(g(far, noCommands, 0.0)[0][0]) == pytest.approx(gear.frictionCoefficient * g(far, noCommands, 0.0)[0][2], rel=1e-6)


def testProgressiveStrutStiffensWithStroke():
    g = LandingGear(cfg.spacecraft, still)
    shallow = g(onPad(0.05), noCommands, 0.0)[0][2] / 0.05
    deep = g(onPad(0.30), noCommands, 0.0)[0][2] / 0.30
    assert deep > 1.5 * shallow


def testSlopedPadNormalAndDistance():
    sloped = dataclasses.replace(still, padSlope=0.1, padSlopeDirection=0.0)
    n = padNormal(sloped)
    assert n[2] > 0.99 and n[0] > 0.0                                 # surface drops toward +x, so its upward normal leans toward +x
    assert heightAbovePad(sloped, np.array([10.0, 0.0, 0.0])) > 0.0  # +x side is downhill, so z=0 there is above the surface
    assert heightAbovePad(sloped, np.array([-10.0, 0.0, 0.0])) < 0.0


def testVehicleRestsOnSlopedPad():
    sloped = dataclasses.replace(cfg, environment=dataclasses.replace(still, padSlope=0.02, padSlopeDirection=0.3))
    from landingSim.dynamics.simulation import Simulation
    from landingSim.dynamics.terminators import LandingTerminator
    g = LandingGear(sloped.spacecraft, sloped.environment)
    sim = Simulation(sloped, producers=[g], dt=0.002, terminator=LandingTerminator(sloped.spacecraft, g))
    traj = sim.run(onPad(-0.1))                                       # set down from 0.1 m above the pad
    assert traj.endReason == "landed"
    final = traj.state(len(traj.time) - 1)
    # The vehicle tilts to sit flat on the slope, which swings its high CG sideways; the base, where the feet are, must not move.
    base = final.position + attitude.rotateBodyToWorld(final.quaternion, np.array([0.0, 0.0, -massProperties(final, cfg.spacecraft).cgHeight]))
    start = traj.state(0)
    startBase = start.position + attitude.rotateBodyToWorld(start.quaternion, np.array([0.0, 0.0, -massProperties(start, cfg.spacecraft).cgHeight]))
    assert np.linalg.norm(base[0:2] - startBase[0:2]) < 0.05          # stiction holds the feet: settling only, no creeping downhill


def testDragRisesWithAngleOfAttackAndCpMoves():
    falling = State(np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, -50.0]), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.3)
    tilted = State(falling.position, falling.velocity, q, np.zeros(3), falling.propellantMass)
    assert angleOfAttack(falling, falling.velocity) == pytest.approx(0.0)
    assert angleOfAttack(tilted, tilted.velocity) == pytest.approx(0.3)
    straight = np.linalg.norm(Drag(cfg.spacecraft, still)(falling, noCommands, 0.0)[0])
    withAlpha = np.linalg.norm(Drag(cfg.spacecraft, still)(tilted, noCommands, 0.0)[0])
    aero = cfg.spacecraft.aero
    body, fins = aero.referenceArea, aero.gridFinArea                # crossflow raises the body's drag; the fins' stays put
    expected = (body * (1 + aero.dragAlphaFactor * np.sin(0.3) ** 2) + fins) / (body + fins)
    assert withAlpha / straight == pytest.approx(expected, rel=1e-3)


def testCrosswindHoverIsAerodynamicallyStable():
    """Hovering upright in a side wind (alpha ~ 90 deg): the drag torque must push the nose downwind, i.e. restore, because the CP sits above the CG."""
    windy = dataclasses.replace(still, wind=dataclasses.replace(still.wind, meanSpeed=8.0, direction=0.0, directionShearPerKm=0.0))
    hover = State(np.array([0.0, 0.0, 20.0]), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    force, torque, _ = Drag(cfg.spacecraft, windy)(hover, noCommands, 0.0)
    cg = massProperties(hover, cfg.spacecraft).cgHeight
    assert force[0] > 0.0 and torque[1] > 0.0                         # +x force above the CG -> +y torque (nose toward +x, downwind)
    assert cfg.spacecraft.aero.cpHeightCrossflow > cg


def testGridFinForceAndTorqueSigns():
    fast = State(np.array([0.0, 0.0, 500.0]), np.array([0.0, 0.0, -60.0]), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    force, torque, _ = GridFins(cfg.spacecraft, still)(fast, Commands(0.0, np.zeros(2), fins=np.array([0.1, 0.0])), 0.0)
    assert force[0] > 0.0 and torque[1] > 0.0                         # +x side force at the top tips the nose toward +x
    assert np.allclose(GridFins(cfg.spacecraft, still)(fast, noCommands, 0.0)[1], 0.0)


def testControllerUsesFinsWhenAirIsFastAndNotWhenSlow():
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.01)
    leaning = State(np.array([0.0, 0.0, 500.0]), np.zeros(3), q, np.zeros(3), cfg.spacecraft.propellant.initialMass)
    tvc = ThrustVectorController(cfg.spacecraft, cfg.guidance.kp, cfg.guidance.kd)
    crossover = cfg.spacecraft.aero.finMinDynamicPressure
    fastAir = tvc(leaning, np.array([0.0, 0.0, 1.0]), 0.0, engineOn=False, dynamicPressure=3 * crossover)
    slowAir = tvc(leaning, np.array([0.0, 0.0, 1.0]), 0.0, engineOn=False, dynamicPressure=0.5 * crossover)
    assert abs(fastAir.fins[0]) > 0.0 and np.allclose(slowAir.fins, 0.0)
    assert abs(fastAir.rcs[1]) < abs(slowAir.rcs[1])                  # fins took most of the torque
    burningFast = tvc(leaning, np.array([0.0, 0.0, 12.0]), 0.0, engineOn=True, dynamicPressure=3 * crossover)
    burningSlow = tvc(leaning, np.array([0.0, 0.0, 12.0]), 0.0, engineOn=True, dynamicPressure=0.5 * crossover)
    assert abs(burningFast.fins[0]) > 0.0                             # fins stay active through the burn while the air is fast
    assert np.allclose(burningSlow.fins, 0.0)                         # and hand over to the gimbal near the ground
    assert abs(burningSlow.gimbal[0]) > abs(burningFast.gimbal[0])


def testClosedLoopLandsWithContactAndAeroRealism():
    traj, result = runEpisode(cfg, LandingController(cfg, seed=2), sensorSeed=2)
    assert result.endReason == "landed", result
    assert result.strutPass and result.loadFactorPass and result.finalTilt < 0.05
