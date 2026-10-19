""" testSlosh.py
Slosh: domed-tank fill, SP-106 pendulum parameters, kick response, steady tilt, wall stop, mass bookkeeping, and the closed loop.
Author: Jack Rendleman Date: 10/18/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.analysis.monteCarlo import runEpisode
from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.state import State
from landingSim.gnc.controller import LandingController
from landingSim.vehicle.slosh import Slosh, besselRoot, sloshDerivative, sloshParameters

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
prop = cfg.spacecraft.propellant
lox = prop.oxidizer
landingResidual = 1000.0          # kg in both tanks near touchdown (about 0.7 t of it LOX)


def tank(propellant=None, slosh=(0.0, 0.0, 0.0, 0.0)):
    """A vehicle at rest with the given total propellant; slosh is the oxidizer tank's mode (the fuel tank's is quiet)."""
    return State(np.array([0.0, 0.0, 100.0]), np.zeros(3), attitude.identity.copy(), np.zeros(3),
                 prop.initialMass if propellant is None else propellant, np.concatenate([np.array(slosh, float), np.zeros(4)]))


def loxMass(total):
    from landingSim.vehicle.tank import tankMasses
    return tankMasses(total, prop)[0]


def testDomeFillMatchesTheAnalyticVolume():
    from landingSim.vehicle.tank import bowlVolume, domeDepth, domeVolume, tankFill
    R, d = lox.tankRadius, domeDepth(lox)
    assert domeVolume(lox) == pytest.approx(2 / 3 * np.pi * R**2 * d)
    for mass in (5.0, 700.0, 5600.0):
        fill = tankFill(mass, lox)
        assert bowlVolume(fill.height, R, d) == pytest.approx(mass / lox.density, rel=1e-9)
        assert fill.surfaceRadius == pytest.approx(R * np.sqrt(1 - ((d - fill.height) / d) ** 2))
        assert 0.5 * fill.height < fill.centroidHeight < fill.height     # a bowl holds more liquid near the top
    above = tankFill(1.2 * domeVolume(lox) * lox.density, lox)
    assert above.height == pytest.approx(d + 0.2 * domeVolume(lox) / (np.pi * R**2))
    assert above.surfaceRadius == R


def testPropellantSplitsByMixtureRatio():
    from landingSim.vehicle.tank import tankMasses
    oxidizer, fuel = tankMasses(prop.initialMass, prop)
    assert oxidizer + fuel == pytest.approx(prop.initialMass)
    assert oxidizer / fuel == pytest.approx(prop.mixtureRatio)


def testLowFillPendulumIsTheBowlCurvature():
    # A shallow pool in a bowl swings like a pendulum of the dome's bottom curvature radius R^2 / d (Lamb). The
    # equivalent cylinder tends to 4 / xi^2 of it as the pool empties, and is within about 5 % at the landing fill.
    from landingSim.vehicle.tank import domeDepth
    curvature = lox.tankRadius**2 / domeDepth(lox)
    nearlyEmpty = sloshParameters(tank(0.5), cfg.spacecraft, 9.81)
    assert nearlyEmpty.length == pytest.approx(4 / besselRoot**2 * curvature, rel=0.02)
    landing = sloshParameters(tank(landingResidual), cfg.spacecraft, 9.81)
    assert landing.length == pytest.approx(curvature, rel=0.06)
    assert landing.omega == pytest.approx(np.sqrt(9.81 / curvature), rel=0.03)


def testParametersFollowSp106():
    from landingSim.vehicle.tank import tankFill
    s = tank()
    fill = tankFill(loxMass(s.propellantMass), lox)
    r, hm, xi = fill.surfaceRadius, fill.meanDepth, besselRoot
    params = sloshParameters(s, cfg.spacecraft, 9.81)
    assert params.mass == pytest.approx((2 * r / (xi * hm)) * np.tanh(xi * hm / r) / (xi**2 - 1) * loxMass(s.propellantMass))
    assert params.length == pytest.approx(r / (xi * np.tanh(xi * hm / r)))
    assert 0.0 < params.mass < loxMass(s.propellantMass)
    assert sloshParameters(tank(0.0), cfg.spacecraft, 9.81).mass == 0.0
    assert sloshParameters(s, cfg.spacecraft, 9.81, "fuel").mass > 0.0                # the fuel tank has its own mode


def testFrequencyRisesWithAxialAcceleration():
    assert sloshParameters(tank(), cfg.spacecraft, 20.0).omega > sloshParameters(tank(), cfg.spacecraft, 9.81).omega


def testLateralKickSwingsTheLiquidAgainstTheKick():
    d = sloshDerivative(tank(), cfg.spacecraft, np.array([2.0, 0.0, 9.81]))
    assert d.shape == (8,)
    assert d[2] < 0.0 and d[3] == 0.0                        # tank pushed +x, oxidizer lags toward -x
    assert d[6] < 0.0 and d[7] == 0.0                        # and so does the fuel


def testSwungLiquidPullsTheTankAndMakesTorque():
    swung = tank(slosh=(0.1, 0.0, 0.0, 0.0))
    force, torque, _ = Slosh(cfg.spacecraft)(swung, None, 0.0)
    assert force[0] > 0.0                                   # the rod pulls the hinge toward the swung liquid
    assert force[2] < 0.0                                   # and carries the liquid's weight
    assert torque[1] != 0.0


def testSteadySideForceTiltsThePendulum():
    f = np.array([1.5, 0.0, 9.81])
    theta = np.arctan2(-f[0], f[2])                         # equilibrium: tan(theta) = -f_x / a, both tanks
    tilted = dataclasses.replace(tank(), slosh=np.array([theta, 0.0, 0.0, 0.0, theta, 0.0, 0.0, 0.0]))
    d = sloshDerivative(tilted, cfg.spacecraft, f)
    assert d[2] == pytest.approx(0.0, abs=1e-9) and d[6] == pytest.approx(0.0, abs=1e-9)
    force, _, _ = Slosh(cfg.spacecraft)(tilted, None, 0.0, specificForceBody=f)
    m1 = sum(sloshParameters(tank(), cfg.spacecraft, 9.81, name).mass for name in ("oxidizer", "fuel"))
    assert force[0] == pytest.approx(-m1 * f[0]) and force[2] == pytest.approx(-m1 * f[2])   # exactly what the slosh masses need


def testSloshDecaysWhenLeftAlone():
    from landingSim.dynamics.simulation import Simulation
    from landingSim.dynamics.state import Commands
    from landingSim.vehicle.mass import massProperties
    # A damped-mode check with a clearly damped tank (zeta 0.1); the 0.01 default would need minutes to show it.
    # The booster cannot hover on its engine (the throttle floor out-lifts it), so a support force stands in.
    damped = dataclasses.replace(cfg.spacecraft, propellant=dataclasses.replace(prop, sloshDamping=0.1))
    oneG = dataclasses.replace(cfg, spacecraft=damped, mission=dataclasses.replace(cfg.mission, maxTime=30.0))
    start = tank(slosh=(0.1, 0.0, 0.0, 0.0))
    total = massProperties(start, damped).total

    class Support:
        def __call__(self, state, commands, t):
            inertia = massProperties(state, damped).inertia
            torque = -inertia * (100.0 * 2 * state.quaternion[1:4] + 20.0 * state.omega)
            return np.array([0.0, 0.0, total * cfg.environment.gravity]), torque, 0.0

    traj = Simulation(oneG, producers=[Support(), Slosh(damped)], dt=0.01, actuators=False).run(
        start, controller=lambda state, t: Commands(throttle=0.0, gimbal=np.zeros(2)))
    amplitudeEarly = np.max(np.abs(traj.states[:300, 14]))
    amplitudeLate = np.max(np.abs(traj.states[-300:, 14]))
    assert amplitudeLate < 0.5 * amplitudeEarly


def testDisabledSloshIsInert():
    off = dataclasses.replace(cfg.spacecraft, propellant=dataclasses.replace(prop, sloshEnabled=False))
    force, torque, _ = Slosh(off)(tank(slosh=(0.1, 0.0, 0.0, 0.0)), None, 0.0)
    assert np.allclose(force, 0.0) and np.allclose(torque, 0.0)


def testClosedLoopLandsWithSlosh():
    traj, result = runEpisode(cfg, LandingController(cfg, seed=4), sensorSeed=4)
    assert result.endReason == "landed", result               # sloshing tanks do not upset the landing
    assert np.max(np.abs(traj.states[:, 14:16])) > 0.0        # the oxidizer mode was excited at some point
    assert np.max(np.abs(traj.states[:, 18:20])) > 0.0        # and so was the fuel's


def testHardSideKickStaysInsideTheTank():
    # Ten seconds of a 1 g side push on a hovering, attitude-held tank: the liquid swings into the wall stop and stays there.
    from landingSim.dynamics.simulation import Simulation
    from landingSim.dynamics.state import Commands
    from landingSim.vehicle.mass import massProperties
    config = dataclasses.replace(cfg, mission=dataclasses.replace(cfg.mission, maxTime=10.0))
    start = tank()
    total = massProperties(start, cfg.spacecraft).total

    class Kick:
        def __call__(self, state, commands, t):
            inertia = massProperties(state, cfg.spacecraft).inertia
            torque = -inertia * (100.0 * 2 * state.quaternion[1:4] + 20.0 * state.omega)
            return np.array([total * 9.81, 0.0, total * 9.81]), torque, 0.0

    traj = Simulation(config, producers=[Kick(), Slosh(cfg.spacecraft)], dt=0.01, actuators=False).run(
        start, controller=lambda state, t: Commands(throttle=0.0, gimbal=np.zeros(2)))
    params = sloshParameters(start, cfg.spacecraft, 9.81)
    offsets = params.length * np.abs(np.sin(traj.states[:, 14]))
    assert np.all(np.isfinite(traj.states))
    assert np.max(offsets) < lox.tankRadius


def testSloshMassIsNotCountedTwice():
    # A steady sideways force on a hovering tank accelerates the whole vehicle at F / total mass once the slosh
    # mode settles. Counting the mode mass in the rigid body as well would give F / (total + slosh mass).
    from landingSim.dynamics.simulation import Simulation
    from landingSim.dynamics.state import Commands
    from landingSim.vehicle.mass import massProperties
    damped = dataclasses.replace(cfg.spacecraft, propellant=dataclasses.replace(prop, sloshDamping=0.3))
    config = dataclasses.replace(cfg, spacecraft=damped, mission=dataclasses.replace(cfg.mission, maxTime=40.0))
    start = tank()
    total = massProperties(start, damped).total
    sideForce = 5000.0

    class Push:
        """Constant side force and weight support, plus a stiff attitude hold so the slosh torque cannot tip the tank."""
        def __call__(self, state, commands, t):
            inertia = massProperties(state, damped).inertia
            torque = -inertia * (100.0 * 2 * state.quaternion[1:4] + 20.0 * state.omega)
            return np.array([sideForce, 0.0, total * cfg.environment.gravity]), torque, 0.0

    traj = Simulation(config, producers=[Push(), Slosh(damped)], dt=0.01, actuators=False).run(
        start, controller=lambda state, t: Commands(throttle=0.0, gimbal=np.zeros(2)))
    late = traj.time > 25.0
    accel = np.polyfit(traj.time[late], traj.states[late, 3], 1)[0]
    assert sloshParameters(start, damped, 9.81).mass > 0.05 * total
    assert accel == pytest.approx(sideForce / total, rel=0.01)
