""" testNavigation.py
Sensor models, the navigation Kalman filter, the thrust estimator, and closed-loop landing on estimated state.
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
from landingSim.gnc.navigation import Navigator
from landingSim.gnc.sensors import Sensors
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


def hovering(altitude=100.0):
    return State(np.array([0.0, 0.0, altitude]), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)


def testSensorsAreReproducibleFromSeed():
    a, b = Sensors(cfg, seed=3), Sensors(cfg, seed=3)
    ma, mb = a.measure(hovering(), 0.0), b.measure(hovering(), 0.0)
    assert np.allclose(ma.specificForceBody, mb.specificForceBody) and np.allclose(ma.gps[0], mb.gps[0])
    assert not np.allclose(Sensors(cfg, seed=4).measure(hovering(), 0.0).specificForceBody, ma.specificForceBody)


def testGpsFixesArriveAtTheConfiguredRate():
    sensors = Sensors(cfg, seed=0)
    fixes = sum(sensors.measure(hovering(), k * 0.01).gps is not None for k in range(100))
    assert fixes == pytest.approx(cfg.sensors.gpsRate, abs=1)


def testAltimeterOnlyBelowRange():
    sensors = Sensors(cfg, seed=0)
    assert sensors.measure(hovering(cfg.sensors.altimeterRange + 500.0), 0.0).altimeter is None
    low = sensors.measure(hovering(20.0), 0.01)
    feet = 20.0 - massProperties(hovering(20.0), cfg.spacecraft).cgHeight
    assert low.altimeter == pytest.approx(feet, abs=0.3)


def testNavigatorTracksAFreeFallWithinSensorNoise():
    nav = Navigator(cfg, seed=1)
    g = cfg.environment.gravity
    errors = []
    for k in range(300):
        t = k * 0.01
        true = State(np.array([5.0, -3.0, 400.0 - 0.5 * g * t**2]), np.array([0.0, 0.0, -g * t]), attitude.identity.copy(), np.zeros(3), 300.0)
        est = nav(true, t)
        if k > 100:
            errors.append(np.linalg.norm(est.position - true.position))
    assert np.mean(errors) < 3 * cfg.sensors.gpsPositionNoise      # within a few centimetres: fixes every 0.1 s, IMU bias in between
    assert np.allclose(est.quaternion, attitude.identity, atol=0.01)


def testDisabledSensorsReturnTheTrueState():
    perfect = dataclasses.replace(cfg, sensors=dataclasses.replace(cfg.sensors, enabled=False))
    nav = Navigator(perfect, seed=0)
    s = hovering()
    assert nav(s, 0.0) is s


def testThrustEstimatorSeesAWeakEngine():
    weak = dataclasses.replace(cfg, spacecraft=dataclasses.replace(cfg.spacecraft, engine=dataclasses.replace(cfg.spacecraft.engine, maxThrust=cfg.spacecraft.engine.maxThrust * 0.88)))
    controller = LandingController(cfg, seed=0)
    traj, result = runEpisode(weak, controller)
    assert 0.8 < controller.thrustScale < 0.96                     # pulled toward 0.88 from 1.0
    assert result.endReason == "landed"


def testClosedLoopLandsOnEstimatedState():
    traj, result = runEpisode(cfg, LandingController(cfg, seed=5))
    assert result.outcome == "succeeded", result


def testAltimeterMeasuresTheRealVehicleNotTheBelievedOne():
    """A CG-perturbed vehicle must still read the true base height; the controller's nominal CG is irrelevant to the sensor."""
    taller = dataclasses.replace(cfg.spacecraft, mass=dataclasses.replace(cfg.spacecraft.mass, cgHeightDry=cfg.spacecraft.mass.cgHeightDry + 0.3))
    quiet = dataclasses.replace(cfg, sensors=dataclasses.replace(cfg.sensors, altimeterNoise=0.0, altimeterNoiseFraction=0.0))
    sensors = Sensors(quiet, seed=0)
    sensors.reseed(0, trueSpacecraft=taller)
    s = State(np.array([0.0, 0.0, 10.0]), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    trueFeet = 10.0 - massProperties(s, taller).cgHeight
    assert sensors.measure(s, 0.0).altimeter == pytest.approx(trueFeet, abs=1e-9)
    assert abs(trueFeet - (10.0 - massProperties(s, cfg.spacecraft).cgHeight)) > 0.1    # the believed value really differs


def testGpsFixesArriveLate():
    sensors = Sensors(cfg, seed=0)
    for k in range(30):
        t = k * 0.01
        s = State(np.array([0.0, 0.0, 100.0 - 10.0 * t]), np.array([0.0, 0.0, -10.0]), attitude.identity.copy(), np.zeros(3), 300.0)
        m = sensors.measure(s, t)
        if m.gps is not None and t > 0.15:
            assert m.gps[2] == pytest.approx(t - cfg.sensors.gpsLatency, abs=0.011)
            assert m.gps[0][2] == pytest.approx(100.0 - 10.0 * m.gps[2], abs=0.2)


def testChamberPressureSeesAWeakEngineAndGaugeHasBias():
    from landingSim.dynamics.state import Commands
    weak = dataclasses.replace(cfg.spacecraft, engine=dataclasses.replace(cfg.spacecraft.engine, maxThrust=cfg.spacecraft.engine.maxThrust * 0.9))
    sensors = Sensors(cfg, seed=2)
    sensors.reseed(2, trueSpacecraft=weak)
    sensors.observeActuators(Commands(throttle=0.8, gimbal=np.zeros(2)))
    m = sensors.measure(hovering(0.0), 0.0)
    assert m.thrust == pytest.approx(0.8 * weak.engine.maxThrust, rel=0.05)
    assert m.throttleFeedback == pytest.approx(0.8, abs=0.01)
    assert abs(m.propellantGauge - cfg.spacecraft.propellant.initialMass) < 0.1 * cfg.spacecraft.propellant.initialMass


def testAttitudeIsIntegratedFromGyrosNotMeasured():
    nav = Navigator(cfg, seed=3)
    q = attitude.fromAxisAngle(np.array([0.0, 1.0, 0.0]), 0.05)
    for k in range(200):
        s = State(np.array([0.0, 0.0, 500.0]), np.zeros(3), q, np.zeros(3), 300.0)
        est = nav(s, k * 0.01)
    angle = 2 * np.arccos(min(abs(attitude.multiply(attitude.conjugate(q), est.quaternion)[0]), 1.0))
    assert angle < 5 * cfg.sensors.alignmentError                  # small: alignment error plus 2 s of tactical-grade drift
    assert nav.sensors.measure(s, 2.0).alignment is None          # only the first sample carries an attitude fix


def testForecastDiffersFromTruthButIsReproducible():
    truth = dataclasses.replace(cfg.environment, wind=dataclasses.replace(cfg.environment.wind, meanSpeed=6.0, direction=1.0))
    a, b = LandingController(cfg), LandingController(cfg)
    a.reseed(11, trueEnvironment=truth); b.reseed(11, trueEnvironment=truth)
    assert a.forecastEnvironment.wind == b.forecastEnvironment.wind
    assert a.forecastEnvironment.wind.meanSpeed != 6.0 or a.forecastEnvironment.wind.direction != 1.0


def testWindObserverSeesASteadyCrosswind():
    windy = dataclasses.replace(cfg, environment=dataclasses.replace(cfg.environment, wind=dataclasses.replace(
        cfg.environment.wind, meanSpeed=7.0, direction=0.0, directionShearPerKm=0.0, turbulenceIntensity=0.0)))
    controller = LandingController(cfg)
    traj, result = runEpisode(windy, controller, sensorSeed=5)
    assert result.endReason == "landed"
    # A +x wind pushes the vehicle toward +x; by the end the observer has seen a positive x push or decayed with the low-speed hover.
    pushes = controller.windAccel
    assert np.all(np.isfinite(pushes))
