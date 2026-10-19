""" testTurbulence.py
Gust field: reproducibility, zero-intensity equivalence, gust statistics, direction shear, and closed loop in turbulence.
Author: Jack Rendleman Date: 10/18/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.analysis.monteCarlo import runEpisode
from landingSim.config import loadConfig
from landingSim.environment.wind import WindField, windDirectionAt, windVelocity
from landingSim.gnc.controller import LandingController

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")
env = cfg.environment


def testGustSeriesIsReproducibleFromSeed():
    a, b, c = WindField(env, seed=4), WindField(env, seed=4), WindField(env, seed=5)
    p = np.array([0.0, 0.0, 300.0])
    assert np.allclose(a.velocity(p, 12.3), b.velocity(p, 12.3))
    assert not np.allclose(a.velocity(p, 12.3), c.velocity(p, 12.3))


def testZeroIntensityEqualsSteadyWind():
    calm = dataclasses.replace(env, wind=dataclasses.replace(env.wind, turbulenceIntensity=0.0))
    field = WindField(calm, seed=1)
    p = np.array([10.0, -5.0, 250.0])
    assert np.allclose(field.velocity(p, 7.0), windVelocity(calm, p, 7.0))


def testGustStandardDeviationFollowsIntensity():
    field = WindField(env, seed=2, duration=2000.0)
    p = np.array([0.0, 0.0, 300.0])
    samples = np.array([field.gust(p, t) for t in np.arange(0.0, 1990.0, 0.5)])
    steady = windVelocity(env, p, 0.0)
    sigma = env.wind.turbulenceIntensity * np.linalg.norm(steady)
    assert np.std(samples[:, 0]) == pytest.approx(sigma, rel=0.3)
    assert np.std(samples[:, 2]) == pytest.approx(sigma * env.wind.verticalGustFraction, rel=0.3)
    assert abs(np.mean(samples[:, 0])) < 0.3 * sigma


def testDirectionShearTurnsWithAltitude():
    assert windDirectionAt(env.wind, 1000.0) - windDirectionAt(env.wind, 0.0) == pytest.approx(env.wind.directionShearPerKm)
    low, high = windVelocity(env, np.array([0, 0, 10.0]), 0.0), windVelocity(env, np.array([0, 0, 1000.0]), 0.0)
    angle = np.arctan2(high[1], high[0]) - np.arctan2(low[1], low[0])
    assert angle == pytest.approx(env.wind.directionShearPerKm * 0.99, abs=0.02)


def testClosedLoopLandsInTurbulence():
    traj, result = runEpisode(cfg, LandingController(cfg, seed=3), sensorSeed=3)
    assert result.outcome == "succeeded", result
