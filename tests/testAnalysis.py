""" testAnalysis.py
Success-criteria scoring of logged episodes and the Monte Carlo perturbation and summary helpers.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest

from landingSim.analysis import criteria
from landingSim.analysis.criteria import evaluate
from landingSim.analysis.monteCarlo import missionStart, perturbConfig, runEpisode, runMonteCarlo, summarize
from landingSim.config import loadConfig
from landingSim.core import attitude
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import State
from landingSim.dynamics.terminators import LandingTerminator
from landingSim.gnc.controller import LandingController
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


def feetAt(height):
    probe = State(np.zeros(3), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)
    cgHeight = massProperties(probe, cfg.spacecraft).cgHeight
    return State(np.array([0.0, 0.0, cgHeight + height]), np.zeros(3), attitude.identity.copy(), np.zeros(3), cfg.spacecraft.propellant.initialMass)


def gentleDrop(startHeight):
    gear = LandingGear(cfg.spacecraft)
    sim = Simulation(cfg, producers=[gear], dt=0.001, terminator=LandingTerminator(cfg.spacecraft, gear))
    return sim.run(feetAt(startHeight))


def testGentleDropSucceeds():
    result = evaluate(gentleDrop(0.2), cfg)                       # ~2 m/s touchdown, the controller's normal arrival
    assert result.endReason == "landed" and result.outcome == "succeeded"
    assert result.strutPass and result.propellantPass and result.offsetPass
    assert result.touchdownSpeed < 0.0
    assert result.peakStrutForce > 0.0
    assert result.peakStrutStress == pytest.approx(result.peakStrutForce / cfg.spacecraft.landingGear.strutCrossSection)


def testGentleTouchdownStaysUnderLoadFactorLimit():
    result = evaluate(gentleDrop(0.2), cfg)
    assert result.loadFactorPass and 0.5 < result.peakLoadFactor < cfg.spacecraft.structure.loadFactorLimit


def testLoadFactorIsSensedNotInertial():
    """A free fall feels zero g; the criterion must not count gravity itself."""
    traj = Simulation(cfg, producers=[], dt=0.01).run(feetAt(200.0))
    result = evaluate(traj, cfg)
    assert result.peakLoadFactor < 0.05 or result.endReason == "touchdown"


def testHardDropIsHardFailure():
    result = evaluate(gentleDrop(50.0), cfg)
    assert result.endReason == "crashed_strut" and result.outcome == "hardFailed"


def testOffsetLandingIsSoftFailure():
    traj = gentleDrop(0.2)
    shifted = traj.states.copy()
    shifted[:, 0] += 2.0                                          # whole episode 2 m off the pad centre
    import dataclasses
    result = evaluate(dataclasses.replace(traj, states=shifted), cfg)
    assert result.endReason == "landed" and result.outcome == "softFailed"
    assert not result.offsetPass and result.finalOffset == pytest.approx(2.0)


def testPeakStrutForceMatchesGearModel():
    traj = gentleDrop(0.2)
    gear = LandingGear(cfg.spacecraft)
    expected = max(f[2] for i in range(len(traj.time)) for _, f in gear.footForces(traj.state(i)))
    assert evaluate(traj, cfg).peakStrutForce == pytest.approx(expected)


def testPerturbConfigChangesOnlyTheDrawnFields():
    rng = np.random.default_rng(3)
    drawn = perturbConfig(cfg, rng)
    assert drawn.spacecraft.engine.maxThrust != cfg.spacecraft.engine.maxThrust
    assert drawn.spacecraft.mass.dryMass != cfg.spacecraft.mass.dryMass
    assert drawn.environment.wind.meanSpeed != cfg.environment.wind.meanSpeed
    assert drawn.spacecraft.engine.isp == cfg.spacecraft.engine.isp
    assert drawn.spacecraft.landingGear == cfg.spacecraft.landingGear
    assert drawn.mission.maxTime == cfg.mission.maxTime
    assert drawn.environment.padSlope >= 0.0


def testPerturbConfigDrawsNonNegativeSpeedAndRandomDirection():
    rng = np.random.default_rng(0)
    draws = [perturbConfig(cfg, rng) for _ in range(50)]
    assert all(d.environment.wind.meanSpeed >= 0.0 for d in draws)
    directions = np.array([d.environment.wind.direction for d in draws])
    assert directions.min() < np.pi / 2 and directions.max() > 3 * np.pi / 2       # spread around the circle
    assert all(d.environment.wind.profileExponent == cfg.environment.wind.profileExponent for d in draws)


def testPerturbConfigIsReproducible():
    a = perturbConfig(cfg, np.random.default_rng(7))
    b = perturbConfig(cfg, np.random.default_rng(7))
    assert a == b


def testMissionStartLiftsCgByCgHeight():
    start = missionStart(cfg)
    assert start.position[2] - massProperties(start, cfg.spacecraft).cgHeight == pytest.approx(cfg.mission.initialAltitude)


def testNominalEpisodeSucceeds():
    _, result = runEpisode(cfg, LandingController(cfg))
    assert result.outcome == "succeeded"


def testMonteCarloSummaryRatesSumToOne():
    results = runMonteCarlo(cfg, episodes=2, seed=1)
    summary = summarize(results)
    assert summary["episodes"] == 2
    assert summary["successRate"] + summary["softFailRate"] + summary["hardFailRate"] == pytest.approx(1.0)
    assert 0.0 <= summary["offsetPassRate"] <= 1.0


def testThresholdsLiveInAnalysis():
    assert criteria.propellantReserveFraction == 0.05 and criteria.landingRadius == 0.5


def testParallelMonteCarloMatchesSerial():
    serial = runMonteCarlo(cfg, episodes=4, seed=5)
    parallel = runMonteCarlo(cfg, episodes=4, seed=5, workers=2)
    assert [r for _, r in serial] == [r for _, r in parallel]
    assert [c for c, _ in serial] == [c for c, _ in parallel]


def testSummaryReportsMissionSuccessAndFuel():
    summary = summarize(runMonteCarlo(cfg, episodes=2, seed=1))
    assert 0.0 <= summary["meanPropellantUsed"] <= 1.0
    assert summary["missionSuccess"] == (summary["successRate"] >= 0.95)
