""" testWindSchedule.py
Wind-scheduled guidance: the blend factor, the blended knobs, the away-from-pad wind input, and that a schedule with equal endpoints changes nothing.
Author: Jack Rendleman Date: 10/19/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import numpy as np
import pytest

from landingSim.analysis.monteCarlo import buildEpisode, runEpisode
from landingSim.config import loadConfig
from landingSim.gnc.controller import LandingController, blendFactor, blendedGuidance, windScheduleKnobs

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


def scheduled(config, windSpeed=None, direction=None, **strong):
    """Schedule on, exact forecast, and optionally a different mean wind or strong-wind knob values."""
    wind = config.environment.wind
    wind = dataclasses.replace(wind, meanSpeed=wind.meanSpeed if windSpeed is None else windSpeed,
                               direction=wind.direction if direction is None else direction)
    return dataclasses.replace(config,
                               environment=dataclasses.replace(config.environment, wind=wind),
                               windSchedule=dataclasses.replace(config.windSchedule, enabled=True, **strong),
                               sensors=dataclasses.replace(config.sensors, forecastSpeedError=0.0, forecastDirectionError=0.0))


def blendAtStart(config):
    _, start = buildEpisode(config, 0.01, 1)
    controller = LandingController(config)
    controller.reseed(3, trueSpacecraft=config.spacecraft, trueEnvironment=config.environment)
    controller.updateEstimate(start, 0.0)
    return controller, start


def testBlendFactorIsLinearAndClamped():
    ws = cfg.windSchedule
    assert blendFactor(ws.averageWindSpeed, ws) == 0.0
    assert blendFactor(ws.strongWindSpeed, ws) == 1.0
    assert blendFactor(0.5 * (ws.averageWindSpeed + ws.strongWindSpeed), ws) == pytest.approx(0.5)
    assert blendFactor(-3.0, ws) == 0.0
    assert blendFactor(20.0, ws) == 1.0
    assert blendFactor(6.0, dataclasses.replace(ws, strongWindSpeed=ws.averageWindSpeed)) == 0.0


def testBlendedGuidanceMovesOnlyTheScheduledKnobs():
    ws = dataclasses.replace(cfg.windSchedule, referenceDecel=4.0, maxTilt=0.4, kVelocity=3.0)
    g = cfg.guidance
    assert blendedGuidance(g, ws, 0.0) == g
    full = blendedGuidance(g, ws, 1.0)
    assert all(getattr(full, k) == pytest.approx(getattr(ws, k)) for k in windScheduleKnobs)
    half = blendedGuidance(g, ws, 0.5)
    assert half.referenceDecel == pytest.approx(0.5 * (g.referenceDecel + 4.0))
    unscheduled = [f.name for f in dataclasses.fields(g) if f.name not in windScheduleKnobs]
    assert all(getattr(half, k) == getattr(g, k) for k in unscheduled)


def testStrongWindAwayFromThePadMovesTheKnobs():
    _, start = buildEpisode(cfg, 0.01, 1)
    awayDirection = float(np.arctan2(start.position[1], start.position[0]))
    config = scheduled(cfg, windSpeed=9.0, direction=awayDirection, referenceDecel=4.0)
    controller, _ = blendAtStart(config)
    assert controller.blend > 0.9
    assert controller.guidance.referenceDecel == pytest.approx(controller.guidanceConfig.referenceDecel)
    assert controller.guidanceConfig.referenceDecel > 3.8


def testWindTowardThePadKeepsTheAverageKnobs():
    _, start = buildEpisode(cfg, 0.01, 1)
    towardPad = float(np.arctan2(-start.position[1], -start.position[0]))
    controller, _ = blendAtStart(scheduled(cfg, windSpeed=9.0, direction=towardPad, referenceDecel=4.0))
    assert controller.blend == 0.0
    assert controller.guidanceConfig.referenceDecel == cfg.guidance.referenceDecel


def testEqualEndpointsChangeNothing():
    # The shipped scenario files start with strong-wind values equal to [guidance], so switching the schedule on
    # must reproduce the unscheduled landing exactly.
    on = dataclasses.replace(cfg, windSchedule=dataclasses.replace(cfg.windSchedule, enabled=True))
    trajOff, resultOff = runEpisode(cfg, LandingController(cfg), sensorSeed=5)
    trajOn, resultOn = runEpisode(on, LandingController(on), sensorSeed=5)
    assert resultOn.outcome == resultOff.outcome
    assert np.allclose(trajOn.states, trajOff.states)


def testScheduledControllerNeverReadsTruth():
    from testTruthBoundary import flyThroughProxy
    config = scheduled(cfg, windSpeed=7.5, referenceDecel=3.5, maxTilt=0.3)
    traj = flyThroughProxy(LandingController(config), config=config)
    assert len(traj.time) > 1000
