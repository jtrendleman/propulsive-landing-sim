""" testOptimize.py
Fuel optimizer helpers: guidance overrides, the penalised objective, the braking sweep, and the TOML round trip.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import pytest

from landingSim.analysis.monteCarlo import missionSuccessRate
from landingSim.analysis.optimize import configToToml, objective, penaltyWeight, sweepReferenceDecel, withGuidance, writeConfig
from landingSim.config import loadConfig

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


def testWithGuidanceChangesOnlyTheNamedKnob():
    changed = withGuidance(cfg, referenceDecel=20.0)
    assert changed.guidance.referenceDecel == 20.0
    assert changed.guidance.descentRate == cfg.guidance.descentRate
    assert changed.spacecraft == cfg.spacecraft and cfg.guidance.referenceDecel == 18.0


def testObjectivePenalisesShortfallOnly():
    ok = {"meanPropellantUsed": 0.40, "successRate": 0.97}
    short = {"meanPropellantUsed": 0.40, "successRate": 0.90}
    assert objective(ok) == pytest.approx(0.40)
    assert objective(short) == pytest.approx(0.40 + penaltyWeight * (missionSuccessRate - 0.90))


def testConfigRoundTripsThroughToml(tmp_path):
    path = writeConfig(withGuidance(cfg, referenceDecel=3.25, coast=False), tmp_path / "variant.toml", header="test")
    back = loadConfig(path)
    assert back.guidance.referenceDecel == 3.25 and back.guidance.coast is False
    assert back.spacecraft == cfg.spacecraft and back.mission == cfg.mission
    assert "[spacecraft.landingGear]" in configToToml(cfg)


def testSweepReturnsOneSummaryPerValue():
    rows = sweepReferenceDecel(cfg, [2.0, 3.0], episodes=2, seed=3)
    assert [v for v, _ in rows] == [2.0, 3.0]
    assert all("meanPropellantUsed" in s for _, s in rows)


def testWithScheduleSwitchesTheScheduleOnAndSetsStrongKnobs():
    from landingSim.analysis.optimize import tables, withSchedule
    tuned = withSchedule(cfg, referenceDecel=3.7, kVelocity=2.5)
    assert tuned.windSchedule.enabled is True
    assert tuned.windSchedule.referenceDecel == 3.7 and tuned.windSchedule.kVelocity == 2.5
    assert tuned.guidance == cfg.guidance
    assert tables["windSchedule"] is withSchedule
