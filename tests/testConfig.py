""" testConfig.py
Tests for the scenario config loader: nominal load, immutability, strict keys.
Author: Jack Rendleman Date: 10/14/2026 Version: 0-1-0A
"""

import dataclasses
import pathlib

import pytest

from landingSim.config import ConfigError, loadConfig

repoRoot = pathlib.Path(__file__).resolve().parents[1]
nominalPath = repoRoot / "config" / "earthBooster.toml"


def writeVariant(tmpPath, oldText, newText):
    """Copy the nominal TOML to tmpPath with one substring replaced, return its path."""
    text = nominalPath.read_text(encoding="utf-8")
    assert oldText in text, f"test setup: {oldText!r} not found in nominal config"
    variantPath = tmpPath / "variant.toml"
    variantPath.write_text(text.replace(oldText, newText), encoding="utf-8")
    return variantPath


def testLoadsNominalFile():
    cfg = loadConfig(nominalPath)
    assert cfg.spacecraft.mass.dryMass == 25600.0                # Falcon 9 Block 5-class first stage
    assert cfg.environment.gravity == 9.81
    assert cfg.mission.maxTime == 120.0
    assert isinstance(cfg.spacecraft.mass.inertiaDry, tuple)
    assert len(cfg.spacecraft.mass.inertiaDry) == 3


def testConfigIsFrozen():
    cfg = loadConfig(nominalPath)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.mission.maxTime = 5.0


def testUnknownKeyRaises(tmp_path):
    badPath = writeVariant(tmp_path, "maxTime = 120.0", "maxTime = 120.0\ntypoKey = 1.0")
    with pytest.raises(ConfigError) as err:
        loadConfig(badPath)
    assert "typoKey" in str(err.value)


def testMissingKeyRaises(tmp_path):
    badPath = writeVariant(tmp_path, "gravity = 9.81", "")
    with pytest.raises(ConfigError) as err:
        loadConfig(badPath)
    assert "gravity" in str(err.value)


def testGuidanceTableLoads():
    cfg = loadConfig(nominalPath)
    assert cfg.guidance.referenceDecel == 18.0
    assert cfg.guidance.coast is True
    assert cfg.guidance.descentRate < 0.0


def testExtremeScenarioSharesTheVehicle():
    average = loadConfig(nominalPath)
    extreme = loadConfig(repoRoot / "config" / "earthBoosterKscExtreme.toml")
    assert extreme.spacecraft == average.spacecraft and extreme.mission == average.mission
    assert extreme.environment.wind.meanSpeed > average.environment.wind.meanSpeed
