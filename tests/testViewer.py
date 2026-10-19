""" testViewer.py
Replay export: frame downsampling, field contents, and the self-contained HTML build.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import json
import pathlib

import numpy as np
import pytest

from landingSim.config import loadConfig
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import initialState
from landingSim.viewer.export import dataMarker, replayData, templatePath, writeReplay

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


@pytest.fixture(scope="module")
def dropTrajectory():
    return Simulation(cfg, producers=[], dt=0.01).run(initialState(cfg.mission, cfg.spacecraft))


def testFramesAreDownsampledAndKeepTheLastRow(dropTrajectory):
    data = replayData(dropTrajectory, cfg, maxFrames=100)
    assert 50 <= len(data["frames"]) <= 101
    assert data["frames"][0]["t"] == 0.0
    assert data["frames"][-1]["t"] == pytest.approx(dropTrajectory.time[-1])
    assert data["endReason"] == dropTrajectory.endReason


def testFrameFieldsMatchTheLog(dropTrajectory):
    data = replayData(dropTrajectory, cfg, maxFrames=10)
    first = data["frames"][0]
    assert np.allclose(first["p"], dropTrajectory.states[0, 0:3])
    assert np.allclose(first["q"], dropTrajectory.states[0, 6:10])
    assert first["throttle"] == 0.0 and first["propellant"] == pytest.approx(1.0)
    assert first["cg"] > 0.0 and first["vz"] == pytest.approx(cfg.mission.initialVerticalSpeed)
    assert first["feet"] == pytest.approx(first["p"][2] - first["cg"])
    assert first["tilt"] == pytest.approx(0.0, abs=1e-9) and first["load"] >= 0.0
    assert data["environment"]["gravity"] == cfg.environment.gravity
    assert data["geometry"]["bodyLength"] == cfg.spacecraft.geometry.bodyLength
    assert data["geometry"]["strutCount"] == cfg.spacecraft.landingGear.strutCount


def testReplayDataIsJsonSerializable(dropTrajectory):
    json.dumps(replayData(dropTrajectory, cfg, maxFrames=20))


def testZeroEffortImpactLandsAhead():
    from landingSim.viewer.export import zeroEffortImpact
    x, y, t = zeroEffortImpact(np.array([10.0, 0.0, 103.0]), np.array([5.0, 0.0, -20.0]), 100.0, 9.81)
    assert t == pytest.approx((-20.0 + np.sqrt(20.0**2 + 2 * 9.81 * 100.0)) / 9.81)
    assert x == pytest.approx(10.0 + 5.0 * t) and y == 0.0
    assert zeroEffortImpact(np.zeros(3), np.zeros(3), 0.0, 9.81)[2] == 0.0


def testTemplateHasExactlyOneDataMarker():
    assert templatePath.read_text(encoding="utf-8").count(dataMarker) == 1


def testWriteReplayInlinesTheData(tmp_path, dropTrajectory):
    out = writeReplay(dropTrajectory, cfg, tmp_path / "replay.html", maxFrames=20)
    html = out.read_text(encoding="utf-8")
    assert dataMarker not in html
    assert '"endReason": "touchdown"' in html
    assert "<script type=\"module\">" in html and "three.module.js" in html
