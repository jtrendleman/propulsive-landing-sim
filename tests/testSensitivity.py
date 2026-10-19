""" testSensitivity.py
Sensitivity helpers: perturbation table, separation statistic, one-at-a-time sweep, and the tornado figure.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from landingSim.analysis.monteCarlo import runMonteCarlo
from landingSim.analysis.sensitivity import oneAtATime, perturbationTable, plotTornado, separation
from landingSim.config import loadConfig

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


@pytest.fixture(scope="module")
def results():
    return runMonteCarlo(cfg, episodes=6, seed=2)


def testPerturbationTableHasOneRowPerEpisode(results):
    table = perturbationTable(results)
    assert all(len(v) == 6 for v in table.values())
    assert np.all(table["wind speed (m/s)"] >= 0.0)


def testSeparationIsSortedByMagnitude(results):
    rows = separation(results)
    magnitudes = [abs(d) for _, d in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(rows) == 6


def testOneAtATimeAndTornado(tmp_path, results):
    rows = oneAtATime(cfg, sigmas=1.0)
    assert len(rows) == 6 and all(hasattr(r[1], "outcome") for r in rows)
    plotTornado(separation(results), rows, tmp_path / "tornado.png")
    assert (tmp_path / "tornado.png").stat().st_size > 10_000
