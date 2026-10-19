""" testPlots3d.py
Smoke tests: the 3D figures render from a logged trajectory and write image files.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from landingSim.analysis.criteria import evaluate
from landingSim.analysis.plots3d import landingDensitySurface, plotDispersion3d, plotLanding3d, plotLandingMontage
from landingSim.config import loadConfig
from landingSim.dynamics.simulation import Simulation
from landingSim.dynamics.state import initialState

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


@pytest.fixture(scope="module")
def dropTrajectory():
    return Simulation(cfg, producers=[], dt=0.02).run(initialState(cfg.mission, cfg.spacecraft))


def testLandingFiguresAreWritten(tmp_path, dropTrajectory):
    plotLanding3d(dropTrajectory, cfg, tmp_path / "a.png")
    plotLandingMontage(dropTrajectory, cfg, tmp_path / "b.png")
    plt.close("all")
    assert (tmp_path / "a.png").stat().st_size > 10_000 and (tmp_path / "b.png").stat().st_size > 10_000


def testDispersionFigureIsWritten(tmp_path, dropTrajectory):
    results = [evaluate(dropTrajectory, cfg)] * 3
    plotDispersion3d([dropTrajectory] * 3, results, cfg, tmp_path / "c.png")
    plt.close("all")
    assert (tmp_path / "c.png").stat().st_size > 10_000


def testDensitySurfacePeaksAtTouchdownPoints():
    gx, gy, z = landingDensitySurface([(2.0, -1.0)], halfWidth=5.0, cells=51, sigma=0.5)
    i = np.unravel_index(np.argmax(z), z.shape)
    assert gx[i] == pytest.approx(2.0, abs=0.2) and gy[i] == pytest.approx(-1.0, abs=0.2)
