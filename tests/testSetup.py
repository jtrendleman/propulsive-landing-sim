""" testSetup.py
Smoke test confirming the Python environment and package skeleton import correctly.
Author: Jack Rendleman Date: 10/14/2026 Version: 0-1-0A
"""

import importlib

import pytest


@pytest.mark.parametrize("moduleName", [
    "numpy", "scipy", "matplotlib", "torch", "gymnasium", "stable_baselines3",
    "landingSim", "landingSim.core", "landingSim.environment", "landingSim.vehicle",
    "landingSim.dynamics", "landingSim.gnc", "landingSim.rl", "landingSim.analysis",
])
def testModuleImports(moduleName):
    importlib.import_module(moduleName)
