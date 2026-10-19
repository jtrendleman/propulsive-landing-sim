""" testTruthBoundary.py
The flight software may read the true state only through the sensor models: anything else is a test failure.
Author: Jack Rendleman Date: 10/18/2026 Version: 0-1-0A

The controller is handed a proxy instead of the true State. Every attribute read checks the call stack and
raises unless it happens inside landingSim.gnc.sensors. A full closed-loop landing must complete without
tripping it, for both the classical controller and the policy-wrapped one.
"""

import pathlib
import sys

import numpy as np

from landingSim.analysis.monteCarlo import buildEpisode
from landingSim.config import loadConfig
from landingSim.gnc.controller import LandingController
from landingSim.rl.residualController import ResidualController

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


class TruthAccessError(AssertionError):
    pass


class TruthProxy:
    """Stands in for State; only code running under landingSim.gnc.sensors may look inside."""

    def __init__(self, state):
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        frame = sys._getframe(1)
        depth = 0
        while frame is not None and depth < 25:
            if frame.f_globals.get("__name__") == "landingSim.gnc.sensors":
                return getattr(object.__getattribute__(self, "_state"), name)
            frame, depth = frame.f_back, depth + 1
        caller = sys._getframe(1).f_globals.get("__name__")
        raise TruthAccessError(f"flight software read true state field {name!r} from {caller}")


class ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(3, dtype=np.float32), None


def flyThroughProxy(controller, config=cfg, sensorSeed=3):
    sim, start = buildEpisode(config, 0.01, sensorSeed + 1)
    controller.reseed(sensorSeed, trueSpacecraft=config.spacecraft, trueEnvironment=config.environment)
    wrapped = _Wrapped(controller)
    return sim.run(start, controller=wrapped)


class _Wrapped:
    """Passes a proxy to the controller and forwards actuator feedback."""

    def __init__(self, controller):
        self.controller = controller

    def observeActuators(self, achieved):
        self.controller.observeActuators(achieved)

    def __call__(self, state, t):
        return self.controller(TruthProxy(state), t)


def testProxyBlocksDirectReads():
    from landingSim.dynamics.state import State
    from landingSim.core import attitude
    proxy = TruthProxy(State(np.zeros(3), np.zeros(3), attitude.identity.copy(), np.zeros(3), 1.0))
    try:
        _ = proxy.position
    except TruthAccessError:
        return
    raise AssertionError("the proxy let a test read the true state")


def testClassicalControllerNeverReadsTruth():
    traj = flyThroughProxy(LandingController(cfg))
    assert traj.endReason in ("landed", "crashed_tipped", "crashed_strut", "crashed_tumbled", "timeout")
    assert len(traj.time) > 1000


def testPolicyWrappedControllerNeverReadsTruth():
    traj = flyThroughProxy(ResidualController(cfg, ZeroModel()))
    assert len(traj.time) > 1000
