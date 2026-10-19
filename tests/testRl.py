""" testRl.py
Gymnasium environment contract, residual plumbing, reward tiers, and the policy-wrapping controller.
Author: Jack Rendleman Date: 10/17/2026 Version: 0-1-0A
"""

import pathlib

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from landingSim.analysis.criteria import LandingResult
from landingSim.analysis.monteCarlo import runEpisode
from landingSim.config import loadConfig
from landingSim.gnc.controller import LandingController
from landingSim.rl.landingEnv import LandingEnv, crashReward, landedReward, rewardScale
from landingSim.rl.observation import observationSize, observe
from landingSim.rl.residualController import ResidualController

repoRoot = pathlib.Path(__file__).resolve().parents[1]
cfg = loadConfig(repoRoot / "config" / "earthBooster.toml")


class ConstantModel:
    def __init__(self, action):
        self.action = np.array(action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action, None


def testEnvPassesGymnasiumChecker():
    check_env(LandingEnv(cfg), warn=False)


def testObservationShapeAndScale():
    env = LandingEnv(cfg, randomize=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (observationSize,) and obs.dtype == np.float32
    assert np.all(np.abs(obs) < 10.0)
    from landingSim.analysis.monteCarlo import missionStart
    from landingSim.vehicle.mass import massProperties
    cgHeight = massProperties(missionStart(cfg), cfg.spacecraft).cgHeight
    assert obs[2] == pytest.approx((cfg.mission.initialAltitude + cgHeight) / 1000.0, abs=0.01)   # feet altitude plus CG height, in km


def testResetWithSameSeedDrawsSameVehicle():
    a, b = LandingEnv(cfg), LandingEnv(cfg)
    a.reset(seed=11)
    b.reset(seed=11)
    assert a.episodeConfig == b.episodeConfig
    assert a.episodeConfig != cfg                               # randomize=True really perturbed it


def testZeroResidualNominalEpisodeSucceeds():
    env = LandingEnv(cfg, randomize=False)
    env.reset(seed=0)
    total, done, info = 0.0, False, {}
    while not done:
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        total += reward
        done = terminated or truncated
    assert info["result"].outcome == "succeeded"
    assert total > (landedReward - 10.0) * rewardScale            # closing the 50 m offset earns shaping, effort is zero
    assert env.trajectory().endReason == "landed"


def testResidualReachesTheController():
    env = LandingEnv(cfg, randomize=False)
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(env.controller.residual, [cfg.rl.maxResidualAccel, 0.0, 0.0])


def testTerminalRewardTiers():
    def result(endReason, offset, strutPass=True, propellantPass=True):
        return LandingResult(endReason, "x", 0.0, 0.0, offset, 0.0, 0.5, 0.0, 0.0, 0.0, strutPass, propellantPass, offset <= 0.5, True)

    assert LandingEnv.terminalReward(result("crashed_strut", 0.0)) == crashReward
    assert LandingEnv.terminalReward(result("timeout", 0.0)) == crashReward
    assert LandingEnv.terminalReward(result("landed", 0.0)) == landedReward
    assert LandingEnv.terminalReward(result("landed", 10.0)) < LandingEnv.terminalReward(result("landed", 1.0))
    assert LandingEnv.terminalReward(result("landed", 0.0, strutPass=False)) < landedReward


def testZeroPolicyControllerMatchesBaseline():
    _, baseline = runEpisode(cfg, LandingController(cfg), sensorSeed=7)
    _, wrapped = runEpisode(cfg, ResidualController(cfg, ConstantModel([0.0, 0.0, 0.0])), sensorSeed=7)
    assert wrapped == baseline                                   # same sensor seed, so identical flights


def testObservationUsesGuidanceCommand():
    env = LandingEnv(cfg, randomize=False)
    obs, _ = env.reset(seed=0)
    aCmd = env.controller.guidanceCommand(env.controller.estimate, 0.0)        # the policy sees the sensor-based estimate, not the truth
    assert np.allclose(obs[13:16], aCmd / 20.0, atol=1e-6)
    assert np.allclose(observe(env.state, 0.0, env.controller), obs)


def testTrainingCurveAndFactoryArePicklable(tmp_path):
    import pickle
    from landingSim.rl.report import ResidualControllerFactory, plotTrainingCurve
    path = plotTrainingCurve([(100 * i, "succeeded" if i % 3 else "softFailed", 0.5 + 0.01 * i) for i in range(60)], tmp_path / "curve.png")
    assert path.stat().st_size > 5_000
    factory = ResidualControllerFactory(tmp_path / "model.zip")
    assert pickle.loads(pickle.dumps(factory)).modelPath == str(tmp_path / "model.zip")
