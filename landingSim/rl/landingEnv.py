""" landingEnv.py
Gymnasium environment: the agent adds a residual acceleration to ZEM/ZEV guidance on a randomized vehicle.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

One env step holds the residual for decisionInterval seconds of simulation. The per-step reward is
potential-based shaping on horizontal offset (dense, does not change the optimal policy) plus a
small effort cost; the landing outcome arrives as a terminal reward. Each reset draws a
perturbed config (thrust, mass, CG, wind, offset) for the VEHICLE while the baseline controller
keeps believing the nominal config, so the residual's job is to cover that gap.
"""

import gymnasium as gym
import numpy as np

from landingSim.analysis.criteria import evaluate
from landingSim.analysis.monteCarlo import buildEpisode, perturbConfig
from landingSim.dynamics.trajectory import Trajectory
from landingSim.gnc.controller import LandingController
from landingSim.rl.observation import observationSize, observe

rewardScale = 0.01                    # rewards are quoted in "landing points" below and scaled to about [-1, 1] for PPO
crashReward = -100.0
landedReward = 100.0
offsetPenaltyPerMetre = 2.0           # taken off landedReward, capped at 50 m
criterionPenalty = 10.0               # per failed strut or propellant criterion on a landed episode
offsetPotentialPerMetre = 0.5         # potential-based shaping: closing 1 m of horizontal offset earns this, opening it costs it
effortPenaltyPerSecond = 0.1          # per unit of squared normalized action, per second


class LandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, dt=0.01, randomize=True):
        super().__init__()
        self.nominalConfig = config
        self.dt = dt
        self.randomize = randomize
        self.decisionInterval = config.rl.decisionInterval
        self.maxResidual = config.rl.maxResidualAccel
        self.stepsPerDecision = max(1, int(round(self.decisionInterval / dt)))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(observationSize,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.episodeConfig = None
        self.controller = None
        self.sim = None
        self.state = None
        self.t = 0.0
        self.rows = None
        self.endReason = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episodeConfig = perturbConfig(self.nominalConfig, self.np_random) if self.randomize else self.nominalConfig
        self.controller = LandingController(self.nominalConfig)
        self.controller.reseed(int(self.np_random.integers(0, 2**31 - 1)), trueSpacecraft=self.episodeConfig.spacecraft,
                               trueEnvironment=self.episodeConfig.environment)
        self.sim, self.state = buildEpisode(self.episodeConfig, self.dt, windSeed=int(self.np_random.integers(0, 2**31 - 1)))
        self.actuators = self.sim.newActuators()
        self.t = 0.0
        self.rows = {"time": [], "states": [], "commands": [], "achieved": []}
        self.endReason = None
        self.controller.updateEstimate(self.state, self.t)
        return observe(self.state, self.t, self.controller), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.controller.residual = action * self.maxResidual

        endReason = None
        offsetBefore = np.linalg.norm(self.state.position[0:2])
        for _ in range(self.stepsPerDecision):
            commands = self.controller(self.state, self.t)
            achieved = self.sim.actuate(self.actuators, commands, self.dt)
            self.controller.observeActuators(achieved)
            self.rows["time"].append(self.t)
            self.rows["states"].append(self.state.toArray())
            self.rows["commands"].append(commands.toArray())
            self.rows["achieved"].append(achieved.toArray())

            endReason = self.sim.terminator(self.state, self.t)
            if endReason is not None:
                break
            if self.t >= self.sim.maxTime - 1e-9:
                endReason = "timeout"
                break
            self.state = self.sim.advance(self.state, achieved, self.t)
            self.t += self.dt

        offsetAfter = np.linalg.norm(self.state.position[0:2])
        reward = offsetPotentialPerMetre * (offsetBefore - offsetAfter) - effortPenaltyPerSecond * float(action @ action) * self.decisionInterval
        terminated = endReason is not None and endReason != "timeout"
        truncated = endReason == "timeout"
        info = {}
        if endReason is not None:
            self.endReason = endReason
            result = evaluate(self.trajectory(), self.episodeConfig)
            reward += self.terminalReward(result)
            info["result"] = result
        self.controller.updateEstimate(self.state, self.t)
        return observe(self.state, self.t, self.controller), float(reward * rewardScale), terminated, truncated, info

    @staticmethod
    def terminalReward(result):
        if result.endReason != "landed":
            return crashReward
        reward = landedReward - offsetPenaltyPerMetre * min(result.finalOffset, 50.0)
        reward -= criterionPenalty * ((not result.strutPass) + (not result.propellantPass))
        return reward

    def trajectory(self):
        """The current episode's log as a Trajectory, for evaluation or the viewer."""
        return Trajectory(np.array(self.rows["time"]), np.array(self.rows["states"]), np.array(self.rows["commands"]),
                          "running" if self.endReason is None else self.endReason, np.array(self.rows["achieved"]))
