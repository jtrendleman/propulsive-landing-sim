""" residualController.py
Wraps the baseline LandingController with a trained policy that updates the residual every decision interval.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

Drop-in replacement for LandingController in Simulation.run and the Monte Carlo loop.
"""

import numpy as np

from landingSim.gnc.controller import LandingController
from landingSim.rl.observation import observe


class ResidualController:
    def __init__(self, config, model, deterministic=True):
        self.baseline = LandingController(config)
        self.model = model
        self.deterministic = deterministic
        self.decisionInterval = config.rl.decisionInterval
        self.maxResidual = config.rl.maxResidualAccel
        self.nextDecisionTime = 0.0

    def reseed(self, seed=None, trueSpacecraft=None, trueEnvironment=None):
        self.baseline.reseed(seed, trueSpacecraft, trueEnvironment)

    def observeActuators(self, achieved):
        self.baseline.observeActuators(achieved)

    def __call__(self, state, t):
        self.baseline.updateEstimate(state, t)                 # the policy sees the same sensor-based estimate
        if t >= self.nextDecisionTime - 1e-9:
            action, _ = self.model.predict(observe(state, t, self.baseline), deterministic=self.deterministic)
            self.baseline.residual = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.maxResidual
            self.nextDecisionTime = t + self.decisionInterval
        return self.baseline(state, t)
