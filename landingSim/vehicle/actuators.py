""" actuators.py
Engine actuator dynamics: ignition delay, first-order throttle lag, and a gimbal rate limit between command and engine.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

Stepped once per logging step by the Simulation, between the controller and the producers, so the
RK4 stages inside a step see constant achieved commands. Achieved throttle is either zero or at least
minThrottle: ignition brings it to minThrottle after ignitionDelay, then it relaxes toward the command;
a cutoff command closes the valve immediately. Has memory, so build one per episode.
"""

import numpy as np

from landingSim.dynamics.state import Commands


class Actuators:
    def __init__(self, spacecraftConfig):
        self.engine = spacecraftConfig.engine
        self.throttle = 0.0
        self.gimbal = np.zeros(2)
        self.ignitionTimer = 0.0
        self.lit = False

    def step(self, commanded, dt):
        eng = self.engine
        wantsThrust = commanded.throttle >= eng.minThrottle
        if not wantsThrust:
            self.throttle, self.lit, self.ignitionTimer = 0.0, False, 0.0
        elif not self.lit:
            self.ignitionTimer += dt
            if self.ignitionTimer >= eng.ignitionDelay:
                self.lit = True
                self.throttle = eng.minThrottle
        else:
            target = float(np.clip(commanded.throttle, eng.minThrottle, 1.0))
            alpha = dt / max(eng.throttleTimeConstant, 1e-6)
            self.throttle = float(np.clip(self.throttle + (target - self.throttle) * min(alpha, 1.0), eng.minThrottle, 1.0))

        maxStep = eng.maxGimbalRate * dt
        target = np.clip(np.asarray(commanded.gimbal, float), -eng.maxGimbalAngle, eng.maxGimbalAngle)
        self.gimbal = self.gimbal + np.clip(target - self.gimbal, -maxStep, maxStep)
        return Commands(throttle=self.throttle, gimbal=self.gimbal.copy(), rcs=np.asarray(commanded.rcs, float), fins=np.asarray(commanded.fins, float))
