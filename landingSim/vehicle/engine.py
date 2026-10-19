""" engine.py
Main engine force producer: throttle and gimbal angles become thrust, torque about the CG, and propellant flow.
Author: Jack Rendleman  Date: 10-17-2026    Version: 2.0.0

Gimbal [gx, gy] tilts the thrust vector toward +x and +y body; a fixed thrustMisalignment adds to it.
Thrust grows with altitude: maxThrust is the sea-level value and nozzleExitArea * (p0 - p) is added as
ambient pressure falls (propellant flow stays throttle-proportional). The engine sits on the base plane
on the body axis, so its lever arm from the CG is [-cgLateral, -cgHeight]. Below minThrottle the engine
is off (not clamped up); an empty tank is also off.
"""

import numpy as np

from landingSim.core import attitude, constants
from landingSim.environment.atmosphere import pressure
from landingSim.vehicle.mass import massProperties


class Engine:
    def __init__(self, spacecraftConfig, environmentConfig=None):
        self.spacecraftConfig = spacecraftConfig
        self.engineConfig = spacecraftConfig.engine
        self.environmentConfig = environmentConfig          # None means sea-level thrust everywhere

    def maxThrustAt(self, altitude):
        eng = self.engineConfig
        if self.environmentConfig is None:
            return eng.maxThrust * eng.count
        env = self.environmentConfig
        return (eng.maxThrust + eng.nozzleExitArea * (env.seaLevelPressure - pressure(env, altitude))) * eng.count

    def __call__(self, state, commands, t):
        eng = self.engineConfig
        if commands.throttle < eng.minThrottle or state.propellantMass <= 0.0:
            return np.zeros(3), np.zeros(3), 0.0

        throttle = float(np.clip(commands.throttle, eng.minThrottle, 1.0))
        thrust = throttle * self.maxThrustAt(state.position[2])
        gimbal = np.clip(commands.gimbal, -eng.maxGimbalAngle, eng.maxGimbalAngle) + np.array(eng.thrustMisalignment)
        directionBody = attitude.normalize(np.array([np.tan(gimbal[0]), np.tan(gimbal[1]), 1.0]))
        forceBody = thrust * directionBody

        props = massProperties(state, self.spacecraftConfig)
        leverArm = np.array([-props.cgLateral[0], -props.cgLateral[1], -props.cgHeight])     # CG -> engine, body frame
        torqueBody = attitude.cross3(leverArm, forceBody)

        forceWorld = attitude.rotateBodyToWorld(state.quaternion, forceBody)
        massFlow = throttle * eng.maxThrust * eng.count / (eng.isp * constants.g0)
        return forceWorld, torqueBody, massFlow
