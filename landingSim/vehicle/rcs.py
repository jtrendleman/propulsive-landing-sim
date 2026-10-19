""" rcs.py
Reaction control system producer: a normalized torque command per body axis becomes a pure torque.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

Thrusters fire in opposing pairs, so the net force is zero and each axis can deliver up to
thrustPerThruster * momentArm of torque. RCS propellant is not tracked: the config has no
RCS Isp, and the quantities used in a landing are small next to the main tank.
"""

import numpy as np


class Rcs:
    def __init__(self, spacecraftConfig):
        self.rcsConfig = spacecraftConfig.rcs
        self.maxTorque = self.rcsConfig.thrustPerThruster * self.rcsConfig.momentArm

    def __call__(self, state, commands, t):
        command = np.clip(commands.rcs, -1.0, 1.0)
        return np.zeros(3), command * self.maxTorque, 0.0
