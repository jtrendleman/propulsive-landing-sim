""" terminators.py
Episode end conditions: callables that return an end reason string or None to continue.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.1.0
"""

import numpy as np

from landingSim.core import attitude


def groundTerminator(state, t):
    """Point-mass rule: the episode ends when the CG reaches the ground."""
    return "touchdown" if state.position[2] <= 0.0 else None


class LandingTerminator:
    """Gear-aware rule: landed when at rest on the pad, crashed if a strut bottoms out or the vehicle tips.

    Has memory (restStart), so build a new instance for every run.
    """

    def __init__(self, spacecraftConfig, gear, restSpeed=0.05, restRate=0.02, restTime=1.0, maxTilt=np.deg2rad(20), tumbleRate=10.0):
        self.landingGear = spacecraftConfig.landingGear
        self.gear = gear
        self.restSpeed = restSpeed
        self.restRate = restRate
        self.restTime = restTime
        self.maxTilt = maxTilt
        self.tumbleRate = tumbleRate
        self.restStart = None

    def __call__(self, state, t):
        # Tumble guard: a vehicle spinning this fast is lost, and the contact model blows up numerically
        # within a step or two once a tipped body drives its feet metres into the ground.
        if not np.all(np.isfinite(state.toArray())) or np.linalg.norm(state.omega) > self.tumbleRate:
            return "crashed_tumbled"
        penetration = self.gear.maxPenetration(state)
        if penetration > self.landingGear.strutStroke:
            return "crashed_strut"

        bodyZWorld = attitude.rotateBodyToWorld(state.quaternion, np.array([0.0, 0.0, 1.0]))
        tilt = np.arccos(np.clip(bodyZWorld[2], -1.0, 1.0))
        if tilt > self.maxTilt and penetration > 0.0:
            return "crashed_tipped"

        atRest = penetration > 0.0 and np.linalg.norm(state.velocity) < self.restSpeed and np.linalg.norm(state.omega) < self.restRate
        if atRest:
            if self.restStart is None:
                self.restStart = t
            elif t - self.restStart >= self.restTime:
                return "landed"
        else:
            self.restStart = None
        return None
