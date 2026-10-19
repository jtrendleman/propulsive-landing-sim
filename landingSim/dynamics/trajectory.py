""" trajectory.py
Per-timestep log of an episode: time, state rows, command rows, and how it ended.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

from dataclasses import dataclass

import numpy as np

from landingSim.dynamics.state import State


@dataclass
class Trajectory:
    time: np.ndarray        # (N,)
    states: np.ndarray      # (N, 14), rows are State.toArray(); layout owned by state.py
    commands: np.ndarray    # (N, 8), rows are Commands.toArray(): [throttle, gimbal0, gimbal1, rcs0, rcs1, rcs2, fin0, fin1]
    endReason: str          # "touchdown", "timeout", or a terminator's reason
    achieved: np.ndarray = None   # (N, 8), what the actuators delivered; None means the commands were applied directly

    def state(self, i):
        """Row i as a State, so callers never slice columns by hand."""
        return State.fromArray(self.states[i])

    def __post_init__(self):
        if self.achieved is None:
            self.achieved = self.commands

    def save(self, path):
        np.savez(path, time=self.time, states=self.states, commands=self.commands, endReason=np.array(self.endReason), achieved=self.achieved)

    @staticmethod
    def load(path):
        data = np.load(path)
        achieved = data["achieved"] if "achieved" in data.files else None
        return Trajectory(data["time"], data["states"], data["commands"], str(data["endReason"]), achieved)
