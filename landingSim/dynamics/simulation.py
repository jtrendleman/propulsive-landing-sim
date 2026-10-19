""" simulation.py
Episode loop: ask the controller, record, check for an end, take one RK4 step, repeat.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0
"""

import numpy as np

from landingSim.dynamics.integrator import step
from landingSim.dynamics.state import Commands
from landingSim.dynamics.terminators import groundTerminator
from landingSim.dynamics.trajectory import Trajectory
from landingSim.vehicle.actuators import Actuators
from landingSim.vehicle.mass import massProperties


def zeroCommands(state, t):
    """Default controller: engine off, gimbal centered."""
    return Commands(throttle=0.0, gimbal=np.zeros(2))


class Simulation:
    """Episode loop. Near the ground the strut springs and dampers are stiff enough that RK4 at the
    logging step can go unstable for a light vehicle, so each step is split into contactSubsteps
    sub-steps while the feet are within contactHeight of the pad. The log keeps the dt spacing."""

    def __init__(self, config, producers, dt, terminator=groundTerminator, contactSubsteps=4, contactHeight=1.0, actuators=True):
        self.config = config
        self.useActuators = actuators
        self.producers = producers
        self.dt = dt
        self.terminator = terminator
        self.spacecraft = config.spacecraft
        self.environment = config.environment
        self.maxTime = config.mission.maxTime
        self.contactSubsteps = contactSubsteps
        self.contactHeight = contactHeight

    def newActuators(self):
        """Fresh actuator state for an episode; None when actuator dynamics are switched off."""
        return Actuators(self.spacecraft) if self.useActuators else None

    @staticmethod
    def actuate(actuators, commands, dt):
        """Commanded -> achieved commands for one logging step."""
        return actuators.step(commands, dt) if actuators is not None else commands

    def advance(self, state, commands, t):
        """One logging step of dt, sub-stepped when the feet are near the ground."""
        feetHeight = state.position[2] - massProperties(state, self.spacecraft).cgHeight
        n = self.contactSubsteps if feetHeight < self.contactHeight else 1
        h = self.dt / n
        for k in range(n):
            state = step(state, self.producers, commands, t + k * h, h, self.spacecraft, self.environment)
            for producer in self.producers:
                if hasattr(producer, "commit"):
                    producer.commit(state)          # between-step memory, such as where a foot is stuck
        return state

    def run(self, initialState, controller=zeroCommands):
        state, t = initialState, 0.0
        times, stateRows, commandRows, achievedRows = [], [], [], []
        actuators = self.newActuators()

        while True:
            commands = controller(state, t)
            achieved = self.actuate(actuators, commands, self.dt)
            if hasattr(controller, "observeActuators"):
                controller.observeActuators(achieved)       # actuator feedback reaches the sensors next step

            # Record first so the final state is always the last row.
            times.append(t)
            stateRows.append(state.toArray())
            commandRows.append(commands.toArray())
            achievedRows.append(achieved.toArray())

            endReason = self.terminator(state, t)
            if endReason is not None:
                break
            if t >= self.maxTime - 1e-9:
                endReason = "timeout"
                break

            state = self.advance(state, achieved, t)
            t += self.dt

        return Trajectory(np.array(times), np.array(stateRows), np.array(commandRows), endReason, np.array(achievedRows))
