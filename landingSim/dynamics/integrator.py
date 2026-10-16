""" integrator.py
Rigid-body equations of motion and the RK4 step that advances the state in time.
Author: Jack Rendleman  Date: 10-16-2026    Version: 1.0.0

Force producers share one signature:
    producer(state, commands, t) -> (forceWorld (3,), torqueBody (3,))
Gravity is a body force (scales with mass) and enters as an acceleration instead.
"""

import numpy as np

from landingSim.core import attitude
from landingSim.dynamics.state import State
from landingSim.environment.gravity import gravityAcceleration
from landingSim.vehicle.mass import massProperties


def rk4Step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def stateDerivative(state, producers, commands, t, spacecraftConfig, environmentConfig):
    props = massProperties(state, spacecraftConfig)

    force = np.zeros(3)
    torque = np.zeros(3)
    for producer in producers:
        producerForce, producerTorque = producer(state, commands, t)
        force = force + producerForce
        torque = torque + producerTorque

    acceleration = force / props.total + gravityAcceleration(environmentConfig, state.position)

    # Euler's rotational equation with principal (diagonal) inertia.
    gyro = np.cross(state.omega, props.inertia * state.omega)
    dOmega = (torque - gyro) / props.inertia

    dPropellant = 0.0  # no engine producer yet, so no mass flow

    return np.concatenate([
        state.velocity,
        acceleration,
        attitude.quatDerivative(state.quaternion, state.omega),
        dOmega,
        [dPropellant],
    ])


def step(state, producers, commands, t, dt, spacecraftConfig, environmentConfig):
    def f(time, arr):
        return stateDerivative(State.fromArray(arr), producers, commands, time, spacecraftConfig, environmentConfig)

    return State.fromArray(rk4Step(f, state.toArray(), t, dt))
