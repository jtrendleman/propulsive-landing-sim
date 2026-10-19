""" integrator.py
Rigid-body equations of motion and the RK4 step that advances the state in time.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.2.0

Force producers share one signature:
    producer(state, commands, t) -> (forceWorld (3,), torqueBody (3,), massFlow kg/s >= 0)
Gravity is a body force (scales with mass) and enters as an acceleration instead.
The slosh producer is called last, with the specific force of the other forces, and its pendulum mass is
taken out of the rigid body (see slosh.py).
"""

import numpy as np

from landingSim.core import attitude
from landingSim.dynamics.state import State
from landingSim.environment.gravity import gravityAcceleration
from landingSim.vehicle.mass import massProperties
from landingSim.vehicle.slosh import Slosh, sloshDerivative, totalSloshMass


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
    totalMassFlow = 0.0
    sloshProducers = [p for p in producers if isinstance(p, Slosh)]
    for producer in producers:
        if producer in sloshProducers:
            continue
        producerForce, producerTorque, producerMassFlow = producer(state, commands, t)
        force = force + producerForce
        torque = torque + producerTorque
        totalMassFlow += producerMassFlow

    # The slosh mass rides on its pendulum, so the rigid body is the rest of the vehicle; the pendulum is driven by,
    # and reacts with, the specific force the other forces give the whole vehicle (exact in steady flight).
    # The tank's own specific force depends on the reaction, so start from the whole-vehicle value (exact in steady
    # flight) and refine it once with the reaction it produces.
    sloshMass = totalSloshMass(state, spacecraftConfig) if sloshProducers else 0.0
    rigidMass = props.total - sloshMass
    forceBody = attitude.rotateWorldToBody(state.quaternion, force)
    tankSpecificForceBody = forceBody / props.total
    for _ in range(2):
        sloshForce, sloshTorque = np.zeros(3), np.zeros(3)
        for producer in sloshProducers:
            producerForce, producerTorque, _ = producer(state, commands, t, specificForceBody=tankSpecificForceBody)
            sloshForce, sloshTorque = sloshForce + producerForce, sloshTorque + producerTorque
        tankSpecificForceBody = (forceBody + attitude.rotateWorldToBody(state.quaternion, sloshForce)) / rigidMass
    force = force + sloshForce
    torque = torque + sloshTorque

    specificForce = force / rigidMass                          # what the structure (and the IMU on it) feels
    acceleration = specificForce + gravityAcceleration(environmentConfig, state.position)

    # Euler's rotational equation with principal (diagonal) inertia.
    gyro = attitude.cross3(state.omega, props.inertia * state.omega)
    dOmega = (torque - gyro) / props.inertia

    dPropellant = -totalMassFlow
    dSlosh = sloshDerivative(state, spacecraftConfig, tankSpecificForceBody if sloshProducers else attitude.rotateWorldToBody(state.quaternion, specificForce))

    return np.concatenate([
        state.velocity,
        acceleration,
        attitude.quatDerivative(state.quaternion, state.omega),
        dOmega,
        [dPropellant],
        dSlosh,
    ])


def step(state, producers, commands, t, dt, spacecraftConfig, environmentConfig):
    def f(time, arr):
        return stateDerivative(State.fromArray(arr), producers, commands, time, spacecraftConfig, environmentConfig)

    return State.fromArray(rk4Step(f, state.toArray(), t, dt))
