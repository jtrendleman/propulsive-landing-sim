""" state.py
Vehicle state and command containers, plus the fixed flat-array layout the integrator uses.
Author: Jack Rendleman  Date: 10-15-2026    Version: 1.0.0

Array layout (contract; nothing outside this file indexes by number):
    0:3   position, world frame, m
    3:6   velocity, world frame, m/s
    6:10  quaternion, scalar-first, body-to-world
    10:13 angular velocity, body frame, rad/s
    13    propellant mass, kg
"""

from dataclasses import dataclass

import numpy as np

from landingSim.core import attitude


@dataclass(frozen=True)
class State:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    omega: np.ndarray
    propellantMass: float

    def toArray(self):
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.omega,
            [self.propellantMass],
        ])

    @staticmethod
    def fromArray(arr):
        return State(
            position=arr[0:3].copy(),
            velocity=arr[3:6].copy(),
            quaternion=attitude.normalize(arr[6:10]),
            omega=arr[10:13].copy(),
            propellantMass=float(arr[13]),
        )


@dataclass(frozen=True)
class Commands:
    throttle: float        # 0..1
    gimbal: np.ndarray     # rad, two gimbal axes


def initialState(missionConfig, spacecraftConfig):
    return State(
        position=np.array([missionConfig.initialHorizontalOffset, 0.0, missionConfig.initialAltitude]),
        velocity=np.array([0.0, 0.0, missionConfig.initialVerticalSpeed]),
        quaternion=attitude.identity.copy(),
        omega=np.zeros(3),
        propellantMass=spacecraftConfig.propellant.initialMass,
    )
