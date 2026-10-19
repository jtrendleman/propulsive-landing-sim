""" state.py
Vehicle state and command containers, plus the fixed flat-array layout the integrator uses.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.1.0

Array layout (contract; nothing outside this file indexes by number):
    0:3   position of the CG, world frame, m (mission initialAltitude is the FEET altitude)
    3:6   velocity, world frame, m/s
    6:10  quaternion, scalar-first, body-to-world
    10:13 angular velocity, body frame, rad/s
    13    propellant mass, kg (both tanks; split by the mixture ratio, see tank.py)
    14:18 oxidizer-tank slosh pendulum: swing angles [x, y] (rad, body) and their rates (rad/s)
    18:22 fuel-tank slosh pendulum: the same
"""

from dataclasses import dataclass, field

import numpy as np

from landingSim.core import attitude


@dataclass(frozen=True)
class State:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    omega: np.ndarray
    propellantMass: float
    slosh: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def toArray(self):
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.omega,
            [self.propellantMass],
            self.slosh,
        ])

    @staticmethod
    def fromArray(arr):
        return State(
            position=arr[0:3].copy(),
            velocity=arr[3:6].copy(),
            quaternion=attitude.normalize(arr[6:10]),
            omega=arr[10:13].copy(),
            propellantMass=max(0.0, float(arr[13])),  # RK4 can overshoot an empty tank
            slosh=_slosh(arr),
        )


def _slosh(arr):
    """Older logs load with quiet tanks: 14 slots had no slosh, 18 had one tank."""
    if len(arr) >= 22:
        return arr[14:22].copy()
    if len(arr) >= 18:
        return np.concatenate([arr[14:18], np.zeros(4)])
    return np.zeros(8)


@dataclass(frozen=True)
class Commands:
    throttle: float        # 0..1
    gimbal: np.ndarray     # rad, two gimbal axes
    rcs: np.ndarray = field(default_factory=lambda: np.zeros(3))   # -1..1 per body axis, fraction of RCS torque
    fins: np.ndarray = field(default_factory=lambda: np.zeros(2))  # rad, grid fin deflections producing body +x and +y side force

    def toArray(self):
        return np.array([self.throttle, self.gimbal[0], self.gimbal[1], self.rcs[0], self.rcs[1], self.rcs[2], self.fins[0], self.fins[1]])


def initialState(missionConfig, spacecraftConfig):
    return State(
        position=np.array([missionConfig.initialHorizontalOffset, 0.0, missionConfig.initialAltitude]),
        velocity=np.array([0.0, 0.0, missionConfig.initialVerticalSpeed]),
        quaternion=attitude.identity.copy(),
        omega=np.zeros(3),
        propellantMass=spacecraftConfig.propellant.initialMass,
    )
