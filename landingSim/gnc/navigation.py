""" navigation.py
Navigation: gyro-integrated attitude, and a per-axis Kalman filter fusing body-frame IMU specific force with delayed GPS fixes and the radar altimeter.
Author: Jack Rendleman  Date: 10-18-2026    Version: 3.0.0

Attitude starts from the alignment fix and is propagated by integrating the measured body rates, so gyro
bias turns into a slowly growing attitude error, as on a real inertial unit. The accelerometer is rotated
into the world frame with that estimate. GPS fixes arrive gpsLatency late and are carried forward to the
current time with their own velocity and the IMU acceleration before the update. The altimeter corrects vertical position below
its range, converted to CG height with the gauged propellant. Never reads the true state: it hands the
state to the sensors and works from the Measurement. With sensors disabled it returns the true state.
"""

import numpy as np

from landingSim.core import attitude
from landingSim.dynamics.state import State
from landingSim.gnc.sensors import Sensors
from landingSim.vehicle.mass import massProperties


def navigate(state):
    """Perfect navigation (kept as the seam the controller used before sensors existed)."""
    return state


class Navigator:
    def __init__(self, config, seed=None):
        self.config = config
        self.enabled = config.sensors.enabled
        self.sensors = Sensors(config, seed)
        self.reset()

    def reseed(self, seed=None, trueSpacecraft=None, trueEnvironment=None):
        self.sensors.reseed(seed, trueSpacecraft, trueEnvironment)
        self.reset()

    def observeActuators(self, achieved):
        self.sensors.observeActuators(achieved)

    def reset(self):
        self.x = None                      # (3, 2): position, velocity per axis
        self.P = None                      # (3, 2, 2)
        self.q = None                      # attitude estimate
        self.lastTime = None
        self.lastMeasurement = None
        self.accel = np.zeros(3)

    def __call__(self, state, t):
        if not self.enabled:
            return state
        m = self.sensors.measure(state, t)
        self.lastMeasurement = m
        sc = self.config.sensors
        g = np.array([0.0, 0.0, -self.config.environment.gravity])
        if self.q is None:
            self.q = m.alignment.copy()
        if self.x is None:
            position, velocity, tValid = m.gps
            self.x = np.column_stack([position + velocity * (t - tValid), velocity]).astype(float)
            self.P = np.array([np.diag([sc.gpsPositionNoise**2, sc.gpsVelocityNoise**2])] * 3)
            self.lastTime = t
        else:
            dt = max(t - self.lastTime, 0.0)
            self.lastTime = t
            if dt > 0.0:
                self.q = attitude.normalize(self.q + attitude.quatDerivative(self.q, m.omega) * dt)
                accel = attitude.rotateBodyToWorld(self.q, m.specificForceBody) + g
                self.accel = accel
                F = np.array([[1.0, dt], [0.0, 1.0]])
                B = np.array([0.5 * dt**2, dt])
                sigmaA = sc.accelNoise + sc.accelBias + self.config.environment.gravity * sc.alignmentError
                Q = sigmaA**2 * np.outer(B, B)
                for k in range(3):
                    self.x[k] = F @ self.x[k] + B * accel[k]
                    self.P[k] = F @ self.P[k] @ F.T + Q
            if m.gps is not None:
                position, velocity, tValid = m.gps
                lag = t - tValid
                R = np.diag([sc.gpsPositionNoise**2, sc.gpsVelocityNoise**2])
                for k in range(3):
                    z = np.array([position[k] + velocity[k] * lag + 0.5 * self.accel[k] * lag**2, velocity[k] + self.accel[k] * lag])
                    K = self.P[k] @ np.linalg.inv(self.P[k] + R)
                    self.x[k] = self.x[k] + K @ (z - self.x[k])
                    self.P[k] = (np.eye(2) - K) @ self.P[k]
            if m.altimeter is not None:
                cgHeight = massProperties(State(self.x[:, 0], self.x[:, 1], self.q, m.omega, m.propellantGauge), self.config.spacecraft).cgHeight
                H = np.array([1.0, 0.0])
                noise = max(sc.altimeterNoise, sc.altimeterNoiseFraction * abs(m.altimeter))
                S = H @ self.P[2] @ H + noise**2
                K = self.P[2] @ H / S
                self.x[2] = self.x[2] + K * (m.altimeter + cgHeight - self.x[2, 0])
                self.P[2] = (np.eye(2) - np.outer(K, H)) @ self.P[2]
        return State(position=self.x[:, 0].copy(), velocity=self.x[:, 1].copy(), quaternion=self.q.copy(),
                     omega=m.omega, propellantMass=m.propellantGauge)
