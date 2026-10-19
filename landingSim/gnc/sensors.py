""" sensors.py
The flight computer's view of the vehicle: IMU, GPS with latency, radar altimeter, chamber pressure, tank gauging, actuator feedback.
Author: Jack Rendleman  Date: 10-18-2026    Version: 2.0.0

This is the only module that reads the true state; everything downstream (navigation, guidance, control,
the RL policy) works from a Measurement. Noise is drawn from a per-episode generator, so an episode is
reproducible from its seed. Per-episode biases model calibration errors. The IMU reports specific force
in the BODY frame (the navigator rotates it with its own attitude estimate) and body rates. The attitude
itself is only measured once, as the alignment at the start of the landing; after that the navigator
integrates the gyros. The sensors ride on the real vehicle (trueSpacecraft, trueEnvironment): a Monte
Carlo draw with a weak engine shows up as low chamber pressure, not as a number the controller is told.
"""

from dataclasses import dataclass

import numpy as np

from landingSim.core import attitude
from landingSim.environment.atmosphere import pressure
from landingSim.vehicle.mass import massProperties


@dataclass
class Measurement:
    t: float
    specificForceBody: np.ndarray    # m/s^2, body frame, accelerometer (acceleration minus gravity) + bias + noise
    omega: np.ndarray                # rad/s, body frame, gyro + bias + noise
    alignment: np.ndarray | None     # initial attitude fix (first sample only), else None
    gps: tuple | None                # (position, velocity, timeOfValidity) when a fix arrives, else None
    altimeter: float | None          # m, base height above the pad when in range, else None
    propellantGauge: float           # kg
    thrust: float                    # N, from chamber pressure
    throttleFeedback: float          # throttle valve position
    gimbalFeedback: np.ndarray       # rad, measured gimbal angles


class Sensors:
    def __init__(self, config, seed=None):
        self.config = config
        self.sensorConfig = config.sensors
        self.reseed(seed)

    def reseed(self, seed=None, trueSpacecraft=None, trueEnvironment=None):
        sc = self.sensorConfig
        self.trueSpacecraft = trueSpacecraft if trueSpacecraft is not None else self.config.spacecraft
        self.trueEnvironment = trueEnvironment if trueEnvironment is not None else self.config.environment
        self.rng = np.random.default_rng(seed)
        self.accelBias = self.rng.normal(0.0, sc.accelBias, 3)
        self.gyroBias = self.rng.normal(0.0, sc.gyroBias, 3)
        self.chamberBias = self.rng.normal(0.0, sc.chamberPressureBias)
        self.gaugeBias = self.rng.normal(0.0, sc.propellantGaugeBias)
        self.lastVelocity = None
        self.lastTime = None
        self.nextGpsTime = 0.0
        self.history = []                # (t, position, velocity) of the true state, for GPS latency
        self.achieved = None             # actuator positions delivered last step
        self.aligned = False

    def observeActuators(self, achieved):
        self.achieved = achieved

    def measure(self, state, t):
        sc = self.sensorConfig
        g = np.array([0.0, 0.0, -self.config.environment.gravity])
        if self.lastVelocity is None or t <= self.lastTime:
            accelWorld = np.zeros(3)
        else:
            accelWorld = (state.velocity - self.lastVelocity) / (t - self.lastTime)
        self.lastVelocity, self.lastTime = state.velocity.copy(), t
        specificForceBody = (attitude.rotateWorldToBody(state.quaternion, accelWorld - g)
                             + self.accelBias + self.rng.normal(0.0, sc.accelNoise, 3))
        omega = state.omega + self.gyroBias + self.rng.normal(0.0, sc.gyroNoise, 3)

        alignment = None
        if not self.aligned:
            tilt = self.rng.normal(0.0, sc.alignmentError, 3)
            angle = np.linalg.norm(tilt)
            alignment = attitude.multiply(state.quaternion, attitude.fromAxisAngle(tilt / angle, angle)) if angle > 0 else state.quaternion.copy()
            self.aligned = True

        self.history.append((t, state.position.copy(), state.velocity.copy()))
        while len(self.history) > 2 and self.history[1][0] <= t - sc.gpsLatency - 0.05:
            self.history.pop(0)
        gps = None
        if t + 1e-9 >= self.nextGpsTime:
            older = [h for h in self.history if h[0] <= t - sc.gpsLatency + 1e-9]
            tValid, position, velocity = older[-1] if older else self.history[0]
            gps = (position + self.rng.normal(0.0, sc.gpsPositionNoise, 3),
                   velocity + self.rng.normal(0.0, sc.gpsVelocityNoise, 3), tValid)
            self.nextGpsTime = t + 1.0 / sc.gpsRate

        feet = state.position[2] - massProperties(state, self.trueSpacecraft).cgHeight      # the real base height
        altimeter = None
        if feet <= sc.altimeterRange:
            altimeter = feet + self.rng.normal(0.0, max(sc.altimeterNoise, sc.altimeterNoiseFraction * abs(feet)))

        gauge = max(state.propellantMass * (1.0 + self.gaugeBias) + self.rng.normal(0.0, sc.propellantGaugeNoise), 0.0)

        throttle = float(self.achieved.throttle) if self.achieved is not None else 0.0
        gimbal = np.asarray(self.achieved.gimbal, float) if self.achieved is not None else np.zeros(2)
        eng = self.trueSpacecraft.engine
        if throttle >= eng.minThrottle and state.propellantMass > 0.0:
            ambient = pressure(self.trueEnvironment, state.position[2])
            trueThrust = throttle * (eng.maxThrust + eng.nozzleExitArea * (self.trueEnvironment.seaLevelPressure - ambient)) * eng.count
            thrust = trueThrust * (1.0 + self.chamberBias + self.rng.normal(0.0, sc.chamberPressureNoise))
            throttleFeedback = throttle + self.rng.normal(0.0, 0.002)
        else:
            thrust, throttleFeedback = 0.0, 0.0
        gimbalFeedback = gimbal + self.rng.normal(0.0, sc.gimbalSensorNoise, 2)
        return Measurement(t, specificForceBody, omega, alignment, gps, altimeter, gauge, thrust, throttleFeedback, gimbalFeedback)
