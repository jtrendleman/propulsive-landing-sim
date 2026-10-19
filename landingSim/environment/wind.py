""" wind.py
Wind velocity field: power-law mean profile, direction shear with altitude, and per-episode Gauss-Markov gusts.
Author: Jack Rendleman  Date: 10-19-2026    Version: 2.1.0

Mean speed: v(z) = meanSpeed * (z / referenceHeight)^profileExponent, held at its 1 m value below 1 m.
Direction turns with altitude by directionShearPerKm (Ekman-layer turning). Both describe the atmospheric
boundary layer: above boundaryLayerHeight the wind keeps its value at the top of the layer (the gradient wind),
so a mission that starts kilometres up does not see the near-surface profile extrapolated there. Gusts are a first-order
Gauss-Markov process per axis (the Dryden spectrum's first-order form) with sigma = turbulenceIntensity
times the local mean speed and correlation time turbulenceLength / local speed. A WindField generates its
gust series once from a seed, so the field is deterministic inside an episode and the sim stays repeatable.
"""

import numpy as np

minimumHeight = 1.0


def windSpeedAt(windConfig, altitude):
    z = min(max(altitude, minimumHeight), windConfig.boundaryLayerHeight)
    return windConfig.meanSpeed * (z / windConfig.referenceHeight) ** windConfig.profileExponent


def windDirectionAt(windConfig, altitude):
    return windConfig.direction + windConfig.directionShearPerKm * min(max(altitude, 0.0), windConfig.boundaryLayerHeight) / 1000.0


def windVelocity(environmentConfig, position, t):
    """Steady part only: mean profile with shear, no gusts."""
    wind = environmentConfig.wind
    speed = windSpeedAt(wind, position[2])
    direction = windDirectionAt(wind, position[2])
    return np.array([speed * np.cos(direction), speed * np.sin(direction), 0.0])


class WindField:
    """Steady profile plus gusts. velocity(position, t) is what the drag producer asks for."""

    def __init__(self, environmentConfig, seed=None, duration=200.0, dt=0.05, referenceAltitude=300.0):
        self.environment = environmentConfig
        wind = environmentConfig.wind
        self.dt = dt
        n = int(duration / dt) + 2
        rng = np.random.default_rng(seed)
        self.unitGusts = np.zeros((n, 3))
        if wind.turbulenceIntensity > 0.0 and wind.meanSpeed > 0.0:
            # Unit-variance Gauss-Markov series per axis; scaled by the local sigma when sampled.
            speedRef = max(windSpeedAt(wind, referenceAltitude), 1.0)
            tau = wind.turbulenceLength / speedRef
            a = np.exp(-dt / tau)
            noise = rng.normal(0.0, np.sqrt(1.0 - a * a), (n, 3))
            noise[:, 2] *= wind.verticalGustFraction
            for k in range(1, n):
                self.unitGusts[k] = a * self.unitGusts[k - 1] + noise[k]

    def gust(self, position, t):
        wind = self.environment.wind
        if wind.turbulenceIntensity <= 0.0:
            return np.zeros(3)
        k = min(int(t / self.dt), len(self.unitGusts) - 2)
        s = (t - k * self.dt) / self.dt
        unit = (1 - s) * self.unitGusts[k] + s * self.unitGusts[k + 1]
        sigma = wind.turbulenceIntensity * windSpeedAt(wind, position[2])
        direction = windDirectionAt(wind, position[2])
        along = np.array([np.cos(direction), np.sin(direction), 0.0])
        across = np.array([-np.sin(direction), np.cos(direction), 0.0])
        return sigma * (unit[0] * along + unit[1] * across + unit[2] * np.array([0.0, 0.0, 1.0]))

    def velocity(self, position, t):
        return windVelocity(self.environment, position, t) + self.gust(position, t)
