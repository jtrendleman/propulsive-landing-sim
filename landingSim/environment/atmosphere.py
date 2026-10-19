""" atmosphere.py
International Standard Atmosphere troposphere: temperature, pressure, and density against altitude.
Author: Jack Rendleman  Date: 10-17-2026    Version: 2.0.0

U.S. Standard Atmosphere 1976 below 11 km: T = T0 - L h, p = p0 (T / T0)^(g / (R L)), rho = p / (R T).
Altitudes below zero are treated as sea level.
"""

import numpy as np


def temperature(environmentConfig, altitude):
    return environmentConfig.seaLevelTemperature - environmentConfig.lapseRate * max(altitude, 0.0)


def pressure(environmentConfig, altitude):
    env = environmentConfig
    exponent = env.gravity / (env.gasConstant * env.lapseRate)
    return env.seaLevelPressure * (temperature(env, altitude) / env.seaLevelTemperature) ** exponent


def airDensity(environmentConfig, altitude):
    env = environmentConfig
    return pressure(env, altitude) / (env.gasConstant * temperature(env, altitude))
