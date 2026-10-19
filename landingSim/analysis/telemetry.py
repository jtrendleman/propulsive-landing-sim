""" telemetry.py
Falcon 9 first-stage webcast telemetry: loading, landing-burn detection, and the numbers the vehicle config is fitted to.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.0.0

The files in data/telemetry/ hold time, speed and altitude read off the SpaceX webcast overlay (public domain,
see data/telemetry/README.txt). Speed is the total speed, which is close to vertical by the landing burn.
Altitude is shown to 0.1 km, so everything here works on a 1 s grid and treats low-altitude values as coarse.

landingBurn finds ignition as the first point below 6.5 km where the deceleration of speed jumps by more
than jumpThreshold within 3 s: engine-off, the booster falls near its terminal speed and decelerates slowly;
the burn adds roughly 10 m/s^2 at once. ballisticCoefficient fits m / (Cd A) from the engine-off fall just
before ignition: dv/dt = g - rho v^2 / (2 beta), with ISA density.
"""

from dataclasses import dataclass
import json
import pathlib

import numpy as np

missions = {"CRS-11": "crs11Stage1.json", "CRS-12": "crs12Stage1.json", "SAOCOM 1A": "saocom1aStage1.json", "ZUMA": "zumaStage1.json"}
gravity = 9.81


@dataclass(frozen=True)
class Telemetry:
    name: str
    time: np.ndarray        # s since liftoff
    speed: np.ndarray       # m/s
    altitude: np.ndarray    # m


@dataclass(frozen=True)
class LandingBurn:
    ignitionTime: float     # s since liftoff
    ignitionAltitude: float # m
    ignitionSpeed: float    # m/s
    stopTime: float         # s, first point under 1 m/s
    duration: float         # s, ignition to stop
    ballisticCoefficient: float   # kg/m^2, m / (Cd A) in the fall before ignition (median of the fit window)


def isaDensity(altitude):
    t = 288.15 - 0.0065 * altitude
    return 101325.0 * (t / 288.15) ** (9.80665 / (287.05 * 0.0065)) / (287.05 * t)


def loadTelemetry(folder, name):
    raw = json.loads((pathlib.Path(folder) / missions[name]).read_text(encoding="utf-8"))
    return Telemetry(name, np.asarray(raw["time"], float), np.asarray(raw["velocity"], float), np.asarray(raw["altitude"], float) * 1000.0)


def resample(telemetry, start, stop, step=1.0):
    """(time, speed, altitude) on a regular grid."""
    grid = np.arange(start, stop, step)
    return grid, np.interp(grid, telemetry.time, telemetry.speed), np.interp(grid, telemetry.time, telemetry.altitude)


def landingBurn(telemetry, window=80.0, jumpThreshold=4.0, fitSeconds=6):
    t, v, h = resample(telemetry, telemetry.time[-1] - window, telemetry.time[-1])
    a = np.gradient(v, t)
    candidates = [i for i in range(2, len(t) - 2) if h[i] < 6500.0 and a[i + 2] - a[i - 1] < -jumpThreshold]
    if not candidates:
        raise ValueError(f"no landing burn found in {telemetry.name}")
    b = candidates[0]
    fit = range(max(b - fitSeconds - 2, 0), b - 2)
    betas = [isaDensity(h[i]) * v[i] ** 2 / (2 * (gravity - a[i])) for i in fit if gravity - a[i] > 1.0]
    stop = int(np.argmax(v < 1.0)) if np.any(v < 1.0) else len(t) - 1
    return LandingBurn(float(t[b]), float(h[b]), float(v[b]), float(t[stop]), float(t[stop] - t[b]), float(np.median(betas)))
