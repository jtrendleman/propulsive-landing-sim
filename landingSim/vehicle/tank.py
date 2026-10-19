""" tank.py
Propellant tank geometry: a cylinder with a half-ellipsoid (domed) bottom, and where the liquid sits in it.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.1.0

The dome has the tank radius R across and depth d = domeRatio * R (0.707 is the classic elliptical bulkhead).
tankBottomHeight is the dome's lowest point. Heights z below are measured up from there.

In the dome the liquid surface at height z has radius r(z) = R sqrt(1 - ((d - z) / d)^2), so the wetted area
is A(z) = pi R^2 (2 z / d - z^2 / d^2) and the volume filled to height h is
    V(h) = pi R^2 (h^2 / d - h^3 / (3 d^2))                        for h <= d,
plus a cylinder of area pi R^2 above the dome. A nearly empty tank's propellant therefore pools in the bowl
(400 kg of water-density liquid in a 1.8 m dome: about 22 cm deep and 1 m across) instead of spreading as a
film over the whole floor. domeRatio = 0 gives the flat-bottomed cylinder.

The vehicle has two tanks (oxidizer above fuel, as on a kerosene booster). Both drain together at the engine's
mixture ratio, so one propellant mass in the state splits into the two by tankMasses.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TankFill:
    height: float            # m, liquid surface above the dome's lowest point
    surfaceRadius: float     # m, radius of the free surface
    meanDepth: float         # m, volume / free-surface area: the depth of the equivalent flat-bottomed pool
    centroidHeight: float    # m, liquid centroid above the dome's lowest point


tanks = ("oxidizer", "fuel")


def tankMasses(propellantMass, propellantConfig):
    """(oxidizer, fuel) kg: the engine burns them at the mixture ratio, so they stay in that ratio."""
    r = propellantConfig.mixtureRatio
    m = max(propellantMass, 0.0)
    return m * r / (1 + r), m / (1 + r)


def domeDepth(tankConfig):
    return tankConfig.domeRatio * tankConfig.tankRadius


def domeVolume(tankConfig):
    return 2.0 / 3.0 * np.pi * tankConfig.tankRadius**2 * domeDepth(tankConfig)


def bowlVolume(h, R, d):
    return np.pi * R**2 * (h**2 / d - h**3 / (3 * d**2))


def tankFill(liquidMass, tankConfig):
    R = tankConfig.tankRadius
    d = domeDepth(tankConfig)
    volume = max(liquidMass, 0.0) / tankConfig.density
    if volume <= 0.0:
        return TankFill(0.0, 0.0, 0.0, 0.0)
    area = np.pi * R**2

    if d > 0.0 and volume <= domeVolume(tankConfig):
        # With u = h / d and c = V / (pi R^2 d), V(h) = V is u^3 - 3 u^2 + 3 c = 0; its root in [0, 1] in closed form.
        c = min(volume / (area * d), 2.0 / 3.0)
        h = d * (1 + 2 * np.cos(np.arccos(np.clip(1 - 1.5 * c, -1.0, 1.0)) / 3 - 2 * np.pi / 3))
        r = R * np.sqrt(max(1.0 - ((d - h) / d) ** 2, 0.0))
        moment = area * (2 * h**3 / (3 * d) - h**4 / (4 * d**2))   # integral of z A(z) dz over the bowl
    else:
        h = d + (volume - domeVolume(tankConfig)) / area
        r = R
        moment = (area * (2 * d**3 / (3 * d) - d**4 / (4 * d**2)) if d > 0.0 else 0.0) + area * (h**2 - d**2) / 2

    return TankFill(height=float(h), surfaceRadius=float(r), meanDepth=float(volume / (np.pi * r**2)) if r > 0 else 0.0,
                    centroidHeight=float(moment / volume))
