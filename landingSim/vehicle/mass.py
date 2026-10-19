""" mass.py
Total mass, CG height, and principal inertia of the vehicle as propellant drains.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.2.0

Propellant sits where the tank geometry puts it (tank.py: oxidizer and fuel tanks with domed bottoms), each
pool treated as a solid of its surface radius and height for inertia.
inertiaDry is about the dry CG; the result is about the total CG (parallel axis).
"""

from dataclasses import dataclass

import numpy as np

from landingSim.vehicle.tank import tankFill, tankMasses, tanks


@dataclass(frozen=True)
class MassProperties:
    total: float
    cgHeight: float
    inertia: np.ndarray
    cgLateral: np.ndarray = None      # m, CG offset from the body axis in body [x, y]; the propellant is on the axis


def massProperties(state, spacecraftConfig):
    prop = spacecraftConfig.propellant
    dry = spacecraftConfig.mass
    mP = state.propellantMass
    total = dry.dryMass + mP

    # Each tank's liquid as a solid cylinder of its pool's surface radius and height, at the pool centroid.
    pools = []
    for name, mass in zip(tanks, tankMasses(mP, prop)):
        tank = getattr(prop, name)
        fill = tankFill(mass, tank)
        pools.append((mass, tank.tankBottomHeight + fill.centroidHeight, fill.surfaceRadius, fill.height))

    cgHeight = (dry.dryMass * dry.cgHeightDry + sum(m * z for m, z, _, _ in pools)) / total

    # Parallel-axis shift of every body to the total CG along the rocket axis.
    iTransverse = dry.dryMass * (dry.cgHeightDry - cgHeight) ** 2
    iAxial = 0.0
    for m, z, r, h in pools:
        iTransverse += m * (3 * r**2 + h**2) / 12 + m * (z - cgHeight) ** 2
        iAxial += m * r**2 / 2
    inertia = np.array([
        dry.inertiaDry[0] + iTransverse,
        dry.inertiaDry[1] + iTransverse,
        dry.inertiaDry[2] + iAxial,
    ])

    cgLateral = np.array(dry.cgLateralOffset, float) * dry.dryMass / total
    return MassProperties(total=total, cgHeight=cgHeight, inertia=inertia, cgLateral=cgLateral)
