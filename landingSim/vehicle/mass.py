""" mass.py
Total mass, CG height, and principal inertia of the vehicle as propellant drains.
Author: Jack Rendleman  Date: 10-16-2026    Version: 1.0.0

Propellant is modeled as a solid cylinder of liquid standing on the tank floor.
inertiaDry is about the dry CG; the result is about the total CG (parallel axis).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassProperties:
    total: float
    cgHeight: float
    inertia: np.ndarray


def massProperties(state, spacecraftConfig):
    prop = spacecraftConfig.propellant
    dry = spacecraftConfig.mass
    mP = state.propellantMass

    tankArea = np.pi * prop.tankRadius**2
    liquidHeight = mP / (prop.density * tankArea)
    cgPropellant = prop.tankBottomHeight + liquidHeight / 2

    total = dry.dryMass + mP
    cgHeight = (dry.dryMass * dry.cgHeightDry + mP * cgPropellant) / total

    # Solid cylinder about its own CG: transverse (x, y) and axial (z).
    iPropTransverse = mP * (3 * prop.tankRadius**2 + liquidHeight**2) / 12
    iPropAxial = mP * prop.tankRadius**2 / 2

    # Parallel-axis shift of both bodies to the total CG along the rocket axis.
    dDry = dry.cgHeightDry - cgHeight
    dProp = cgPropellant - cgHeight
    iTransverse = dry.dryMass * dDry**2 + iPropTransverse + mP * dProp**2
    inertia = np.array([
        dry.inertiaDry[0] + iTransverse,
        dry.inertiaDry[1] + iTransverse,
        dry.inertiaDry[2] + iPropAxial,
    ])

    return MassProperties(total=total, cgHeight=cgHeight, inertia=inertia)
