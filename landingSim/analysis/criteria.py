""" criteria.py
Scores one logged episode against the success criteria and sorts it into succeeded / softFailed / hardFailed.
Author: Jack Rendleman  Date: 10-17-2026    Version: 1.0.0

Pass/fail thresholds live here, not in the config: the config holds numbers the simulation
uses, the analysis holds the judgement applied afterwards.
"""

from dataclasses import dataclass

import numpy as np

from landingSim.core import attitude
from landingSim.vehicle.landingGear import LandingGear
from landingSim.vehicle.mass import massProperties

propellantReserveFraction = 0.05       # criterion 2: at least this fraction of loaded propellant remains
landingRadius = 0.5                    # m, criterion 3: final footprint centre (the base, between the feet) within this of the pad centre


@dataclass(frozen=True)
class LandingResult:
    endReason: str
    outcome: str                       # "succeeded", "softFailed", or "hardFailed"
    flightTime: float
    touchdownSpeed: float              # m/s, vz at the first row with feet at or below the pad (nan if never)
    finalOffset: float                 # m, horizontal distance of the footprint centre from the pad centre
    finalTilt: float                   # rad
    propellantFraction: float
    peakStrutForce: float              # N, largest single-foot normal force in the log
    peakStrutStress: float             # Pa, peakStrutForce / strutCrossSection
    peakLoadFactor: float              # g, largest sensed acceleration (acceleration minus gravity) / g in the log
    strutPass: bool
    propellantPass: bool
    offsetPass: bool
    loadFactorPass: bool


def evaluate(trajectory, config):
    spacecraft = config.spacecraft
    gear = LandingGear(spacecraft)
    n = len(trajectory.time)

    feetHeight = np.empty(n)
    peakStrutForce = 0.0
    for i in range(n):
        s = trajectory.state(i)
        feetHeight[i] = s.position[2] - massProperties(s, spacecraft).cgHeight
        if feetHeight[i] < 1.0:                           # feet can only touch when they are near the pad
            for _, footForceWorld in gear.footForces(s):
                peakStrutForce = max(peakStrutForce, footForceWorld[2])

    touchdownRows = np.where(feetHeight <= 0.0)[0]
    touchdownSpeed = float(trajectory.states[touchdownRows[0], 5]) if len(touchdownRows) else float("nan")

    final = trajectory.state(n - 1)
    # Measured where the vehicle stands, not at its CG: on a tall booster a fraction of a degree of rocking swings the CG
    # tens of centimetres while the feet do not move.
    finalProps = massProperties(final, spacecraft)
    base = final.position + attitude.rotateBodyToWorld(final.quaternion, np.array([-finalProps.cgLateral[0], -finalProps.cgLateral[1], -finalProps.cgHeight]))
    finalOffset = float(np.linalg.norm(base[0:2]))
    bodyZ = attitude.rotateBodyToWorld(final.quaternion, np.array([0.0, 0.0, 1.0]))
    finalTilt = float(np.arccos(np.clip(bodyZ[2], -1.0, 1.0)))
    propellantFraction = final.propellantMass / spacecraft.propellant.initialMass

    # Sensed (non-gravitational) acceleration: what an accelerometer and the structure feel.
    velocity = trajectory.states[:, 3:6]
    acceleration = np.gradient(velocity, trajectory.time, axis=0) if n > 1 else np.zeros_like(velocity)
    sensed = acceleration - np.array([0.0, 0.0, -config.environment.gravity])
    peakLoadFactor = float(np.max(np.linalg.norm(sensed, axis=1)) / config.environment.gravity)

    peakStrutStress = peakStrutForce / spacecraft.landingGear.strutCrossSection
    strutPass = peakStrutStress * spacecraft.structure.factorOfSafety < spacecraft.structure.yieldStrength
    propellantPass = propellantFraction >= propellantReserveFraction
    offsetPass = finalOffset <= landingRadius
    loadFactorPass = peakLoadFactor <= spacecraft.structure.loadFactorLimit

    if trajectory.endReason != "landed":
        outcome = "hardFailed"
    elif strutPass and propellantPass and offsetPass and loadFactorPass:
        outcome = "succeeded"
    else:
        outcome = "softFailed"

    return LandingResult(
        endReason=trajectory.endReason, outcome=outcome, flightTime=float(trajectory.time[-1]),
        touchdownSpeed=touchdownSpeed, finalOffset=finalOffset, finalTilt=finalTilt,
        propellantFraction=float(propellantFraction), peakStrutForce=float(peakStrutForce),
        peakStrutStress=float(peakStrutStress), peakLoadFactor=peakLoadFactor,
        strutPass=bool(strutPass), propellantPass=bool(propellantPass), offsetPass=bool(offsetPass),
        loadFactorPass=bool(loadFactorPass),
    )
