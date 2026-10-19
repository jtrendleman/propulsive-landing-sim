""" aero.py
Drag producer with angle-of-attack dependence: Cd rises with crossflow and the centre of pressure moves along the body.
Author: Jack Rendleman  Date: 10-19-2026    Version: 2.1.0

Drag opposes the air-relative velocity with Cd(alpha) = Cd0 (1 + dragAlphaFactor sin^2 alpha), where alpha
is the angle between the body axis and the relative wind, and acts at a centre of pressure that blends from
cpHeight (axial flow) to cpHeightCrossflow (90 degrees, mid-body for a cylinder) by sin^2 alpha. A crosswind
large angle of attack is crossflow, so the crossflow station matters. The grid fins' drag acts at the fins.
While the engine fires, the plume of the single central nozzle shields the body and its drag falls to
plumeDragFactor of the engine-off value (supersonic retropropulsion, Korzun et al. 2008). The mission starts
near Mach 1.1 and the drag coefficient is fitted to flight telemetry at about Mach 1; there is no further Mach
dependence.
"""

import numpy as np

from landingSim.core import attitude
from landingSim.environment.atmosphere import airDensity
from landingSim.environment.wind import windVelocity
from landingSim.vehicle.mass import massProperties


def relativeWind(state, environmentConfig, windField, t):
    wind = windField.velocity(state.position, t) if windField is not None else windVelocity(environmentConfig, state.position, t)
    return state.velocity - wind


def angleOfAttack(state, vRel):
    speed = np.linalg.norm(vRel)
    if speed < 1e-9:
        return 0.0
    zBody = attitude.rotateBodyToWorld(state.quaternion, np.array([0.0, 0.0, 1.0]))
    return float(np.arccos(np.clip(abs(zBody @ vRel) / speed, 0.0, 1.0)))


class Drag:
    def __init__(self, spacecraftConfig, environmentConfig, windField=None):
        self.config = spacecraftConfig
        self.environment = environmentConfig
        self.windField = windField

    def __call__(self, state, commands, t):
        aero = self.config.aero
        rho = airDensity(self.environment, state.position[2])
        vRel = relativeWind(state, self.environment, self.windField, t)
        speed = np.linalg.norm(vRel)
        if speed < 1e-9:
            return np.zeros(3), np.zeros(3), 0.0
        alpha = angleOfAttack(state, vRel)
        props = massProperties(state, self.config)
        # Body drag rises with crossflow and acts at the body's centre of pressure; the grid fins' drag acts at the
        # fins, at the top of the vehicle. Flying base-first, that fin drag is what keeps the descent stable.
        plume = aero.plumeDragFactor if commands is not None and commands.throttle > 0.0 else 1.0
        bodyForce = -0.5 * rho * aero.dragCoefficient * (1 + aero.dragAlphaFactor * np.sin(alpha) ** 2) * aero.referenceArea * speed * vRel * plume
        finForce = -0.5 * rho * aero.dragCoefficient * aero.gridFinArea * speed * vRel
        cpHeight = aero.cpHeight + (aero.cpHeightCrossflow - aero.cpHeight) * np.sin(alpha) ** 2
        torqueBody = np.zeros(3)
        for force, height in ((bodyForce, cpHeight), (finForce, aero.finHeight)):
            rBody = np.array([-props.cgLateral[0], -props.cgLateral[1], height - props.cgHeight])
            torqueBody += attitude.cross3(rBody, attitude.rotateWorldToBody(state.quaternion, force))
        return bodyForce + finForce, torqueBody, 0.0


class GridFins:
    """Side force from deflected grid fins at the top of the vehicle: F_body = q * A_fin * CLalpha * [dx, dy, 0]."""

    def __init__(self, spacecraftConfig, environmentConfig, windField=None):
        self.config = spacecraftConfig
        self.environment = environmentConfig
        self.windField = windField

    def dynamicPressure(self, state, t):
        rho = airDensity(self.environment, state.position[2])
        vRel = relativeWind(state, self.environment, self.windField, t)
        return 0.5 * rho * float(vRel @ vRel)

    def __call__(self, state, commands, t):
        aero = self.config.aero
        if aero.gridFinArea <= 0.0:
            return np.zeros(3), np.zeros(3), 0.0
        q = self.dynamicPressure(state, t)
        deflection = np.clip(np.asarray(commands.fins, float), -aero.maxFinDeflection, aero.maxFinDeflection)
        forceBody = q * aero.gridFinArea * aero.finLiftSlope * np.array([deflection[0], deflection[1], 0.0])
        props = massProperties(state, self.config)
        rBody = np.array([-props.cgLateral[0], -props.cgLateral[1], aero.finHeight - props.cgHeight])
        torqueBody = attitude.cross3(rBody, forceBody)
        return attitude.rotateBodyToWorld(state.quaternion, forceBody), torqueBody, 0.0
