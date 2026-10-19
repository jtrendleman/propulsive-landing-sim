""" slosh.py
Propellant slosh as the first-mode equivalent pendulum of the liquid in a domed tank (NASA SP-106, Abramson 1966), reacting on the vehicle.
Author: Jack Rendleman  Date: 10-19-2026    Version: 2.1.0

The liquid's free surface has radius r and the pool a mean depth hm (tank.py). With xi = 1.841, the root of
the Bessel derivative for the first antisymmetric mode, the SP-106 equivalent-cylinder results give
    slosh mass fraction   m1 / m = (2 r / (xi hm)) tanh(xi hm / r) / (xi^2 - 1)
    pendulum length       L = r / (xi tanh(xi hm / r))         so w1^2 = a / L with a the axial specific force
    mass station          the pool's base + hm/2 - (r / xi) tanh(xi hm / (2 r)), hinge L above it.
For a nearly empty domed tank the pool is a shallow lump in the bowl. Lamb's shallow-water solution for a
bowl gives exactly a pendulum of the dome's bottom curvature radius R^2 / d; the equivalent cylinder tends
to 4 / xi^2 = 1.18 times that as the pool empties (frequency 8 % low), and is about 5 % long at the landing
fill (400 kg here: 1.91 rad/s at 1 g against 1.96).

Dynamics, per body axis i in {x, y}, with theta_i the swing angle, f the tank's specific force (body frame),
a = f_z, and q_i the damping and wall forces on the mass along its path:
    theta_i'' = -(a sin(theta_i) + f_i cos(theta_i)) / L + q_i / (m1 L)
    rod tension T = m1 (L |theta'|^2 + a cos(theta_x) cos(theta_y) - f_x sin(theta_x) - f_y sin(theta_y))
The full sin / cos restoring force is kept, so the liquid climbing the curved wall is pulled back the way
the bowl pulls it back. The rod pulls the hinge with T toward the mass, and q reacts on the tank wall.
The mass m1 is carried only through the rod, so the rigid body is the total mass minus m1 in every
direction; in steady flight T balances exactly what m1 would have needed, and the vehicle accelerates at
F / total mass. Past the angle at which half the pool is against the cylinder wall, a stiff damped stop
holds the liquid inside the tank. Damping of an unbaffled tank (NASA SP-8031).

Each tank (oxidizer above, fuel below) has its own pendulum: state.slosh[0:4] is the oxidizer tank's swing
angles and rates, state.slosh[4:8] the fuel tank's.
"""

from dataclasses import dataclass

import numpy as np

from landingSim.core import attitude
from landingSim.vehicle.mass import massProperties
from landingSim.vehicle.tank import tankFill, tankMasses, tanks

besselRoot = 1.841
wallStiffness = 50.0          # wall stop stiffness in units of the pendulum's own w1^2
wallDamping = 0.5             # damping ratio of the wall stop


@dataclass(frozen=True)
class SloshParameters:
    mass: float             # kg, first-mode slosh mass
    length: float           # m, equivalent pendulum length
    omega: float            # rad/s, natural frequency at the current axial specific force
    damping: float          # ratio
    hingeHeight: float      # m above the base plane, where the rod force acts on the tank
    maxAngle: float         # rad, swing at which the wall stop starts


def tankSlice(index):
    return slice(4 * index, 4 * index + 4)


def sloshParameters(state, spacecraftConfig, axialSpecificForce, tankName="oxidizer"):
    prop = spacecraftConfig.propellant
    tank = getattr(prop, tankName)
    liquidMass = tankMasses(state.propellantMass, prop)[tanks.index(tankName)]
    fill = tankFill(liquidMass, tank)
    r, hm = fill.surfaceRadius, fill.meanDepth
    if liquidMass <= 1e-6 or r < 1e-3 or hm < 1e-4 or not prop.sloshEnabled:
        return SloshParameters(0.0, 1.0, 0.0, prop.sloshDamping, tank.tankBottomHeight, 0.0)
    xi = besselRoot
    t = np.tanh(xi * hm / r)
    massFraction = min((2 * r / (xi * hm)) * t / (xi**2 - 1), 1.0)
    length = r / (xi * t)
    omega = np.sqrt(max(axialSpecificForce, 0.5) / length)
    station = tank.tankBottomHeight + fill.height - hm + hm / 2 - (r / xi) * np.tanh(xi * hm / (2 * r))
    maxOffset = tank.tankRadius - r / 2                 # the pool centre can swing until half the pool meets the wall
    maxAngle = float(np.arcsin(min(maxOffset / length, 1.0)))
    return SloshParameters(float(massFraction * liquidMass), float(length), float(omega), prop.sloshDamping,
                           float(station + length), maxAngle)


def pathForce(theta, rate, params):
    """Damping and wall-stop acceleration of the slosh mass along its path, per axis (q / (m1 L) times L)."""
    w2 = params.omega**2
    accel = -2 * params.damping * params.omega * rate
    excess = np.abs(theta) - params.maxAngle
    beyond = excess > 0.0
    k = wallStiffness * w2
    accel = accel - np.where(beyond, k * np.sign(theta) * excess + 2 * wallDamping * np.sqrt(k) * rate, 0.0)
    return accel


def totalSloshMass(state, spacecraftConfig):
    return sum(sloshParameters(state, spacecraftConfig, 9.81, name).mass for name in tanks)


def sloshDerivative(state, spacecraftConfig, specificForceBody):
    """d/dt of both tanks' [theta_x, theta_y, rate_x, rate_y], driven by the specific force of the tank the liquid rides in."""
    a = max(specificForceBody[2], 0.0)
    out = []
    for i, name in enumerate(tanks):
        params = sloshParameters(state, spacecraftConfig, a, name)
        mode = state.slosh[tankSlice(i)]
        theta, rate = mode[0:2], mode[2:4]
        if params.mass <= 0.0:
            out.append(np.concatenate([np.zeros(2), -5.0 * rate]))      # empty tank: the mode decays away
            continue
        accel = -(a * np.sin(theta) + specificForceBody[0:2] * np.cos(theta)) / params.length + pathForce(theta, rate, params)
        out.append(np.concatenate([rate, accel]))
    return np.concatenate(out)


def sloshReaction(state, spacecraftConfig, specificForceBody, tankName="oxidizer"):
    """(forceBody, params): what one tank's swinging liquid does to the tank, through the rod and along its path."""
    a = max(specificForceBody[2], 0.0)
    params = sloshParameters(state, spacecraftConfig, a, tankName)
    if params.mass <= 0.0:
        return np.zeros(3), params
    mode = state.slosh[tankSlice(tanks.index(tankName))]
    theta, rate = mode[0:2], mode[2:4]
    s, c = np.sin(theta), np.cos(theta)
    tension = params.mass * (params.length * float(rate @ rate) + a * c[0] * c[1] - float(specificForceBody[0:2] @ s))
    tension = max(tension, 0.0)                                  # liquid cannot pull on the tank
    q = params.mass * params.length * pathForce(theta, rate, params)
    lateral = tension * s - q * c
    axial = -tension * c[0] * c[1] - float(q @ s)
    return np.array([lateral[0], lateral[1], axial]), params


class Slosh:
    """Force producer: both tanks' slosh pendulums reacting on the vehicle, each at its hinge. The integrator passes
    the tank's specific force (from the other forces) and takes the slosh masses out of the rigid body."""

    def __init__(self, spacecraftConfig):
        self.config = spacecraftConfig

    def __call__(self, state, commands, t, specificForceBody=(0.0, 0.0, 9.81)):
        f = np.asarray(specificForceBody, float)
        props = massProperties(state, self.config)
        forceBody, torqueBody = np.zeros(3), np.zeros(3)
        for name in tanks:
            force, params = sloshReaction(state, self.config, f, name)
            if params.mass <= 0.0:
                continue
            rBody = np.array([-props.cgLateral[0], -props.cgLateral[1], params.hingeHeight - props.cgHeight])
            forceBody += force
            torqueBody += attitude.cross3(rBody, force)
        return attitude.rotateBodyToWorld(state.quaternion, forceBody), torqueBody, 0.0
