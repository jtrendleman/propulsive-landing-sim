""" landingGear.py
Ground contact producer: progressive strut springs, damping, and regularised Coulomb friction against a tilted pad.
Author: Jack Rendleman  Date: 10-19-2026    Version: 3.0.0

Feet sit on the base plane, evenly spaced on a circle of footRadius, foot 0 on +x body. Contact is
measured along the pad normal, so a sloped pad works the same as a flat one. Normal force:
k x (1 + hardening (x / stroke)^2) - c v_n, never negative. Tangential force: stick-slip friction. A foot
that touches down pins an anchor point; friction is a stiff tangential spring (tangentialStiffness) and
damper (footFriction) back to that anchor, capped at mu N. Past the cap the foot slides and the anchor is
dragged along so the spring sits exactly at mu N. That gives true stiction (a vehicle stands on a slope or
against a sloshing tank without creeping) while the spring stays far inside RK4's stability limit, which a
steep velocity-regularised Coulomb law does not at booster mass. Anchors are updated between integration
steps (commit), so every RK4 stage of a step sees the same anchors.
Returns zero without any work when the CG is higher above the pad than any foot can reach.
"""

import numpy as np

from landingSim.core import attitude
from landingSim.environment.ground import padNormal
from landingSim.vehicle.mass import massProperties


class LandingGear:
    def __init__(self, spacecraftConfig, environmentConfig=None):
        self.config = spacecraftConfig
        self.environment = environmentConfig
        gear = spacecraftConfig.landingGear
        angles = 2 * np.pi * np.arange(gear.strutCount) / gear.strutCount
        self.footOffsetsXY = [(gear.footRadius * np.cos(a), gear.footRadius * np.sin(a)) for a in angles]
        self._footXY = np.array(self.footOffsetsXY)
        self.normal = padNormal(environmentConfig) if environmentConfig is not None else np.array([0.0, 0.0, 1.0])
        self.anchors = None            # world point each foot is stuck to, or None while it is off the pad

    def footStates(self, state, cgHeight, cgLateral=(0.0, 0.0)):
        """Each foot as (rBody from CG, world position, world velocity)."""
        rBody, posWorld, velWorld = self._feet(state, cgHeight, cgLateral)
        return [(rBody[i], posWorld[i], velWorld[i]) for i in range(len(rBody))]

    def _feet(self, state, cgHeight, cgLateral=(0.0, 0.0)):
        rotation = attitude.quatToRotationMatrix(state.quaternion)
        rBody = np.column_stack([self._footXY - np.asarray(cgLateral), np.full(len(self._footXY), -cgHeight)])
        posWorld = state.position + rBody @ rotation.T
        omegaCross = np.cross(np.broadcast_to(state.omega, rBody.shape), rBody)
        velWorld = state.velocity + omegaCross @ rotation.T
        return rBody, posWorld, velWorld

    def _outOfReach(self, state, cgHeight):
        gear = self.config.landingGear
        return state.position @ self.normal > np.hypot(cgHeight, gear.footRadius + 0.1) + 1e-6

    def _contactForces(self, posWorld, velWorld):
        """World-frame force per foot (n, 3) from penetration along the pad normal, and the penetrations."""
        gear = self.config.landingGear
        n = self.normal
        penetration = -(posWorld @ n)
        contact = penetration > 0.0
        vNormal = velWorld @ n
        spring = gear.strutStiffness * penetration * (1 + gear.strutHardening * (penetration / gear.strutStroke) ** 2)
        normal = np.where(contact, np.maximum(spring - gear.strutDamping * vNormal, 0.0), 0.0)
        vTangent = velWorld - np.outer(vNormal, n)
        anchors = posWorld if self.anchors is None else np.where(np.isnan(self.anchors), posWorld, self.anchors)
        stretch = posWorld - anchors
        stretch = stretch - np.outer(stretch @ n, n)
        tangential = -(gear.tangentialStiffness * stretch + gear.footFriction * vTangent)
        limit = gear.frictionCoefficient * normal
        size = np.linalg.norm(tangential, axis=1)
        scale = np.where(size > limit, limit / np.maximum(size, 1e-12), 1.0)
        tangential = np.where(contact[:, None], tangential * scale[:, None], 0.0)
        return np.outer(normal, n) + tangential, penetration

    def commit(self, state):
        """Between steps: pin newly landed feet, drag the anchors of sliding feet, release feet off the pad."""
        gear = self.config.landingGear
        props = massProperties(state, self.config)
        if self._outOfReach(state, props.cgHeight):
            self.anchors = None
            return
        _, posWorld, _ = self._feet(state, props.cgHeight, props.cgLateral)
        n = self.normal
        penetration = -(posWorld @ n)
        anchors = np.full_like(posWorld, np.nan) if self.anchors is None else self.anchors.copy()
        for i in range(len(posWorld)):
            if penetration[i] <= 0.0:
                anchors[i] = np.nan
                continue
            if np.isnan(anchors[i, 0]):
                anchors[i] = posWorld[i]
                continue
            stretch = posWorld[i] - anchors[i]
            stretch = stretch - (stretch @ n) * n
            spring = np.linalg.norm(stretch)
            normal = gear.strutStiffness * penetration[i] * (1 + gear.strutHardening * (penetration[i] / gear.strutStroke) ** 2)
            reach = gear.frictionCoefficient * normal / gear.tangentialStiffness
            if spring > reach:
                anchors[i] = posWorld[i] - stretch * (reach / spring)     # sliding: the anchor follows at the friction limit
        self.anchors = None if np.all(np.isnan(anchors)) else anchors

    def footForces(self, state):
        """Each foot as (rBody from CG, world contact force); zero force for feet above the pad."""
        props = massProperties(state, self.config)
        if self._outOfReach(state, props.cgHeight):
            return [(np.array([x - props.cgLateral[0], y - props.cgLateral[1], -props.cgHeight]), np.zeros(3)) for x, y in self.footOffsetsXY]
        rBody, posWorld, velWorld = self._feet(state, props.cgHeight, props.cgLateral)
        forces, _ = self._contactForces(posWorld, velWorld)
        return [(rBody[i], forces[i]) for i in range(len(rBody))]

    def __call__(self, state, commands, t):
        props = massProperties(state, self.config)
        if self._outOfReach(state, props.cgHeight):
            return np.zeros(3), np.zeros(3), 0.0
        rotation = attitude.quatToRotationMatrix(state.quaternion)
        rBody, posWorld, velWorld = self._feet(state, props.cgHeight, props.cgLateral)
        forcesWorld, penetration = self._contactForces(posWorld, velWorld)
        if not np.any(penetration > 0.0):
            return np.zeros(3), np.zeros(3), 0.0
        force = forcesWorld.sum(axis=0)
        forcesBody = forcesWorld @ rotation
        torque = np.cross(rBody, forcesBody).sum(axis=0)
        return force, torque, 0.0

    def maxPenetration(self, state):
        """Deepest foot below the pad along its normal, m; zero if all feet are above."""
        props = massProperties(state, self.config)
        if self._outOfReach(state, props.cgHeight):
            return 0.0
        _, posWorld, _ = self._feet(state, props.cgHeight, props.cgLateral)
        return float(max(0.0, -(posWorld @ self.normal).min()))
