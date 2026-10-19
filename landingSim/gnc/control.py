""" control.py
Thrust-vector controller: throttle from |aCmd|, gimbal angles from the attitude error, RCS for the rest.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.2.0

PD on the tilt error between body +z and the commanded thrust direction gives a desired
torque. The grid fins are deployed throughout, as on a Falcon 9-class booster, and lead the pitch and
yaw control while the airflow gives them more authority than the gimbal; their share fades to zero as
the dynamic pressure falls from 2 x finMinDynamicPressure to finMinDynamicPressure (the authority
crossover), so near the ground, where they would only saturate and push the top sideways, the gimbal
leads. The gimbal takes the rest through the engine's lever arm (step 5 sign convention). RCS takes
roll, which one gimballed engine cannot produce, plus whatever is left.
"""

import numpy as np

from landingSim.core import attitude
from landingSim.dynamics.state import Commands
from landingSim.vehicle.mass import massProperties


class ThrustVectorController:
    def __init__(self, spacecraftConfig, kp=20.0, kd=12.0):
        self.config = spacecraftConfig
        self.engine = spacecraftConfig.engine
        self.rcsMaxTorque = spacecraftConfig.rcs.thrustPerThruster * spacecraftConfig.rcs.momentArm
        self.kp = kp
        self.kd = kd

    def finShare(self, dynamicPressure):
        """Fraction of the pitch/yaw torque demand the fins lead: 1 above twice the crossover pressure, 0 below it."""
        q0 = self.config.aero.finMinDynamicPressure
        return float(np.clip((dynamicPressure - q0) / q0, 0.0, 1.0)) if q0 > 0.0 else 1.0

    def finAllocation(self, torqueDes, props, dynamicPressure):
        """Grid fins take as much of the pitch/yaw torque as the airflow allows; returns (deflections, torque delivered)."""
        aero = self.config.aero
        if aero.gridFinArea <= 0.0 or dynamicPressure < aero.finMinDynamicPressure:
            return np.zeros(2), np.zeros(3)
        arm = aero.finHeight - props.cgHeight
        gain = dynamicPressure * aero.gridFinArea * aero.finLiftSlope * arm        # N m per rad of deflection
        # tau_y = arm * Fx -> dx = tau_y / gain ; tau_x = -arm * Fy -> dy = -tau_x / gain
        deflection = np.clip(np.array([torqueDes[1], -torqueDes[0]]) / gain, -aero.maxFinDeflection, aero.maxFinDeflection)
        delivered = np.array([-gain * deflection[1], gain * deflection[0], 0.0])
        return deflection, delivered

    def __call__(self, state, aCmd, t, engineOn=True, dynamicPressure=0.0):
        """engineOn=False: coast. Throttle zero, gimbal centred; fins and RCS carry the attitude torque."""
        eng = self.engine
        props = massProperties(state, self.config)
        tMax = eng.maxThrust * eng.count
        aMag = np.linalg.norm(aCmd)
        throttle = float(np.clip(props.total * aMag / tMax, eng.minThrottle, 1.0)) if engineOn else 0.0

        zDes = aCmd / aMag if aMag > 1e-6 else np.array([0.0, 0.0, 1.0])
        zBody = attitude.rotateBodyToWorld(state.quaternion, np.array([0.0, 0.0, 1.0]))
        errWorld = attitude.cross3(zBody, zDes)                  # axis * sin(angle) taking zBody to zDes
        errBody = attitude.rotateWorldToBody(state.quaternion, errWorld)

        alphaDes = self.kp * errBody - self.kd * state.omega
        torqueDes = props.inertia * alphaDes             # principal axes, elementwise

        if not engineOn:
            fins, finTorque = self.finAllocation(torqueDes, props, dynamicPressure)
            torqueDes = torqueDes - finTorque
            return Commands(throttle=0.0, gimbal=np.zeros(2), rcs=np.clip(torqueDes / self.rcsMaxTorque, -1.0, 1.0), fins=fins)

        # The fins lead their share of the pitch/yaw demand while the airflow is strong; the gimbal takes the rest.
        fins, finTorque = self.finAllocation(self.finShare(dynamicPressure) * torqueDes, props, dynamicPressure)
        torqueDes = torqueDes - finTorque

        thrust = throttle * tMax
        h = props.cgHeight
        fx = -torqueDes[1] / h                           # tau_y = -h * Fx
        fy = torqueDes[0] / h                            # tau_x =  h * Fy
        gx = np.clip(np.arctan2(fx, thrust), -eng.maxGimbalAngle, eng.maxGimbalAngle)
        gy = np.clip(np.arctan2(fy, thrust), -eng.maxGimbalAngle, eng.maxGimbalAngle)

        # Torque the clamped gimbal actually delivers, and what is left for the RCS.
        gimbalTorque = np.array([h * thrust * np.tan(gy), -h * thrust * np.tan(gx), 0.0])
        remainder = torqueDes - gimbalTorque
        rcs = np.clip(remainder / self.rcsMaxTorque, -1.0, 1.0)
        return Commands(throttle=throttle, gimbal=np.array([gx, gy]), rcs=rcs, fins=fins)
