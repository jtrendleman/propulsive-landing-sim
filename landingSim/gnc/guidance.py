""" guidance.py
ZEM/ZEV guidance: the thrust acceleration that arrives at the target with the target velocity in time tgo.
Author: Jack Rendleman  Date: 10-19-2026    Version: 1.1.0

tgo comes from a constant-deceleration reference profile: at altitude h the profile speed is
sqrt(vf^2 + 2*aRef*h). Using the faster of actual and profile speed keeps tgo from stretching
when drag has already slowed the vehicle, and from collapsing when it is still fast.

The lateral channel uses tgo - lateralLead, so the sideways error and speed are nulled before the
vertical channel reaches the pad. With one shared tgo the vehicle arrives at the terminal phase
still swinging toward the pad, and the tilt limit near the ground cannot absorb that in time.

The lateral channel's time-to-go never drops below lateralTgoMin. ZEM/ZEV's lateral loop has a bandwidth of
about sqrt(6) / tgo, and a large vehicle can only tilt so fast: below that floor the lateral command would
outrun the attitude loop that has to deliver it.
"""

import numpy as np

from landingSim.environment.gravity import gravityAcceleration


class ZemZevGuidance:
    def __init__(self, environmentConfig, targetPosition=(0.0, 0.0, 0.0), targetVelocity=(0.0, 0.0, -1.0),
                 referenceDecel=2.0, tgoMin=1.0, tgoMax=60.0, lateralLead=0.0, lateralTgoMin=None):
        self.environment = environmentConfig
        self.targetPosition = np.array(targetPosition, float)
        self.targetVelocity = np.array(targetVelocity, float)
        self.referenceDecel = referenceDecel
        self.tgoMin = tgoMin
        self.tgoMax = tgoMax
        self.lateralLead = lateralLead
        self.lateralTgoMin = tgoMin if lateralTgoMin is None else lateralTgoMin

    def timeToGo(self, state, targetPosition):
        altitude = max(state.position[2] - targetPosition[2], 0.0)
        vf = abs(self.targetVelocity[2])
        u = max(-state.velocity[2], 0.0)
        uRef = np.sqrt(vf**2 + 2 * self.referenceDecel * altitude)
        return float(np.clip(2 * altitude / (max(u, uRef) + vf), self.tgoMin, self.tgoMax))

    def __call__(self, state, t, targetPosition=None):
        rTarget = self.targetPosition if targetPosition is None else np.array(targetPosition, float)
        tgo = self.timeToGo(state, rTarget)
        g = gravityAcceleration(self.environment, state.position)
        aCmd = self.correction(state, rTarget, g, tgo)
        if self.lateralLead > 0.0 or self.lateralTgoMin > self.tgoMin:
            tgoLateral = max(tgo - self.lateralLead, self.lateralTgoMin)
            aCmd[0:2] = self.correction(state, rTarget, g, tgoLateral)[0:2]
        return aCmd

    def correction(self, state, rTarget, g, tgo):
        zem = rTarget - (state.position + state.velocity * tgo + 0.5 * g * tgo**2)
        zev = self.targetVelocity - (state.velocity + g * tgo)
        return 6 * zem / tgo**2 - 2 * zev / tgo
