""" observation.py
Builds the policy's observation vector from what the flight computer has: the navigation estimate, the guidance command, the wind forecast, and the onboard estimators.
Author: Jack Rendleman  Date: 10-18-2026    Version: 2.0.0

Everything is scaled to roughly [-1, 1] by hand so no running normalizer is needed. Nothing here reads
the true state: the controller's estimate is built from sensors (see sensors.py).
"""

import numpy as np

from landingSim.core import attitude
from landingSim.environment.wind import windVelocity

observationSize = 21


def observe(state, t, controller):
    """`state` is only a fallback for controllers without an estimate (perfect-navigation tests)."""
    if getattr(controller, "estimate", None) is not None:
        state = controller.estimate
    aCmd = controller.guidanceCommand(state, t)
    bodyZ = attitude.rotateBodyToWorld(state.quaternion, np.array([0.0, 0.0, 1.0]))
    forecast = windVelocity(getattr(controller, "forecastEnvironment", controller.config.environment), state.position, t)[0:2]
    return np.concatenate([
        state.position / np.array([100.0, 100.0, 1000.0]),
        state.velocity / np.array([30.0, 30.0, 100.0]),
        bodyZ,
        state.omega,
        [state.propellantMass / controller.config.spacecraft.propellant.initialMass],
        aCmd / 20.0,
        forecast / 15.0,
        getattr(controller, "windAccel", np.zeros(2)) / 3.0,
        [5.0 * (getattr(controller, "thrustScale", 1.0) - 1.0)],
    ]).astype(np.float32)
