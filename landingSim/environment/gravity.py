""" gravity.py
Gravitational acceleration field in the world frame; dynamics multiplies by mass.
Author: Jack Rendleman  Date: 10-16-2026    Version: 1.0.0
"""

import numpy as np


def gravityAcceleration(environmentConfig, position):
    # position is unused for constant gravity; kept so an altitude model can drop in later.
    return np.array([0.0, 0.0, -environmentConfig.gravity])
