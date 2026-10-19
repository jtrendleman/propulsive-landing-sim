""" ground.py
The landing pad as a tilted plane through the origin: surface height, unit normal, and signed distance of a point.
Author: Jack Rendleman  Date: 10-18-2026    Version: 1.0.0
"""

import numpy as np


def padNormal(environmentConfig):
    s, d = environmentConfig.padSlope, environmentConfig.padSlopeDirection
    n = np.array([np.tan(s) * np.cos(d), np.tan(s) * np.sin(d), 1.0])
    return n / np.linalg.norm(n)


def padHeight(environmentConfig, x, y):
    s, d = environmentConfig.padSlope, environmentConfig.padSlopeDirection
    return -np.tan(s) * (x * np.cos(d) + y * np.sin(d))


def heightAbovePad(environmentConfig, points):
    """Signed distance along the pad normal for an (n, 3) array or a single point; negative is below the surface."""
    return np.asarray(points) @ padNormal(environmentConfig)
