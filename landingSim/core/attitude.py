""" attitude.py
Quaternion attitude math shared by every module; this docstring is the frame contract.
Author: Jack Rendleman  Date: 10-15-2026    Version: 1.0.0

Conventions (binding for the whole repo):
- World frame: +z up, landing pad at the origin.
- Body frame: +z along the rocket's length; the engine thrusts along +z body.
- Quaternion: numpy array shape (4,), scalar-first [w, x, y, z], unit norm.
- q is the body-to-world rotation. identity [1, 0, 0, 0] means upright.
- Hamilton product. multiply(q1, q2) means "apply q2 first, then q1".
- Euler angles: 3-2-1 yaw-pitch-roll, radians. R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
- Angular velocity omega is a (3,) array in the body frame, rad/s.
"""

import numpy as np

identity = np.array([1.0, 0.0, 0.0, 0.0])   # module constant

def normalize(q):
    return q / np.linalg.norm(q)

def conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def inverse(q):
    return conjugate(q) / np.dot(q, q)

def multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def fromAxisAngle(axis, angle):
   axis = axis / np.linalg.norm(axis)
   half = angle / 2
   return np.array([np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)])

def quatToRotationMatrix(q):
    # R @ vBody = vWorld. Standard formula from the unit quaternion:
    w, x, y, z = q
    row0 = [1 - 2*(y**2 + z**2),   2*(x*y - w*z),         2*(x*z + w*y)      ]
    row1 = [2*(x*y + w*z),         1 - 2*(x**2 + z**2),   2*(y*z - w*x)      ]
    row2 = [2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x**2 + y**2)]
    return np.array([row0, row1, row2])

def cross3(a, b):
    """Cross product of two 3-vectors. numpy.cross spends most of its time on axis bookkeeping for inputs this small."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def rotateBodyToWorld(q, v):
    """v' = v + 2 w (u x v) + 2 u x (u x v) with q = [w, u]; same result as quatToRotationMatrix(q) @ v, several times faster."""
    w = q[0]
    u = q[1:4]
    t = 2.0 * cross3(u, v)
    return v + w * t + cross3(u, t)

def rotateWorldToBody(q, v):
    """Inverse rotation: conjugate quaternion, same closed form."""
    w = q[0]
    u = -q[1:4]
    t = 2.0 * cross3(u, v)
    return v + w * t + cross3(u, t)

def quatDerivative(q, omega):
    # omega in BODY frame. For a body-to-world quaternion:
    #     qdot = 0.5 * q (x) [0, omega]
    omegaQuat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * multiply(q, omegaQuat)

def eulerToQuat(roll, pitch, yaw):
    qz = fromAxisAngle(np.array([0, 0, 1]), yaw)
    qy = fromAxisAngle(np.array([0, 1, 0]), pitch)
    qx = fromAxisAngle(np.array([1, 0, 0]), roll)
    return multiply(multiply(qz, qy), qx)

def quatToEuler(q):
    R = quatToRotationMatrix(q)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[1,0], R[0,0])
    return np.array([roll, pitch, yaw])

