""" testAttitude.py
Tests for core attitude math against hand cases and scipy's Rotation oracle.
Conventions: +z up world, +z body thrust axis, scalar-first [w,x,y,z],
q = body-to-world, Hamilton product, 3-2-1 Euler (roll, pitch, yaw).
Author: Jack Rendleman Date: 10/15/2026 Version: 0-1-0A
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from landingSim.core import attitude, constants

rng = np.random.default_rng(seed=42)
bodyUp = np.array([0.0, 0.0, 1.0])  # thrust axis in body frame
worldUp = np.array([0.0, 0.0, 1.0])


def randomUnitQuat():
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def randomVector():
    return rng.normal(size=3)


def scipyRotation(q):
    """Build the scipy oracle from our scalar-first quaternion."""
    return Rotation.from_quat(q, scalar_first=True)


def testIdentityIsUpright():
    assert np.allclose(attitude.rotateBodyToWorld(attitude.identity, bodyUp), worldUp)


def testNormalizeGivesUnitNorm():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    assert np.allclose(np.linalg.norm(attitude.normalize(q)), 1.0)
    assert np.allclose(attitude.normalize(q), attitude.identity)


def testNinetyDegreesAboutXHandCase():
    q = attitude.fromAxisAngle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
    # Rotating body +z by +90 deg about world x lays it along world -y.
    assert np.allclose(attitude.rotateBodyToWorld(q, bodyUp), [0.0, -1.0, 0.0], atol=1e-12)


def testWorldToBodyInvertsBodyToWorld():
    for _ in range(10):
        q, v = randomUnitQuat(), randomVector()
        vWorld = attitude.rotateBodyToWorld(q, v)
        assert np.allclose(attitude.rotateWorldToBody(q, vWorld), v)


def testRotateMatchesScipy():
    for _ in range(10):
        q, v = randomUnitQuat(), randomVector()
        assert np.allclose(attitude.rotateBodyToWorld(q, v), scipyRotation(q).apply(v))


def testRotationMatrixMatchesScipy():
    for _ in range(10):
        q = randomUnitQuat()
        assert np.allclose(attitude.quatToRotationMatrix(q), scipyRotation(q).as_matrix())


def testRotationMatrixIsProper():
    for _ in range(10):
        R = attitude.quatToRotationMatrix(randomUnitQuat())
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)


def testMultiplyMatchesScipy():
    for _ in range(10):
        q1, q2 = randomUnitQuat(), randomUnitQuat()
        ours = attitude.multiply(q1, q2)
        ref = (scipyRotation(q1) * scipyRotation(q2)).as_quat(scalar_first=True)
        # q and -q are the same rotation; compare up to sign.
        assert np.allclose(ours, ref) or np.allclose(ours, -ref)


def testInverseUndoesRotation():
    for _ in range(10):
        q = randomUnitQuat()
        product = attitude.multiply(q, attitude.inverse(q))
        assert np.allclose(product, attitude.identity, atol=1e-12)


def testEulerRoundTrip():
    for _ in range(10):
        roll, yaw = rng.uniform(-np.pi, np.pi, size=2)
        pitch = rng.uniform(-1.4, 1.4)  # stay clear of gimbal lock at +-pi/2
        q = attitude.eulerToQuat(roll, pitch, yaw)
        assert np.allclose(attitude.quatToEuler(q), [roll, pitch, yaw])


def testEulerMatchesScipy():
    for _ in range(10):
        q = randomUnitQuat()
        yawPitchRoll = scipyRotation(q).as_euler("ZYX")  # intrinsic z, then y, then x
        assert np.allclose(attitude.quatToEuler(q), yawPitchRoll[::-1])


def testQuatDerivativeHandCase():
    omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw about body z
    assert np.allclose(attitude.quatDerivative(attitude.identity, omega), [0.0, 0.0, 0.0, 0.5])


def testQuatDerivativeMatchesSmallRotation():
    dt = 1e-4
    for _ in range(10):
        q, omega = randomUnitQuat(), randomVector()
        stepped = attitude.normalize(q + attitude.quatDerivative(q, omega) * dt)
        angle = np.linalg.norm(omega) * dt
        exact = attitude.multiply(q, attitude.fromAxisAngle(omega, angle))
        assert np.allclose(stepped, exact, atol=1e-7)
        # Derivative must be perpendicular to q so the norm stays 1.
        assert np.isclose(np.dot(q, attitude.quatDerivative(q, omega)), 0.0, atol=1e-12)


def testStandardGravityConstant():
    assert constants.g0 == pytest.approx(9.80665)
