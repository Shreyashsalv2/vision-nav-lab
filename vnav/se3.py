"""Rigid-body transforms in NumPy. Poses are 4x4 matrices T_a_b mapping points from frame b to frame a.

Quaternions use the (x, y, z, w) order of the TUM trajectory format and ROS.
"""
from __future__ import annotations

import numpy as np

# Camera optical frame (x right, y down, z forward) -> robot body frame (x forward, y left, z up), ROS REP-103.
R_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_rot(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w) with w >= 0 (Shepperd's method, stable for all angles)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def pose_to_vec(T: np.ndarray) -> np.ndarray:
    """4x4 -> [tx, ty, tz, qx, qy, qz, qw]."""
    return np.concatenate([T[:3, 3], rot_to_quat(T[:3, :3])])


def vec_to_pose(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return make_T(quat_to_rot(v[3:7]), v[:3])


def rotation_angle(R: np.ndarray) -> float:
    """Angle of a rotation matrix in radians."""
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def optical_to_body_np(T_w_c: np.ndarray) -> np.ndarray:
    """Re-express a camera pose given in an optical-convention world into ROS body convention (pure NumPy)."""
    C = make_T(R_BODY_FROM_OPTICAL, np.zeros(3))
    return C @ T_w_c @ inv_T(C)
