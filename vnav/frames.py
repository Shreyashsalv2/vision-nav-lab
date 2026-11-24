"""Coordinate-frame conversions, backed by the C++17 library in cpp/ when its Python module is built.

Build the module with scripts/build_cpp.sh; without it, the identical NumPy implementation in se3.py is used.
"""
from __future__ import annotations

import numpy as np

from . import se3

try:  # the pybind11 module compiled from cpp/src/bindings.cpp
    from . import _frames as _cpp  # type: ignore
except ImportError:  # pragma: no cover - depends on the build
    _cpp = None

BACKEND = "cpp" if _cpp is not None else "numpy"


def optical_to_body(poses: np.ndarray, backend: str | None = None) -> np.ndarray:
    """Convert camera poses (N,4,4) from optical convention (x right, y down, z forward)
    to ROS body convention (x forward, y left, z up), for both the world frame and the moving frame."""
    poses = np.asarray(poses, dtype=np.float64)
    single = poses.ndim == 2
    poses = poses[None] if single else poses
    use = backend or BACKEND
    if use == "cpp":
        if _cpp is None:
            raise RuntimeError("C++ frames module not built; run scripts/build_cpp.sh")
        vecs = [se3.pose_to_vec(T).tolist() for T in poses]
        out = np.array([se3.vec_to_pose(v) for v in _cpp.optical_to_body(vecs)])
    else:
        out = np.array([se3.optical_to_body_np(T) for T in poses])
    return out[0] if single else out
