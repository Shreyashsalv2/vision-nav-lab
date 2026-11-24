"""Pinhole camera with radial-tangential (Brown-Conrady) distortion, as used by OpenCV."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=np.float64)

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Project 3-D points in the camera's optical frame (z forward) to distorted pixels."""
        pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return np.empty((0, 2))
        px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.D)
        return px.reshape(-1, 2)

    def normalize(self, px: np.ndarray) -> np.ndarray:
        """Undistort pixels to normalized image coordinates (x/z, y/z)."""
        pts = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        return cv2.undistortPoints(pts, self.K, self.D).reshape(-1, 2)

    def in_image(self, px: np.ndarray, margin: float = 0.0) -> np.ndarray:
        px = np.asarray(px).reshape(-1, 2)
        return (px[:, 0] >= margin) & (px[:, 0] < self.width - margin) & (px[:, 1] >= margin) & (px[:, 1] < self.height - margin)

    # ---- serialisation ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["dist"] = [float(x) for x in self.dist]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PinholeCamera":
        return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
                   [float(x) for x in d.get("dist", [0, 0, 0, 0, 0])])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PinholeCamera":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_projection_matrix(cls, P: np.ndarray, width: int, height: int) -> "PinholeCamera":
        """Build an undistorted camera from a 3x4 projection matrix, e.g. KITTI's calib.txt P0 (images are rectified)."""
        P = np.asarray(P, dtype=np.float64).reshape(3, 4)
        return cls(width, height, P[0, 0], P[1, 1], P[0, 2], P[1, 2])
