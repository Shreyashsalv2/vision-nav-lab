"""Relative camera motion from matched points: essential matrix (5-point + RANSAC) and its decomposition."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import PinholeCamera


@dataclass
class RelativePose:
    R: np.ndarray        # rotation taking points from the previous camera frame to the current one
    t: np.ndarray        # unit translation (direction only: monocular scale is unknown)
    inliers: int
    median_flow_px: float


def estimate(camera: PinholeCamera, p0: np.ndarray, p1: np.ndarray, ransac_px: float = 1.0) -> RelativePose | None:
    """Estimate motion between two frames from tracked pixel coordinates (distorted)."""
    if len(p0) < 8:
        return None
    flow = float(np.median(np.linalg.norm(p1 - p0, axis=1)))
    n0, n1 = camera.normalize(p0), camera.normalize(p1)
    I = np.eye(3)
    E, mask = cv2.findEssentialMat(n0, n1, I, method=cv2.RANSAC, prob=0.999, threshold=ransac_px / camera.fx)
    if E is None:
        return None
    if E.shape != (3, 3):  # several solutions stacked: keep the first
        E = E[:3, :3]
    inliers, R, t, _ = cv2.recoverPose(E, n0, n1, I, mask=mask)
    return RelativePose(R=R, t=t.ravel() / max(np.linalg.norm(t), 1e-12), inliers=int(inliers), median_flow_px=flow)
