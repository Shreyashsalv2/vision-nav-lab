"""Checkerboard camera calibration (Zhang's method via OpenCV), plus synthetic checkerboard views for testing."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import se3
from .camera import PinholeCamera


@dataclass
class CalibrationResult:
    camera: PinholeCamera
    rms_reprojection_px: float
    views_used: int


def board_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    """Inner-corner coordinates of a checkerboard in its own plane (z = 0)."""
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


def calibrate(images: list[np.ndarray], pattern: tuple[int, int] = (9, 6), square_m: float = 0.025) -> CalibrationResult:
    """Detect the board in each grayscale image and solve for intrinsics and distortion."""
    obj, img_pts = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    for img in images:
        found, corners = cv2.findChessboardCorners(img, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), crit)
        obj.append(board_points(pattern, square_m))
        img_pts.append(corners)
    if len(obj) < 3:
        raise ValueError(f"board found in only {len(obj)} images; need at least 3")
    h, w = images[0].shape[:2]
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img_pts, (w, h), None, None)
    D = D.ravel().tolist() + [0.0] * (5 - D.size)
    cam = PinholeCamera(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), D[:5])
    return CalibrationResult(cam, float(rms), len(obj))


def render_board_views(camera: PinholeCamera, n: int = 12, pattern: tuple[int, int] = (9, 6),
                       square_m: float = 0.025, seed: int = 0) -> list[np.ndarray]:
    """Render a checkerboard seen by `camera` from n random poses (for tests and demos)."""
    rng = np.random.default_rng(seed)
    cols, rows = pattern
    sq = 24  # pixels per square in the board texture
    tex = np.zeros(((rows + 1) * sq + 2 * sq, (cols + 1) * sq + 2 * sq), np.uint8)
    tex[:] = 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                tex[sq + r * sq: sq + (r + 1) * sq, sq + c * sq: sq + (c + 1) * sq] = 0
    # texture pixel (u, v) <-> board metres: inner corner (0,0) sits at pixel (2*sq, 2*sq)
    m_per_px = square_m / sq
    views = []
    yy, xx = np.mgrid[0:camera.height, 0:camera.width].astype(np.float32)
    for _ in range(n):
        R = se3.rot_x(rng.uniform(-0.5, 0.5)) @ se3.rot_y(rng.uniform(-0.5, 0.5)) @ se3.rot_z(rng.uniform(-0.3, 0.3))
        centre = np.array([(cols - 1) * square_m / 2, (rows - 1) * square_m / 2, 0.0])
        t = np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03), rng.uniform(0.35, 0.55)]) - R @ centre
        # inverse map: for each image pixel, intersect its ray with the board plane
        rays = camera.normalize(np.stack([xx.ravel(), yy.ravel()], 1))
        d = np.column_stack([rays, np.ones(len(rays))]) @ R  # ray directions in board frame
        o = -R.T @ t                                         # camera centre in board frame
        lam = -o[2] / d[:, 2]
        pb = o + lam[:, None] * d
        u = (pb[:, 0] / m_per_px + 2 * sq).reshape(camera.height, camera.width).astype(np.float32)
        v = (pb[:, 1] / m_per_px + 2 * sq).reshape(camera.height, camera.width).astype(np.float32)
        img = cv2.remap(tex, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=200)
        views.append(img)
    return views
