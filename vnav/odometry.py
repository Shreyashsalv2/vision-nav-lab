"""Monocular visual odometry: track features, estimate frame-to-frame motion, chain the poses.

Monocular vision recovers the direction of motion but not its length. Each step is scaled by an
external speed source: ground truth here (standard practice when evaluating monocular VO), and on a
robot an altimeter, wheel odometry or an IMU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import motion, se3
from .camera import PinholeCamera
from .features import FeatureTracker, TrackerConfig


@dataclass
class StepInfo:
    tracked: int = 0
    inliers: int = 0
    used_fallback: bool = False


@dataclass
class VOResult:
    poses: np.ndarray                      # (N,4,4) T_world_camera, optical convention, first pose = identity
    steps: list = field(default_factory=list)

    @property
    def fallback_rate(self) -> float:
        return float(np.mean([s.used_fallback for s in self.steps])) if self.steps else 0.0


class VisualOdometry:
    def __init__(self, camera: PinholeCamera, tracker: FeatureTracker | None = None,
                 min_inliers: int = 25, min_flow_px: float = 0.3):
        self.camera = camera
        self.tracker = tracker or FeatureTracker(TrackerConfig())
        self.min_inliers = min_inliers
        self.min_flow_px = min_flow_px
        self.reset()

    def reset(self) -> None:
        self.pose = np.eye(4)
        self._prev = None
        self._pts = np.empty((0, 2), np.float32)
        self._last_rel = np.eye(4)
        self.last_p0 = self.last_p1 = np.empty((0, 2))

    def step(self, img: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, StepInfo]:
        """Process the next grayscale frame. Returns the new camera pose and diagnostics."""
        info = StepInfo()
        if self._prev is None:
            self._prev = img
            self._pts = self.tracker.detect(img)
            return self.pose.copy(), info
        p0, p1 = self.tracker.track(self._prev, img, self._pts)
        self.last_p0, self.last_p1 = p0, p1
        info.tracked = len(p1)
        rel = motion.estimate(self.camera, p0, p1)
        if rel is not None and rel.inliers >= self.min_inliers and rel.median_flow_px >= self.min_flow_px:
            info.inliers = rel.inliers
            T_curr_prev = se3.make_T(rel.R, rel.t * scale)
            T_prev_curr = se3.inv_T(T_curr_prev)
        else:  # not enough evidence: assume the previous motion continued (constant velocity)
            info.used_fallback = True
            T_prev_curr = self._last_rel.copy()
            if np.linalg.norm(T_prev_curr[:3, 3]) > 1e-12:
                T_prev_curr[:3, 3] *= scale / np.linalg.norm(T_prev_curr[:3, 3])
        self.pose = self.pose @ T_prev_curr
        self._last_rel = T_prev_curr
        self._pts = p1
        if len(self._pts) < self.tracker.cfg.min_tracked:
            new = self.tracker.detect(img, existing=self._pts)
            self._pts = np.vstack([self._pts, new]) if len(self._pts) else new
        self._prev = img
        return self.pose.copy(), info

    def run(self, images, scales) -> VOResult:
        """Run over a sequence. scales[k] is the distance travelled from frame k-1 to k (scales[0] unused)."""
        self.reset()
        poses, steps = [], []
        for img, s in zip(images, scales):
            T, info = self.step(img, float(s))
            poses.append(T)
            steps.append(info)
        return VOResult(poses=np.array(poses), steps=steps[1:])


def step_lengths(gt_poses: np.ndarray) -> np.ndarray:
    """Distance travelled between consecutive ground-truth poses (first entry 0)."""
    p = gt_poses[:, :3, 3]
    return np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0), axis=1)])
