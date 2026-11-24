"""Feature detection (Shi-Tomasi corners) and frame-to-frame tracking (pyramidal Lucas-Kanade)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackerConfig:
    max_corners: int = 900
    min_tracked: int = 350       # re-detect when fewer features survive
    quality: float = 0.01
    min_distance: int = 9
    win_size: int = 21
    levels: int = 3
    fb_threshold_px: float = 1.0  # forward-backward consistency check


class FeatureTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self._lk = dict(winSize=(self.cfg.win_size, self.cfg.win_size), maxLevel=self.cfg.levels,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def detect(self, img: np.ndarray, existing: np.ndarray | None = None) -> np.ndarray:
        """Detect corners, avoiding the neighbourhood of points already being tracked."""
        mask = np.full(img.shape[:2], 255, np.uint8)
        n_existing = 0
        if existing is not None and len(existing):
            n_existing = len(existing)
            for x, y in existing.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), self.cfg.min_distance, 0, -1)
        want = self.cfg.max_corners - n_existing
        if want <= 0:
            return np.empty((0, 2), np.float32)
        pts = cv2.goodFeaturesToTrack(img, maxCorners=want, qualityLevel=self.cfg.quality,
                                      minDistance=self.cfg.min_distance, mask=mask)
        return np.empty((0, 2), np.float32) if pts is None else pts.reshape(-1, 2).astype(np.float32)

    def track(self, prev: np.ndarray, curr: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Track pts from prev to curr. Returns the matched (prev_pts, curr_pts) that pass all checks."""
        if len(pts) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **self._lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None, **self._lk)
        fb = np.linalg.norm(p0 - p0r, axis=2).ravel()
        h, w = curr.shape[:2]
        q = p1.reshape(-1, 2)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg.fb_threshold_px)
        ok &= (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        return p0.reshape(-1, 2)[ok], q[ok]
