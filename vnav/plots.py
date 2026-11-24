"""Result plots (matplotlib, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GREEN, RED, GREY = "#1B4332", "#C0392B", "#6C757D"


def trajectory_topdown(gt: np.ndarray, est: np.ndarray, path: str | Path, title: str) -> None:
    """Top-down view in the optical world: forward (z) along the page, sideways (x) up the page."""
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=130)
    ax.plot(gt[:, 2, 3], gt[:, 0, 3], color=GREY, lw=2.5, label="Ground truth")
    ax.plot(est[:, 2, 3], est[:, 0, 3], color=GREEN, lw=1.6, ls="--", label="Visual odometry")
    ax.scatter([gt[0, 2, 3]], [gt[0, 0, 3]], color=GREEN, zorder=5, s=30)
    ax.annotate("start", (gt[0, 2, 3], gt[0, 0, 3]), textcoords="offset points", xytext=(4, -12), fontsize=9, color=GREY)
    ax.set_xlabel("forward, z (m)")
    ax.set_ylabel("right, x (m)\n(scale exaggerated)")
    ax.set_title(title, fontsize=11, loc="left")
    lo = min(gt[:, 0, 3].min(), est[:, 0, 3].min())
    hi = max(gt[:, 0, 3].max(), est[:, 0, 3].max())
    pad = 0.35 * max(hi - lo, 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_over_time(errors_m: np.ndarray, stamps: np.ndarray, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=130)
    ax.plot(stamps - stamps[0], errors_m, color=GREEN, lw=1.5)
    ax.fill_between(stamps - stamps[0], 0, errors_m, color=GREEN, alpha=0.12)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tracks_overlay(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, path: str | Path) -> None:
    """A frame with feature tracks drawn on it."""
    import cv2

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for a, b in zip(p0.astype(int), p1.astype(int)):
        cv2.line(vis, tuple(int(x) for x in a), tuple(int(x) for x in b), (79, 106, 45), 1, cv2.LINE_AA)
        cv2.circle(vis, tuple(int(x) for x in b), 2, (43, 57, 192), -1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)
