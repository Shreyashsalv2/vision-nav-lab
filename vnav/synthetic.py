"""Synthetic recorded dataset: a camera flying through a field of 3-D landmarks.

Every frame is rendered from a known camera pose, so the ground-truth trajectory is exact.
Output layout (TUM-like):  images/000000.png ...,  rgb.txt,  groundtruth.txt (TUM format),  camera.json
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import se3
from .camera import PinholeCamera

DEFAULT_CAMERA = PinholeCamera(640, 480, 500.0, 500.0, 320.0, 240.0, [-0.12, 0.03, 0.0005, -0.0004, 0.0])


@dataclass
class SyntheticConfig:
    frames: int = 300
    step_m: float = 0.30        # distance travelled per frame
    fps: float = 10.0
    n_landmarks: int = 9000
    seed: int = 7
    noise_sigma: float = 2.0    # image noise, grey levels


def make_trajectory(cfg: SyntheticConfig) -> np.ndarray:
    """Smooth path in the optical-convention world (x right, y down, z forward); first pose is identity."""
    poses = []
    pos = np.zeros(3)
    for k in range(cfg.frames):
        yaw = 0.45 * np.sin(2 * np.pi * k / 160.0)          # gentle left/right turns
        pitch = 0.04 * np.sin(2 * np.pi * k / 60.0)          # small nodding
        if k > 0:
            pos = pos + cfg.step_m * np.array([np.sin(yaw), 0.0, np.cos(yaw)])
            pos[1] = 0.25 * np.sin(2 * np.pi * k / 90.0)     # vertical bob (y is down)
        poses.append(se3.make_T(se3.rot_y(yaw) @ se3.rot_x(pitch), pos.copy()))
    T0_inv = se3.inv_T(poses[0])
    return np.array([T0_inv @ T for T in poses])


def make_landmarks(traj: np.ndarray, cfg: SyntheticConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    path = traj[:, :3, 3]
    lo, hi = path.min(0) - [18, 5, 5], path.max(0) + [18, 5, 30]
    pts = rng.uniform(lo, hi, size=(cfg.n_landmarks * 2, 3))
    pts[:, 1] = rng.uniform(-5.0, 5.0, size=len(pts))
    # keep landmarks at least 2 m from the flight path so nothing passes through the lens
    d = np.min(np.linalg.norm(pts[:, None, [0, 2]] - path[None, ::5][:, :, [0, 2]], axis=2), axis=1)
    pts = pts[d > 2.0][: cfg.n_landmarks]
    shades = rng.integers(20, 236, size=len(pts))
    return pts, shades


def render(camera: PinholeCamera, T_w_c: np.ndarray, landmarks: np.ndarray, shades: np.ndarray,
           rng: np.random.Generator, noise_sigma: float) -> np.ndarray:
    T_c_w = se3.inv_T(T_w_c)
    pc = landmarks @ T_c_w[:3, :3].T + T_c_w[:3, 3]
    keep = (pc[:, 2] > 0.8) & (pc[:, 2] < 45.0)
    pc, sh = pc[keep], shades[keep]
    # coarse frustum cull before the (slower) distorted projection
    fov = (np.abs(pc[:, 0] / pc[:, 2]) < 0.9) & (np.abs(pc[:, 1] / pc[:, 2]) < 0.7)
    pc, sh = pc[fov], sh[fov]
    px = camera.project(pc)
    vis = camera.in_image(px, margin=-8)
    pc, sh, px = pc[vis], sh[vis], px[vis]
    img = np.full((camera.height, camera.width), 128, np.uint8)
    yy = np.linspace(-1, 1, camera.height)[:, None]
    img = np.clip(img + (18 * yy).astype(np.int16), 0, 255).astype(np.uint8)
    order = np.argsort(-pc[:, 2])  # painter's algorithm: far first
    for i in order:
        half = int(np.clip(round(camera.fx * 0.05 / pc[i, 2]), 1, 7))
        u, v = int(round(px[i, 0])), int(round(px[i, 1]))
        cv2.rectangle(img, (u - half, v - half), (u + half, v + half), int(sh[i]), thickness=-1)
    noise = rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def generate(out_dir: str | Path, cfg: SyntheticConfig | None = None, camera: PinholeCamera = DEFAULT_CAMERA) -> Path:
    from .io import write_tum

    cfg = cfg or SyntheticConfig()
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    traj = make_trajectory(cfg)
    landmarks, shades = make_landmarks(traj, cfg, rng)
    stamps = np.arange(cfg.frames) / cfg.fps
    lines = ["# timestamp filename"]
    for k, T in enumerate(traj):
        img = render(camera, T, landmarks, shades, rng, cfg.noise_sigma)
        name = f"images/{k:06d}.png"
        cv2.imwrite(str(out / name), img)
        lines.append(f"{stamps[k]:.6f} {name}")
    (out / "rgb.txt").write_text("\n".join(lines) + "\n")
    write_tum(out / "groundtruth.txt", stamps, traj)
    camera.save(out / "camera.json")
    return out
