"""Readers and writers for recorded datasets: TUM trajectory format, TUM RGB-D sequences, KITTI odometry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import se3
from .camera import PinholeCamera


def write_tum(path: str | Path, stamps, poses: np.ndarray) -> None:
    """TUM format: one line per pose, 'timestamp tx ty tz qx qy qz qw'."""
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for ts, T in zip(stamps, poses):
        v = se3.pose_to_vec(T)
        lines.append(f"{float(ts):.6f} " + " ".join(f"{x:.9f}" for x in v))
    Path(path).write_text("\n".join(lines) + "\n")


def read_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stamps, poses = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        stamps.append(vals[0])
        poses.append(se3.vec_to_pose(vals[1:8]))
    return np.array(stamps), np.array(poses)


def read_image_list(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """TUM rgb.txt: 'timestamp filename' per line."""
    stamps, files = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ts, name = line.split()[:2]
        stamps.append(float(ts))
        files.append(name)
    return np.array(stamps), files


def associate(stamps_a: np.ndarray, stamps_b: np.ndarray, max_dt: float = 0.02) -> list[tuple[int, int]]:
    """Match each timestamp in a to the nearest in b within max_dt (as in the TUM benchmark tools)."""
    pairs = []
    for i, t in enumerate(stamps_a):
        j = int(np.argmin(np.abs(stamps_b - t)))
        if abs(stamps_b[j] - t) <= max_dt:
            pairs.append((i, j))
    return pairs


@dataclass
class Sequence:
    camera: PinholeCamera
    image_paths: list
    gt_poses: np.ndarray       # T_world_camera, re-anchored so the first pose is identity
    stamps: np.ndarray

    def images(self):
        for p in self.image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(p)
            yield img


def _anchor(poses: np.ndarray) -> np.ndarray:
    T0 = se3.inv_T(poses[0])
    return np.array([T0 @ T for T in poses])


def load_tum_style(root: str | Path, camera_file: str | Path | None = None) -> Sequence:
    """Synthetic sequences and TUM RGB-D sequences (rgb.txt + groundtruth.txt).
    TUM sequences need a camera JSON with that sequence's published calibration (see README)."""
    root = Path(root)
    camera = PinholeCamera.load(camera_file or root / "camera.json")
    img_ts, files = read_image_list(root / "rgb.txt")
    gt_ts, gt = read_tum(root / "groundtruth.txt")
    pairs = associate(img_ts, gt_ts)
    return Sequence(camera, [root / files[i] for i, _ in pairs], _anchor(gt[[j for _, j in pairs]]), img_ts[[i for i, _ in pairs]])


def load_kitti(root: str | Path, seq: str = "00") -> Sequence:
    """KITTI odometry layout: sequences/XX/{image_0/, calib.txt, times.txt}, poses/XX.txt (3x4 per line)."""
    root = Path(root)
    sdir = root / "sequences" / seq
    P0 = next(np.array([float(x) for x in l.split()[1:]]) for l in (sdir / "calib.txt").read_text().splitlines() if l.startswith("P0:"))
    files = sorted((sdir / "image_0").glob("*.png"))
    h, w = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE).shape
    camera = PinholeCamera.from_projection_matrix(P0, w, h)
    rows = np.loadtxt(root / "poses" / f"{seq}.txt").reshape(-1, 3, 4)
    poses = np.array([np.vstack([r, [0, 0, 0, 1]]) for r in rows])
    stamps = np.loadtxt(sdir / "times.txt")
    return Sequence(camera, files, _anchor(poses), stamps)
