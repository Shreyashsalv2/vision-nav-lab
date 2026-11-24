"""Trajectory accuracy: absolute trajectory error (ATE), relative pose error (RPE) and drift."""
from __future__ import annotations

import numpy as np

from . import se3


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares similarity transform with dst ≈ s * R @ src + t (Umeyama, 1991)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(est: np.ndarray, gt: np.ndarray, with_scale: bool = False) -> dict:
    """Absolute trajectory error of positions after best-fit alignment (SE(3), or Sim(3) with scale)."""
    pe, pg = est[:, :3, 3], gt[:, :3, 3]
    s, R, t = umeyama(pe, pg, with_scale)
    aligned = (s * (R @ pe.T)).T + t
    err = np.linalg.norm(aligned - pg, axis=1)
    return {"rmse_m": float(np.sqrt(np.mean(err ** 2))), "mean_m": float(err.mean()), "median_m": float(np.median(err)),
            "max_m": float(err.max()), "scale": s, "errors_m": err, "aligned": aligned}


def rpe(est: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Relative pose error over a fixed frame gap: local accuracy, independent of accumulated drift."""
    t_err, r_err = [], []
    for i in range(len(est) - delta):
        d_est = se3.inv_T(est[i]) @ est[i + delta]
        d_gt = se3.inv_T(gt[i]) @ gt[i + delta]
        E = se3.inv_T(d_gt) @ d_est
        t_err.append(np.linalg.norm(E[:3, 3]))
        r_err.append(np.degrees(se3.rotation_angle(E[:3, :3])))
    t_err, r_err = np.array(t_err), np.array(r_err)
    return {"delta": delta, "trans_rmse_m": float(np.sqrt(np.mean(t_err ** 2))), "rot_rmse_deg": float(np.sqrt(np.mean(r_err ** 2)))}


def path_length(poses: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))


def final_drift_pct(est: np.ndarray, gt: np.ndarray) -> float:
    """Final position error as % of distance travelled, with both trajectories starting at the same pose."""
    return 100.0 * float(np.linalg.norm(est[-1, :3, 3] - gt[-1, :3, 3])) / max(path_length(gt), 1e-9)


def summary(est: np.ndarray, gt: np.ndarray) -> dict:
    a = ate(est, gt)
    return {
        "frames": int(len(est)),
        "path_length_m": path_length(gt),
        "ate_rmse_m": a["rmse_m"],
        "ate_max_m": a["max_m"],
        "rpe1_trans_rmse_m": rpe(est, gt, 1)["trans_rmse_m"],
        "rpe1_rot_rmse_deg": rpe(est, gt, 1)["rot_rmse_deg"],
        "rpe10_trans_rmse_m": rpe(est, gt, 10)["trans_rmse_m"],
        "rpe10_rot_rmse_deg": rpe(est, gt, 10)["rot_rmse_deg"],
        "final_drift_pct": final_drift_pct(est, gt),
    }
