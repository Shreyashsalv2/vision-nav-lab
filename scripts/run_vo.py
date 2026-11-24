"""Run visual odometry on a recorded sequence and score it against ground truth.

  python scripts/run_vo.py --dataset synthetic --path data/synthetic --out results/synthetic
  python scripts/run_vo.py --dataset tum --path rgbd_dataset_freiburg1_xyz --camera configs/my_tum_camera.json --out results/tum
  python scripts/run_vo.py --dataset kitti --path kitti_odometry --seq 00 --out results/kitti

--max-ate / --max-drift make the script exit non-zero above a threshold (used as a CI accuracy gate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnav import frames, io, metrics, plots  # noqa: E402
from vnav.odometry import VisualOdometry, VOResult, step_lengths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["synthetic", "tum", "kitti"], default="synthetic")
    ap.add_argument("--path", default="data/synthetic")
    ap.add_argument("--camera", help="camera JSON (required for TUM sequences)")
    ap.add_argument("--seq", default="00", help="KITTI sequence number")
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--max-ate", type=float, help="fail if ATE RMSE (m) exceeds this")
    ap.add_argument("--max-drift", type=float, help="fail if final drift (%% of distance) exceeds this")
    a = ap.parse_args()

    seq = io.load_kitti(a.path, a.seq) if a.dataset == "kitti" else io.load_tum_style(a.path, a.camera)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    vo = VisualOdometry(seq.camera)
    scales = step_lengths(seq.gt_poses)
    overlay_at = min(120, len(seq.image_paths) - 1)
    poses, steps = [], []
    for k, (img, s) in enumerate(zip(seq.images(), scales)):
        T, info = vo.step(img, float(s))
        poses.append(T)
        steps.append(info)
        if k == overlay_at:
            plots.tracks_overlay(img, vo.last_p0, vo.last_p1, out / "tracks.png")
    res = VOResult(poses=np.array(poses), steps=steps[1:])
    secs = time.perf_counter() - t0

    m = metrics.summary(res.poses, seq.gt_poses)
    m.update({
        "dataset": a.dataset, "fallback_rate": res.fallback_rate,
        "mean_tracked_features": float(np.mean([s.tracked for s in res.steps])),
        "mean_inliers": float(np.mean([s.inliers for s in res.steps])),
        "ms_per_frame": 1000 * secs / len(res.poses), "frames_backend": frames.BACKEND,
    })
    io.write_tum(out / "estimated_optical.txt", seq.stamps, res.poses)
    io.write_tum(out / "estimated_body.txt", seq.stamps, frames.optical_to_body(res.poses))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    a_res = metrics.ate(res.poses, seq.gt_poses)
    plots.trajectory_topdown(seq.gt_poses, res.poses, out / "trajectory.png",
                             f"{a.dataset}: {m['path_length_m']:.0f} m path, ATE RMSE {m['ate_rmse_m']:.2f} m")
    plots.error_over_time(a_res["errors_m"], seq.stamps, out / "error.png", "Position error after alignment")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))

    failed = []
    if a.max_ate is not None and m["ate_rmse_m"] > a.max_ate:
        failed.append(f"ATE RMSE {m['ate_rmse_m']:.3f} m > {a.max_ate} m")
    if a.max_drift is not None and m["final_drift_pct"] > a.max_drift:
        failed.append(f"drift {m['final_drift_pct']:.2f}% > {a.max_drift}%")
    if failed:
        print("ACCURACY GATE FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    print("accuracy gate passed" if (a.max_ate or a.max_drift) else "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
