"""Calibrate a camera from checkerboard photos and save camera.json.

  python scripts/calibrate.py --images "calib/*.png" --pattern 9x6 --square 0.025 --out camera.json
  python scripts/calibrate.py --demo     # renders synthetic board views from a known camera and checks the result
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnav import calibration, synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="glob of checkerboard images")
    ap.add_argument("--pattern", default="9x6", help="inner corners, cols x rows")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--out", default="camera.json")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    pattern = tuple(int(x) for x in a.pattern.lower().split("x"))
    if a.demo:
        truth = synthetic.DEFAULT_CAMERA
        imgs = calibration.render_board_views(truth, n=14, pattern=pattern, square_m=a.square)
    else:
        imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in sorted(glob.glob(a.images))]
    res = calibration.calibrate(imgs, pattern, a.square)
    res.camera.save(a.out)
    c = res.camera
    print(f"{res.views_used} views, RMS reprojection {res.rms_reprojection_px:.3f} px")
    print(f"fx={c.fx:.2f} fy={c.fy:.2f} cx={c.cx:.2f} cy={c.cy:.2f} dist={[round(d, 4) for d in c.dist]}")
    if a.demo:
        print(f"truth: fx={truth.fx} fy={truth.fy} cx={truth.cx} cy={truth.cy} dist={truth.dist}")


if __name__ == "__main__":
    main()
