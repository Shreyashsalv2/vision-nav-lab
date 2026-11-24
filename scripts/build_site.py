"""Build the static results page (site/) from results/synthetic.   python scripts/build_site.py"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "synthetic"
SITE = ROOT / "site"
REPO = "https://github.com/Shreyashsalv2/vision-nav-lab"


def main() -> None:
    m = json.loads((RES / "metrics.json").read_text())
    SITE.mkdir(exist_ok=True)
    for f in ("trajectory.png", "error.png", "tracks.png"):
        shutil.copy2(RES / f, SITE / f)
    cards = [
        (f"{m['ate_rmse_m']:.2f} m", "ATE RMSE", f"absolute trajectory error over a {m['path_length_m']:.0f} m path"),
        (f"{m['final_drift_pct']:.1f}%", "Final drift", "end-point error as a share of distance travelled"),
        (f"{m['rpe1_rot_rmse_deg']:.2f}°", "Rotation error per frame", "relative pose error, RMSE"),
        (f"{m['mean_tracked_features']:.0f}", "Features tracked per frame", f"{m['mean_inliers']:.0f} RANSAC inliers on average"),
    ]
    card_html = "".join(f"<div class='stat'><b>{a}</b><span>{b}</span><small>{c}</small></div>" for a, b, c in cards)
    steps = [
        ("Calibrate", "Checkerboard calibration (Zhang's method) gives focal length, image centre and lens distortion; pixels are undistorted before any geometry."),
        ("Track", "Shi-Tomasi corners tracked frame to frame with pyramidal Lucas-Kanade; a forward-backward check drops bad tracks, and new corners are added when fewer than 350 survive."),
        ("Estimate motion", "Five-point essential matrix with RANSAC, decomposed into rotation and direction of travel; weak frames fall back to constant velocity."),
        ("Scale and chain", "Monocular vision has no metric scale, so each step's length comes from a speed source (ground truth here; an altimeter, wheel odometry or IMU on a robot)."),
        ("Change frames", "A C++17 library (pybind11 bindings, GoogleTest) converts camera-optical poses to ROS body convention for odometry and TF."),
        ("Score", "ATE after best-fit alignment, RPE over 1 and 10 frames, and final drift; CI fails if accuracy drops below the gate."),
    ]
    step_html = "".join(f"<li><b>{h}.</b> {t}</li>" for h, t in steps)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>vision-nav-lab · monocular visual odometry · Shreyash Singh Salv</title>
<meta name="description" content="Monocular visual odometry in Python and OpenCV with a C++17 frames library and a ROS2 node, scored against ground truth: ATE RMSE {m['ate_rmse_m']:.2f} m over {m['path_length_m']:.0f} m.">
<style>
:root{{--g:#1B4332;--g2:#2D6A4F;--l:#EEF5F0;--t:#1B1B1B;--m:#5c6670}}*{{box-sizing:border-box}}
body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--t);line-height:1.55}}
header{{background:var(--g);color:#fff;padding:52px 20px 40px}}.w{{max-width:980px;margin:0 auto;padding:0 20px}}
h1{{font-size:clamp(28px,4.2vw,42px);line-height:1.15;margin:0 0 12px}}h2{{color:var(--g);font-size:24px;margin:0 0 14px}}
header p{{color:#B7E4C7;font-size:18px;max-width:780px;margin:0 0 20px}}.btns a{{display:inline-block;margin:6px 10px 0 0;padding:10px 16px;border-radius:8px;background:#fff;color:var(--g);font-weight:700;text-decoration:none}}
section{{padding:36px 0;border-bottom:1px solid #e4ebe6}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.stat{{background:var(--l);border-radius:12px;padding:16px}}.stat b{{display:block;font-size:30px;color:var(--g)}}.stat span{{display:block;font-weight:700}}.stat small{{color:var(--m)}}
img{{max-width:100%;height:auto;border-radius:8px;border:1px solid #e1e7e3}}ol{{padding-left:20px}}li{{margin:8px 0}}.muted{{color:var(--m);font-size:14px}}
.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;align-items:start}}code{{background:var(--l);padding:1px 5px;border-radius:4px}}
footer{{padding:28px 0 46px;color:var(--m);font-size:14px}}a{{color:var(--g2)}}
</style></head><body>
<header><div class="w"><h1>Where am I without GPS? Monocular visual odometry, scored against ground truth.</h1>
<p>A camera-only navigation pipeline in Python and OpenCV, with a C++17 coordinate-frames library and a ROS2 node. On a {m['frames']}-frame recorded sequence it tracks a {m['path_length_m']:.0f} m path to within {m['ate_rmse_m']:.2f} m RMSE.</p>
<div class="btns"><a href="{REPO}" target="_blank" rel="noopener">Code on GitHub ↗</a><a href="{REPO}#run-it" target="_blank" rel="noopener">Run it ↗</a></div></div></header>
<main class="w">
<section><h2>Results on the synthetic sequence</h2><div class="stats">{card_html}</div>
<p class="muted">The test video is generated with an exact ground-truth path and lens distortion, so every error is measurable. The same code reads TUM RGB-D and KITTI sequences.</p></section>
<section><h2>Estimated path vs ground truth</h2><img src="trajectory.png" alt="Top-down plot: the visual-odometry path follows the ground-truth path closely over 90 metres"><img src="error.png" alt="Position error over time after alignment" style="margin-top:14px"></section>
<section class="two"><div><h2>How it works</h2><ol>{step_html}</ol></div>
<div><h2>Tracked features</h2><img src="tracks.png" alt="A frame with tracked feature points and their motion since the previous frame"><p class="muted">Red: tracked corners. Green: their motion since the previous frame.</p></div></section>
<section><h2>Engineering</h2><ul>
<li><b>Python</b> package <code>vnav</code>: camera model, tracker, motion, odometry, metrics, dataset readers, calibration.</li>
<li><b>C++17</b> frames library with a CLI, pybind11 bindings and 13 GoogleTest tests.</li>
<li><b>ROS2</b> package <code>vnav_ros</code>: plays back recorded sequences or subscribes to <code>camera/image_raw</code>; publishes <code>nav_msgs/Odometry</code>, <code>nav_msgs/Path</code> and TF.</li>
<li><b>CI</b> on Ubuntu Linux: C++ tests, 21 Python tests, a full run with an accuracy gate, and a ROS2 Humble build, node test and smoke test.</li></ul>
<p class="muted">Limits: monocular only, scale from an external source, no loop closure or IMU fusion; results here are on synthetic data.</p></section>
</main>
<footer class="w">Built by Shreyash Singh Salv · <a href="https://github.com/Shreyashsalv2" target="_blank" rel="noopener">github.com/Shreyashsalv2</a></footer>
</body></html>
"""
    (SITE / "index.html").write_text(page)
    print(SITE / "index.html")


if __name__ == "__main__":
    main()
