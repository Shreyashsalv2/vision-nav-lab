import cv2
import numpy as np
import pytest

from vnav import calibration, io, metrics, motion, se3, synthetic
from vnav.features import FeatureTracker
from vnav.odometry import VisualOdometry, step_lengths

CAM = synthetic.DEFAULT_CAMERA


@pytest.fixture(scope="module")
def short_sequence(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    synthetic.generate(out, synthetic.SyntheticConfig(frames=80, seed=11))
    return io.load_tum_style(out)


def test_synthetic_sequence_layout(short_sequence):
    seq = short_sequence
    assert len(seq.image_paths) == 80 and all(p.exists() for p in seq.image_paths)
    assert np.allclose(seq.gt_poses[0], np.eye(4))
    assert metrics.path_length(seq.gt_poses) == pytest.approx(79 * 0.30, rel=0.05)


def test_synthetic_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cfg = synthetic.SyntheticConfig(frames=3, seed=5)
    synthetic.generate(a, cfg)
    synthetic.generate(b, cfg)
    assert (a / "images/000002.png").read_bytes() == (b / "images/000002.png").read_bytes()


def test_tracker_recovers_known_image_shift(short_sequence):
    img = cv2.imread(str(short_sequence.image_paths[10]), cv2.IMREAD_GRAYSCALE)
    shifted = cv2.warpAffine(img, np.float32([[1, 0, 3.0], [0, 1, -2.0]]), (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_REFLECT)
    tr = FeatureTracker()
    p0, p1 = tr.track(img, shifted, tr.detect(img))
    assert len(p0) > 200
    assert np.allclose(np.median(p1 - p0, axis=0), [3.0, -2.0], atol=0.1)


def test_tracker_detect_respects_existing_points(short_sequence):
    img = cv2.imread(str(short_sequence.image_paths[0]), cv2.IMREAD_GRAYSCALE)
    tr = FeatureTracker()
    first = tr.detect(img)
    more = tr.detect(img, existing=first)
    assert len(first) + len(more) <= tr.cfg.max_corners
    if len(more):
        d = np.min(np.linalg.norm(more[:, None] - first[None], axis=2), axis=1)
        assert d.min() >= tr.cfg.min_distance - 1


def test_relative_pose_from_exact_correspondences():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-4, 4, 300), rng.uniform(-3, 3, 300), rng.uniform(4, 25, 300)])
    R = se3.rot_y(0.05) @ se3.rot_x(-0.02)
    t = np.array([0.1, 0.0, 0.5])
    p0 = CAM.project(X)
    p1 = CAM.project(X @ R.T + t)
    rel = motion.estimate(CAM, p0, p1)
    assert rel is not None and rel.inliers > 250
    assert np.degrees(se3.rotation_angle(rel.R @ R.T)) < 0.05
    cos = rel.t @ (t / np.linalg.norm(t))
    assert np.degrees(np.arccos(min(1.0, cos))) < 0.5


def test_relative_pose_needs_enough_points():
    assert motion.estimate(CAM, np.zeros((5, 2)), np.ones((5, 2))) is None


def test_odometry_accuracy_on_short_sequence(short_sequence):
    seq = short_sequence
    res = VisualOdometry(seq.camera).run(seq.images(), step_lengths(seq.gt_poses))
    m = metrics.summary(res.poses, seq.gt_poses)
    assert res.fallback_rate == 0.0
    assert m["ate_rmse_m"] < 0.15        # over a ~24 m path
    assert m["rpe1_rot_rmse_deg"] < 0.3


def test_calibration_recovers_known_camera():
    views = calibration.render_board_views(CAM, n=12)
    res = calibration.calibrate(views)
    c = res.camera
    assert res.rms_reprojection_px < 0.3 and res.views_used >= 10
    assert c.fx == pytest.approx(CAM.fx, rel=0.01) and c.fy == pytest.approx(CAM.fy, rel=0.01)
    assert abs(c.cx - CAM.cx) < 2 and abs(c.cy - CAM.cy) < 2
    # inside the area the boards covered, the estimated model maps pixels to rays like the true one
    # (k2 and k3 trade off against each other, so compare behaviour, not coefficients)
    grid = np.stack(np.meshgrid(np.linspace(120, 520, 9), np.linspace(100, 380, 7)), -1).reshape(-1, 2)
    assert np.max(np.abs(c.normalize(grid) - CAM.normalize(grid))) * CAM.fx < 2.0


def test_calibration_rejects_images_without_board():
    blank = [np.full((480, 640), 128, np.uint8)] * 4
    with pytest.raises(ValueError):
        calibration.calibrate(blank)
