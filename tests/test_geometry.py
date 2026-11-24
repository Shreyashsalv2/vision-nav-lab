import numpy as np
import pytest

from vnav import frames, metrics, se3
from vnav.camera import PinholeCamera
from vnav.synthetic import DEFAULT_CAMERA


def random_pose(rng):
    R = se3.rot_z(rng.uniform(-3, 3)) @ se3.rot_y(rng.uniform(-1.5, 1.5)) @ se3.rot_x(rng.uniform(-3, 3))
    return se3.make_T(R, rng.uniform(-5, 5, 3))


def test_quaternion_round_trip_all_angles():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = random_pose(rng)[:3, :3]
        assert np.allclose(se3.quat_to_rot(se3.rot_to_quat(R)), R, atol=1e-12)
    # half-turn: trace is negative, the branch that naive formulas get wrong
    assert np.allclose(se3.quat_to_rot(se3.rot_to_quat(se3.rot_x(np.pi))), se3.rot_x(np.pi), atol=1e-12)


def test_inverse_and_vec_round_trip():
    rng = np.random.default_rng(1)
    T = random_pose(rng)
    assert np.allclose(T @ se3.inv_T(T), np.eye(4), atol=1e-12)
    assert np.allclose(se3.vec_to_pose(se3.pose_to_vec(T)), T, atol=1e-12)


def test_optical_to_body_conventions():
    forward = se3.make_T(np.eye(3), [0, 0, 5])  # 5 m along optical z
    assert np.allclose(frames.optical_to_body(forward, backend="numpy")[:3, 3], [5, 0, 0])
    right_turn = se3.make_T(se3.rot_y(0.3), [0, 0, 0])  # optical yaw to the right
    assert np.allclose(frames.optical_to_body(right_turn, backend="numpy")[:3, :3], se3.rot_z(-0.3), atol=1e-12)


@pytest.mark.skipif(frames.BACKEND != "cpp", reason="C++ module not built (scripts/build_cpp.sh)")
def test_cpp_and_numpy_frames_agree():
    rng = np.random.default_rng(2)
    poses = np.array([random_pose(rng) for _ in range(50)])
    a = frames.optical_to_body(poses, backend="cpp")
    b = frames.optical_to_body(poses, backend="numpy")
    assert np.allclose(a, b, atol=1e-9)


def test_camera_project_then_normalize_round_trip():
    cam = DEFAULT_CAMERA
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-2, 2, 300), rng.uniform(-1.5, 1.5, 300), rng.uniform(3, 20, 300)])
    px = cam.project(pts)
    n = cam.normalize(px)
    assert np.allclose(n, pts[:, :2] / pts[:, 2:3], atol=1e-6)


def test_camera_save_load(tmp_path):
    p = tmp_path / "cam.json"
    DEFAULT_CAMERA.save(p)
    assert PinholeCamera.load(p) == DEFAULT_CAMERA


def test_camera_from_kitti_projection_matrix():
    P = np.array([[718.856, 0, 607.1928, 0], [0, 718.856, 185.2157, 0], [0, 0, 1, 0]])
    cam = PinholeCamera.from_projection_matrix(P, 1241, 376)
    assert (cam.fx, cam.cy, cam.dist) == (718.856, 185.2157, [0.0] * 5)


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(100, 3))
    R = se3.rot_z(0.7) @ se3.rot_x(-0.2)
    dst = 2.5 * src @ R.T + [1, -2, 3]
    s, R_est, t = metrics.umeyama(src, dst, with_scale=True)
    assert s == pytest.approx(2.5) and np.allclose(R_est, R) and np.allclose(t, [1, -2, 3])


def test_ate_rpe_zero_for_identical_and_drift_measured():
    rng = np.random.default_rng(5)
    gt = np.array([se3.make_T(se3.rot_y(0.01 * k), [0.1 * k, 0, k]) for k in range(50)])
    assert metrics.ate(gt, gt)["rmse_m"] == pytest.approx(0, abs=1e-9)
    assert metrics.rpe(gt, gt, 5)["trans_rmse_m"] == pytest.approx(0, abs=1e-9)
    est = gt.copy()
    est[-1, :3, 3] += [0, 0, 0.49]  # end 0.49 m off
    assert metrics.final_drift_pct(est, gt) == pytest.approx(100 * 0.49 / metrics.path_length(gt))
    noisy = gt.copy()
    noisy[:, :3, 3] += rng.normal(0, 0.1, (50, 3))
    assert 0.05 < metrics.ate(noisy, gt)["rmse_m"] < 0.25
