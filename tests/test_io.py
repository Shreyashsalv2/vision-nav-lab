import cv2
import numpy as np

from vnav import io, se3


def test_tum_round_trip(tmp_path):
    poses = np.array([se3.make_T(se3.rot_z(0.1 * k), [k, -k, 0.5 * k]) for k in range(5)])
    io.write_tum(tmp_path / "t.txt", np.arange(5) * 0.1, poses)
    stamps, back = io.read_tum(tmp_path / "t.txt")
    assert np.allclose(stamps, np.arange(5) * 0.1) and np.allclose(back, poses, atol=1e-8)


def test_associate_matches_nearest_within_tolerance():
    pairs = io.associate(np.array([0.0, 0.10, 0.20, 0.50]), np.array([0.005, 0.099, 0.3]), max_dt=0.02)
    assert pairs == [(0, 0), (1, 1)]


def test_kitti_loader(tmp_path):
    seq = tmp_path / "sequences" / "07"
    (seq / "image_0").mkdir(parents=True)
    (tmp_path / "poses").mkdir()
    for k in range(3):
        cv2.imwrite(str(seq / "image_0" / f"{k:06d}.png"), np.zeros((376, 1241), np.uint8))
    (seq / "calib.txt").write_text("P0: 718.856 0 607.1928 0 0 718.856 185.2157 0 0 0 1 0\nP1: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    (seq / "times.txt").write_text("0.0\n0.1\n0.2\n")
    rows = ["1 0 0 0 0 1 0 0 0 0 1 0", "1 0 0 0 0 1 0 0 0 0 1 1", "1 0 0 0.1 0 1 0 0 0 0 1 2"]
    (tmp_path / "poses" / "07.txt").write_text("\n".join(rows) + "\n")
    s = io.load_kitti(tmp_path, "07")
    assert s.camera.fx == 718.856 and s.camera.width == 1241
    assert len(s.image_paths) == 3 and np.allclose(s.gt_poses[2][:3, 3], [0.1, 0, 2])
