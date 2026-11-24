#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

#include "vnav/frames.hpp"

using namespace vnav;

namespace {
constexpr double kEps = 1e-9;
constexpr double kPi = 3.14159265358979323846;

void expect_vec(const Vec3& a, const Vec3& b, double eps = kEps) {
  EXPECT_NEAR(a.x, b.x, eps);
  EXPECT_NEAR(a.y, b.y, eps);
  EXPECT_NEAR(a.z, b.z, eps);
}

Pose make(double x, double y, double z, const Mat3& R) { return {{x, y, z}, mat_to_quat(R)}; }
}  // namespace

TEST(Rotation, QuaternionRoundTripForManyAngles) {
  for (double a = -3.0; a <= 3.0; a += 0.37) {
    const Mat3 R = mul(rot_z(a), mul(rot_y(0.5 * a), rot_x(-0.3 * a)));
    const Mat3 R2 = quat_to_mat(mat_to_quat(R));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) EXPECT_NEAR(R[i][j], R2[i][j], 1e-12);
  }
}

TEST(Rotation, HandlesHalfTurnWhereTraceIsNegative) {
  const Quat q = mat_to_quat(rot_x(kPi));
  EXPECT_NEAR(std::abs(q.x), 1.0, 1e-12);
  EXPECT_NEAR(q.w, 0.0, 1e-12);
}

TEST(Rotation, AngleBetweenQuaternions) {
  EXPECT_NEAR(angle_between(mat_to_quat(rot_z(0.0)), mat_to_quat(rot_z(0.7))), 0.7, 1e-12);
  EXPECT_NEAR(angle_between(mat_to_quat(rot_y(0.2)), mat_to_quat(rot_y(0.2))), 0.0, 1e-7);
}

TEST(Rotation, RejectsZeroQuaternion) { EXPECT_THROW(normalize({0, 0, 0, 0}), std::invalid_argument); }

TEST(Transform, ComposeWithInverseIsIdentity) {
  const Pose p = make(1.0, -2.0, 0.5, mul(rot_z(0.4), rot_x(0.2)));
  const Pose i1 = compose(p, inverse(p)), i2 = compose(inverse(p), p);
  expect_vec(i1.t, {0, 0, 0});
  expect_vec(i2.t, {0, 0, 0});
  EXPECT_NEAR(angle_between(i1.q, Quat{}), 0.0, 1e-7);
}

TEST(Transform, YawThenTranslate) {
  // A frame turned 90 degrees left and moved 2 m along x: a point 1 m ahead lands at (2, 1, 0).
  const Pose p = make(2.0, 0.0, 0.0, rot_z(kPi / 2));
  expect_vec(transform_point(p, {1.0, 0.0, 0.0}), {2.0, 1.0, 0.0});
}

TEST(Transform, CompositionIsAssociative) {
  const Pose a = make(1, 2, 3, rot_x(0.3)), b = make(-1, 0.5, 2, rot_y(-0.7)), c = make(0, 0, 1, rot_z(1.1));
  const Pose l = compose(compose(a, b), c), r = compose(a, compose(b, c));
  expect_vec(l.t, r.t, 1e-12);
  EXPECT_NEAR(angle_between(l.q, r.q), 0.0, 1e-7);
}

TEST(Convention, OpticalToBodyAxes) {
  // Optical z (forward) is body x; optical x (right) is body -y; optical y (down) is body -z.
  const Mat3& C = body_from_optical();
  expect_vec(rotate(C, {0, 0, 1}), {1, 0, 0});
  expect_vec(rotate(C, {1, 0, 0}), {0, -1, 0});
  expect_vec(rotate(C, {0, 1, 0}), {0, 0, -1});
}

TEST(Convention, IdentityStaysIdentity) {
  const Pose b = optical_to_body(identity());
  expect_vec(b.t, {0, 0, 0});
  EXPECT_NEAR(angle_between(b.q, Quat{}), 0.0, 1e-7);
}

TEST(Convention, ForwardMotionBecomesBodyX) {
  // Camera moved 5 m forward (optical +z): in body convention the robot moved 5 m along +x.
  const Pose b = optical_to_body(make(0, 0, 5, rot_y(0)));
  expect_vec(b.t, {5, 0, 0});
}

TEST(Convention, TurningRightBecomesNegativeYaw) {
  // Optical yaw is a rotation about +y (down); turning right by a is a body rotation about +z (up) of -a.
  const Pose b = optical_to_body(make(0, 0, 0, rot_y(0.3)));
  const Pose expected{{0, 0, 0}, mat_to_quat(rot_z(-0.3))};
  EXPECT_NEAR(angle_between(b.q, expected.q), 0.0, 1e-7);
}

TEST(TumFormat, WriteThenReadRoundTrip) {
  const std::string path = ::testing::TempDir() + "vnav_traj.txt";
  std::vector<StampedPose> traj{{0.0, identity()}, {0.1, make(1, 2, 3, rot_z(0.5))}};
  write_tum(path, traj);
  const auto back = read_tum(path);
  ASSERT_EQ(back.size(), 2u);
  EXPECT_NEAR(back[1].stamp, 0.1, 1e-6);
  expect_vec(back[1].pose.t, {1, 2, 3}, 1e-8);
  EXPECT_NEAR(angle_between(back[1].pose.q, traj[1].pose.q), 0.0, 1e-7);
  std::remove(path.c_str());
}

TEST(TumFormat, MissingFileThrows) { EXPECT_THROW(read_tum("/nonexistent/vnav.txt"), std::runtime_error); }
