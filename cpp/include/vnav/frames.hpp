// Coordinate-frame utilities for vision-aided navigation (C++17, no dependencies).
//
// Poses follow the TUM / ROS convention: translation (x, y, z) + unit quaternion (x, y, z, w).
// A pose T_a_b maps points from frame b into frame a.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace vnav {

struct Vec3 {
  double x{0}, y{0}, z{0};
};

struct Quat {
  double x{0}, y{0}, z{0}, w{1};
};

using Mat3 = std::array<std::array<double, 3>, 3>;

struct Pose {
  Vec3 t;
  Quat q;
};

// --- rotations ---
Quat normalize(const Quat& q);
Mat3 quat_to_mat(const Quat& q);
Quat mat_to_quat(const Mat3& R);            // returns w >= 0
Mat3 mul(const Mat3& A, const Mat3& B);
Mat3 transpose(const Mat3& A);
Vec3 rotate(const Mat3& R, const Vec3& v);
Mat3 rot_x(double a);
Mat3 rot_y(double a);
Mat3 rot_z(double a);
double angle_between(const Quat& a, const Quat& b);  // radians, in [0, pi]

// --- rigid transforms ---
Pose identity();
Pose compose(const Pose& a, const Pose& b);  // a * b
Pose inverse(const Pose& p);
Vec3 transform_point(const Pose& p, const Vec3& v);

// --- camera conventions ---
// Rotation taking camera optical axes (x right, y down, z forward)
// to robot body axes (x forward, y left, z up), per ROS REP-103.
const Mat3& body_from_optical();

// Re-express a camera pose given in an optical-convention world in body convention,
// for both the world frame and the moving frame: C * T_w_c * C^-1.
Pose optical_to_body(const Pose& T_w_c);
std::vector<Pose> optical_to_body(const std::vector<Pose>& poses);

// --- TUM trajectory text format: "timestamp tx ty tz qx qy qz qw" ---
struct StampedPose {
  double stamp{0};
  Pose pose;
};
std::vector<StampedPose> read_tum(const std::string& path);
void write_tum(const std::string& path, const std::vector<StampedPose>& traj);

}  // namespace vnav
