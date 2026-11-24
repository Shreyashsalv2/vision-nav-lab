#include "vnav/frames.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace vnav {

Quat normalize(const Quat& q) {
  const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n < 1e-15) throw std::invalid_argument("zero-length quaternion");
  return {q.x / n, q.y / n, q.z / n, q.w / n};
}

Mat3 quat_to_mat(const Quat& qin) {
  const Quat q = normalize(qin);
  const double x = q.x, y = q.y, z = q.z, w = q.w;
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
           {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
           {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}

Quat mat_to_quat(const Mat3& R) {
  // Shepperd's method: pick the largest diagonal term for numerical stability.
  Quat q;
  const double tr = R[0][0] + R[1][1] + R[2][2];
  if (tr > 0) {
    const double s = std::sqrt(tr + 1.0) * 2;
    q = {(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s};
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const double s = std::sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2;
    q = {0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s};
  } else if (R[1][1] > R[2][2]) {
    const double s = std::sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2;
    q = {(R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s};
  } else {
    const double s = std::sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2;
    q = {(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s};
  }
  q = normalize(q);
  if (q.w < 0) q = {-q.x, -q.y, -q.z, -q.w};
  return q;
}

Mat3 mul(const Mat3& A, const Mat3& B) {
  Mat3 C{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) C[i][j] += A[i][k] * B[k][j];
  return C;
}

Mat3 transpose(const Mat3& A) {
  Mat3 T{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) T[i][j] = A[j][i];
  return T;
}

Vec3 rotate(const Mat3& R, const Vec3& v) {
  return {R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z,
          R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z,
          R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z};
}

Mat3 rot_x(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{1, 0, 0}, {0, c, -s}, {0, s, c}}};
}
Mat3 rot_y(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}};
}
Mat3 rot_z(double a) {
  const double c = std::cos(a), s = std::sin(a);
  return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

double angle_between(const Quat& a, const Quat& b) {
  const Quat qa = normalize(a), qb = normalize(b);
  const double d = std::abs(qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w);
  return 2.0 * std::acos(std::min(1.0, d));
}

Pose identity() { return Pose{}; }

Pose compose(const Pose& a, const Pose& b) {
  const Mat3 Ra = quat_to_mat(a.q), Rb = quat_to_mat(b.q);
  const Vec3 t = rotate(Ra, b.t);
  return {{a.t.x + t.x, a.t.y + t.y, a.t.z + t.z}, mat_to_quat(mul(Ra, Rb))};
}

Pose inverse(const Pose& p) {
  const Mat3 Rt = transpose(quat_to_mat(p.q));
  const Vec3 t = rotate(Rt, p.t);
  return {{-t.x, -t.y, -t.z}, mat_to_quat(Rt)};
}

Vec3 transform_point(const Pose& p, const Vec3& v) {
  const Vec3 r = rotate(quat_to_mat(p.q), v);
  return {r.x + p.t.x, r.y + p.t.y, r.z + p.t.z};
}

const Mat3& body_from_optical() {
  static const Mat3 R{{{0, 0, 1}, {-1, 0, 0}, {0, -1, 0}}};
  return R;
}

Pose optical_to_body(const Pose& T_w_c) {
  const Pose C{{0, 0, 0}, mat_to_quat(body_from_optical())};
  return compose(compose(C, T_w_c), inverse(C));
}

std::vector<Pose> optical_to_body(const std::vector<Pose>& poses) {
  std::vector<Pose> out;
  out.reserve(poses.size());
  for (const auto& p : poses) out.push_back(optical_to_body(p));
  return out;
}

std::vector<StampedPose> read_tum(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<StampedPose> traj;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    StampedPose sp;
    if (!(ss >> sp.stamp >> sp.pose.t.x >> sp.pose.t.y >> sp.pose.t.z >> sp.pose.q.x >> sp.pose.q.y >> sp.pose.q.z >>
          sp.pose.q.w))
      throw std::runtime_error("bad TUM line: " + line);
    sp.pose.q = normalize(sp.pose.q);
    traj.push_back(sp);
  }
  return traj;
}

void write_tum(const std::string& path, const std::vector<StampedPose>& traj) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "# timestamp tx ty tz qx qy qz qw\n" << std::fixed;
  for (const auto& sp : traj) {
    const auto& p = sp.pose;
    out << std::setprecision(6) << sp.stamp << ' ' << std::setprecision(9) << p.t.x << ' ' << p.t.y << ' ' << p.t.z
        << ' ' << p.q.x << ' ' << p.q.y << ' ' << p.q.z << ' ' << p.q.w << '\n';
  }
}

}  // namespace vnav
