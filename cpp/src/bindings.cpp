// Python bindings: vnav._frames
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vnav/frames.hpp"

namespace py = pybind11;

namespace {
using Vec7 = std::array<double, 7>;  // tx ty tz qx qy qz qw

vnav::Pose to_pose(const Vec7& v) { return {{v[0], v[1], v[2]}, vnav::normalize({v[3], v[4], v[5], v[6]})}; }
Vec7 to_vec(const vnav::Pose& p) { return {p.t.x, p.t.y, p.t.z, p.q.x, p.q.y, p.q.z, p.q.w}; }
}  // namespace

PYBIND11_MODULE(_frames, m) {
  m.doc() = "C++17 coordinate-frame utilities (poses as [tx, ty, tz, qx, qy, qz, qw])";
  m.def("compose", [](const Vec7& a, const Vec7& b) { return to_vec(vnav::compose(to_pose(a), to_pose(b))); });
  m.def("inverse", [](const Vec7& a) { return to_vec(vnav::inverse(to_pose(a))); });
  m.def("transform_point", [](const Vec7& a, const std::array<double, 3>& p) {
    const auto r = vnav::transform_point(to_pose(a), {p[0], p[1], p[2]});
    return std::array<double, 3>{r.x, r.y, r.z};
  });
  m.def("optical_to_body", [](const std::vector<Vec7>& poses) {
    std::vector<Vec7> out;
    out.reserve(poses.size());
    for (const auto& v : poses) out.push_back(to_vec(vnav::optical_to_body(to_pose(v))));
    return out;
  });
}
