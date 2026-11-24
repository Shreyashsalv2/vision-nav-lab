// frames_cli: convert a TUM trajectory of camera poses from optical to body convention.
//   frames_cli to-body <in.txt> <out.txt>
#include <iostream>

#include "vnav/frames.hpp"

int main(int argc, char** argv) {
  if (argc != 4 || std::string(argv[1]) != "to-body") {
    std::cerr << "usage: frames_cli to-body <in_tum.txt> <out_tum.txt>\n";
    return 2;
  }
  try {
    auto traj = vnav::read_tum(argv[2]);
    for (auto& sp : traj) sp.pose = vnav::optical_to_body(sp.pose);
    vnav::write_tum(argv[3], traj);
    std::cout << "converted " << traj.size() << " poses\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
