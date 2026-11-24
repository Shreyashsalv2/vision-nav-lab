#!/usr/bin/env bash
# Build the C++ library, CLI and tests, and the Python module vnav/_frames*.so.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
cmake -S cpp -B build/cpp -DVNAV_BUILD_PYTHON=ON -DPython_EXECUTABLE="$(command -v "$PY")" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
cmake --build build/cpp -j"$(nproc 2>/dev/null || echo 2)"
cp build/cpp/_frames*.so vnav/
echo "built: build/cpp/frames_cli, build/cpp/frames_test, vnav/$(basename build/cpp/_frames*.so)"
