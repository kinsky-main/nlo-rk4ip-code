#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
cmake_args=()

if [[ -x /usr/bin/python3 ]]; then
  cmake_args+=(-D Python3_EXECUTABLE=/usr/bin/python3)
fi

cmake -S . -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure
