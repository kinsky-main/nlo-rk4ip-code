#!/usr/bin/env bash
set -euo pipefail

build_dir="build"
do_build=0
install_deps=1
cmake_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      build_dir="$2"
      shift 2
      ;;
    --build)
      do_build=1
      shift
      ;;
    --no-install)
      install_deps=0
      shift
      ;;
    --cmake-arg)
      cmake_args+=("$2")
      shift 2
      ;;
    *)
      echo "Usage: $0 [--build-dir DIR] [--build] [--no-install] [--cmake-arg ARG]"
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${repo_root}/CMakeLists.txt" ]]; then
  echo "Run this script from the nlolib repo root or place setup.sh there."
  exit 1
fi

if [[ "${install_deps}" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "${EUID}" -ne 0 ]]; then
      if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo not available; install dependencies manually."
        exit 1
      fi
      sudo apt-get update
      sudo apt-get install -y cmake ninja-build build-essential pkg-config libfftw3-dev
    else
      apt-get update
      apt-get install -y cmake ninja-build build-essential pkg-config libfftw3-dev
    fi
  else
    echo "apt-get not found; install cmake, build tools, and FFTW3 dev headers manually."
  fi
fi

generator=()
if command -v ninja >/dev/null 2>&1; then
  generator=(-G Ninja)
fi

cmake -S "${repo_root}" -B "${repo_root}/${build_dir}" \
  "${generator[@]}" \
  -D NLO_FFT_BACKEND_FFTW=ON \
  -D NLO_FFT_BACKEND_CUFFT=OFF \
  "${cmake_args[@]}"

if [[ "${do_build}" -eq 1 ]]; then
  cmake --build "${repo_root}/${build_dir}"
fi
