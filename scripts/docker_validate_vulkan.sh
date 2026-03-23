#!/usr/bin/env bash
set -euo pipefail

image_tag="${NLO_DOCKER_VK_IMAGE:-nlolib-vulkan-dev}"
build_dir="${NLO_DOCKER_BUILD_DIR:-build-docker-vulkan}"

if [[ "${NLO_DOCKER_VALIDATE_DRY_RUN:-0}" != "0" ]]; then
  echo "[nlolib] docker_validate_vulkan.sh dry-run"
  echo "image_tag=${image_tag}"
  echo "build_dir=${build_dir}"
  exit 0
fi

docker build -t "${image_tag}" -f docker/vulkan-dev.Dockerfile .
docker run --rm \
  -v "$(pwd):/workspace/nlolib" \
  -w /workspace/nlolib \
  "${image_tag}" \
  bash -lc "cmake -S . -B ${build_dir} -GNinja \
    -DNLO_ENABLE_VULKAN_BACKEND=ON \
    -DNLO_ENABLE_CUDA_BACKEND=OFF \
    -DNLO_INSTALL_GIT_HOOKS=OFF \
    -DNLO_BUMP_PATCH_ON_BUILD=OFF \
    -DBUILD_TESTING=ON && \
    cmake --build ${build_dir} && \
    ctest --test-dir ${build_dir} -R '^(test_core_state_alloc|test_fft|test_nlo_vector_backend|test_nlo_vector_backend_vulkan)$' --output-on-failure"
