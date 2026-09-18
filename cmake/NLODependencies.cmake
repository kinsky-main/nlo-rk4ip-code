# NLODependencies.cmake - the single place where every third-party source that
# nlolib compiles into itself is pinned.
#
# Each dependency is fixed to a released archive and its SHA-256, so a build is
# reproducible and a tampered or moved archive is rejected at configure time.
# Bump a version here, refresh the hash (sha256sum on the new archive), and keep
# sbom.cdx.json in step; nothing else in the tree needs to change.
#
# Offline hosts: point FETCHCONTENT_SOURCE_DIR_<NAME> at a pre-extracted copy
# (see docker/scripts/make_offline_bundle.sh); the hash is then not consulted.

include_guard(GLOBAL)

# ---- FFTW 3.3.10 (GPL-2.0-or-later) -- CPU FFT backend, built in-tree, static
set(FFTW_VERSION "3.3.10")
set(FFTW_GIT_TAG "fftw-${FFTW_VERSION}"
  CACHE STRING "FFTW release used for the in-tree static build")
set(FFTW_URL "https://www.fftw.org/${FFTW_GIT_TAG}.tar.gz"
  CACHE STRING "FFTW source archive")
set(FFTW_SHA256 "56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
  CACHE STRING "SHA-256 of FFTW_URL")

# ---- SQLite 3.49.2 (public domain) -- snapshot store, amalgamation compiled in
set(SQLITE_VERSION "3.49.2")
set(SQLITE_AMALGAMATION_URL
  "https://www.sqlite.org/2025/sqlite-amalgamation-3490200.zip"
  CACHE STRING "SQLite amalgamation archive")
set(SQLITE_AMALGAMATION_SHA256
  "921fc725517a694df7df38a2a3dfede6684024b5788d9de464187c612afb5918"
  CACHE STRING "SHA-256 of SQLITE_AMALGAMATION_URL")

# ---- VkFFT 1.3.4 (MIT) -- header-only Vulkan FFT, used when ENABLE_VKFFT
set(VKFFT_VERSION "1.3.4")
set(VKFFT_URL
  "https://github.com/DTolm/VkFFT/archive/refs/tags/v${VKFFT_VERSION}.tar.gz"
  CACHE STRING "VkFFT source archive")
set(VKFFT_SHA256 "b61055393adb3adc79009fe12401cbfbbdfba584e665e9c35fcbf4b32fb31f30"
  CACHE STRING "SHA-256 of VKFFT_URL")

# ---- glslang 12.3.1 (BSD-3-Clause) -- runtime GLSL->SPIR-V for VkFFT, static
#      Only when NLOLIB_GLSLANG_PROVIDER=FETCH (the default).
set(GLSLANG_VERSION "12.3.1")
set(GLSLANG_GIT_TAG "${GLSLANG_VERSION}"
  CACHE STRING "glslang release used when NLOLIB_GLSLANG_PROVIDER=FETCH")
set(GLSLANG_URL
  "https://github.com/KhronosGroup/glslang/archive/refs/tags/${GLSLANG_GIT_TAG}.tar.gz"
  CACHE STRING "glslang source archive")
set(GLSLANG_SHA256 "a57836a583b3044087ac51bb0d5d2d803ff84591d55f89087fc29ace42a8b9a8"
  CACHE STRING "SHA-256 of GLSLANG_URL")

# ---- Vulkan-Headers 1.4.341 (Apache-2.0 OR MIT) -- only fetched when no
#      Vulkan SDK / libvulkan-dev headers are found. Matches SDK 1.4.341.x.
set(VULKAN_HEADERS_VERSION "1.4.341.0")
set(VULKAN_HEADERS_URL
  "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-${VULKAN_HEADERS_VERSION}.tar.gz"
  CACHE STRING "Vulkan-Headers source archive used when local headers are unavailable")
set(VULKAN_HEADERS_SHA256 "d73bc5036b6556b741f6985ff600ca720308c5f2850e4a43ceb498bd3de069e7"
  CACHE STRING "SHA-256 of VULKAN_HEADERS_URL")

# ---- doxygen-awesome-css 2.5.0 (MIT) -- docs theme, `docs` target only
set(DOXYGEN_AWESOME_CSS_VERSION "2.5.0")
set(DOXYGEN_AWESOME_CSS_URL
  "https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v${DOXYGEN_AWESOME_CSS_VERSION}.tar.gz"
  CACHE STRING "doxygen-awesome-css source archive")
set(DOXYGEN_AWESOME_CSS_SHA256 "959b6e9369a0f5c7c0a9beba34b18a02bc5e788e68f6733975908f133a784dd5"
  CACHE STRING "SHA-256 of DOXYGEN_AWESOME_CSS_URL")
