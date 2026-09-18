#!/usr/bin/env python3
"""Generate sbom.cdx.json (CycloneDX 1.6) from cmake/NLODependencies.cmake.

The CMake pin table is the single source of truth for every third-party
version, URL and SHA-256 that is compiled into nlolib. This script reads it and
emits the SBOM, so the two can never disagree. Re-run after bumping a pin:

    python tools/write_sbom.py            # writes ./sbom.cdx.json
    python tools/write_sbom.py --check    # exit 1 if sbom.cdx.json is stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN_FILE = ROOT / "cmake" / "NLODependencies.cmake"
SBOM_FILE = ROOT / "sbom.cdx.json"


def read_pins() -> dict[str, str]:
    raw = dict(re.findall(r'set\((\w+)\s+"([^"]*)"', PIN_FILE.read_text()))

    def expand(value: str) -> str:
        return re.sub(r"\$\{(\w+)\}", lambda m: expand(raw[m.group(1)]), value)

    return {k: expand(v) for k, v in raw.items()}


def project_version() -> str:
    text = (ROOT / "CMakeLists.txt").read_text()
    return re.search(r"project\(nlolib VERSION ([0-9.]+)", text).group(1)


def comp(name, version, ctype, lic, role, scope, desc, *, purl=None, url=None,
         sha=None, props=None, bomref=None):
    c = {
        "type": ctype,
        "bom-ref": bomref or f"{name}@{version}",
        "name": name,
        "version": version,
        "description": desc,
        "scope": scope,
        "properties": [{"name": "nlolib:role", "value": role}]
        + [{"name": k, "value": v} for k, v in (props or {}).items()],
    }
    if lic:
        if any(op in lic for op in (" AND ", " OR ", " WITH ")):
            c["licenses"] = [{"expression": lic}]
        else:
            c["licenses"] = [{"license": {"id": lic}}]
    if purl:
        c["purl"] = purl
    if url:
        ref = {"type": "distribution", "url": url}
        if sha:
            ref["hashes"] = [{"alg": "SHA-256", "content": sha}]
        c["externalReferences"] = [ref]
    return c


def build_bom(P: dict[str, str], version: str, serial: str) -> dict:
    fftw, sqlite, vkfft = P["FFTW_VERSION"], P["SQLITE_VERSION"], P["VKFFT_VERSION"]
    glslang, vkh, dox = P["GLSLANG_VERSION"], P["VULKAN_HEADERS_VERSION"], P["DOXYGEN_AWESOME_CSS_VERSION"]

    components = [
        # ---- compiled into libnlolib -------------------------------------
        comp("fftw", fftw, "library", "GPL-2.0-or-later", "compiled-in", "required",
             "CPU FFT backend. Built in-tree as a static library (double precision, "
             "no threads, no SIMD dispatch) and linked into nlolib.",
             purl=f"pkg:generic/fftw@{fftw}?download_url={P['FFTW_URL']}&checksum=sha256:{P['FFTW_SHA256']}",
             url=P["FFTW_URL"], sha=P["FFTW_SHA256"],
             props={"nlolib:cmake": "cmake/NLOFftw.cmake (FFTW_URL, FFTW_SHA256)",
                    "nlolib:linkage": "static"}),
        comp("sqlite", sqlite, "library", "blessing", "compiled-in", "required",
             "Snapshot/checkpoint store. Amalgamation compiled directly into nlolib.",
             purl=f"pkg:generic/sqlite@{sqlite}?download_url={P['SQLITE_AMALGAMATION_URL']}"
                  f"&checksum=sha256:{P['SQLITE_AMALGAMATION_SHA256']}",
             url=P["SQLITE_AMALGAMATION_URL"], sha=P["SQLITE_AMALGAMATION_SHA256"],
             props={"nlolib:cmake": "cmake/NLOSQLite.cmake (SQLITE_AMALGAMATION_URL, SQLITE_AMALGAMATION_SHA256)",
                    "nlolib:linkage": "static"}),
        comp("VkFFT", vkfft, "library", "MIT", "compiled-in", "required",
             "Vulkan FFT, header-only. Compiled into nlolib when ENABLE_VKFFT=ON (default). "
             "Needs a C++17 compiler for the translation units that include it.",
             purl=f"pkg:github/DTolm/VkFFT@v{vkfft}", url=P["VKFFT_URL"], sha=P["VKFFT_SHA256"],
             props={"nlolib:cmake": "src/fft/CMakeLists.txt (VKFFT_URL, VKFFT_SHA256); option ENABLE_VKFFT",
                    "nlolib:linkage": "header-only"}),
        comp("glslang", glslang, "library", "BSD-3-Clause AND BSD-2-Clause AND MIT AND Apache-2.0",
             "compiled-in", "required",
             "Runtime GLSL to SPIR-V compiler used by VkFFT to build its kernels on first use. "
             "Built in-tree as static libraries (glslang, SPIRV; HLSL, optimizer and binaries off) "
             "when NLOLIB_GLSLANG_PROVIDER=FETCH (default). C++17.",
             purl=f"pkg:github/KhronosGroup/glslang@{glslang}", url=P["GLSLANG_URL"], sha=P["GLSLANG_SHA256"],
             props={"nlolib:cmake": "cmake/ResolveGlslang.cmake (GLSLANG_URL, GLSLANG_SHA256); NLOLIB_GLSLANG_PROVIDER",
                    "nlolib:linkage": "static",
                    "nlolib:alternative": "NLOLIB_GLSLANG_PROVIDER=SYSTEM links the distro/SDK glslang "
                                          "instead (Ubuntu 24.04: glslang-dev 15.1.0)"}),
        comp("Vulkan-Headers", vkh, "library", "Apache-2.0 OR MIT", "compiled-in", "optional",
             "Vulkan API headers. Fetched only when neither VULKAN_SDK nor system headers "
             "(libvulkan-dev) are found; otherwise the installed headers are used. "
             "Header version matches Vulkan SDK 1.4.341.x.",
             purl=f"pkg:github/KhronosGroup/Vulkan-Headers@vulkan-sdk-{vkh}",
             url=P["VULKAN_HEADERS_URL"], sha=P["VULKAN_HEADERS_SHA256"],
             props={"nlolib:cmake": "cmake/ResolveVulkan.cmake (VULKAN_HEADERS_URL, VULKAN_HEADERS_SHA256)",
                    "nlolib:linkage": "header-only"}),
        # ---- runtime, resolved on the target machine ---------------------
        comp("vulkan-loader", "1.2", "library", "Apache-2.0", "runtime", "required",
             "Khronos Vulkan ICD loader. The embedded kernels target Vulkan 1.2, so any loader plus "
             "a conformant driver exposing API >= 1.2 is needed for the GPU backend. "
             "Linux: libvulkan.so.1 is a link-time dependency (Ubuntu package libvulkan1). "
             "Windows: vulkan-1.dll is delay-loaded and probed by src/backend/vk_auto_context.c; "
             "when absent nlolib still loads and uses the CPU backend.",
             purl="pkg:generic/vulkan-loader@1.2", bomref="vulkan-loader@>=1.2",
             props={"nlolib:constraint": ">=1.2 (any 1.x loader)",
                    "nlolib:platform": "linux: hard; windows/macos: soft (delay-loaded / dlopen)"}),
        comp("libstdc++", "6", "library", "GPL-3.0-or-later WITH GCC-exception-3.1", "runtime", "required",
             "C++ standard library shared object required on Linux because glslang and the VkFFT "
             "translation units are C++ (libstdc++.so.6 plus libgcc_s.so.1). The target needs a "
             "libstdc++ at least as new as the GCC that built nlolib (13.3 for the Ubuntu 24.04 "
             "artifact). On Windows the MSVC runtime is linked statically "
             "(NLOLIB_STATIC_MSVC_RUNTIME=ON by default) so no redistributable is needed.",
             purl="pkg:generic/libstdc%2B%2B@6", bomref="libstdc++@6",
             props={"nlolib:platform": "linux only"}),
        # ---- build tools (not shipped) -----------------------------------
        comp("cmake", "3.22.1", "application", "BSD-3-Clause", "build-tool", "excluded",
             "Build system generator. cmake_minimum_required(VERSION 3.22.1); 3.28.3 verified on "
             "Ubuntu 24.04, 4.x on Windows.",
             purl="pkg:generic/cmake@3.22.1", bomref="cmake@>=3.22.1",
             props={"nlolib:constraint": ">=3.22.1"}),
        comp("glslangValidator", "vulkan1.2", "application", "BSD-3-Clause", "build-tool", "excluded",
             "Compiles src/backend/vulkan/kernels/*.comp to SPIR-V at build time; the words are then "
             "embedded into nlolib by cmake/embed_spirv.cmake, so the tool is not needed at runtime. "
             "Any glslang release that accepts --target-env vulkan1.2 works. Sources: LunarG Vulkan "
             "SDK (Windows dev machine: 1.4.341.1) or Ubuntu package glslang-tools (24.04: 15.1.0).",
             purl="pkg:generic/glslang-tools", bomref="glslangValidator",
             props={"nlolib:constraint": "supports --target-env vulkan1.2",
                    "nlolib:cmake": "cmake/NLOVulkanBackend.cmake find_program(GLSLANG_VALIDATOR)"}),
        comp("c-compiler", "C99", "application", None, "build-tool", "excluded",
             "C99 compiler for nlolib itself (CMAKE_C_STANDARD 99). Verified: MSVC 19.44 "
             "(VS 2022 17.14), GCC 13.3.",
             purl="pkg:generic/c-compiler", bomref="c-compiler", props={"nlolib:constraint": "C99"}),
        comp("cxx-compiler", "C++17", "application", None, "build-tool", "excluded",
             "C++17 compiler, needed only because glslang and the VkFFT translation units are C++ "
             "(enable_language(CXX) when ENABLE_VKFFT=ON). Verified: MSVC 19.44, GCC 13.3.",
             purl="pkg:generic/cxx-compiler", bomref="cxx-compiler", props={"nlolib:constraint": "C++17"}),
        # ---- optional: docs, tests, bindings -----------------------------
        comp("doxygen-awesome-css", dox, "library", "MIT", "docs", "excluded",
             "Doxygen HTML theme. Fetched only for the docs target (NLOLIB_BUILD_DOCS=ON and "
             "Doxygen found); not part of the library.",
             purl=f"pkg:github/jothepro/doxygen-awesome-css@v{dox}",
             url=P["DOXYGEN_AWESOME_CSS_URL"], sha=P["DOXYGEN_AWESOME_CSS_SHA256"],
             props={"nlolib:cmake": "cmake/NLODocs.cmake (DOXYGEN_AWESOME_CSS_URL, DOXYGEN_AWESOME_CSS_SHA256)"}),
        comp("doxygen", "1.9", "application", "GPL-2.0-only", "docs", "excluded",
             "API documentation generator for the optional docs target (Graphviz for graphs). "
             "1.9.8 verified on Ubuntu 24.04.",
             purl="pkg:generic/doxygen@1.9", bomref="doxygen@>=1.9", props={"nlolib:constraint": ">=1.9"}),
        comp("python", "3.9", "application", "PSF-2.0", "test", "excluded",
             "Interpreter for the Python binding (python/nlolib, ctypes, no compiled extension) and "
             "the CTest Python suite. pyproject.toml: requires-python >=3.9. 3.12 verified.",
             purl="pkg:generic/python@3.9", bomref="python@>=3.9",
             props={"nlolib:constraint": ">=3.9",
                    "nlolib:cmake": "tests/python/CMakeLists.txt find_package(Python3)"}),
        comp("numpy", "1.24", "library", "BSD-3-Clause", "test", "excluded",
             "Required by the Python binding, tests and examples (examples/python/requirements.txt).",
             purl="pkg:pypi/numpy@1.24", bomref="numpy@>=1.24", props={"nlolib:constraint": ">=1.24"}),
        comp("matplotlib", "3.8", "library", "PSF-2.0", "test", "excluded",
             "Plotting in Python examples and several CTest cases (requirements.txt: >=3.8; "
             "3.6.3 from Ubuntu 24.04 apt also passed the suite).",
             purl="pkg:pypi/matplotlib@3.8", bomref="matplotlib@>=3.8", props={"nlolib:constraint": ">=3.8"}),
        comp("imageio", "2.37", "library", "BSD-2-Clause", "examples", "excluded",
             "Animation output in a few Python examples only (with imageio-ffmpeg>=0.5).",
             purl="pkg:pypi/imageio@2.37", bomref="imageio@>=2.37", props={"nlolib:constraint": ">=2.37"}),
        comp("julia", "1.10", "application", "MIT", "test", "excluded",
             "Julia binding (julia/Project.toml compat julia = 1.10); CTest skips the Julia cases "
             "when no julia binary is found.",
             purl="pkg:generic/julia@1.10", bomref="julia@1.10", props={"nlolib:constraint": "^1.10"}),
        comp("MATLAB", "R2019b", "application", None, "bindings", "excluded",
             "MATLAB toolbox target (matlab_stage / package_mltbx.m). matlab/README.md: R2019b or "
             "later, no compiler needed on the client.",
             purl="pkg:generic/matlab@R2019b", bomref="MATLAB@>=R2019b",
             props={"nlolib:constraint": ">=R2019b"}),
    ]

    return {
        "$schema": "http://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": serial,
        "version": 1,
        "metadata": {
            "timestamp": "2026-09-18T00:00:00Z",
            "authors": [{"name": "Wenzel Kinsky"}],
            "component": {
                "type": "library",
                "bom-ref": f"nlolib@{version}",
                "name": "nlolib",
                "version": version,
                "description": "C99 nonlinear-optics RK4IP solver with CPU (FFTW) and Vulkan (VkFFT) "
                               "backends; SPIR-V kernels embedded, third-party code linked statically, "
                               "Python/MATLAB/Julia bindings.",
                "licenses": [{"license": {"id": "GPL-2.0-or-later"}}],
                "purl": f"pkg:github/kinsky-main/nlo-rk4ip-code@{version}",
                "externalReferences": [{"type": "vcs", "url": "https://github.com/kinsky-main/nlo-rk4ip-code"}],
            },
            "properties": [
                {"name": "nlolib:source-of-truth",
                 "value": "Generated by tools/write_sbom.py from cmake/NLODependencies.cmake; edit the pin table, then re-run."},
                {"name": "nlolib:scope-legend",
                 "value": "required = in or needed by libnlolib at runtime; optional = fallback only; "
                          "excluded = build/test/docs tooling, never shipped."},
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": f"nlolib@{version}",
             "dependsOn": [f"fftw@{fftw}", f"sqlite@{sqlite}", f"VkFFT@{vkfft}", f"glslang@{glslang}",
                           "vulkan-loader@>=1.2", "libstdc++@6"]},
            {"ref": f"VkFFT@{vkfft}", "dependsOn": [f"glslang@{glslang}", f"Vulkan-Headers@{vkh}"]},
            {"ref": f"glslang@{glslang}", "dependsOn": ["libstdc++@6"]},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify sbom.cdx.json matches the pins; do not write")
    args = ap.parse_args()

    pins = read_pins()
    version = project_version()

    if args.check:
        if not SBOM_FILE.exists():
            print("sbom.cdx.json missing; run tools/write_sbom.py", file=sys.stderr)
            return 1
        current = json.loads(SBOM_FILE.read_text())
        expected = build_bom(pins, version, current.get("serialNumber", "urn:uuid:0"))
        strip = lambda b: {k: v for k, v in b.items() if k != "metadata"} | {"metadata": {k: v for k, v in b["metadata"].items() if k != "timestamp"}}
        if strip(current) != strip(expected):
            print("sbom.cdx.json is stale relative to cmake/NLODependencies.cmake; run tools/write_sbom.py", file=sys.stderr)
            return 1
        print("sbom.cdx.json matches cmake/NLODependencies.cmake")
        return 0

    serial = f"urn:uuid:{uuid.uuid4()}"
    if SBOM_FILE.exists():
        try:
            serial = json.loads(SBOM_FILE.read_text()).get("serialNumber", serial)
        except json.JSONDecodeError:
            pass
    bom = build_bom(pins, version, serial)
    SBOM_FILE.write_text(json.dumps(bom, indent=2) + "\n", newline="\n")
    print(f"wrote {SBOM_FILE.relative_to(ROOT)}: {len(bom['components'])} components, nlolib {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
