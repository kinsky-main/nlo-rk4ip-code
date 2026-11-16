"""Vulkan operator JIT from a relocated install, not from the source tree.

The archived GPU branch read the JIT's GLSL includes at runtime from the
build-time source path, so an installed wheel found nothing there and fell back
to the operator interpreter: correct results, no speedup, no error. The port
embeds those sources at build time, and `operator_jit_programs` /
`operator_jit_fallbacks` in the perf-profile snapshot are what make the two
paths distinguishable at all.

So this builds the Python package, installs it into a fresh virtualenv under
the system temp directory -- no part of which is the source tree -- and from
there asserts that a small Vulkan propagate bound at least one program to a
JIT kernel and left none on the interpreter. The counters accumulate whether or
not profiling is enabled, so nothing here turns it on.

Skips (exit 0) when no Vulkan device answers the loader, and when the
environment cannot build the package at all (no build backend reachable).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NAME = "test_python_wheel_jit_relocated"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _case():
    """A small Kerr 1-D run; every operator program in it is JIT-lowerable."""
    import math

    n, dt = 256, 0.01
    field = [complex(math.exp(-(((i - n / 2) * dt / 0.2) ** 2)), 0.0) for i in range(n)]
    omega = [
        complex(2.0 * math.pi * (i if i < (n + 1) // 2 else i - n) / (n * dt), 0.0)
        for i in range(n)
    ]
    return n, dt, field, omega


def _config(nlolib):
    n, dt, field, omega = _case()
    cfg = nlolib.prepare_sim_config(
        n,
        propagation_distance=0.005,
        starting_step_size=1.0e-3,
        max_step_size=1.0e-3,
        min_step_size=1.0e-3,
        error_tolerance=1.0e-7,
        pulse_period=n * dt,
        delta_time=dt,
        frequency_grid=omega,
        runtime=nlolib.RuntimeOperators(
            linear_factor_expr="i*c0*(w*w)",
            linear_expr="exp(h*D)",
            nonlinear_expr="i*A*c1*I",
            constants=[-0.005, 80.0],
        ),
    )
    return cfg, field


def _stage_package(stage_dir, library):
    """Reproduce what tools/release/build_wheel_linux.sh stages, out of tree."""
    shutil.copy2(REPO_ROOT / "pyproject.toml", stage_dir / "pyproject.toml")
    shutil.copy2(REPO_ROOT / "setup.py", stage_dir / "setup.py")
    shutil.copytree(
        REPO_ROOT / "python",
        stage_dir / "python",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    package_dir = stage_dir / "python" / "nlolib"
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library, package_dir / library.name)


def _venv_python(venv_dir):
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_installed():
    """Body of the test, run by the virtualenv's interpreter (see main)."""
    import nlolib

    # Nothing may resolve to the checkout: no PYTHONPATH, no script directory
    # and no working directory on sys.path (python -I), and no NLOLIB_LIBRARY.
    assert "NLOLIB_LIBRARY" not in os.environ, "library discovery must not be overridden"
    package = Path(nlolib.__file__).resolve()
    assert REPO_ROOT not in package.parents, f"imported the source tree: {package}"
    assert Path(sys.prefix).resolve() in package.parents, f"not installed in the venv: {package}"
    nlolib.load()

    api = nlolib.NLolib()
    api.set_progress_options(enabled=False)
    cfg, field = _config(nlolib)

    # The counters have to carry the signal with profiling off, which is how
    # every ordinary run is configured.
    assert not nlolib.perf_profile_is_enabled(), "profiling must stay disabled"

    # The CPU backend has no JIT, so neither counter may move on it: a
    # non-zero reading there would mean the Vulkan assertion below proves
    # nothing about the GPU path.
    cpu = nlolib.default_execution_options(nlolib.VECTOR_BACKEND_CPU)
    nlolib.perf_profile_reset()
    api.propagate(cfg, field, 2, cpu)
    on_cpu = nlolib.perf_profile_read()
    assert on_cpu.operator_jit_programs == 0, on_cpu.operator_jit_programs
    assert on_cpu.operator_jit_fallbacks == 0, on_cpu.operator_jit_fallbacks

    vulkan = nlolib.default_execution_options(nlolib.VECTOR_BACKEND_VULKAN)
    nlolib.perf_profile_reset()
    api.propagate(cfg, field, 2, vulkan)
    on_gpu = nlolib.perf_profile_read()
    print(
        f"{NAME}: installed at {package.parent}, Vulkan run compiled "
        f"{on_gpu.operator_jit_programs} JIT program(s), "
        f"{on_gpu.operator_jit_fallbacks} interpreter fallback(s)"
    )
    assert on_gpu.operator_jit_programs >= 1, (
        "the relocated install fell back to the operator interpreter: "
        f"programs={on_gpu.operator_jit_programs} fallbacks={on_gpu.operator_jit_fallbacks}"
    )
    assert on_gpu.operator_jit_fallbacks == 0, on_gpu.operator_jit_fallbacks


def _vulkan_available():
    """Probe the in-tree package, so no wheel is built on a machine with no GPU."""
    import nlolib

    api = nlolib.NLolib()
    api.set_progress_options(enabled=False)
    cfg, field = _config(nlolib)
    try:
        api.propagate(cfg, field, 2, nlolib.default_execution_options(nlolib.VECTOR_BACKEND_VULKAN))
    except Exception as exc:  # noqa: BLE001 - any failure means no usable Vulkan device
        print(f"{NAME}: Vulkan unavailable ({exc}); skipped.")
        return False
    return True


def main():
    library = os.environ.get("NLOLIB_LIBRARY")
    if not library or not Path(library).is_file():
        print(f"{NAME}: NLOLIB_LIBRARY is not set to a built library; skipped.")
        return
    if not _vulkan_available():
        return

    work = Path(tempfile.mkdtemp(prefix="nlolib-jit-relocated-")).resolve()
    assert REPO_ROOT not in work.parents, f"temp directory is inside the source tree: {work}"
    try:
        stage = work / "stage"
        stage.mkdir()
        _stage_package(stage, Path(library).resolve())

        venv = work / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = _venv_python(venv)
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "--no-input", str(stage)],
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            # No reachable build backend (pip builds the package in an isolated
            # environment). That is an environment limit, not a JIT regression.
            tail = (install.stdout + install.stderr).strip().splitlines()[-5:]
            print(f"{NAME}: cannot build the package here; skipped. " + " | ".join(tail))
            return

        # -I drops PYTHONPATH, the script directory and the working directory
        # from sys.path, and NLOLIB_LIBRARY is removed, so the run has nothing
        # to resolve against but the installed package. NLO_OPERATOR_JIT is
        # deliberately left alone: setting it to 0 must make this test fail.
        env = {k: v for k, v in os.environ.items() if k not in ("NLOLIB_LIBRARY", "PYTHONPATH")}
        proof = subprocess.run(
            [str(python), "-I", str(Path(__file__).resolve()), "--installed"],
            cwd=str(work),
            env=env,
        )
        assert proof.returncode == 0, f"the installed package failed the JIT check ({proof.returncode})"
        print(f"{NAME}: the relocated install took the Vulkan JIT path.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    if "--installed" in sys.argv[1:]:
        run_installed()
    else:
        main()
