import importlib.util
import inspect
import sys
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples" / "python"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    plotting = _load_module("backend_plotting", examples_dir / "backend" / "plotting.py")
    second_order = _load_module("second_order_soliton_rk4ip", examples_dir / "second_order_soliton_rk4ip.py")

    signature = inspect.signature(plotting.plot_wavelength_step_history)
    assert "proposed_step_sizes" not in signature.parameters
    plotting_source = inspect.getsource(plotting.plot_wavelength_step_history)
    assert "proposed_step_sizes" not in plotting_source
    assert ".legend(" not in plotting_source

    save_plots_source = inspect.getsource(second_order.save_plots)
    assert "proposed_step_sizes" not in save_plots_source
    assert "show_step_legend" not in save_plots_source
    print("test_python_second_order_soliton_plot_cleanup: plotting path only uses accepted step sizes.")


if __name__ == "__main__":
    main()
