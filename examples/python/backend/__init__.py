"""Internal OOP helpers for nlolib Python examples."""

from .app_base import ExampleAppBase
from .cli import build_example_parser
from .metrics import (
    DEFAULT_RELATIVE_ERROR_EPS,
    mean_pointwise_abs_relative_error,
    mean_pointwise_abs_relative_error_curve,
    pointwise_abs_relative_error,
)
from .plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_convergence_loglog,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_mode_power_exchange,
    plot_phase_shift_comparison,
    plot_summary_curve,
    plot_three_curve_drift,
    plot_total_error_over_propagation,
    plot_two_curve_comparison,
    plot_wavelength_step_history,
)
from .runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)
from .spectral import (
    SPEED_OF_LIGHT_M_PER_S,
    carrier_wavelength_nm_to_frequency_hz,
    frequency_hz_to_wavelength_nm,
    omega_centroid_to_wavelength_nm,
    omega_detuning_to_frequency_hz,
    omega_detuning_to_wavelength_nm,
)
from .storage import CaseListing, ExampleRunDB, LoadedCase

__all__ = [
    "ExampleAppBase",
    "NloExampleRunner",
    "SimulationOptions",
    "TemporalSimulationConfig",
    "centered_time_grid",
    "build_example_parser",
    "DEFAULT_RELATIVE_ERROR_EPS",
    "pointwise_abs_relative_error",
    "mean_pointwise_abs_relative_error",
    "mean_pointwise_abs_relative_error_curve",
    "SPEED_OF_LIGHT_M_PER_S",
    "carrier_wavelength_nm_to_frequency_hz",
    "frequency_hz_to_wavelength_nm",
    "omega_detuning_to_frequency_hz",
    "omega_detuning_to_wavelength_nm",
    "omega_centroid_to_wavelength_nm",
    "ExampleRunDB",
    "CaseListing",
    "LoadedCase",
    "plot_3d_intensity_scatter_propagation",
    "plot_convergence_loglog",
    "plot_intensity_colormap_vs_propagation",
    "plot_final_re_im_comparison",
    "plot_final_intensity_comparison",
    "plot_mode_power_exchange",
    "plot_phase_shift_comparison",
    "plot_summary_curve",
    "plot_three_curve_drift",
    "plot_total_error_over_propagation",
    "plot_two_curve_comparison",
    "plot_wavelength_step_history",
]
