"""Internal OOP helpers for nlolib Python examples."""

from .plotting import (
    plot_3d_intensity_scatter_propagation,
    plot_final_intensity_comparison,
    plot_final_re_im_comparison,
    plot_intensity_colormap_vs_propagation,
    plot_total_error_over_propagation,
)
from .runner import (
    NloExampleRunner,
    SimulationOptions,
    TemporalSimulationConfig,
    centered_time_grid,
)

__all__ = [
    "NloExampleRunner",
    "SimulationOptions",
    "TemporalSimulationConfig",
    "centered_time_grid",
    "plot_3d_intensity_scatter_propagation",
    "plot_intensity_colormap_vs_propagation",
    "plot_final_re_im_comparison",
    "plot_final_intensity_comparison",
    "plot_total_error_over_propagation",
]
