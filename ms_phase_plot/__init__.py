"""Adobe Audition-style Mid/Side Phase Analysis plotter and interactive scope."""

from .dsp import (
    ChartType,
    ColorMode,
    PhaseMetrics,
    ScaleMode,
    apply_scale_points,
    compute_mid_side,
    compute_phase_metrics,
    compute_phase_wheel,
    compute_polar_histogram,
    extract_stereo_pair,
    generate_test_signal,
    load_audio,
    load_raw_channels,
    splat_phosphor_density,
)
from .static_plot import plot_mid_side, plot_phase_analysis

__all__ = [
    "ChartType",
    "ColorMode",
    "PhaseMetrics",
    "ScaleMode",
    "apply_scale_points",
    "compute_mid_side",
    "compute_phase_metrics",
    "compute_phase_wheel",
    "compute_polar_histogram",
    "extract_stereo_pair",
    "generate_test_signal",
    "load_audio",
    "load_raw_channels",
    "plot_mid_side",
    "plot_phase_analysis",
    "splat_phosphor_density",
]
