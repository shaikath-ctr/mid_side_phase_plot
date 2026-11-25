"""High-resolution Matplotlib renderer styled after Adobe Audition's Phase Analysis panel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np

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
    load_audio,
    splat_phosphor_density,
)

# Adobe Audition-inspired dark studio palette
BG_OUTER = "#14171D"
BG_PLOT = "#090C10"
GRID_MAIN = "#262E3A"
GRID_SUB = "#1B212A"
TEXT_MUTED = "#8894A6"
TEXT_BRIGHT = "#D9E2EC"
ACCENT_PHOSPHOR = "#26F7B4"
ACCENT_CORAL = "#FF5A6E"


def _density_to_rgba(
    intensity: np.ndarray,
    anti_ratio: np.ndarray,
    color_mode: ColorMode = "phosphor",
) -> np.ndarray:
    """Convert [H, W] intensity and anti-phase ratio maps into an RGBA uint8 image."""
    h, w = intensity.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    i = np.clip(intensity, 0.0, 1.0)
    # Two-stage tone curve for dark-teal halo + glowing mint-white core
    core = np.power(i, 2.1)
    glow = np.power(i, 0.85)

    if color_mode == "phase_zones":
        # Blend in-phase (cyan-emerald) vs out-of-phase (amber-coral)
        anti = np.clip(anti_ratio, 0.0, 1.0)
        in_p = 1.0 - anti
        r = glow * (0.04 * in_p + 1.00 * anti) + core * (0.65 * in_p + 0.35 * anti)
        g = glow * (0.92 * in_p + 0.34 * anti) + core * (0.35 * in_p + 0.30 * anti)
        b = glow * (0.72 * in_p + 0.38 * anti) + core * (0.30 * in_p + 0.25 * anti)
    else:
        # Classic Audition phosphor cyan-green
        r = glow * 0.05 + core * 0.78
        g = glow * 0.92 + core * 0.14
        b = glow * 0.70 + core * 0.30

    rgba[..., 0] = np.clip(r, 0.0, 1.0)
    rgba[..., 1] = np.clip(g, 0.0, 1.0)
    rgba[..., 2] = np.clip(b, 0.0, 1.0)
    rgba[..., 3] = np.clip(np.power(i, 0.72), 0.0, 1.0)
    return (rgba * 255.0).astype(np.uint8)


def _draw_mid_side_grid(ax: Axes, scale_mode: ScaleMode, normalize: bool) -> None:
    """Draw Adobe Audition Mid/Side axes, L/R diagonals, and dB diamond/circle graticule."""
    # Concentric reference diamonds (0 dB, -6 dB, -12 dB)
    for lvl, ls, col in [(1.0, "-", GRID_MAIN), (0.5, "--", GRID_SUB), (0.25, ":", GRID_SUB)]:
        diamond = Polygon(
            [(0.0, lvl), (lvl, 0.0), (0.0, -lvl), (-lvl, 0.0)],
            closed=True,
            fill=False,
            edgecolor=col,
            linewidth=1.0 if lvl == 1.0 else 0.85,
            linestyle=ls,
            zorder=1,
        )
        ax.add_patch(diamond)

    # Subtle outer boundary circle/square ticks
    ax.axhline(0.0, color=GRID_MAIN, linewidth=1.0, zorder=1)
    ax.axvline(0.0, color=GRID_MAIN, linewidth=1.0, zorder=1)

    # Diagonal L (-45 deg) and R (+45 deg) axes
    d = 0.76
    ax.plot([-d, d], [d, -d], color=GRID_MAIN, linewidth=0.95, linestyle="-.", zorder=1)
    ax.plot([-d, d], [-d, d], color=GRID_MAIN, linewidth=0.95, linestyle="-.", zorder=1)

    # Axis labels matching Adobe Audition
    pad = 1.045
    ax.text(0.0, pad, "+M", color=TEXT_BRIGHT, fontsize=9.5, fontweight="bold", ha="center", va="bottom")
    ax.text(0.0, -pad, "-M", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="center", va="top")
    ax.text(-pad, 0.0, "-S", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="right", va="center")
    ax.text(pad, 0.0, "+S", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="left", va="center")

    ax.text(-0.60, 0.60, "L", color=ACCENT_PHOSPHOR, alpha=0.8, fontsize=9.5, fontweight="bold", ha="right", va="bottom")
    ax.text(0.60, 0.60, "R", color=ACCENT_PHOSPHOR, alpha=0.8, fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    ax.text(-0.60, -0.60, "-R", color=TEXT_MUTED, alpha=0.65, fontsize=8.0, ha="right", va="top")
    ax.text(0.60, -0.60, "-L", color=TEXT_MUTED, alpha=0.65, fontsize=8.0, ha="left", va="top")

    if scale_mode == "linear":
        tag = "Normalized" if normalize else "0 dBFS"
        ax.text(0.52, -0.49, tag, color=TEXT_MUTED, fontsize=7.5, alpha=0.85)
    else:
        floor_txt = "-36 dB" if scale_mode == "log_zoom" else "-72 dB"
        ax.text(0.52, -0.49, f"Log ({floor_txt})", color=TEXT_MUTED, fontsize=7.5, alpha=0.85)


def _draw_lissajous_grid(ax: Axes) -> None:
    """Draw standard L/R Lissajous XY scope grid with M/S diagonals."""
    rect = Rectangle((-1.0, -1.0), 2.0, 2.0, fill=False, edgecolor=GRID_MAIN, linewidth=1.0, zorder=1)
    ax.add_patch(rect)
    for lvl in (0.5, 0.25):
        ax.add_patch(
            Rectangle(
                (-lvl, -lvl),
                2.0 * lvl,
                2.0 * lvl,
                fill=False,
                edgecolor=GRID_SUB,
                linewidth=0.8,
                linestyle=":",
                zorder=1,
            )
        )
    ax.axhline(0.0, color=GRID_MAIN, linewidth=0.95, zorder=1)
    ax.axvline(0.0, color=GRID_MAIN, linewidth=0.95, zorder=1)
    ax.plot([-1.0, 1.0], [-1.0, 1.0], color=GRID_MAIN, linewidth=0.95, linestyle="-.", zorder=1)
    ax.plot([-1.0, 1.0], [1.0, -1.0], color=GRID_MAIN, linewidth=0.95, linestyle="-.", zorder=1)

    ax.text(1.04, 0.0, "+L", color=TEXT_BRIGHT, fontsize=9.0, fontweight="bold", ha="left", va="center")
    ax.text(-1.04, 0.0, "-L", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="right", va="center")
    ax.text(0.0, 1.04, "+R", color=TEXT_BRIGHT, fontsize=9.0, fontweight="bold", ha="center", va="bottom")
    ax.text(0.0, -1.04, "-R", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="center", va="top")
    ax.text(0.92, 0.92, "M", color=ACCENT_PHOSPHOR, alpha=0.85, fontsize=9.0, fontweight="bold", ha="right", va="top")
    ax.text(-0.92, 0.92, "S", color=TEXT_MUTED, alpha=0.85, fontsize=9.0, fontweight="bold", ha="left", va="top")


def _draw_polar_grid(ax: Axes, rings: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0), labels: Optional[Tuple[str, ...]] = None) -> None:
    """Draw circular polar graticule for Histogram and Phase Wheel modes."""
    for idx, r in enumerate(rings):
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                r,
                fill=False,
                edgecolor=GRID_MAIN if r == 1.0 else GRID_SUB,
                linewidth=1.0 if r == 1.0 else 0.8,
                linestyle="-" if r == 1.0 else ":",
                zorder=1,
            )
        )
        if labels and idx < len(labels):
            ax.text(
                0.03,
                r - 0.03,
                labels[idx],
                color=TEXT_MUTED,
                fontsize=7.5,
                ha="left",
                va="top",
                alpha=0.92,
                bbox=dict(boxstyle="round,pad=0.14", facecolor=BG_PLOT, edgecolor="none", alpha=0.72),
                zorder=5,
            )

    ax.axhline(0.0, color=GRID_MAIN, linewidth=0.9, zorder=1)
    ax.axvline(0.0, color=GRID_MAIN, linewidth=0.9, zorder=1)
    d = 1.0 / np.sqrt(2.0)
    ax.plot([-d, d], [d, -d], color=GRID_SUB, linewidth=0.85, linestyle="-.", zorder=1)
    ax.plot([-d, d], [-d, d], color=GRID_SUB, linewidth=0.85, linestyle="-.", zorder=1)


def _draw_correlation_bar(ax_corr: Axes, metrics: PhaseMetrics) -> None:
    """Render Adobe Audition-style -1 to +1 Phase Correlation bar below the main plot."""
    ax_corr.set_facecolor(BG_PLOT)
    ax_corr.set_xlim(-1.05, 1.05)
    ax_corr.set_ylim(0.0, 1.0)
    ax_corr.set_yticks([])
    ax_corr.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_corr.set_xticklabels(["-1.0", "-0.5", "0.0", "+0.5", "+1.0"], color=TEXT_MUTED, fontsize=8.5)
    ax_corr.tick_params(axis="x", colors=TEXT_MUTED, length=3, pad=2)
    for spine in ax_corr.spines.values():
        spine.set_color(GRID_MAIN)

    # Background color gradient (-1 coral -> 0 amber -> +1 cyan-green)
    grad = np.linspace(-1.0, 1.0, 256)
    rgb = np.zeros((1, 256, 3), dtype=np.float32)
    for i, val in enumerate(grad):
        if val < 0.0:
            t = 1.0 + val  # 0 at -1, 1 at 0
            rgb[0, i] = [0.75 * (1.0 - t) + 0.85 * t, 0.16 * (1.0 - t) + 0.55 * t, 0.22 * (1.0 - t) + 0.12 * t]
        else:
            t = val  # 0 at 0, 1 at +1
            rgb[0, i] = [0.85 * (1.0 - t) + 0.08 * t, 0.55 * (1.0 - t) + 0.85 * t, 0.12 * (1.0 - t) + 0.62 * t]

    ax_corr.imshow(rgb, extent=(-1.0, 1.0, 0.18, 0.55), aspect="auto", alpha=0.32, zorder=1)
    ax_corr.axvline(0.0, color=TEXT_MUTED, linewidth=0.9, linestyle="--", alpha=0.65, zorder=2)

    corr = float(np.clip(metrics.correlation, -1.0, 1.0))
    fill_color = ACCENT_CORAL if corr < 0.0 else ("#F5B841" if corr < 0.35 else ACCENT_PHOSPHOR)
    if corr >= 0.0:
        ax_corr.add_patch(Rectangle((0.0, 0.22), corr, 0.29, facecolor=fill_color, alpha=0.85, zorder=3))
    else:
        ax_corr.add_patch(Rectangle((corr, 0.22), -corr, 0.29, facecolor=fill_color, alpha=0.85, zorder=3))

    ax_corr.axvline(corr, color="#FFFFFF", linewidth=2.0, zorder=4)
    ax_corr.text(
        -1.0,
        0.78,
        "PHASE CORRELATION",
        color=TEXT_MUTED,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax_corr.text(
        1.0,
        0.78,
        f"r = {corr:+.3f}   |   Width: {metrics.stereo_width_pct:.1f}%   |   In-Phase: {metrics.in_phase_pct:.1f}%",
        color=TEXT_BRIGHT,
        fontsize=8.5,
        fontfamily="monospace",
        ha="right",
        va="center",
    )


def plot_phase_analysis(
    stereo_lr: np.ndarray,
    sample_rate: int = 48000,
    chart_type: ChartType = "mid_side",
    scale_mode: ScaleMode = "linear",
    normalize: bool = True,
    color_mode: ColorMode = "phosphor",
    connect_lines: bool = True,
    title: Optional[str] = None,
    show_correlation: bool = True,
    show_metrics_box: bool = True,
    figsize: Tuple[float, float] = (7.2, 8.2),
    resolution: int = 640,
) -> Tuple[Figure, Axes]:
    """Render an Adobe Audition-style Phase Analysis plot using Matplotlib.

    Returns (fig, ax) ready for display or `fig.savefig(...)`.
    """
    lr = np.asarray(stereo_lr, dtype=np.float32)
    if lr.ndim == 1:
        lr = np.column_stack([lr, lr])
    elif lr.ndim != 2 or lr.shape[1] < 2:
        raise ValueError(f"Expected [N, 2] stereo array, got shape {lr.shape}")

    metrics = compute_phase_metrics(lr)

    fig = plt.figure(figsize=figsize, facecolor=BG_OUTER)
    if show_correlation:
        gs = fig.add_gridspec(2, 1, height_ratios=[8.4, 1.35], hspace=0.14)
        ax = fig.add_subplot(gs[0])
        ax_corr = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(111)
        ax_corr = None

    ax.set_facecolor(BG_PLOT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.14, 1.14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID_MAIN)
        spine.set_linewidth(1.1)

    if chart_type in ("mid_side", "lissajous"):
        if chart_type == "mid_side":
            mid, side = compute_mid_side(lr, orthonormal=False)
            x_raw, y_raw = side, mid
            _draw_mid_side_grid(ax, scale_mode=scale_mode, normalize=normalize)
            default_title = "Phase Analysis — Mid / Side"
        else:
            x_raw, y_raw = lr[:, 0], lr[:, 1]
            _draw_lissajous_grid(ax)
            default_title = "Phase Analysis — Lissajous (L / R)"

        x_pts, y_pts, _ = apply_scale_points(x_raw, y_raw, normalize=normalize, scale_mode=scale_mode)
        intensity, anti = splat_phosphor_density(
            x_pts,
            y_pts,
            size=resolution,
            connect_lines=connect_lines,
        )
        # Note: anti-phase mask is physically |S| > |M|; in Lissajous that corresponds to L*R < 0
        if chart_type == "lissajous" and x_pts.size > 0:
            # Re-evaluate L*R < 0 color mask on canvas by using M/S anti-ratio? Simpler: in Lissajous,
            # x*y < 0 means 2nd & 4th quadrants are out-of-phase. Let's override anti_ratio geometrically:
            yy, xx = np.mgrid[1.0 : -1.0 : complex(0, resolution), -1.0 : 1.0 : complex(0, resolution)]
            anti = ((xx * yy) < 0.0).astype(np.float32)

        rgba = _density_to_rgba(intensity, anti, color_mode=color_mode)
        ax.imshow(rgba, extent=(-1.0, 1.0, -1.0, 1.0), origin="upper", interpolation="bilinear", zorder=3)

    elif chart_type == "histogram":
        hist_scale: ScaleMode = scale_mode
        angles, radii, mean_angle, mean_r = compute_polar_histogram(
            lr, num_bins=180, scale_mode=hist_scale, normalize=normalize
        )
        ring_labels = ("-45 dB", "-30 dB", "-15 dB", "0 dB") if hist_scale == "log_full" else (
            ("-22 dB", "-15 dB", "-7.5 dB", "0 dB") if hist_scale == "log_zoom" else ("0.25", "0.50", "0.75", "1.0")
        )
        _draw_polar_grid(ax, labels=ring_labels)
        default_title = f"Phase Analysis — Histogram ({hist_scale.replace('_', ' ').title()})"

        ax.text(0.0, 1.045, "+M", color=TEXT_BRIGHT, fontsize=9.5, fontweight="bold", ha="center", va="bottom")
        ax.text(0.0, -1.045, "-M", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="center", va="top")
        ax.text(-1.045, 0.0, "-S", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="right", va="center")
        ax.text(1.045, 0.0, "+S", color=TEXT_MUTED, fontsize=9.0, fontweight="bold", ha="left", va="center")
        ax.text(-0.75, 0.75, "L", color=ACCENT_PHOSPHOR, fontsize=9.0, fontweight="bold", ha="right", va="bottom")
        ax.text(0.75, 0.75, "R", color=ACCENT_PHOSPHOR, fontsize=9.0, fontweight="bold", ha="left", va="bottom")

        # Close loop for smooth polygon
        ang_c = np.concatenate([angles, angles[:1]])
        rad_c = np.concatenate([radii, radii[:1]])
        px = rad_c * np.sin(ang_c)
        py = rad_c * np.cos(ang_c)
        ax.fill(px, py, facecolor=ACCENT_PHOSPHOR, alpha=0.22, zorder=3)
        ax.plot(px, py, color=ACCENT_PHOSPHOR, linewidth=1.8, zorder=4)

        # Tracking ball indicator (Audition style)
        bx = mean_r * np.sin(mean_angle)
        by = mean_r * np.cos(mean_angle)
        ax.plot([0.0, bx], [0.0, by], color="#FFFFFF", alpha=0.45, linewidth=1.0, zorder=4)
        ax.scatter([bx], [by], s=56, color="#FFFFFF", edgecolors=ACCENT_PHOSPHOR, linewidths=1.8, zorder=5)

    elif chart_type == "phase_wheel":
        xw, yw, freqs, mags = compute_phase_wheel(lr, sample_rate=sample_rate)
        _draw_polar_grid(ax, rings=(0.25, 0.5, 0.75, 1.0), labels=("110 Hz", "630 Hz", "3.5 kHz", "20 kHz"))
        default_title = "Phase Analysis — Phase Wheel"

        ax.text(0.0, 1.045, "0° (In-Phase)", color=TEXT_BRIGHT, fontsize=8.5, fontweight="bold", ha="center", va="bottom")
        ax.text(0.0, -1.045, "180° (Inverted)", color=ACCENT_CORAL, fontsize=8.5, fontweight="bold", ha="center", va="top")
        ax.text(-1.045, 0.0, "-90°", color=TEXT_MUTED, fontsize=8.5, fontweight="bold", ha="right", va="center")
        ax.text(1.045, 0.0, "+90°", color=TEXT_MUTED, fontsize=8.5, fontweight="bold", ha="left", va="center")

        if xw.size > 0:
            # Color by phase coherence: top half green/cyan, bottom half coral
            phase_cos = yw / np.maximum(np.hypot(xw, yw), 1e-6)
            colors = np.zeros((xw.size, 4), dtype=np.float32)
            in_p = np.clip((phase_cos + 1.0) * 0.5, 0.0, 1.0)
            colors[:, 0] = 1.0 * (1.0 - in_p) + 0.15 * in_p
            colors[:, 1] = 0.35 * (1.0 - in_p) + 0.96 * in_p
            colors[:, 2] = 0.42 * (1.0 - in_p) + 0.72 * in_p
            colors[:, 3] = np.clip(0.25 + 0.75 * mags, 0.15, 0.95)
            ax.scatter(xw, yw, s=6.0 + 26.0 * (mags**1.5), c=colors, edgecolors="none", zorder=4)
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type!r}")

    ax.set_title(title or default_title, color=TEXT_BRIGHT, fontsize=11.5, fontweight="bold", pad=10)

    if show_metrics_box:
        info_txt = (
            f"M: {metrics.mid_rms_db:+.1f} dB (pk {metrics.mid_peak_db:+.1f})\n"
            f"S: {metrics.side_rms_db:+.1f} dB (pk {metrics.side_peak_db:+.1f})\n"
            f"L/R Bal: {metrics.balance_db:+.1f} dB"
        )
        ax.text(
            -1.08,
            1.08,
            info_txt,
            color=TEXT_MUTED,
            fontsize=7.6,
            fontfamily="monospace",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#10141B", edgecolor=GRID_MAIN, alpha=0.85),
            zorder=6,
        )

    if ax_corr is not None:
        _draw_correlation_bar(ax_corr, metrics)

    return fig, ax


def plot_mid_side(
    audio: np.ndarray | str | Path,
    sample_rate: int = 48000,
    left_ch: int = 0,
    right_ch: int = 1,
    normalize: bool = True,
    scale_mode: ScaleMode = "linear",
    color_mode: ColorMode = "phosphor",
    save_path: Optional[str | Path] = None,
    dpi: int = 180,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Convenience wrapper accepting either a NumPy [N, 2] array or an audio file path."""
    if isinstance(audio, (str, Path)):
        stereo, sample_rate, _ = load_audio(audio, left_ch=left_ch, right_ch=right_ch)
        if "title" not in kwargs:
            kwargs["title"] = f"Mid / Side Phase Analysis — {Path(audio).name}"
    else:
        stereo = np.asarray(audio, dtype=np.float32)

    fig, ax = plot_phase_analysis(
        stereo,
        sample_rate=sample_rate,
        chart_type="mid_side",
        scale_mode=scale_mode,
        normalize=normalize,
        color_mode=color_mode,
        **kwargs,
    )
    if save_path is not None:
        out = Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig, ax
