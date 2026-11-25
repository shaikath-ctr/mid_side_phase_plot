"""Command-line interface for launching the GUI or exporting headless phase plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .dsp import extract_stereo_pair, generate_test_signal, load_raw_channels
from .static_plot import plot_phase_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ms-phase-plot",
        description="Adobe Audition-style Mid/Side Phase Analysis Plotter & Interactive Scope",
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        default=None,
        help="Optional audio file (.wav, .flac, .aiff, .ogg, .mp3, .m4a) to analyze",
    )
    parser.add_argument(
        "--preset",
        default="stereo_mix",
        choices=[
            "stereo_mix",
            "mono",
            "out_of_phase",
            "hard_left",
            "hard_right",
            "quadrature",
            "delayed_comb",
        ],
        help="Built-in stereo test signal preset when no audio_file is provided",
    )
    parser.add_argument(
        "--chart",
        default="mid_side",
        choices=["mid_side", "lissajous", "histogram", "phase_wheel"],
        help="Phase analysis chart type (default: mid_side)",
    )
    parser.add_argument(
        "--scale",
        default="linear",
        choices=["linear", "log_zoom", "log_full"],
        help="Display scaling mode (default: linear)",
    )
    parser.add_argument(
        "--color",
        default="phosphor",
        choices=["phosphor", "phase_zones"],
        help="Color palette mode (default: phosphor)",
    )
    parser.add_argument(
        "--left-ch",
        type=int,
        default=0,
        help="0-based channel index for Left (default: 0)",
    )
    parser.add_argument(
        "--right-ch",
        type=int,
        default=1,
        help="0-based channel index for Right (default: 1)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start offset in seconds for headless export (default: 0.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds to slice for headless export",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable automatic display normalization (show true 0 dBFS scale)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Export plot headlessly to PNG/SVG/PDF instead of opening GUI",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI when using --save (default: 200)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.save is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.audio_file is not None:
            raw, sr = load_raw_channels(args.audio_file)
            stereo = extract_stereo_pair(raw, left_ch=args.left_ch, right_ch=args.right_ch)
            title = f"Phase Analysis — {args.audio_file.name}"
        else:
            stereo, sr = generate_test_signal(preset=args.preset, duration_s=4.0, sample_rate=48000)
            title = f"Phase Analysis — Test Signal ({args.preset})"

        start_idx = max(0, int(round(args.start * sr)))
        if args.duration is not None:
            end_idx = start_idx + max(1, int(round(args.duration * sr)))
            stereo = stereo[start_idx:end_idx]
        elif start_idx > 0:
            stereo = stereo[start_idx:]

        fig, _ = plot_phase_analysis(
            stereo,
            sample_rate=sr,
            chart_type=args.chart,
            scale_mode=args.scale,
            normalize=not args.no_normalize,
            color_mode=args.color,
            title=title,
        )
        out = args.save.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved phase analysis plot to {out}")
        return 0

    from .gui import launch_gui

    return launch_gui(initial_file=args.audio_file, preset=args.preset)
