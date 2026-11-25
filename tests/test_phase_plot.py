"""Automated test suite for ms_phase_plot DSP, static rendering, CLI, and GUI."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf

from ms_phase_plot import (
    apply_scale_points,
    compute_mid_side,
    compute_phase_metrics,
    compute_phase_wheel,
    compute_polar_histogram,
    generate_test_signal,
    load_audio,
    plot_mid_side,
    plot_phase_analysis,
    splat_phosphor_density,
)
from ms_phase_plot.cli import main as cli_main


def test_mid_side_geometry_identities() -> None:
    n = 1024
    sig = np.sin(np.linspace(0.0, 12.0 * np.pi, n, dtype=np.float32))

    # Pure mono: S == 0, M == sig
    mono = np.column_stack([sig, sig])
    m, s = compute_mid_side(mono)
    assert np.allclose(s, 0.0, atol=1e-6)
    assert np.allclose(m, sig, atol=1e-6)

    # Inverted mono: M == 0, S == -sig
    inv = np.column_stack([sig, -sig])
    m, s = compute_mid_side(inv)
    assert np.allclose(m, 0.0, atol=1e-6)
    assert np.allclose(s, -sig, atol=1e-6)

    # Hard Left (+1, 0) -> S = -0.5, M = +0.5 (top-left L diagonal)
    left_only = np.array([[1.0, 0.0], [-0.8, 0.0]], dtype=np.float32)
    m, s = compute_mid_side(left_only)
    assert np.allclose(m, [0.5, -0.4], atol=1e-6)
    assert np.allclose(s, [-0.5, 0.4], atol=1e-6)

    # Hard Right (0, +1) -> S = +0.5, M = +0.5 (top-right R diagonal)
    right_only = np.array([[0.0, 1.0], [0.0, -0.8]], dtype=np.float32)
    m, s = compute_mid_side(right_only)
    assert np.allclose(m, [0.5, -0.4], atol=1e-6)
    assert np.allclose(s, [0.5, -0.4], atol=1e-6)

    # Orthonormal transform preserves instantaneous energy
    stereo, _ = generate_test_signal("stereo_mix", duration_s=0.25)
    m_o, s_o = compute_mid_side(stereo, orthonormal=True)
    e_lr = np.sum(stereo[:, 0] ** 2 + stereo[:, 1] ** 2)
    e_ms = np.sum(m_o**2 + s_o**2)
    assert np.isclose(e_ms, e_lr, rtol=1e-5)


def test_phase_metrics_cases() -> None:
    mono, _ = generate_test_signal("mono", duration_s=0.5)
    mm = compute_phase_metrics(mono)
    assert mm.correlation == pytest.approx(1.0, abs=1e-4)
    assert mm.stereo_width_pct == pytest.approx(0.0, abs=1e-3)
    assert mm.in_phase_pct == pytest.approx(100.0, abs=1e-3)
    assert mm.side_rms_db <= -90.0

    oop, _ = generate_test_signal("out_of_phase", duration_s=0.5)
    mo = compute_phase_metrics(oop)
    assert mo.correlation == pytest.approx(-1.0, abs=1e-4)
    assert mo.stereo_width_pct == pytest.approx(100.0, abs=1e-3)
    assert mo.in_phase_pct == pytest.approx(0.0, abs=1e-3)
    assert mo.mid_rms_db <= -90.0

    quad, _ = generate_test_signal("quadrature", duration_s=1.0)
    mq = compute_phase_metrics(quad)
    assert abs(mq.correlation) < 0.05
    assert 45.0 < mq.stereo_width_pct < 55.0

    hl, _ = generate_test_signal("hard_left", duration_s=0.5)
    ml = compute_phase_metrics(hl)
    assert ml.balance_db < -40.0

    hr, _ = generate_test_signal("hard_right", duration_s=0.5)
    mr = compute_phase_metrics(hr)
    assert mr.balance_db > 40.0

    # Silence safety
    silent = np.zeros((512, 2), dtype=np.float32)
    ms = compute_phase_metrics(silent)
    assert np.isfinite(ms.correlation)
    assert ms.mid_rms_db == -96.0


def test_phosphor_splat_orientation() -> None:
    # Quiet mono signal normalized should sit strictly on vertical center column (x = 0)
    t = np.linspace(0, 4 * np.pi, 1024, dtype=np.float32)
    sig = 0.05 * np.sin(t)
    mono = np.column_stack([sig, sig])
    m, s = compute_mid_side(mono)
    x, y, gain = apply_scale_points(s, m, normalize=True, scale_mode="linear")
    assert gain > 10.0
    assert np.max(np.abs(y)) == pytest.approx(0.92, abs=1e-3)

    intensity, anti = splat_phosphor_density(x, y, size=129, bloom_sigma=0.0)
    mid_col = 64
    assert float(np.sum(intensity[:, mid_col])) > 0.0
    assert float(np.sum(intensity[:, : mid_col - 1])) == 0.0
    assert float(np.sum(intensity[:, mid_col + 2 :])) == 0.0
    assert float(np.max(anti)) == 0.0


@pytest.mark.parametrize("scale_mode", ["linear", "log_zoom", "log_full"])
def test_polar_histogram_peaks(scale_mode: str) -> None:
    mono, _ = generate_test_signal("mono", duration_s=0.5)
    angles, radii, mean_ang, mean_r = compute_polar_histogram(mono, scale_mode=scale_mode)
    assert np.all(np.isfinite(radii))
    assert 0.0 <= float(np.max(radii)) <= 1.0 + 1e-5
    # Energy weighted ball for mono should point near 0 radians (+M)
    assert abs(mean_ang) < 0.08
    assert mean_r > 0.9

    hard_r, _ = generate_test_signal("hard_right", duration_s=0.5)
    _, _, r_ang, _ = compute_polar_histogram(hard_r, scale_mode=scale_mode)
    assert r_ang == pytest.approx(np.pi / 4.0, abs=0.08)


def test_phase_wheel_delay_detection() -> None:
    sr = 48000
    n = 16384
    rng = np.random.default_rng(7)
    base = rng.standard_normal(n).astype(np.float32) * 0.4
    # Delay right channel by 4 samples -> at 3 kHz, phase lag = -2*pi*3000*(4/48000) = -pi/2
    right = np.roll(base, 4)
    right[:4] = 0.0
    lr = np.column_stack([base, right])
    xw, yw, freqs, mags = compute_phase_wheel(lr, sample_rate=sr, n_fft=4096)
    assert freqs.size > 10
    # Near 3000 Hz, dphi = atan2(x, y) should be around -pi/2 (or +pi/2 depending on sign of R*conj(L))
    # Since R is delayed: R(w) = e^{-jw tau} L(w) -> angle(R * conj(L)) = -w * tau
    idx_3k = int(np.argmin(np.abs(freqs - 3000.0)))
    measured_dphi = np.arctan2(xw[idx_3k], yw[idx_3k])
    expected_dphi = -2.0 * np.pi * freqs[idx_3k] * (4.0 / sr)
    assert measured_dphi == pytest.approx(expected_dphi, abs=0.15)


def test_multichannel_audio_load_and_static_plots(tmp_path: Path) -> None:
    sr = 44100
    n = 4410
    t = np.arange(n, dtype=np.float32) / sr
    ch0 = 0.5 * np.sin(2 * np.pi * 200 * t)
    ch1 = 0.5 * np.sin(2 * np.pi * 200 * t)
    ch2 = -ch0
    ch3 = np.zeros_like(ch0)
    multi = np.column_stack([ch0, ch1, ch2, ch3])

    wav_path = tmp_path / "quad_test.wav"
    sf.write(str(wav_path), multi, sr)

    pair_01, loaded_sr, total_ch = load_audio(wav_path, left_ch=0, right_ch=1)
    assert loaded_sr == sr
    assert total_ch == 4
    assert compute_phase_metrics(pair_01).correlation == pytest.approx(1.0, abs=1e-3)

    pair_02, _, _ = load_audio(wav_path, left_ch=0, right_ch=2)
    assert compute_phase_metrics(pair_02).correlation == pytest.approx(-1.0, abs=1e-3)

    for chart in ("mid_side", "lissajous", "histogram", "phase_wheel"):
        fig, _ = plot_phase_analysis(pair_01, sample_rate=sr, chart_type=chart, resolution=256)
        out = tmp_path / f"{chart}.png"
        fig.savefig(str(out), dpi=100)
        plt.close(fig)
        assert out.exists() and out.stat().st_size > 2000

    # Test plot_mid_side directly from file path
    out_ms = tmp_path / "from_file.png"
    fig2, _ = plot_mid_side(wav_path, save_path=out_ms, dpi=100, resolution=256)
    plt.close(fig2)
    assert out_ms.exists() and out_ms.stat().st_size > 2000


def test_cli_headless_save(tmp_path: Path) -> None:
    out_png = tmp_path / "cli_out.png"
    rc = cli_main(["--preset", "quadrature", "--chart", "mid_side", "--color", "phase_zones", "--save", str(out_png)])
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 2000


def test_gui_offscreen_smoke(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from ms_phase_plot.gui import PhaseAnalysisWindow

    app = QApplication.instance() or QApplication([])
    win = PhaseAnalysisWindow(initial_preset="stereo_mix")
    win.show()
    app.processEvents()

    # Exercise all chart modes and polarity invert
    for idx in range(win._combo_chart.count()):
        win._combo_chart.setCurrentIndex(idx)
        app.processEvents()
        img = win.grab()
        assert img.width() > 100 and img.height() > 100

    win._combo_chart.setCurrentIndex(0)
    win._btn_invert_r.setChecked(True)
    app.processEvents()

    win.load_preset("mono")
    win._btn_invert_r.setChecked(False)
    app.processEvents()
    assert win._corr_meter._corr == pytest.approx(1.0, abs=1e-2)

    win.close()
    win.deleteLater()
    app.processEvents()
