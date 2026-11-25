"""Core DSP, Mid/Side phase math, phosphor density rendering, and audio I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

ChartType = Literal["mid_side", "lissajous", "histogram", "phase_wheel"]
ScaleMode = Literal["linear", "log_zoom", "log_full"]
ColorMode = Literal["phosphor", "phase_zones"]


@dataclass(frozen=True)
class PhaseMetrics:
    """Summary phase and stereo-field telemetry for a stereo audio segment."""

    correlation: float
    mid_rms_db: float
    side_rms_db: float
    left_rms_db: float
    right_rms_db: float
    mid_peak_db: float
    side_peak_db: float
    left_peak_db: float
    right_peak_db: float
    stereo_width_pct: float
    in_phase_pct: float
    balance_db: float
    sample_count: int


def _amp_to_db(val: float, floor_db: float = -96.0) -> float:
    """Convert linear amplitude to dBFS with a finite floor."""
    if val <= 0.0 or not np.isfinite(val):
        return floor_db
    db = 20.0 * np.log10(val)
    return float(max(db, floor_db))


def _decode_audio_array(file_path: Path) -> Tuple[np.ndarray, int]:
    """Decode any supported audio file into float32 [N, C] and sample rate."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    data: Optional[np.ndarray] = None
    sr: int = 48000
    sf_err: Optional[Exception] = None

    try:
        import soundfile as sf

        data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    except Exception as err:
        sf_err = err

    if data is None and file_path.suffix.lower() == ".wav":
        try:
            from scipy.io import wavfile

            sr, raw = wavfile.read(str(file_path))
            data = _normalize_pcm_array(raw)
        except Exception:
            data = None

    if data is None:
        try:
            data, sr = _load_via_transcoder(file_path)
        except Exception as trans_err:
            raise RuntimeError(
                f"Could not decode audio file '{file_path.name}': {sf_err} / {trans_err}"
            ) from sf_err

    if data.ndim == 1:
        data = data[:, np.newaxis]

    return np.nan_to_num(data.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-1.0), int(sr)


def load_audio(
    path: str | Path,
    left_ch: int = 0,
    right_ch: int = 1,
) -> Tuple[np.ndarray, int, int]:
    """Load an audio file and return (stereo_lr [N, 2] float32, sample_rate, total_channels).

    Mono files are duplicated to [L, L] so all phase diagnostics work cleanly.
    Supports WAV/FLAC/AIFF/OGG/MP3 via soundfile, plus ffmpeg (cross-platform) or afconvert (macOS).
    """
    file_path = Path(path).expanduser().resolve()
    data, sr = _decode_audio_array(file_path)

    total_channels = int(data.shape[1])
    if total_channels == 1:
        stereo = np.column_stack([data[:, 0], data[:, 0]]).astype(np.float32, copy=False)
    else:
        li = int(np.clip(left_ch, 0, total_channels - 1))
        ri = int(np.clip(right_ch, 0, total_channels - 1))
        stereo = np.column_stack([data[:, li], data[:, ri]]).astype(np.float32, copy=False)

    return stereo, int(sr), total_channels


def _normalize_pcm_array(raw: np.ndarray) -> np.ndarray:
    """Convert integer or float PCM array into [-1, 1] float32 [N, C]."""
    arr = np.asarray(raw)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        return arr.astype(np.float32) / scale
    return arr.astype(np.float32)


def _load_via_transcoder(file_path: Path) -> Tuple[np.ndarray, int]:
    """Decode compressed audio using ffmpeg (cross-platform) or afconvert (macOS)."""
    import soundfile as sf

    ffmpeg_bin = shutil.which("ffmpeg")
    afconvert_bin = shutil.which("afconvert")
    if not ffmpeg_bin and not afconvert_bin:
        raise RuntimeError("Neither ffmpeg nor macOS afconvert is available for compressed audio fallback")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "decoded.wav"
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-i",
                str(file_path),
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                str(tmp_wav),
            ]
        else:
            cmd = [
                afconvert_bin,
                "-f",
                "WAVE",
                "-d",
                "LEF32",
                str(file_path),
                str(tmp_wav),
            ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
        return data, int(sr)


def load_raw_channels(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load all channels of an audio file as float32 [N, C] along with sample rate."""
    file_path = Path(path).expanduser().resolve()
    return _decode_audio_array(file_path)


def extract_stereo_pair(data: np.ndarray, left_ch: int = 0, right_ch: int = 1) -> np.ndarray:
    """Select [L, R] stereo float32 [N, 2] from an [N] or [N, C] array."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return np.column_stack([arr, arr])
    if arr.shape[1] == 1:
        return np.column_stack([arr[:, 0], arr[:, 0]])
    c = arr.shape[1]
    li = int(np.clip(left_ch, 0, c - 1))
    ri = int(np.clip(right_ch, 0, c - 1))
    return np.column_stack([arr[:, li], arr[:, ri]])


def compute_mid_side(
    stereo_lr: np.ndarray,
    orthonormal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert [N, 2] (L, R) audio into (Mid, Side).

    With orthonormal=False (default):
        M = (L + R) / 2,  S = (R - L) / 2
    so full-scale [-1, 1] x [-1, 1] maps to the unit diamond |M| + |S| <= 1,
    placing Left-only (+L) on the -45 deg top-left diagonal (S < 0, M > 0) and
    Right-only (+R) on the +45 deg top-right diagonal (S > 0, M > 0).

    With orthonormal=True:
        M = (L + R) / sqrt(2),  S = (R - L) / sqrt(2)
    which preserves total Euclidean signal power M^2 + S^2 = L^2 + R^2.
    """
    lr = np.asarray(stereo_lr, dtype=np.float32)
    if lr.ndim != 2 or lr.shape[1] < 2:
        raise ValueError(f"Expected [N, 2] stereo array, got shape {lr.shape}")

    left = lr[:, 0]
    right = lr[:, 1]
    factor = np.float32(1.0 / np.sqrt(2.0) if orthonormal else 0.5)
    mid = (left + right) * factor
    side = (right - left) * factor
    return mid, side


def compute_phase_metrics(stereo_lr: np.ndarray) -> PhaseMetrics:
    """Compute Audition-compatible phase correlation, M/S RMS/peaks, width, and balance."""
    lr = np.asarray(stereo_lr, dtype=np.float64)
    if lr.size == 0:
        return PhaseMetrics(
            correlation=1.0,
            mid_rms_db=-96.0,
            side_rms_db=-96.0,
            left_rms_db=-96.0,
            right_rms_db=-96.0,
            mid_peak_db=-96.0,
            side_peak_db=-96.0,
            left_peak_db=-96.0,
            right_peak_db=-96.0,
            stereo_width_pct=0.0,
            in_phase_pct=100.0,
            balance_db=0.0,
            sample_count=0,
        )

    left = lr[:, 0]
    right = lr[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (right - left)

    l_sq = float(np.sum(left * left))
    r_sq = float(np.sum(right * right))
    denom = np.sqrt(l_sq * r_sq)
    if denom <= 1e-24:
        corr = 1.0 if (l_sq <= 1e-24 and r_sq <= 1e-24) else 0.0
    else:
        corr = float(np.clip(np.sum(left * right) / denom, -1.0, 1.0))

    n = len(left)
    l_rms = float(np.sqrt(l_sq / n))
    r_rms = float(np.sqrt(r_sq / n))
    m_rms = float(np.sqrt(np.mean(mid * mid)))
    s_rms = float(np.sqrt(np.mean(side * side)))

    m_peak = float(np.max(np.abs(mid)))
    s_peak = float(np.max(np.abs(side)))
    l_peak = float(np.max(np.abs(left)))
    r_peak = float(np.max(np.abs(right)))

    # Stereo width: 0% = pure mono (S=0), 50% = uncorrelated / hard panned (S=M), 100% = pure anti-phase (M=0)
    ms_angle = np.arctan2(s_rms, m_rms) if (m_rms > 1e-12 or s_rms > 1e-12) else 0.0
    stereo_width_pct = float(np.clip((2.0 / np.pi) * ms_angle * 100.0, 0.0, 100.0))

    # In-phase sample fraction (where L*R >= 0, i.e., inside the +-45 deg M cone)
    prod = left * right
    active = (np.abs(left) + np.abs(right)) > 1e-7
    if np.any(active):
        in_phase_pct = float(np.mean(prod[active] >= 0.0) * 100.0)
    else:
        in_phase_pct = 100.0

    # L/R balance in dB (+ = Right louder, - = Left louder)
    balance_db = _amp_to_db(r_rms) - _amp_to_db(l_rms)
    if l_rms <= 1e-12 and r_rms <= 1e-12:
        balance_db = 0.0

    return PhaseMetrics(
        correlation=corr,
        mid_rms_db=_amp_to_db(m_rms),
        side_rms_db=_amp_to_db(s_rms),
        left_rms_db=_amp_to_db(l_rms),
        right_rms_db=_amp_to_db(r_rms),
        mid_peak_db=_amp_to_db(m_peak),
        side_peak_db=_amp_to_db(s_peak),
        left_peak_db=_amp_to_db(l_peak),
        right_peak_db=_amp_to_db(r_peak),
        stereo_width_pct=stereo_width_pct,
        in_phase_pct=in_phase_pct,
        balance_db=float(np.clip(balance_db, -60.0, 60.0)),
        sample_count=int(n),
    )


def apply_scale_points(
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
    scale_mode: ScaleMode = "linear",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale (x, y) coordinates for display, optionally applying normalization and log compression.

    Returns (x_scaled, y_scaled, applied_gain).
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if x_arr.size == 0:
        return x_arr, y_arr, 1.0

    gain = 1.0
    if normalize:
        peak = float(max(np.max(np.abs(x_arr)), np.max(np.abs(y_arr))))
        if peak > 1e-7:
            gain = 0.92 / peak
            x_arr = x_arr * gain
            y_arr = y_arr * gain

    if scale_mode != "linear":
        # Radial logarithmic / dB-style expansion for low-level detail
        r = np.hypot(x_arr, y_arr)
        floor_db = -36.0 if scale_mode == "log_zoom" else -72.0
        r_safe = np.maximum(r, 10.0 ** (floor_db / 20.0))
        r_db = 20.0 * np.log10(r_safe)
        r_norm = np.clip((r_db - floor_db) / (-floor_db), 0.0, 1.0)
        scale = np.where(r > 1e-12, r_norm / r_safe, 0.0)
        x_arr = x_arr * scale
        y_arr = y_arr * scale

    return x_arr, y_arr, gain


def splat_phosphor_density(
    x: np.ndarray,
    y: np.ndarray,
    size: int = 512,
    connect_lines: bool = True,
    steps_per_segment: int = 6,
    bloom_sigma: float = 0.85,
    max_samples: int = 65536,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize (x, y) in [-1, 1]^2 into a 2D phosphor density map and phase-zone mask.

    Returns:
        intensity: float32 [size, size] normalized in [0, 1] (origin top-left for image display: row 0 = y=+1).
        out_of_phase_ratio: float32 [size, size] in [0, 1], fraction of energy outside in-phase cone (|x| > |y|).
    """
    density = np.zeros((size, size), dtype=np.float32)
    anti_density = np.zeros((size, size), dtype=np.float32)

    if x.size == 0:
        return density, anti_density

    # Downsample uniformly if static selection is huge to keep rasterization sub-15ms
    if x.size > max_samples:
        stride = int(np.ceil(x.size / max_samples))
        x = x[::stride]
        y = y[::stride]

    if connect_lines and x.size >= 2:
        # Vectorized linear interpolation along consecutive sample segments
        t = np.linspace(0.0, 1.0, steps_per_segment, endpoint=False, dtype=np.float32)
        dx = np.diff(x)
        dy = np.diff(y)
        # Skip huge discontinuities when strided or clipped
        xs = (x[:-1, None] + dx[:, None] * t[None, :]).ravel()
        ys = (y[:-1, None] + dy[:, None] * t[None, :]).ravel()
        xs = np.concatenate([xs, x[-1:]])
        ys = np.concatenate([ys, y[-1:]])
    else:
        xs = x
        ys = y

    # Map x in [-1, 1] -> col [0, size-1]; y in [-1, 1] -> row [size-1, 0]
    half = (size - 1) * 0.5
    cols = np.rint(xs * half + half).astype(np.int32)
    rows = np.rint(-ys * half + half).astype(np.int32)

    valid = (cols >= 0) & (cols < size) & (rows >= 0) & (rows < size)
    if not np.any(valid):
        return density, anti_density

    cols = cols[valid]
    rows = rows[valid]
    xs_v = xs[valid]
    ys_v = ys[valid]

    np.add.at(density, (rows, cols), 1.0)
    anti_mask = (np.abs(xs_v) > np.abs(ys_v)).astype(np.float32)
    np.add.at(anti_density, (rows, cols), anti_mask)

    if bloom_sigma > 0.0:
        density = gaussian_filter(density, sigma=bloom_sigma, mode="constant")
        anti_density = gaussian_filter(anti_density, sigma=bloom_sigma, mode="constant")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(density > 1e-6, np.clip(anti_density / density, 0.0, 1.0), 0.0)

    # Logarithmic phosphor tone-mapping: crisp low trails + bright overlapping cores
    peak = float(np.max(density))
    if peak > 0.0:
        # Adaptive compression factor so both sparse 1024-sample frames and dense 65k frames shine
        k = 18.0 / np.sqrt(max(peak, 1.0))
        mapped = np.log1p(density * k) / np.log1p(peak * k)
        intensity = np.clip(mapped, 0.0, 1.0).astype(np.float32)
    else:
        intensity = density

    return intensity, ratio.astype(np.float32)


def compute_polar_histogram(
    stereo_lr: np.ndarray,
    num_bins: int = 180,
    scale_mode: ScaleMode = "log_zoom",
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute Audition-style polar stereo phase histogram.

    Maps each sample into Mid/Side polar angle theta in [-pi, pi], where:
        theta = 0       -> +Mid (center, in-phase)
        theta = -pi/4   -> +Left
        theta = +pi/4   -> +Right
        theta = +-pi/2  -> Side (anti-phase)

    Folded into directional symmetry around [0, 2*pi) (0 = top/+M, clockwise toward +R, +S, -M, -L).
    Returns:
        angles_rad: bin centers [num_bins] where 0 = top (+M), clockwise.
        radii: normalized magnitude [num_bins] in [0, 1].
        mean_angle_rad: energy-weighted tracking-ball angle in upper hemisphere.
        mean_radius: tracking-ball coherence magnitude in [0, 1].
    """
    mid, side = compute_mid_side(stereo_lr, orthonormal=False)
    mag = np.hypot(side, mid)
    if mag.size == 0 or float(np.max(mag)) <= 1e-9:
        angles = np.linspace(0.0, 2.0 * np.pi, num_bins, endpoint=False)
        return angles, np.zeros(num_bins, dtype=np.float32), 0.0, 0.0

    # Fold opposite polarity halves (+/-) symmetrically onto upper/lower or full circle
    # Standard goniometer/histogram treats sample polarity symmetrically (x, y) and (-x, -y)
    # Wait: plotting full signed angle atan2(side, mid) captures asymmetry, while folding
    # upper hemisphere [-pi/2, +pi/2] or reflecting keeps L/R distinct regardless of waveform sign.
    # Let's fold negative Mid into upper half by multiplying both (side, mid) by sign(mid):
    # That way a pure Left sine wave stays on -45 deg (top-left) instead of splitting 50/50 to bottom-right!
    sign_m = np.where(mid >= 0.0, 1.0, -1.0)
    s_folded = side * sign_m
    m_folded = mid * sign_m

    # Angle clockwise from +M (top): atan2(s_folded, m_folded) in [-pi/2, +pi/2]
    # Note: an anti-phase signal (M=0) spans both -pi/2 and +pi/2!
    theta = np.arctan2(s_folded, m_folded)  # [-pi/2, +pi/2]

    # Bin over full [-pi, pi) for symmetric polar display (mirror across origin so it forms a classic Audition flower!)
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Symmetrize energy across origin (theta and theta + pi) like Audition's circular histogram
    theta_opp = np.where(theta >= 0.0, theta - np.pi, theta + np.pi)
    weights = mag * mag  # energy weighting

    hist_pos, _ = np.histogram(theta, bins=bin_edges, weights=weights)
    hist_neg, _ = np.histogram(theta_opp, bins=bin_edges, weights=weights)
    hist = (hist_pos + hist_neg).astype(np.float64)

    # Circular smoothing for clean Audition-like envelope
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    hist = np.convolve(padded, kernel, mode="valid")

    # RMS amplitude per bin
    hist = np.sqrt(np.maximum(hist, 0.0))

    if normalize:
        peak = float(np.max(hist))
        if peak > 1e-12:
            hist = hist / peak
    else:
        # Normalize relative to full-scale energy density
        scale_ref = np.sqrt(np.mean(weights)) * np.sqrt(num_bins * 0.25) + 1e-12
        hist = np.clip(hist / max(scale_ref, 1e-9), 0.0, 1.0)

    if scale_mode == "log_zoom":
        floor_db = -30.0
        db = 20.0 * np.log10(np.maximum(hist, 10.0 ** (floor_db / 20.0)))
        radii = np.clip((db - floor_db) / (-floor_db), 0.0, 1.0)
    elif scale_mode == "log_full":
        floor_db = -60.0
        db = 20.0 * np.log10(np.maximum(hist, 10.0 ** (floor_db / 20.0)))
        radii = np.clip((db - floor_db) / (-floor_db), 0.0, 1.0)
    else:
        radii = np.clip(hist, 0.0, 1.0)

    # Energy-weighted tracking ball in upper hemisphere
    total_w = float(np.sum(weights)) + 1e-18
    cx = float(np.sum(weights * np.sin(theta)) / total_w)
    cy = float(np.sum(weights * np.cos(theta)) / total_w)
    mean_angle = float(np.arctan2(cx, cy))
    mean_radius = float(np.clip(np.hypot(cx, cy), 0.0, 1.0))

    return bin_centers.astype(np.float32), radii.astype(np.float32), mean_angle, mean_radius


def compute_phase_wheel(
    stereo_lr: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    min_freq: float = 20.0,
    max_freq: float = 20000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Adobe Audition-style frequency-resolved Phase Wheel coordinates.

    For each frequency bin f:
        - radius r in [0, 1] maps log-frequency from min_freq (center) to min(max_freq, sr/2) (outer rim)
        - angle delta_phi = angle(R(f) * conj(L(f))) in [-pi, pi]:
            0 deg (top) = in-phase
            +-90 deg (left/right) = quadrature
            +-180 deg (bottom) = polarity inverted
        - weight = combined spectral magnitude (normalized in [0, 1])

    Returns:
        x_wheel, y_wheel, freqs_hz, mag_norm
    """
    lr = np.asarray(stereo_lr, dtype=np.float64)
    n = lr.shape[0]
    if n < 32:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    eff_fft = min(n_fft, n)
    # Average cross-spectrum across overlapping Hann-windowed frames for low-variance phase estimates
    hop = max(eff_fft // 2, 1)
    window = np.hanning(eff_fft)

    cross_spec = np.zeros(eff_fft // 2 + 1, dtype=np.complex128)
    power_l = np.zeros(eff_fft // 2 + 1, dtype=np.float64)
    power_r = np.zeros(eff_fft // 2 + 1, dtype=np.float64)

    frames = 0
    for start in range(0, max(1, n - eff_fft + 1), hop):
        seg_l = lr[start : start + eff_fft, 0] * window
        seg_r = lr[start : start + eff_fft, 1] * window
        fl = np.fft.rfft(seg_l)
        fr = np.fft.rfft(seg_r)
        cross_spec += fr * np.conj(fl)
        power_l += np.abs(fl) ** 2
        power_r += np.abs(fr) ** 2
        frames += 1

    if frames == 0:
        seg_l = np.pad(lr[:, 0], (0, eff_fft - n)) * window
        seg_r = np.pad(lr[:, 1], (0, eff_fft - n)) * window
        fl = np.fft.rfft(seg_l)
        fr = np.fft.rfft(seg_r)
        cross_spec = fr * np.conj(fl)
        power_l = np.abs(fl) ** 2
        power_r = np.abs(fr) ** 2

    freqs = np.fft.rfftfreq(eff_fft, d=1.0 / sample_rate)
    nyq = sample_rate * 0.5
    hi = min(max_freq, nyq)
    lo = min(min_freq, hi * 0.5)
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    f_sel = freqs[mask]
    cs_sel = cross_spec[mask]
    mag = np.sqrt(power_l[mask] + power_r[mask])

    mag_peak = float(np.max(mag))
    if mag_peak <= 1e-18:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    mag_db = 20.0 * np.log10(np.maximum(mag / mag_peak, 1e-6))
    mag_norm = np.clip((mag_db + 60.0) / 60.0, 0.0, 1.0)

    # Filter out silent frequency bins so noise floor doesn't clutter the wheel
    active = mag_norm > 0.05
    f_sel = f_sel[active]
    cs_sel = cs_sel[active]
    mag_norm = mag_norm[active]

    if f_sel.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    # Log-frequency radial coordinate: 0 at lo (center), 1 at hi (rim)
    r = np.log10(f_sel / lo) / max(np.log10(hi / lo), 1e-6)
    r = np.clip(r, 0.02, 0.98)

    # Phase difference angle: 0 at top (+Y), clockwise positive when R leads L
    dphi = np.angle(cs_sel)
    x_w = r * np.sin(dphi)
    y_w = r * np.cos(dphi)

    return (
        x_w.astype(np.float32),
        y_w.astype(np.float32),
        f_sel.astype(np.float32),
        mag_norm.astype(np.float32),
    )


def generate_test_signal(
    preset: str = "stereo_mix",
    duration_s: float = 4.0,
    sample_rate: int = 48000,
) -> Tuple[np.ndarray, int]:
    """Generate realistic stereo DSP test signals for verification and demonstration."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(42)

    if preset == "mono":
        sig = 0.65 * np.sin(2.0 * np.pi * 220.0 * t) + 0.25 * np.sin(2.0 * np.pi * 660.0 * t)
        lr = np.column_stack([sig, sig])
    elif preset == "out_of_phase":
        sig = 0.6 * np.sin(2.0 * np.pi * 330.0 * t) + 0.2 * np.sin(2.0 * np.pi * 880.0 * t)
        lr = np.column_stack([sig, -sig])
    elif preset == "hard_left":
        sig = 0.7 * np.sin(2.0 * np.pi * 440.0 * t)
        lr = np.column_stack([sig, np.zeros_like(sig)])
    elif preset == "hard_right":
        sig = 0.7 * np.sin(2.0 * np.pi * 440.0 * t)
        lr = np.column_stack([np.zeros_like(sig), sig])
    elif preset == "quadrature":
        # 90-degree phase shift -> circle on both Lissajous and Mid/Side
        env = 0.55 + 0.25 * np.sin(2.0 * np.pi * 0.5 * t)
        left = env * np.sin(2.0 * np.pi * 250.0 * t)
        right = env * np.cos(2.0 * np.pi * 250.0 * t)
        lr = np.column_stack([left, right])
    elif preset == "delayed_comb":
        # Common mic phase issue: Right delayed by 0.6 ms relative to Left
        sig = rng.standard_normal(n)
        # Simple lowpass smoothing to sound musical / noise-like
        from scipy.signal import butter, filtfilt

        b, a = butter(3, [120.0 / (sample_rate * 0.5), 6000.0 / (sample_rate * 0.5)], btype="band")
        band = filtfilt(b, a, sig)
        band = band / (np.max(np.abs(band)) + 1e-9) * 0.65
        delay_samples = int(0.0006 * sample_rate)
        right = np.roll(band, delay_samples)
        right[:delay_samples] = 0.0
        lr = np.column_stack([band, right])
    else:
        # "stereo_mix": Rich musical-style synth with centered bass/kick + wide stereo pad/harmonics
        # Centered bass pulse + fundamental
        kick_env = np.exp(-((t % 0.5) * 9.0))
        mid_bass = 0.55 * np.sin(2.0 * np.pi * (55.0 + 45.0 * kick_env) * t) * (0.4 + 0.6 * kick_env)
        mid_lead = 0.28 * np.sin(2.0 * np.pi * 330.0 * t) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 1.5 * t))
        mid_total = mid_bass + mid_lead

        # Wide stereo shimmer + slightly detuned chorus pad (non-phase-locked for natural symmetry)
        side_pad = (
            0.18 * np.sin(2.0 * np.pi * 497.3 * t + 0.6 * np.sin(2.0 * np.pi * 0.75 * t))
            + 0.12 * np.sin(2.0 * np.pi * 991.7 * t + 1.1)
        )
        # Occasional gentle stereo pan sweep
        pan = np.sin(2.0 * np.pi * 0.25 * t)
        left = mid_total + side_pad * (0.7 + 0.3 * pan)
        right = mid_total - side_pad * (0.7 - 0.3 * pan)
        # Subtle stereo room ambience
        amb = rng.standard_normal((n, 2)) * 0.025
        lr = np.column_stack([left, right]) + amb
        peak = np.max(np.abs(lr))
        if peak > 0.9:
            lr *= 0.88 / peak

    return lr.astype(np.float32), int(sample_rate)
