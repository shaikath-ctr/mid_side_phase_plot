# Mid / Side Phase Plot (Adobe Audition-Style)

A Python-based interactive stereo Phase Analysis tool and DSP plotting library modeled after **Adobe Audition's Phase Analysis** panel. Works on both **Windows** and **macOS**.

## Features

- **Mid / Side Plot (45° rotated vectorscope)**
  - Vertical axis (`+M` / `-M`): Mid sum $(L + R) / 2$
  - Horizontal axis (`-S` / `+S`): Side difference $(R - L) / 2$
  - Diagonal `L` ($-45^\circ$) and `R` ($+45^\circ$) reference axes with concentric $0\text{ dBFS} / -6\text{ dB} / -12\text{ dB}$ diamonds
  - CRT-style phosphor density rasterizer with persistence trails and optional **Phase Zones** coloring (in-phase cyan-emerald vs. out-of-phase coral)
- **Additional Audition Chart Types**
  - **Lissajous (L/R)**: classic X-Y vectorscope
  - **Histogram**: polar stereo energy distribution (`Linear`, `Log Zoom`, `Log Full`) with tracking ball
  - **Phase Wheel**: frequency-resolved ($20\text{ Hz} - 20\text{ kHz}$) inter-channel phase difference wheel
- **Phase Correlation Meter (`-1.0` to `+1.0`)** & real-time Mid, Side, Left, Right RMS/peak, stereo width %, in-phase %, and L/R balance readouts
- **Interactive Studio GUI** (`PySide6`): drag-and-drop audio files, waveform scrubber, audio playback, 180° polarity invert toggle, multichannel pair selector, live audio input monitor, and high-res PNG/SVG/PDF export
- **Scriptable Python API & Headless CLI** for batch analysis and Matplotlib figures

---

## Requirements

- **Python 3.10+** (tested on Python 3.11)
- Core packages installed automatically via `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `soundfile`, `sounddevice`, `PySide6`

---

## Running on macOS

### 1. Clone and Install

```bash
git clone https://github.com/shaikath-ctr/mid_side_phase_plot.git
cd mid_side_phase_plot

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .
```

*(Note: If you already created this folder locally with `.venv`, you can skip straight to launching below.)*

### 2. Launch the App

```bash
# Launch the interactive GUI (starts with built-in Stereo Mix demo signal)
./run.sh

# Open an audio file directly in the GUI
./run.sh /path/to/audio.wav

# Launch with a specific built-in test signal
./run.sh --preset delayed_comb
```

### 3. Headless Plot Export (CLI)

```bash
./run.sh /path/to/audio.wav --chart mid_side --save phase.png
./run.sh --preset delayed_comb --chart phase_wheel --save wheel.png
```

- **Supported macOS audio formats:** WAV, FLAC, AIFF, OGG, and MP3 work natively via `soundfile`; `.m4a`, `.aac`, and `.caf` are decoded automatically via built-in macOS `afconvert` (or `ffmpeg` if installed).

---

## Running on Windows

### 1. Clone and Install

> **Important:** If you copied this folder from a Mac, delete any existing `.venv` folder first and create a fresh Windows virtual environment.

**Using Command Prompt (`cmd`):**

```bat
git clone https://github.com/shaikath-ctr/mid_side_phase_plot.git
cd mid_side_phase_plot

python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-deps -e .
```

**Using PowerShell:**

```powershell
git clone https://github.com/shaikath-ctr/mid_side_phase_plot.git
cd mid_side_phase_plot

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install --no-deps -e .
```

*(If `python` is not on your `PATH`, use `py -3 -m venv .venv` with the Windows Python Launcher.)*

### 2. Launch the App

Once installed, you can **double-click `run.bat`** in File Explorer, or run from a terminal:

**Command Prompt (`cmd`):**

```bat
run.bat
run.bat C:\path\to\audio.wav
run.bat --preset delayed_comb
```

**PowerShell:**

```powershell
.\run.bat
.\run.bat "C:\path\to\audio.wav"
.\run.bat --preset delayed_comb
```

### 3. Headless Plot Export (CLI)

```bat
run.bat C:\path\to\audio.wav --chart mid_side --save phase.png
run.bat --preset delayed_comb --chart phase_wheel --save wheel.png
```

- **Supported Windows audio formats:** WAV, FLAC, AIFF, OGG, and MP3 work natively via `soundfile`. For `.m4a`, `.aac`, or `.wma` files on Windows, install `ffmpeg` and ensure it is on your system `PATH`.

---

## GUI Shortcuts & Tips

- **Drag & Drop:** Drop any supported audio file directly onto the window to load it.
- **`Space`:** Play / Pause audio playback.
- **`Cmd+O` (macOS) / `Ctrl+O` (Windows):** Open an audio file.
- **`Cmd+S` (macOS) / `Ctrl+S` (Windows):** Export the current view as a high-resolution PNG, SVG, or PDF.
- **Scrubbing:** Click or drag across the bottom L/R waveform strip to inspect any region in the file.
- **Invert Right Polarity (180°):** Quickly verify how phase inversion collapses Mid energy into the horizontal Side axis.

---

## CLI Options Reference

| Flag | Values / Description |
| --- | --- |
| `audio_file` | Optional path to `.wav`, `.flac`, `.aiff`, `.ogg`, `.mp3`, `.m4a`, etc. |
| `--preset` | `stereo_mix` (default), `mono`, `out_of_phase`, `hard_left`, `hard_right`, `quadrature`, `delayed_comb` |
| `--chart` | `mid_side` (default), `lissajous`, `histogram`, `phase_wheel` |
| `--scale` | `linear` (default), `log_zoom`, `log_full` |
| `--color` | `phosphor` (default), `phase_zones` |
| `--left-ch` / `--right-ch` | 0-based channel indices for multichannel files (default: `0` and `1`) |
| `--start` / `--duration` | Optional slice start & duration in seconds for `--save` exports |
| `--no-normalize` | Disable automatic display normalization (show true `0 dBFS` scale) |
| `--save` | Export plot headlessly to `.png`, `.svg`, or `.pdf` without opening the GUI |
| `--dpi` | Output DPI for `--save` (default: `200`) |

---

## Python API Usage

```python
from ms_phase_plot import compute_phase_metrics, load_audio, plot_mid_side

stereo, sr, n_ch = load_audio("mix.wav")
metrics = compute_phase_metrics(stereo)
print(f"Correlation: {metrics.correlation:+.3f}, Width: {metrics.stereo_width_pct:.1f}%")

fig, ax = plot_mid_side(stereo, sample_rate=sr, normalize=True, save_path="phase.png")
```

---

## Running Tests

**macOS:**

```bash
cd mid_side_phase_plot
.venv/bin/pytest -v
```

**Windows:**

```bat
cd mid_side_phase_plot
.venv\Scripts\pytest -v
```
