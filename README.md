# Mid / Side Phase Plot (Adobe Audition-Style)

A Python-based interactive stereo Phase Analysis tool and DSP plotting library modeled after **Adobe Audition's Phase Analysis** panel.

## Features

- **Mid / Side Plot (45° rotated vectorscope)**
  - Vertical axis (`+M` / `-M`): Mid sum $(L + R) / 2$
  - Horizontal axis (`-S` / `+S`): Side difference $(R - L) / 2$
  - Diagonal `L` ($-45^\circ$) and `R` ($+45^\circ$) reference axes, concentric $0\text{ dBFS} / -6\text{ dB} / -12\text{ dB}$ diamonds
  - CRT-style phosphor density rasterizer with persistence trails and optional **Phase Zones** coloring (in-phase cyan-emerald vs out-of-phase coral)
- **Additional Audition Chart Types**
  - **Lissajous (L/R)** standard X-Y vectorscope
  - **Histogram** (`Linear`, `Log Zoom`, `Log Full`) polar stereo energy distribution with tracking ball
  - **Phase Wheel** frequency-resolved ($20\text{ Hz} - 20\text{ kHz}$) inter-channel phase difference wheel
- **Phase Correlation Meter (`-1.0` to `+1.0`)** & live M/S / L/R RMS, peak, stereo width %, balance, and in-phase % readouts
- **Interactive Studio GUI** (PySide6): load/drag-and-drop WAV/FLAC/AIFF/OGG/MP3/M4A files, waveform scrubber, audio playback, polarity invert test, multichannel pair selector, live audio input monitor, and high-res PNG/SVG export
- **Scriptable Python API & CLI** for headless batch analysis and Matplotlib figures

## Quick Start

### macOS

Launch the interactive desktop application:

```bash
cd ~/mid_side_phase_plot
./run.sh
```

Open a specific audio file in the GUI:

```bash
./run.sh /path/to/audio.wav
```

Export a high-resolution static plot headlessly:

```bash
./run.sh /path/to/audio.wav --chart mid_side --save out.png
./run.sh --preset delayed_comb --chart phase_wheel --save wheel.png
```

### Windows

Create a Windows virtual environment (do not reuse the macOS `.venv` if copying the folder across machines):

```bat
cd %USERPROFILE%\mid_side_phase_plot
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-deps -e .
```

Then launch the GUI or export headlessly:

```bat
run.bat
run.bat C:\path\to\audio.wav
run.bat C:\path\to\audio.wav --chart mid_side --save out.png
```

*(Note: WAV, FLAC, AIFF, OGG, and MP3 work natively via `soundfile`. For M4A/AAC/WMA files on Windows, install `ffmpeg` on your `PATH`.)*

## Python API Usage

```python
from ms_phase_plot import compute_phase_metrics, load_audio, plot_mid_side

stereo, sr, n_ch = load_audio("mix.wav")
metrics = compute_phase_metrics(stereo)
print(f"Correlation: {metrics.correlation:+.3f}, Width: {metrics.stereo_width_pct:.1f}%")

fig, ax = plot_mid_side(stereo, sample_rate=sr, normalize=True, save_path="phase.png")
```

## Running Tests

```bash
cd ~/mid_side_phase_plot
.venv/bin/pytest -v
```
