"""Interactive Adobe Audition-style Mid/Side Phase Analysis GUI built with PySide6."""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

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
    load_raw_channels,
    splat_phosphor_density,
)
from .static_plot import _density_to_rgba, plot_phase_analysis

def _ui_font(point_size: int, bold: bool = False) -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    font.setPointSize(point_size)
    if bold:
        font.setWeight(QFont.Bold)
    return font


def _mono_font(point_size: int, bold: bool = False) -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPointSize(point_size)
    if bold:
        font.setWeight(QFont.Bold)
    return font


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #16181E;
    color: #D9E2EC;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
}
QGroupBox {
    background-color: #1D2028;
    border: 1px solid #2B313D;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: #9FB0C3;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #8BA0B8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
QPushButton {
    background-color: #252A35;
    border: 1px solid #343C4A;
    border-radius: 5px;
    padding: 6px 12px;
    color: #E2E8F0;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2E3543;
    border-color: #465164;
}
QPushButton:pressed {
    background-color: #1D222B;
}
QPushButton:checked {
    background-color: #10493D;
    border-color: #26F7B4;
    color: #A7FFE3;
}
QPushButton#primaryBtn {
    background-color: #124E41;
    border-color: #22D9A0;
    color: #E6FFF7;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #176252;
}
QPushButton#dangerToggle:checked {
    background-color: #5C1D28;
    border-color: #FF5A6E;
    color: #FFD6DC;
}
QComboBox {
    background-color: #14171D;
    border: 1px solid #313846;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
    color: #E2E8F0;
}
QComboBox:hover {
    border-color: #475266;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #2A313D;
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8FA0B5;
}
QComboBox QAbstractItemView {
    background-color: #1D2028;
    border: 1px solid #3A4252;
    selection-background-color: #145446;
    color: #E2E8F0;
}
QCheckBox {
    spacing: 7px;
    color: #CBD5E1;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 3px;
    border: 1px solid #3D4657;
    background-color: #12151B;
}
QCheckBox::indicator:checked {
    background-color: #19B888;
    border-color: #26F7B4;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #11141A;
    border: 1px solid #2A313D;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #26F7B4;
    border: 1px solid #0B261E;
    width: 13px;
    margin: -5px 0;
    border-radius: 6px;
}
QLabel#monoReadout {
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    color: #D9E2EC;
}
"""


class PhaseCanvasWidget(QWidget):
    """High-FPS Adobe Audition-style Phase Analysis canvas with phosphor persistence."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(460, 460)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.chart_type: ChartType = "mid_side"
        self.scale_mode: ScaleMode = "linear"
        self.normalize: bool = True
        self.color_mode: ColorMode = "phosphor"
        self.connect_lines: bool = True
        self.persistence: float = 0.72
        self.resolution: int = 512

        self._persist_int = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self._persist_anti = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self._qimage: Optional[QImage] = None

        # Histogram & Phase Wheel state
        self._hist_angles: Optional[np.ndarray] = None
        self._hist_radii: Optional[np.ndarray] = None
        self._hist_ball: tuple[float, float] = (0.0, 0.0)

        self._wheel_pts: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        self._hover_text: str = ""
        self._applied_gain: float = 1.0

    def reset_persistence(self) -> None:
        """Clear accumulated phosphor trail."""
        self._persist_int.fill(0.0)
        self._persist_anti.fill(0.0)
        self._qimage = None
        self.update()

    def update_audio_frame(
        self,
        stereo_lr: np.ndarray,
        sample_rate: int,
        is_streaming: bool = False,
    ) -> None:
        """Process a stereo [N, 2] chunk and update the active visualization."""
        if stereo_lr.size == 0:
            self.reset_persistence()
            return

        if self.chart_type in ("mid_side", "lissajous"):
            if self.chart_type == "mid_side":
                mid, side = compute_mid_side(stereo_lr, orthonormal=False)
                x_raw, y_raw = side, mid
            else:
                x_raw, y_raw = stereo_lr[:, 0], stereo_lr[:, 1]

            x_pts, y_pts, gain = apply_scale_points(
                x_raw, y_raw, normalize=self.normalize, scale_mode=self.scale_mode
            )
            self._applied_gain = gain
            inst, anti = splat_phosphor_density(
                x_pts,
                y_pts,
                size=self.resolution,
                connect_lines=self.connect_lines,
                steps_per_segment=5 if is_streaming else 7,
                bloom_sigma=0.80,
                max_samples=24576 if is_streaming else 65536,
            )

            if self.chart_type == "lissajous":
                yy, xx = np.mgrid[
                    1.0 : -1.0 : complex(0, self.resolution),
                    -1.0 : 1.0 : complex(0, self.resolution),
                ]
                anti = ((xx * yy) < 0.0).astype(np.float32)

            if is_streaming and self.persistence > 0.01:
                decay = float(np.clip(self.persistence, 0.0, 0.96))
                self._persist_int = np.maximum(self._persist_int * decay, inst)
                # Blend anti-phase ratio smoothly where new trace arrives
                blend = inst > 0.05
                self._persist_anti = np.where(
                    blend,
                    np.maximum(self._persist_anti * decay, anti),
                    self._persist_anti * decay,
                )
            else:
                self._persist_int = inst
                self._persist_anti = anti

            rgba = np.ascontiguousarray(
                _density_to_rgba(self._persist_int, self._persist_anti, color_mode=self.color_mode)
            )
            self._qimage = QImage(
                rgba.data,
                self.resolution,
                self.resolution,
                self.resolution * 4,
                QImage.Format_RGBA8888,
            ).copy()

        elif self.chart_type == "histogram":
            angles, radii, mean_ang, mean_r = compute_polar_histogram(
                stereo_lr,
                num_bins=180,
                scale_mode=self.scale_mode,
                normalize=self.normalize,
            )
            if is_streaming and self._hist_radii is not None and self.persistence > 0.01:
                alpha = 1.0 - float(np.clip(self.persistence * 0.82, 0.0, 0.85))
                self._hist_radii = (1.0 - alpha) * self._hist_radii + alpha * radii
                bx0 = self._hist_ball[1] * np.sin(self._hist_ball[0])
                by0 = self._hist_ball[1] * np.cos(self._hist_ball[0])
                bx1 = mean_r * np.sin(mean_ang)
                by1 = mean_r * np.cos(mean_ang)
                bx = (1.0 - alpha) * bx0 + alpha * bx1
                by = (1.0 - alpha) * by0 + alpha * by1
                self._hist_ball = (float(np.arctan2(bx, by)), float(np.hypot(bx, by)))
            else:
                self._hist_radii = radii
                self._hist_ball = (mean_ang, mean_r)
            self._hist_angles = angles

        elif self.chart_type == "phase_wheel":
            xw, yw, _, mags = compute_phase_wheel(stereo_lr, sample_rate=sample_rate)
            self._wheel_pts = (xw, yw, mags)

        self.update()

    def _plot_square(self) -> QRectF:
        pad = 34.0
        w = max(10.0, self.width() - 2.0 * pad)
        h = max(10.0, self.height() - 2.0 * pad)
        side = min(w, h)
        x0 = (self.width() - side) * 0.5
        y0 = (self.height() - side) * 0.5
        return QRectF(x0, y0, side, side)

    def mouseMoveEvent(self, event) -> None:
        rect = self._plot_square()
        pos = event.position()
        if not rect.contains(pos):
            self._hover_text = ""
            self.update()
            return

        nx = ((pos.x() - rect.left()) / rect.width()) * 2.0 - 1.0
        ny = 1.0 - ((pos.y() - rect.top()) / rect.height()) * 2.0
        if self.chart_type == "mid_side":
            ang = np.degrees(np.arctan2(nx, ny))
            self._hover_text = f"S: {nx:+.2f}  M: {ny:+.2f}  ({ang:+.1f}°)"
        elif self.chart_type == "lissajous":
            self._hover_text = f"L: {nx:+.2f}  R: {ny:+.2f}"
        elif self.chart_type == "histogram":
            ang = np.degrees(np.arctan2(nx, ny))
            mag = np.hypot(nx, ny)
            self._hover_text = f"Angle: {ang:+.1f}°  Mag: {mag:.2f}"
        else:
            ang = np.degrees(np.arctan2(nx, ny))
            r = np.clip(np.hypot(nx, ny), 0.0, 1.0)
            freq = 20.0 * (1000.0**r)
            self._hover_text = f"Phase: {ang:+.0f}°  Freq: ~{freq:.0f} Hz"
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_text = ""
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Studio near-black backdrop
        painter.fillRect(self.rect(), QColor("#090C10"))
        rect = self._plot_square()
        cx, cy = rect.center().x(), rect.center().y()
        half = rect.width() * 0.5

        def to_px(x: float, y: float) -> QPointF:
            return QPointF(cx + x * half, cy - y * half)

        # Draw background graticule depending on chart mode
        if self.chart_type == "mid_side":
            self._paint_mid_side_grid(painter, rect, to_px)
        elif self.chart_type == "lissajous":
            self._paint_lissajous_grid(painter, rect, to_px)
        else:
            self._paint_polar_grid(painter, rect, to_px)

        # Draw data layer
        if self.chart_type in ("mid_side", "lissajous"):
            if self._qimage is not None:
                painter.drawImage(rect, self._qimage)
        elif self.chart_type == "histogram":
            self._paint_histogram(painter, to_px)
        elif self.chart_type == "phase_wheel":
            self._paint_phase_wheel(painter, to_px)

        # Top-left mode badge & hover telemetry
        painter.setFont(_mono_font(10))
        painter.setPen(QColor("#788699"))
        mode_names = {
            "mid_side": "MID / SIDE",
            "lissajous": "LISSAJOUS (L/R)",
            "histogram": f"HISTOGRAM ({self.scale_mode.upper()})",
            "phase_wheel": "PHASE WHEEL",
        }
        badge = mode_names[self.chart_type]
        if self.normalize and self.chart_type != "phase_wheel":
            badge += f"  •  NORM x{self._applied_gain:.2f}" if self._applied_gain > 1.05 else "  •  NORM"
        painter.drawText(12, 20, badge)

        if self._hover_text:
            painter.setPen(QColor("#A3F7DC"))
            painter.drawText(
                QRectF(12, self.height() - 24, self.width() - 24, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                self._hover_text,
            )

    def _paint_mid_side_grid(self, painter: QPainter, rect: QRectF, to_px) -> None:
        # Outer square border
        painter.setPen(QPen(QColor("#1E242E"), 1.0))
        painter.drawRect(rect)

        # Concentric diamonds: 0 dBFS, -6 dB, -12 dB
        for lvl, color, style in [
            (1.0, "#2D3644", Qt.SolidLine),
            (0.5, "#1E2530", Qt.DashLine),
            (0.25, "#171C25", Qt.DotLine),
        ]:
            poly = QPolygonF([to_px(0, lvl), to_px(lvl, 0), to_px(0, -lvl), to_px(-lvl, 0)])
            painter.setPen(QPen(QColor(color), 1.0, style))
            painter.drawPolygon(poly)

        # Vertical (+M/-M) and horizontal (-S/+S) axes
        painter.setPen(QPen(QColor("#2A3340"), 1.0))
        painter.drawLine(to_px(0, -1.0), to_px(0, 1.0))
        painter.drawLine(to_px(-1.0, 0), to_px(1.0, 0))

        # Diagonal L (-45 deg) and R (+45 deg) axes
        d = 0.76
        painter.setPen(QPen(QColor("#27303D"), 1.0, Qt.DashDotLine))
        painter.drawLine(to_px(-d, d), to_px(d, -d))
        painter.drawLine(to_px(-d, -d), to_px(d, d))

        # Axis labels
        painter.setFont(_ui_font(10, bold=True))
        painter.setPen(QColor("#E2E8F0"))
        painter.drawText(QRectF(rect.center().x() - 24, rect.top() - 24, 48, 20), Qt.AlignCenter, "+M")
        painter.setPen(QColor("#7F8C9D"))
        painter.drawText(QRectF(rect.center().x() - 24, rect.bottom() + 4, 48, 20), Qt.AlignCenter, "-M")
        painter.drawText(QRectF(rect.left() - 28, rect.center().y() - 10, 24, 20), Qt.AlignRight | Qt.AlignVCenter, "-S")
        painter.drawText(QRectF(rect.right() + 4, rect.center().y() - 10, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "+S")

        painter.setPen(QColor("#26F7B4"))
        p_l = to_px(-0.62, 0.62)
        painter.drawText(QRectF(p_l.x() - 22, p_l.y() - 18, 20, 18), Qt.AlignCenter, "L")
        p_r = to_px(0.62, 0.62)
        painter.drawText(QRectF(p_r.x() + 2, p_r.y() - 18, 20, 18), Qt.AlignCenter, "R")

        painter.setFont(_ui_font(8, bold=True))
        painter.setPen(QColor("#546173"))
        p_nr = to_px(-0.62, -0.62)
        painter.drawText(QRectF(p_nr.x() - 22, p_nr.y(), 22, 16), Qt.AlignCenter, "-R")
        p_nl = to_px(0.62, -0.62)
        painter.drawText(QRectF(p_nl.x(), p_nl.y(), 22, 16), Qt.AlignCenter, "-L")

    def _paint_lissajous_grid(self, painter: QPainter, rect: QRectF, to_px) -> None:
        painter.setPen(QPen(QColor("#2D3644"), 1.0))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#1B212B"), 1.0, Qt.DotLine))
        half_w, half_h = rect.width() * 0.25, rect.height() * 0.25
        painter.drawRect(rect.adjusted(half_w, half_h, -half_w, -half_h))

        painter.setPen(QPen(QColor("#2A3340"), 1.0))
        painter.drawLine(to_px(0, -1.0), to_px(0, 1.0))
        painter.drawLine(to_px(-1.0, 0), to_px(1.0, 0))
        painter.setPen(QPen(QColor("#27303D"), 1.0, Qt.DashDotLine))
        painter.drawLine(to_px(-1.0, -1.0), to_px(1.0, 1.0))
        painter.drawLine(to_px(-1.0, 1.0), to_px(1.0, -1.0))

        painter.setFont(_ui_font(10, bold=True))
        painter.setPen(QColor("#E2E8F0"))
        painter.drawText(QRectF(rect.right() + 4, rect.center().y() - 10, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "+L")
        painter.drawText(QRectF(rect.center().x() - 20, rect.top() - 24, 40, 20), Qt.AlignCenter, "+R")
        painter.setPen(QColor("#26F7B4"))
        pm = to_px(0.88, 0.88)
        painter.drawText(QRectF(pm.x() - 16, pm.y() - 16, 20, 18), Qt.AlignCenter, "M")

    def _paint_polar_grid(self, painter: QPainter, rect: QRectF, to_px) -> None:
        for r, col, st in [(1.0, "#2D3644", Qt.SolidLine), (0.75, "#1E2530", Qt.DotLine), (0.5, "#1E2530", Qt.DashLine), (0.25, "#171C25", Qt.DotLine)]:
            rad_x = rect.width() * 0.5 * r
            rad_y = rect.height() * 0.5 * r
            painter.setPen(QPen(QColor(col), 1.0, st))
            painter.drawEllipse(rect.center(), rad_x, rad_y)

        painter.setPen(QPen(QColor("#2A3340"), 1.0))
        painter.drawLine(to_px(0, -1.0), to_px(0, 1.0))
        painter.drawLine(to_px(-1.0, 0), to_px(1.0, 0))
        d = 1.0 / np.sqrt(2.0)
        painter.setPen(QPen(QColor("#202732"), 1.0, Qt.DashDotLine))
        painter.drawLine(to_px(-d, d), to_px(d, -d))
        painter.drawLine(to_px(-d, -d), to_px(d, d))

        painter.setFont(_ui_font(9, bold=True))
        if self.chart_type == "histogram":
            painter.setPen(QColor("#E2E8F0"))
            painter.drawText(QRectF(rect.center().x() - 24, rect.top() - 24, 48, 20), Qt.AlignCenter, "+M")
            painter.setPen(QColor("#7F8C9D"))
            painter.drawText(QRectF(rect.center().x() - 24, rect.bottom() + 4, 48, 20), Qt.AlignCenter, "-M")
            painter.drawText(QRectF(rect.left() - 28, rect.center().y() - 10, 24, 20), Qt.AlignRight | Qt.AlignVCenter, "-S")
            painter.drawText(QRectF(rect.right() + 4, rect.center().y() - 10, 24, 20), Qt.AlignLeft | Qt.AlignVCenter, "+S")
        else:
            painter.setPen(QColor("#26F7B4"))
            painter.drawText(QRectF(rect.center().x() - 40, rect.top() - 24, 80, 20), Qt.AlignCenter, "0° In-Phase")
            painter.setPen(QColor("#FF5A6E"))
            painter.drawText(QRectF(rect.center().x() - 45, rect.bottom() + 4, 90, 20), Qt.AlignCenter, "180° Inverted")
            painter.setPen(QColor("#7F8C9D"))
            painter.drawText(QRectF(rect.left() - 32, rect.center().y() - 10, 28, 20), Qt.AlignRight | Qt.AlignVCenter, "-90°")
            painter.drawText(QRectF(rect.right() + 4, rect.center().y() - 10, 28, 20), Qt.AlignLeft | Qt.AlignVCenter, "+90°")

    def _paint_histogram(self, painter: QPainter, to_px) -> None:
        if self._hist_angles is None or self._hist_radii is None:
            return
        angles = self._hist_angles
        radii = self._hist_radii
        pts = [to_px(float(r * np.sin(a)), float(r * np.cos(a))) for a, r in zip(angles, radii)]
        if not pts:
            return
        poly = QPolygonF(pts)
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()

        painter.setPen(QPen(QColor("#26F7B4"), 2.0))
        painter.setBrush(QColor(38, 247, 180, 48))
        painter.drawPath(path)

        # Tracking ball
        ang, rad = self._hist_ball
        ball_pt = to_px(float(rad * np.sin(ang)), float(rad * np.cos(ang)))
        origin = to_px(0.0, 0.0)
        painter.setPen(QPen(QColor(255, 255, 255, 110), 1.2))
        painter.drawLine(origin, ball_pt)
        painter.setPen(QPen(QColor("#26F7B4"), 2.0))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(ball_pt, 5.5, 5.5)

    def _paint_phase_wheel(self, painter: QPainter, to_px) -> None:
        if self._wheel_pts is None:
            return
        xw, yw, mags = self._wheel_pts
        if xw.size == 0:
            return
        painter.setPen(Qt.NoPen)
        for x, y, m in zip(xw, yw, mags):
            r = np.hypot(x, y)
            in_p = np.clip((y / max(r, 1e-6) + 1.0) * 0.5, 0.0, 1.0)
            col = QColor(
                int(255 * (1.0 - in_p) + 38 * in_p),
                int(90 * (1.0 - in_p) + 247 * in_p),
                int(110 * (1.0 - in_p) + 180 * in_p),
                int(70 + 180 * m),
            )
            painter.setBrush(col)
            rad = 2.0 + 3.8 * (m**1.4)
            painter.drawEllipse(to_px(float(x), float(y)), rad, rad)


class CorrelationMeterWidget(QWidget):
    """Adobe Audition-style -1.0 to +1.0 Phase Correlation horizontal meter."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self._corr: float = 1.0
        self._hold_min: float = 1.0
        self._hold_time: float = time.monotonic()

    def set_correlation(self, corr: float, is_streaming: bool = False) -> None:
        c = float(np.clip(corr, -1.0, 1.0))
        if is_streaming:
            self._corr = 0.65 * self._corr + 0.35 * c
            now = time.monotonic()
            if c < self._hold_min or (now - self._hold_time) > 1.6:
                self._hold_min = c
                self._hold_time = now
        else:
            self._corr = c
            self._hold_min = c
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#12151B"))

        pad_x = 18.0
        bar_y = 22.0
        bar_h = 14.0
        bar_w = max(20.0, self.width() - 2.0 * pad_x)
        bar_rect = QRectF(pad_x, bar_y, bar_w, bar_h)

        # Header labels
        painter.setFont(_ui_font(9, bold=True))
        painter.setPen(QColor("#8A98AB"))
        painter.drawText(QRectF(pad_x, 3, 200, 16), Qt.AlignLeft | Qt.AlignVCenter, "PHASE CORRELATION")

        painter.setFont(_mono_font(10, bold=True))
        val_color = QColor("#FF5A6E") if self._corr < 0.0 else (QColor("#F5B841") if self._corr < 0.35 else QColor("#26F7B4"))
        painter.setPen(val_color)
        painter.drawText(QRectF(pad_x, 2, bar_w, 17), Qt.AlignRight | Qt.AlignVCenter, f"r = {self._corr:+.3f}")

        # Gradient bar background
        grad = QLinearGradient(bar_rect.left(), 0, bar_rect.right(), 0)
        grad.setColorAt(0.0, QColor(255, 70, 90, 55))
        grad.setColorAt(0.5, QColor(245, 184, 65, 45))
        grad.setColorAt(1.0, QColor(38, 247, 180, 55))
        painter.setBrush(grad)
        painter.setPen(QPen(QColor("#2C3442"), 1.0))
        painter.drawRoundedRect(bar_rect, 3.0, 3.0)

        def val_to_x(v: float) -> float:
            return pad_x + (v + 1.0) * 0.5 * bar_w

        # Active fill from 0 to corr
        x0 = val_to_x(0.0)
        xc = val_to_x(self._corr)
        fill_rect = QRectF(min(x0, xc), bar_y + 2, abs(xc - x0), bar_h - 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(val_color.red(), val_color.green(), val_color.blue(), 190))
        painter.drawRect(fill_rect)

        # Min hold marker
        xh = val_to_x(self._hold_min)
        painter.setPen(QPen(QColor(255, 190, 110, 180), 1.5))
        painter.drawLine(QPointF(xh, bar_y + 1), QPointF(xh, bar_y + bar_h - 1))

        # Center 0 tick & live needle
        painter.setPen(QPen(QColor("#6E7B8F"), 1.0))
        painter.drawLine(QPointF(x0, bar_y - 2), QPointF(x0, bar_y + bar_h + 2))
        painter.setPen(QPen(QColor("#FFFFFF"), 2.2))
        painter.drawLine(QPointF(xc, bar_y - 2), QPointF(xc, bar_y + bar_h + 2))

        # Scale labels
        painter.setFont(_mono_font(8))
        painter.setPen(QColor("#788598"))
        for tick, txt in [(-1.0, "-1"), (-0.5, "-0.5"), (0.0, "0"), (0.5, "+0.5"), (1.0, "+1")]:
            xt = val_to_x(tick)
            painter.drawText(QRectF(xt - 16, bar_y + bar_h + 2, 32, 14), Qt.AlignCenter, txt)


class WaveformScrubberWidget(QWidget):
    """Interactive stereo waveform overview and playhead scrubber."""

    seekRequested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(68)
        self._total_samples: int = 0
        self._sample_rate: int = 48000
        self._playhead: int = 0
        self._window_samples: int = 4096
        self._env_l: Optional[np.ndarray] = None  # [cols, 2] min/max
        self._env_r: Optional[np.ndarray] = None
        self._dragging: bool = False

    def set_audio(self, stereo_lr: np.ndarray, sample_rate: int) -> None:
        self._total_samples = int(stereo_lr.shape[0])
        self._sample_rate = int(sample_rate)
        self._playhead = 0
        cols = 900
        if self._total_samples == 0:
            self._env_l = None
            self._env_r = None
            self.update()
            return

        if self._total_samples <= cols:
            self._env_l = np.column_stack([stereo_lr[:, 0], stereo_lr[:, 0]])
            self._env_r = np.column_stack([stereo_lr[:, 1], stereo_lr[:, 1]])
        else:
            edges = np.linspace(0, self._total_samples, cols + 1, dtype=np.int64)
            el = np.zeros((cols, 2), dtype=np.float32)
            er = np.zeros((cols, 2), dtype=np.float32)
            for i in range(cols):
                seg = stereo_lr[edges[i] : edges[i + 1]]
                if seg.size > 0:
                    el[i, 0] = float(np.min(seg[:, 0]))
                    el[i, 1] = float(np.max(seg[:, 0]))
                    er[i, 0] = float(np.min(seg[:, 1]))
                    er[i, 1] = float(np.max(seg[:, 1]))
            self._env_l = el
            self._env_r = er
        self.update()

    def set_playhead(self, sample_idx: int, window_samples: int) -> None:
        self._playhead = int(np.clip(sample_idx, 0, max(0, self._total_samples - 1)))
        self._window_samples = max(1, int(window_samples))
        self.update()

    def _emit_seek(self, x: float) -> None:
        if self._total_samples <= 0:
            return
        frac = float(np.clip(x / max(1.0, float(self.width())), 0.0, 1.0))
        self.seekRequested.emit(int(frac * max(0, self._total_samples - 1)))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._emit_seek(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._emit_seek(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0B0E13"))
        painter.setPen(QPen(QColor("#242B36"), 1.0))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        w = float(self.width())
        h = float(self.height())
        if self._env_l is None or self._env_r is None or self._total_samples <= 0:
            painter.setPen(QColor("#627084"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Load an audio file or select a test signal")
            return

        # Divider between L (top) and R (bottom)
        mid_y = h * 0.5
        painter.setPen(QPen(QColor("#1D232C"), 1.0))
        painter.drawLine(QPointF(0, mid_y), QPointF(w, mid_y))

        cols = self._env_l.shape[0]
        dx = w / cols
        qtr = h * 0.23
        yc_l = h * 0.25
        yc_r = h * 0.75

        painter.setPen(QPen(QColor(38, 247, 180, 175), max(1.0, dx)))
        for i in range(cols):
            x = (i + 0.5) * dx
            y0 = yc_l - float(self._env_l[i, 1]) * qtr
            y1 = yc_l - float(self._env_l[i, 0]) * qtr
            painter.drawLine(QPointF(x, y0), QPointF(x, y1))

        painter.setPen(QPen(QColor(56, 189, 248, 165), max(1.0, dx)))
        for i in range(cols):
            x = (i + 0.5) * dx
            y0 = yc_r - float(self._env_r[i, 1]) * qtr
            y1 = yc_r - float(self._env_r[i, 0]) * qtr
            painter.drawLine(QPointF(x, y0), QPointF(x, y1))

        # Active analysis window highlight & playhead
        px = (self._playhead / max(1, self._total_samples)) * w
        win_w = max(2.0, (min(self._window_samples, self._total_samples) / max(1, self._total_samples)) * w)
        painter.fillRect(QRectF(px, 1, min(win_w, w - px), h - 2), QColor(255, 255, 255, 26))
        painter.setPen(QPen(QColor("#FFFFFF"), 1.6))
        painter.drawLine(QPointF(px, 0), QPointF(px, h))

        # Time badge with dark backdrop pill
        cur_s = self._playhead / max(1, self._sample_rate)
        tot_s = self._total_samples / max(1, self._sample_rate)
        time_txt = f"{cur_s:05.2f}s / {tot_s:05.2f}s"
        pill = QRectF(w - 114, 4, 108, 17)
        painter.fillRect(pill, QColor(11, 14, 19, 210))
        painter.setFont(_mono_font(9))
        painter.setPen(QColor("#D9E2EC"))
        painter.drawText(pill.adjusted(0, 0, -4, 0), Qt.AlignRight | Qt.AlignVCenter, time_txt)


class PhaseAnalysisWindow(QMainWindow):
    """Main interactive Adobe Audition-style Phase Analysis application window."""

    def __init__(
        self,
        initial_file: Optional[Path] = None,
        initial_preset: str = "stereo_mix",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Mid / Side Phase Analysis  —  Audition-Style Stereo Scope")
        self.resize(1120, 790)
        self.setAcceptDrops(True)

        self._raw_data: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._stereo: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._sample_rate: int = 48000
        self._source_label: str = "Test Signal"

        # Transport & Audio IO state
        self._lock = threading.Lock()
        self._play_pos: int = 0
        self._is_playing: bool = False
        self._is_live_input: bool = False
        self._loop: bool = True
        self._invert_r: bool = False
        self._out_stream = None
        self._in_stream = None
        self._live_ring = np.zeros((16384, 2), dtype=np.float32)
        self._live_write_idx: int = 0

        self._build_ui()
        self._populate_input_devices()

        # 50 Hz UI refresh timer for smooth real-time animation
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._on_timer_tick)
        self._timer.start()

        # Keyboard shortcuts
        QShortcut(QKeySequence("Space"), self, activated=self._toggle_play)
        QShortcut(QKeySequence.Open, self, activated=self._open_file_dialog)
        QShortcut(QKeySequence.Save, self, activated=self._export_plot_dialog)

        if initial_file is not None:
            self.load_file(initial_file)
        else:
            self.load_preset(initial_preset)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        # Top toolbar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self._btn_open = QPushButton("Open Audio…")
        self._btn_open.setObjectName("primaryBtn")
        self._btn_open.clicked.connect(self._open_file_dialog)
        top_bar.addWidget(self._btn_open)

        top_bar.addWidget(QLabel("Test Signal:"))
        self._combo_preset = QComboBox()
        self._presets = [
            ("Stereo Mix (M/S Demo)", "stereo_mix"),
            ("Pure Mono (+1.0)", "mono"),
            ("Out-of-Phase (-1.0)", "out_of_phase"),
            ("Hard Left", "hard_left"),
            ("Hard Right", "hard_right"),
            ("90° Quadrature Circle", "quadrature"),
            ("Delayed Comb (0.6 ms)", "delayed_comb"),
        ]
        for label, _ in self._presets:
            self._combo_preset.addItem(label)
        self._combo_preset.activated.connect(self._on_preset_selected)
        top_bar.addWidget(self._combo_preset)

        top_bar.addSpacing(8)
        self._btn_play = QPushButton("▶  Play")
        self._btn_play.setCheckable(True)
        self._btn_play.clicked.connect(self._toggle_play)
        top_bar.addWidget(self._btn_play)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.clicked.connect(self._stop_playback)
        top_bar.addWidget(self._btn_stop)

        self._chk_loop = QCheckBox("Loop")
        self._chk_loop.setChecked(True)
        self._chk_loop.toggled.connect(lambda v: setattr(self, "_loop", bool(v)))
        top_bar.addWidget(self._chk_loop)

        top_bar.addStretch(1)

        self._btn_export = QPushButton("Export High-Res PNG…")
        self._btn_export.clicked.connect(self._export_plot_dialog)
        top_bar.addWidget(self._btn_export)

        outer.addLayout(top_bar)

        # Main workspace split: Left (Phase Scope + Correlation) | Right (Controls + Telemetry)
        body = QHBoxLayout()
        body.setSpacing(12)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        self._canvas = PhaseCanvasWidget()
        left_col.addWidget(self._canvas, stretch=1)

        self._corr_meter = CorrelationMeterWidget()
        left_col.addWidget(self._corr_meter)

        body.addLayout(left_col, stretch=3)

        # Right panel
        right_panel = QWidget()
        right_panel.setFixedWidth(308)
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(10)

        # Source info
        self._lbl_source = QLabel("Source: —")
        self._lbl_source.setWordWrap(True)
        self._lbl_source.setStyleSheet("color: #9FB0C3; font-weight: 600; padding: 2px 4px;")
        right_lay.addWidget(self._lbl_source)

        # Chart & Audition settings group
        grp_chart = QGroupBox("Phase Analysis Settings")
        grid = QGridLayout(grp_chart)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Chart Type:"), 0, 0)
        self._combo_chart = QComboBox()
        self._chart_opts: list[tuple[str, ChartType]] = [
            ("Mid / Side", "mid_side"),
            ("Lissajous (L/R)", "lissajous"),
            ("Histogram", "histogram"),
            ("Phase Wheel", "phase_wheel"),
        ]
        for label, _ in self._chart_opts:
            self._combo_chart.addItem(label)
        self._combo_chart.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self._combo_chart, 0, 1)

        grid.addWidget(QLabel("Scale:"), 1, 0)
        self._combo_scale = QComboBox()
        self._scale_opts: list[tuple[str, ScaleMode]] = [
            ("Linear", "linear"),
            ("Log Zoom (-36 dB)", "log_zoom"),
            ("Log Full (-72 dB)", "log_full"),
        ]
        for label, _ in self._scale_opts:
            self._combo_scale.addItem(label)
        self._combo_scale.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self._combo_scale, 1, 1)

        grid.addWidget(QLabel("Samples:"), 2, 0)
        self._combo_samples = QComboBox()
        self._sample_opts = [512, 1024, 2048, 4096, 8192, 16384, -1]
        for s in self._sample_opts:
            self._combo_samples.addItem("All (Full File)" if s < 0 else str(s))
        self._combo_samples.setCurrentIndex(3)  # 4096 default
        self._combo_samples.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self._combo_samples, 2, 1)

        grid.addWidget(QLabel("Left Ch:"), 3, 0)
        self._combo_ch_l = QComboBox()
        self._combo_ch_l.currentIndexChanged.connect(self._on_channel_changed)
        grid.addWidget(self._combo_ch_l, 3, 1)

        grid.addWidget(QLabel("Right Ch:"), 4, 0)
        self._combo_ch_r = QComboBox()
        self._combo_ch_r.currentIndexChanged.connect(self._on_channel_changed)
        grid.addWidget(self._combo_ch_r, 4, 1)

        grid.addWidget(QLabel("Color:"), 5, 0)
        self._combo_color = QComboBox()
        self._combo_color.addItem("Audition Phosphor", "phosphor")
        self._combo_color.addItem("Phase Zones (In/Out)", "phase_zones")
        self._combo_color.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self._combo_color, 5, 1)

        self._chk_normalize = QCheckBox("Normalize Display")
        self._chk_normalize.setChecked(True)
        self._chk_normalize.toggled.connect(self._on_settings_changed)
        grid.addWidget(self._chk_normalize, 6, 0, 1, 2)

        self._chk_connect = QCheckBox("Connect Trace Vectors")
        self._chk_connect.setChecked(True)
        self._chk_connect.toggled.connect(self._on_settings_changed)
        grid.addWidget(self._chk_connect, 7, 0, 1, 2)

        grid.addWidget(QLabel("Persistence:"), 8, 0)
        self._slider_persist = QSlider(Qt.Horizontal)
        self._slider_persist.setRange(0, 95)
        self._slider_persist.setValue(72)
        self._slider_persist.valueChanged.connect(self._on_settings_changed)
        grid.addWidget(self._slider_persist, 8, 1)

        self._btn_invert_r = QPushButton("Invert Right Polarity (180°)")
        self._btn_invert_r.setObjectName("dangerToggle")
        self._btn_invert_r.setCheckable(True)
        self._btn_invert_r.toggled.connect(self._on_invert_toggled)
        grid.addWidget(self._btn_invert_r, 9, 0, 1, 2)

        right_lay.addWidget(grp_chart)

        # Live input group
        grp_live = QGroupBox("Live Audio Input Monitor")
        live_lay = QVBoxLayout(grp_live)
        live_lay.setSpacing(6)
        self._combo_input_dev = QComboBox()
        live_lay.addWidget(self._combo_input_dev)
        self._btn_live = QPushButton("Start Live Input Monitor")
        self._btn_live.setCheckable(True)
        self._btn_live.clicked.connect(self._toggle_live_input)
        live_lay.addWidget(self._btn_live)
        right_lay.addWidget(grp_live)

        # Telemetry group
        grp_stats = QGroupBox("Stereo && M/S Telemetry")
        stat_lay = QGridLayout(grp_stats)
        stat_lay.setVerticalSpacing(5)
        self._stat_labels: dict[str, QLabel] = {}
        rows = [
            ("mid_rms", "Mid RMS / Peak:"),
            ("side_rms", "Side RMS / Peak:"),
            ("left_rms", "Left RMS / Peak:"),
            ("right_rms", "Right RMS / Peak:"),
            ("width", "Stereo Width:"),
            ("in_phase", "In-Phase Energy:"),
            ("balance", "L/R Balance:"),
        ]
        for idx, (key, title) in enumerate(rows):
            lbl_t = QLabel(title)
            lbl_t.setStyleSheet("color: #8796A8;")
            lbl_v = QLabel("—")
            lbl_v.setObjectName("monoReadout")
            lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            stat_lay.addWidget(lbl_t, idx, 0)
            stat_lay.addWidget(lbl_v, idx, 1)
            self._stat_labels[key] = lbl_v

        right_lay.addWidget(grp_stats)
        right_lay.addStretch(1)

        body.addWidget(right_panel)
        outer.addLayout(body, stretch=1)

        # Waveform overview scrubber at bottom
        self._scrubber = WaveformScrubberWidget()
        self._scrubber.seekRequested.connect(self._on_seek)
        outer.addWidget(self._scrubber)

        self.statusBar().showMessage("Ready — Drop a stereo/multichannel audio file or press Space to play.")

    def _populate_input_devices(self) -> None:
        self._combo_input_dev.clear()
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            default_in = sd.default.device[0]
            default_idx = 0
            for idx, dev in enumerate(devices):
                if int(dev["max_input_channels"]) >= 2:
                    self._combo_input_dev.addItem(f"{dev['name']} ({dev['max_input_channels']}ch)", idx)
                    if idx == default_in:
                        default_idx = self._combo_input_dev.count() - 1
            if self._combo_input_dev.count() == 0:
                for idx, dev in enumerate(devices):
                    if int(dev["max_input_channels"]) >= 1:
                        self._combo_input_dev.addItem(f"{dev['name']} (mono)", idx)
            if self._combo_input_dev.count() > 0:
                self._combo_input_dev.setCurrentIndex(default_idx)
            else:
                self._combo_input_dev.addItem("No input devices detected", -1)
                self._btn_live.setEnabled(False)
        except Exception:
            self._combo_input_dev.addItem("Audio input unavailable", -1)
            self._btn_live.setEnabled(False)

    def _refresh_channel_selectors(self, num_channels: int) -> None:
        self._combo_ch_l.blockSignals(True)
        self._combo_ch_r.blockSignals(True)
        self._combo_ch_l.clear()
        self._combo_ch_r.clear()
        for ch in range(max(1, num_channels)):
            tag = f"Ch {ch + 1}"
            if num_channels == 2:
                tag += " (L)" if ch == 0 else " (R)"
            self._combo_ch_l.addItem(tag, ch)
            self._combo_ch_r.addItem(tag, ch)
        self._combo_ch_l.setCurrentIndex(0)
        self._combo_ch_r.setCurrentIndex(min(1, max(0, num_channels - 1)))
        self._combo_ch_l.blockSignals(False)
        self._combo_ch_r.blockSignals(False)

    def _rebuild_stereo_buffer(self) -> None:
        li = self._combo_ch_l.currentData()
        ri = self._combo_ch_r.currentData()
        li = 0 if li is None else int(li)
        ri = min(1, max(0, self._raw_data.shape[1] - 1)) if ri is None else int(ri)
        stereo = extract_stereo_pair(self._raw_data, left_ch=li, right_ch=ri)
        if self._invert_r:
            stereo[:, 1] = -stereo[:, 1]
        with self._lock:
            self._stereo = stereo
            self._play_pos = min(self._play_pos, max(0, len(stereo) - 1))
        self._scrubber.set_audio(stereo, self._sample_rate)
        self._refresh_static_view()

    def load_file(self, path: str | Path) -> None:
        self._stop_all_streams()
        try:
            raw, sr = load_raw_channels(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could Not Open Audio", str(exc))
            return

        self._raw_data = raw
        self._sample_rate = sr
        self._source_label = f"{Path(path).name}  ({sr} Hz, {raw.shape[1]} ch, {len(raw)/sr:.2f}s)"
        self._lbl_source.setText(self._source_label)
        self._play_pos = 0
        self._refresh_channel_selectors(raw.shape[1])
        self._rebuild_stereo_buffer()
        self.statusBar().showMessage(f"Loaded {Path(path).name}")

    def load_preset(self, preset: str) -> None:
        self._stop_all_streams()
        lr, sr = generate_test_signal(preset=preset, duration_s=4.0, sample_rate=48000)
        self._raw_data = lr
        self._sample_rate = sr
        self._source_label = f"Test Signal: {preset}  ({sr} Hz, 2 ch, 4.00s)"
        self._lbl_source.setText(self._source_label)
        self._play_pos = 0
        self._refresh_channel_selectors(2)
        self._rebuild_stereo_buffer()
        self.statusBar().showMessage(f"Generated test signal '{preset}'")

    def _on_preset_selected(self, idx: int) -> None:
        if 0 <= idx < len(self._presets):
            self.load_preset(self._presets[idx][1])

    def _on_channel_changed(self) -> None:
        if self._raw_data.size > 0:
            self._rebuild_stereo_buffer()

    def _on_invert_toggled(self, checked: bool) -> None:
        self._invert_r = bool(checked)
        if self._raw_data.size > 0:
            self._rebuild_stereo_buffer()

    def _on_settings_changed(self) -> None:
        _, chart = self._chart_opts[self._combo_chart.currentIndex()]
        _, scale = self._scale_opts[self._combo_scale.currentIndex()]
        color = self._combo_color.currentData() or "phosphor"

        self._canvas.chart_type = chart
        self._canvas.scale_mode = scale
        self._canvas.normalize = self._chk_normalize.isChecked()
        self._canvas.connect_lines = self._chk_connect.isChecked()
        self._canvas.color_mode = color
        self._canvas.persistence = self._slider_persist.value() / 100.0
        self._canvas.reset_persistence()

        if not self._is_playing and not self._is_live_input:
            self._refresh_static_view()

    def _current_window_samples(self) -> int:
        val = self._sample_opts[self._combo_samples.currentIndex()]
        if val < 0:
            return max(1, len(self._stereo))
        return int(val)

    def _current_slice(self) -> np.ndarray:
        if self._stereo.size == 0:
            return self._stereo
        win = self._current_window_samples()
        n = len(self._stereo)
        if win >= n:
            return self._stereo
        start = int(np.clip(self._play_pos, 0, max(0, n - win)))
        return self._stereo[start : start + win]

    def _refresh_static_view(self) -> None:
        seg = self._current_slice()
        win = self._current_window_samples()
        self._scrubber.set_playhead(self._play_pos, win)
        self._update_readouts(seg, is_streaming=False)
        self._canvas.update_audio_frame(seg, self._sample_rate, is_streaming=False)

    def _update_readouts(self, seg: np.ndarray, is_streaming: bool) -> None:
        metrics: PhaseMetrics = compute_phase_metrics(seg)
        self._corr_meter.set_correlation(metrics.correlation, is_streaming=is_streaming)
        self._stat_labels["mid_rms"].setText(f"{metrics.mid_rms_db:+.1f} / {metrics.mid_peak_db:+.1f} dB")
        self._stat_labels["side_rms"].setText(f"{metrics.side_rms_db:+.1f} / {metrics.side_peak_db:+.1f} dB")
        self._stat_labels["left_rms"].setText(f"{metrics.left_rms_db:+.1f} / {metrics.left_peak_db:+.1f} dB")
        self._stat_labels["right_rms"].setText(f"{metrics.right_rms_db:+.1f} / {metrics.right_peak_db:+.1f} dB")
        self._stat_labels["width"].setText(f"{metrics.stereo_width_pct:.1f} %")
        self._stat_labels["in_phase"].setText(f"{metrics.in_phase_pct:.1f} %")
        self._stat_labels["balance"].setText(f"{metrics.balance_db:+.2f} dB")

    def _on_seek(self, sample_idx: int) -> None:
        with self._lock:
            self._play_pos = int(np.clip(sample_idx, 0, max(0, len(self._stereo) - 1)))
        if not self._is_playing:
            self._refresh_static_view()

    def _toggle_play(self) -> None:
        if self._is_live_input:
            self._stop_live_input()
        if self._is_playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._stereo.size == 0:
            self._btn_play.setChecked(False)
            return
        self._stop_out_stream()
        self._is_playing = True
        self._btn_play.setChecked(True)
        self._btn_play.setText("❚❚  Pause")
        self._canvas.reset_persistence()

        try:
            import sounddevice as sd

            def audio_cb(outdata, frames, time_info, status):
                with self._lock:
                    n = len(self._stereo)
                    if n == 0 or not self._is_playing:
                        outdata.fill(0.0)
                        return
                    pos = self._play_pos
                    remain = frames
                    write_off = 0
                    while remain > 0:
                        avail = n - pos
                        if avail <= 0:
                            if self._loop:
                                pos = 0
                                avail = n
                            else:
                                outdata[write_off:].fill(0.0)
                                self._is_playing = False
                                break
                        chunk = min(remain, avail)
                        outdata[write_off : write_off + chunk] = self._stereo[pos : pos + chunk]
                        pos += chunk
                        write_off += chunk
                        remain -= chunk
                    self._play_pos = pos if pos < n else 0

            self._out_stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=2,
                dtype="float32",
                blocksize=1024,
                callback=audio_cb,
            )
            self._out_stream.start()
            self.statusBar().showMessage("Playing audio…")
        except Exception as exc:
            # Visual-only playback fallback if audio output device is unavailable
            self._out_stream = None
            self.statusBar().showMessage(f"Visual preview (no audio output: {exc})")

    def _pause_playback(self) -> None:
        self._is_playing = False
        self._stop_out_stream()
        self._btn_play.setChecked(False)
        self._btn_play.setText("▶  Play")
        self._refresh_static_view()
        self.statusBar().showMessage("Paused")

    def _stop_playback(self) -> None:
        self._is_playing = False
        self._stop_out_stream()
        self._btn_play.setChecked(False)
        self._btn_play.setText("▶  Play")
        with self._lock:
            self._play_pos = 0
        self._refresh_static_view()
        self.statusBar().showMessage("Stopped")

    def _stop_out_stream(self) -> None:
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

    def _toggle_live_input(self) -> None:
        if self._is_live_input:
            self._stop_live_input()
        else:
            self._start_live_input()

    def _start_live_input(self) -> None:
        self._stop_playback()
        dev_idx = self._combo_input_dev.currentData()
        if dev_idx is None or int(dev_idx) < 0:
            self._btn_live.setChecked(False)
            return
        try:
            import sounddevice as sd

            dev_info = sd.query_devices(int(dev_idx))
            in_ch = min(2, int(dev_info["max_input_channels"]))
            sr = int(dev_info.get("default_samplerate", 48000)) or 48000
            self._sample_rate = sr
            self._live_ring.fill(0.0)
            self._live_write_idx = 0

            def in_cb(indata, frames, time_info, status):
                with self._lock:
                    ring_n = len(self._live_ring)
                    block = indata[:, :2] if indata.shape[1] >= 2 else np.column_stack([indata[:, 0], indata[:, 0]])
                    if self._invert_r:
                        block = block.copy()
                        block[:, 1] = -block[:, 1]
                    if frames >= ring_n:
                        self._live_ring[:] = block[-ring_n:]
                        self._live_write_idx = 0
                    else:
                        end = self._live_write_idx + frames
                        if end <= ring_n:
                            self._live_ring[self._live_write_idx : end] = block
                        else:
                            first = ring_n - self._live_write_idx
                            self._live_ring[self._live_write_idx :] = block[:first]
                            self._live_ring[: frames - first] = block[first:]
                        self._live_write_idx = end % ring_n

            self._in_stream = sd.InputStream(
                device=int(dev_idx),
                channels=in_ch,
                samplerate=sr,
                dtype="float32",
                blocksize=512,
                callback=in_cb,
            )
            self._in_stream.start()
            self._is_live_input = True
            self._btn_live.setChecked(True)
            self._btn_live.setText("Stop Live Input Monitor")
            self._canvas.reset_persistence()
            self.statusBar().showMessage(f"Monitoring live input: {dev_info['name']} ({sr} Hz)")
        except Exception as exc:
            self._btn_live.setChecked(False)
            QMessageBox.warning(self, "Live Input Error", f"Could not open audio input:\n{exc}")

    def _stop_live_input(self) -> None:
        self._is_live_input = False
        if self._in_stream is not None:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None
        self._btn_live.setChecked(False)
        self._btn_live.setText("Start Live Input Monitor")
        self._refresh_static_view()

    def _stop_all_streams(self) -> None:
        self._stop_playback()
        self._stop_live_input()

    def _on_timer_tick(self) -> None:
        if self._is_live_input:
            win = min(max(512, self._current_window_samples()), len(self._live_ring))
            with self._lock:
                idx = self._live_write_idx
                ring_n = len(self._live_ring)
                ordered = np.roll(self._live_ring, -idx, axis=0)
                seg = ordered[ring_n - win :].copy()
            self._update_readouts(seg, is_streaming=True)
            self._canvas.update_audio_frame(seg, self._sample_rate, is_streaming=True)
            return

        if not self._is_playing:
            return

        # If audio output stream wasn't opened, advance playhead by ~20ms
        if self._out_stream is None and len(self._stereo) > 0:
            reached_end = False
            with self._lock:
                step = int(self._sample_rate * 0.02)
                nxt = self._play_pos + step
                if nxt >= len(self._stereo):
                    if self._loop:
                        self._play_pos = 0
                    else:
                        self._play_pos = len(self._stereo) - 1
                        reached_end = True
                else:
                    self._play_pos = nxt
            if reached_end:
                self._pause_playback()
                return

        if not self._is_playing:
            self._pause_playback()
            return

        win = self._current_window_samples()
        with self._lock:
            n = len(self._stereo)
            if n == 0:
                return
            eff_win = min(win, n)
            pos = self._play_pos
            if pos + eff_win <= n:
                seg = self._stereo[pos : pos + eff_win].copy()
            else:
                part1 = self._stereo[pos:]
                part2 = self._stereo[: eff_win - len(part1)]
                seg = np.vstack([part1, part2])

        self._scrubber.set_playhead(pos, eff_win)
        self._update_readouts(seg, is_streaming=True)
        self._canvas.update_audio_frame(seg, self._sample_rate, is_streaming=True)

    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Stereo / Multichannel Audio File",
            str(Path.home()),
            "Audio Files (*.wav *.flac *.aiff *.aif *.ogg *.mp3 *.m4a *.caf);;All Files (*)",
        )
        if path:
            self.load_file(path)

    def _export_plot_dialog(self) -> None:
        if self._stereo.size == 0 and not self._is_live_input:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export High-Resolution Phase Analysis Plot",
            str(Path.home() / "mid_side_phase_plot.png"),
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf)",
        )
        if not out_path:
            return

        seg = self._current_slice() if not self._is_live_input else self._live_ring.copy()
        _, chart = self._chart_opts[self._combo_chart.currentIndex()]
        _, scale = self._scale_opts[self._combo_scale.currentIndex()]
        color = self._combo_color.currentData() or "phosphor"

        fig, _ = plot_phase_analysis(
            seg,
            sample_rate=self._sample_rate,
            chart_type=chart,
            scale_mode=scale,
            normalize=self._chk_normalize.isChecked(),
            color_mode=color,
            connect_lines=self._chk_connect.isChecked(),
            title=f"{self._source_label}",
            resolution=800,
        )
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt

        plt.close(fig)
        self.statusBar().showMessage(f"Saved plot to {out_path}")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local:
                self.load_file(local)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._stop_all_streams()
        self._canvas.reset_persistence()
        super().closeEvent(event)


def launch_gui(initial_file: Optional[Path] = None, preset: str = "stereo_mix") -> int:
    """Create the QApplication, open PhaseAnalysisWindow, and run the event loop."""
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK_STYLESHEET)
    win = PhaseAnalysisWindow(initial_file=initial_file, initial_preset=preset)
    win.show()
    return int(app.exec())
