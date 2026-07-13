"""PyQt6 GUI for VideoMAE fall detection: single video, batch, webcam."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from test_fall_model import (
    OUTPUT_DIR,
    classify_events,
    load_model,
    time_predictions,
)
from webcam_fall_app import DEFAULT_OUTPUT_DIR, WebcamSession

BATCH_OUTPUT_DIR = OUTPUT_DIR / "batch"
VIDEO_FILETYPES = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
CPU_SLOWDOWN_NOTE = (
    "CPU inference is roughly 15-20x slower than GPU. Realtime webcam "
    "detection will lag behind the live feed on CPU."
)
STYLESHEET = """
QWidget { font-size: 13px; }
QTabWidget::pane { border: 1px solid #3a3f4b; border-radius: 6px; top: -1px; }
QTabBar::tab {
    background: #2b2f38; color: #d7dae0; padding: 8px 18px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px;
}
QTabBar::tab:selected { background: #3d7cf4; color: white; }
QPushButton {
    background: #3d7cf4; color: white; border: none; border-radius: 5px;
    padding: 6px 16px;
}
QPushButton:disabled { background: #555b66; color: #aaa; }
QPushButton:hover:!disabled { background: #2f66d0; }
QTableWidget { gridline-color: #3a3f4b; }
QTextEdit, QTableWidget { border: 1px solid #3a3f4b; border-radius: 4px; }
"""


def write_report(
    video_path: Path,
    device: torch.device,
    threshold: float,
    whole: dict[str, Any],
    clip_df,
    result: dict[str, Any],
    out_dir: Path,
    timings: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    """Write per-video CSV/JSON reports, matching test_fall_model.py's format."""

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fall_clip_scores.csv"
    json_path = out_dir / "fall_events.json"
    clip_df.to_csv(csv_path, index=False)
    output = {
        "video": str(video_path),
        "device": str(device),
        "threshold": threshold,
        "whole_video_prediction": whole,
        "timings": timings,
        **result,
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    return csv_path, json_path


@dataclass
class AppSettings:
    device: str = "auto"
    threshold: float = 0.7


class SingleVideoWorker(QThread):
    done = pyqtSignal(dict, dict, Path, Path, object, dict)
    failed = pyqtSignal(Exception)

    def __init__(self, app: "FallDetectionGUI", video_path: Path):
        super().__init__()
        self.app = app
        self.video_path = video_path

    def run(self) -> None:
        try:
            model, processor, device = self.app.get_model(self.app.settings.device)
            threshold = self.app.settings.threshold
            whole, clip_df, timings = time_predictions(
                self.video_path,
                model,
                processor,
                device,
                threshold=threshold,
            )
            result = classify_events(clip_df, threshold=threshold)
            csv_path, json_path = write_report(
                self.video_path,
                device,
                threshold,
                whole,
                clip_df,
                result,
                OUTPUT_DIR,
                timings=timings,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(exc)
            return
        self.done.emit(whole, result, csv_path, json_path, device, timings)


class BatchWorker(QThread):
    load_failed = pyqtSignal(Exception)
    row_done = pyqtSignal(dict)
    status = pyqtSignal(str)
    finished_all = pyqtSignal(Path)

    def __init__(self, app: "FallDetectionGUI", video_paths: list[Path]):
        super().__init__()
        self.app = app
        self.video_paths = video_paths

    def run(self) -> None:
        try:
            model, processor, device = self.app.get_model(self.app.settings.device)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.load_failed.emit(exc)
            return

        threshold = self.app.settings.threshold
        summary_rows: list[dict[str, Any]] = []
        total = len(self.video_paths)
        for index, video_path in enumerate(self.video_paths, start=1):
            self.status.emit(f"Processing {index}/{total}: {video_path.name}")
            try:
                whole, clip_df, timings = time_predictions(
                    video_path,
                    model,
                    processor,
                    device,
                    threshold=threshold,
                )
                result = classify_events(clip_df, threshold=threshold)
                out_dir = BATCH_OUTPUT_DIR / video_path.stem
                write_report(
                    video_path,
                    device,
                    threshold,
                    whole,
                    clip_df,
                    result,
                    out_dir,
                    timings=timings,
                )
                max_score = (
                    float(clip_df["fall_down_score"].max()) if not clip_df.empty else 0.0
                )
                row = {
                    "video": str(video_path),
                    "decision": result["decision"],
                    "max_fall_down_score": max_score,
                    "num_events": len(result["events"]),
                    "output_dir": str(out_dir),
                    "total_seconds": timings["total_seconds"],
                }
            except Exception as exc:  # noqa: BLE001 - continue the batch
                row = {
                    "video": str(video_path),
                    "decision": "Error",
                    "max_fall_down_score": 0.0,
                    "num_events": 0,
                    "output_dir": f"Error: {exc}",
                    "total_seconds": 0.0,
                }
            summary_rows.append(row)
            self.row_done.emit(row)

        BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_csv = BATCH_OUTPUT_DIR / "summary.csv"
        with summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "video",
                    "decision",
                    "max_fall_down_score",
                    "num_events",
                    "output_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        self.finished_all.emit(summary_csv)


class WebcamOpenWorker(QThread):
    opened = pyqtSignal()
    failed = pyqtSignal(Exception)

    def __init__(self, session: WebcamSession):
        super().__init__()
        self.session = session

    def run(self) -> None:
        try:
            self.session.open()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(exc)
            return
        self.opened.emit()


class WebcamFinalizeWorker(QThread):
    def __init__(self, session: WebcamSession):
        super().__init__()
        self.session = session

    def run(self) -> None:
        self.session.finalize()
        self.session.close()


class WebcamStepWorker(QThread):
    """Runs the capture+inference loop off the GUI thread so a ~330ms
    inference call no longer freezes the video preview and UI once a second."""

    stepped = pyqtSignal(dict)

    def __init__(self, session: WebcamSession):
        super().__init__()
        self.session = session
        self._should_stop = False

    def request_stop(self) -> None:
        self._should_stop = True

    def run(self) -> None:
        while not self._should_stop:
            result = self.session.step()
            self.stepped.emit(result)
            if not result["ok"]:
                break


class FallDetectionGUI(QMainWindow):
    """Owns shared state (settings, model cache) across all tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fall Detection")
        self.resize(1000, 740)

        self.settings = AppSettings()
        self._model_cache: dict[str, tuple[Any, Any, torch.device]] = {}

        notebook = QTabWidget()
        self.setCentralWidget(notebook)

        self.single_tab = SingleVideoTab(self)
        self.batch_tab = BatchTab(self)
        self.webcam_tab = WebcamTab(self)
        self.settings_tab = SettingsTab(self)

        notebook.addTab(self.single_tab, "Single Video")
        notebook.addTab(self.batch_tab, "Batch")
        notebook.addTab(self.webcam_tab, "Webcam")
        notebook.addTab(self.settings_tab, "Settings")

    def get_model(self, device_pref: str) -> tuple[Any, Any, torch.device]:
        if device_pref not in self._model_cache:
            self._model_cache[device_pref] = load_model(device_pref)
        return self._model_cache[device_pref]

    def is_cpu(self) -> bool:
        if self.settings.device == "cpu":
            return True
        if self.settings.device == "cuda":
            return False
        return not torch.cuda.is_available()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.webcam_tab.running:
            self.webcam_tab.stop(reason="window closed")
        event.accept()


class SingleVideoTab(QWidget):
    def __init__(self, app: FallDetectionGUI):
        super().__init__()
        self.app = app
        self.video_path: Path | None = None
        self.worker: SingleVideoWorker | None = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.path_label = QLabel("No video selected")
        top.addWidget(self.path_label, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse)
        top.addWidget(browse_button)
        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)
        top.addWidget(self.run_button)
        layout.addLayout(top)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select a video", "", VIDEO_FILETYPES)
        if not path:
            return
        self.video_path = Path(path)
        self.path_label.setText(str(self.video_path))
        self.run_button.setEnabled(True)

    def _run(self) -> None:
        if self.video_path is None:
            return
        self.run_button.setEnabled(False)
        self.status_label.setText("Loading model / analyzing...")
        self.result_text.setPlainText("")
        self.worker = SingleVideoWorker(self.app, self.video_path)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_error)
        self.worker.start()

    def _on_error(self, exc: Exception) -> None:
        self.status_label.setText("Error.")
        QMessageBox.critical(self, "Single video analysis", str(exc))
        self.run_button.setEnabled(True)

    def _on_done(
        self,
        whole: dict[str, Any],
        result: dict[str, Any],
        csv_path: Path,
        json_path: Path,
        device: torch.device,
        timings: dict[str, float],
    ) -> None:
        self.status_label.setText(f"Done. Device: {device}")
        lines = [
            f"Decision: {result['decision']}",
            f"Whole-video prediction: {whole['top_label']} "
            f"(score={whole['top_score']:.4f})",
            f"Saved: {csv_path}",
            f"Saved: {json_path}",
            "Timings: "
            f"decode={timings['decode_seconds']:.2f}s, "
            f"whole_video={timings['whole_video_seconds']:.2f}s, "
            f"sliding_window={timings['sliding_window_seconds']:.2f}s "
            f"({timings['num_clips']} clips, "
            f"avg {timings['avg_clip_ms']:.0f}ms/clip), "
            f"total={timings['total_seconds']:.2f}s",
            "",
        ]
        if result["events"]:
            for event in result["events"]:
                lines.append(
                    f"- {event['event_type']}: "
                    f"{event['fall_start_seconds']:.1f}s-"
                    f"{event['fall_end_seconds']:.1f}s, "
                    f"max FallDown score={event['max_fall_down_score']:.4f}, "
                    f"recovery_observed={event['recovery_observed']}"
                )
        else:
            lines.append("No fall candidates detected.")
        self.result_text.setPlainText("\n".join(lines))
        self.run_button.setEnabled(True)


class BatchTab(QWidget):
    def __init__(self, app: FallDetectionGUI):
        super().__init__()
        self.app = app
        self.video_paths: list[Path] = []
        self.worker: BatchWorker | None = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        browse_button = QPushButton("Select videos...")
        browse_button.clicked.connect(self._browse)
        top.addWidget(browse_button)
        self.selected_label = QLabel("0 videos selected")
        top.addWidget(self.selected_label)
        self.run_button = QPushButton("Run batch")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)
        top.addWidget(self.run_button)
        top.addStretch(1)
        layout.addLayout(top)

        self.columns = ("video", "decision", "max_fall_down_score", "events", "output", "time")
        headers = ("Video", "Decision", "Max FallDown", "Events", "Output dir", "Time (s)")
        self.table = QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select videos", "", VIDEO_FILETYPES)
        if not paths:
            return
        self.video_paths = [Path(path) for path in paths]
        self.selected_label.setText(f"{len(self.video_paths)} videos selected")
        self.run_button.setEnabled(True)
        self.table.setRowCount(0)

    def _run(self) -> None:
        if not self.video_paths:
            return
        self.run_button.setEnabled(False)
        self.table.setRowCount(0)
        self.worker = BatchWorker(self.app, self.video_paths)
        self.worker.load_failed.connect(self._on_load_error)
        self.worker.status.connect(self.status_label.setText)
        self.worker.row_done.connect(self._add_row)
        self.worker.finished_all.connect(self._on_done)
        self.worker.start()

    def _add_row(self, row: dict[str, Any]) -> None:
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        values = (
            Path(row["video"]).name,
            row["decision"],
            f"{row['max_fall_down_score']:.4f}",
            str(row["num_events"]),
            row["output_dir"],
            f"{row.get('total_seconds', 0.0):.2f}",
        )
        for column, value in enumerate(values):
            self.table.setItem(row_index, column, QTableWidgetItem(value))

    def _on_load_error(self, exc: Exception) -> None:
        QMessageBox.critical(self, "Batch", f"Could not load model: {exc}")
        self.run_button.setEnabled(True)

    def _on_done(self, summary_csv: Path) -> None:
        self.status_label.setText(f"Done. Summary: {summary_csv}")
        self.run_button.setEnabled(True)


class WebcamTab(QWidget):
    def __init__(self, app: FallDetectionGUI):
        super().__init__()
        self.app = app
        self.session: WebcamSession | None = None
        self.running = False
        self.latencies: list[float] = []
        self.alerts_fired = 0
        self.start_time: float | None = None
        self._last_summary: dict[str, Any] | None = None
        self.open_worker: WebcamOpenWorker | None = None
        self.finalize_worker: WebcamFinalizeWorker | None = None
        self.step_worker: WebcamStepWorker | None = None
        self._pending_stop_reason: str | None = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera index:"))
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 8)
        top.addWidget(self.camera_index_spin)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start)
        top.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.stop())
        top.addWidget(self.stop_button)
        self.save_button = QPushButton("Save summary")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_summary)
        top.addWidget(self.save_button)
        top.addStretch(1)
        layout.addLayout(top)

        self.cpu_warning = QLabel("")
        self.cpu_warning.setStyleSheet(
            "color: white; background-color: #b00020; padding: 6px;"
        )
        self.cpu_warning.setVisible(False)
        layout.addWidget(self.cpu_warning)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        self.stats_label = QLabel("")
        layout.addWidget(self.stats_label)

        self.alert_label = QLabel("No alerts yet.")
        self.alert_label.setStyleSheet("color: #b00020;")
        layout.addWidget(self.alert_label)

    def _start(self) -> None:
        if self.running:
            return

        if self.app.is_cpu():
            self.cpu_warning.setText("Running on CPU - " + CPU_SLOWDOWN_NOTE)
            self.cpu_warning.setVisible(True)
        else:
            self.cpu_warning.setVisible(False)

        args = SimpleNamespace(
            camera_index=self.camera_index_spin.value(),
            device=self.app.settings.device,
            clip_seconds=2.0,
            stride_seconds=1.0,
            buffer_seconds=12.0,
            threshold=self.app.settings.threshold,
            consecutive=1,
            recovery_window_seconds=8.0,
            recovery_threshold=0.5,
            ground_threshold=0.5,
            output_dir=str(DEFAULT_OUTPUT_DIR),
        )
        self.session = WebcamSession(args)
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.stats_label.setText("Starting camera and loading model...")
        self.open_worker = WebcamOpenWorker(self.session)
        self.open_worker.opened.connect(self._open_succeeded)
        self.open_worker.failed.connect(self._open_failed)
        self.open_worker.start()

    def _open_failed(self, exc: Exception) -> None:
        QMessageBox.critical(self, "Webcam", f"Could not start webcam session: {exc}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.session = None

    def _open_succeeded(self) -> None:
        self.running = True
        self.latencies = []
        self.alerts_fired = 0
        self.start_time = time.monotonic()
        self.stop_button.setEnabled(True)
        self.step_worker = WebcamStepWorker(self.session)
        self.step_worker.stepped.connect(self._on_step)
        self.step_worker.start()

    def _on_step(self, result: dict[str, Any]) -> None:
        if not self.running or self.session is None:
            return
        if not result["ok"]:
            self.stop(reason="Camera frame read failed.")
            return
        self._render_frame(result["display_frame"])
        if result["latency_seconds"] is not None:
            self.latencies.append(result["latency_seconds"])
        if result["event"] is not None:
            self.alerts_fired += 1
            self.alert_label.setText(
                f"ALERT: {result['event'].get('event_type', 'Fall')} at "
                f"{result['stream_seconds']:.1f}s"
            )
        self._update_stats(result)

    def _render_frame(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            640,
            480,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def _update_stats(self, result: dict[str, Any]) -> None:
        latency = result["latency_seconds"]
        avg_latency = (
            sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        )
        lag_note = ""
        if self.session is not None and avg_latency > self.session.args.stride_seconds:
            lag_note = "   WARNING: falling behind real-time"
        latency_text = f"{latency:.2f}s" if latency is not None else "-"
        text = (
            f"Stream time: {result['stream_seconds']:.1f}s   "
            f"Last clip latency: {latency_text}\n"
            f"Avg latency: {avg_latency:.2f}s   "
            f"Clips processed: {len(self.latencies)}   "
            f"Alerts: {self.alerts_fired}{lag_note}"
        )
        self.stats_label.setText(text)

    def stop(self, reason: str | None = None) -> None:
        if not self.running:
            return
        self.running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._pending_stop_reason = reason
        if self.step_worker is not None and self.step_worker.isRunning():
            self.step_worker.request_stop()
            self.step_worker.finished.connect(self._on_step_worker_finished)
        else:
            self._on_step_worker_finished()

    def _on_step_worker_finished(self) -> None:
        session = self.session
        if session is not None:
            self.finalize_worker = WebcamFinalizeWorker(session)
            self.finalize_worker.start()
        self._show_summary(self._pending_stop_reason)

    def _show_summary(self, reason: str | None) -> None:
        if self.start_time is None or self.session is None:
            return
        runtime = time.monotonic() - self.start_time
        avg_latency = (
            sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        )
        summary = {
            "runtime_seconds": runtime,
            "clips_processed": len(self.latencies),
            "avg_latency_seconds": avg_latency,
            "min_latency_seconds": min(self.latencies) if self.latencies else 0.0,
            "max_latency_seconds": max(self.latencies) if self.latencies else 0.0,
            "alerts_fired": self.alerts_fired,
            "device": str(self.session.device),
            "reason_stopped": reason or "user stopped",
        }
        self._last_summary = summary
        self.save_button.setEnabled(True)
        message = (
            f"Runtime: {summary['runtime_seconds']:.1f}s\n"
            f"Clips processed: {summary['clips_processed']}\n"
            f"Avg latency: {summary['avg_latency_seconds']:.2f}s "
            f"(min {summary['min_latency_seconds']:.2f}s, "
            f"max {summary['max_latency_seconds']:.2f}s)\n"
            f"Alerts fired: {summary['alerts_fired']}\n"
            f"Device: {summary['device']}"
        )
        QMessageBox.information(self, "Session summary", message)

    def _save_summary(self) -> None:
        if self._last_summary is None:
            return
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DEFAULT_OUTPUT_DIR / f"session_summary_{timestamp}.json"
        path.write_text(json.dumps(self._last_summary, indent=2) + "\n")
        QMessageBox.information(self, "Saved", f"Summary saved to {path}")


class SettingsTab(QWidget):
    def __init__(self, app: FallDetectionGUI):
        super().__init__()
        self.app = app
        self._build_widgets()

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)

        grid.addWidget(QLabel("Device:"), 0, 0)
        self.device_group = QButtonGroup(self)
        for index, value in enumerate(("auto", "cuda", "cpu")):
            radio = QRadioButton(value)
            radio.setChecked(value == self.app.settings.device)
            radio.setProperty("device_value", value)
            self.device_group.addButton(radio)
            grid.addWidget(radio, 0, 1 + index)

        grid.addWidget(QLabel("Detection threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.app.settings.threshold)
        grid.addWidget(self.threshold_spin, 1, 1)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._apply)
        grid.addWidget(apply_button, 2, 0)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #b00020;")
        self.info_label.setWordWrap(True)
        grid.addWidget(self.info_label, 3, 0, 1, 4)

        layout.addStretch(1)
        self._refresh_info()

    def _apply(self) -> None:
        for button in self.device_group.buttons():
            if button.isChecked():
                self.app.settings.device = button.property("device_value")
                break
        self.app.settings.threshold = float(self.threshold_spin.value())
        self._refresh_info()

    def _refresh_info(self) -> None:
        if self.app.is_cpu():
            self.info_label.setText("Running on CPU. " + CPU_SLOWDOWN_NOTE)
        else:
            self.info_label.setText("")


def main() -> None:
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = FallDetectionGUI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
