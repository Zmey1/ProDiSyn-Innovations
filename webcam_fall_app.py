"""Realtime webcam fall detection with the existing VideoMAE model."""

from __future__ import annotations

import argparse
import os
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch

from test_fall_model import (
    RECOVERY_LABELS,
    load_model as load_video_model,
    predict_frames as predict_video_frames,
)


DEFAULT_OUTPUT_DIR = Path("outputs/realtime_events")
WINDOW_NAME = "Realtime VideoMAE Fall Detection"


def load_model(device_arg: str) -> tuple[Any, Any, torch.device]:
    """Load the existing model, selecting CPU if requested CUDA is unavailable."""

    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable. Using CPU.")
        device_arg = "cpu"
    try:
        return load_video_model(device_arg)
    except RuntimeError as exc:
        cuda_was_selected = (
            device_arg == "cuda"
            or (device_arg == "auto" and torch.cuda.is_available())
        )
        if not cuda_was_selected:
            raise
        print(f"CUDA model initialization failed: {exc}")
        print("Loading the model on CPU and continuing.")
        return load_video_model("cpu")


def open_camera(camera_source: int | str) -> cv2.VideoCapture:
    """Open a local webcam or network (RTSP/HTTP) stream, failing clearly.

    A non-numeric string source is treated as a network stream: it is opened
    through the FFmpeg backend with TCP transport and minimal buffering so the
    capture thread always reads the newest frame instead of a growing backlog of
    stale frames. Behavior for integer/local-index sources is unchanged.
    """

    is_stream = isinstance(camera_source, str) and not camera_source.isdigit()
    if is_stream:
        # Must be set before the capture is constructed. TCP avoids torn frames
        # (more reliable than UDP), and a low max_delay keeps latency down.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|max_delay;500000"
        )
        camera = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
    else:
        camera = cv2.VideoCapture(int(camera_source))

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera source {camera_source!r}")

    if is_stream:
        # Keep only the latest frame so inference never processes stale frames.
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def sample_frames_from_buffer(
    frame_buffer: deque[tuple[float, np.ndarray]],
    clip_seconds: float,
    num_frames: int = 16,
) -> tuple[list[np.ndarray], float, float, np.ndarray]:
    """Uniformly sample one RGB clip from the latest buffered webcam frames."""

    if not frame_buffer:
        raise ValueError("Cannot sample an empty frame buffer")

    clip_end_seconds = float(frame_buffer[-1][0])
    requested_start = max(0.0, clip_end_seconds - clip_seconds)
    eligible = [
        (timestamp, frame)
        for timestamp, frame in frame_buffer
        if timestamp >= requested_start
    ]
    if not eligible:
        eligible = [frame_buffer[-1]]

    timestamps = np.asarray(
        [item[0] for item in eligible],
        dtype=np.float64,
    )
    targets = np.linspace(
        requested_start,
        clip_end_seconds,
        num=num_frames,
    )
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = (
        np.abs(timestamps[left] - targets)
        <= np.abs(timestamps[right] - targets)
    )
    indices = np.where(choose_left, left, right)

    rgb_frames = [
        cv2.cvtColor(eligible[int(index)][1], cv2.COLOR_BGR2RGB)
        for index in indices
    ]
    middle_index = int(indices[len(indices) // 2])
    key_frame = eligible[middle_index][1].copy()
    return (
        rgb_frames,
        requested_start,
        clip_end_seconds,
        key_frame,
    )


def predict_frames(
    frames: list[np.ndarray],
    model: Any,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Run the existing VideoMAE frame-prediction helper."""

    return predict_video_frames(frames, model, processor, device)


def make_prediction_row(
    prediction: dict[str, Any],
    clip_start_seconds: float,
    clip_end_seconds: float,
) -> dict[str, Any]:
    """Create one realtime score-timeline row."""

    scores = prediction["scores"]
    fall_down_score = float(scores.get("FallDown", 0.0))
    lying_down_score = float(scores.get("LyingDown", 0.0))
    stand_up_score = float(scores.get("StandUp", 0.0))
    standing_score = float(scores.get("Standing", 0.0))
    walking_score = float(scores.get("Walking", 0.0))

    return {
        "timestamp": float(clip_end_seconds),
        "clip_start_seconds": float(clip_start_seconds),
        "clip_end_seconds": float(clip_end_seconds),
        "top_label": str(prediction["top_label"]),
        "top_score": float(prediction["top_score"]),
        "fall_down_score": fall_down_score,
        "lying_down_score": lying_down_score,
        "stand_up_score": stand_up_score,
        "standing_score": standing_score,
        "walking_score": walking_score,
        "ground_score": max(fall_down_score, lying_down_score),
        "recovery_score": max(
            stand_up_score,
            standing_score,
            walking_score,
        ),
    }


def new_event_state() -> dict[str, Any]:
    """Return an empty event state."""

    return {
        "phase": "no_active_event",
        "consecutive_fall_predictions": 0,
        "fall_start_seconds": None,
        "fall_end_seconds": None,
        "max_fall_down_score": 0.0,
        "peak_frame": None,
        "recovery_deadline_seconds": None,
        "recovery_observed": False,
        "sustained_ground_observed": False,
        "recovery_evidence_label": None,
        "recovery_evidence_start_seconds": None,
        "recovery_evidence_end_seconds": None,
    }


def _reset_event_state(state: dict[str, Any]) -> None:
    state.clear()
    state.update(new_event_state())


def _recovery_evidence_label(row: dict[str, Any]) -> str:
    if row["top_label"] in RECOVERY_LABELS:
        return str(row["top_label"])
    score_columns = {
        "StandUp": "stand_up_score",
        "Standing": "standing_score",
        "Walking": "walking_score",
    }
    return max(
        score_columns,
        key=lambda label: float(row[score_columns[label]]),
    )


def _finalize_event(
    state: dict[str, Any],
    row: dict[str, Any],
    event_type: str,
    decision_reason: str,
) -> dict[str, Any]:
    event = {
        "event_type": event_type,
        "wall_clock_timestamp": datetime.now().astimezone().isoformat(),
        "stream_seconds": float(row["timestamp"]),
        "fall_start_seconds": float(state["fall_start_seconds"]),
        "fall_end_seconds": float(state["fall_end_seconds"]),
        "max_fall_down_score": float(state["max_fall_down_score"]),
        "recovery_observed": bool(state["recovery_observed"]),
        "sustained_ground_observed": bool(
            state["sustained_ground_observed"]
        ),
        "recovery_evidence_label": state["recovery_evidence_label"],
        "recovery_evidence_start_seconds": state[
            "recovery_evidence_start_seconds"
        ],
        "recovery_evidence_end_seconds": state[
            "recovery_evidence_end_seconds"
        ],
        "decision_reason": decision_reason,
        "_peak_frame": state["peak_frame"],
    }
    state["phase"] = "event_finalized"
    state["last_event_type"] = event_type
    return event


def _check_post_fall(
    row: dict[str, Any],
    state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    recovery_observed = (
        row["top_label"] in RECOVERY_LABELS
        or row["recovery_score"] >= args.recovery_threshold
    )
    if recovery_observed:
        state["recovery_observed"] = True
        state["recovery_evidence_label"] = _recovery_evidence_label(row)
        state["recovery_evidence_start_seconds"] = float(
            row["clip_start_seconds"]
        )
        state["recovery_evidence_end_seconds"] = float(
            row["clip_end_seconds"]
        )
        return _finalize_event(
            state,
            row,
            "Partial Fall",
            "FallDown detected, followed by "
            "StandUp/Standing/Walking within recovery window.",
        )

    if row["ground_score"] >= args.ground_threshold:
        state["sustained_ground_observed"] = True

    if row["timestamp"] < state["recovery_deadline_seconds"]:
        return None

    if state["sustained_ground_observed"]:
        return _finalize_event(
            state,
            row,
            "Full Fall",
            "FallDown detected with no recovery and continued "
            "FallDown/LyingDown evidence during the recovery window.",
        )
    return _finalize_event(
        state,
        row,
        "Fall Detected, Recovery Unknown",
        "FallDown detected, but the recovery window contained no clear "
        "recovery or sustained ground evidence.",
    )


def update_event_state(
    prediction_row: dict[str, Any],
    state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Advance the simple fall-candidate and recovery state machine."""

    if state["phase"] == "event_finalized":
        _reset_event_state(state)

    above_fall_threshold = (
        prediction_row["fall_down_score"] >= args.threshold
    )

    if state["phase"] == "no_active_event":
        if not above_fall_threshold:
            return None
        state.update(
            {
                "phase": "fall_candidate_active",
                "consecutive_fall_predictions": 1,
                "fall_start_seconds": float(
                    prediction_row["clip_start_seconds"]
                ),
                "fall_end_seconds": float(
                    prediction_row["clip_end_seconds"]
                ),
                "max_fall_down_score": float(
                    prediction_row["fall_down_score"]
                ),
                "peak_frame": prediction_row.get("_key_frame"),
            }
        )
        return None

    if state["phase"] == "fall_candidate_active":
        if above_fall_threshold:
            state["consecutive_fall_predictions"] += 1
            state["fall_end_seconds"] = float(
                prediction_row["clip_end_seconds"]
            )
            if (
                prediction_row["fall_down_score"]
                > state["max_fall_down_score"]
            ):
                state["max_fall_down_score"] = float(
                    prediction_row["fall_down_score"]
                )
                state["peak_frame"] = prediction_row.get("_key_frame")
            return None

        if (
            state["consecutive_fall_predictions"]
            < args.consecutive
        ):
            _reset_event_state(state)
            return None

        state["phase"] = "waiting_for_recovery"
        state["recovery_deadline_seconds"] = (
            float(state["fall_end_seconds"])
            + args.recovery_window_seconds
        )
        print(
            "Fall candidate detected: "
            f"{state['fall_start_seconds']:.1f}s to "
            f"{state['fall_end_seconds']:.1f}s, "
            "max FallDown score="
            f"{state['max_fall_down_score']:.4f}"
        )
        return _check_post_fall(prediction_row, state, args)

    if state["phase"] == "waiting_for_recovery":
        return _check_post_fall(prediction_row, state, args)

    return None


def _unique_event_stem(output_dir: Path) -> str:
    base = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = base
    suffix = 1
    while any(
        (output_dir / f"{candidate}{extension}").exists()
        for extension in (".json", ".jpg", ".mp4")
    ):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def save_event_outputs(
    event: dict[str, Any],
    frame_buffer: deque[tuple[float, np.ndarray]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Save an event JSON record, peak image, and available buffered video."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    stem = _unique_event_stem(output_path)
    image_path = output_path / f"{stem}.jpg"
    clip_path = output_path / f"{stem}.mp4"
    json_path = output_path / f"{stem}.json"

    peak_frame = event.pop("_peak_frame", None)
    if peak_frame is None and frame_buffer:
        peak_frame = frame_buffer[-1][1]
    saved_image_path: str | None = None
    if peak_frame is not None and cv2.imwrite(str(image_path), peak_frame):
        saved_image_path = str(image_path)

    clip_start = float(event["fall_start_seconds"]) - 2.0
    clip_end = float(event["stream_seconds"])
    clip_frames = [
        (timestamp, frame)
        for timestamp, frame in frame_buffer
        if clip_start <= timestamp <= clip_end
    ]
    saved_clip_path: str | None = None
    if len(clip_frames) >= 2:
        duration = clip_frames[-1][0] - clip_frames[0][0]
        fps = (len(clip_frames) - 1) / duration if duration > 0 else 10.0
        fps = float(np.clip(fps, 5.0, 60.0))
        height, width = clip_frames[0][1].shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if writer.isOpened():
            try:
                for _, frame in clip_frames:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
            finally:
                writer.release()
            saved_clip_path = str(clip_path)
        else:
            writer.release()

    event["image_path"] = saved_image_path
    event["clip_path"] = saved_clip_path
    json_path.write_text(json.dumps(event, indent=2) + "\n")
    event["json_path"] = str(json_path)
    return event


def draw_overlay(
    frame: np.ndarray,
    current_prediction: dict[str, Any] | None,
    event_state: dict[str, Any],
    latest_alert: dict[str, Any] | None,
    hint_text: str = "Press q to quit",
) -> np.ndarray:
    """Draw current prediction, state, and recent alert on a webcam frame.

    ``hint_text`` is the small bottom-left hint line; callers replaying a file
    in a GUI (where there is no ``q`` key) can pass ``""`` to hide it.
    """

    output = frame.copy()
    if current_prediction is None:
        prediction_text = "Model: warming up"
    else:
        prediction_text = (
            f"Label: {current_prediction['top_label']}  "
            f"FallDown: {current_prediction['fall_down_score']:.3f}"
        )
    state_text = f"State: {event_state['phase'].replace('_', ' ')}"

    cv2.putText(
        output,
        prediction_text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        state_text,
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if latest_alert is not None:
        cv2.putText(
            output,
            f"ALERT: {latest_alert['event_type']}",
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if hint_text:
        cv2.putText(
            output,
            hint_text,
            (20, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return output


def _alert(event: dict[str, Any]) -> None:
    print("\a")
    print(
        f"ALERT: {event['event_type']} at "
        f"{event['stream_seconds']:.1f}s"
    )
    print(f"Reason: {event['decision_reason']}")
    print(f"JSON: {event['json_path']}")


def _finalize_on_quit(
    state: dict[str, Any],
    stream_seconds: float,
    consecutive_required: int,
) -> dict[str, Any] | None:
    confirmed_candidate = (
        state["phase"] == "waiting_for_recovery"
        or (
            state["phase"] == "fall_candidate_active"
            and state["consecutive_fall_predictions"]
            >= consecutive_required
        )
    )
    if not confirmed_candidate:
        return None
    row = {
        "timestamp": stream_seconds,
    }
    return _finalize_event(
        state,
        row,
        "Fall Detected, Recovery Unknown",
        "The webcam session ended before post-fall recovery could be "
        "determined.",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime webcam fall detection with VideoMAE."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--camera-url",
        default=None,
        help="RTSP/HTTP stream URL. Takes precedence over --camera-index.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--buffer-seconds", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--consecutive", type=int, default=1)
    parser.add_argument(
        "--recovery-window-seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument("--recovery-threshold", type=float, default=0.5)
    parser.add_argument("--ground-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "clip_seconds",
        "stride_seconds",
        "buffer_seconds",
        "recovery_window_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.buffer_seconds < args.clip_seconds:
        raise ValueError("--buffer-seconds must be at least --clip-seconds")
    if args.consecutive <= 0:
        raise ValueError("--consecutive must be positive")
    for name in (
        "threshold",
        "recovery_threshold",
        "ground_threshold",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be between 0 and 1"
            )


class WebcamSession:
    """Drive the webcam fall-detection loop one frame at a time.

    Holds all the state that main()'s while-loop used to keep in locals, so a
    caller (CLI or GUI) can call step() repeatedly instead of owning the loop
    itself. Behavior is identical to the original inline loop.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.model: Any = None
        self.processor: Any = None
        self.device: torch.device | None = None
        self.camera: cv2.VideoCapture | None = None
        self.frame_buffer: deque[tuple[float, np.ndarray]] = deque()
        self.event_state = new_event_state()
        self.current_prediction: dict[str, Any] | None = None
        self.latest_alert: dict[str, Any] | None = None
        self.alert_expires_at = 0.0
        self.stream_started_at = 0.0
        self.last_inference_at = -float("inf")
        self.last_stream_seconds = 0.0
        self.last_latency_seconds: float | None = None
        self.buffer_lock = threading.Lock()

    def open(self) -> None:
        """Load the model and open the camera."""

        self.model, self.processor, self.device = load_model(self.args.device)
        print(f"Selected device: {self.device}")
        source = getattr(self.args, "camera_url", None) or self.args.camera_index
        self.camera = open_camera(source)
        self.stream_started_at = time.monotonic()

    def capture_step(self) -> dict[str, Any]:
        """Read one camera frame and update the buffer. Never blocks on inference."""

        ok, frame = self.camera.read()
        if not ok:
            return {"ok": False}

        now = time.monotonic()
        stream_seconds = now - self.stream_started_at
        self.last_stream_seconds = stream_seconds
        with self.buffer_lock:
            self.frame_buffer.append((stream_seconds, frame.copy()))
            while (
                self.frame_buffer
                and stream_seconds - self.frame_buffer[0][0]
                > self.args.buffer_seconds
            ):
                self.frame_buffer.popleft()

        if (
            self.latest_alert is not None
            and time.monotonic() > self.alert_expires_at
        ):
            self.latest_alert = None

        display_frame = draw_overlay(
            frame,
            self.current_prediction,
            self.event_state,
            self.latest_alert,
        )

        return {
            "ok": True,
            "frame": frame,
            "display_frame": display_frame,
            "stream_seconds": stream_seconds,
        }

    def try_infer(self) -> dict[str, Any] | None:
        """Run one inference cycle if due. Returns None immediately if not due yet.

        Never queues backlog: always samples from whatever the buffer currently
        holds, so a slow (e.g. CPU) inference pass simply results in a longer
        effective interval between cycles rather than falling further behind.
        """

        with self.buffer_lock:
            buffer_duration = (
                self.frame_buffer[-1][0] - self.frame_buffer[0][0]
                if len(self.frame_buffer) >= 2
                else 0.0
            )
            stream_seconds = self.last_stream_seconds
            inference_due = (
                buffer_duration >= self.args.clip_seconds
                and stream_seconds - self.last_inference_at
                >= self.args.stride_seconds
            )
            buffer_snapshot = list(self.frame_buffer) if inference_due else None

        if not inference_due or not buffer_snapshot:
            return None

        (
            sampled_frames,
            clip_start,
            clip_end,
            key_frame,
        ) = sample_frames_from_buffer(
            buffer_snapshot,
            self.args.clip_seconds,
            num_frames=16,
        )
        inference_start = time.monotonic()
        try:
            prediction = predict_frames(
                sampled_frames,
                self.model,
                self.processor,
                self.device,
            )
        except RuntimeError as exc:
            if self.device.type != "cuda":
                raise
            print(f"CUDA inference failed: {exc}")
            print("Moving the model to CPU and continuing.")
            self.device = torch.device("cpu")
            try:
                self.model.to(self.device)
            except RuntimeError:
                self.model, self.processor, self.device = load_model("cpu")
            self.model.eval()
            prediction = predict_frames(
                sampled_frames,
                self.model,
                self.processor,
                self.device,
            )
        self.last_latency_seconds = time.monotonic() - inference_start

        self.current_prediction = make_prediction_row(
            prediction,
            clip_start,
            clip_end,
        )
        self.current_prediction["_key_frame"] = key_frame
        self.last_inference_at = stream_seconds
        event = update_event_state(
            self.current_prediction,
            self.event_state,
            self.args,
        )
        saved_event: dict[str, Any] | None = None
        if event is not None:
            saved_event = save_event_outputs(
                event,
                buffer_snapshot,
                self.output_dir,
            )
            _alert(saved_event)
            self.latest_alert = saved_event
            self.alert_expires_at = time.monotonic() + 5.0

        return {
            "inference_ran": True,
            "prediction": self.current_prediction,
            "stream_seconds": stream_seconds,
            "latency_seconds": self.last_latency_seconds,
            "event": saved_event,
        }

    def step(self) -> dict[str, Any]:
        """Read one frame and run inference if due. Used by the single-threaded CLI."""

        result = self.capture_step()
        if not result["ok"]:
            return result

        inference_result = self.try_infer()
        result["inference_ran"] = inference_result is not None
        result["prediction"] = (
            inference_result["prediction"] if inference_result else self.current_prediction
        )
        result["latency_seconds"] = (
            inference_result["latency_seconds"] if inference_result else None
        )
        result["event"] = inference_result["event"] if inference_result else None
        if inference_result is not None:
            result["display_frame"] = draw_overlay(
                result["frame"],
                self.current_prediction,
                self.event_state,
                self.latest_alert,
            )
        return result

    def finalize(self) -> dict[str, Any] | None:
        """Flush a pending fall candidate at session end, if any."""

        pending_event = _finalize_on_quit(
            self.event_state,
            self.last_stream_seconds,
            self.args.consecutive,
        )
        if pending_event is None:
            return None
        saved_event = save_event_outputs(
            pending_event,
            self.frame_buffer,
            self.output_dir,
        )
        _alert(saved_event)
        return saved_event

    def close(self) -> None:
        """Release the camera."""

        if self.camera is not None:
            self.camera.release()
        self.camera = None


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    session = WebcamSession(args)
    session.open()
    try:
        while True:
            result = session.step()
            if not result["ok"]:
                print("Camera frame read failed; stopping.")
                break
            cv2.imshow(WINDOW_NAME, result["display_frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        session.finalize()
        session.close()
        cv2.destroyAllWindows()
        print("Webcam app stopped cleanly.")


if __name__ == "__main__":
    main()
