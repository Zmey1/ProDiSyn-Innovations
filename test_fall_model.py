"""Minimal local VideoMAE fall detection for one video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# This project uses PyTorch only. Avoid importing an unrelated TensorFlow
# installation through Transformers.
os.environ.setdefault("USE_TF", "0")

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification


LOCAL_MODEL_NAME = "local-videomae-fall-model"
BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"
RECOVERY_LABELS = {"StandUp", "Standing", "Walking"}


def _model_source() -> str:
    """Return the required local model directory."""

    required_files = (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    )
    missing = [
        name for name in required_files
        if not (LOCAL_MODEL_DIR / name).is_file()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Local model is incomplete at {LOCAL_MODEL_DIR}. "
            f"Missing: {missing_list}. Run 'git lfs pull' in the repository."
        )
    return str(LOCAL_MODEL_DIR)


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    device_preference: str = "auto",
) -> tuple[Any, Any, torch.device]:
    """Load the image processor and VideoMAE classifier."""

    source = _model_source()
    device = _select_device(device_preference)

    print(f"Loading model from: {source}")
    processor = AutoImageProcessor.from_pretrained(
        source,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForVideoClassification.from_pretrained(
        source,
        local_files_only=True,
    )

    model.to(device)
    model.eval()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return model, processor, device


def _open_video(video_path: str | Path) -> tuple[cv2.VideoCapture, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(
            f"Video has invalid metadata: frame_count={frame_count}, fps={fps}"
        )
    return capture, frame_count, fps


def _resize_shortest_edge(frame: np.ndarray, target_shortest_edge: int) -> np.ndarray:
    """Downscale so the shorter side equals target_shortest_edge (no-op if already smaller).

    Matches the processor's own resize target (preprocessor_config.json:
    size.shortest_edge=224), so this produces the same crop the processor
    would compute from the raw frame, just without decoding/resizing at
    full 4K first.
    """

    height, width = frame.shape[:2]
    shortest_edge = min(height, width)
    if shortest_edge <= target_shortest_edge:
        return frame
    scale = target_shortest_edge / shortest_edge
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


_FFMPEG_HWACCEL_AVAILABLE: bool | None = None


def _ffmpeg_hwaccel_available() -> bool:
    """Cache whether ffmpeg with CUDA/NVDEC hwaccel is usable on this machine."""

    global _FFMPEG_HWACCEL_AVAILABLE
    if _FFMPEG_HWACCEL_AVAILABLE is not None:
        return _FFMPEG_HWACCEL_AVAILABLE
    if shutil.which("ffmpeg") is None:
        _FFMPEG_HWACCEL_AVAILABLE = False
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        _FFMPEG_HWACCEL_AVAILABLE = "cuda" in result.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        _FFMPEG_HWACCEL_AVAILABLE = False
    return _FFMPEG_HWACCEL_AVAILABLE


def _decode_all_frames_ffmpeg(
    video_path: str | Path,
    target_shortest_edge: int,
) -> list[np.ndarray] | None:
    """Decode via ffmpeg's NVDEC hwaccel, scaling at decode time. None on any failure."""

    probe = cv2.VideoCapture(str(video_path))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    if width <= 0 or height <= 0:
        return None

    shortest_edge = min(width, height)
    if shortest_edge > target_shortest_edge:
        scale = target_shortest_edge / shortest_edge
        out_width = round(width * scale)
        out_height = round(height * scale)
    else:
        out_width, out_height = width, height
    out_width -= out_width % 2
    out_height -= out_height % 2
    if out_width <= 0 or out_height <= 0:
        return None

    frame_bytes = out_width * out_height * 3
    command = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-i", str(video_path),
        "-vf", f"scale={out_width}:{out_height}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-",
    ]
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except OSError:
        return None

    frames: list[np.ndarray] = []
    try:
        assert process.stdout is not None
        while True:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            frame = np.frombuffer(buffer, dtype=np.uint8).reshape(
                (out_height, out_width, 3)
            )
            frames.append(frame.copy())
    finally:
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return None

    if process.returncode != 0 or not frames:
        return None
    return frames


def _decode_all_frames_cv2(
    video_path: str | Path,
    target_shortest_edge: int | None = None,
) -> list[np.ndarray]:
    """Decode every frame sequentially (no seeking), downscaling as it goes."""

    capture, frame_count, _ = _open_video(video_path)
    frames: list[np.ndarray] = []
    try:
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            if target_shortest_edge is not None:
                frame = _resize_shortest_edge(frame, target_shortest_edge)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Could not decode any frames from {video_path}")
    return frames


def decode_video_frames(
    video_path: str | Path,
    target_shortest_edge: int | None = None,
) -> list[np.ndarray]:
    """Decode every frame of the video once, downscaled to target_shortest_edge.

    Callers that need multiple analysis passes over the same video (whole-
    video + sliding-window) should decode once here and pass the result to
    both, instead of letting each pass re-decode overlapping regions.
    """

    if target_shortest_edge is not None and _ffmpeg_hwaccel_available():
        frames = _decode_all_frames_ffmpeg(video_path, target_shortest_edge)
        if frames is not None:
            return frames
    return _decode_all_frames_cv2(video_path, target_shortest_edge)


def _select_indices(
    num_available: int,
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    last_frame = num_available - 1
    start_frame = int(np.clip(start_frame, 0, last_frame))
    end_frame = int(np.clip(end_frame, start_frame, last_frame))
    return np.linspace(start_frame, end_frame, num=num_frames).round().astype(int)


def sample_uniform_frames(
    video_path: str | Path,
    num_frames: int = 16,
    target_shortest_edge: int | None = None,
    frames: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Sample frames uniformly across the complete video.

    Pass a pre-decoded `frames` list (from `decode_video_frames`) to avoid
    re-decoding when the caller already decoded this video for another pass.
    """

    if frames is None:
        frames = decode_video_frames(video_path, target_shortest_edge)
    indices = _select_indices(len(frames), 0, len(frames) - 1, num_frames)
    return [frames[i] for i in indices]


def _processor_shortest_edge(processor: Any) -> int:
    """Read the processor's configured resize target instead of a guessed constant."""

    size = getattr(processor, "size", None) or {}
    return int(size.get("shortest_edge", 224))


def predict_frames(
    frames: list[np.ndarray],
    model: Any,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Return class probabilities and the top class for one 16-frame clip."""

    expected_frames = int(getattr(model.config, "num_frames", 16))
    if len(frames) != expected_frames:
        raise ValueError(
            f"Model expects {expected_frames} frames, received {len(frames)}"
        )

    inputs = processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        logits = model(pixel_values=pixel_values).logits
        probabilities = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()

    labels = {
        int(index): label for index, label in model.config.id2label.items()
    }
    top_index = int(np.argmax(probabilities))
    scores = {
        labels[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }
    return {
        "top_label": labels[top_index],
        "top_score": float(probabilities[top_index]),
        "scores": scores,
        "fall_down_score": float(scores.get("FallDown", 0.0)),
    }


def predict_video_whole(
    video_path: str | Path,
    model: Any,
    processor: Any,
    device: torch.device,
    frames: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Predict one clip sampled uniformly across the entire video.

    Pass a pre-decoded `frames` list (from `decode_video_frames`) when the
    caller is also running `analyze_video_sliding` on the same video, to
    avoid decoding it twice.
    """

    num_frames = int(getattr(model.config, "num_frames", 16))
    sampled = sample_uniform_frames(
        video_path,
        num_frames=num_frames,
        target_shortest_edge=_processor_shortest_edge(processor),
        frames=frames,
    )
    return predict_frames(sampled, model, processor, device)


def classify_events(
    df: pd.DataFrame,
    threshold: float = 0.7,
    consecutive_required: int = 2,
    recovery_window_seconds: float = 8.0,
    recovery_threshold: float = 0.5,
    ground_threshold: float = 0.5,
) -> dict[str, Any]:
    """Detect fall candidates, then classify their post-fall state."""

    if consecutive_required <= 0:
        raise ValueError("consecutive_required must be positive")
    if recovery_window_seconds <= 0:
        raise ValueError("recovery_window_seconds must be positive")
    for name, value in (
        ("threshold", threshold),
        ("recovery_threshold", recovery_threshold),
        ("ground_threshold", ground_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if df.empty:
        return {"decision": "Normal", "events": []}

    above_threshold = df["fall_down_score"].to_numpy() >= threshold
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for index, is_above in enumerate(above_threshold):
        if is_above and run_start is None:
            run_start = index
        if run_start is not None and (not is_above or index == len(df) - 1):
            run_end = index if is_above else index - 1
            runs.append((run_start, run_end))
            run_start = None

    candidates = [
        (start_index, end_index)
        for start_index, end_index in runs
        if end_index - start_index + 1 >= consecutive_required
    ]

    # Sliding clips overlap. Merge qualifying candidates whose time windows
    # touch or overlap, without adding more complex temporal logic.
    merged_candidates: list[tuple[int, int]] = []
    for start_index, end_index in candidates:
        if not merged_candidates:
            merged_candidates.append((start_index, end_index))
            continue
        previous_start, previous_end = merged_candidates[-1]
        previous_end_seconds = float(df.iloc[previous_end]["end_seconds"])
        current_start_seconds = float(df.iloc[start_index]["start_seconds"])
        if current_start_seconds <= previous_end_seconds:
            merged_candidates[-1] = (
                previous_start,
                max(previous_end, end_index),
            )
        else:
            merged_candidates.append((start_index, end_index))

    events: list[dict[str, Any]] = []
    video_end_seconds = float(df["end_seconds"].max())
    for start_index, end_index in merged_candidates:
        candidate = df.iloc[start_index : end_index + 1]
        fall_start_seconds = float(candidate.iloc[0]["start_seconds"])
        fall_end_seconds = float(candidate.iloc[-1]["end_seconds"])
        max_fall_down_score = float(candidate["fall_down_score"].max())
        recovery_window_end = fall_end_seconds + recovery_window_seconds

        # Use clips following the candidate in the score timeline. The first
        # post-fall clip may overlap the candidate because sliding windows
        # overlap.
        post_fall = df.iloc[end_index + 1 :]
        post_fall = post_fall[
            post_fall["start_seconds"] < recovery_window_end
        ]
        recovery_clips = post_fall[
            post_fall["top_label"].isin(RECOVERY_LABELS)
            | (post_fall["recovery_score"] >= recovery_threshold)
        ]
        recovery_observed = not recovery_clips.empty
        recovery_evidence = (
            recovery_clips.iloc[0] if recovery_observed else None
        )
        sustained_ground_observed = bool(
            (post_fall["ground_score"] >= ground_threshold).any()
        )

        if recovery_observed:
            event_type = "Partial Fall"
            decision_reason = (
                "FallDown detected, followed by "
                "StandUp/Standing/Walking within recovery window."
            )
        elif sustained_ground_observed:
            event_type = "Full Fall"
            decision_reason = (
                "FallDown detected with no recovery and continued "
                "FallDown/LyingDown evidence within recovery window."
            )
        elif video_end_seconds < recovery_window_end:
            event_type = "Fall Detected, Recovery Unknown"
            decision_reason = (
                "FallDown detected, but the video ended before the full "
                "recovery window and provided no clear recovery or ground "
                "evidence."
            )
        else:
            event_type = "Fall Detected, Recovery Unknown"
            decision_reason = (
                "FallDown detected, but post-fall clips provided no clear "
                "recovery or ground evidence."
            )

        events.append(
            {
                "event_type": event_type,
                "fall_start_seconds": fall_start_seconds,
                "fall_end_seconds": fall_end_seconds,
                "max_fall_down_score": max_fall_down_score,
                "recovery_observed": recovery_observed,
                "recovery_evidence_label": (
                    str(recovery_evidence["top_label"])
                    if recovery_evidence is not None
                    else None
                ),
                "recovery_evidence_start_seconds": (
                    float(recovery_evidence["start_seconds"])
                    if recovery_evidence is not None
                    else None
                ),
                "recovery_evidence_end_seconds": (
                    float(recovery_evidence["end_seconds"])
                    if recovery_evidence is not None
                    else None
                ),
                "sustained_ground_observed": sustained_ground_observed,
                "recovery_window_seconds": float(recovery_window_seconds),
                "decision_reason": decision_reason,
            }
        )

    if any(event["event_type"] == "Full Fall" for event in events):
        decision = "Full Fall"
    elif any(
        event["event_type"] == "Fall Detected, Recovery Unknown"
        for event in events
    ):
        decision = "Fall Detected, Recovery Unknown"
    elif any(event["event_type"] == "Partial Fall" for event in events):
        decision = "Partial Fall"
    else:
        decision = "Normal"

    return {"decision": decision, "events": events}


def analyze_video_sliding(
    video_path: str | Path,
    model: Any,
    processor: Any,
    device: torch.device,
    clip_seconds: float = 2.0,
    stride_seconds: float = 1.0,
    threshold: float = 0.7,
    consecutive_required: int = 2,
    frames: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Run VideoMAE on fixed-duration clips across the complete video.

    Pass a pre-decoded `frames` list (from `decode_video_frames`) when the
    caller is also running `predict_video_whole` on the same video, to avoid
    decoding it twice. Otherwise the video is decoded once here, up front,
    instead of once per overlapping sliding window.
    """

    if clip_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("clip_seconds and stride_seconds must be positive")

    capture, frame_count, fps = _open_video(video_path)
    capture.release()
    duration = frame_count / fps
    num_frames = int(getattr(model.config, "num_frames", 16))

    if frames is None:
        frames = decode_video_frames(video_path, _processor_shortest_edge(processor))

    if duration <= clip_seconds:
        starts = [0.0]
    else:
        last_full_start = duration - clip_seconds
        starts = list(
            np.arange(0.0, last_full_start + 1e-9, stride_seconds)
        )
        if not starts or starts[-1] < last_full_start - 1e-6:
            starts.append(last_full_start)

    rows: list[dict[str, Any]] = []
    for clip_index, start_seconds in enumerate(starts):
        end_seconds = min(start_seconds + clip_seconds, duration)
        start_frame = int(round(start_seconds * fps))
        end_frame = min(
            frame_count - 1,
            max(start_frame, int(round(end_seconds * fps)) - 1),
        )
        indices = _select_indices(len(frames), start_frame, end_frame, num_frames)
        sampled = [frames[i] for i in indices]
        prediction = predict_frames(sampled, model, processor, device)
        scores = prediction["scores"]
        fall_down_score = float(scores.get("FallDown", 0.0))
        lying_down_score = float(scores.get("LyingDown", 0.0))
        stand_up_score = float(scores.get("StandUp", 0.0))
        standing_score = float(scores.get("Standing", 0.0))
        walking_score = float(scores.get("Walking", 0.0))
        rows.append(
            {
                "clip_index": clip_index,
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds),
                "top_label": prediction["top_label"],
                "top_score": prediction["top_score"],
                "fall_down_score": fall_down_score,
                "lying_down_score": lying_down_score,
                "stand_up_score": stand_up_score,
                "standing_score": standing_score,
                "walking_score": walking_score,
                "ground_score": max(
                    fall_down_score,
                    lying_down_score,
                ),
                "recovery_score": max(
                    stand_up_score,
                    standing_score,
                    walking_score,
                ),
                "above_threshold": fall_down_score >= threshold,
            }
        )

    return pd.DataFrame(rows)


def time_predictions(
    video_path: str | Path,
    model: Any,
    processor: Any,
    device: torch.device,
    **sliding_kwargs: Any,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    """Run whole-video + sliding-window predictions, timing each stage.

    Shared by the CLI and the GUI's Single Video / Batch tabs so timing
    stays consistent and isn't duplicated per caller.
    """

    decode_start = time.perf_counter()
    frames = decode_video_frames(video_path, _processor_shortest_edge(processor))
    decode_seconds = time.perf_counter() - decode_start

    whole_start = time.perf_counter()
    whole = predict_video_whole(video_path, model, processor, device, frames=frames)
    whole_video_seconds = time.perf_counter() - whole_start

    sliding_start = time.perf_counter()
    clip_df = analyze_video_sliding(
        video_path, model, processor, device, frames=frames, **sliding_kwargs
    )
    sliding_window_seconds = time.perf_counter() - sliding_start

    num_clips = len(clip_df)
    total_seconds = decode_seconds + whole_video_seconds + sliding_window_seconds
    timings = {
        "decode_seconds": decode_seconds,
        "whole_video_seconds": whole_video_seconds,
        "sliding_window_seconds": sliding_window_seconds,
        "num_clips": num_clips,
        "avg_clip_ms": (
            sliding_window_seconds / num_clips * 1000 if num_clips else 0.0
        ),
        "total_seconds": total_seconds,
    }
    return whole, clip_df, timings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the pretrained VideoMAE fall detector on one video."
    )
    parser.add_argument("--video", required=True, help="Path to a local video")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--consecutive", type=int, default=1)
    parser.add_argument(
        "--recovery-window-seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument("--recovery-threshold", type=float, default=0.5)
    parser.add_argument("--ground-threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def _run_predictions(
    video_path: Path,
    model: Any,
    processor: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    return time_predictions(
        video_path,
        model,
        processor,
        device,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        threshold=args.threshold,
        consecutive_required=args.consecutive,
    )


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.consecutive <= 0:
        raise ValueError("--consecutive must be positive")
    if args.recovery_window_seconds <= 0:
        raise ValueError("--recovery-window-seconds must be positive")
    for name, value in (
        ("--recovery-threshold", args.recovery_threshold),
        ("--ground-threshold", args.ground_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    model, processor, device = load_model(args.device)
    labels = {
        int(index): label for index, label in model.config.id2label.items()
    }
    print(f"Labels: {labels}")

    whole, clip_df, timings = _run_predictions(
        video_path,
        model,
        processor,
        device,
        args,
    )

    print(
        "\nTimings: "
        f"decode={timings['decode_seconds']:.2f}s, "
        f"whole_video={timings['whole_video_seconds']:.2f}s, "
        f"sliding_window={timings['sliding_window_seconds']:.2f}s "
        f"({timings['num_clips']} clips, "
        f"avg {timings['avg_clip_ms']:.0f}ms/clip), "
        f"total={timings['total_seconds']:.2f}s"
    )

    print(
        "Whole video prediction: "
        f"{whole['top_label']}, score={whole['top_score']:.4f}"
    )
    print(
        f"Whole video FallDown score: {whole['fall_down_score']:.4f}"
    )

    print("\nPer-clip FallDown scores:")
    print(
        clip_df[
            [
                "clip_index",
                "start_seconds",
                "end_seconds",
                "fall_down_score",
                "top_label",
            ]
        ].to_string(index=False)
    )

    print("\nTop clip scores:")
    print(
        clip_df.sort_values("fall_down_score", ascending=False)[
            [
                "start_seconds",
                "end_seconds",
                "fall_down_score",
                "top_label",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    result = classify_events(
        clip_df,
        threshold=args.threshold,
        consecutive_required=args.consecutive,
        recovery_window_seconds=args.recovery_window_seconds,
        recovery_threshold=args.recovery_threshold,
        ground_threshold=args.ground_threshold,
    )
    if result["events"]:
        for event in result["events"]:
            print(
                "\nFall candidate: "
                f"{event['fall_start_seconds']:.1f}s to "
                f"{event['fall_end_seconds']:.1f}s, "
                "max FallDown score="
                f"{event['max_fall_down_score']:.4f}"
            )
            print(
                "Post-fall recovery observed: "
                f"{'yes' if event['recovery_observed'] else 'no'}"
            )
            if event["recovery_observed"]:
                print(
                    "Recovery evidence: "
                    f"top_label={event['recovery_evidence_label']} from "
                    f"{event['recovery_evidence_start_seconds']:.1f}s to "
                    f"{event['recovery_evidence_end_seconds']:.1f}s"
                )
            print(
                "Sustained ground position observed: "
                f"{'yes' if event['sustained_ground_observed'] else 'no'}"
            )
            print(f"Event decision: {event['event_type']}")
    else:
        print("\nFall candidate: none")
    print(f"Final decision: {result['decision']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "fall_clip_scores.csv"
    json_path = OUTPUT_DIR / "fall_events.json"
    clip_df.to_csv(csv_path, index=False)

    output = {
        "video": str(video_path),
        "model": LOCAL_MODEL_NAME,
        "model_source": _model_source(),
        "device": str(device),
        "threshold": args.threshold,
        "clip_seconds": args.clip_seconds,
        "stride_seconds": args.stride_seconds,
        "consecutive_required": args.consecutive,
        "recovery_window_seconds": args.recovery_window_seconds,
        "recovery_threshold": args.recovery_threshold,
        "ground_threshold": args.ground_threshold,
        "whole_video_prediction": whole,
        "timings": timings,
        **result,
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
