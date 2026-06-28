"""
CNN-based fall detection module.

Usage:
    python cnn_fall_detector.py video.mp4
    python cnn_fall_detector.py video.mp4 --annotate output.mp4
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 30
NUM_KEYPOINTS = 17
KPT_CONF_THRESH = 0.3
INPUT_FEATURES = NUM_KEYPOINTS * 3 + 7  # 58
THRESHOLD = 0.7
CONFIRM_WINDOWS = 3


def normalize_keypoints(keypoints: np.ndarray, bbox_xyxy: list) -> np.ndarray:
    """Normalize keypoints relative to bounding box."""
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    kpts = np.array(keypoints, dtype=np.float32).copy()
    kpts[:, 0] = (kpts[:, 0] - x1) / bw
    kpts[:, 1] = (kpts[:, 1] - y1) / bh
    return kpts


def extract_features(record: dict) -> np.ndarray:
    """Convert pose record to 58-dim feature vector."""
    frame_w = float(record.get("frame_w", 640))
    frame_h = float(record.get("frame_h", 480))

    norm_kpts = normalize_keypoints(record["keypoints"], record["bbox_xyxy"])
    kpt_features = norm_kpts.reshape(-1).astype(np.float32)

    x1, y1, x2, y2 = record["bbox_xyxy"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    visible_count = sum(1 for kp in record["keypoints"] if kp[2] >= KPT_CONF_THRESH)

    bbox_features = np.array([
        cx / frame_w,
        cy / frame_h,
        w / frame_w,
        h / frame_h,
        w / max(h, 1.0),
        (w * h) / (frame_w * frame_h),
        visible_count / NUM_KEYPOINTS,
    ], dtype=np.float32)

    return np.concatenate([kpt_features, bbox_features])


class FallCNN1D(nn.Module):
    """Temporal CNN for fall classification from pose sequences."""

    def __init__(self, input_features: int = INPUT_FEATURES, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x))


class FallDetector:
    """Fall detection using YOLOv8-pose + temporal CNN."""

    def __init__(
        self,
        model_path: str,
        threshold: float = THRESHOLD,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.seq_len = SEQ_LEN
        self.confirm_windows = CONFIRM_WINDOWS

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = FallCNN1D(INPUT_FEATURES, num_classes=2).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self._pose_model = None

    @property
    def pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO
            self._pose_model = YOLO("yolov8n-pose.pt")
        return self._pose_model

    def reset(self):
        """Reset state for new video."""
        self.buffers = defaultdict(lambda: deque(maxlen=self.seq_len))
        self.track_state = defaultdict(lambda: {
            "consecutive_fall": 0,
            "in_event": False,
            "event_start": None,
            "last_fall_frame": None,
            "max_conf": 0.0,
        })
        self.events: List[Dict[str, Any]] = []

    @torch.no_grad()
    def predict_window(self, window: list) -> Tuple[bool, float]:
        """Classify a 30-frame pose sequence. Each element is a pose record dict."""
        features = np.stack([extract_features(r) for r in window])
        tensor = torch.tensor(features, dtype=torch.float32)
        tensor = tensor.transpose(0, 1).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        fall_prob = float(probs[1].item())
        return fall_prob >= self.threshold, fall_prob

    def _update_event_state(
        self,
        video_name: str,
        track_id: int,
        frame_idx: int,
        is_fall: bool,
        confidence: float,
    ):
        """Update fall event state machine."""
        state = self.track_state[track_id]

        if is_fall:
            state["consecutive_fall"] += 1
            state["last_fall_frame"] = frame_idx
            state["max_conf"] = max(state["max_conf"], confidence)

            if not state["in_event"] and state["consecutive_fall"] >= self.confirm_windows:
                state["in_event"] = True
                state["event_start"] = frame_idx
        else:
            if state["in_event"]:
                self.events.append({
                    "video_name": video_name,
                    "track_id": int(track_id),
                    "start_frame": int(state["event_start"]),
                    "end_frame": int(state["last_fall_frame"]),
                    "confidence": float(state["max_conf"]),
                })
            state["consecutive_fall"] = 0
            state["in_event"] = False
            state["event_start"] = None
            state["last_fall_frame"] = None
            state["max_conf"] = 0.0

    def _finalize_events(self):
        """Close any open events at end of video."""
        for tid, state in self.track_state.items():
            if state["in_event"]:
                self.events.append({
                    "video_name": getattr(self, "_current_video", "unknown"),
                    "track_id": int(tid),
                    "start_frame": int(state["event_start"]),
                    "end_frame": int(state["last_fall_frame"]),
                    "confidence": float(state["max_conf"]),
                })

    def run_on_video(
        self,
        video_path: str,
        progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process video file. Returns list of fall events."""
        self.reset()
        video_path = str(video_path)
        video_name = Path(video_path).name
        self._current_video = video_name

        cap = cv2.VideoCapture(video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        results = self.pose_model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False,
        )

        frame_idx = 0
        for r in results:
            frame_idx += 1

            if progress and frame_idx % 100 == 0:
                pct = 100 * frame_idx / max(total_frames, 1)
                print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%)", end="\r")

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else np.arange(len(boxes))
            )
            keypoints = (
                r.keypoints.data.cpu().numpy()
                if r.keypoints is not None
                else np.zeros((len(boxes), NUM_KEYPOINTS, 3))
            )

            for i in range(len(boxes)):
                tid = int(track_ids[i])
                kpts = keypoints[i]
                if kpts.shape[-1] == 2:
                    kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=-1)

                record = {
                    "frame_idx": frame_idx,
                    "track_id": tid,
                    "bbox_xyxy": boxes[i].tolist(),
                    "keypoints": kpts.tolist(),
                    "frame_w": frame_w,
                    "frame_h": frame_h,
                }

                self.buffers[tid].append(record)

                if len(self.buffers[tid]) < self.seq_len:
                    continue

                is_fall, conf = self.predict_window(list(self.buffers[tid]))
                self._update_event_state(video_name, tid, frame_idx, is_fall, conf)

        if progress:
            print()

        self._finalize_events()
        return self.events
