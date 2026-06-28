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
