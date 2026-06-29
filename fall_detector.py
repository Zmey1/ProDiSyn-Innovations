"""
================================================================================
Name        : detector.py
Author      : ProDiSyn Innovations
Version     : 1.0.0
Description : Fall detection engine for the SafeLensAI module.
              Implements multi-signal fall classification using YOLOv8-pose
              bounding-box geometry, keypoint voting, and temporal smoothing.
================================================================================
"""

import logging
import math
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import PersonTracker

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection thresholds / tuning constants  (values identical to original)
# ---------------------------------------------------------------------------

# Signal 1 — Bounding-box geometry
ASPECT_RATIO_THRESHOLD: float        = 1.2
VELOCITY_THRESHOLD: float            = 250.0
EMERGENCY_VELOCITY_MULTIPLIER: float = 1.5

# Signal 2 — Keypoint voting
TORSO_ANGLE_THRESHOLD: float  = 35.0
HIP_HEIGHT_THRESHOLD: float   = 0.45
HEAD_HIP_DIST_THRESHOLD: float = 0.25
KEYPOINT_CONF_THRESHOLD: float = 0.4
KP_TRUSTWORTHY_CONF: float     = 0.55

# Floor-posture signal  (primary forward-fall detector)
FLOOR_CONF_THRESHOLD: float = 0.28

# Temporal smoothing
REQUIRED_FALL_FRAMES: int     = 5
REQUIRED_COLLAPSE_FRAMES: int = 8

# Height-collapse fallback
HEIGHT_COLLAPSE_THRESHOLD: float = 0.50
MIN_STANDING_BBOX_H: int         = 80
UPRIGHT_RATIO_CEILING: float     = 0.9

# Visibility gate
MIN_UPPER_BODY_KEYPOINTS: int = 2
MIN_LOWER_BODY_KEYPOINTS: int = 0

# Occlusion guard values
HORIZONTAL_VELOCITY_GUARD: float = 80.0
H_TO_V_RATIO_GUARD: float        = 1.5
ONSET_VELOCITY_THRESHOLD: float  = 2.5
STATIC_ONSET_THRESHOLD: float    = 5.0
TOP_STABLE_DELTA_PX: int         = 20
BOTTOM_RISING_DELTA_PX: int      = 35
VELOCITY_HISTORY_WINDOW: int     = 5
Y_HISTORY_LOOKBACK: int          = 8
BBOX_DEQUE_MAXLEN: int           = 10

# Standalone demo
DEMO_BUFFER_SIZE: int   = 2
DEMO_IOU_THRESHOLD: float = 0.45


class FallDetector:
    """Multi-signal fall detector built on YOLOv8-pose.

    Combines bounding-box geometry (Signal 1), keypoint voting (Signal 2),
    floor-posture analysis, and temporal smoothing to confirm fall events
    with low false-positive rates on elevated CCTV cameras.
    """

    def __init__(self, model_path: str = 'yolov8s-pose.pt', device: str = 'mps') -> None:

        self.model = YOLO(model_path)
        self.device = device

        # Robust ID tracker — maintains spatial history and handles ID re-assignment
        self.tracker = PersonTracker(history_size=30, max_inactive_time=2.0)

        # Per-PID fall state booleans: {pid: bool}
        self.fall_states = {}

        # --- SIGNAL 1: Bounding Box + Velocity thresholds ---

        self.aspect_ratio_threshold = ASPECT_RATIO_THRESHOLD
        # width/height ratio above which the person is considered horizontal.

        self.velocity_threshold = VELOCITY_THRESHOLD
        # Vertical pixel shift per second to trigger falling-fast signal.

        self.emergency_velocity_multiplier = EMERGENCY_VELOCITY_MULTIPLIER
        # If velocity exceeds velocity_threshold × this, entry visibility gate is bypassed.

        # --- SIGNAL 2: Keypoint Majority thresholds ---

        self.torso_angle_threshold = TORSO_ANGLE_THRESHOLD
        # Torso angle (degrees from horizontal) below which torso is considered fallen.

        self.hip_height_threshold = HIP_HEIGHT_THRESHOLD
        # Normalised position of hips inside bbox above which hips are considered low.

        self.head_hip_dist_threshold = HEAD_HIP_DIST_THRESHOLD
        # Normalised head-to-hip distance below which head and hips are considered close.

        self.keypoint_conf_threshold = KEYPOINT_CONF_THRESHOLD
        # Minimum YOLO keypoint confidence to consider a keypoint valid for voting.

        # --- Temporal smoothing ---

        self.fall_frame_counts = {}   # pid -> consecutive frames with raw_fall_flag True
        self.required_fall_frames = REQUIRED_FALL_FRAMES
        # Number of consecutive flagged frames before a fall is confirmed.

        self.post_fall_still_frames = {}  # pid -> frames of stillness after fall flag

        # --- Height Collapse Signal (last-resort fallback) ---

        self.max_standing_height = {}
        # pid -> maximum bbox height observed while person was standing upright.

        self.height_collapse_frame_counts = {}
        # pid -> consecutive frames where height collapse fired as fallback.

        self.required_collapse_frames = REQUIRED_COLLAPSE_FRAMES
        # Frames of sustained height collapse before confirming via this fallback.
        # 8 frames ≈ 0.5–0.6 seconds at 13–15 FPS.

        self.bbox_center_history = {}
        # pid -> deque of (cx, timestamp) tuples, maxlen=10.

        self.bbox_y_history = {}
        # pid -> deque of (y1, y2) tuples, maxlen=10.

        self.height_drop_observed = {}
        # pid -> bool: set True once a rapid height-drop onset was observed for this pid.
        # Guards against false positives from seated/static persons.

        # --- Visibility gate parameters ---

        self.min_upper_body_keypoints = MIN_UPPER_BODY_KEYPOINTS
        self.min_lower_body_keypoints = MIN_LOWER_BODY_KEYPOINTS

    # -----------------------------------------------------------------------
    # GEOMETRY HELPERS
    # -----------------------------------------------------------------------

    def _calculate_aspect_ratio(self, bbox: tuple) -> float:
        """Return the width-to-height ratio of a bounding box.

        Args:
            bbox: Bounding box as ``(x1, y1, x2, y2)``.

        Returns:
            Width divided by height, or ``0`` when height is zero.
        """
        x1, y1, x2, y2 = bbox
        width  = x2 - x1
        height = y2 - y1
        if height == 0:
            return 0
        return width / height

    def _calculate_vertical_velocity(self, history) -> float:
        """Estimate the downward centre-Y velocity over the last five history frames.

        Args:
            history: Sequence of history dicts with keys ``'bbox'`` and ``'time'``.

        Returns:
            Signed velocity in pixels per second (positive = downward),
            or ``0.0`` when fewer than five frames are available.
        """
        if len(history) < VELOCITY_HISTORY_WINDOW:
            return 0.0
        recent = history[-1]
        past   = history[-VELOCITY_HISTORY_WINDOW]
        recent_y = (recent['bbox'][3] + recent['bbox'][1]) / 2.0
        past_y   = (past['bbox'][3]   + past['bbox'][1])   / 2.0
        dy = recent_y - past_y
        dt = recent['time'] - past['time']
        if dt == 0:
            return 0.0
        return dy / dt

    # -----------------------------------------------------------------------
    # KEYPOINT METRIC HELPERS
    # -----------------------------------------------------------------------

    def calculate_torso_angle(self, keypoints) -> Optional[float]:
        """Return the torso angle in degrees relative to horizontal.

        Args:
            keypoints: YOLO keypoints array of shape ``(N, 2)`` or ``(N, 3)``.

        Returns:
            Angle in degrees between mid-shoulder and mid-hip, measured from
            horizontal.  ``None`` when any required keypoint is below
            ``keypoint_conf_threshold``.
        """
        if keypoints is None or len(keypoints) < 13:
            return None
        if keypoints.shape[-1] < 3:
            kp5, kp6   = keypoints[5],  keypoints[6]
            kp11, kp12 = keypoints[11], keypoints[12]
            conf5 = conf6 = conf11 = conf12 = 1.0
        else:
            kp5,  kp6  = keypoints[5],  keypoints[6]
            kp11, kp12 = keypoints[11], keypoints[12]
            conf5, conf6   = kp5[2],  kp6[2]
            conf11, conf12 = kp11[2], kp12[2]

        if any(c < self.keypoint_conf_threshold for c in [conf5, conf6, conf11, conf12]):
            return None

        mid_shoulder = ((kp5[0] + kp6[0]) / 2.0, (kp5[1] + kp6[1]) / 2.0)
        mid_hip      = ((kp11[0] + kp12[0]) / 2.0, (kp11[1] + kp12[1]) / 2.0)

        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]

        angle = abs(math.degrees(math.atan2(dy, dx)))
        return angle

    def calculate_hip_height_ratio(self, keypoints, bbox: tuple) -> Optional[float]:
        """Return the normalised Y-position of the mid-hip within the bounding box.

        Args:
            keypoints: YOLO keypoints array of shape ``(N, 2)`` or ``(N, 3)``.
            bbox: Bounding box as ``(x1, y1, x2, y2)``.

        Returns:
            Mid-hip Y offset as a fraction of bbox height, or ``None`` when
            hip keypoints fall below ``keypoint_conf_threshold``.
        """
        if keypoints is None or len(keypoints) < 13:
            return None
        if keypoints.shape[-1] < 3:
            kp11, kp12 = keypoints[11], keypoints[12]
            conf11 = conf12 = 1.0
        else:
            kp11, kp12 = keypoints[11], keypoints[12]
            conf11, conf12 = kp11[2], kp12[2]

        if conf11 < self.keypoint_conf_threshold or conf12 < self.keypoint_conf_threshold:
            return None

        mid_hip_y = (kp11[1] + kp12[1]) / 2.0
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        if bbox_height == 0:
            return None
        return (mid_hip_y - y1) / bbox_height

    def calculate_head_hip_dist(self, keypoints, bbox: tuple) -> Optional[float]:
        """Return the normalised distance between the nose and mid-hip.

        Args:
            keypoints: YOLO keypoints array of shape ``(N, 2)`` or ``(N, 3)``.
            bbox: Bounding box as ``(x1, y1, x2, y2)``.

        Returns:
            Absolute head-to-hip Y distance divided by bbox height, or
            ``None`` when any required keypoint is below ``keypoint_conf_threshold``.
        """
        if keypoints is None or len(keypoints) < 13:
            return None
        if keypoints.shape[-1] < 3:
            kp0, kp11, kp12 = keypoints[0], keypoints[11], keypoints[12]
            conf0 = conf11 = conf12 = 1.0
        else:
            kp0, kp11, kp12 = keypoints[0], keypoints[11], keypoints[12]
            conf0, conf11, conf12 = kp0[2], kp11[2], kp12[2]

        if any(c < self.keypoint_conf_threshold for c in [conf0, conf11, conf12]):
            return None

        head_y    = kp0[1]
        mid_hip_y = (kp11[1] + kp12[1]) / 2.0

        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        if bbox_height == 0:
            return None
        return abs(head_y - mid_hip_y) / bbox_height

    def _calculate_floor_posture_signals(
        self, keypoints, bbox: tuple
    ) -> Tuple[Optional[bool], Optional[float]]:
        """
        PRIMARY signal for forward/backward fall detection from elevated CCTV cameras.

        CORE INSIGHT — Shoulder-vs-Ankle Y inversion:
        ─────────────────────────────────────────────────────────────────────────
        In an elevated-camera view (Y increases downward in image):

          STANDING / BENDING / SITTING / CROUCHING:
            Shoulders (upper body) are always ABOVE ankles in the image.
            → shoulder_y  <  ankle_y  (shoulder has SMALLER Y)
            → shoulder_below_ankles = False  (safe, no false trigger)

          FALLEN FORWARD (feet toward camera):
            Feet end up closer to camera → ankles near TOP of image (small Y)
            Head/shoulders farther away  → shoulders near BOTTOM (large Y)
            → shoulder_y  >  ankle_y  (shoulder has LARGER Y)
            → shoulder_below_ankles = True  ← FALL DETECTED

          FALLEN BACKWARD (head toward camera):
            Shoulders at top (normal), ankles at bottom (normal).
            shoulder_ankle_proximity is LARGE (full body span in image).
            shoulder_below_ankles = False — NOT caught by this signal.
            ⚠ Backward fall is handled by existing sideways aspect-ratio signals
              and Timer 2 (10-second sustained floor posture) in main.py.
        ─────────────────────────────────────────────────────────────────────────

        Uses SHOULDERS (kp5, kp6) instead of nose: shoulders are reliably detected
        even when the person is face-down and the nose is hidden.

        Uses a LOWER confidence threshold (0.28) because lying down reduces YOLO's
        keypoint confidence scores significantly.

        Returns:
          shoulder_below_ankles (bool | None) — True = forward fall posture
          shoulder_ankle_proximity (float | None) — |shoulder_y - ankle_y| / bbox_height
        """
        if keypoints is None or len(keypoints) < 17:
            return None, None

        has_conf = keypoints.shape[-1] == 3
        # Lower threshold: keypoint quality degrades when person is on the floor.
        # 0.28 is still well above noise (< 0.10) but tolerant of lying-down detection.
        floor_conf_threshold = FLOOR_CONF_THRESHOLD

        kp5,  kp6  = keypoints[5],  keypoints[6]
        kp15, kp16 = keypoints[15], keypoints[16]

        shoulder_ys = []
        if (float(kp5[2])  if has_conf else 1.0) >= floor_conf_threshold:
            shoulder_ys.append(float(kp5[1]))
        if (float(kp6[2])  if has_conf else 1.0) >= floor_conf_threshold:
            shoulder_ys.append(float(kp6[1]))

        ankle_ys = []
        if (float(kp15[2]) if has_conf else 1.0) >= floor_conf_threshold:
            ankle_ys.append(float(kp15[1]))
        if (float(kp16[2]) if has_conf else 1.0) >= floor_conf_threshold:
            ankle_ys.append(float(kp16[1]))

        if not shoulder_ys or not ankle_ys:
            return None, None

        avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
        avg_ankle_y    = sum(ankle_ys)    / len(ankle_ys)

        x1, y1, x2, y2 = bbox
        bbox_height = max(1.0, float(y2 - y1))

        # shoulder_below_ankles: True when avg_shoulder_y > avg_ankle_y
        # (shoulders have LARGER Y = lower in frame = farther from camera)
        shoulder_below_ankles  = bool(avg_shoulder_y > avg_ankle_y)
        shoulder_ankle_proximity = abs(avg_shoulder_y - avg_ankle_y) / bbox_height

        return shoulder_below_ankles, shoulder_ankle_proximity

    # -----------------------------------------------------------------------
    # VISIBILITY GATE HELPERS
    # -----------------------------------------------------------------------

    def is_body_sufficiently_visible(self, keypoints) -> bool:
        """
        Returns True only if enough upper and lower body keypoints are visible
        above keypoint_conf_threshold to make a reliable fall determination.
        """
        if keypoints is None or len(keypoints) < 17:
            return False

        upper_body_indices = [0, 5, 6, 11, 12]
        lower_body_indices = [13, 14, 15, 16]
        has_conf = keypoints.shape[-1] == 3

        upper_visible = sum(
            1 for i in upper_body_indices
            if (keypoints[i][2] if has_conf else 1.0) > self.keypoint_conf_threshold
        )
        lower_visible = sum(
            1 for i in lower_body_indices
            if (keypoints[i][2] if has_conf else 1.0) > self.keypoint_conf_threshold
        )

        return (upper_visible >= self.min_upper_body_keypoints and
                lower_visible >= self.min_lower_body_keypoints)

    def _is_hip_visible(self, keypoints) -> bool:  # noqa: D401
        """
        Returns True if at least one hip keypoint has YOLO confidence >= threshold.
        Guards against false positives when only the scalp is visible.
        """
        if keypoints is None or len(keypoints) < 13:
            return False
        has_conf = keypoints.shape[-1] == 3
        conf11 = keypoints[11][2] if has_conf else 1.0
        conf12 = keypoints[12][2] if has_conf else 1.0
        return (conf11 >= self.keypoint_conf_threshold or
                conf12 >= self.keypoint_conf_threshold)

    # -----------------------------------------------------------------------
    # HEIGHT COLLAPSE + OCCLUSION GUARD HELPERS
    # -----------------------------------------------------------------------

    def _calculate_height_collapse(
        self, p_id, current_bbox: tuple
    ) -> Tuple[float, bool]:
        """
        Computes the bbox height collapse ratio vs the person's known standing
        height baseline. Returns (ratio, is_collapsed).

        Acts as a LAST-RESORT fallback for forward/backward fall when the primary
        shoulder_below_ankles signal is unavailable (face-down, keypoints occluded).

        Contains THREE independent occlusion guards.

        IMPORTANT DESIGN DECISION — threshold is 0.50 (not 0.55):
          0.55 caused false positives for seated workers — their desk-occluded bbox
          (head-to-hip only) reached 55-65% of standing height, triggering Timer 2
          in main.py and forcing them into CRITICAL state.
          0.50 is tight enough that a seated person (ratio ≈ 0.65-0.75) will NOT
          trigger, but a genuinely fallen person (ratio ≈ 0.35-0.50) will.

        Guard 3 onset threshold is 2.5 px/frame (not 5.0) to catch slower forward
        falls, but height_drop_observed prevents static-seated persons from being
        misidentified.
        """
        x1, y1, x2, y2 = current_bbox
        current_height = y2 - y1
        if current_height <= 0:
            return 1.0, False

        # Update standing height baseline when person appears upright.
        aspect_ratio = self._calculate_aspect_ratio(current_bbox)
        if aspect_ratio < UPRIGHT_RATIO_CEILING and current_height > MIN_STANDING_BBOX_H:
            prev_max = self.max_standing_height.get(p_id, 0)
            if current_height > prev_max:
                self.max_standing_height[p_id] = current_height

        max_h = self.max_standing_height.get(p_id, 0)
        if max_h == 0:
            return 1.0, False  # No baseline yet

        ratio = current_height / max_h
        height_collapse_threshold = HEIGHT_COLLAPSE_THRESHOLD
        raw_is_collapsed = ratio < height_collapse_threshold

        if not raw_is_collapsed:
            return ratio, False

        # ── OCCLUSION GUARD 1: Horizontal Motion Guard ───────────────────────
        h_vel = self._get_horizontal_velocity(p_id)
        v_vel = self._get_vertical_velocity_from_history(p_id)
        is_moving_sideways = (
            h_vel > HORIZONTAL_VELOCITY_GUARD
            and h_vel > (v_vel * H_TO_V_RATIO_GUARD)
        )

        # ── OCCLUSION GUARD 2: Bbox Top vs Bottom Asymmetry ──────────────────
        is_top_stable_bottom_rising = self._is_top_stable_bottom_rising(p_id)

        # ── OCCLUSION GUARD 3: Collapse Onset Velocity ───────────────────────
        # Real fall:          20–50 px/frame → onset_velocity > 2.5 → Guard 3 passes
        # Gradual forward fall: 2–5 px/frame → might pass at threshold 2.5
        # Seated/static:       0 px/frame   → onset_velocity = 0.0 → Guard 3 blocks
        onset_velocity = self._get_height_drop_velocity(p_id)

        # Mark rapid onset observed. Once True, Guard 3 does not suppress the
        # collapsed state (the person has genuinely fallen and is now lying still).
        if onset_velocity >= ONSET_VELOCITY_THRESHOLD:
            self.height_drop_observed[p_id] = True

        # is_static_occlusion: True if no rapid drop ever occurred → table / chair occlusion
        is_static_occlusion = (
            (onset_velocity < STATIC_ONSET_THRESHOLD)
            and not self.height_drop_observed.get(p_id, False)
        )

        if is_moving_sideways or is_top_stable_bottom_rising or is_static_occlusion:
            return ratio, False

        return ratio, True

    def _get_horizontal_velocity(self, p_id) -> float:
        history = self.bbox_center_history.get(p_id)
        if history is None or len(history) < 5:
            return 0.0
        past_cx, past_time = history[-5]
        curr_cx, curr_time = history[-1]
        dt = curr_time - past_time
        if dt == 0:
            return 0.0
        return abs(curr_cx - past_cx) / dt

    def _get_vertical_velocity_from_history(self, p_id) -> float:
        y_history  = self.bbox_y_history.get(p_id)
        cx_history = self.bbox_center_history.get(p_id)
        if y_history is None or len(y_history) < 5:
            return 0.0
        if cx_history is None or len(cx_history) < 5:
            return 0.0
        past_y1, past_y2 = y_history[-5]
        curr_y1, curr_y2 = y_history[-1]
        past_cy = (past_y1 + past_y2) / 2.0
        curr_cy = (curr_y1 + curr_y2) / 2.0
        _, past_time = cx_history[-5]
        _, curr_time = cx_history[-1]
        dt = curr_time - past_time
        if dt == 0:
            return 0.0
        return abs(curr_cy - past_cy) / dt

    def _is_top_stable_bottom_rising(self, p_id) -> bool:
        """Return True when the top edge is steady but the bottom edge is rising.

        Used as Occlusion Guard 2 inside :meth:`_calculate_height_collapse`
        to distinguish a person walking behind an obstruction (bottom of bbox
        rises as they disappear) from a genuine height collapse.

        Args:
            p_id: Tracking ID of the person.

        Returns:
            ``True`` if the top-edge shift is below ``TOP_STABLE_DELTA_PX``
            and the bottom-edge shift exceeds ``BOTTOM_RISING_DELTA_PX``.
        """
        history = self.bbox_y_history.get(p_id)
        if history is None or len(history) < 8:
            return False
        past_y1, past_y2 = history[-8]
        curr_y1, curr_y2 = history[-1]
        y1_delta = abs(curr_y1 - past_y1)
        y2_delta = abs(curr_y2 - past_y2)
        top_is_stable    = y1_delta < TOP_STABLE_DELTA_PX
        bottom_is_rising = y2_delta > BOTTOM_RISING_DELTA_PX
        return top_is_stable and bottom_is_rising

    def _get_height_drop_velocity(self, p_id) -> float:
        """Return the average per-frame bbox height drop over the recent history.

        Used as the Collapse Onset Velocity in Occlusion Guard 3.
        A value near zero means the person has been at this height for a while
        (static occlusion), whereas a high value indicates a sudden collapse.

        Args:
            p_id: Tracking ID of the person.

        Returns:
            Mean positive height drop in pixels per frame, or ``0.0``
            when fewer than five history entries exist.
        """
        history = self.bbox_y_history.get(p_id)
        if history is None or len(history) < 5:
            return 0.0
        heights = [(e[1] - e[0]) for e in history]
        drops = [max(0, heights[i] - heights[i + 1]) for i in range(len(heights) - 1)]
        if not drops:
            return 0.0
        return sum(drops) / len(drops)

    # -----------------------------------------------------------------------
    # MAIN DETECTION METHOD
    # -----------------------------------------------------------------------

    def detect_fall(self, p_id, current_bbox, history, keypoints=None) -> bool:
        """Analyse spatial and keypoint attributes to determine if a fall occurred.

        Evaluates the multi-signal decision tree (Hard Bypasses A/C, Signals B,
        Priorities 1-3, Explicit Negative) with temporal debouncing and returns
        the stable per-PID fall state.

        Args:
            p_id: Tracking ID assigned by YOLO.
            current_bbox: Current bounding box as ``(x1, y1, x2, y2)``.
            history: List of history dicts from :class:`tracker.PersonTracker`.
            keypoints: Optional YOLO keypoints array of shape ``(N, 2|3)``.

        Returns:
            ``True`` if a fall is confirmed for this *p_id*, ``False`` otherwise.

        Detection priority order
        ------------------------
        HARD BYPASS A — Extreme horizontal (aspect_ratio > 1.8, height < 250 px)
                        Always fires, no keypoints needed.  [SIDEWAYS FALL]

        SIGNAL B     — Shoulder-below-ankles (shoulder_y > ankle_y in image)
                        Primary forward fall detector.
                        [FORWARD FALL / FEET-TOWARD-CAMERA]

        HARD BYPASS C — Height collapse + rapid onset + no trustworthy keypoints
                        Last-resort fallback when Signal B keypoints unavailable.
                        [FORWARD/BACKWARD FALL, OCCLUDED KEYPOINTS]

        PRIORITY 1   — Signal 2: Keypoint Majority, kp_trustworthy=True
                        REQUIRES is_horizontal to prevent bending-over FPs.
                        [SIDEWAYS FALL WITH KEYPOINT CONFIRMATION]

        PRIORITY 2   — Bbox horizontal, no usable keypoints
                        [SIDEWAYS FALL, NO KEYPOINTS]

        PRIORITY 3   — Rapid downward velocity  [MID-FALL / FAST COLLAPSE]

        NEGATIVE     — Trustworthy keypoints say upright, no floor signals.
                        Explicitly blocks false positives.

        KEY FIXES vs previous version:
          FIX 1 — forward fall: Added Signal B (shoulder_below_ankles).
          FIX 2 — bending FP:   Priority 1 NOW REQUIRES is_horizontal as gate.
          FIX 3 — seated FP:    Removed Priority 1b branch.
          FIX 4 — seated FP:    Removed kp_spread Metric 4 from voting.
          FIX 5 — Timer 2 FP:   height_collapse_threshold reverted to 0.50.
        """
        # Populate bbox position histories BEFORE the visibility gate.
        _x1, _y1, _x2, _y2 = current_bbox
        _cx = (_x1 + _x2) / 2.0
        if p_id not in self.bbox_center_history:
            self.bbox_center_history[p_id] = deque(maxlen=BBOX_DEQUE_MAXLEN)
        self.bbox_center_history[p_id].append((_cx, time.time()))
        if p_id not in self.bbox_y_history:
            self.bbox_y_history[p_id] = deque(maxlen=BBOX_DEQUE_MAXLEN)
        self.bbox_y_history[p_id].append((_y1, _y2))

        # ── TWO-PHASE VISIBILITY GATE ─────────────────────────────────────────
        current_state = self.fall_states.get(p_id, False)

        if current_state:
            # EXIT IMMUNITY: Fall already confirmed — bypass entry gate.
            pass
        else:
            gate_velocity    = self._calculate_vertical_velocity(history)
            emergency_thresh = self.velocity_threshold * self.emergency_velocity_multiplier
            is_emergency_entry = gate_velocity > emergency_thresh

            if not is_emergency_entry:
                if not self.is_body_sufficiently_visible(keypoints):
                    return False

        # ── STEP 1: Signal 1 inputs ───────────────────────────────────────────
        aspect_ratio      = self._calculate_aspect_ratio(current_bbox)
        vertical_velocity = self._calculate_vertical_velocity(history)

        is_horizontal   = aspect_ratio > self.aspect_ratio_threshold
        is_falling_fast = vertical_velocity > self.velocity_threshold

        # ── STEP 1b: Height collapse (last-resort fallback) ───────────────────
        height_collapse_ratio, is_height_collapsed = self._calculate_height_collapse(
            p_id, current_bbox
        )

        # ── STEP 2: Keypoint metrics ──────────────────────────────────────────
        torso_angle = self.calculate_torso_angle(keypoints)
        hip_ratio   = self.calculate_hip_height_ratio(keypoints, current_bbox)
        head_hip    = self.calculate_head_hip_dist(keypoints, current_bbox)

        # ── STEP 3: Keypoint voting (Metrics 1–3 only) ───────────────────────
        # NOTE: kp_spread (Metric 4) REMOVED — it caused false positives for
        # seated workers viewed from elevated cameras (all keypoints at similar
        # Y level when sitting → kp_spread < 0.35 → incorrect fall vote).
        available_metrics = 0
        kp_votes = 0

        # Metric 1: Torso Angle
        if torso_angle is not None:
            available_metrics += 1
            if torso_angle < self.torso_angle_threshold:
                kp_votes += 1

        # Metric 2: Hip Height Ratio (skip when bbox is horizontal — unreliable)
        if hip_ratio is not None and not is_horizontal:
            available_metrics += 1
            if hip_ratio > self.hip_height_threshold:
                kp_votes += 1

        # Metric 3: Head-Hip Distance
        if head_hip is not None:
            available_metrics += 1
            if head_hip < self.head_hip_dist_threshold:
                kp_votes += 1

        # Dynamic voting threshold
        if available_metrics >= 2:
            kp_confirms_fall = (kp_votes >= 2)
        elif available_metrics == 1:
            kp_confirms_fall = (kp_votes >= 1)
        else:
            kp_confirms_fall = False

        # torso_is_fallen: mandatory gate for Priority 1
        torso_is_fallen = (torso_angle is not None and
                           torso_angle < self.torso_angle_threshold)

        # ── STEP 3b: Keypoint trustworthiness ────────────────────────────────
        if not self._is_hip_visible(keypoints):
            kp_available   = False
            kp_trustworthy = False
        else:
            kp_available = (available_metrics > 0)
            if keypoints is not None and keypoints.shape[-1] == 3:
                conf_scores   = keypoints[:, 2]
                visible_confs = conf_scores[conf_scores > self.keypoint_conf_threshold]
                avg_kp_confidence = float(np.mean(visible_confs)) if len(visible_confs) > 0 else 0.0
            else:
                avg_kp_confidence = 1.0
            kp_trustworthy = kp_available and (avg_kp_confidence >= KP_TRUSTWORTHY_CONF)

        # ── STEP 3c: Floor posture signals (primary forward fall detector) ────
        shoulder_below_ankles, shoulder_ankle_proximity = \
            self._calculate_floor_posture_signals(keypoints, current_bbox)

        # ── STEP 4: Fall flag decision tree (priority order) ─────────────────
        x1, y1, x2, y2 = current_bbox
        bbox_height = y2 - y1

        # HARD BYPASS A: Extreme horizontal — definite sideways fall
        is_extreme_horizontal = (aspect_ratio > 1.8) and (bbox_height < 250)

        # HARD BYPASS C: Height collapse fallback
        # ONLY fires when:
        #   (a) height_drop_observed = True (rapid onset actually occurred — blocks seated persons)
        #   (b) height < 45% of baseline (very dramatic collapse)
        #   (c) Signal 2 (keypoints) is NOT trustworthy
        # This catches forward falls where shoulder_below_ankles is unavailable
        # (person face-down, YOLO cannot detect shoulders at all).
        is_extreme_collapse = (
            not kp_trustworthy
            and is_height_collapsed
            and height_collapse_ratio < 0.45
            and self.height_drop_observed.get(p_id, False)    # MUST have seen rapid onset
            and self.max_standing_height.get(p_id, 0) >= 100
        )

        # ── DECISION TREE ─────────────────────────────────────────────────────

        if is_extreme_horizontal:
            # Sideways hard bypass — extreme wide box, unconditional
            raw_fall_flag = True

        elif shoulder_below_ankles is True:
            # ── SIGNAL B: Forward fall (primary) ────────────────────────────
            # Shoulders are LOWER in the image than ankles.
            # This is geometrically impossible when standing, bending, or sitting:
            #   • Standing: shoulders at top, ankles at bottom → ankle_y > shoulder_y
            #   • Bending over desk: shoulders drop but ankles remain at floor below → False
            #   • Sitting: shoulders at mid-height, ankles visible below → False
            #   • Crouching/tying shoes: ankles at floor, still below shoulders → False
            # ONLY True when body is prone with feet toward camera (forward fall).
            raw_fall_flag = True

        elif is_extreme_collapse:
            # ── HARD BYPASS C: Height collapse fallback ──────────────────────
            # Gated by height_drop_observed so seated workers never trigger.
            # Use separate sustained frame counter.
            self.height_collapse_frame_counts[p_id] = (
                self.height_collapse_frame_counts.get(p_id, 0) + 1
            )
            raw_fall_flag = (
                self.height_collapse_frame_counts.get(p_id, 0) >= self.required_collapse_frames
            )

        elif kp_trustworthy and is_horizontal and torso_is_fallen and kp_confirms_fall:
            # ── PRIORITY 1: Sideways fall with keypoint confirmation ──────────
            # REQUIRES is_horizontal to prevent bending-over-desk false positives.
            #
            # Previous bug: this branch lacked `is_horizontal`, so a person leaning
            # forward at their desk (torso < 35°, portrait bbox) triggered it.
            # Fix: `is_horizontal` is now a HARD gate — bbox must be at least 1.2 wide.
            # A person leaning forward keeps a portrait bbox → is_horizontal=False → blocked.
            raw_fall_flag = True

        elif is_horizontal and not kp_available:
            # ── PRIORITY 2: Bbox horizontal, no usable keypoints ─────────────
            raw_fall_flag = True

        elif is_falling_fast:
            # ── PRIORITY 3: Rapid downward velocity ──────────────────────────
            raw_fall_flag = True

        elif kp_trustworthy and not torso_is_fallen and not is_horizontal \
                and shoulder_below_ankles is not True:
            # ── EXPLICIT NEGATIVE: trustworthy keypoints say upright ──────────
            # Strong negative: reliable keypoints, upright torso, portrait bbox,
            # no forward fall posture. Explicitly returns False to counteract
            # any incremented fall_frame_counts from earlier transient flags.
            raw_fall_flag = False

        else:
            raw_fall_flag = False

        # Reset height collapse counter on any non-collapse frame
        if not is_height_collapsed:
            self.height_collapse_frame_counts[p_id] = 0

        # ── STEP 5: Temporal smoothing (debounce) ─────────────────────────────
        if p_id not in self.fall_states:
            self.fall_frame_counts[p_id] = 0

        if raw_fall_flag:
            self.fall_frame_counts[p_id] = self.fall_frame_counts.get(p_id, 0) + 1
        else:
            self.fall_frame_counts[p_id] = max(0, self.fall_frame_counts.get(p_id, 0) - 1)

        confirmed_fall = self.fall_frame_counts.get(p_id, 0) >= self.required_fall_frames

        # ── STEP 6: Update fall states ────────────────────────────────────────
        if confirmed_fall:
            self.fall_states[p_id] = True
        elif aspect_ratio < 0.8 and shoulder_below_ankles is not True:
            # Person is upright AND not showing floor posture.
            # The `shoulder_below_ankles is not True` guard prevents accidentally
            # clearing the fall state of someone lying face-down on the floor
            # (their portrait bbox would otherwise satisfy aspect_ratio < 0.8).
            self.fall_states[p_id] = False
            self.fall_frame_counts[p_id] = 0
            self.height_drop_observed.pop(p_id, None)

        return self.fall_states.get(p_id, False)

    # -----------------------------------------------------------------------
    # STANDALONE VIDEO PROCESSING (testing / demo mode)
    # -----------------------------------------------------------------------

    def process_video(self, video_path: str) -> None:
        """Run the fall detector on a local video file (demo / test mode).

        Opens the video, runs YOLO tracking on each frame, and renders
        annotated output to an OpenCV window.  Press ``q`` to quit.

        Args:
            video_path: Path to the input video file.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, DEMO_BUFFER_SIZE)

        if not cap.isOpened():
            logger.error("Error opening video %s", video_path)
            return

        cv2.namedWindow('Fall Detection System', cv2.WINDOW_NORMAL)
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            results = self.model.track(frame, persist=True, device=self.device,
                                       classes=[0], verbose=False)
            current_detections = []

            if results[0].boxes.id is not None:
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    kpts = results[0].keypoints.data.cpu().numpy()
                else:
                    kpts = [None] * len(boxes)

                for box, track_id, kpt in zip(boxes, track_ids, kpts):
                    current_detections.append({
                        'id': track_id,
                        'bbox': box,
                        'keypoints': kpt
                    })

            current_detections = self.tracker.suppress_duplicates(
                current_detections, iou_threshold=DEMO_IOU_THRESHOLD
            )
            self.tracker.update(current_detections, current_time)

            for det in current_detections:
                p_id      = det['id']
                bbox      = det['bbox']
                keypoints = det['keypoints']
                history   = self.tracker.get_history(p_id)

                is_fallen = self.detect_fall(p_id, bbox, history, keypoints)

                color = (0, 0, 255) if is_fallen else (0, 255, 0)
                label = f"ID:{p_id} {'FALL DETECTED' if is_fallen else 'SAFE'}"

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Fall Detection System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FallDetector(model_path='yolov8s-pose.pt', device='mps')
    # detector.process_video('test_video.mp4')
