"""Minimal PersonTracker for fall_detector.py integration."""

from collections import defaultdict
from typing import Dict, List


class PersonTracker:
    """Maintains per-person spatial history for fall detection."""

    def __init__(self, history_size: int = 30, max_inactive_time: float = 2.0):
        self.history_size = history_size
        self.max_inactive_time = max_inactive_time
        self.histories: Dict[int, List[dict]] = defaultdict(list)
        self.last_seen: Dict[int, float] = {}

    def suppress_duplicates(
        self, detections: List[dict], iou_threshold: float = 0.5
    ) -> List[dict]:
        """Remove duplicate detections based on IoU overlap."""
        if not detections:
            return []

        keep = []
        for det in detections:
            is_dup = False
            for kept in keep:
                if self._iou(det['bbox'], kept['bbox']) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(det)
        return keep

    def update(self, detections: List[dict], current_time: float) -> None:
        """Update history with current detections."""
        seen_ids = set()
        for det in detections:
            pid = det['id']
            seen_ids.add(pid)
            self.last_seen[pid] = current_time
            self.histories[pid].append({
                'bbox': det['bbox'],
                'time': current_time
            })
            if len(self.histories[pid]) > self.history_size:
                self.histories[pid] = self.histories[pid][-self.history_size:]

        stale = [
            pid for pid, t in self.last_seen.items()
            if current_time - t > self.max_inactive_time and pid not in seen_ids
        ]
        for pid in stale:
            del self.histories[pid]
            del self.last_seen[pid]

    def get_history(self, person_id: int) -> List[dict]:
        """Return bbox/time history for a person ID."""
        return self.histories.get(person_id, [])

    @staticmethod
    def _iou(box1, box2) -> float:
        """Calculate IoU between two bboxes (x1,y1,x2,y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
