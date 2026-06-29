"""Evaluate rule-based fall detector on test set using pre-extracted tracks."""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from fall_detector import FallDetector
from tracker import PersonTracker


def load_tracks(track_file: Path) -> list:
    """Load pre-extracted YOLO tracks from JSON."""
    with open(track_file) as f:
        return json.load(f)


def run_rule_based_detector(tracks: list, fps: float = 30.0) -> dict:
    """Run rule-based detector on pre-extracted tracks.

    Returns dict with:
        fall_detected: bool
        fall_frames: list of frame indices where fall was detected
        first_fall_frame: int or None
    """
    detector = FallDetector()
    tracker = PersonTracker(history_size=30, max_inactive_time=2.0)

    fall_frames = []
    frames_by_idx = defaultdict(list)

    for track in tracks:
        frames_by_idx[track['frame_idx']].append(track)

    for frame_idx in sorted(frames_by_idx.keys()):
        frame_tracks = frames_by_idx[frame_idx]
        current_time = frame_idx / fps

        detections = []
        for t in frame_tracks:
            detections.append({
                'id': t['track_id'],
                'bbox': tuple(t['bbox_xyxy']),
                'keypoints': t.get('keypoints')
            })

        detections = tracker.suppress_duplicates(detections, iou_threshold=0.5)
        tracker.update(detections, current_time)

        for det in detections:
            history = tracker.get_history(det['id'])
            kpts = det['keypoints']
            if kpts:
                import numpy as np
                kpts = np.array(kpts)

            is_fallen = detector.detect_fall(
                det['id'], det['bbox'], history, kpts
            )
            if is_fallen:
                fall_frames.append(frame_idx)

    return {
        'fall_detected': len(fall_frames) > 0,
        'fall_frames': fall_frames,
        'first_fall_frame': fall_frames[0] if fall_frames else None
    }


def match_video_to_track(video_uid: str, track_files: dict) -> Path:
    """Match video UID to track file."""
    if video_uid in track_files:
        return track_files[video_uid]

    uid_normalized = video_uid.replace('-', '_').lower()
    for tf_name, tf_path in track_files.items():
        tf_normalized = tf_name.lower()
        if uid_normalized in tf_normalized or tf_normalized in uid_normalized:
            return tf_path
    return None


def main():
    proj_dir = Path(__file__).parent

    with open(proj_dir / 'test_manifest.json') as f:
        test_videos = json.load(f)['videos']

    track_dir = proj_dir / 'extras' / 'outputs' / 'tracks'
    track_files = {f.stem.replace('_tracks', ''): f for f in track_dir.glob('*_tracks.json')}

    tp, fp, fn, tn = 0, 0, 0, 0
    results = []

    for v in test_videos:
        uid = v['video_uid']
        has_fall_gt = v['has_fall']

        track_file = match_video_to_track(uid, track_files)
        if not track_file:
            print(f'SKIP: {uid} (no track file)')
            continue

        tracks = load_tracks(track_file)
        if not tracks:
            print(f'SKIP: {uid} (empty tracks)')
            continue

        result = run_rule_based_detector(tracks)
        pred_fall = result['fall_detected']

        if has_fall_gt and pred_fall:
            tp += 1
            status = 'TP'
        elif not has_fall_gt and pred_fall:
            fp += 1
            status = 'FP'
        elif has_fall_gt and not pred_fall:
            fn += 1
            status = 'FN'
        else:
            tn += 1
            status = 'TN'

        results.append({
            'video_uid': uid,
            'gt': has_fall_gt,
            'pred': pred_fall,
            'status': status,
            'first_fall_frame': result['first_fall_frame']
        })
        print(f'{status}: {uid}')

    print('\n' + '='*60)
    print('RULE-BASED DETECTOR RESULTS')
    print('='*60)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Evaluated: {total} videos')
    print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
    print(f'Precision: {precision:.1%}')
    print(f'Recall: {recall:.1%}')
    print(f'F1: {f1:.1%}')

    with open(proj_dir / 'rule_based_results.json', 'w') as f:
        json.dump({
            'metrics': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                       'precision': precision, 'recall': recall, 'f1': f1},
            'results': results
        }, f, indent=2)
    print(f'\nDetailed results saved to rule_based_results.json')


if __name__ == '__main__':
    main()
