# Offline VideoMAE fall detector

This folder contains a local fall-detection model with a GUI app plus two
command-line tools:

- `fall_detection_gui.py` — PyQt6 app (Single Video, Batch, Webcam tabs).
- `test_fall_model.py` analyzes a video file.
- `webcam_fall_app.py` analyzes rolling clips from a webcam.

All three load the bundled model from `model/`. They do not train or
fine-tune it and do not download model files at runtime.

## Quick start

```bash
git lfs install
git lfs pull
conda activate fallgpu
cd /home/zmey1/VSCODE_FILES/prodesyn/final_test
pip install -r requirements.txt
python fall_detection_gui.py
```

The configured PyTorch build uses CUDA 12.1. All apps accept
`--device auto`, `--device cuda`, or `--device cpu` (or the equivalent
Settings option in the GUI).

## GUI app

Tabs: Single Video, Batch, Webcam. Device and detection threshold are under
Settings. If running on CPU, the app will show a clear warning — CPU
inference is roughly 15-20x slower than GPU and webcam detection will lag
behind the live feed.

## Event decisions

- `Partial Fall`: a confirmed fall candidate followed by `StandUp`,
  `Standing`, or `Walking` evidence.
- `Full Fall`: a confirmed fall candidate with continued ground evidence and
  no recovery during the recovery window.
- `Fall Detected, Recovery Unknown`: a confirmed candidate without enough
  evidence to determine recovery or sustained ground position.
- `Normal`: no confirmed fall candidate.

## Command-line usage

### Single video

From inside `final_test`:

```bash
python test_fall_model.py --video fall_20260629_182439.mp4 --device auto
```

The script samples the whole video and then evaluates 2-second sliding clips
with a 1-second stride. It writes:

- `outputs/fall_clip_scores.csv`
- `outputs/fall_events.json`

Useful options:

```text
--threshold 0.7
--clip-seconds 2.0
--stride-seconds 1.0
--consecutive 2
--recovery-window-seconds 8.0
--recovery-threshold 0.5
--ground-threshold 0.5
--device auto
```

### Webcam

```bash
python webcam_fall_app.py --camera-index 0 --device auto
```

Press `q` to stop. Finalized events are written under
`outputs/realtime_events/` as JSON, JPG, and MP4 files.
