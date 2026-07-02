# Offline VideoMAE fall detector

This folder contains two local inference applications:

- `test_fall_model.py` analyzes a video file.
- `webcam_fall_app.py` analyzes rolling clips from a webcam.

Both applications load the bundled model from `model/`. They do not train or
fine-tune it and do not download model files at runtime.

## Setup

Install Git LFS before cloning or pulling the model weights:

```bash
git lfs install
git lfs pull
```

Create or activate the Python environment, then install dependencies:

```bash
conda activate fallgpu
cd /home/zmey1/VSCODE_FILES/prodesyn/final_test
pip install -r requirements.txt
```

The configured PyTorch build uses CUDA 12.1. The applications accept
`--device auto`, `--device cuda`, or `--device cpu`.

## Test the included video

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

## Run the webcam application

```bash
python webcam_fall_app.py --camera-index 0 --device auto
```

Press `q` to stop. Finalized events are written under
`outputs/realtime_events/` as JSON, JPG, and MP4 files.

## Event decisions

- `Partial Fall`: a confirmed fall candidate followed by `StandUp`,
  `Standing`, or `Walking` evidence.
- `Full Fall`: a confirmed fall candidate with continued ground evidence and
  no recovery during the recovery window.
- `Fall Detected, Recovery Unknown`: a confirmed candidate without enough
  evidence to determine recovery or sustained ground position.
- `Normal`: no confirmed fall candidate.

The model is licensed for non-commercial use under CC BY-NC 4.0. See
`model/MODEL_NOTICE.md`.
