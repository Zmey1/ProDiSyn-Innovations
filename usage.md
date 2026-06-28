# cnn_fall_detector.py — Usage

## Basic inference

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4
```

## JSON output

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 -o results.json
```

## Annotated video (single video only)

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 --annotate output.mp4
```

## JSON output + annotated video

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 -o results.json --annotate output.mp4
```

## Batch processing

```bash
conda run -n fallgpu python cnn_fall_detector.py videos/*.mp4 -o all_results.json
```

## Custom threshold

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 --threshold 0.8
```

## Suppress progress output

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 --quiet
```

## Specify device

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 --device cuda
conda run -n fallgpu python cnn_fall_detector.py video.mp4 --device cpu
```

## Custom model path

```bash
conda run -n fallgpu python cnn_fall_detector.py video.mp4 --model runs/meow1/fall_cnn1d_v4_best.pt
```

## Output locations

| Output | Flag | Location |
|--------|------|----------|
| JSON results | `-o results.json` | Path you specify — relative to cwd |
| Annotated video | `--annotate output.mp4` | Path you specify — relative to cwd |
| Console fall events | (always) | stdout — `Track N: frames X-Y (conf=0.XX)` |
| Progress counter | (default) | stdout — `Frame N/Total (XX%)`, suppressed with `--quiet` |

**Example output paths:**
```bash
# saves to project root
conda run -n fallgpu python cnn_fall_detector.py video.mp4 -o results.json --annotate annotated.mp4

# saves to a dedicated output dir
mkdir -p out
conda run -n fallgpu python cnn_fall_detector.py video.mp4 -o out/results.json --annotate out/annotated.mp4
```

**JSON event format:**
```json
[
  {
    "video_name": "fall-01-cam0-rgb.mp4",
    "track_id": 1,
    "start_frame": 214,
    "end_frame": 216,
    "confidence": 0.7615
  }
]
```

## All flags

```
positional arguments:
  videos                Video file(s) to process

options:
  --model PATH          Model checkpoint (default: runs/meow1/fall_cnn1d_v4_best.pt)
  --threshold FLOAT     Fall confidence threshold (default: 0.7)
  --device DEVICE       Device: cuda, cpu, mps
  -o, --output PATH     Save results as JSON
  --annotate PATH       Create annotated video (single video only)
  -q, --quiet           Suppress progress output
```

## Run tests

```bash
conda run -n fallgpu python -m pytest tests/test_cnn_fall_detector.py -v
```

## Integration test (requires model + video)

```bash
conda run -n fallgpu python -m pytest tests/test_cnn_fall_detector.py::test_integration_real_video -v -s
```
