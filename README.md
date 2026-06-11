# Fall Detection V1

Real-time fall detection using pose estimation and temporal CNN classification.

## Results

| Metric | Value |
|--------|-------|
| **Recall** | 81.0% |
| **Precision** | 91.9% |
| **F1 Score** | 86.1% |
| False Alarms | 3/67 videos |
| Detection Delay | 6.9 frames (~230ms @ 30fps) |

### Confusion Matrix
```
                Predicted
              Fall    Normal
Actual Fall    34        8     (8 missed)
Actual Normal   3       22     (3 false alarms)
```

## Architecture

```
Video → YOLOv8n-pose → ByteTrack → 30-frame windows → CNN1D → Fall/Normal
```

- **Pose Model**: YOLOv8n-pose (17 keypoints)
- **Tracker**: ByteTrack (streaming mode)
- **Classifier**: 1D CNN over 30-frame sequences
- **Features**: 58 per frame (51 keypoint + 7 bbox)
- **Threshold**: 0.7 with 3-window confirmation

## Dataset

| Source | Videos | Falls | Normal |
|--------|--------|-------|--------|
| Le2i | 190 | 156 | 34 |
| GMDCSA24 | 160 | 79 | 81 |
| URFall | 70 | 30 | 40 |
| **Total** | **420** | **265** | **155** |

**Split**: 70% train (292) / 15% val (61) / 15% test (67)

> **Note**: Current dataset is small with limited diversity. Performance expected to improve with larger, properly annotated real-world data.

## Files

```
├── local_f_kaggle.ipynb          # Training + evaluation notebook
├── manifests/
│   ├── train_manifest_kaggle.json
│   ├── val_manifest_kaggle.json
│   └── test_manifest_kaggle.json
└── results/v4_final/
    ├── fall_cnn1d_v4_best.pt     # Trained model (355KB)
    ├── training_history.json
    ├── train_features.pkl        # Pre-extracted features
    └── val_features.pkl
```

## Usage

### Kaggle
1. Add dataset: `ayushkumar10/Fall-Detection-v4-industrial`
2. Run `local_f_kaggle.ipynb`

### Local
```python
from ultralytics import YOLO
import torch

# Load models
pose_model = YOLO("yolov8n-pose.pt")
classifier = torch.load("results/v4_final/fall_cnn1d_v4_best.pt")
```

## Training

- Epochs: 30
- Optimizer: AdamW (lr=1e-3, wd=1e-4)
- Scheduler: Cosine annealing
- Class weights: Balanced
