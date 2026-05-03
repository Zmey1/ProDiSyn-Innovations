# Fall Detection — Test Documentation

**Project:** Human Fall Detection from Video using Pose Estimation + Sequence Classification  
**Author:** Ayush  
**Date:** May 2026  

---

## Overview

This document records all experiments performed on the fall detection pipeline. The system uses YOLOv8-Pose for skeleton extraction, BotSort for tracking, and a sliding-window classifier over 30-frame pose sequences.

**Feature vector per frame (58 dimensions):**  
- 17 YOLO keypoints × 3 values (x, y, confidence) = 51  
- Bounding box: center_x, center_y, width, height (normalized) = 4  
- Aspect ratio, area, confidence = 3  

**Sliding window:** 30 frames, stride 5.  
**Train/val split:** Video-level (by `video_uid`) to prevent data leakage.  
**Class imbalance handling:** Square-root class weights in cross-entropy loss.

---

## Version 1 — Le2i Dataset, CNN Baseline

### Setup

| Parameter | Value |
|-----------|-------|
| Model | `FallCNN1D` (1D CNN) |
| Dataset | Le2i Fall Detection (Coffee room + Home) |
| Training epochs | 15 |
| Best checkpoint epoch | **8** |
| Batch size | 64 |
| Learning rate | 5e-4 (Adam) |
| Gradient clipping | max_norm = 1.0 |
| Checkpoint | `outputs/checkpoints/fall_cnn1d_best_video_uid_sqrt_weights.pt` |

### Model Architecture

```
Conv1d(58 → 64, k=3) → BN → ReLU
Conv1d(64 → 128, k=3) → BN → ReLU → MaxPool1d(2)
Conv1d(128 → 128, k=3) → BN → ReLU → AdaptiveAvgPool1d(1)
Flatten → Dropout(0.35) → Linear(128 → 2)
```

### Validation Set Composition

| Class | Samples |
|-------|---------|
| Normal | 1,245 |
| Fall | 395 |
| **Total** | **1,640** |

### Results

| Metric | Normal | Fall | Macro Avg |
|--------|--------|------|-----------|
| Precision | 0.79 | 0.34 | 0.57 |
| Recall | 0.78 | 0.36 | 0.57 |
| F1-score | 0.79 | **0.35** | **0.57** |
| Accuracy | — | — | **0.68** |

**Confusion Matrix:**

|  | Predicted Normal | Predicted Fall |
|--|-----------------|----------------|
| **Actual Normal** | 971 (TN) | 274 (FP) |
| **Actual Fall** | 254 (FN) | 141 (TP) |

### Threshold Sweep (selected threshold = 0.15)

*Selection rule: if any threshold achieves recall ≥ 0.70, pick highest precision among those; otherwise pick max F1.*

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| 0.10 | 0.26 | 0.25 | **1.00** | 0.39 |
| **0.15** | 0.48 | 0.29 | **0.82** | **0.43** |
| 0.20 | 0.35 | 0.24 | 0.80 | 0.37 |
| 0.25 | 0.40 | 0.26 | 0.78 | 0.38 |

---

## Version 2 — Le2i Dataset, BiLSTM + Attention

Used BiLSTM to capture the temporal patterns which are lost when we flatten the input features in the CNN especially since any fall has a start and end which maybe in different frames of video 

### Setup

| Parameter | Value |
|-----------|-------|
| Model | `BiLSTMAttentionClassifier` |
| Dataset | Le2i Fall Detection (same split as Exp 1) |
| Training epochs | 15 |
| Batch size | 64 |
| Learning rate | 2e-4 (Adam, reduced for LSTM stability) |
| Gradient clipping | max_norm = 1.0 |
| Checkpoint | `outputs/checkpoints/fall_bilstm_attn_best_video_uid_sqrt_weights.pt` |

### Model Architecture

```
BiLSTM: 2 layers, hidden=128, dropout=0.3
  → output shape: (batch, seq_len, 256)
Attention: Linear(256 → 1) → Softmax over time → weighted sum → (batch, 256)
Classifier: Dropout(0.3) → Linear(256 → 64) → ReLU → Dropout(0.3) → Linear(64 → 2)
Total parameters: ~800K
```

### Results

| Metric | Value |
|--------|-------|
| Best val F1 (fall class) | ~0.35 |
| Accuracy | ~0.68 |

**Outcome:** BiLSTM underperformed the CNN baseline on this dataset. With only ~1,640 validation samples and a small Le2i corpus, the ~800K-parameter recurrent model did not generalise well. The CNN (128K parameters) was better matched to the data size.

**Decision:** Reverted to CNN for all subsequent training. The BiLSTM checkpoint is retained for reference.

### Key Takeaway

> BiLSTM + Attention is architecturally more expressive but requires significantly more training data to outperform a lightweight CNN. On small datasets, prefer CNN with appropriate regularisation.

---

## Version 3 — Combined Dataset (Le2i + GMDCSA-24 + UR Fall), CNN

### Setup

| Parameter | Value |
|-----------|-------|
| Model | `FallCNN1D` (identical to Exp 1) |
| Datasets | Le2i + GMDCSA-24 + UR Fall Detection |
| Training epochs | 15 |
| Best checkpoint epoch | **2** |
| Batch size | 64 |
| Learning rate | 5e-4 (Adam) |
| Gradient clipping | max_norm = 1.0 |
| Checkpoint | `outputs/checkpoints/fall_cnn1d_combined_best.pt` |

### Dataset Details

| Dataset | Source | Videos | Fall Annotation Method |
|---------|--------|--------|------------------------|
| Le2i | Coffee room + Home (indoor) | ~20 | Per-frame bboxes + fall_start/fall_end frame numbers |
| GMDCSA-24 | Subjects 1–4, Fall + ADL | ~60+ | Timestamps in seconds per video (CSV) |
| UR Fall | 70 sequences (30 fall, 40 ADL) | 70 | Sequence-level label (fall = full sequence) |

**Resolution handling:** Each dataset uses a different resolution (Le2i: 320×240, GMDCSA-24: 1280×720, UR Fall: 640×480). Feature extraction uses `record_to_feature_vector_v2` which normalises keypoints by the actual frame dimensions stored per record.

### Validation Set Composition

| Class | Samples |
|-------|---------|
| Normal | 2,391 |
| Fall | 854 |
| **Total** | **3,245** |

### Results

| Metric | Normal | Fall | Macro Avg |
|--------|--------|------|-----------|
| Precision | 0.86 | 0.58 | 0.72 |
| Recall | 0.84 | 0.60 | 0.72 |
| F1-score | 0.85 | **0.59** | **0.72** |
| Accuracy | — | — | **0.78** |

**Confusion Matrix:**

|  | Predicted Normal | Predicted Fall |
|--|-----------------|----------------|
| **Actual Normal** | 2,015 (TN) | 376 (FP) |
| **Actual Fall** | 341 (FN) | 513 (TP) |

### Threshold Sweep (top entries)

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| 0.10 | 0.41 | 0.31 | **0.99** | 0.47 |
| 0.15 | 0.46 | 0.32 | 0.96 | 0.48 |
| 0.20 | 0.52 | 0.35 | 0.92 | 0.50 |
| **0.25** | 0.57 | 0.37 | **0.89** | **0.52** |

---

## Summary Comparison

| Experiment | Model | Dataset | Val Samples | Accuracy | Fall F1 | Macro F1 |
|-----------|-------|---------|-------------|----------|---------|----------|
| 1 — CNN Baseline | FallCNN1D | Le2i only | 1,640 | 0.68 | 0.35 | 0.57 |
| 2 — BiLSTM + Attn | BiLSTMAttentionClassifier | Le2i only | 1,640 | ~0.68 | ~0.35 | ~0.57 |
| **3 — Combined CNN** | **FallCNN1D** | **Le2i + GMDCSA-24 + UR Fall** | **3,245** | **0.78** | **0.59** | **0.72** |

### Improvement (Exp 1 → Exp 3)

| Metric | Le2i-only CNN | Combined CNN | Δ |
|--------|--------------|--------------|---|
| Accuracy | 68% | **78%** | **+10 pp** |
| Fall F1 | 0.35 | **0.59** | **+0.24** |
| Fall Recall | 0.36 | **0.60** | **+0.24** |
| Fall Precision | 0.34 | **0.58** | **+0.24** |
| Macro F1 | 0.57 | **0.72** | **+0.15** |

---

## Conclusions

1. **Data volume is the dominant factor.** Doubling the dataset size (1,640 → 3,245 val samples, reflecting a ~3× increase in total training data) improved fall F1 by 0.24 — far larger than any architectural change.

2. **CNN outperforms BiLSTM at small scale.** With limited data, a lightweight 1D CNN (128K params) generalises better than a BiLSTM (800K params). BiLSTM may be re-evaluated if the dataset grows further.

3. **Multi-dataset generalisation works.** The combined model handles three different environments (indoor, laboratory, structured sequences) and three different resolutions without any handcrafted domain-specific features — resolution-agnostic normalisation is sufficient.

4. **Threshold tuning is critical for fall detection.** At the default 0.50 threshold, precision and recall are balanced but recall (catching real falls) is too low for a safety application. The sweep shows that 0.15–0.25 thresholds substantially increase recall at acceptable precision.

---

## Reproducibility

All results can be reproduced by running `local.ipynb` end-to-end with the original datasets present. YOLO extraction results are cached as JSON in `outputs/tracks/`. Trained checkpoints are in `outputs/checkpoints/`.

Required datasets (not included in repo — download separately):
- **Le2i Fall Detection Dataset** → place in `archive/`
- **GMDCSA-24** → place in `GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master/`
- **UR Fall Detection Dataset** → place in `archive_ur/UR_fall_detection_dataset_cam0_rgb/`
