import numpy as np
import pytest


def test_normalize_keypoints_centers_at_bbox():
    from cnn_fall_detector import normalize_keypoints

    keypoints = np.array([
        [100, 200, 0.9],
        [150, 250, 0.8],
    ])
    bbox = [100, 200, 200, 400]  # 100x200 box

    result = normalize_keypoints(keypoints, bbox)

    assert result.shape == (2, 3)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[1, 0] == pytest.approx(0.5)
    assert result[1, 1] == pytest.approx(0.25)
    assert result[0, 2] == pytest.approx(0.9)  # conf unchanged


def test_extract_features_shape_and_dtype():
    from cnn_fall_detector import extract_features, NUM_KEYPOINTS, INPUT_FEATURES

    record = {
        "bbox_xyxy": [100, 100, 200, 300],
        "keypoints": [[150, 150 + i*10, 0.9] for i in range(NUM_KEYPOINTS)],
        "frame_w": 640,
        "frame_h": 480,
    }

    features = extract_features(record)

    assert features.shape == (INPUT_FEATURES,)
    assert features.dtype == np.float32


def test_extract_features_bbox_features():
    from cnn_fall_detector import extract_features, NUM_KEYPOINTS

    record = {
        "bbox_xyxy": [0, 0, 100, 200],  # w=100, h=200, cx=50, cy=100
        "keypoints": [[50, 100, 0.9] for _ in range(NUM_KEYPOINTS)],
        "frame_w": 200,
        "frame_h": 400,
    }

    features = extract_features(record)

    # Last 7 features are bbox: cx/fw, cy/fh, w/fw, h/fh, w/h, area_ratio, vis_ratio
    bbox_feats = features[-7:]
    assert bbox_feats[0] == pytest.approx(50/200)   # cx / frame_w
    assert bbox_feats[1] == pytest.approx(100/400)  # cy / frame_h
    assert bbox_feats[2] == pytest.approx(100/200)  # w / frame_w
    assert bbox_feats[3] == pytest.approx(200/400)  # h / frame_h
    assert bbox_feats[4] == pytest.approx(0.5)      # w / h


def test_cnn_model_forward_pass():
    import torch
    from cnn_fall_detector import FallCNN1D, INPUT_FEATURES, SEQ_LEN

    model = FallCNN1D(INPUT_FEATURES, num_classes=2)
    batch = torch.randn(4, INPUT_FEATURES, SEQ_LEN)
    output = model(batch)

    assert output.shape == (4, 2)


def test_cnn_model_output_is_logits():
    import torch
    from cnn_fall_detector import FallCNN1D, INPUT_FEATURES, SEQ_LEN

    model = FallCNN1D()
    x = torch.randn(1, INPUT_FEATURES, SEQ_LEN)

    output = model(x)
    probs = torch.softmax(output, dim=1)

    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_falldetector_loads_model(tmp_path):
    import torch
    from cnn_fall_detector import FallDetector, FallCNN1D

    dummy_model = FallCNN1D()
    ckpt_path = tmp_path / "dummy.pt"
    torch.save({"model_state_dict": dummy_model.state_dict()}, ckpt_path)

    detector = FallDetector(str(ckpt_path), device="cpu")

    assert detector.model is not None
    assert detector.threshold == 0.7
    assert detector.seq_len == 30


def test_falldetector_predict_window(tmp_path):
    import torch
    from cnn_fall_detector import FallDetector, FallCNN1D, SEQ_LEN, NUM_KEYPOINTS

    dummy_model = FallCNN1D()
    ckpt_path = tmp_path / "dummy.pt"
    torch.save({"model_state_dict": dummy_model.state_dict()}, ckpt_path)

    detector = FallDetector(str(ckpt_path), device="cpu")

    window = [{
        "bbox_xyxy": [100, 100, 200, 300],
        "keypoints": [[150, 150, 0.9] for _ in range(NUM_KEYPOINTS)],
        "frame_w": 640,
        "frame_h": 480,
    } for _ in range(SEQ_LEN)]

    is_fall, conf = detector.predict_window(window)

    assert isinstance(is_fall, bool)
    assert 0.0 <= conf <= 1.0
