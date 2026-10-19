import json

import numpy as np
import pytest

from utils.metrics import Evaluator


def test_counts_are_pooled_per_lead_time_then_averaged():
    # Lead 1: TP=1,FN=1,FP=0,TN=2 -> CSI=1/2,HSS=1/2.
    # Lead 2: TP=1,FN=0,FP=2,TN=1 -> CSI=1/3,HSS=1/5.
    truth = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=float).reshape(1, 2, 1, 1, 4)
    pred = np.array([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=float).reshape(1, 2, 1, 1, 4)
    metric = Evaluator(2, 1, thresholds=(1,), compute_image_metrics=False)
    metric.evaluate(truth, pred)
    result = metric.done(is_main_process=False)
    assert result["csi"] == pytest.approx((1 / 2 + 1 / 3) / 2)
    assert result["hss"] == pytest.approx((1 / 2 + 1 / 5) / 2)
    assert result["csi"] != pytest.approx(2 / 5)


def test_chunking_does_not_change_metrics_and_json_is_finite(tmp_path):
    rng = np.random.default_rng(3)
    truth = rng.random((3, 2, 1, 16, 16), dtype=np.float32)
    pred = rng.random(truth.shape, dtype=np.float32)
    one, chunks = Evaluator(2, 90, (12, 32)), Evaluator(2, 90, (12, 32))
    one.evaluate(truth, pred)
    for i in range(3):
        chunks.evaluate(truth[i : i + 1], pred[i : i + 1])
    a, b = one.done(False), chunks.done(json_save_dir=tmp_path)
    for key in ("csi", "hss", "mse", "ssim", "rmse"):
        assert a[key] == pytest.approx(b[key], rel=1e-6)
    json.loads((tmp_path / "eval_results.json").read_text())


def test_float32_threshold_scaling_matches_original():
    truth = np.full((1, 1, 1, 1, 1), 32 / 90, dtype=np.float32)
    metric = Evaluator(1, 90, (32,), compute_image_metrics=False)
    metric.evaluate(truth, truth)
    assert metric.done(False)["csi"] == 1.0


def test_perfect_images_and_empty_input():
    metric = Evaluator(1, 255, (16,))
    with pytest.raises(ValueError, match="empty"):
        metric.done(False)
    image = np.ones((1, 1, 1, 16, 16))
    metric.evaluate(image, image)
    result = metric.done(False)
    assert result["mse"] == 0
    assert result["ssim"] == pytest.approx(1)
