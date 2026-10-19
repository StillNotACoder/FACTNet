"""Streaming radar metrics with the original FACTNet aggregation convention.

Inputs have shape [batch, lead time, 1, height, width] and are normalized to
[0, 1]. See ``docs/metrics.md`` for scales, averaging, and undefined scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLDS = (20, 30, 35, 40)


def cal_ssim(pred, true, data_range=255.0):
    """Return the original 11x11 Gaussian-window SSIM for two 2-D images.

    OpenCV is imported only when image metrics are requested. The five-pixel
    border is excluded, matching the original research implementation.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "SSIM requires opencv-python-headless. Install the project "
            "requirements, or use compute_image_metrics=False."
        ) from exc

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != true.shape or min(pred.shape) < 11:
        raise ValueError("SSIM requires matching 2-D images of at least 11x11 pixels.")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("SSIM data_range must be finite and positive.")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("SSIM inputs must contain only finite values.")

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu_pred = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu_true = cv2.filter2D(true, -1, window)[5:-5, 5:-5]
    pred_variance = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu_pred**2
    true_variance = cv2.filter2D(true**2, -1, window)[5:-5, 5:-5] - mu_true**2
    covariance = cv2.filter2D(pred * true, -1, window)[5:-5, 5:-5] - mu_pred * mu_true
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu_pred * mu_true + c1) * (2 * covariance + c2)) / (
        (mu_pred**2 + mu_true**2 + c1) * (pred_variance + true_variance + c2)
    )
    result = float(ssim.mean())
    if not np.isfinite(result):
        raise ValueError("SSIM produced a non-finite value; check the input scale.")
    return result


def _as_numpy(value):
    """Accept NumPy arrays or tensor-like objects without importing PyTorch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise ValueError("Expected a real numeric array.")
        return array
    except (TypeError, ValueError) as exc:
        raise ValueError("Metric inputs must be numeric arrays or tensors.") from exc


def _divide_or_zero(numerator, denominator):
    """The original evaluation assigns zero to an undefined categorical score."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


class Evaluator:
    """Accumulate sufficient statistics, using memory independent of dataset size.

    Categorical counts are summed across samples separately at each lead time.
    Scores are then averaged over lead times and thresholds, in that order.
    Image errors are computed per image and averaged over samples and lead times.
    """

    def __init__(
        self,
        seq_len,
        value_scale,
        thresholds=DEFAULT_THRESHOLDS,
        *,
        compute_image_metrics=True,
    ):
        if (
            isinstance(seq_len, (bool, np.bool_))
            or not isinstance(seq_len, (int, np.integer))
            or seq_len <= 0
        ):
            raise ValueError("seq_len must be a positive integer.")
        if (
            not np.isscalar(value_scale)
            or not np.isfinite(value_scale)
            or not 0 < value_scale <= 65535
        ):
            raise ValueError("value_scale must be finite and in (0, 65535] for uint16 conversion.")
        try:
            threshold_array = np.asarray(tuple(thresholds), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("thresholds must be a nonempty sequence of finite numbers.") from exc
        if (
            threshold_array.ndim != 1
            or not threshold_array.size
            or not np.isfinite(threshold_array).all()
        ):
            raise ValueError("thresholds must be a nonempty sequence of finite numbers.")
        if np.unique(threshold_array).size != threshold_array.size:
            raise ValueError("thresholds must not contain duplicates.")

        self.seq_len = int(seq_len)
        self.value_scale = float(value_scale)
        self.thresholds = [int(x) if x.is_integer() else float(x) for x in threshold_array]
        self.compute_image_metrics = bool(compute_image_metrics)
        self.total = 0
        self._spatial_shape = None
        # Last axis: hits, misses, false alarms, correct negatives.
        self._counts = np.zeros((len(self.thresholds), self.seq_len, 4), dtype=np.int64)
        self._image_sums = {
            name: np.zeros(self.seq_len, dtype=np.float64)
            for name in ("mse", "mae", "rmse", "ssim")
        }

    def evaluate(self, true_batch, pred_batch):
        """Add a batch; reject malformed or non-finite inputs before updating state."""
        true = _as_numpy(true_batch)
        pred = _as_numpy(pred_batch)
        if true.shape != pred.shape:
            raise ValueError(f"True and predicted shapes differ: {true.shape} vs {pred.shape}.")
        if true.ndim != 5 or true.shape[2] != 1:
            raise ValueError("Expected shape [batch, lead time, 1, height, width].")
        if true.shape[0] == 0 or min(true.shape[-2:]) == 0:
            raise ValueError("Cannot evaluate an empty batch or empty image.")
        if true.shape[1] != self.seq_len:
            raise ValueError(f"Expected {self.seq_len} lead times, got {true.shape[1]}.")
        if not np.isfinite(true).all() or not np.isfinite(pred).all():
            raise ValueError("Metric inputs must contain only finite values.")
        if self._spatial_shape is not None and true.shape[-2:] != self._spatial_shape:
            raise ValueError("All batches in one evaluation must have the same image dimensions.")
        if self.compute_image_metrics and min(true.shape[-2:]) < 11:
            raise ValueError("Image metrics require images of at least 11x11 pixels for SSIM.")

        true = np.clip(true, 0.0, 1.0)
        pred = np.clip(pred, 0.0, 1.0)
        true_scaled = true * self.value_scale
        pred_scaled = pred * self.value_scale
        true_int = true_scaled.astype(np.uint16)
        pred_int = pred_scaled.astype(np.uint16)

        batch_counts = np.zeros_like(self._counts)
        axes = (0, 2, 3, 4)
        for index, threshold in enumerate(self.thresholds):
            observed = true_int >= threshold
            forecast = pred_int >= threshold
            batch_counts[index, :, 0] = np.sum(observed & forecast, axis=axes)
            batch_counts[index, :, 1] = np.sum(observed & ~forecast, axis=axes)
            batch_counts[index, :, 2] = np.sum(~observed & forecast, axis=axes)
            batch_counts[index, :, 3] = np.sum(~observed & ~forecast, axis=axes)

        image_sums = {}
        if self.compute_image_metrics:
            error = pred_scaled - true_scaled
            per_image_mse = np.mean(error**2, axis=(2, 3, 4))
            image_sums["mse"] = per_image_mse.sum(axis=0)
            image_sums["mae"] = np.mean(np.abs(error), axis=(2, 3, 4)).sum(axis=0)
            image_sums["rmse"] = np.sqrt(per_image_mse).sum(axis=0)
            image_sums["ssim"] = np.zeros(self.seq_len, dtype=np.float64)
            for sample in range(true.shape[0]):
                for lead in range(self.seq_len):
                    image_sums["ssim"][lead] += cal_ssim(
                        pred_scaled[sample, lead, 0],
                        true_scaled[sample, lead, 0],
                        data_range=self.value_scale,
                    )
            if any(not np.isfinite(values).all() for values in image_sums.values()):
                raise ValueError("Image metrics produced non-finite values.")

        self._counts += batch_counts
        for name, values in image_sums.items():
            self._image_sums[name] += values
        self.total += int(true.shape[0])
        self._spatial_shape = true.shape[-2:]

    def done(self, is_main_process=True, json_save_dir=None):
        """Return Python scalars/lists and optionally write strict ``eval_results.json``.

        Calling this method does not reset or otherwise change accumulated state.
        Undefined categorical ratios are zero; an empty evaluation is an error.
        """
        if self.total == 0:
            raise ValueError(
                "Cannot finalize an empty evaluation; call evaluate() with samples first."
            )

        hits, misses, false_alarms, correct_negatives = np.moveaxis(
            self._counts.astype(np.float64), -1, 0
        )
        scores = {
            "csi": _divide_or_zero(hits, hits + misses + false_alarms),
            "far": _divide_or_zero(false_alarms, hits + false_alarms),
            "pod": _divide_or_zero(hits, hits + misses),
            "hss": _divide_or_zero(
                2 * (hits * correct_negatives - misses * false_alarms),
                (hits + misses) * (misses + correct_negatives)
                + (hits + false_alarms) * (false_alarms + correct_negatives),
            ),
        }
        result = {name: float(values.mean(axis=1).mean()) for name, values in scores.items()}
        result["num_samples"] = self.total
        result["seq_len"] = self.seq_len
        result["value_scale"] = self.value_scale
        result["thresholds"] = list(self.thresholds)
        result["compute_image_metrics"] = self.compute_image_metrics
        result["lead_times"] = list(range(1, self.seq_len + 1))
        result["per_lead_time"] = {
            name: values.mean(axis=0).tolist() for name, values in scores.items()
        }
        result["summary"] = {
            "total_avg_csi": result["csi"],
            "avg_far": result["far"],
            "avg_pod": result["pod"],
            "avg_hss": result["hss"],
        }
        result["threshold_details"] = {}
        count_names = ("hits", "misses", "falsealarms", "correctnegs")
        for index, threshold in enumerate(self.thresholds):
            detail = {}
            for name, values in scores.items():
                detail[f"avg_{name}"] = float(values[index].mean())
                detail[f"{name}_per_frame"] = values[index].tolist()
            detail["counts"] = {
                name: self._counts[index, :, column].tolist()
                for column, name in enumerate(count_names)
            }
            result["threshold_details"][str(threshold)] = detail

        result["losses"] = {}
        if self.compute_image_metrics:
            for name, total in self._image_sums.items():
                per_frame = total / self.total
                average = float(per_frame.mean())
                result[name] = average
                result["per_lead_time"][name] = per_frame.tolist()
                result["losses"][name] = {"avg": average, "per_frame": per_frame.tolist()}
                result["summary"][f"avg_{name}"] = average

        # Validate even when not writing a file: no NaN or Infinity can escape.
        encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if is_main_process:
            LOGGER.info(
                "Evaluation: %s samples; CSI %.6f; HSS %.6f",
                self.total,
                result["csi"],
                result["hss"],
            )
            if json_save_dir is not None:
                output_dir = Path(json_save_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "eval_results.json").write_text(encoded + "\n", encoding="utf-8")
        return result
