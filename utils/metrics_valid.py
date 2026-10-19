"""Compatibility import for validation using categorical radar metrics only."""

from .metrics import DEFAULT_THRESHOLDS
from .metrics import Evaluator as _Evaluator


class Evaluator(_Evaluator):
    """Use the same counts and aggregation as testing, with image metrics off."""

    def __init__(
        self, seq_len, value_scale, thresholds=DEFAULT_THRESHOLDS, *, compute_image_metrics=False
    ):
        super().__init__(
            seq_len,
            value_scale,
            thresholds,
            compute_image_metrics=compute_image_metrics,
        )
