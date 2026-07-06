from typing import Sequence

from ..abc import Extrinsic
from .extrinsic import plot_extrinsics


def plot_predictor_output(
    predictions: Sequence[Extrinsic],
):
    """Plot extrinsics returned by a predictor."""
    return plot_extrinsics(predictions)
