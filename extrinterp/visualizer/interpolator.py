from typing import Sequence

from ..abc import Extrinsic
from .extrinsic import plot_extrinsics


def plot_interpolator_output(
    extrinsics: Sequence[Extrinsic],
    ax,
):
    """Plot extrinsics returned by an interpolator."""
    return plot_extrinsics(extrinsics, ax=ax)
