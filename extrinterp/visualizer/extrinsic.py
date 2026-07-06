from typing import Sequence, Tuple

import torch

from ..abc import Extrinsic


RGB_AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")


def plot_extrinsic(
    extrinsic: Extrinsic,
    ax,
    axis_length: float = 1.0,
    axis_colors: Tuple[str, str, str] = RGB_AXIS_COLORS,
):
    """Plot one extrinsic on an existing matplotlib 3D axes."""
    position = extrinsic.T.detach().cpu().reshape(-1)[:3]
    rotation = extrinsic.R.detach().cpu()

    ax.scatter(
        [position[0].item()],
        [position[1].item()],
        [position[2].item()],
        color="black",
    )
    for axis_index, color in enumerate(axis_colors):
        direction = rotation[:, axis_index].reshape(-1)[:3].mul(axis_length)
        ax.quiver(
            position[0].item(),
            position[1].item(),
            position[2].item(),
            direction[0].item(),
            direction[1].item(),
            direction[2].item(),
            color=color,
        )

    return ax


def plot_extrinsics(
    extrinsics: Sequence[Extrinsic],
    ax,
    axis_colors: Tuple[str, str, str] = RGB_AXIS_COLORS,
):
    """Plot multiple extrinsics and their translation trajectory."""
    extrinsics = list(extrinsics)
    positions = torch.stack([
        extrinsic.T.detach().cpu().reshape(-1)[:3]
        for extrinsic in extrinsics
    ])
    ranges = positions.max(dim=0).values - positions.min(dim=0).values
    axis_length = max(ranges.max().item(), 1.0) * 0.08

    ax.plot(
        positions[:, 0].tolist(),
        positions[:, 1].tolist(),
        positions[:, 2].tolist(),
        color="tab:gray",
        linewidth=1.5,
        marker="o",
        markersize=3,
    )
    for extrinsic in extrinsics:
        plot_extrinsic(
            extrinsic,
            ax,
            axis_length=axis_length,
            axis_colors=axis_colors,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mins = positions.min(dim=0).values
    maxs = positions.max(dim=0).values
    centers = (mins + maxs).div(2)
    radius = max((maxs - mins).max().item() / 2, 1.0)
    ax.set_xlim(centers[0].item() - radius, centers[0].item() + radius)
    ax.set_ylim(centers[1].item() - radius, centers[1].item() + radius)
    ax.set_zlim(centers[2].item() - radius, centers[2].item() + radius)
    ax.set_box_aspect((1, 1, 1))

    return ax
