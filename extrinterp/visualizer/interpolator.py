from typing import List, Sequence

import open3d as o3d
import torch
from gaussian_splatting.prepare import prepare_dataset

from ..abc import Extrinsic
from ..interpolator.interp import smooth_interpolation
from .extrinsic import plot_extrinsics


def prepare_visualization(
        source: str, device: str, n: int, window_size: int,
        trainable_camera: bool = False, load_camera: str = None
) -> tuple[List[Extrinsic], List[Extrinsic]]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_depth=False)
    inputs = [Extrinsic.from_camera(dataset[idx]) for idx in range(len(dataset))]
    outputs = smooth_interpolation(dataset=dataset, n=n, window_size=window_size)
    return inputs, outputs


def visualize(inputs: List[Extrinsic], outputs: List[Extrinsic]) -> None:
    geometries = plot_extrinsics(outputs, 0.25, line_color=(0.0, 0.35, 1.0)) + plot_extrinsics(inputs, 0.5, line_color=(0.0, 0.0, 0.0))
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--interp_n", required=True, type=int)
    parser.add_argument("--interp_window_size", type=int, default=3)
    args = parser.parse_args()
    with torch.no_grad():
        inputs, outputs = prepare_visualization(
            source=args.source, device=args.device,
            n=args.interp_n, window_size=args.interp_window_size,
            trainable_camera=args.mode == "camera",
            load_camera=args.load_camera,
        )
        visualize(inputs, outputs)
