from typing import List

import torch
from gaussian_splatting.prepare import prepare_dataset

from .abc import Extrinsic
from .interpolator.interp import smooth_interpolation
from .visualizer.interpolator import plot_interpolator_output


def prepare_visualization(
        source: str, device: str, n: int, window_size: int,
        trainable_camera: bool = False, load_camera: str = None
) -> List[Extrinsic]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_depth=False)
    return smooth_interpolation(dataset=dataset, n=n, window_size=window_size)


def visualize(extrinsics: List[Extrinsic]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_interpolator_output(extrinsics, ax)
    plt.show()


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
        extrinsics = prepare_visualization(
            source=args.source, device=args.device,
            n=args.interp_n, window_size=args.interp_window_size,
            trainable_camera=args.mode == "camera",
            load_camera=args.load_camera,
        )
        visualize(extrinsics)
