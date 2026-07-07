from typing import List, Sequence

import torch
from gaussian_splatting.prepare import prepare_dataset

from ..predictor.abc import AbstractExtrinsicPredictor
from ..predictor.builder import build_predictor
from ..abc import Extrinsic
from ..interpolator.interp import sort_cameras
from .extrinsic import draw_geometries, plot_extrinsics


def prepare_prediction_visualization(
        predictor: AbstractExtrinsicPredictor,
        groundtruth: Sequence[Extrinsic],
        update_interval: int,
        predict_n: int
) -> tuple[List[Extrinsic], List[List[Extrinsic]]]:
    groundtruth = list(groundtruth)
    predictor.reset()
    branches = []
    for idx, extrinsic in enumerate(groundtruth):
        predictor.update(extrinsic)
        if (idx + 1) % update_interval == 0:
            branches.append([extrinsic] + predictor.predict(predict_n))
    return list(groundtruth), branches


def visualize_prediction(groundtruth: List[Extrinsic], branches: List[List[Extrinsic]]) -> None:
    geometries = plot_extrinsics(
        groundtruth,
        0.5,
        line_color=(0.0, 0.0, 0.0),
        name="prediction groundtruth",
    )
    for idx, branch in enumerate(branches):
        geometries += plot_extrinsics(
            branch,
            0.25,
            line_color=(1.0, 0.25, 0.0),
            name="prediction branch {:03d}".format(idx),
        )
    draw_geometries(geometries, title="Extrinsic Prediction Visualizer")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--predictor", choices=["angular", "kalman", "var"], default="angular")
    parser.add_argument("--predictor_path", default=None, type=str)
    parser.add_argument("-o", "--option", action="append", default=[])
    parser.add_argument("--update_interval", required=True, type=int)
    parser.add_argument("--predict_n", required=True, type=int)
    args = parser.parse_args()
    with torch.no_grad():
        dataset = prepare_dataset(
            source=args.source,
            device=args.device,
            trainable_camera=args.mode == "camera",
            load_camera=args.load_camera,
            load_depth=False,
        )
        groundtruth = sort_cameras([
            Extrinsic.from_camera(dataset[idx])
            for idx in range(len(dataset))
        ])
        configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
        groundtruth, branches = prepare_prediction_visualization(
            predictor=build_predictor(
                name=args.predictor,
                path=args.predictor_path,
                **configs,
            ),
            groundtruth=groundtruth,
            update_interval=args.update_interval,
            predict_n=args.predict_n,
        )
        visualize_prediction(groundtruth, branches)
