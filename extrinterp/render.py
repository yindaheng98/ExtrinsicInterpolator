from typing import Tuple
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import tifffile
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_dataset, prepare_gaussians
from extrinterp import ExtrinsicsInterpolationDataset


def prepare_rendering(
        sh_degree: int, source: str, device: str, n: int, window_size: int,
        trainable_camera: bool = False, load_ply: str = None, load_camera: str = None, load_depth=False,
        use_intrinsics: int | dict = 0
) -> Tuple[CameraDataset, GaussianModel]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_depth=load_depth)
    if isinstance(use_intrinsics, int):
        i = use_intrinsics
        use_intrinsics = dict(
            image_height=dataset[i].image_height, image_width=dataset[i].image_width,
            FoVx=dataset[i].FoVx, FoVy=dataset[i].FoVy)
    elif not isinstance(use_intrinsics, dict):
        raise ValueError("Invalid use_intrinsics format")
    dataset = ExtrinsicsInterpolationDataset(dataset=dataset, n=n, window_size=window_size, **use_intrinsics)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply)
    return dataset, gaussians


def rendering(dataset: CameraDataset, gaussians: GaussianModel, save: str) -> None:
    os.makedirs(save, exist_ok=True)
    dataset.save_cameras(os.path.join(save, "cameras.json"))
    render_path = os.path.join(save, "renders")
    makedirs(render_path, exist_ok=True)
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        out = gaussians(camera)
        rendering = out["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        depth = out["depth"].squeeze(0)
        tifffile.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.tiff"), depth.cpu().numpy())


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--interp_n", required=True, type=int)
    parser.add_argument("--interp_window_size", type=int, default=3)
    parser.add_argument("--use_intrinsics", type=str, default="0", help="Use intrinsics for rendering, can be an integer index or a dict with keys: image_height, image_width, FoVx, FoVy")
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device,
            n=args.interp_n, window_size=args.interp_window_size,
            trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera, load_depth=True,
            use_intrinsics=eval(args.use_intrinsics)
        )
        rendering(dataset, gaussians, save)
