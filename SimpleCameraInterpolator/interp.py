from typing import List
import math
import torch
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_to_matrix
from .abc import SimpleCamera


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*torch.atan(pixels/(2*focal))


def linspace_cameras(start: SimpleCamera, end: SimpleCamera, n: int) -> List[SimpleCamera]:
    ratio = torch.linspace(0, 1, n, device=start.R.device).unsqueeze(-1)
    Ts = start.T.unsqueeze(0) + ratio * (end.T.unsqueeze(0) - start.T.unsqueeze(0))
    q_start, q_end = matrix_to_quaternion(start.R), matrix_to_quaternion(end.R)
    qs = q_start.unsqueeze(0) + ratio * (q_end.unsqueeze(0) - q_start.unsqueeze(0))
    Rs = quaternion_to_matrix(qs)
    image_height = (start.image_height + ratio * (end.image_height - start.image_height)).int()
    image_width = (start.image_width + ratio * (end.image_width - start.image_width)).int()
    focal_x = fov2focal(start.FoVx, image_width) + ratio * (fov2focal(end.FoVx, image_width) - fov2focal(start.FoVx, image_width))
    focal_y = fov2focal(start.FoVy, image_height) + ratio * (fov2focal(end.FoVy, image_height) - fov2focal(start.FoVy, image_height))
    FoVx = focal2fov(focal_x, image_width)
    FoVy = focal2fov(focal_y, image_height)
    return [
        SimpleCamera(R=R, T=T, FoVx=FoVx.item(), FoVy=FoVy.item(), image_height=ih.item(), image_width=iw.item())
        for R, T, FoVx, FoVy, ih, iw in zip(Rs, Ts, FoVx, FoVy, image_height, image_width)
    ]


def sort_cameras(cameras: List[SimpleCamera]) -> List[SimpleCamera]:
    Ts = torch.stack([camera.T for camera in cameras])
    distances = torch.cdist(Ts, Ts)
    next_idx = distances.mean(0).argmax()
    sorted_cameras = [cameras.pop(next_idx)]
    while cameras:
        distances = torch.cdist(sorted_cameras[-1].T.unsqueeze(0), torch.stack([camera.T for camera in cameras])).squeeze(0)
        next_idx = distances.argmin()
        sorted_cameras.append(cameras.pop(next_idx))
    return sorted_cameras


def interpolation(cameras: List[SimpleCamera], n: int) -> List[SimpleCamera]:
    cameras = sort_cameras(cameras)
    new_cameras = []
    for i in range(len(cameras) - 1):
        k = round((n - 1) / (len(cameras) - 1 - i))
        new_cameras.extend(linspace_cameras(cameras[i], cameras[i + 1], k + 1)[:-1])
        n -= k
    new_cameras.append(cameras[-1])
    return new_cameras


def smooth_1d(inputs: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    shape = inputs.shape
    inputs = inputs.view(shape[0], -1).T.unsqueeze(1)
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    kernel = torch.ones((1, 1, window_size), device=inputs.device) / window_size
    outputs = torch.nn.functional.conv1d(inputs, kernel, stride=1, padding=0)
    return outputs.squeeze(1).T.view(shape[0] - window_size//2*2, *shape[1:]).contiguous()


def smooth(cameras: List[SimpleCamera], window_size: int = 3) -> List[SimpleCamera]:
    Ts = smooth_1d(torch.stack([camera.T for camera in cameras]), window_size=window_size)
    Rs = quaternion_to_matrix(smooth_1d(matrix_to_quaternion(torch.stack([camera.R for camera in cameras])), window_size=window_size))
    Ks = smooth_1d(torch.stack([camera.K for camera in cameras]), window_size=window_size)
    return [
        cameras[i]._replace(R=R, T=T, K=K)
        for i, (R, T, K) in enumerate(zip(Rs, Ts, Ks))
    ]


def smooth_interpolation(dataset: CameraDataset, n: int, window_size: int = 3) -> List[SimpleCamera]:
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    cameras = [SimpleCamera.from_camera(camera, timestamp=idx) for idx, camera in enumerate(dataset)]
    return smooth(interpolation(cameras, n + window_size // 2*2), window_size=window_size)
