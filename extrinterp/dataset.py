from gaussian_splatting.dataset import CameraDataset
from .abc import Extrinsic, ExtrinsicDataset, Extrinsic2CameraDataset
from .interp import smooth_interpolation


class ExtrinsicInterpolator(ExtrinsicDataset):

    def __init__(self, dataset: CameraDataset, n: int, window_size: int = 3):
        self.cameras = smooth_interpolation(dataset=dataset, n=n, window_size=window_size)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Extrinsic:
        return self.cameras[idx]

    def to(self, device):
        self.cameras = [camera.to(device) for camera in self.cameras]
        return self


def ExtrinsicInterpolationDataset(dataset: CameraDataset, n: int, *args, window_size: int = 3, **kwargs) -> Extrinsic2CameraDataset:
    return Extrinsic2CameraDataset(
        dataset=ExtrinsicInterpolator(dataset=dataset, n=n, window_size=window_size),
        *args, **kwargs
    )
