import torch
from torch.utils.data import Dataset
from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset
from .abc import Extrinsic
from .interp import smooth_interpolation


class ExtrinsicInterpolator(Dataset):

    def __init__(self, dataset: CameraDataset, n: int, window_size: int = 3):
        self.cameras = smooth_interpolation(dataset=dataset, n=n, window_size=window_size)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Extrinsic:
        return self.cameras[idx]

    def to(self, device):
        self.cameras = [camera.to(device) for camera in self.cameras]
        return self


class ExtrinsicInterpolationDataset(CameraDataset):
    def __init__(
        self,
            dataset: CameraDataset, n: int, window_size: int = 3,
            image_height: int = 1000, image_width: int = 1000,
            FoVx: float = 90.0*torch.pi/180, FoVy: float = 90.0*torch.pi/180):
        self.cameras = ExtrinsicInterpolator(dataset=dataset, n=n, window_size=window_size)
        self.image_height = image_height
        self.image_width = image_width
        self.FoVx = FoVx
        self.FoVy = FoVy

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Camera:
        return self.cameras[idx].to_camera(
            image_height=self.image_height,
            image_width=self.image_width,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            device=self.cameras[idx].R.device
        )

    def to(self, device):
        self.cameras = self.cameras.to(device)
        return self
