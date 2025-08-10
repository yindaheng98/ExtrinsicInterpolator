from abc import abstractmethod
from typing import NamedTuple

import torch
from torch.utils.data import Dataset

from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.camera import build_camera


class Extrinsic(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor

    @classmethod
    def from_camera(cls, camera: Camera) -> 'Extrinsic':
        return cls(
            R=camera.R,
            T=camera.T,
        )

    def to_camera(
        self,
        image_height: int, image_width: int,
        FoVx: float, FoVy: float,
        *args, **kwargs
    ) -> Camera:
        return build_camera(
            image_width=image_width,
            image_height=image_height,
            FoVx=FoVx,
            FoVy=FoVy,
            R=self.R,
            T=self.T,
            *args, **kwargs
        )

    def to(self, device):
        return Camera(
            R=self.R.to(device),
            T=self.T.to(device),
        )


class ExtrinsicDataset(Dataset):
    @abstractmethod
    def to(self, device) -> 'ExtrinsicDataset':
        return self

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> Extrinsic:
        raise NotImplementedError


class Extrinsic2CameraDataset(CameraDataset):
    def __init__(
        self,
            dataset: ExtrinsicDataset,
            image_height: int = 1000, image_width: int = 1000,
            FoVx: float = 90.0*torch.pi/180, FoVy: float = 90.0*torch.pi/180):
        self.cameras = dataset
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
