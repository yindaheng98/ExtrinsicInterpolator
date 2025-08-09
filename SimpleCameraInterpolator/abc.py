from typing import NamedTuple

import torch
from gaussian_splatting import Camera
from gaussian_splatting.camera import build_camera


class SimpleCamera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor

    @classmethod
    def from_camera(cls, camera: Camera) -> 'SimpleCamera':
        return cls(
            image_height=camera.image_height,
            image_width=camera.image_width,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            R=camera.R,
            T=camera.T,
        )

    def to_camera(self, *args, **kwargs) -> Camera:
        return build_camera(
            image_width=self.image_width,
            image_height=self.image_height,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            R=self.R,
            T=self.T,
            *args, **kwargs
        )

    def to(self, device):
        return Camera(
            R=self.R.to(device),
            T=self.T.to(device),
        )
