from typing import NamedTuple

import torch
from gaussian_splatting import Camera
from gaussian_splatting.camera import build_camera


class Extrinsics(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor

    @classmethod
    def from_camera(cls, camera: Camera) -> 'Extrinsics':
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
