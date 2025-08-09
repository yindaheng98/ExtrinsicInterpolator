from torch.utils.data import Dataset
from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset
from .abc import SimpleCamera
from .interp import smooth_interpolation


class SimpleCameraInterpolator(Dataset):

    def __init__(self, dataset: CameraDataset, n: int, window_size: int = 3):
        self.cameras = smooth_interpolation(dataset=dataset, n=n, window_size=window_size)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> SimpleCamera:
        return self.cameras[idx]

    def to(self, device):
        self.cameras = [camera.to(device) for camera in self.cameras]
        return self


class SimpleCameraInterpolationDataset(CameraDataset):
    def __init__(self, dataset: CameraDataset, n: int, window_size: int = 3):
        self.cameras = SimpleCameraInterpolator(dataset=dataset, n=n, window_size=window_size)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Camera:
        return self.cameras[idx].to_camera()

    def to(self, device):
        self.cameras = self.cameras.to(device)
        return self
