from typing import List, Tuple

import torch
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_to_matrix

from ..abc import Extrinsic
from .abc import AbstractExtrinsicPredictor


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True)


def align_quaternion_to_reference(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if torch.dot(q, reference).item() < 0:
        return -q
    return q


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.concat((q[:1], -q[1:]), dim=0)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_w, left_x, left_y, left_z = left.unbind(dim=-1)
    right_w, right_x, right_y, right_z = right.unbind(dim=-1)
    return torch.stack((
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    ), dim=-1)


def quaternion_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = rotation_vector.norm()
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 0,
        torch.sin(half_angle) / angle,
        torch.ones((), device=rotation_vector.device, dtype=rotation_vector.dtype) * 0.5,
    )
    return normalize_quaternion(torch.concat((
        torch.cos(half_angle).reshape(1),
        rotation_vector * scale,
    ), dim=0))


def quaternion_log(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    vector = q[1:]
    vector_norm = vector.norm()
    angle = 2 * torch.atan2(vector_norm, q[0])
    scale = torch.where(
        vector_norm > 0,
        angle / vector_norm,
        torch.ones((), device=q.device, dtype=q.dtype) * 2,
    )
    return vector * scale


def extrinsics_to_aligned_quaternions(extrinsics: List[Extrinsic]) -> torch.Tensor:
    qs = normalize_quaternion(matrix_to_quaternion(torch.stack([extrinsic.R for extrinsic in extrinsics]))).clone()
    for idx in range(1, qs.shape[0]):
        qs[idx] = align_quaternion_to_reference(qs[idx], qs[idx - 1])
    return qs


def body_angular_velocity_samples(extrinsics: List[Extrinsic], sample_interval: float) -> torch.Tensor:
    qs = extrinsics_to_aligned_quaternions(extrinsics)
    velocities = []
    for idx in range(1, qs.shape[0]):
        delta = quaternion_multiply(quaternion_conjugate(qs[idx - 1]), qs[idx])
        velocities.append(quaternion_log(delta) / sample_interval)
    return torch.stack(velocities)


def fit_current_motion(samples: torch.Tensor, sample_interval: float, latest_sample_time: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = samples.shape[0]
    degree = min(2, sample_count - 1)
    times = (
        torch.arange(sample_count, device=samples.device, dtype=samples.dtype)
        - sample_count
        + 1
    ) * sample_interval + latest_sample_time
    design = torch.stack([
        times ** power
        for power in range(degree + 1)
    ], dim=1)
    coefficients = torch.linalg.lstsq(design, samples).solution
    value = coefficients[0]
    velocity = coefficients[1] if degree >= 1 else torch.zeros_like(value)
    acceleration = 2 * coefficients[2] if degree >= 2 else torch.zeros_like(value)
    return value, velocity, acceleration


def estimate_angular_velocity_and_acceleration(
    extrinsics: List[Extrinsic],
    sample_interval: float,
    smoothing_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = matrix_to_quaternion(extrinsics[-1].R)
    if len(extrinsics) == 1:
        return torch.zeros(3, device=q.device, dtype=q.dtype), torch.zeros(3, device=q.device, dtype=q.dtype)
    samples = body_angular_velocity_samples(extrinsics[-smoothing_window:], sample_interval)
    angular_motion = fit_current_motion(
        samples=samples,
        sample_interval=sample_interval,
        latest_sample_time=-0.5 * sample_interval,
    )
    return angular_motion[0], angular_motion[1]


def estimate_translation_velocity_and_acceleration(
    extrinsics: List[Extrinsic],
    sample_interval: float,
    smoothing_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    positions = torch.stack([extrinsic.T for extrinsic in extrinsics[-smoothing_window:]])
    position, velocity, acceleration = fit_current_motion(
        samples=positions,
        sample_interval=sample_interval,
    )
    return velocity, acceleration


def integrate_angular_acceleration(
    q: torch.Tensor,
    angular_velocity: torch.Tensor,
    angular_acceleration: torch.Tensor,
    horizon: float,
    integration_step: float,
) -> torch.Tensor:
    elapsed = 0.0
    while elapsed < horizon:
        step = min(integration_step, horizon - elapsed)
        q = quaternion_multiply(q, quaternion_exp(angular_velocity * step))
        angular_velocity = angular_velocity + angular_acceleration * step
        elapsed = elapsed + step
    return normalize_quaternion(q)


def speed_adjusted_horizon(
    horizon: float,
    angular_speed: torch.Tensor,
    low_speed_threshold: float,
    low_speed_horizon: float,
) -> float:
    horizon_tensor = torch.as_tensor(horizon, device=angular_speed.device, dtype=angular_speed.dtype)
    speed_ratio = (angular_speed / low_speed_threshold).clamp(max=1)
    limited_horizon = low_speed_horizon + (horizon_tensor - low_speed_horizon).clamp_min(0) * speed_ratio
    return torch.minimum(horizon_tensor, limited_horizon).item()


class ConstantAngularAccelerationExtrinsicPredictor(AbstractExtrinsicPredictor):
    def __init__(
        self,
        sample_interval: float = 1.0 / 90.0,
        integration_step: float = 0.001,
        smoothing_window: int = 5,
        low_speed_threshold: float = 0.05,
        low_speed_horizon: float = 0.02,
    ):
        self.sample_interval = sample_interval
        self.integration_step = integration_step
        self.smoothing_window = smoothing_window
        self.low_speed_threshold = low_speed_threshold
        self.low_speed_horizon = low_speed_horizon
        self.history: List[Extrinsic] = []

    def to(self, device) -> 'ConstantAngularAccelerationExtrinsicPredictor':
        self.history = [
            extrinsic.to(device)
            for extrinsic in self.history
        ]
        return self

    def reset(self) -> None:
        self.history = []

    def update(self, extrinsic: Extrinsic) -> None:
        self.history.append(extrinsic)

    def predict(self, n: int) -> List[Extrinsic]:
        assert len(self.history) > 0, RuntimeError("Predictor has not received any extrinsic.")
        assert n >= 0, ValueError("Prediction count must be non-negative.")
        if n == 0:
            return []

        current = self.history[-1]
        current_q = normalize_quaternion(matrix_to_quaternion(current.R))
        angular_velocity, angular_acceleration = estimate_angular_velocity_and_acceleration(
            extrinsics=self.history,
            sample_interval=self.sample_interval,
            smoothing_window=self.smoothing_window,
        )
        linear_velocity, linear_acceleration = estimate_translation_velocity_and_acceleration(
            extrinsics=self.history,
            sample_interval=self.sample_interval,
            smoothing_window=self.smoothing_window,
        )

        predictions = []
        angular_speed = angular_velocity.norm()
        for idx in range(n):
            horizon = speed_adjusted_horizon(
                horizon=(idx + 1) * self.sample_interval,
                angular_speed=angular_speed,
                low_speed_threshold=self.low_speed_threshold,
                low_speed_horizon=self.low_speed_horizon,
            )
            q = integrate_angular_acceleration(
                q=current_q,
                angular_velocity=angular_velocity,
                angular_acceleration=angular_acceleration,
                horizon=horizon,
                integration_step=self.integration_step,
            )
            T = current.T + linear_velocity * horizon + 0.5 * linear_acceleration * horizon ** 2
            predictions.append(Extrinsic(R=quaternion_to_matrix(q), T=T))
        return predictions
