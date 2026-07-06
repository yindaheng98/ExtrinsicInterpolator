from typing import List, Optional, Tuple

import torch
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_to_matrix

from ..abc import Extrinsic
from .abc import AbstractExtrinsicPredictor


class KalmanExtrinsicPredictor(AbstractExtrinsicPredictor):
    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        initial_covariance: float = 1.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.state: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.state = None
        self.covariance = None

    def update(self, extrinsic: Extrinsic) -> None:
        observation = self.extrinsic_to_observation(extrinsic)
        if self.state is None:
            self.state = torch.concat((observation, torch.zeros_like(observation)), dim=0)
            self.covariance = torch.eye(
                self.state.shape[0],
                device=self.state.device,
                dtype=self.state.dtype,
            ) * self.initial_covariance
            return

        assert self.covariance is not None, RuntimeError("Kalman covariance is not initialized.")
        self.state = self.state.to(device=observation.device, dtype=observation.dtype)
        self.covariance = self.covariance.to(device=observation.device, dtype=observation.dtype)
        predicted_state, predicted_covariance = self.predict_state(self.state, self.covariance)
        self.state, self.covariance = self.update_state(
            predicted_state,
            predicted_covariance,
            observation,
        )

    def predict(self, n: int) -> List[Extrinsic]:
        assert self.state is not None, RuntimeError("Kalman predictor has not received any extrinsic.")
        assert self.covariance is not None, RuntimeError("Kalman covariance is not initialized.")
        assert n >= 0, ValueError("Prediction count must be non-negative.")
        if n == 0:
            return []

        state = self.state
        covariance = self.covariance
        predictions = []
        while len(predictions) < n:
            state, covariance = self.predict_state(state, covariance)
            predictions.append(self.state_to_extrinsic(state))
        return predictions

    def extrinsic_to_observation(self, extrinsic: Extrinsic) -> torch.Tensor:
        q = matrix_to_quaternion(extrinsic.R)
        q = self.normalize_quaternion(q)
        if self.state is not None:
            previous_q = self.state[:4].to(device=q.device, dtype=q.dtype)
            if torch.dot(q, previous_q).item() < 0:
                q = -q
        return torch.concat((q, extrinsic.T), dim=-1)

    def state_to_extrinsic(self, state: torch.Tensor) -> Extrinsic:
        q = self.normalize_quaternion(state[:4])
        T = state[4:7]
        R = quaternion_to_matrix(q)
        return Extrinsic(R=R, T=T)

    def predict_state(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transition = self.constant_velocity_transition_matrix(state.device, state.dtype)
        process_noise = torch.eye(
            state.shape[0],
            device=state.device,
            dtype=state.dtype,
        ) * self.process_noise
        predicted_state = transition @ state
        predicted_state = predicted_state.clone()
        predicted_state[:4] = self.normalize_quaternion(predicted_state[:4])
        predicted_covariance = transition @ covariance @ transition.T + process_noise
        return predicted_state, predicted_covariance

    def update_state(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observation_matrix = self.extrinsic_observation_matrix(state.device, state.dtype)
        measurement_noise = torch.eye(
            observation.shape[0],
            device=state.device,
            dtype=state.dtype,
        ) * self.measurement_noise

        residual = observation - observation_matrix @ state
        innovation_covariance = observation_matrix @ covariance @ observation_matrix.T + measurement_noise
        kalman_gain = torch.linalg.solve(
            innovation_covariance.T,
            (covariance @ observation_matrix.T).T,
        ).T
        updated_state = state + kalman_gain @ residual
        updated_state = updated_state.clone()
        updated_state[:4] = self.normalize_quaternion(updated_state[:4])

        identity = torch.eye(state.shape[0], device=state.device, dtype=state.dtype)
        updated_covariance = (identity - kalman_gain @ observation_matrix) @ covariance
        return updated_state, updated_covariance

    @staticmethod
    def constant_velocity_transition_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        transition = torch.eye(14, device=device, dtype=dtype)
        transition[:7, 7:] = torch.eye(7, device=device, dtype=dtype)
        return transition

    @staticmethod
    def extrinsic_observation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        observation = torch.zeros((7, 14), device=device, dtype=dtype)
        observation[:, :7] = torch.eye(7, device=device, dtype=dtype)
        return observation

    @staticmethod
    def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
        return q / q.norm().clamp_min(torch.finfo(q.dtype).eps)
