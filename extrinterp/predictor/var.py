from typing import List

import torch
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_to_matrix
from statsmodels.iolib.smpickle import load_pickle
from statsmodels.tsa.api import VAR

from ..abc import Extrinsic
from .abc import AbstractTrainableExtrinsicPredictor


def extrinsics_to_tensor(extrinsics: List[Extrinsic]) -> torch.Tensor:
    Rs = torch.stack([extrinsic.R for extrinsic in extrinsics])
    Ts = torch.stack([extrinsic.T for extrinsic in extrinsics])
    Qs = matrix_to_quaternion(Rs)
    return torch.concat((Qs, Ts), dim=-1)


def tensor_to_extrinsics(data: torch.Tensor) -> List[Extrinsic]:
    Qs = data[..., :4]
    Ts = data[..., 4:]
    Rs = quaternion_to_matrix(Qs)
    return [Extrinsic(R=R, T=T) for R, T in zip(Rs, Ts)]


class VARExtrinsicPredictor(AbstractTrainableExtrinsicPredictor):
    def __init__(
        self,
        lag_order: int = 2,
    ):
        self.model = None
        self.lag_order = lag_order
        self.history: List[Extrinsic] = []

    def reset(self) -> None:
        self.history = []

    def update(self, extrinsic: Extrinsic) -> None:
        self.history.append(extrinsic)

    def predict(self, n: int) -> List[Extrinsic]:
        assert self.model is not None, RuntimeError("VAR model has not been trained or loaded.")
        assert len(self.history) >= self.model.k_ar, RuntimeError("VAR prediction requires at least k_ar history items.")
        assert n >= 0, ValueError("Prediction count must be non-negative.")
        if n == 0:
            return []

        last_extrinsic = self.history[-1]
        history = extrinsics_to_tensor(self.history[-self.model.k_ar:]).detach().cpu().numpy()
        predictions = self.model.forecast(history, steps=n)
        prediction_tensor = torch.tensor(
            predictions,
            device=last_extrinsic.R.device,
            dtype=last_extrinsic.R.dtype,
        )
        return tensor_to_extrinsics(prediction_tensor)

    def train(self, dataset: List[List[Extrinsic]]) -> None:
        sequences = [sequence for sequence in dataset if sequence]
        assert sequences, ValueError("Training dataset must contain at least one non-empty sequence.")

        data = torch.concat([
            extrinsics_to_tensor(sequence).detach().cpu()
            for sequence in sequences
        ], dim=0)
        assert data.shape[0] > self.lag_order, ValueError("Training data length must be greater than lag_order.")

        self.model = VAR(endog=data.numpy()).fit(self.lag_order)

    def save(self, path: str) -> None:
        assert self.model is not None, RuntimeError("VAR model has not been trained or loaded.")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = load_pickle(path)
