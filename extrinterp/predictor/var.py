import pickle
from typing import List, Tuple

import torch
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_to_matrix

from ..abc import Extrinsic, ExtrinsicDataset
from .abc import AbstractTrainableExtrinsicPredictor


NORMALIZATION_EPSILON = 1e-8


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


def sequence_mean_and_scale(sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = sequence.std(dim=0, unbiased=False)
    return sequence.mean(dim=0), scale.clamp_min(NORMALIZATION_EPSILON)


def normalize_sequence(sequence: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (sequence - mean) / scale


def build_var_training_matrices(
    sequence: torch.Tensor,
    lag_order: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for idx in range(lag_order, sequence.shape[0]):
        history = sequence[idx - lag_order:idx]
        mean, scale = sequence_mean_and_scale(history)
        history = normalize_sequence(history, mean, scale)
        target = normalize_sequence(sequence[idx], mean, scale)
        inputs.append(torch.concat([
            history[-lag]
            for lag in range(1, lag_order + 1)
        ], dim=0))
        targets.append(target)
    return torch.stack(inputs), torch.stack(targets)


def forecast_var_sequence(
    history: torch.Tensor,
    coefficients: torch.Tensor,
    intercept: torch.Tensor,
    steps: int,
    lag_order: int,
) -> torch.Tensor:
    assert steps >= 0, ValueError("Prediction count must be non-negative.")
    assert history.shape[0] >= lag_order, RuntimeError("VAR prediction requires enough history items.")
    if steps == 0:
        return torch.empty((0, history.shape[1]), device=history.device, dtype=history.dtype)

    values = [history[idx] for idx in range(history.shape[0])]
    predictions = []
    while len(predictions) < steps:
        window = torch.stack(values[-lag_order:])
        mean, scale = sequence_mean_and_scale(window)
        normalized_window = normalize_sequence(window, mean, scale)
        model_input = torch.concat([
            normalized_window[-lag]
            for lag in range(1, lag_order + 1)
        ], dim=0)
        normalized_prediction = intercept + model_input @ coefficients
        prediction = normalized_prediction * scale + mean
        values.append(prediction)
        predictions.append(prediction)
    return torch.stack(predictions)


class VARModel:
    def __init__(
        self,
        coefficients: torch.Tensor,
        intercept: torch.Tensor,
        lag_order: int,
    ):
        self.coefficients = coefficients
        self.intercept = intercept
        self.k_ar = lag_order

    def forecast(self, data, steps: int):
        history = torch.as_tensor(data, dtype=self.coefficients.dtype)
        assert history.ndim == 2, ValueError("VAR history must be a 2D tensor or array.")
        assert history.shape[1] == self.intercept.shape[0], ValueError("VAR history feature count is invalid.")
        predictions = forecast_var_sequence(
            history=history,
            coefficients=self.coefficients,
            intercept=self.intercept,
            steps=steps,
            lag_order=self.k_ar,
        )
        return predictions.detach().cpu().numpy()


def fit_var_model(
    sequences: List[torch.Tensor],
    lag_order: int,
) -> VARModel:
    input_matrices = []
    target_matrices = []
    for sequence in sequences:
        sequence = sequence.detach().cpu().to(dtype=torch.float64)
        inputs, targets = build_var_training_matrices(
            sequence=sequence,
            lag_order=lag_order,
        )
        input_matrices.append(inputs)
        target_matrices.append(targets)

    inputs = torch.concat(input_matrices, dim=0)
    targets = torch.concat(target_matrices, dim=0)
    design = torch.concat((
        torch.ones((inputs.shape[0], 1), device=inputs.device, dtype=inputs.dtype),
        inputs,
    ), dim=1)
    parameters = torch.linalg.lstsq(design, targets).solution
    intercept = parameters[0]
    coefficients = parameters[1:]
    return VARModel(
        coefficients=coefficients,
        intercept=intercept,
        lag_order=lag_order,
    )


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

    def train(self, dataset: List[ExtrinsicDataset]) -> None:
        sequences = [
            extrinsics_to_tensor([
                extrinsic_dataset[i]
                for i in range(len(extrinsic_dataset))
            ]).detach().cpu()
            for extrinsic_dataset in dataset
        ]

        self.model = fit_var_model(
            sequences=sequences,
            lag_order=self.lag_order,
        )

    def save(self, path: str) -> None:
        assert self.model is not None, RuntimeError("VAR model has not been trained or loaded.")
        with open(path, "wb") as file:
            pickle.dump(self.model, file)

    def load(self, path: str) -> None:
        with open(path, "rb") as file:
            self.model = pickle.load(file)
        self.lag_order = self.model.k_ar
