import pickle
from typing import List, Tuple

import torch
from gaussian_splatting.utils import matrix_to_quaternion, quaternion_to_matrix

from ..abc import Extrinsic, ExtrinsicDataset
from .abc import AbstractTrainableExtrinsicPredictor


NORMALIZATION_EPSILON = 1e-8


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(q)
    fallback[..., 0] = 1
    norm = q.norm(dim=-1, keepdim=True)
    normalized = q / norm.clamp_min(NORMALIZATION_EPSILON)
    valid = torch.isfinite(normalized).all(dim=-1, keepdim=True)
    return torch.where(valid, normalized, fallback)


def align_quaternion_sequence(qs: torch.Tensor) -> torch.Tensor:
    qs = normalize_quaternion(qs).clone()
    for idx in range(1, qs.shape[0]):
        if torch.dot(qs[idx - 1], qs[idx]).item() < 0:
            qs[idx] = -qs[idx]
    return qs


def normalize_extrinsic_tensor(data: torch.Tensor, reference: torch.Tensor = None) -> torch.Tensor:
    data = data.clone()
    q = normalize_quaternion(data[..., :4])
    if reference is not None:
        reference_q = normalize_quaternion(reference[..., :4]).to(device=q.device, dtype=q.dtype)
        if torch.dot(reference_q.reshape(-1)[:4], q.reshape(-1)[:4]).item() < 0:
            q = -q
    data[..., :4] = q
    return data


def extrinsics_to_tensor(extrinsics: List[Extrinsic]) -> torch.Tensor:
    Rs = torch.stack([extrinsic.R for extrinsic in extrinsics])
    Ts = torch.stack([extrinsic.T for extrinsic in extrinsics])
    Qs = align_quaternion_sequence(matrix_to_quaternion(Rs))
    return torch.concat((Qs, Ts), dim=-1)


def tensor_to_extrinsics(data: torch.Tensor) -> List[Extrinsic]:
    data = normalize_extrinsic_tensor(data)
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
        inputs.append(torch.concat([
            history[-lag]
            for lag in range(1, lag_order + 1)
        ], dim=0))
        targets.append(sequence[idx])
    return torch.stack(inputs), torch.stack(targets)


def forecast_var_sequence(
    history: torch.Tensor,
    coefficients: torch.Tensor,
    intercept: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    steps: int,
    lag_order: int,
) -> torch.Tensor:
    assert steps >= 0, ValueError("Prediction count must be non-negative.")
    assert history.shape[0] >= lag_order, RuntimeError("VAR prediction requires enough history items.")
    if steps == 0:
        return torch.empty((0, history.shape[1]), device=history.device, dtype=history.dtype)

    normalized_history = normalize_sequence(history, mean, scale)
    values = [normalized_history[idx] for idx in range(normalized_history.shape[0])]
    predictions = []
    while len(predictions) < steps:
        window = torch.stack(values[-lag_order:])
        model_input = torch.concat([
            window[-lag]
            for lag in range(1, lag_order + 1)
        ], dim=0)
        prediction = intercept + model_input @ coefficients
        prediction = normalize_extrinsic_tensor(
            prediction * scale + mean,
            reference=values[-1] * scale + mean,
        )
        values.append(normalize_sequence(prediction, mean, scale))
        predictions.append(prediction)
    return torch.stack(predictions)


class VARModel:
    def __init__(
        self,
        coefficients: torch.Tensor,
        intercept: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        lag_order: int,
    ):
        self.coefficients = coefficients
        self.intercept = intercept
        self.mean = mean
        self.scale = scale
        self.k_ar = lag_order

    def forecast(self, data, steps: int):
        history = torch.as_tensor(data, dtype=self.coefficients.dtype)
        assert history.ndim == 2, ValueError("VAR history must be a 2D tensor or array.")
        assert history.shape[1] == self.intercept.shape[0], ValueError("VAR history feature count is invalid.")
        predictions = forecast_var_sequence(
            history=history,
            coefficients=self.coefficients,
            intercept=self.intercept,
            mean=self.mean,
            scale=self.scale,
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
    sequences = [
        sequence.detach().cpu().to(dtype=torch.float64)
        for sequence in sequences
    ]
    mean, scale = sequence_mean_and_scale(torch.concat(sequences, dim=0))
    for sequence in sequences:
        sequence = normalize_sequence(sequence, mean, scale)
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
        mean=mean,
        scale=scale,
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
