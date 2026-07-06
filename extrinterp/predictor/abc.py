from abc import ABC, abstractmethod
from os import PathLike
from typing import List

from ..abc import Extrinsic


class AbstractExtrinsicPredictor(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, extrinsic: Extrinsic) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, n: int) -> List[Extrinsic]:
        raise NotImplementedError


class AbstractTrainableExtrinsicPredictor(AbstractExtrinsicPredictor):
    @abstractmethod
    def train(self, dataset: List[List[Extrinsic]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | PathLike) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | PathLike) -> None:
        raise NotImplementedError
