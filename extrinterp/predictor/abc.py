from abc import ABC, abstractmethod
from typing import List

from ..abc import Extrinsic, ExtrinsicDataset


class AbstractExtrinsicPredictor(ABC):
    @abstractmethod
    def to(self, device) -> 'AbstractExtrinsicPredictor':
        raise NotImplementedError

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
    def train(self, dataset: List[ExtrinsicDataset]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError
