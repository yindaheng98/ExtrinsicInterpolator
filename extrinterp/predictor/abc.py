from abc import ABC, abstractmethod
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
