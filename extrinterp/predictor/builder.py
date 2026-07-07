from typing import Dict, Optional, Type

from .abc import AbstractExtrinsicPredictor, AbstractTrainableExtrinsicPredictor
from .angular import ConstantAngularAccelerationExtrinsicPredictor
from .kalman import KalmanExtrinsicPredictor
from .var import VARExtrinsicPredictor


PREDICTOR_CLASSES: Dict[str, Type[AbstractExtrinsicPredictor]] = {
    "angular": ConstantAngularAccelerationExtrinsicPredictor,
    "kalman": KalmanExtrinsicPredictor,
    "var": VARExtrinsicPredictor,
}


def build_predictor(
    name: str,
    path: Optional[str] = None,
    **configs,
) -> AbstractExtrinsicPredictor:
    try:
        predictor_cls = PREDICTOR_CLASSES[name]
    except KeyError as exc:
        raise ValueError("Unknown predictor name: {}".format(name)) from exc

    predictor = predictor_cls(**configs)
    if path is not None:
        if not isinstance(predictor, AbstractTrainableExtrinsicPredictor):
            raise TypeError("Predictor '{}' does not support loading from a path.".format(name))
        predictor.load(path)
    return predictor
