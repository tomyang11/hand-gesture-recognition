import torch
from torch import Tensor
from yolov5.models.experimental import attempt_load


class YOLOv5HandDetector:
    def __init__(self, weights: str):
        self.model = attempt_load(weights)

    def __call__(self, img: Tensor) -> Tensor:
        return self.model(img)

    def to(self, device: str):
        self.model.to(device)
