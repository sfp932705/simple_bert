from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn

from data.base import T_DataItem


@dataclass
class TrainForwardPassOutput:
    loss: torch.Tensor


T_FPOutput = TypeVar("T_FPOutput", bound=TrainForwardPassOutput)


class TrainableModel(ABC, nn.Module, Generic[T_DataItem, T_FPOutput]):
    @abstractmethod
    def train_forward_from_dataset_batch(self, batch: T_DataItem) -> T_FPOutput: ...

    def load(self, checkpoint_path: str, device: str | torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
