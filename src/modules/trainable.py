from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import torch
from torch import nn

from data.base import T_DataItem


class TrainableModel(ABC, nn.Module, Generic[T_DataItem]):
    @abstractmethod
    def compute_loss_from_dataset_batch(self, batch: T_DataItem) -> torch.Tensor: ...

    def load(self, checkpoint_path: str, device: str | torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
