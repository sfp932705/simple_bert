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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
