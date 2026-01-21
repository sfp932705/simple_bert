from __future__ import annotations

from enum import Enum

import torch
from torch import nn

from settings import BertSettings


class Activation(Enum):
    RELU = "relu"
    GELU = "gelu"

    @staticmethod
    def from_string(activation: str) -> Activation:
        return Activation(activation.lower())

    def get_act_fn(self) -> nn.Module:
        match self:
            case Activation.GELU:
                return nn.GELU()
            case _:
                return nn.ReLU()


class FeedForwardLayer(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.dense_expansion = nn.Linear(
            settings.hidden_size, settings.intermediate_size
        )
        self.activation = Activation.from_string(settings.hidden_act).get_act_fn()
        self.dense_contraction = nn.Linear(
            settings.intermediate_size, settings.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(settings.hidden_size, eps=settings.layer_norm_eps)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        interm_output = self.activation(self.dense_expansion(hidden_states))
        output = self.dropout(self.dense_contraction(interm_output))
        return self.LayerNorm(output + hidden_states)
