import torch
from torch import nn

from modules.encoders.encoder import Encoder
from settings import BertSettings


class StackedEncoder(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.layer = nn.ModuleList(
            [Encoder(settings) for _ in range(settings.num_hidden_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
