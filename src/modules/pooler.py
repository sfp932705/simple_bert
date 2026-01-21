import torch
from torch import nn

from settings import LayerCommonSettings


class Pooler(nn.Module):
    def __init__(self, settings: LayerCommonSettings):
        super().__init__()
        self.dense = nn.Linear(settings.hidden_size, settings.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        return self.activation(self.dense(first_token_tensor))
