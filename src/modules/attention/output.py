import torch
from torch import nn

from settings import BertSettings


class AttentionOutput(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.dense = nn.Linear(settings.hidden_size, settings.hidden_size)
        self.LayerNorm = nn.LayerNorm(settings.hidden_size, eps=settings.layer_norm_eps)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
