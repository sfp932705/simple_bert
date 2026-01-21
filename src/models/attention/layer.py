import torch
from torch import nn

from models.attention.multihead import MultiHeadAttention
from models.attention.output import AttentionOutput
from settings import AttentionSettings


class AttentionLayer(nn.Module):
    def __init__(self, settings: AttentionSettings):
        super().__init__()
        self.multi_head = MultiHeadAttention(settings)
        self.output = AttentionOutput(settings)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        head_output = self.multi_head(hidden_states, attention_mask)
        return self.output(head_output, hidden_states)
