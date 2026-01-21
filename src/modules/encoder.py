import torch
from torch import nn

from modules.attention.layer import MultiHeadAttention
from modules.feed_forward import FeedForwardLayer
from settings import AttentionSettings, FeedForwardSettings


class Encoder(nn.Module):
    def __init__(
        self, attention_settings: AttentionSettings, ff_settings: FeedForwardSettings
    ):
        super().__init__()
        self.attention = MultiHeadAttention(attention_settings)
        self.feed_forward = FeedForwardLayer(ff_settings)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.feed_forward(self.attention(hidden_states, attention_mask))
