import torch
from torch import nn

from modules.attention.layer import MultiHeadAttention
from modules.feed_forward import FeedForwardLayer
from settings import EncoderSettings


class Encoder(nn.Module):
    def __init__(self, settings: EncoderSettings):
        super().__init__()
        self.attention = MultiHeadAttention(settings.attention)
        self.feed_forward = FeedForwardLayer(settings.ff)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.feed_forward(self.attention(hidden_states, attention_mask))
