import math

import torch
from torch import nn

from settings import AttentionSettings


class MultiHeadAttention(nn.Module):
    def __init__(self, settings: AttentionSettings):
        super().__init__()
        if settings.hidden_size % settings.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({settings.hidden_size}) must be divisible by "
                f"number of heads ({settings.num_attention_heads})"
            )

        self.num_heads = settings.num_attention_heads
        self.head_dim = settings.hidden_size // settings.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = nn.Linear(settings.hidden_size, self.all_head_size)
        self.key = nn.Linear(settings.hidden_size, self.all_head_size)
        self.value = nn.Linear(settings.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(settings.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # new shape will be: (Batch, Seq_Len, Num_Heads, Head_Dim)
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        # After permutation: (Batch, Num_Heads, Seq_Len, Head_Dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Project to (batch, num_heads, seq_len, head_dim)
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Calculate attention scores: (Q * K^T) / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
