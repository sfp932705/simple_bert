import pytest
import torch

from models.attention.multihead import MultiHeadAttention
from settings import AttentionSettings


def test_multi_head_initialization(attention_settings: AttentionSettings):
    layer = MultiHeadAttention(attention_settings)
    assert layer.num_heads == attention_settings.num_attention_heads
    assert layer.head_dim == 8
    assert layer.query.in_features == attention_settings.hidden_size
    assert layer.query.out_features == attention_settings.hidden_size


def test_multi_head_initialization_failure(bad_attention_settings: AttentionSettings):
    with pytest.raises(ValueError):
        MultiHeadAttention(bad_attention_settings)


def test_transpose_for_scores(
    attention_settings: AttentionSettings, batch_size: int, seq_len: int
):
    layer = MultiHeadAttention(attention_settings)
    hidden = attention_settings.hidden_size
    dummy_input = torch.randn(batch_size, seq_len, hidden)
    transposed = layer.transpose_for_scores(dummy_input)
    expected_shape = (
        batch_size,
        attention_settings.num_attention_heads,
        seq_len,
        layer.head_dim,
    )
    assert transposed.shape == expected_shape


def test_multi_head_forward_shape(
    attention_settings: AttentionSettings, sample_hidden_states: torch.Tensor
):
    heads = MultiHeadAttention(attention_settings)
    output = heads(sample_hidden_states)
    assert output.dim() == 3
    assert output.size() == sample_hidden_states.size()
