import pytest
import torch

from modules.attention.multihead import MultiHeadAttention
from settings import BertSettings


def test_multi_head_initialization(settings: BertSettings):
    layer = MultiHeadAttention(settings)
    assert layer.num_heads == settings.num_attention_heads
    assert layer.head_dim == 8
    assert layer.query.in_features == settings.hidden_size
    assert layer.query.out_features == settings.hidden_size


def test_multi_head_initialization_failure(bad_attention_settings: BertSettings):
    with pytest.raises(ValueError):
        MultiHeadAttention(bad_attention_settings)


def test_transpose_for_scores(settings: BertSettings, batch_size: int, seq_len: int):
    layer = MultiHeadAttention(settings)
    hidden = settings.hidden_size
    dummy_input = torch.randn(batch_size, seq_len, hidden)
    transposed = layer.transpose_for_scores(dummy_input)
    expected_shape = (
        batch_size,
        settings.num_attention_heads,
        seq_len,
        layer.head_dim,
    )
    assert transposed.shape == expected_shape


def test_multi_head_forward_shape(
    settings: BertSettings, sample_hidden_states: torch.Tensor
):
    heads = MultiHeadAttention(settings)
    output = heads(sample_hidden_states)
    assert output.dim() == 3
    assert output.size() == sample_hidden_states.size()
