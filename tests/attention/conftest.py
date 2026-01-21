import pytest
import torch

from settings import AttentionSettings


@pytest.fixture
def attention_settings() -> AttentionSettings:
    return AttentionSettings(
        vocab_size=100,
        hidden_size=32,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    )

@pytest.fixture
def bad_attention_settings(attention_settings:AttentionSettings):
    attention_settings.num_attention_heads = 5
    return attention_settings

@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 5


@pytest.fixture
def sample_hidden_states(
    attention_settings: AttentionSettings, batch_size: int, seq_len: int
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, attention_settings.hidden_size)


@pytest.fixture
def sample_attention_mask() -> torch.Tensor:
    return torch.zeros(2, 1, 1, 5)
