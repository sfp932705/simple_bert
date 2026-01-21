import pytest
import torch

from modules.embeddings import Embeddings
from modules.encoders.encoder import Encoder
from modules.encoders.layer import StackedEncoder
from modules.feed_forward import FeedForwardLayer
from modules.pooler import Pooler
from settings import (
    AttentionSettings,
    EmbeddingSettings,
    EncoderSettings,
    FeedForwardSettings,
    LayerCommonSettings,
)


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 5


@pytest.fixture
def hidden_size() -> int:
    return 32


@pytest.fixture
def embedding_settings(hidden_size: int) -> EmbeddingSettings:
    return EmbeddingSettings(
        vocab_size=100, hidden_size=hidden_size, max_position_embeddings=50
    )


@pytest.fixture
def ff_settings(hidden_size: int) -> FeedForwardSettings:
    return FeedForwardSettings(hidden_size=hidden_size, intermediate_size=64)


@pytest.fixture
def ff(ff_settings: FeedForwardSettings) -> FeedForwardLayer:
    return FeedForwardLayer(ff_settings)


@pytest.fixture
def ff_relu(ff_settings: FeedForwardSettings) -> FeedForwardLayer:
    ff_settings.hidden_act = "relu"
    return FeedForwardLayer(ff_settings)


@pytest.fixture
def embeddings(embedding_settings: EmbeddingSettings) -> Embeddings:
    return Embeddings(embedding_settings)


@pytest.fixture
def attention_settings(hidden_size: int) -> AttentionSettings:
    return AttentionSettings(
        vocab_size=100,
        hidden_size=hidden_size,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    )


@pytest.fixture
def bad_attention_settings(attention_settings: AttentionSettings):
    attention_settings.num_attention_heads = 5
    return attention_settings


@pytest.fixture
def encoder(
    attention_settings: AttentionSettings, ff_settings: FeedForwardSettings
) -> Encoder:
    return Encoder(EncoderSettings(attention=attention_settings, ff=ff_settings))


@pytest.fixture
def encoder_layer(
    attention_settings: AttentionSettings, ff_settings: FeedForwardSettings
) -> StackedEncoder:
    return StackedEncoder(EncoderSettings(attention=attention_settings, ff=ff_settings))


@pytest.fixture
def sample_hidden_states(
    attention_settings: AttentionSettings, batch_size: int, seq_len: int
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, attention_settings.hidden_size)


@pytest.fixture
def sample_attention_mask() -> torch.Tensor:
    return torch.zeros(2, 1, 1, 5)


@pytest.fixture
def pooler(hidden_size: int) -> Pooler:
    settings = LayerCommonSettings()
    settings.hidden_size = hidden_size
    return Pooler(settings)
