import pytest
import torch

from modules.bert.backbone import BertBackbone
from modules.bert.finetuning import BertForSequenceClassification
from modules.bert.pretraining import BertForPreTraining
from modules.embeddings import Embeddings
from modules.encoders.encoder import Encoder
from modules.encoders.layer import StackedEncoder
from modules.feed_forward import FeedForwardLayer
from modules.pooler import Pooler
from settings import BertSettings


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
def vocab_size() -> int:
    return 100


@pytest.fixture
def num_classes() -> int:
    return 5


@pytest.fixture
def settings(hidden_size: int, vocab_size: int) -> BertSettings:
    return BertSettings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
    )


@pytest.fixture
def ff(settings: BertSettings) -> FeedForwardLayer:
    return FeedForwardLayer(settings)


@pytest.fixture
def ff_relu(settings: BertSettings) -> FeedForwardLayer:
    settings.hidden_act = "relu"
    return FeedForwardLayer(settings)


@pytest.fixture
def embeddings(settings: BertSettings) -> Embeddings:
    return Embeddings(settings)


@pytest.fixture
def bad_attention_settings(settings: BertSettings):
    settings.num_attention_heads = 5
    return settings


@pytest.fixture
def encoder(settings: BertSettings) -> Encoder:
    return Encoder(settings)


@pytest.fixture
def encoder_layer(settings: BertSettings) -> StackedEncoder:
    return StackedEncoder(settings)


@pytest.fixture
def sample_hidden_states(
    settings: BertSettings, batch_size: int, seq_len: int
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, settings.hidden_size)


@pytest.fixture
def sample_attention_mask() -> torch.Tensor:
    return torch.zeros(2, 1, 1, 5)


@pytest.fixture
def sample_indices(
    settings: BertSettings, batch_size: int, seq_len: int
) -> torch.Tensor:
    return torch.randint(0, settings.vocab_size, (batch_size, seq_len))


@pytest.fixture
def pooler(hidden_size: int) -> Pooler:
    return Pooler(hidden_size)


@pytest.fixture
def bert_backbone(settings: BertSettings) -> BertBackbone:
    return BertBackbone(settings=settings)


@pytest.fixture
def pretraining_bert(settings: BertSettings) -> BertForPreTraining:
    return BertForPreTraining(settings)


@pytest.fixture
def finetuning_bert(settings: BertSettings, num_classes: int):
    return BertForSequenceClassification(settings, num_classes=num_classes)
