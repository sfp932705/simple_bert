import torch

from models.attention.layer import AttentionLayer
from settings import AttentionSettings


def test_attention_layer_integration(
    attention_settings: AttentionSettings,
    sample_hidden_states: torch.Tensor,
    sample_attention_mask: torch.Tensor,
):
    layer = AttentionLayer(attention_settings)
    output = layer(sample_hidden_states, attention_mask=sample_attention_mask)
    assert output.shape == sample_hidden_states.shape
    output.sum().backward()
    assert layer.multi_head.query.weight.grad is not None
    assert layer.output.dense.weight.grad is not None


def test_attention_layer_deterministic_when_no_dropout(
    attention_settings: AttentionSettings, sample_hidden_states: torch.Tensor
):
    settings = attention_settings.model_copy(
        update={"attention_probs_dropout_prob": 0.0, "hidden_dropout_prob": 0.0}
    )

    layer = AttentionLayer(settings)
    assert torch.allclose(layer(sample_hidden_states), layer(sample_hidden_states))


def test_attention_layer_deterministic_when_eval(
    attention_settings: AttentionSettings, sample_hidden_states: torch.Tensor
):
    layer = AttentionLayer(attention_settings)
    layer.eval()
    assert torch.allclose(layer(sample_hidden_states), layer(sample_hidden_states))
