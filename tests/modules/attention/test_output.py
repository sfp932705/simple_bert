import torch

from modules.attention.output import AttentionOutput
from settings import AttentionSettings


def test_output_initialization(attention_settings: AttentionSettings):
    layer = AttentionOutput(attention_settings)
    assert layer.dense.in_features == attention_settings.hidden_size
    assert layer.dense.out_features == attention_settings.hidden_size
    assert layer.LayerNorm.normalized_shape == (attention_settings.hidden_size,)


def test_output_forward_residual(
    attention_settings: AttentionSettings, sample_hidden_states: torch.Tensor
):
    layer = AttentionOutput(attention_settings)
    output = layer(
        hidden_states=sample_hidden_states, input_tensor=sample_hidden_states
    )
    assert output.size() == sample_hidden_states.size()
    assert not torch.isnan(output).any()


def test_output_forward_residual_deterministic_when_eval(
    attention_settings: AttentionSettings, sample_hidden_states: torch.Tensor
):
    layer = AttentionOutput(attention_settings)
    layer.eval()
    out1 = layer(hidden_states=sample_hidden_states, input_tensor=sample_hidden_states)
    out2 = layer(hidden_states=sample_hidden_states, input_tensor=sample_hidden_states)
    assert torch.allclose(out1, out2)
