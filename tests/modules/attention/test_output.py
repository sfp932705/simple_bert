import torch

from modules.attention.output import AttentionOutput
from settings import BertSettings


def test_output_initialization(settings: BertSettings):
    layer = AttentionOutput(settings)
    assert layer.dense.in_features == settings.hidden_size
    assert layer.dense.out_features == settings.hidden_size
    assert layer.LayerNorm.normalized_shape == (settings.hidden_size,)


def test_output_forward_residual(
    settings: BertSettings, sample_hidden_states: torch.Tensor
):
    layer = AttentionOutput(settings)
    output = layer(
        hidden_states=sample_hidden_states, input_tensor=sample_hidden_states
    )
    assert output.size() == sample_hidden_states.size()
    assert not torch.isnan(output).any()


def test_output_forward_residual_deterministic_when_eval(
    settings: BertSettings, sample_hidden_states: torch.Tensor
):
    layer = AttentionOutput(settings)
    layer.eval()
    out1 = layer(hidden_states=sample_hidden_states, input_tensor=sample_hidden_states)
    out2 = layer(hidden_states=sample_hidden_states, input_tensor=sample_hidden_states)
    assert torch.allclose(out1, out2)
