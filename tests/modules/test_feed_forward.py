import pytest
import torch
from torch import nn

from modules.feed_forward import Activation, FeedForwardLayer
from settings import FeedForwardSettings


@pytest.mark.parametrize(
    "act_string, expected_class",
    [
        ("GELU", nn.GELU),
        ("gelu", nn.GELU),
        ("relu", nn.ReLU),
        ("RELU", nn.ReLU),
    ],
)
def test_activation_enum_logic(act_string, expected_class):
    act_fn = Activation.from_string(act_string).get_act_fn()
    assert isinstance(act_fn, expected_class)


def test_activation_enum_logic_fails():
    with pytest.raises(ValueError):
        Activation.from_string("sigmoid").get_act_fn()


def test_layer_initialization(ff: FeedForwardLayer):
    assert ff.dense_expansion.in_features == 32
    assert ff.dense_expansion.out_features == 64
    assert ff.dense_contraction.in_features == 64
    assert ff.dense_contraction.out_features == 32
    assert ff.LayerNorm.normalized_shape == (32,)
    assert ff.LayerNorm.eps == 1e-12


def test_forward_pass_shape(ff: FeedForwardLayer, ff_settings: FeedForwardSettings):
    batch_size, seq_len = 2, 10
    input_tensor = torch.randn(batch_size, seq_len, ff_settings.hidden_size)
    output_tensor = ff(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    assert output_tensor.shape == (batch_size, seq_len, ff_settings.hidden_size)


def test_residual_connection(ff: FeedForwardLayer, ff_settings: FeedForwardSettings):
    ff.eval()
    nn.init.constant_(ff.dense_expansion.weight, 0.0)
    nn.init.constant_(ff.dense_expansion.bias, 0.0)
    nn.init.constant_(ff.dense_contraction.weight, 0.0)
    nn.init.constant_(ff.dense_contraction.bias, 0.0)
    input_tensor = torch.randn(1, 5, ff_settings.hidden_size)
    output = ff(input_tensor)
    expected_norm = nn.functional.layer_norm(
        input_tensor, (ff_settings.hidden_size,), eps=ff_settings.layer_norm_eps
    )
    assert torch.allclose(output, expected_norm, atol=1e-6)


def test_dropout_is_applied(ff: FeedForwardLayer, ff_settings: FeedForwardSettings):
    assert isinstance(ff.dropout, nn.Dropout)
    assert ff.dropout.p == ff_settings.hidden_dropout_prob


def test_gelu_vs_relu_math(ff: FeedForwardLayer, ff_relu: FeedForwardLayer):
    negative_val = torch.tensor([[-1.0]])
    out_gelu = ff.activation(negative_val)
    out_relu = ff_relu.activation(negative_val)
    assert out_gelu.item() < 0.0
    assert out_relu.item() == 0.0
