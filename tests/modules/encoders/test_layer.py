import torch
from torch import nn

from modules.encoders.encoder import Encoder
from modules.encoders.layer import StackedEncoder


def test_stack_initialization(encoder_layer: StackedEncoder):
    assert isinstance(encoder_layer.layer, nn.ModuleList)
    assert len(encoder_layer.layer) == 12
    assert isinstance(encoder_layer.layer[0], Encoder)


def test_stack_forward_shape(
    encoder_layer: StackedEncoder, batch_size: int, seq_len: int, hidden_size: int
):

    input_states = torch.randn(batch_size, seq_len, hidden_size)
    output = encoder_layer(input_states)
    assert output.shape == input_states.shape
    assert output.shape == (batch_size, seq_len, hidden_size)


def test_gradient_propagation(
    encoder_layer: StackedEncoder, batch_size: int, seq_len: int, hidden_size: int
):
    encoder_layer.train()
    input_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    output = encoder_layer(input_states)
    loss = output.sum()
    loss.backward()
    last_layer_index = 12 - 1
    grad_last: torch.Tensor = encoder_layer.layer[
        last_layer_index
    ].feed_forward.dense_expansion.weight.grad  # type:ignore
    assert grad_last is not None
    assert torch.abs(grad_last).sum() > 0
    grad_first: torch.Tensor = encoder_layer.layer[
        0
    ].feed_forward.dense_expansion.weight.grad  # type:ignore
    assert grad_first is not None
    assert torch.abs(grad_first).sum() > 0


def test_mask_propagation(
    encoder_layer: StackedEncoder, batch_size: int, seq_len: int, hidden_size: int
):
    input_states = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.zeros(batch_size, 1, 1, seq_len)
    output = encoder_layer(input_states, attention_mask=mask)
    assert output.shape == input_states.shape
