import torch
from torch import nn

from modules.pooler import Pooler


def test_pooler_initialization(pooler: Pooler, hidden_size: int):
    assert isinstance(pooler.dense, nn.Linear)
    assert isinstance(pooler.activation, nn.Tanh)
    assert pooler.dense.in_features == hidden_size
    assert pooler.dense.out_features == hidden_size


def test_pooler_output_shape(
    pooler: Pooler, batch_size: int, seq_len: int, hidden_size: int
):

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    assert pooler(input_tensor).shape == (batch_size, hidden_size)


def test_pooler_selects_first_token(
    pooler: Pooler, batch_size: int, seq_len: int, hidden_size: int
):
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    input_tensor[:, 0, :] = 1.0  # The [CLS] tokens
    input_tensor[:, 1:, :] = 100.0  # The rest
    output = pooler(input_tensor)
    cls_token = input_tensor[:, 0]
    expected_output = torch.tanh(pooler.dense(cls_token))
    assert torch.allclose(output, expected_output, atol=1e-6)


def test_pooler_tanh_range(
    pooler: Pooler, batch_size: int, seq_len: int, hidden_size: int
):
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    output = pooler(input_tensor)
    assert output.max().item() <= 1.0
    assert output.min().item() >= -1.0


def test_pooler_gradient_flow(
    pooler: Pooler, batch_size: int, seq_len: int, hidden_size: int
):
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    loss = pooler(input_tensor).sum()
    loss.backward()
    grad_cls = input_tensor.grad[0, 0].abs().sum().item()  # type:ignore
    assert grad_cls > 0.0
    grad_token1 = input_tensor.grad[0, 1].abs().sum().item()  # type:ignore
    assert grad_token1 == 0.0
