import torch

from modules.encoders.encoder import Encoder


def test_layer_shape_preservation(
    encoder: Encoder, batch_size: int, seq_len: int, hidden_size: int
):
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, 1, 1, seq_len)
    assert encoder(input_tensor, mask).shape == (batch_size, seq_len, hidden_size)


def test_layer_data_flow(
    encoder: Encoder, batch_size: int, seq_len: int, hidden_size: int
):
    encoder.train()
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    loss = encoder(input_tensor).sum()
    loss.backward()
    assert encoder.attention.query.weight.grad is not None
    assert encoder.feed_forward.dense_expansion.weight.grad is not None


def test_layer_masking_integration(
    encoder: Encoder, batch_size: int, seq_len: int, hidden_size: int
):
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.zeros(batch_size, 1, 1, seq_len)
    output = encoder(input_tensor, mask)
    assert output.shape == input_tensor.shape
