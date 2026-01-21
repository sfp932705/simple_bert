import torch

from modules.embeddings import Embeddings


def test_embeddings_shape(embeddings: Embeddings):
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    output = embeddings(input_ids)
    assert output.shape == (2, 5, 32)


def test_embeddings_auto_position_logic(embeddings: Embeddings):
    input_ids = torch.tensor([[10, 20]])
    output = embeddings(input_ids)
    assert output.shape == (1, 2, 32)
    assert not torch.isnan(output).any()


def test_embeddings_position_affects_summation(embeddings: Embeddings):
    embeddings.eval()
    input_ids = torch.tensor([[10, 10]])
    output = embeddings(input_ids)
    assert not torch.allclose(output[0, 0], output[0, 1])
    output = embeddings(input_ids, position_ids=torch.tensor([[0, 0]]))
    assert torch.allclose(output[0, 0], output[0, 1])
