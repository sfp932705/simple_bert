import torch

from modules.bert.mlm_head import MaskedLanguageModellingBert


def test_mlm_output_shape(
    mlm_bert: MaskedLanguageModellingBert,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):

    prediction_scores = mlm_bert(sample_indices)
    assert prediction_scores.shape == (batch_size, seq_len, vocab_size)


def test_weight_tying_is_active(mlm_bert: MaskedLanguageModellingBert):
    embedding_weight = mlm_bert.bert.embeddings.word_embeddings.weight
    decoder_weight = mlm_bert.cls.decoder.weight
    assert embedding_weight is decoder_weight

    with torch.no_grad():
        embedding_weight[0, 0] = 999.99
    assert decoder_weight[0, 0] == 999.99


def test_bias_handling(mlm_bert: MaskedLanguageModellingBert, vocab_size: int):
    assert isinstance(mlm_bert.cls.bias, torch.nn.Parameter)
    assert mlm_bert.cls.bias.shape[0] == vocab_size
    assert mlm_bert.cls.decoder.bias is mlm_bert.cls.bias


def test_compute_loss(
    mlm_bert: MaskedLanguageModellingBert,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):

    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    preds = mlm_bert(sample_indices)
    loss = mlm_bert.compute_loss(preds, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert preds.shape == (batch_size, seq_len, vocab_size)
    assert loss.item() > 0


def test_gradient_flow_through_head(
    mlm_bert: MaskedLanguageModellingBert,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = mlm_bert.compute_loss(mlm_bert(sample_indices),labels)
    loss.backward()
    assert mlm_bert.cls.bias.grad is not None
    assert torch.sum(torch.abs(mlm_bert.cls.bias.grad)) > 0
    embed_grad = mlm_bert.bert.embeddings.word_embeddings.weight.grad
    assert embed_grad is not None
    assert torch.sum(torch.abs(embed_grad)) > 0
