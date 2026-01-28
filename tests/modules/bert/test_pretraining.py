import torch

from modules.bert.pretraining import BertForPreTraining


def test_pretraining_output_shapes(
    pretraining_bert: BertForPreTraining,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):
    prediction_scores, seq_relationship_score = pretraining_bert(sample_indices)
    assert prediction_scores.shape == (batch_size, seq_len, vocab_size)
    assert seq_relationship_score.shape == (batch_size, 2)


def test_weight_tying_is_active(pretraining_bert: BertForPreTraining):
    embedding_weight = pretraining_bert.bert.embeddings.word_embeddings.weight
    decoder_weight = pretraining_bert.mlm.decoder.weight
    assert embedding_weight is decoder_weight
    with torch.no_grad():
        embedding_weight[0, 0] = 1234.5
    assert decoder_weight[0, 0] == 1234.5


def test_pretraining_loss_computation(
    pretraining_bert: BertForPreTraining,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):
    prediction_scores, seq_relationship_score = pretraining_bert(sample_indices)
    mlm_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    nsp_labels = torch.randint(0, 2, (batch_size,))
    loss = pretraining_bert.compute_loss(
        prediction_scores, seq_relationship_score, mlm_labels, nsp_labels
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_gradient_flow_both_heads(
    pretraining_bert: BertForPreTraining,
    sample_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):
    preds, nsp_scores = pretraining_bert(sample_indices)
    mlm_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    nsp_labels = torch.randint(0, 2, (batch_size,))
    loss = pretraining_bert.compute_loss(preds, nsp_scores, mlm_labels, nsp_labels)
    loss.backward()
    assert pretraining_bert.mlm.bias.grad is not None
    assert torch.sum(torch.abs(pretraining_bert.mlm.bias.grad)) > 0
    assert pretraining_bert.nsp.seq_relationship.weight.grad is not None
    assert torch.sum(torch.abs(pretraining_bert.nsp.seq_relationship.weight.grad)) > 0
    embed_grad = pretraining_bert.bert.embeddings.word_embeddings.weight.grad
    assert embed_grad is not None
    assert torch.sum(torch.abs(embed_grad)) > 0
