import torch
from torch import nn

from modules.bert.cls_head import SequenceClassificationBert


def test_classification_init(
    cls_bert: SequenceClassificationBert, num_classes: int, hidden_size: int
):
    assert isinstance(cls_bert.classifier, nn.Linear)
    assert cls_bert.classifier.in_features == hidden_size
    assert cls_bert.classifier.out_features == num_classes
    assert isinstance(cls_bert.dropout, nn.Dropout)


def test_forward_shape(
    cls_bert: SequenceClassificationBert,
    num_classes: int,
    sample_indices: torch.Tensor,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
):
    logits = cls_bert(sample_indices)
    assert logits.shape == (batch_size, num_classes)


def test_gradient_flow(
    cls_bert: SequenceClassificationBert, sample_indices: torch.Tensor
):
    logits = cls_bert(sample_indices)
    loss = logits.sum()
    loss.backward()
    assert cls_bert.classifier.weight.grad is not None
    assert torch.sum(torch.abs(cls_bert.classifier.weight.grad)) > 0
    embed_grad = cls_bert.bert.embeddings.word_embeddings.weight.grad
    assert embed_grad is not None
    assert torch.sum(torch.abs(embed_grad)) > 0


def test_dropout_behavior(
    cls_bert: SequenceClassificationBert, sample_indices: torch.Tensor
):
    cls_bert.train()
    out1 = cls_bert(sample_indices)
    out2 = cls_bert(sample_indices)
    assert not torch.allclose(out1, out2)
    cls_bert.eval()
    out3 = cls_bert(sample_indices)
    out4 = cls_bert(sample_indices)
    assert torch.allclose(out3, out4)
