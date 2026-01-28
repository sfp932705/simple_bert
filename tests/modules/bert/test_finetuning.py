import torch
from torch import nn

from modules.bert.finetuning import BertForSequenceClassification
from modules.bert.heads.cls import SequenceClassificationHead


def test_classification_structure(
    finetuning_bert: BertForSequenceClassification, num_classes: int, hidden_size: int
):
    assert isinstance(finetuning_bert.classifier, SequenceClassificationHead)
    assert isinstance(finetuning_bert.classifier.classifier, nn.Linear)
    assert finetuning_bert.classifier.classifier.out_features == num_classes
    assert finetuning_bert.classifier.classifier.in_features == hidden_size


def test_classification_forward_shape(
    finetuning_bert: BertForSequenceClassification,
    sample_indices: torch.Tensor,
    batch_size: int,
    num_classes: int,
):
    logits = finetuning_bert(sample_indices)
    assert logits.shape == (batch_size, num_classes)


def test_classification_loss(
    finetuning_bert: BertForSequenceClassification,
    sample_indices: torch.Tensor,
    batch_size: int,
    num_classes: int,
):
    logits = finetuning_bert(sample_indices)
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = finetuning_bert.compute_loss(logits, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_dropout_behavior(
    finetuning_bert: BertForSequenceClassification, sample_indices: torch.Tensor
):
    finetuning_bert.train()
    out1 = finetuning_bert(sample_indices)
    out2 = finetuning_bert(sample_indices)
    assert not torch.allclose(out1, out2)

    finetuning_bert.eval()
    out3 = finetuning_bert(sample_indices)
    out4 = finetuning_bert(sample_indices)
    assert torch.allclose(out3, out4)
