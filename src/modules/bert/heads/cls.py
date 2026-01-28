import torch
from torch import nn

from settings import BertSettings


class SequenceClassificationHead(nn.Module):
    def __init__(self, settings: BertSettings, num_classes: int):
        super().__init__()
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)
        self.classifier = nn.Linear(settings.hidden_size, num_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        output = self.dropout(pooled_output)
        return self.classifier(output)
