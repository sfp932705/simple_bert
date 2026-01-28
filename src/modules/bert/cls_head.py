import torch
from torch import nn

from modules.bert.backbone import BertBackbone
from settings import BertSettings


class SequenceClassificationBert(nn.Module):
    def __init__(self, settings: BertSettings, num_classes: int):
        super().__init__()
        self.bert = BertBackbone(settings)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)
        self.classifier = nn.Linear(settings.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        return self.classifier(output)
