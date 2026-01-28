import torch
from torch import nn

from modules.bert.heads.cls import SequenceClassificationHead
from src.modules.bert.backbone import BertBackbone
from src.settings import BertSettings


class BertForSequenceClassification(nn.Module):
    def __init__(self, settings: BertSettings, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bert = BertBackbone(settings)
        self.classifier = SequenceClassificationHead(settings, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        _, pooled_output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return self.classifier(pooled_output)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
