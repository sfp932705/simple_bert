import torch
from torch import nn

from modules.bert.backbone import BertBackbone
from settings import BertSettings


class PredictionHeadTransform(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.dense = nn.Linear(settings.hidden_size, settings.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(settings.hidden_size, eps=settings.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLanguageModellingPredictionHead(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.transform = PredictionHeadTransform(settings)
        self.decoder = nn.Linear(settings.hidden_size, settings.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(settings.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)


class MaskedLanguageModellingBert(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.bert = BertBackbone(settings)
        self.cls = BertLanguageModellingPredictionHead(settings)
        self.cls.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None
    ) -> torch.Tensor:
        sequence_output, _ = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return self.cls(sequence_output)

    def compute_loss(self, prediction: torch.Tensor, target=None) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(
            prediction.view(-1, self.bert.settings.vocab_size), target.view(-1)
        )
