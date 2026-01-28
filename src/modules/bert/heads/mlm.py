import torch
from torch import nn

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


class MaskedLanguageModellingHead(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.transform = PredictionHeadTransform(settings)
        self.decoder = nn.Linear(settings.hidden_size, settings.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(settings.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        return self.decoder(hidden_states)
