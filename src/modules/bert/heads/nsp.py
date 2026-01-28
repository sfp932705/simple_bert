import torch
from torch import nn

from settings import BertSettings


class NextSentencePredictionHead(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.seq_relationship = nn.Linear(settings.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
