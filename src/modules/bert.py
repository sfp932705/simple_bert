import torch
from torch import nn

from modules.embeddings import Embeddings
from modules.encoders.layer import StackedEncoder
from modules.pooler import Pooler
from settings import BertSettings


class Bert(nn.Module):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.embeddings = Embeddings(settings)
        self.encoder = StackedEncoder(settings)
        self.pooler = Pooler(settings.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        sequence_output = self.encoder(
            hidden_states=embedding_output, attention_mask=extended_attention_mask
        )
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output
