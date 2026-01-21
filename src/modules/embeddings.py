import torch
from torch import nn

from settings import EmbeddingSettings


class Embeddings(nn.Module):
    position_ids: torch.Tensor

    def __init__(self, settings: EmbeddingSettings):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            settings.vocab_size, settings.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            settings.max_position_embeddings, settings.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            settings.type_vocab_size, settings.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(settings.hidden_size, eps=settings.layer_norm_eps)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(settings.max_position_embeddings).expand((1, -1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=input_ids.device
            )

        inputs_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings
        return self.dropout(self.LayerNorm(embeddings))
