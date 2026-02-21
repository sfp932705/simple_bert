from dataclasses import dataclass

import torch
from torch import nn

from data.types.outputs.pretraining import PretrainingOutput
from modules.bert.backbone import BertBackbone
from modules.bert.heads.mlm import MaskedLanguageModellingHead
from modules.bert.heads.nsp import NextSentencePredictionHead
from modules.trainable import TrainableModel, TrainForwardPassOutput
from settings import BertSettings


@dataclass
class PretrainingForwardPassOutput(TrainForwardPassOutput):
    mlm_preds: torch.Tensor
    nsp_preds: torch.Tensor


class BertForPreTraining(
    TrainableModel[PretrainingOutput, PretrainingForwardPassOutput]
):
    def __init__(self, settings: BertSettings):
        super().__init__()
        self.bert = BertBackbone(settings)
        self.mlm = MaskedLanguageModellingHead(settings)
        self.nsp = NextSentencePredictionHead(settings)
        self.apply(self._init_weights)
        self.mlm.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        prediction_scores = self.mlm(sequence_output)
        seq_relationship_score = self.nsp(pooled_output)
        return prediction_scores, seq_relationship_score

    def compute_loss(
        self,
        prediction_scores: torch.Tensor,
        seq_relationship_score: torch.Tensor,
        mlm_labels: torch.Tensor,
        nsp_labels: torch.Tensor,
    ) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.bert.settings.vocab_size),
            mlm_labels.view(-1),
        )

        next_sentence_loss = loss_fct(
            seq_relationship_score.view(-1, 2), nsp_labels.view(-1)
        )
        return masked_lm_loss + next_sentence_loss

    def train_forward_from_dataset_batch(
        self, batch: PretrainingOutput
    ) -> PretrainingForwardPassOutput:
        prediction_scores, seq_relationship_score = self.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
        )
        loss = self.compute_loss(
            prediction_scores,
            seq_relationship_score,
            batch.mlm_labels,
            batch.nsp_labels,
        )
        return PretrainingForwardPassOutput(
            loss=loss, mlm_preds=prediction_scores, nsp_preds=seq_relationship_score
        )
