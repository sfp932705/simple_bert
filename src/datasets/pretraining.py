import random
from enum import IntEnum

import torch

from datasets.base import BaseDataset, T_Tokenizer
from datasets.types.inputs.pretraining import PretrainingCorpusData
from datasets.types.outputs.pretraining import PretrainingOutput
from settings import LoaderSettings


class NSPLabel(IntEnum):
    IS_NEXT = 0
    NOT_NEXT = 1


class PretrainingDataset(
    BaseDataset[T_Tokenizer, PretrainingCorpusData, PretrainingOutput]
):
    def __init__(
        self,
        data: PretrainingCorpusData,
        tokenizer: T_Tokenizer,
        loader_settings: LoaderSettings | None = None,
    ):
        super().__init__(data, tokenizer, loader_settings)
        self.samples_index = []
        for doc_idx, doc in enumerate(self.data):
            for sent_idx in range(len(doc)):
                self.samples_index.append((doc_idx, sent_idx))

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, index) -> PretrainingOutput:
        doc_idx, sent_idx = self.samples_index[index]
        document = self.data[doc_idx]
        sent_a = document[sent_idx]

        tokens_a = self.tokenizer.encode(sent_a)
        sent_b, label = self._get_next_sentence(document, sent_idx)
        tokens_b = self.tokenizer.encode(sent_b)
        self._truncate_pair(tokens_a, tokens_b)
        input_ids = (
            [self.tokenizer.cls_token_id]
            + tokens_a
            + [self.tokenizer.sep_token_id]
            + tokens_b
            + [self.tokenizer.sep_token_id]
        )

        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_ids_tensor, attention_mask_tensor = self.pad_and_tensorize(
            input_ids, self.tokenizer.pad_token_id
        )

        assert self.settings is not None
        padding_len = self.settings.max_seq_len - len(input_ids)
        if padding_len > 0:
            token_type_ids = token_type_ids + [0] * padding_len
        token_type_ids_tensor = torch.tensor(token_type_ids, dtype=torch.long)

        input_ids_masked, mlm_labels = self._mask_tokens(input_ids_tensor)
        return PretrainingOutput(
            input_ids=input_ids_masked,
            attention_mask=attention_mask_tensor,
            token_type_ids=token_type_ids_tensor,
            mlm_labels=mlm_labels,
            nsp_labels=torch.tensor(label, dtype=torch.long),
        )

    def _get_next_sentence(
        self, document: list[str], sentence_idx: int
    ) -> tuple[str, IntEnum]:
        label = NSPLabel.IS_NEXT
        if random.random() > 0.5 and sentence_idx + 1 < len(document):
            second_sentence = document[sentence_idx + 1]
        else:
            # Label noise here since we might pick randomly same document and same
            # sentence, but extremely unlikely in huge dataset, so not worth checking.
            label = NSPLabel.NOT_NEXT
            rand_doc_idx = random.randint(0, len(self.data) - 1)
            rand_doc = self.data[rand_doc_idx]
            rand_sent_idx = random.randint(0, len(rand_doc) - 1)
            second_sentence = rand_doc[rand_sent_idx]
        return second_sentence, label

    def _truncate_pair(self, tokens_a, tokens_b):
        assert self.settings is not None
        while True:
            if len(tokens_a) + len(tokens_b) <= self.settings.max_seq_len - 3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _mask_tokens(self, inputs: torch.Tensor):
        mlm_labels = inputs.clone()
        probability_matrix = torch.full(mlm_labels.shape, 0.15)
        special_tokens_mask = (
            (inputs == self.tokenizer.cls_token_id)
            | (inputs == self.tokenizer.sep_token_id)
            | (inputs == self.tokenizer.pad_token_id)
        )

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100

        #  80% of masked tokens -> Replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        #  10% of masked tokens -> Replace with Random Word
        # remaining 10% are left as-is
        indices_random = (
            torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.tokenizer.vocab_size, mlm_labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]
        return inputs, mlm_labels
