import torch

from datasets.types.inputs.finetuning import FinetuningData
from datasets.types.outputs.finetuning import FinetuningOutput
from settings import LoaderSettings

from .base import BaseDataset, T_Tokenizer


class FinetuningDataset(BaseDataset[T_Tokenizer, FinetuningData, FinetuningOutput]):
    def __init__(
        self,
        data: FinetuningData,
        tokenizer: T_Tokenizer,
        loader_settings: LoaderSettings | None = None,
    ):
        super().__init__(data, tokenizer, loader_settings)

    def __getitem__(self, index) -> FinetuningOutput:
        sample = self.data[index]
        tokens = self.tokenizer.encode(sample.text)
        input_ids = (
            [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        )

        input_ids_tensor, mask_tensor = self.pad_and_tensorize(
            input_ids, padding_value=self.tokenizer.pad_token_id
        )

        token_type_ids = torch.zeros_like(input_ids_tensor)
        return FinetuningOutput(
            input_ids=input_ids_tensor,
            attention_mask=mask_tensor,
            token_type_ids=token_type_ids,
            labels=torch.tensor(sample.label, dtype=torch.long),
        )
