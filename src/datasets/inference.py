import torch

from datasets.base import BaseDataset, T_Tokenizer
from datasets.types.inputs.inference import InferenceData
from datasets.types.outputs.inference import InferenceOutput
from settings import LoaderSettings


class InferenceDataset(BaseDataset[T_Tokenizer, InferenceData, InferenceOutput]):
    def __init__(
        self,
        data: InferenceData,
        tokenizer: T_Tokenizer,
        loader_settings: LoaderSettings | None = None,
    ):
        super().__init__(data, tokenizer, loader_settings)

    def __getitem__(self, index) -> InferenceOutput:
        text = self.data[index]
        tokens = self.tokenizer.encode(text)
        input_ids = (
            [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        )
        input_ids_tensor, mask_tensor = self.pad_and_tensorize(
            input_ids, padding_value=self.tokenizer.pad_token_id
        )
        token_type_ids = torch.zeros_like(input_ids_tensor)
        return InferenceOutput(
            input_ids=input_ids_tensor,
            attention_mask=mask_tensor,
            token_type_ids=token_type_ids,
        )
