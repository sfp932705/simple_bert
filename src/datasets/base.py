from typing import Generic, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.types.inputs.base import BaseData
from datasets.types.outputs.base import BaseOutput
from settings import LoaderSettings
from token_encoders.base import BaseTokenizer

T_Tokenizer = TypeVar("T_Tokenizer", bound=BaseTokenizer)
T_Data = TypeVar("T_Data", bound=BaseData)
T_DataItem = TypeVar("T_DataItem", bound=BaseOutput)


class BaseDataset(Dataset, Generic[T_Tokenizer, T_Data, T_DataItem]):
    def __init__(
        self,
        data: T_Data,
        tokenizer: T_Tokenizer,
        loader_settings: LoaderSettings | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.settings = loader_settings

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> T_DataItem:
        raise NotImplementedError

    def loader(self, override_settings: LoaderSettings | None = None) -> DataLoader:
        config = override_settings or self.settings
        if not config:
            raise ValueError(
                "No loader settings found. Specify them during init, or here "
            )
        return DataLoader(
            self,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
        )

    def pad_and_tensorize(
        self, input_ids: list[int], padding_value: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = self.settings.max_seq_len
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        attention_mask = [1] * len(input_ids)
        padding_len = max_len - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [padding_value] * padding_len
            attention_mask = attention_mask + [0] * padding_len
        return torch.tensor(input_ids), torch.tensor(attention_mask)
