from dataclasses import dataclass

import pytest
import torch
from torch.utils.data import DataLoader

from data.base import BaseDataset
from data.types.inputs.base import BaseData
from data.types.outputs.base import BaseOutput
from settings import LoaderSettings
from token_encoders.bpe import BPETokenizer


@dataclass
class MockedOutput(BaseOutput):
    value: int


@dataclass
class MockedInput(BaseData):
    def __init__(self, data: list[int]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return self.data.__iter__()


class MockedBaseDataset(BaseDataset[BPETokenizer, MockedInput, MockedOutput]):

    def __getitem__(self, index) -> MockedOutput:
        ones = torch.ones(1)
        return MockedOutput(
            value=self.data[index],
            input_ids=ones,
            attention_mask=ones,
            token_type_ids=ones,
        )


@pytest.fixture
def sample_data() -> MockedInput:
    return MockedInput([1, 2, 3, 4, 5])


@pytest.fixture
def dataset(
    sample_data: MockedInput,
    tokenizer: BPETokenizer,
    settings: LoaderSettings,
) -> MockedBaseDataset:
    return MockedBaseDataset(sample_data, tokenizer, settings)


def test_base_dataset_initialization(
    dataset: MockedBaseDataset,
    sample_data: MockedInput,
    tokenizer: BPETokenizer,
    settings: LoaderSettings,
):
    assert dataset.data == sample_data
    assert dataset.tokenizer == tokenizer
    assert dataset.settings == settings
    assert len(dataset) == 5


def test_base_dataset_loader_creation(
    dataset: MockedBaseDataset, settings: LoaderSettings
):
    dl = dataset.loader()
    assert isinstance(dl, DataLoader)
    assert dl.batch_size == settings.batch_size
    assert dl.num_workers == settings.num_workers
    assert dl.drop_last == settings.drop_last


def test_base_dataset_loader_override(dataset: MockedBaseDataset):
    new_settings = LoaderSettings(
        max_seq_len=10,
        batch_size=99,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    assert dataset.settings is not None
    assert dataset.settings.batch_size != 99
    dl = dataset.loader(override_settings=new_settings)
    assert dl.batch_size == 99
    assert dataset.settings.batch_size == 99


def test_base_dataset_loader_error(
    sample_data: MockedInput,
    tokenizer: BPETokenizer,
):
    ds = MockedBaseDataset(sample_data, tokenizer, loader_settings=None)
    with pytest.raises(ValueError, match="No loader settings found"):
        ds.loader()


@pytest.mark.parametrize(
    "input_ids, expected_mask_sum", [([1, 2, 3], 3), ([1] * 12, 12), ([1] * 15, 12)]
)
def test_pad_and_tensorize(
    dataset: MockedBaseDataset,
    settings: LoaderSettings,
    input_ids: list[int],
    expected_mask_sum: int,
):
    pad_val = 999
    ids_tensor, mask_tensor = dataset.pad_and_tensorize(
        input_ids, padding_value=pad_val
    )
    assert isinstance(ids_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert ids_tensor.shape[0] == settings.max_seq_len
    assert mask_tensor.shape[0] == settings.max_seq_len
    assert mask_tensor.sum().item() == expected_mask_sum

    if len(input_ids) < settings.max_seq_len:
        real_len = len(input_ids)
        assert (ids_tensor[real_len:] == pad_val).all()
        assert (mask_tensor[real_len:] == 0).all()


def test_getitem_abstract_enforcement(
    sample_data: MockedInput, tokenizer: BPETokenizer, settings: LoaderSettings
):
    ds = BaseDataset(sample_data, tokenizer, settings)  # type: ignore
    with pytest.raises(NotImplementedError):
        _ = ds[0]


def test_loader_returns_batched_dataclass(
    dataset: MockedBaseDataset, settings: LoaderSettings
):
    dl = dataset.loader()
    batch = next(iter(dl))
    assert not isinstance(batch, list)
    assert isinstance(batch, MockedOutput)
    assert batch.input_ids.shape[0] == settings.batch_size
