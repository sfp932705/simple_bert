import pytest
import torch
from torch.utils.data import DataLoader

from datasets.base import BaseDataset
from settings import LoaderSettings
from token_encoders.bpe import BPETokenizer


class MockedBaseDataset(BaseDataset[BPETokenizer, list[int], dict]):  # type: ignore

    def __getitem__(self, index) -> dict:
        return {"value": self.data[index]}


@pytest.fixture
def sample_data() -> list[int]:
    return [1, 2, 3, 4, 5]


@pytest.fixture
def dataset(
    sample_data: list[int],
    tokenizer: BPETokenizer,
    settings: LoaderSettings,
) -> MockedBaseDataset:
    return MockedBaseDataset(sample_data, tokenizer, settings)


def test_base_dataset_initialization(
    dataset: MockedBaseDataset,
    sample_data: list[int],
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
    sample_data: list[int],
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
    sample_data: list[int], tokenizer: BPETokenizer, settings: LoaderSettings
):
    ds = BaseDataset(sample_data, tokenizer, settings)  # type: ignore
    with pytest.raises(NotImplementedError):
        _ = ds[0]
