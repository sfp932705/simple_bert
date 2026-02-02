import pytest
import torch

from datasets.finetuning import FinetuningDataset
from datasets.types.inputs.finetuning import FinetuningData
from settings import LoaderSettings
from token_encoders.bpe import BPETokenizer


@pytest.fixture
def sample_data() -> FinetuningData:
    return FinetuningData.from_lists(
        texts=["short text", "a slightly longer text example"], labels=[0, 1]
    )


@pytest.fixture
def dataset(
    sample_data: FinetuningData, tokenizer: BPETokenizer, settings: LoaderSettings
) -> FinetuningDataset:
    return FinetuningDataset(sample_data, tokenizer, settings)



def test_finetuning_getitem_structure(
    dataset: FinetuningDataset, settings: LoaderSettings
):
    output = dataset[0]

    assert isinstance(output.input_ids, torch.Tensor)
    assert isinstance(output.attention_mask, torch.Tensor)
    assert isinstance(output.token_type_ids, torch.Tensor)
    assert isinstance(output.labels, torch.Tensor)

    expected_shape = (settings.max_seq_len,)
    assert output.input_ids.shape == expected_shape
    assert output.attention_mask.shape == expected_shape
    assert output.token_type_ids.shape == expected_shape

    assert output.labels.item() == 0  # "short text" has label 0
    assert output.labels.dtype == torch.long


def test_finetuning_special_tokens_insertion(
    dataset: FinetuningDataset, tokenizer: BPETokenizer
):
    item = dataset[0]
    assert item.input_ids[0] == tokenizer.cls_token_id
    assert item.input_ids[3] == tokenizer.sep_token_id


@pytest.mark.parametrize("idx, text_word_count", [(0, 2), (1, 5)])
def test_finetuning_padding_logic(
    dataset: FinetuningDataset, tokenizer: BPETokenizer, idx: int, text_word_count: int
):
    item = dataset[idx]
    real_len = 1 + text_word_count + 1
    assert item.attention_mask[:real_len].sum() == real_len
    assert item.attention_mask[real_len:].sum() == 0
    if real_len < len(item.input_ids):
        assert (item.input_ids[real_len:] == tokenizer.pad_token_id).all()
    assert torch.all(item.token_type_ids == 0)


def test_finetuning_truncation(tokenizer: BPETokenizer, settings: LoaderSettings):
    long_text = " ".join(["word"] * 20)
    data = FinetuningData.from_lists([long_text], [0])
    ds = FinetuningDataset(data, tokenizer, settings)
    item = ds[0]
    assert len(item.input_ids) == settings.max_seq_len
    assert len(item.attention_mask) == settings.max_seq_len
    assert torch.all(item.attention_mask == 1)
