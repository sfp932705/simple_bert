from unittest.mock import patch

import pytest
import torch

from datasets.pretraining import NSPLabel, PretrainingDataset
from datasets.types.inputs.pretraining import PretrainingCorpusData
from settings import LoaderSettings
from token_encoders.bpe import BPETokenizer


@pytest.fixture
def sample_corpus() -> PretrainingCorpusData:
    return PretrainingCorpusData(
        documents=[["doc0_s0", "doc0_s1", "doc0_s2"], ["doc1_s0"]]
    )


@pytest.fixture
def dataset(
    sample_corpus: PretrainingCorpusData,
    tokenizer: BPETokenizer,
    settings: LoaderSettings,
) -> PretrainingDataset:
    return PretrainingDataset(sample_corpus, tokenizer, settings)


def test_init_indexing(dataset: PretrainingDataset):
    assert len(dataset) == 4
    assert len(dataset.samples_index) == 4
    expected_indices = [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert dataset.samples_index == expected_indices


@pytest.mark.parametrize(
    "doc, current_idx, random_val, randint_effects, expected_text, expected_label",
    [
        (["s0", "s1", "s2"], 0, 0.9, None, "s1", NSPLabel.IS_NEXT),
        (["s0", "s1", "s2"], 0, 0.1, [1, 0], "doc1_s0", NSPLabel.NOT_NEXT),
        (["s0", "s1"], 1, 0.9, [0, 0], "doc0_s0", NSPLabel.NOT_NEXT),
    ],
)
def test_get_next_sentence_cases(
    dataset: PretrainingDataset,
    doc: list[str],
    current_idx: int,
    random_val: float,
    randint_effects: list[int] | None,
    expected_text: str,
    expected_label: NSPLabel,
):
    with patch("random.random", return_value=random_val):
        with patch("random.randint", side_effect=randint_effects):
            sent_b, label = dataset._get_next_sentence(doc, current_idx)
    assert sent_b == expected_text
    assert label == expected_label


def test_truncate_pair_logic(dataset: PretrainingDataset, settings: LoaderSettings):
    tokens_a = [1] * 8
    tokens_b = [2] * 4
    dataset._truncate_pair(tokens_a, tokens_b)
    assert len(tokens_a) + len(tokens_b) <= settings.max_seq_len - 3
    assert len(tokens_a) == 5
    assert len(tokens_b) == 4


@pytest.mark.parametrize(
    "len_a, len_b, expected_a, expected_b", [(8, 4, 5, 4), (4, 8, 4, 5)]
)
def test_truncate_pair_logic(
    dataset: PretrainingDataset,
    settings: LoaderSettings,
    len_a: int,
    len_b: int,
    expected_a: int,
    expected_b: int,
):
    tokens_a = [1] * len_a
    tokens_b = [2] * len_b
    dataset._truncate_pair(tokens_a, tokens_b)
    assert len(tokens_a) == expected_a
    assert len(tokens_b) == expected_b
    assert len(tokens_a) + len(tokens_b) <= settings.max_seq_len - 3


def test_mask_tokens_special_tokens_protected(dataset: PretrainingDataset):
    input_ids = torch.tensor([1, 10, 11, 2, 0])
    with patch("torch.full", return_value=torch.ones(input_ids.shape)):
        masked_input, labels = dataset._mask_tokens(input_ids)

    assert masked_input[0] == 1
    assert masked_input[3] == 2
    assert masked_input[4] == 0
    assert labels[0] == -100
    assert labels[3] == -100
    assert labels[4] == -100
    assert masked_input[1] == 3  # mask id
    assert labels[1] == 10


def test_mask_tokens_values(dataset: PretrainingDataset):
    input_ids = torch.tensor([10, 11])
    torch.manual_seed(42)
    masked_input, labels = dataset._mask_tokens(input_ids)

    assert masked_input.shape == input_ids.shape
    assert labels.shape == input_ids.shape

    for i in range(len(input_ids)):
        if labels[i] != -100:
            assert labels[i] == input_ids[i]


def test_getitem_structure(
    dataset: PretrainingDataset, settings: LoaderSettings, tokenizer: BPETokenizer
):
    with patch("random.random", return_value=0.9):
        item = dataset[0]

    assert isinstance(item.input_ids, torch.Tensor)
    assert isinstance(item.mlm_labels, torch.Tensor)
    assert isinstance(item.token_type_ids, torch.Tensor)
    assert isinstance(item.nsp_labels, torch.Tensor)

    assert item.input_ids.shape[0] == settings.max_seq_len
    assert item.token_type_ids.shape[0] == settings.max_seq_len

    token_types = item.token_type_ids.tolist()
    assert 0 in token_types
    assert 1 in token_types
    assert token_types[0] == 0

    assert item.nsp_labels.item() == 0
