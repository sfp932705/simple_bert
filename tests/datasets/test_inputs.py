from pathlib import Path

import pytest

from datasets.types.inputs.finetuning import FinetuningData
from datasets.types.inputs.inference import InferenceData
from datasets.types.inputs.pretraining import PretrainingCorpusData


def test_finetuning_input_data():
    texts = ["hello", "world"]
    labels = [0, 1]
    data = FinetuningData.from_lists(texts, labels)
    assert len(data) == 2
    assert data[0].text == "hello"
    assert data[0].label == 0
    assert len([d for d in data]) == 2
    with pytest.raises(ValueError, match="same length"):
        FinetuningData.from_lists(["one"], [0, 1])


def test_inference_input_data(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("line1\n\nline2\n", encoding="utf-8")
    data = InferenceData.from_file(str(f))
    assert len(data) == 2
    assert len([d for d in data]) == 2
    assert data[0] == "line1"


def test_pretraining_input_data(tmp_path: Path):
    f = tmp_path / "wiki.txt"
    content = "Doc1 Sent1.Doc1 Sent2.\nDoc1 Sent3.|||Doc2 Sent1. Doc2 Sent2."
    f.write_text(content, encoding="utf-8")
    data = PretrainingCorpusData.from_file(f, doc_separator="|||")
    assert len(data) == 2
    assert len(data[0]) == 2
    assert len(data[1]) == 1
    assert len([d for d in data]) == 2
