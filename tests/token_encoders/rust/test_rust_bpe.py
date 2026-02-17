from pathlib import Path

import pytest

from token_encoders.rust.bpe import RustBPETokenizer


@pytest.fixture
def loaded_tokenizer(
    trained_rust_bpe_tokenizer: RustBPETokenizer, tmp_path: Path
) -> RustBPETokenizer:
    save_dir = tmp_path / "rust_bpe_model"
    trained_rust_bpe_tokenizer.save(save_dir)
    loaded_tokenizer = RustBPETokenizer(trained_rust_bpe_tokenizer.settings)
    loaded_tokenizer.load(save_dir)
    return loaded_tokenizer


def test_rust_bpe_training(trained_rust_bpe_tokenizer: RustBPETokenizer):
    assert (
        len(trained_rust_bpe_tokenizer.vocab)
        <= trained_rust_bpe_tokenizer.settings.vocab_size
    )
    assert (
        "the" in trained_rust_bpe_tokenizer.vocab
        or "t" in trained_rust_bpe_tokenizer.vocab
    )


def test_rust_bpe_encoding_decoding(trained_rust_bpe_tokenizer: RustBPETokenizer):
    original_text = "the milk store"
    encoded = trained_rust_bpe_tokenizer.encode(original_text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    decoded = trained_rust_bpe_tokenizer.decode(encoded)
    assert "milk" in decoded


def test_rust_bpe_save(trained_rust_bpe_tokenizer: RustBPETokenizer, tmp_path: Path):
    save_dir = tmp_path / "rust_bpe_model"
    trained_rust_bpe_tokenizer.save(save_dir)
    assert (save_dir / "vocab.txt").exists()
    assert (save_dir / "merges.txt").exists()


def test_rust_bpe_load(
    trained_rust_bpe_tokenizer: RustBPETokenizer,
    loaded_tokenizer: RustBPETokenizer,
):
    assert sorted(loaded_tokenizer.vocab) == sorted(trained_rust_bpe_tokenizer.vocab)
    assert loaded_tokenizer.merges == trained_rust_bpe_tokenizer.merges
    test_text = "milk is good"
    assert loaded_tokenizer.encode(test_text) == trained_rust_bpe_tokenizer.encode(
        test_text
    )


@pytest.mark.parametrize(
    "property_name",
    ["pad_token_id", "mask_token_id", "cls_token_id", "sep_token_id", "unk_token_id"],
)
def test_rust_bpe_load_special_tokens(
    loaded_tokenizer: RustBPETokenizer,
    property_name: str,
):
    assert getattr(loaded_tokenizer, property_name) != -1
