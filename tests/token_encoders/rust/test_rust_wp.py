from pathlib import Path

import pytest

from token_encoders.rust.wp import RustWordPieceTokenizer


@pytest.fixture
def loaded_tokenizer(
    trained_rust_word_piece_tokenizer: RustWordPieceTokenizer, tmp_path: Path
) -> RustWordPieceTokenizer:
    save_dir = tmp_path / "rust_wp_model"
    trained_rust_word_piece_tokenizer.save(save_dir)
    tokenizer = RustWordPieceTokenizer(trained_rust_word_piece_tokenizer.settings)
    tokenizer.load(save_dir)
    return tokenizer


def test_rust_wp_training(trained_rust_word_piece_tokenizer: RustWordPieceTokenizer):
    assert (
        len(trained_rust_word_piece_tokenizer.vocab)
        <= trained_rust_word_piece_tokenizer.settings.vocab_size
    )
    assert (
        "the" in trained_rust_word_piece_tokenizer.vocab
        or "##e" in trained_rust_word_piece_tokenizer.vocab
    )


def test_rust_wp_encoding_decoding(
    trained_rust_word_piece_tokenizer: RustWordPieceTokenizer,
):
    original_text = "the milk store"
    encoded = trained_rust_word_piece_tokenizer.encode(original_text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    decoded = trained_rust_word_piece_tokenizer.decode(encoded)
    assert "milk" in decoded


def test_rust_wp_save(
    trained_rust_word_piece_tokenizer: RustWordPieceTokenizer, tmp_path: Path
):
    save_dir = tmp_path / "rust_wp_model"
    trained_rust_word_piece_tokenizer.save(save_dir)
    assert (save_dir / "vocab.txt").exists()
    assert (save_dir / "merges.txt").exists()


@pytest.mark.parametrize(
    "property_name",
    ["pad_token_id", "mask_token_id", "cls_token_id", "sep_token_id", "unk_token_id"],
)
def test_rust_wp_load_special_tokens(
    loaded_tokenizer: RustWordPieceTokenizer,
    property_name: str,
):
    assert getattr(loaded_tokenizer, property_name) != -1
