from pathlib import Path

from token_encoders.rust.bpe import RustBPETokenizer


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


def test_rust_bpe_save_load(
    trained_rust_bpe_tokenizer: RustBPETokenizer, tmp_path: Path
):
    save_dir = tmp_path / "rust_bpe_model"
    trained_rust_bpe_tokenizer.save(save_dir)

    assert (save_dir / "vocab.txt").exists()
    assert (save_dir / "merges.txt").exists()

    new_tokenizer = RustBPETokenizer(trained_rust_bpe_tokenizer.settings)
    new_tokenizer.load(save_dir)
    assert sorted(new_tokenizer.vocab) == sorted(trained_rust_bpe_tokenizer.vocab)
    assert new_tokenizer.merges == trained_rust_bpe_tokenizer.merges
    test_text = "milk is good"
    assert new_tokenizer.encode(test_text) == trained_rust_bpe_tokenizer.encode(
        test_text
    )
