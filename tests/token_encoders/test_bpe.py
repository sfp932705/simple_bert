from pathlib import Path

import pytest

from token_encoders.bpe import BPETokenizer


def test_bpe_initialization(byte_pair_tokenizer: BPETokenizer):
    byte_pair_tokenizer._initialize_vocab([""])
    assert "[CLS]" in byte_pair_tokenizer.vocab
    assert byte_pair_tokenizer.vocab["[PAD]"] == 0


def test_bpe_training(byte_pair_tokenizer: BPETokenizer, sample_corpus: list[str]):
    byte_pair_tokenizer.train(sample_corpus)
    assert len(byte_pair_tokenizer.vocab) <= byte_pair_tokenizer.settings.vocab_size
    assert "the" in byte_pair_tokenizer.vocab or "t" in byte_pair_tokenizer.vocab
    assert "men" not in byte_pair_tokenizer.vocab


def test_bpe_encoding_decoding(trained_byte_pair_tokenizer: BPETokenizer):
    original_text = "the milk store"
    encoded = trained_byte_pair_tokenizer.encode(original_text)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    decoded = trained_byte_pair_tokenizer.decode(encoded)
    assert original_text == decoded


def test_bpe_unknown_token(trained_byte_pair_tokenizer: BPETokenizer):
    encoded = trained_byte_pair_tokenizer.encode("§")
    assert trained_byte_pair_tokenizer.vocab["[UNK]"] in encoded


def test_bpe_save_load_consistency(
    trained_byte_pair_tokenizer: BPETokenizer, tmp_path: Path
):
    save_dir = tmp_path / "bpe_model"
    trained_byte_pair_tokenizer.save(save_dir)
    assert (save_dir / "vocab.txt").exists()
    assert (save_dir / "merges.txt").exists()
    new_tokenizer = BPETokenizer(trained_byte_pair_tokenizer.settings)
    new_tokenizer.load(save_dir)
    assert sorted(new_tokenizer.vocab) == sorted(trained_byte_pair_tokenizer.vocab)
    assert new_tokenizer.merges == trained_byte_pair_tokenizer.merges
    test_text = "the man bought milk"
    assert new_tokenizer.encode(test_text) == trained_byte_pair_tokenizer.encode(
        test_text
    )


def test_load_fails(trained_byte_pair_tokenizer: BPETokenizer, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        trained_byte_pair_tokenizer.load(tmp_path)
    trained_byte_pair_tokenizer.save(tmp_path)
    (tmp_path / "merges.txt").unlink()
    with pytest.raises(FileNotFoundError):
        trained_byte_pair_tokenizer.load(tmp_path)


def test_bpe_vocab_sorting_on_disk(
    trained_byte_pair_tokenizer: BPETokenizer, tmp_path: Path
):
    save_dir = tmp_path / "bpe_check"
    trained_byte_pair_tokenizer.save(save_dir)
    vocab_lines = (save_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    for line_idx, token in enumerate(vocab_lines):
        assert trained_byte_pair_tokenizer.vocab[token] == line_idx


def test_bpe_merges_format(trained_byte_pair_tokenizer: BPETokenizer, tmp_path):
    save_dir = tmp_path / "bpe_merges"
    trained_byte_pair_tokenizer.save(save_dir)
    merges_lines = (save_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    for i, merge in enumerate(trained_byte_pair_tokenizer.merges):
        assert merges_lines[i] == f"{merge[0]} {merge[1]}"
