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
