from token_encoders.wp import WordPieceTokenizer


def test_word_piece_initialization(word_piece_tokenizer: WordPieceTokenizer):
    word_piece_tokenizer._initialize_vocab([""])
    assert "[CLS]" in word_piece_tokenizer.vocab
    assert word_piece_tokenizer.vocab["[PAD]"] == 0


def test_word_piece_training(
    word_piece_tokenizer: WordPieceTokenizer, sample_corpus: list[str]
):
    word_piece_tokenizer.train(sample_corpus)
    assert len(word_piece_tokenizer.vocab) <= word_piece_tokenizer.settings.vocab_size
    assert "the" in word_piece_tokenizer.vocab or "##e" in word_piece_tokenizer.vocab
    assert "men" not in word_piece_tokenizer.vocab


def test_word_piece_encoding_decoding(trained_word_piece_tokenizer: WordPieceTokenizer):
    original_text = "the milk store"
    encoded = trained_word_piece_tokenizer.encode(original_text)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    decoded = trained_word_piece_tokenizer.decode(encoded)
    assert original_text == decoded


def test_unknown_token(trained_word_piece_tokenizer: WordPieceTokenizer):
    encoded = trained_word_piece_tokenizer.encode("xyz")
    assert trained_word_piece_tokenizer.vocab["[UNK]"] in encoded
