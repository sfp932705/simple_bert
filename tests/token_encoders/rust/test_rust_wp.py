from token_encoders.rust.wp import RustWordPieceTokenizer


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


