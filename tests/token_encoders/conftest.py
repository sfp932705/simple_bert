import pytest

from settings import TokenizerSettings
from token_encoders.bpe import BPETokenizer
from token_encoders.rust.bpe import RustBPETokenizer
from token_encoders.rust.wp import RustWordPieceTokenizer
from token_encoders.wp import WordPieceTokenizer


@pytest.fixture
def sample_corpus() -> list[str]:
    return [
        "the man went to the store",
        "the store was closed",
        "the man bought a gallon of milk",
        "milk is good for bones",
    ]


@pytest.fixture
def word_piece_tokenizer() -> WordPieceTokenizer:
    return WordPieceTokenizer(TokenizerSettings(vocab_size=60, unused_tokens=5))


@pytest.fixture
def trained_word_piece_tokenizer(
    word_piece_tokenizer: WordPieceTokenizer, sample_corpus: list[str]
) -> WordPieceTokenizer:
    word_piece_tokenizer.train(sample_corpus)
    return word_piece_tokenizer


@pytest.fixture
def byte_pair_tokenizer() -> BPETokenizer:
    return BPETokenizer(TokenizerSettings(vocab_size=60, unused_tokens=5))


@pytest.fixture
def trained_byte_pair_tokenizer(
    byte_pair_tokenizer: BPETokenizer, sample_corpus: list[str]
) -> BPETokenizer:
    byte_pair_tokenizer.train(sample_corpus)
    return byte_pair_tokenizer


@pytest.fixture
def rust_bpe_tokenizer() -> RustBPETokenizer:
    return RustBPETokenizer(TokenizerSettings(vocab_size=60, unused_tokens=5))


@pytest.fixture
def trained_rust_bpe_tokenizer(
    rust_bpe_tokenizer: RustBPETokenizer, sample_corpus: list[str]
) -> RustBPETokenizer:
    rust_bpe_tokenizer.train(sample_corpus)
    return rust_bpe_tokenizer


@pytest.fixture
def rust_word_piece_tokenizer() -> RustWordPieceTokenizer:
    return RustWordPieceTokenizer(TokenizerSettings(vocab_size=60, unused_tokens=5))


@pytest.fixture
def trained_rust_word_piece_tokenizer(
    rust_word_piece_tokenizer: RustWordPieceTokenizer, sample_corpus: list[str]
) -> RustWordPieceTokenizer:
    rust_word_piece_tokenizer.train(sample_corpus)
    return rust_word_piece_tokenizer
