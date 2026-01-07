import pytest

from settings import TokenizerSettings
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
    return WordPieceTokenizer(TokenizerSettings(vocab_size=60))

@pytest.fixture
def trained_word_piece_tokenizer(
    word_piece_tokenizer: WordPieceTokenizer, sample_corpus: list[str]
):
    word_piece_tokenizer.train(sample_corpus)
    return word_piece_tokenizer
