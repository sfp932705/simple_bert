import pytest

from settings import LoaderSettings, TokenizerSettings
from token_encoders.bpe import BPETokenizer


@pytest.fixture
def settings() -> LoaderSettings:
    return LoaderSettings(
        max_seq_len=12,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


@pytest.fixture
def tokenizer() -> BPETokenizer:
    settings = TokenizerSettings(vocab_size=60, unused_tokens=5)
    tokenizer = BPETokenizer(settings)
    tokenizer.vocab = {
        settings.pad_token: 0,
        settings.cls_token: 1,
        settings.sep_token: 2,
        settings.mask_token: 3,
        settings.unk_token: 4,
    }

    tokenizer._update_id_cache()

    def simple_encode(text: str):
        if not text:
            return []
        return [hash(w) % (settings.vocab_size - 5) + 5 for w in text.split()]

    tokenizer.encode = simple_encode  # type: ignore
    return tokenizer
