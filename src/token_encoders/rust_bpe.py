import rust

from settings import TokenizerSettings
from token_encoders.bpe import BPETokenizer


class RustBPETokenizer(BPETokenizer):
    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self._backend = rust.token_encoders.RustBPETokenizer(  # type:ignore
            settings.vocab_size,
            settings.special_tokens,
            settings.unused_tokens,
            delimiter=self.delimiter,
        )

    def train(self, corpus: list[str]) -> None:
        self._backend.train(corpus)
        self.vocab = self._backend.get_vocab()
        self.merges = self._backend.get_merges()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids)
