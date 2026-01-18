from abc import abstractmethod
from typing import Any

from settings import TokenizerSettings
from token_encoders.base import BaseTokenizer


class RustBaseTokenizer(BaseTokenizer):
    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self._backend = self.get_backend()  # type:ignore

    @property
    @abstractmethod
    def backend_tokenizer(self) -> Any:
        pass

    @property
    @abstractmethod
    def tokenizer_delimiter(self) -> str:
        pass

    @property
    @abstractmethod
    def wordpiece_mode(self) -> bool:
        pass

    def get_backend(self) -> Any:
        return self.backend_tokenizer(
            self.settings.vocab_size,
            self.settings.special_tokens,
            self.settings.unused_tokens,
            delimiter=self.tokenizer_delimiter,
            wordpiece_mode=self.wordpiece_mode,
        )

    def train(self, corpus: list[str]) -> None:
        self._backend.train("".join(corpus))
        self.vocab = self._backend.get_vocab()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids)
