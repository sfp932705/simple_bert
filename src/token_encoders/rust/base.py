from abc import abstractmethod
from typing import Any

from settings import TokenizerSettings
from token_encoders.base import BaseTokenizer


class RustBaseTokenizer(BaseTokenizer):
    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self._backend = self.get_backend()  # type:ignore

    @abstractmethod
    def get_backend(self) -> Any:
        pass

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids)
