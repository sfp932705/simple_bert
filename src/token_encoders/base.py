from abc import ABC, abstractmethod

from settings import TokenizerSettings


class BaseTokenizer(ABC):
    def __init__(self, settings: TokenizerSettings):
        self.settings = settings
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def _initialize_vocab(self, base_alphabet: list[str]) -> None:
        for idx, token in enumerate(self.settings.special_tokens):
            self.vocab[token] = idx
        for char in base_alphabet:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    @abstractmethod
    def train(self, corpus: list[str]) -> None:
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, text: str) -> list[int]:
        pass
