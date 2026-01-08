from abc import ABC, abstractmethod
from pathlib import Path

from settings import TokenizerSettings


class BaseTokenizer(ABC):
    def __init__(self, settings: TokenizerSettings):
        self.settings = settings
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def _initialize_vocab(self, base_alphabet: list[str]) -> None:
        for idx, token in enumerate(self.settings.special_tokens):
            self.vocab[token] = idx
        for i in range(self.settings.unused_tokens):
            self.vocab[f"[unused{i}]"] = len(self.vocab)
        for char in base_alphabet:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sorted_tokens = sorted(self.vocab.items(), key=lambda item: item[1])
        vocab_content = "\n".join([token for token, _ in sorted_tokens])
        (directory / "vocab.txt").write_text(vocab_content, encoding="utf-8")

    def load(self, directory: Path) -> None:
        vocab_path = directory / "vocab.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"No vocab.txt found in {directory}")
        lines = vocab_path.read_text(encoding="utf-8").splitlines()
        self.vocab = {token: i for i, token in enumerate(lines)}
        self.inverse_vocab = {i: token for i, token in enumerate(lines)}

    @abstractmethod
    def train(self, corpus: list[str]) -> None:
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        pass
