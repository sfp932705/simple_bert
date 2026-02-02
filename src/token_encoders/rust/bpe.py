from pathlib import Path
from typing import Any

import rust

from token_encoders.bpe import BPETokenizer
from token_encoders.rust.base import RustBaseTokenizer


class RustBPETokenizer(RustBaseTokenizer, BPETokenizer):
    @property
    def backend_tokenizer(self) -> Any:
        return rust.token_encoders.RustBPETokenizer  # type:ignore

    @property
    def tokenizer_delimiter(self) -> str:
        return self.delimiter

    @property
    def wordpiece_mode(self) -> bool:
        return False

    def train(self, corpus: list[str]) -> None:
        super().train(corpus)
        self.merges = self._backend.get_merges()

    def load(self, directory: Path) -> None:
        BPETokenizer.load(self, directory)
        self._backend.set_state(self.vocab, self.merges)
        self._update_special_tokens()
