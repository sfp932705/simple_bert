from pathlib import Path
from typing import Any

import rust

from token_encoders.rust.base import RustBaseTokenizer
from token_encoders.wp import WordPieceTokenizer


class RustWordPieceTokenizer(RustBaseTokenizer, WordPieceTokenizer):
    @property
    def backend_tokenizer(self) -> Any:
        return rust.token_encoders.RustWordPieceTokenizer  # type: ignore

    @property
    def tokenizer_delimiter(self) -> str:
        return self.delimiter

    @property
    def wordpiece_mode(self) -> bool:
        return True

    def load(self, directory: Path) -> None:
        WordPieceTokenizer.load(self, directory)
        self._backend.set_state(self.vocab, self.merges)
        self._update_special_tokens()
