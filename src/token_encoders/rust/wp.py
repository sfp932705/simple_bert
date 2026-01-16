import rust

from token_encoders.rust.base import RustBaseTokenizer
from token_encoders.wp import WordPieceTokenizer


class RustWordPieceTokenizer(RustBaseTokenizer, WordPieceTokenizer):

    def get_backend(self):
        return rust.token_encoders.RustWordPieceTokenizer(  # type:ignore
            self.settings.vocab_size,
            self.settings.special_tokens,
            self.settings.unused_tokens,
            delimiter=self.delimiter,
        )

    def train(self, corpus: list[str]) -> None:
        self._backend.train(corpus)
        self.vocab = self._backend.get_vocab()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
