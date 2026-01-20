import unicodedata
from collections import Counter
from pathlib import Path

from settings import TokenizerSettings
from token_encoders.bpe import BPETokenizer


class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.token_id: int | None = None
        self.score: float = 0.0
        self.end_word: bool = False


class WordPieceTokenizer(BPETokenizer):
    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self.delimiter = "##"
        self.do_lower_case = True
        self.trie_root = TrieNode()
        self.min_score = -1e10

    def _build_index(self) -> None:
        self.trie_root = TrieNode()
        for token, token_id in self.vocab.items():
            node = self.trie_root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.token_id = token_id
            node.end_word = True

    def _is_punctuation(self, char: str) -> bool:
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True
        cat = unicodedata.category(char)
        return cat.startswith("P") or cat.startswith("S")

    def _is_cjk(self, char: str) -> bool:
        cp = ord(char)
        return (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
        )

    def _prepare_corpus_counts(self, corpus: list[str]) -> Counter[str]:
        word_freqs: Counter[str] = Counter()
        for text in corpus:
            words = self.pre_tokenize(text)
            word_freqs.update(words)
        return word_freqs

    def _token_to_chars(self, word: str) -> list[str]:
        return [
            char if i == 0 else f"{self.delimiter}{char}" for i, char in enumerate(word)
        ]

    def _merge_strings(self, token_a: str, token_b: str) -> str:
        return token_a + token_b.replace(self.delimiter, "", 1)

    def pre_tokenize(self, text: str) -> list[str]:
        if self.do_lower_case:
            text = text.lower()
            text = unicodedata.normalize("NFD", text)
            text = "".join([c for c in text if unicodedata.category(c) != "Mn"])

        tokens = text.split()
        split_tokens = []

        for token in tokens:
            current_word: list = []
            for char in token:
                if self._is_punctuation(char) or self._is_cjk(char):
                    if current_word:
                        split_tokens.append("".join(current_word))
                        current_word = []
                    split_tokens.append(char)
                else:
                    current_word.append(char)
            if current_word:
                split_tokens.append("".join(current_word))
        return split_tokens

    def train(self, corpus: list[str]) -> None:
        super().train(corpus)
        self._build_index()

    def load(self, directory: Path) -> None:
        super().load(directory)
        self._build_index()

    def encode(self, text: str) -> list[int]:
        if not self.trie_root.children:
            self._build_index()

        words = self.pre_tokenize(text)
        output_ids: list[int] = []
        unk_id = self.vocab.get(self.unk_token)
        assert unk_id is not None
        continuing_root = self.trie_root
        if self.delimiter[0] in continuing_root.children:
            for char in self.delimiter:
                if char in continuing_root.children:
                    continuing_root = continuing_root.children[char]
                else:
                    continuing_root = None  # type:ignore
                    break

        for word in words:
            chars = list(word)
            i = 0
            n = len(chars)
            is_bad = False
            word_ids = []

            while i < n:
                if i == 0:
                    node = self.trie_root
                else:
                    node = continuing_root
                    if node is None:
                        is_bad = True
                        break

                j = i
                last_token_id = None
                last_match_end = -1

                while j < n:
                    char = chars[j]
                    if char in node.children:
                        node = node.children[char]
                        if node.token_id is not None:
                            last_token_id = node.token_id
                            last_match_end = j
                        j += 1
                    else:
                        break

                if last_token_id is not None:
                    word_ids.append(last_token_id)
                    i = last_match_end + 1
                else:
                    is_bad = True
                    break

            if is_bad:
                output_ids.append(unk_id)
            else:
                output_ids.extend(word_ids)
        return output_ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse_vocab.get(i, self.unk_token) for i in ids]
        return " ".join(tokens).replace(" " + self.delimiter, "")
