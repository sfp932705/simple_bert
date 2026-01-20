import re
from collections import Counter, defaultdict
from pathlib import Path

from settings import TokenizerSettings
from token_encoders.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self.merges: list[tuple[str, str]] = []
        self.vocab_list: list[list[str]] = []
        self.vocab_counts: list[int] = []
        self.delimiter = "Ġ"
        self.unk_token = "[UNK]"
        self.split_pattern = re.compile(r"[^\s]+|\n")

    def _prepare_corpus_counts(self, corpus: list[str]) -> Counter[str]:
        word_ctr: Counter[str] = Counter()
        for text in corpus:
            if not text:
                continue
            words = self.split_pattern.findall(text)
            if not words:
                continue
            word_ctr[words[0]] += 1
            for w in words[1:]:
                word_ctr[self.delimiter + w] += 1
        return word_ctr

    def _token_to_chars(self, word: str) -> list[str]:
        return list(word)

    def _merge_strings(self, token_a: str, token_b: str) -> str:
        return token_a + token_b

    def train(self, corpus: list[str]) -> None:
        word_counts = self._prepare_corpus_counts(corpus)
        self.vocab_list = []
        self.vocab_counts = []
        for word, freq in word_counts.items():
            self.vocab_list.append(self._token_to_chars(word))
            self.vocab_counts.append(freq)

        alphabet = {char for word in self.vocab_list for char in word}
        self._initialize_vocab(sorted(list(alphabet)))
        self._run_merge_loop()

    def _run_merge_loop(self) -> None:
        stats: Counter = Counter()
        inverted_index = defaultdict(set)
        for idx, word in enumerate(self.vocab_list):
            freq = self.vocab_counts[idx]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                stats[pair] += freq
                inverted_index[pair].add(idx)

        while len(self.vocab) < self.settings.vocab_size:
            if not stats:
                break

            best_pair = max(stats, key=stats.get)  # type: ignore
            if stats[best_pair] < 1:
                break

            token_a, token_b = best_pair
            new_token = self._merge_strings(token_a, token_b)
            self.merges.append(best_pair)

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = new_token
            indices_to_update = list(inverted_index[best_pair])

            for idx in indices_to_update:
                word = self.vocab_list[idx]
                freq = self.vocab_counts[idx]
                new_word: list = []
                i = 0
                while i < len(word):
                    if (
                        i < len(word) - 1
                        and word[i] == token_a
                        and word[i + 1] == token_b
                    ):
                        if i > 0:
                            prev_token = new_word[-1]
                            stats[(prev_token, token_a)] -= freq
                            inverted_index[(prev_token, token_a)].discard(idx)

                            new_prev_pair = (prev_token, new_token)
                            stats[new_prev_pair] += freq
                            inverted_index[new_prev_pair].add(idx)

                        if i < len(word) - 2:
                            next_token = word[i + 2]
                            stats[(token_b, next_token)] -= freq
                            inverted_index[(token_b, next_token)].discard(idx)

                            new_next_pair = (new_token, next_token)
                            stats[new_next_pair] += freq
                            inverted_index[new_next_pair].add(idx)

                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                self.vocab_list[idx] = new_word

            del stats[best_pair]
            del inverted_index[best_pair]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        words = self.split_pattern.findall(text)
        if not words:
            return []

        encoded_ids = []
        word_list = [list(words[0])] + [list(self.delimiter + w) for w in words[1:]]
        for symbols in word_list:
            for pair in self.merges:
                if pair[0] not in symbols:
                    continue
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and (symbols[j], symbols[j + 1]) == pair:
                        new_symbols.append(pair[0] + pair[1])
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                symbols = new_symbols
            encoded_ids.extend(
                [self.vocab.get(s, self.vocab[self.unk_token]) for s in symbols]
            )
        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse_vocab.get(i, self.unk_token) for i in ids]
        return "".join(tokens).replace(self.delimiter, " ").strip()

    def save(self, directory: Path) -> None:
        super().save(directory)
        merge_content = "\n".join([f"{p[0]} {p[1]}" for p in self.merges])
        (directory / "merges.txt").write_text(merge_content, encoding="utf-8")

    def load(self, directory: Path) -> None:
        super().load(directory)
        merge_path = directory / "merges.txt"
        if not merge_path.exists():
            raise FileNotFoundError(f"No merges.txt found in {directory}")
        self.merges = []
        lines = merge_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2:
                self.merges.append((parts[0], parts[1]))
