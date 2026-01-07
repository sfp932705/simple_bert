import collections

from settings import TokenizerSettings
from token_encoders.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):

    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self.merges: list[tuple[str, str]] = []
        self.delimiter = "Ġ"

    def train(self, corpus: list[str]) -> None:
        processed_corpus = [s.replace(" ", self.delimiter) for s in corpus]
        counts = collections.Counter(tuple(s) for s in processed_corpus)
        alphabet = {char for word_tuple in counts for char in word_tuple}
        self._initialize_vocab(sorted(list(alphabet)))

        while len(self.vocab) < self.settings.vocab_size:
            pair_counts: dict[tuple[str, str], int] = collections.Counter()
            for word_tuple, freq in counts.items():
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i + 1])] += freq
            if not pair_counts:
                break

            best_pair: tuple[str, str] = max(pair_counts, key=pair_counts.get)
            new_token: str = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = new_token
            new_counts = {}
            for word_tuple, freq in counts.items():
                new_word, i = [], 0
                while i < len(word_tuple):
                    if (
                        i < len(word_tuple) - 1
                        and (word_tuple[i], word_tuple[i + 1]) == best_pair
                    ):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_counts[tuple(new_word)] = freq
            counts = new_counts

    def encode(self, text: str) -> list[int]:
        words = text.split(" ")
        encoded_ids = []
        for i, word in enumerate(words):
            word_str = self.delimiter + word if i > 0 else word
            symbols = list(word_str)
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
                [self.vocab.get(s, self.vocab["[UNK]"]) for s in symbols]
            )
        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse_vocab.get(i, "[UNK]") for i in ids]
        return "".join(tokens).replace(self.delimiter, " ").strip()
