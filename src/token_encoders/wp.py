import collections

from token_encoders.base import BaseTokenizer


class WordPieceTokenizer(BaseTokenizer):
    def train(self, corpus: list[str]) -> None:
        counts: dict[str, int] = collections.Counter()
        for sentence in corpus:
            for word in sentence.split():
                counts[word] += 1
        split_counts: dict[tuple[str, ...], int] = {
            tuple([w[0]] + ["##" + c for c in w[1:]]): freq
            for w, freq in counts.items()
        }

        alphabet: set[str] = set()
        for t in split_counts:
            alphabet.update(t)
        self._initialize_vocab(sorted(list(alphabet)))

        while len(self.vocab) < self.settings.vocab_size:
            scores: dict[tuple[str, str], float] = {}
            pair_counts: dict[tuple[str, str], int] = collections.Counter()
            char_counts: dict[str, int] = collections.Counter()

            for word_tuple, freq in split_counts.items():
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i + 1])] += freq
                for char in word_tuple:
                    char_counts[char] += freq

            for pair, freq in pair_counts.items():
                scores[pair] = freq / (char_counts[pair[0]] * char_counts[pair[1]])

            if not scores:
                break
            best_pair: tuple[str, str] = max(scores, key=scores.get)
            new_token: str = best_pair[0] + best_pair[1].replace("##", "")

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = new_token
            new_split_counts = {}
            for word_tuple, freq in split_counts.items():
                new_word = []
                i = 0
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
                new_split_counts[tuple(new_word)] = freq
            split_counts = new_split_counts

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            start = 0
            while start < len(word):
                end = len(word)
                cur_substr = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    ids.append(self.vocab["[UNK]"])
                    break
                ids.append(self.vocab[cur_substr])
                start = end
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse_vocab.get(i, "[UNK]") for i in ids]
        return " ".join(tokens).replace(" ##", "")
