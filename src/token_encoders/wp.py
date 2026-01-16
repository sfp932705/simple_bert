import unicodedata
from collections import Counter, defaultdict

from settings import TokenizerSettings
from token_encoders.base import BaseTokenizer


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, settings: TokenizerSettings):
        super().__init__(settings)
        self.delimiter = "##"
        self.unk_token = "[UNK]"
        self.do_lower_case = True

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

    def pre_tokenize(self, text: str) -> list[str]:
        if self.do_lower_case:
            text = text.lower()
            text = unicodedata.normalize("NFD", text)
            text = "".join([c for c in text if unicodedata.category(c) != "Mn"])

        tokens = text.split()
        split_tokens = []

        for token in tokens:
            current_word = []
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
        print(f"Step 1: Pre-tokenizing (Lower case: {self.do_lower_case})...")
        word_freqs = Counter()
        for text in corpus:
            words = self.pre_tokenize(text)
            word_freqs.update(words)

        unique_words = list(word_freqs.keys())
        word_splits = []

        char_counts = Counter()
        inner_chars = set()

        for word, freq in word_freqs.items():
            for i, char in enumerate(word):
                char_counts[char] += freq
                if i > 0:
                    inner_chars.add(char)
        all_chars = set(char_counts.keys())

        initial_alphabet = []
        for char in sorted(list(all_chars)):
            initial_alphabet.append(char)
            if char in inner_chars:
                initial_alphabet.append(f"{self.delimiter}{char}")

        self._initialize_vocab(initial_alphabet)
        next_id = len(self.vocab)
        pair_counts = defaultdict(int)
        where_pair = defaultdict(set)

        for idx, word in enumerate(unique_words):
            freq = word_freqs[word]
            tokens = [
                char if i == 0 else f"{self.delimiter}{char}"
                for i, char in enumerate(word)
            ]
            word_splits.append(tokens)

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
                where_pair[pair].add(idx)

        while len(self.vocab) < self.settings.vocab_size:
            best_pair = None
            max_score = -1
            current_pairs = list(pair_counts.keys())
            for pair in current_pairs:
                freq = pair_counts[pair]

                if freq > max_score:
                    max_score = freq
                    best_pair = pair
                elif freq == max_score:
                    if best_pair is None or pair < best_pair:
                        best_pair = pair

            if best_pair is None:
                break

            p1, p2 = best_pair
            new_token = p1 + p2.replace(self.delimiter, "", 1)

            self.vocab[new_token] = next_id
            self.inverse_vocab[next_id] = new_token
            next_id += 1

            indices = where_pair[best_pair]
            for word_idx in indices:
                tokens = word_splits[word_idx]
                if len(tokens) < 2:
                    continue

                freq = word_freqs[unique_words[word_idx]]
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] -= freq

                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == p1 and tokens[i + 1] == p2:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_splits[word_idx] = new_tokens

                for i in range(len(new_tokens) - 1):
                    pair = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[pair] += freq
                    where_pair[pair].add(word_idx)

            del pair_counts[best_pair]
            del where_pair[best_pair]

    def encode(self, text: str) -> list[int]:
        words = self.pre_tokenize(text)
        output_ids = []
        max_len = 100

        for word in words:
            chars = list(word)
            start = 0
            sub_tokens = []
            is_bad = False

            while start < len(chars):
                end = min(len(chars), start + max_len)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = self.delimiter + substr

                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_ids.append(self.vocab[self.unk_token])
            else:
                output_ids.extend([self.vocab[t] for t in sub_tokens])
        return output_ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.inverse_vocab.get(i, self.unk_token) for i in ids]
        return " ".join(tokens).replace(" " + self.delimiter, "")
