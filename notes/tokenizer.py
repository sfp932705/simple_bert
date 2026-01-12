from collections import Counter, defaultdict


def bpe_verbose_train(self, corpus: list[str]) -> None:
    print("\n" + "=" * 60)
    print("STARTING BPE TRAINING ALGORITHM")
    print("=" * 60)

    print("\n[Phase 1] Pre-tokenization & Word Counting")
    word_ctr: Counter[tuple[str, ...]] = Counter()
    for text in corpus:
        if not text:
            continue
        words = text.split()
        if not words:
            continue

        tokenized_word_0 = tuple(words[0])
        word_ctr[tokenized_word_0] += 1
        print(f"  Processing raw text: '{text}'")
        print(f"    -> Parsed word: '{words[0]}' -> Tokens: {tokenized_word_0}")

        for w in words[1:]:
            tokenized_word_rest = tuple(self.delimiter + w)
            word_ctr[tokenized_word_rest] += 1
            print(
                f"    -> Parsed word: '{self.delimiter + w}' -> Tokens: {tokenized_word_rest}"
            )

    print(f"\n[Phase 1 Summary] Unique word structures found: {len(word_ctr)}")
    for w, c in word_ctr.items():
        print(f"  - {w}: {c}")

    self.vocab_list = [list(w) for w in word_ctr.keys()]
    self.vocab_counts = list(word_ctr.values())

    print("\n[Phase 2] Building Initial Inverted Index & Stats")
    stats: Counter = Counter()
    inverted_index = defaultdict(set)

    for idx, word in enumerate(self.vocab_list):
        freq = self.vocab_counts[idx]
        print(f"  Scanning word index {idx}: {word} (Freq: {freq})")
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq
            inverted_index[pair].add(idx)
            print(
                f"    -> Found pair {pair}. Count = {stats[pair]}. Added index {idx}."
            )

    print("\n[Phase 2 Summary] Initial Pair Statistics:")
    for p, c in stats.items():
        print(f"  {p}: {c}")

    alphabet = {char for word in self.vocab_list for char in word}
    self._initialize_vocab(sorted(list(alphabet)))
    print(
        f"\n[Init] Base Vocab initialized with characters. Current Size: {len(self.vocab)}"
    )

    print("\n" + "=" * 60)
    print("ENTERING MAIN MERGE LOOP")
    print("=" * 60)

    step_counter = 0

    while len(self.vocab) < self.settings.vocab_size:
        step_counter += 1

        is_start = step_counter <= 3
        is_periodic = step_counter % 10 == 0
        is_last = len(self.vocab) + 1 >= self.settings.vocab_size
        verbose_step = is_start or is_periodic or is_last

        if verbose_step:
            print(
                f"\n--- [Step {step_counter}] Vocab Size: {len(self.vocab)} / Target: {self.settings.vocab_size} ---"
            )

        if not stats:
            print("  No more pairs to merge. Stopping early.")
            break

        best_pair = max(stats, key=stats.get)  # type: ignore
        count = stats[best_pair]

        if verbose_step:
            print(f"  Best candidate pair: {best_pair} (Count: {count})")

        if stats[best_pair] < 1:
            print("  Best pair frequency < 1. Stopping.")
            break

        token_a, token_b = best_pair
        new_token = token_a + token_b
        self.merges.append(best_pair)

        if verbose_step:
            print(f"  Merging {best_pair} -> Created New Token: '{new_token}'")

        if new_token not in self.vocab:
            self.vocab[new_token] = len(self.vocab)
            self.inverse_vocab[len(self.vocab) - 1] = new_token
            if verbose_step:
                print(f"  Added '{new_token}' to vocabulary at ID {len(self.vocab)-1}")

        indices_to_update = list(inverted_index[best_pair])

        if verbose_step:
            print(f"  Updating occurrences in {len(indices_to_update)} unique words...")

        for idx in indices_to_update:
            word = self.vocab_list[idx]
            freq = self.vocab_counts[idx]

            if verbose_step:
                print(f"    Processing Word [{idx}]: {word} (Freq: {freq})")

            new_word: list = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == token_a and word[i + 1] == token_b:
                    if verbose_step:
                        print(f"      -> Match found at pos {i}: {best_pair}")

                    merge_right_side = (
                        i + 3 < len(word)
                        and word[i + 2] == token_a
                        and word[i + 3] == token_b
                    )

                    if i > 0:
                        prev_token = new_word[-1]
                        prev_pair = (prev_token, token_a)
                        stats[prev_pair] -= freq

                        new_prev_pair = (prev_token, new_token)
                        stats[new_prev_pair] += freq
                        inverted_index[new_prev_pair].add(idx)

                        if verbose_step:
                            print(
                                f"        [Stat Adjust] Left: Removed {prev_pair}, Added {new_prev_pair}"
                            )

                    if not merge_right_side and i < len(word) - 2:
                        next_token = word[i + 2]
                        next_pair = (token_b, next_token)
                        stats[next_pair] -= freq

                        new_next_pair = (new_token, next_token)
                        stats[new_next_pair] += freq
                        inverted_index[new_next_pair].add(idx)

                        if verbose_step:
                            print(
                                f"        [Stat Adjust] Right: Removed {next_pair}, Added {new_next_pair}"
                            )

                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            if verbose_step:
                print(f"    Resulting Word: {new_word}")
            self.vocab_list[idx] = new_word

        del stats[best_pair]
        del inverted_index[best_pair]
        if verbose_step:
            print(f"  Cleaned up old stats for {best_pair}")

    print("\n" + "=" * 60)
    print(f"Training complete. Final Vocab size: {len(self.vocab)}")
    print("=" * 60)
