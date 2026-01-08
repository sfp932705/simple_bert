use crate::token_encoders::base::BaseTokenizer;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct UnsafeSlice<T> {
    ptr: *mut T,
}
impl<T> Copy for UnsafeSlice<T> {}
impl<T> Clone for UnsafeSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> Send for UnsafeSlice<T> {}
unsafe impl<T> Sync for UnsafeSlice<T> {}
impl<T> UnsafeSlice<T> {
    fn new(vec: &mut Vec<T>) -> Self {
        Self {
            ptr: vec.as_mut_ptr(),
        }
    }
    unsafe fn get_mut(&self, index: usize) -> &mut T {
        &mut *self.ptr.add(index)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct MergePair {
    count: u32,
    pair: (u32, u32),
}
impl Ord for MergePair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for MergePair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[pyclass]
pub struct RustBPETokenizer {
    base: BaseTokenizer,
    merges: Vec<(String, String)>,
    delimiter: String,
}

impl RustBPETokenizer {
    fn merge_in_word(
        tokens: &mut Vec<u32>,
        freq: i32,
        left_id: u32,
        right_id: u32,
        new_id: u32,
        changes: &mut FxHashMap<(u32, u32), i32>,
    ) -> bool {
        let mut i = 0;
        let mut changed = false;
        while i < tokens.len() - 1 {
            if tokens[i] == left_id && tokens[i + 1] == right_id {
                if i > 0 {
                    *changes.entry((tokens[i - 1], left_id)).or_insert(0) -= freq;
                }
                if i + 1 < tokens.len() - 1 {
                    *changes.entry((right_id, tokens[i + 2])).or_insert(0) -= freq;
                }

                tokens[i] = new_id;
                tokens.remove(i + 1);
                changed = true;

                if i > 0 {
                    *changes.entry((tokens[i - 1], new_id)).or_insert(0) += freq;
                }
                if i < tokens.len() - 1 {
                    *changes.entry((new_id, tokens[i + 1])).or_insert(0) += freq;
                }
            } else {
                i += 1;
            }
        }
        changed
    }
}

#[pymethods]
impl RustBPETokenizer {
    #[new]
    pub fn new(
        vocab_size: usize,
        special_tokens: Vec<String>,
        unused_tokens: usize,
        delimiter: String,
    ) -> Self {
        RustBPETokenizer {
            base: BaseTokenizer::new(vocab_size, special_tokens, unused_tokens),
            merges: Vec::new(),
            delimiter,
        }
    }

    pub fn get_vocab(&self) -> FxHashMap<String, u32> {
        self.base
            .vocab
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    pub fn get_merges(&self) -> Vec<(String, String)> {
        self.merges.clone()
    }

    pub fn train(&mut self, corpus: Vec<String>) {
        if corpus.is_empty() {
            return;
        }
        let text_data = &corpus[0];

        let intermediate_counts = text_data
            .par_split_whitespace()
            .fold(FxHashMap::default, |mut map, w| {
                *map.entry(w).or_insert(0) += 1;
                map
            })
            .reduce(FxHashMap::default, |mut m1, m2| {
                for (k, v) in m2 {
                    *m1.entry(k).or_insert(0) += v;
                }
                m1
            });

        let string_counts: FxHashMap<String, u32> = intermediate_counts
            .into_iter()
            .map(|(word, count)| {
                let s = format!("{}{}", self.delimiter, word);
                (s, count)
            })
            .collect();

        let mut alphabet: FxHashSet<char> = FxHashSet::default();
        for key in string_counts.keys() {
            for c in key.chars() {
                alphabet.insert(c);
            }
        }
        let mut sorted_alphabet: Vec<char> = alphabet.into_iter().collect();
        sorted_alphabet.sort();
        self.base.initialize_vocab(sorted_alphabet);

        let mut word_list: Vec<(Vec<u32>, u32)> = Vec::with_capacity(string_counts.len());
        for (s, freq) in string_counts {
            let ids: Vec<u32> = s
                .chars()
                .map(|c| self.base.get_id(&c.to_string()))
                .collect();
            word_list.push((ids, freq));
        }

        let mut token_to_words: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        let mut pair_counts: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut heap: BinaryHeap<MergePair> = BinaryHeap::new();

        for (word_idx, (tokens, freq)) in word_list.iter().enumerate() {
            for i in 0..tokens.len() {
                let t = tokens[i];
                let entry = token_to_words.entry(t).or_default();

                if entry.last() != Some(&word_idx) {
                    entry.push(word_idx);
                }

                if i < tokens.len() - 1 {
                    let pair = (tokens[i], tokens[i + 1]);
                    *pair_counts.entry(pair).or_insert(0) += *freq;
                }
            }
        }

        for (&pair, &count) in &pair_counts {
            heap.push(MergePair { count, pair });
        }

        let word_list_ptr = UnsafeSlice::new(&mut word_list);
        let parallel_threshold = 8192;

        while self.base.vocab.len() < self.base.vocab_size {
            let mut best_pair = (0, 0);
            let mut found = false;

            while let Some(MergePair { count, pair }) = heap.pop() {
                if let Some(&real_count) = pair_counts.get(&pair) {
                    if real_count == count {
                        best_pair = pair;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                break;
            }

            let (left_id, right_id) = best_pair;

            let part_a = self.base.inverse_vocab.get(&left_id).unwrap().clone();
            let part_b = self.base.inverse_vocab.get(&right_id).unwrap().clone();
            let new_token_str = format!("{}{}", part_a, part_b);

            self.merges.push((part_a, part_b));
            self.base.add_token(new_token_str);
            let new_id = (self.base.vocab.len() - 1) as u32;

            let words_to_check: &Vec<usize> = if let Some(idxs) = token_to_words.get(&left_id) {
                idxs
            } else {
                &Vec::new()
            };

            let (batch_changes, words_with_new_token) = if words_to_check.len() > parallel_threshold
            {
                words_to_check
                    .par_iter()
                    .fold(
                        || (FxHashMap::default(), Vec::new()),
                        move |mut acc, &word_idx| {
                            let (local_changes, local_new_idxs) = &mut acc;
                            let (tokens, freq) = unsafe { word_list_ptr.get_mut(word_idx) };

                            if Self::merge_in_word(
                                tokens,
                                *freq as i32,
                                left_id,
                                right_id,
                                new_id,
                                local_changes,
                            ) {
                                local_new_idxs.push(word_idx);
                            }
                            acc
                        },
                    )
                    .reduce(
                        || (FxHashMap::default(), Vec::new()),
                        |mut a, b| {
                            for (k, v) in b.0 {
                                *a.0.entry(k).or_insert(0) += v;
                            }
                            a.1.extend(b.1);
                            a
                        },
                    )
            } else {
                let mut changes = FxHashMap::default();
                let mut new_idxs = Vec::new();

                for &word_idx in words_to_check {
                    let (tokens, freq) = unsafe { word_list_ptr.get_mut(word_idx) };

                    if Self::merge_in_word(
                        tokens,
                        *freq as i32,
                        left_id,
                        right_id,
                        new_id,
                        &mut changes,
                    ) {
                        new_idxs.push(word_idx);
                    }
                }
                (changes, new_idxs)
            };

            pair_counts.remove(&best_pair);

            if let Some(entry) = token_to_words.get_mut(&new_id) {
                entry.extend(words_with_new_token);
            } else {
                token_to_words.insert(new_id, words_with_new_token);
            }

            for (pair, delta) in batch_changes {
                let count = pair_counts.entry(pair).or_insert(0);
                if delta < 0 {
                    *count = count.saturating_sub(delta.abs() as u32);
                } else {
                    *count += delta as u32;
                }
                if *count == 0 {
                    pair_counts.remove(&pair);
                } else {
                    heap.push(MergePair {
                        count: *count,
                        pair,
                    });
                }
            }
        }
    }

    pub fn encode(&self, text: String) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }
        let mut encoded_ids = Vec::new();
        let mut word_list: Vec<Vec<u32>> = Vec::new();
        word_list.push(
            words[0]
                .chars()
                .map(|c| self.base.get_id(&c.to_string()))
                .collect(),
        );
        for w in &words[1..] {
            let prefixed = format!("{}{}", self.delimiter, w);
            word_list.push(
                prefixed
                    .chars()
                    .map(|c| self.base.get_id(&c.to_string()))
                    .collect(),
            );
        }
        for (p_left, p_right) in &self.merges {
            let left_id = self.base.get_id(p_left);
            let right_id = self.base.get_id(p_right);
            let new_token_str = format!("{}{}", p_left, p_right);
            let new_id = self.base.get_id(&new_token_str);
            for symbols in &mut word_list {
                if symbols.len() < 2 {
                    continue;
                }
                let mut i = 0;
                while i < symbols.len() - 1 {
                    if symbols[i] == left_id && symbols[i + 1] == right_id {
                        symbols[i] = new_id;
                        symbols.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        for symbols in word_list {
            encoded_ids.extend(symbols);
        }
        encoded_ids
    }

    pub fn decode(&self, ids: Vec<u32>) -> String {
        let mut tokens = Vec::new();
        for id in ids {
            if let Some(token) = self.base.inverse_vocab.get(&id) {
                tokens.push(token.as_str());
            } else {
                tokens.push("[UNK]");
            }
        }
        tokens
            .join("")
            .replace(&self.delimiter, " ")
            .trim()
            .to_string()
    }
}
