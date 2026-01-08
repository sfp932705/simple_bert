use crate::token_encoders::base::BaseTokenizer;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

#[pyclass]
pub struct RustBPETokenizer {
    base: BaseTokenizer,
    merges: Vec<(String, String)>,
    delimiter: String,
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

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.base.vocab.clone()
    }

    pub fn get_merges(&self) -> Vec<(String, String)> {
        self.merges.clone()
    }

    pub fn train(&mut self, corpus: Vec<String>) {
        let mut counts: HashMap<Vec<String>, u32> = HashMap::new();
        for sentence in corpus {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.is_empty() {
                continue;
            }

            let first = vec![words[0].to_string()];
            *counts.entry(first).or_insert(0) += 1;

            for w in &words[1..] {
                let token = format!("{}{}", self.delimiter, w);
                let chars: Vec<String> = token.chars().map(|c| c.to_string()).collect();
                *counts.entry(chars).or_insert(0) += 1;
            }
        }

        let mut alphabet: HashSet<char> = HashSet::new();
        for (word_tokens, _) in &counts {
            for token in word_tokens {
                for c in token.chars() {
                    alphabet.insert(c);
                }
            }
        }
        let mut sorted_alphabet: Vec<char> = alphabet.into_iter().collect();
        sorted_alphabet.sort();

        self.base.initialize_vocab(sorted_alphabet);

        while self.base.vocab.len() < self.base.vocab_size {
            let mut pair_counts: HashMap<(String, String), u32> = HashMap::new();

            for (word, freq) in &counts {
                if word.len() < 2 {
                    continue;
                }
                for i in 0..word.len() - 1 {
                    let pair = (word[i].clone(), word[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += *freq;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            let best_pair = pair_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1))
                .map(|(k, _v)| k)
                .unwrap()
                .clone();

            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            self.merges.push(best_pair.clone());
            self.base.add_token(new_token.clone());

            let mut new_counts: HashMap<Vec<String>, u32> = HashMap::new();
            for (word, freq) in counts.iter() {
                let mut new_word = Vec::with_capacity(word.len());
                let mut i = 0;
                while i < word.len() {
                    if i < word.len() - 1 && word[i] == best_pair.0 && word[i + 1] == best_pair.1 {
                        new_word.push(new_token.clone());
                        i += 2;
                    } else {
                        new_word.push(word[i].clone());
                        i += 1;
                    }
                }
                *new_counts.entry(new_word).or_insert(0) += *freq;
            }
            counts = new_counts;
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
        let mut word_list: Vec<Vec<String>> = Vec::new();

        word_list.push(words[0].chars().map(|c| c.to_string()).collect());

        for w in &words[1..] {
            let prefixed = format!("{}{}", self.delimiter, w);
            word_list.push(prefixed.chars().map(|c| c.to_string()).collect());
        }

        for mut symbols in word_list {
            for pair in &self.merges {
                if !symbols.contains(&pair.0) {
                    continue;
                }
                if symbols.len() < 2 {
                    continue;
                }

                let mut new_symbols = Vec::new();
                let mut i = 0;
                while i < symbols.len() {
                    if i < symbols.len() - 1 && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                        new_symbols.push(format!("{}{}", pair.0, pair.1));
                        i += 2;
                    } else {
                        new_symbols.push(symbols[i].clone());
                        i += 1;
                    }
                }
                symbols = new_symbols;
            }

            for s in symbols {
                // DELEGATE: Get ID from base
                encoded_ids.push(self.base.get_id(&s));
            }
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
