use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TokenizerModel {
    BPE,
    WordPiece,
}

impl TokenizerModel {
    pub fn token_to_chars(&self, word: &str, delimiter: &str) -> Vec<String> {
        match self {
            TokenizerModel::BPE => {
                let s = format!("{}{}", delimiter, word);
                s.chars().map(|c| c.to_string()).collect()
            }
            TokenizerModel::WordPiece => word
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if i == 0 {
                        c.to_string()
                    } else {
                        format!("{}{}", delimiter, c)
                    }
                })
                .collect(),
        }
    }

    pub fn merge_strings(&self, part_a: &str, part_b: &str, delimiter: &str) -> String {
        match self {
            TokenizerModel::BPE => {
                format!("{}{}", part_a, part_b)
            }
            TokenizerModel::WordPiece => {
                let clean_b = if part_b.starts_with(delimiter) {
                    &part_b[delimiter.len()..]
                } else {
                    part_b
                };
                format!("{}{}", part_a, clean_b)
            }
        }
    }
}

#[derive(Clone)]
pub struct BaseTokenizer {
    pub vocab: FxHashMap<String, u32>,
    pub inverse_vocab: FxHashMap<u32, String>,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub pad_token_id: u32,
    pub mask_token_id: u32,
    pub cls_token_id: u32,
    pub sep_token_id: u32,
    pub unk_token_id: u32,
}

impl BaseTokenizer {
    pub fn new(vocab_size: usize, mut special_tokens: Vec<String>, unused_tokens: usize) -> Self {
        for i in 0..unused_tokens {
            special_tokens.push(format!("[unused{}]", i));
        }

        BaseTokenizer {
            vocab: FxHashMap::default(),
            inverse_vocab: FxHashMap::default(),
            vocab_size,
            special_tokens,
            pad_token_id: 0,
            mask_token_id: 0,
            cls_token_id: 0,
            sep_token_id: 0,
            unk_token_id: 0,
        }
    }

    pub fn update_special_tokens(&mut self, settings: &Bound<'_, PyAny>) {
        let get_setting = |name: &str| -> String {
            settings
                .getattr(name)
                .expect(&format!("Settings object is missing attribute '{}'", name))
                .extract()
                .expect(&format!("Attribute '{}' must be a string", name))
        };

        let pad_token = get_setting("pad_token");
        let mask_token = get_setting("mask_token");
        let cls_token = get_setting("cls_token");
        let sep_token = get_setting("sep_token");
        let unk_token = get_setting("unk_token");

        self.pad_token_id = *self.vocab.get(&pad_token).unwrap_or(&0);
        self.mask_token_id = *self.vocab.get(&mask_token).unwrap_or(&0);
        self.cls_token_id = *self.vocab.get(&cls_token).unwrap_or(&0);
        self.sep_token_id = *self.vocab.get(&sep_token).unwrap_or(&0);
        self.unk_token_id = *self.vocab.get(&unk_token).unwrap_or(&0);
    }

    pub fn initialize_vocab(&mut self, sorted_alphabet: Vec<String>) {
        self.vocab.clear();
        self.inverse_vocab.clear();
        for (i, token) in self.special_tokens.iter().enumerate() {
            self.vocab.insert(token.clone(), i as u32);
            self.inverse_vocab.insert(i as u32, token.clone());
        }

        for token in sorted_alphabet {
            if !self.vocab.contains_key(&token) {
                if self.vocab.len() >= self.vocab_size {
                    break;
                }
                let idx = self.vocab.len() as u32;
                self.vocab.insert(token.clone(), idx);
                self.inverse_vocab.insert(idx, token);
            }
        }
    }

    pub fn get_id(&self, token: &str) -> u32 {
        *self
            .vocab
            .get(token)
            .unwrap_or(self.vocab.get("[UNK]").unwrap_or(&0))
    }

    pub fn add_token(&mut self, token: String) {
        if !self.vocab.contains_key(&token) {
            let idx = self.vocab.len() as u32;
            self.vocab.insert(token.clone(), idx);
            self.inverse_vocab.insert(idx, token);
        }
    }

    pub fn split_by_special_tokens<'a>(&self, text: &'a str) -> Vec<(bool, &'a str)> {
        let mut tokens: Vec<(bool, &'a str)> = Vec::new();
        let mut cursor = 0;

        let mut special_tokens: Vec<&String> = self
            .special_tokens
            .iter()
            .filter(|t| self.vocab.contains_key(*t))
            .collect();
        special_tokens.sort_by(|a, b| b.len().cmp(&a.len()));

        while cursor < text.len() {
            let remainder = &text[cursor..];
            let mut matched_special: Option<&String> = None;

            for special in &special_tokens {
                if remainder.starts_with(special.as_str()) {
                    matched_special = Some(special);
                    break;
                }
            }

            if let Some(special) = matched_special {
                tokens.push((true, &text[cursor..cursor + special.len()]));
                cursor += special.len();
            } else {
                let mut next_special_idx = text.len();
                for special in &special_tokens {
                    if let Some(pos) = remainder.find(special.as_str()) {
                        if cursor + pos < next_special_idx {
                            next_special_idx = cursor + pos;
                        }
                    }
                }

                if next_special_idx > cursor {
                    tokens.push((false, &text[cursor..next_special_idx]));
                    cursor = next_special_idx;
                } else if cursor < text.len() {
                    tokens.push((false, &text[cursor..cursor + 1]));
                    cursor += 1;
                }
            }
        }
        tokens
    }
}
