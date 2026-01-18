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

// -----------------------------------------------------------------------------
// Base Tokenizer Struct
// -----------------------------------------------------------------------------
#[derive(Clone)]
pub struct BaseTokenizer {
    pub vocab: FxHashMap<String, u32>,
    pub inverse_vocab: FxHashMap<u32, String>,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
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
        }
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
}
