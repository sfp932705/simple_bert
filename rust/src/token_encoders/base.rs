use std::collections::HashMap;

pub struct BaseTokenizer {
    pub vocab: HashMap<String, u32>,
    pub inverse_vocab: HashMap<u32, String>,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
}

impl BaseTokenizer {
    pub fn new(vocab_size: usize, mut special_tokens: Vec<String>, unused_tokens: usize) -> Self {
        for i in 0..unused_tokens {
            special_tokens.push(format!("[unused{}]", i));
        }

        BaseTokenizer {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            vocab_size,
            special_tokens, // This now contains the FULL list
        }
    }

    pub fn initialize_vocab(&mut self, sorted_alphabet: Vec<char>) {
        self.vocab.clear();
        self.inverse_vocab.clear();
        for (i, token) in self.special_tokens.iter().enumerate() {
            self.vocab.insert(token.clone(), i as u32);
            self.inverse_vocab.insert(i as u32, token.clone());
        }

        for c in sorted_alphabet {
            let s = c.to_string();
            if !self.vocab.contains_key(&s) {
                if self.vocab.len() >= self.vocab_size {
                    break;
                }
                let idx = self.vocab.len() as u32;
                self.vocab.insert(s.clone(), idx);
                self.inverse_vocab.insert(idx, s);
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
