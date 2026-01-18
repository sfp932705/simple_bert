use crate::token_encoders::bpe::RustBPETokenizer;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::UnicodeNormalization;

struct TrieNode {
    children: FxHashMap<char, Box<TrieNode>>,
    token_id: Option<u32>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: FxHashMap::default(),
            token_id: None,
        }
    }
}

#[pyclass]
pub struct RustWordPieceTokenizer {
    bpe: RustBPETokenizer,
    trie_root: TrieNode,
    do_lower_case: bool,
    unk_token: String,
    delimiter: String,
}

impl RustWordPieceTokenizer {
    fn is_punctuation(c: char) -> bool {
        let cp = c as u32;
        if (cp >= 33 && cp <= 47)
            || (cp >= 58 && cp <= 64)
            || (cp >= 91 && cp <= 96)
            || (cp >= 123 && cp <= 126)
        {
            return true;
        }

        match get_general_category(c) {
            GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::MathSymbol
            | GeneralCategory::CurrencySymbol
            | GeneralCategory::ModifierSymbol
            | GeneralCategory::OtherSymbol => true,
            _ => false,
        }
    }

    fn is_cjk(c: char) -> bool {
        let cp = c as u32;
        (cp >= 0x4E00 && cp <= 0x9FFF)
            || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF)
    }

    fn pre_tokenize_text(&self, text: &str) -> Vec<String> {
        let mut processed_text = text.to_string();
        if self.do_lower_case {
            processed_text = processed_text.to_lowercase();

            processed_text = processed_text
                .nfd()
                .filter(|c| get_general_category(*c) != GeneralCategory::NonspacingMark)
                .collect();
        }

        let mut split_tokens = Vec::new();

        for token in processed_text.split_whitespace() {
            let chars: Vec<char> = token.chars().collect();
            let mut current_word = String::new();

            for &c in &chars {
                if Self::is_punctuation(c) || Self::is_cjk(c) {
                    if !current_word.is_empty() {
                        split_tokens.push(current_word.clone());
                        current_word.clear();
                    }
                    split_tokens.push(c.to_string());
                } else {
                    current_word.push(c);
                }
            }
            if !current_word.is_empty() {
                split_tokens.push(current_word);
            }
        }
        split_tokens
    }

    fn build_index(&mut self) {
        self.trie_root = TrieNode::new();
        let vocab = self.bpe.get_vocab();

        for (token, id) in vocab {
            let mut node = &mut self.trie_root;
            for c in token.chars() {
                node = node
                    .children
                    .entry(c)
                    .or_insert_with(|| Box::new(TrieNode::new()));
            }
            node.token_id = Some(id);
        }
    }
}

#[pymethods]
impl RustWordPieceTokenizer {
    #[new]
    pub fn new(
        vocab_size: usize,
        special_tokens: Vec<String>,
        unused_tokens: usize,
        delimiter: Option<String>,
    ) -> Self {
        let delim = delimiter.unwrap_or_else(|| "##".to_string());

        RustWordPieceTokenizer {
            bpe: RustBPETokenizer::new(
                vocab_size,
                special_tokens,
                unused_tokens,
                delim.clone(),
                Some(true),
            ),
            trie_root: TrieNode::new(),
            do_lower_case: true,
            unk_token: "[UNK]".to_string(),
            delimiter: delim,
        }
    }

    pub fn get_vocab(&self) -> FxHashMap<String, u32> {
        self.bpe.get_vocab()
    }

    pub fn get_merges(&self) -> Vec<(String, String)> {
        self.bpe.get_merges()
    }

    pub fn train(&mut self, corpus: Vec<String>) {
        let processed_corpus: Vec<String> = corpus
            .par_iter()
            .map(|text| {
                let tokens = self.pre_tokenize_text(text);
                tokens.join(" ")
            })
            .collect();

        self.bpe.train(processed_corpus);
        self.build_index();
    }

    pub fn encode(&mut self, text: String) -> Vec<u32> {
        if self.trie_root.children.is_empty() && !self.bpe.get_vocab().is_empty() {
            self.build_index();
        }

        let words = self.pre_tokenize_text(&text);
        let mut output_ids = Vec::new();
        let unk_id = *self.bpe.base.vocab.get(&self.unk_token).unwrap_or(&0);

        for word in words {
            let chars: Vec<char> = word.chars().collect();
            let n = chars.len();
            let mut i = 0;
            let mut is_bad = false;
            let mut word_ids = Vec::new();

            while i < n {
                let mut node = &self.trie_root;

                if i > 0 {
                    for delimiter_char in self.delimiter.chars() {
                        if let Some(next_node) = node.children.get(&delimiter_char) {
                            node = next_node;
                        } else {
                            is_bad = true;
                            break;
                        }
                    }
                    if is_bad {
                        break;
                    }
                }

                let mut j = i;
                let mut last_token_id = None;
                let mut last_match_end = 0;

                while j < n {
                    let c = chars[j];
                    if let Some(next_node) = node.children.get(&c) {
                        node = next_node;
                        if node.token_id.is_some() {
                            last_token_id = node.token_id;
                            last_match_end = j;
                        }
                        j += 1;
                    } else {
                        break;
                    }
                }

                if let Some(id) = last_token_id {
                    word_ids.push(id);
                    i = last_match_end + 1;
                } else {
                    is_bad = true;
                    break;
                }
            }

            if is_bad {
                output_ids.push(unk_id);
            } else {
                output_ids.extend(word_ids);
            }
        }

        output_ids
    }

    pub fn decode(&self, ids: Vec<u32>) -> String {
        let vocab = &self.bpe.base.inverse_vocab;

        let tokens: Vec<String> = ids
            .iter()
            .map(|id| {
                vocab
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| self.unk_token.clone())
            })
            .collect();

        let joined = tokens.join(" ");
        let target = format!(" {}", self.delimiter);
        joined.replace(&target, "")
    }
}
