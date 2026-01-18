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
    #[inline(always)]
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

    #[inline(always)]
    fn is_cjk(c: char) -> bool {
        let cp = c as u32;
        (cp >= 0x4E00 && cp <= 0x9FFF)
            || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF)
            || (cp >= 0x2A700 && cp <= 0x2B73F)
            || (cp >= 0x2B740 && cp <= 0x2B81F)
            || (cp >= 0x2B820 && cp <= 0x2CEAF)
            || (cp >= 0xF900 && cp <= 0xFAFF)
            || (cp >= 0x2F800 && cp <= 0x2FA1F)
            || (cp >= 0xAC00 && cp <= 0xD7A3)
            || (cp >= 0x1100 && cp <= 0x11FF)
            || (cp >= 0x3130 && cp <= 0x318F)
    }

    #[inline(always)]
    fn is_control(c: char) -> bool {
        if c == '\t' || c == '\n' || c == '\r' {
            return false;
        }
        let cat = get_general_category(c);
        cat == GeneralCategory::Control || cat == GeneralCategory::Format
    }

    /// ULTRA-FAST PRE-TOKENIZER
    /// Writes directly to a single output String to avoid millions of allocations.
    /// Returns: "Hello , world !" (Space separated tokens)
    fn pre_tokenize_to_string(&self, text: &str) -> String {
        // Reserve significant capacity to avoid re-allocations.
        // 1.5x length is a heuristic for adding spaces around punctuation.
        let mut output = String::with_capacity((text.len() as f64 * 1.5) as usize);

        // --- PATH A: FAST ASCII ---
        if text.is_ascii() {
            for c in text.chars() {
                if self.do_lower_case {
                    let lc = c.to_ascii_lowercase();
                    if Self::is_control(lc) {
                        continue;
                    }

                    if Self::is_punctuation(lc) {
                        output.push(' ');
                        output.push(lc);
                        output.push(' ');
                    } else if lc.is_whitespace() {
                        output.push(' ');
                    } else {
                        output.push(lc);
                    }
                } else {
                    if Self::is_control(c) {
                        continue;
                    }
                    if Self::is_punctuation(c) {
                        output.push(' ');
                        output.push(c);
                        output.push(' ');
                    } else if c.is_whitespace() {
                        output.push(' ');
                    } else {
                        output.push(c);
                    }
                }
            }
            return output;
        }

        // --- PATH B: UNICODE (NFD + CATEGORIES) ---
        let chars_iter = if self.do_lower_case {
            text.chars()
                .flat_map(|c| c.to_lowercase())
                .nfd()
                .filter(|c| {
                    !Self::is_control(*c)
                        && get_general_category(*c) != GeneralCategory::NonspacingMark
                })
                .collect::<Vec<char>>() // Collect needed to iterate safely, but simpler than String
        } else {
            text.chars()
                .filter(|c| !Self::is_control(*c))
                .collect::<Vec<char>>()
        };

        for c in chars_iter {
            if c.is_whitespace() {
                output.push(' ');
            } else if Self::is_punctuation(c) || Self::is_cjk(c) {
                output.push(' ');
                output.push(c);
                output.push(' ');
            } else {
                output.push(c);
            }
        }
        output
    }

    // Legacy helper for inference (creates list of strings)
    fn pre_tokenize_text(&self, text: &str) -> Vec<String> {
        self.pre_tokenize_to_string(text)
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
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
        wordpiece_mode: Option<bool>,
    ) -> Self {
        let delim = delimiter.unwrap_or_else(|| "##".to_string());

        RustWordPieceTokenizer {
            bpe: RustBPETokenizer::new(
                vocab_size,
                special_tokens,
                unused_tokens,
                delim.clone(),
                Option::from(wordpiece_mode.unwrap_or(true)),
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

    // OPTIMIZED TRAIN
    pub fn train(&mut self, corpus: String) {
        // 1. Parallel Pre-tokenization using the ZERO-ALLOCATION helper
        let processed_corpus: String = corpus
            .par_lines()
            .map(|line| self.pre_tokenize_to_string(line)) // Returns String directly
            .collect::<Vec<String>>()
            .join("\n");

        // 2. Delegate to BPE
        self.bpe.train(processed_corpus);

        // 3. Build Index
        self.build_index();
    }

    pub fn encode(&mut self, text: String) -> Vec<u32> {
        if self.trie_root.children.is_empty() && !self.bpe.get_vocab().is_empty() {
            self.build_index();
        }
        // We can reuse the string helper here too, but splitting is cheap for single inference
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
        tokens
            .join(" ")
            .replace(&format!(" {}", self.delimiter), "")
    }
}
