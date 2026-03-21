use weav_core::config::TokenCounterType;

/// Token counter supporting multiple counting strategies.
pub struct TokenCounter {
    counter_type: TokenCounterType,
}

impl TokenCounter {
    /// Create a new token counter with the specified strategy.
    pub fn new(counter_type: TokenCounterType) -> Self {
        Self { counter_type }
    }

    /// Count tokens in the given text.
    pub fn count(&self, text: &str) -> u32 {
        match &self.counter_type {
            TokenCounterType::CharDiv4 => char_div_4(text),
            TokenCounterType::TiktokenCl100k => {
                let bpe = tiktoken_rs::cl100k_base_singleton();
                let lock = bpe.lock();
                lock.encode_with_special_tokens(text).len() as u32
            }
            TokenCounterType::TiktokenO200k => {
                let bpe = tiktoken_rs::o200k_base_singleton();
                let lock = bpe.lock();
                lock.encode_with_special_tokens(text).len() as u32
            }
            TokenCounterType::Exact(path) => {
                // v0.1: Custom tokenizer loading is not yet implemented.
                // Falling back to CharDiv4 intentionally until custom BPE
                // loading is supported in a future release.
                eprintln!(
                    "weav-vector: Exact tokenizer '{}' not yet supported, falling back to CharDiv4",
                    path
                );
                char_div_4(text)
            }
        }
    }
}

/// Ceiling division of byte length by 4.
fn char_div_4(text: &str) -> u32 {
    (text.len() as u32).div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_div_4_empty() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        assert_eq!(counter.count(""), 0);
    }

    #[test]
    fn test_char_div_4_short() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        // 1 byte -> ceil(1/4) = 1
        assert_eq!(counter.count("a"), 1);
        // 4 bytes -> ceil(4/4) = 1
        assert_eq!(counter.count("abcd"), 1);
        // 5 bytes -> ceil(5/4) = 2
        assert_eq!(counter.count("abcde"), 2);
    }

    #[test]
    fn test_char_div_4_various() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        // 8 bytes -> 2
        assert_eq!(counter.count("12345678"), 2);
        // 9 bytes -> 3
        assert_eq!(counter.count("123456789"), 3);
        // 12 bytes -> 3
        assert_eq!(counter.count("123456789012"), 3);
        // 13 bytes -> 4
        assert_eq!(counter.count("1234567890123"), 4);
    }

    #[test]
    fn test_tiktoken_cl100k() {
        let counter = TokenCounter::new(TokenCounterType::TiktokenCl100k);
        // "hello world" should produce some tokens
        let count = counter.count("hello world");
        assert!(count > 0);
        assert!(count < 10); // sanity check
    }

    #[test]
    fn test_tiktoken_o200k() {
        let counter = TokenCounter::new(TokenCounterType::TiktokenO200k);
        let count = counter.count("hello world");
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_exact_fallback() {
        let counter = TokenCounter::new(TokenCounterType::Exact("/nonexistent".into()));
        // Falls back to char_div_4
        assert_eq!(counter.count("abcd"), 1);
    }

    #[test]
    fn test_tiktoken_empty() {
        let counter = TokenCounter::new(TokenCounterType::TiktokenCl100k);
        assert_eq!(counter.count(""), 0);
    }

    #[test]
    fn test_exact_fallback_warning() {
        // Exact variant falls back to CharDiv4 with a warning (v0.1 behavior)
        let counter = TokenCounter::new(TokenCounterType::Exact("/some/path".into()));
        let count = counter.count("hello world");
        // "hello world" = 11 bytes, ceil(11/4) = 3
        assert_eq!(count, 3);
    }

    // ── Round 6 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_token_count_multibyte_utf8() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        // "cafe\u{0301}" = "caf\xc3\xa9" in UTF-8, 5 bytes (c, a, f, 0xc3, 0xa9)
        // Actually "café" as a single char e-acute is 5 bytes total
        let text = "caf\u{00e9}";
        let byte_len = text.len(); // 5 bytes (c=1, a=1, f=1, e-acute=2)
        assert_eq!(byte_len, 5);
        // CharDiv4 counts bytes: ceil(5/4) = 2
        assert_eq!(counter.count(text), 2);
    }

    #[test]
    fn test_token_count_emoji() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let text = "Hello \u{1F30D}"; // "Hello 🌍"
        let byte_len = text.len(); // "Hello " = 6 bytes, 🌍 = 4 bytes = 10 total
        assert_eq!(byte_len, 10);
        // CharDiv4: ceil(10/4) = 3
        assert_eq!(counter.count(text), 3);
    }

    #[test]
    fn test_token_count_cjk() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let text = "\u{4F60}\u{597D}\u{4E16}\u{754C}"; // "你好世界"
        let byte_len = text.len(); // Each CJK char is 3 bytes in UTF-8: 4*3 = 12
        assert_eq!(byte_len, 12);
        // CharDiv4: ceil(12/4) = 3
        assert_eq!(counter.count(text), 3);
    }

    #[test]
    fn test_token_count_very_long_string() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let text = "a".repeat(10000);
        // 10000 bytes, CharDiv4: ceil(10000/4) = 2500
        assert_eq!(counter.count(&text), 2500);
    }

}
