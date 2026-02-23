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
            TokenCounterType::Exact(_) => {
                // Fall back to CharDiv4 for now
                char_div_4(text)
            }
        }
    }

    /// Count tokens for a batch of texts.
    pub fn count_batch(&self, texts: &[&str]) -> Vec<u32> {
        texts.iter().map(|t| self.count(t)).collect()
    }
}

/// Ceiling division of byte length by 4.
fn char_div_4(text: &str) -> u32 {
    (text.len() as u32 + 3) / 4
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
    fn test_count_batch() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let results = counter.count_batch(&["", "a", "abcd", "abcde"]);
        assert_eq!(results, vec![0, 1, 1, 2]);
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
    fn test_batch_with_tiktoken() {
        let counter = TokenCounter::new(TokenCounterType::TiktokenCl100k);
        let results = counter.count_batch(&["hello", "world", ""]);
        assert_eq!(results.len(), 3);
        assert!(results[0] > 0);
        assert!(results[1] > 0);
        assert_eq!(results[2], 0);
    }
}
