use weav_core::config::TokenCounterType;
use weav_core::types::{NodeId, Value};

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

    /// Count tokens for a batch of texts.
    pub fn count_batch(&self, texts: &[&str]) -> Vec<u32> {
        texts.iter().map(|t| self.count(t)).collect()
    }

    /// Count tokens for a node by combining its properties into a single string.
    ///
    /// Each property is formatted as `"key: value"` and all pairs are joined
    /// with newlines before counting.
    pub fn count_node(&self, _node_id: NodeId, properties: &[(String, Value)]) -> u32 {
        let combined: String = properties
            .iter()
            .map(|(key, value)| format!("{}: {}", key, value_to_string(value)))
            .collect::<Vec<_>>()
            .join("\n");
        self.count(&combined)
    }

    /// Count tokens for a context chunk given its content, label, and
    /// relationship strings.
    ///
    /// This avoids importing `ContextChunk` from `weav-query` (which depends
    /// on `weav-vector`), thereby preventing a circular dependency.
    pub fn count_chunk(&self, content: &str, label: &str, relationships: &[String]) -> u32 {
        let mut text = String::with_capacity(
            label.len() + content.len() + relationships.iter().map(|r| r.len() + 1).sum::<usize>(),
        );
        text.push_str(label);
        text.push('\n');
        text.push_str(content);
        for rel in relationships {
            text.push('\n');
            text.push_str(rel);
        }
        self.count(&text)
    }
}

/// Convert a `Value` to a human-readable string representation for token
/// counting purposes.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s.to_string(),
        Value::Bytes(b) => format!("<{} bytes>", b.len()),
        Value::Vector(v) => format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", ")),
        Value::List(items) => format!("[{}]", items.iter().map(value_to_string).collect::<Vec<_>>().join(", ")),
        Value::Map(entries) => format!(
            "{{{}}}",
            entries
                .iter()
                .map(|(k, v)| format!("{}: {}", k, value_to_string(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Timestamp(ts) => format!("@{}", ts),
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

    #[test]
    fn test_count_node_empty_properties() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let count = counter.count_node(1, &[]);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_node_with_properties() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let props = vec![
            ("name".to_string(), Value::String("Alice".into())),
            ("age".to_string(), Value::Int(30)),
        ];
        let count = counter.count_node(42, &props);
        // "name: Alice\nage: 30" = 19 bytes, ceil(19/4) = 5
        assert_eq!(count, 5);
    }

    #[test]
    fn test_count_node_various_value_types() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let props = vec![
            ("flag".to_string(), Value::Bool(true)),
            ("score".to_string(), Value::Float(3.14)),
            ("data".to_string(), Value::Null),
        ];
        let count = counter.count_node(1, &props);
        assert!(count > 0);
    }

    #[test]
    fn test_count_chunk_basic() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let count = counter.count_chunk("some content", "Person", &[]);
        // "Person\nsome content" = 19 bytes, ceil(19/4) = 5
        assert_eq!(count, 5);
    }

    #[test]
    fn test_count_chunk_with_relationships() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let rels = vec!["knows Bob".to_string(), "lives in NYC".to_string()];
        let count = counter.count_chunk("Alice", "Person", &rels);
        // "Person\nAlice\nknows Bob\nlives in NYC" = 35 bytes, ceil(35/4) = 9
        assert_eq!(count, 9);
    }

    #[test]
    fn test_count_chunk_empty() {
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let count = counter.count_chunk("", "", &[]);
        // "\n" = 1 byte, ceil(1/4) = 1
        assert_eq!(count, 1);
    }

    #[test]
    fn test_exact_fallback_warning() {
        // Exact variant falls back to CharDiv4 with a warning (v0.1 behavior)
        let counter = TokenCounter::new(TokenCounterType::Exact("/some/path".into()));
        let count = counter.count("hello world");
        // "hello world" = 11 bytes, ceil(11/4) = 3
        assert_eq!(count, 3);
    }

    #[test]
    fn test_value_to_string_coverage() {
        // Verify all Value variants produce non-empty strings
        assert_eq!(value_to_string(&Value::Null), "null");
        assert_eq!(value_to_string(&Value::Bool(false)), "false");
        assert_eq!(value_to_string(&Value::Int(-5)), "-5");
        assert_eq!(value_to_string(&Value::Float(2.5)), "2.5");
        assert_eq!(value_to_string(&Value::String("hi".into())), "hi");
        assert_eq!(value_to_string(&Value::Bytes(vec![1, 2, 3])), "<3 bytes>");
        assert_eq!(value_to_string(&Value::Vector(vec![1.0, 2.0])), "[1, 2]");
        assert_eq!(value_to_string(&Value::Timestamp(12345)), "@12345");

        let list = Value::List(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(value_to_string(&list), "[1, 2]");

        let map = Value::Map(vec![("k".into(), Value::Bool(true))]);
        assert_eq!(value_to_string(&map), "{k: true}");
    }
}
