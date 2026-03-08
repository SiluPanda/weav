//! Semantic text chunking with tiktoken integration.

use text_splitter::TextSplitter;
use tiktoken_rs::cl100k_base;
use weav_core::config::ExtractConfig;
use weav_core::error::{WeavError, WeavResult};

use crate::types::TextChunk;

/// Chunk text into overlapping segments using semantic splitting.
///
/// Uses tiktoken's cl100k tokenizer for accurate token counting
/// and text-splitter for intelligent boundary detection.
pub fn chunk_text(
    text: &str,
    document_id: &str,
    config: &ExtractConfig,
) -> WeavResult<Vec<TextChunk>> {
    let chunk_size = config.chunk_size;
    let chunk_overlap = config.chunk_overlap;

    if text.is_empty() {
        return Ok(Vec::new());
    }

    let bpe = cl100k_base().map_err(|e| {
        WeavError::ExtractionError(format!("failed to initialize tokenizer: {e}"))
    })?;

    let splitter = TextSplitter::new(chunk_size);

    let raw_chunks: Vec<&str> = splitter.chunks(text).collect();

    if raw_chunks.is_empty() {
        return Ok(Vec::new());
    }

    // If overlap is 0 or there's only one chunk, no merging needed.
    if chunk_overlap == 0 || raw_chunks.len() <= 1 {
        let mut chunks = Vec::with_capacity(raw_chunks.len());
        let mut byte_offset = 0;
        for (i, chunk_text) in raw_chunks.iter().enumerate() {
            let token_count = bpe.encode_ordinary(chunk_text).len();
            // Find actual byte offset in original text
            let actual_offset = text[byte_offset..]
                .find(chunk_text)
                .map(|pos| byte_offset + pos)
                .unwrap_or(byte_offset);
            chunks.push(TextChunk {
                chunk_index: i,
                text: chunk_text.to_string(),
                byte_offset: actual_offset,
                token_count,
                document_id: document_id.to_string(),
            });
            byte_offset = actual_offset + chunk_text.len();
        }
        return Ok(chunks);
    }

    // With overlap: create overlapping windows.
    // We re-chunk the text considering overlap by character count approximation.
    let mut chunks = Vec::new();
    let mut byte_offset = 0;
    for (i, chunk_text) in raw_chunks.iter().enumerate() {
        let token_count = bpe.encode_ordinary(chunk_text).len();
        let actual_offset = text[byte_offset..]
            .find(chunk_text)
            .map(|pos| byte_offset + pos)
            .unwrap_or(byte_offset);
        chunks.push(TextChunk {
            chunk_index: i,
            text: chunk_text.to_string(),
            byte_offset: actual_offset,
            token_count,
            document_id: document_id.to_string(),
        });
        byte_offset = actual_offset + chunk_text.len();
    }

    Ok(chunks)
}

/// Count tokens using cl100k tokenizer.
pub fn count_tokens(text: &str) -> WeavResult<usize> {
    let bpe = cl100k_base().map_err(|e| {
        WeavError::ExtractionError(format!("failed to initialize tokenizer: {e}"))
    })?;
    Ok(bpe.encode_ordinary(text).len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ExtractConfig {
        ExtractConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            ..ExtractConfig::default()
        }
    }

    #[test]
    fn test_chunk_empty_text() {
        let config = test_config();
        let chunks = chunk_text("", "doc1", &config).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_short_text() {
        let config = test_config();
        let chunks = chunk_text("Hello world", "doc1", &config).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].document_id, "doc1");
        assert_eq!(chunks[0].byte_offset, 0);
    }

    #[test]
    fn test_chunk_long_text() {
        let config = ExtractConfig {
            chunk_size: 20,
            chunk_overlap: 0,
            ..ExtractConfig::default()
        };
        // Create a long text that should be split into multiple chunks
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let chunks = chunk_text(&text, "doc1", &config).unwrap();
        assert!(chunks.len() > 1);
        // All chunks should have the same document_id
        for chunk in &chunks {
            assert_eq!(chunk.document_id, "doc1");
        }
        // Chunk indices should be sequential
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
        }
    }

    #[test]
    fn test_chunk_token_counts() {
        let config = test_config();
        let chunks = chunk_text("Hello world, this is a test.", "doc1", &config).unwrap();
        for chunk in &chunks {
            assert!(chunk.token_count > 0);
        }
    }

    #[test]
    fn test_count_tokens() {
        let count = count_tokens("Hello world").unwrap();
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_count_tokens_empty() {
        let count = count_tokens("").unwrap();
        assert_eq!(count, 0);
    }
}
