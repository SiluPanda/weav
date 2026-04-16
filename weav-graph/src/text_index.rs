//! Full-text inverted index with BM25 scoring.

use std::collections::{HashMap, HashSet};
use weav_core::types::NodeId;

/// A full-text search index that maps tokens to document (node) IDs
/// and supports BM25-ranked retrieval.
pub struct TextIndex {
    /// token -> set of (node_id, term_frequency)
    inverted: HashMap<String, Vec<(NodeId, u32)>>,
    /// node_id -> document length (total tokens)
    doc_lengths: HashMap<NodeId, u32>,
    /// Total number of indexed documents
    doc_count: u32,
    /// Sum of all document lengths (for avg_dl)
    total_length: u64,
}

impl TextIndex {
    pub fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_count: 0,
            total_length: 0,
        }
    }

    /// Index a node's text content. Tokenizes the text and adds to inverted index.
    /// Call this with concatenated string properties of a node.
    pub fn index_node(&mut self, node_id: NodeId, text: &str) {
        // Remove old entry if re-indexing
        self.remove_node(node_id);

        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }

        let doc_len = tokens.len() as u32;
        self.doc_lengths.insert(node_id, doc_len);
        self.doc_count += 1;
        self.total_length += doc_len as u64;

        // Count term frequencies
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }

        // Add to inverted index
        for (term, count) in tf {
            self.inverted
                .entry(term.to_string())
                .or_default()
                .push((node_id, count));
        }
    }

    /// Remove a node from the index.
    pub fn remove_node(&mut self, node_id: NodeId) {
        if let Some(old_len) = self.doc_lengths.remove(&node_id) {
            self.doc_count = self.doc_count.saturating_sub(1);
            self.total_length = self.total_length.saturating_sub(old_len as u64);

            // Remove from inverted index
            for postings in self.inverted.values_mut() {
                postings.retain(|&(nid, _)| nid != node_id);
            }
            // Clean up empty entries
            self.inverted.retain(|_, v| !v.is_empty());
        }
    }

    /// Search using BM25 scoring. Returns nodes ranked by relevance.
    /// k1 = 1.2, b = 0.75 are standard BM25 parameters.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(NodeId, f32)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || self.doc_count == 0 {
            return Vec::new();
        }

        let k1: f32 = 1.2;
        let b: f32 = 0.75;
        let avg_dl = self.total_length as f32 / self.doc_count as f32;

        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Deduplicate query tokens
        let unique_terms: HashSet<&str> = query_tokens.iter().map(|s| s.as_str()).collect();

        for term in unique_terms {
            if let Some(postings) = self.inverted.get(term) {
                let df = postings.len() as f32;
                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((self.doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(node_id, tf) in postings {
                    let dl = self.doc_lengths.get(&node_id).copied().unwrap_or(1) as f32;
                    // BM25 term score
                    let tf_norm =
                        (tf as f32 * (k1 + 1.0)) / (tf as f32 + k1 * (1.0 - b + b * dl / avg_dl));
                    *scores.entry(node_id).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        // Sort by score descending
        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_count as usize
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Number of unique terms in the index.
    pub fn term_count(&self) -> usize {
        self.inverted.len()
    }
}

impl Default for TextIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple whitespace + punctuation tokenizer. Lowercases and strips non-alphanumeric chars.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() >= 2) // skip single chars
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_and_search_basic() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "The quick brown fox jumps over the lazy dog");
        idx.index_node(2, "A fast brown car drives past the sleeping cat");
        idx.index_node(3, "Quantum computing advances in 2026");

        let results = idx.search("brown fox", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1); // node 1 should rank highest (has both terms)
    }

    #[test]
    fn test_bm25_idf_ranking() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "rust programming language");
        idx.index_node(2, "rust prevention for metal surfaces");
        idx.index_node(3, "programming in python and rust");

        let results = idx.search("rust programming", 10);
        // Node 1 and 3 have both terms; should rank higher than node 2
        assert!(results.len() >= 2);
        let top_ids: Vec<NodeId> = results.iter().take(2).map(|r| r.0).collect();
        assert!(top_ids.contains(&1));
        assert!(top_ids.contains(&3));
    }

    #[test]
    fn test_remove_node() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "hello world");
        idx.index_node(2, "hello there");
        assert_eq!(idx.len(), 2);

        idx.remove_node(1);
        assert_eq!(idx.len(), 1);

        let results = idx.search("world", 10);
        assert!(results.is_empty()); // "world" was only in node 1
    }

    #[test]
    fn test_reindex_node() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "old content here");
        idx.index_node(1, "new content replaced");
        assert_eq!(idx.len(), 1);

        let results = idx.search("old", 10);
        assert!(results.is_empty());

        let results = idx.search("replaced", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_empty_search() {
        let idx = TextIndex::new();
        let results = idx.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_query() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "some content");
        let results = idx.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_term_count() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "hello world hello");
        assert_eq!(idx.term_count(), 2); // "hello" and "world"
    }

    #[test]
    fn test_many_documents_ranking() {
        let mut idx = TextIndex::new();
        // "database" appears in all docs (low IDF)
        // "graph" appears in fewer (higher IDF)
        for i in 1..=20 {
            idx.index_node(i, &format!("database system number {i}"));
        }
        idx.index_node(100, "graph database for AI workloads");
        idx.index_node(101, "graph algorithms and traversal methods overview");

        let results = idx.search("graph database", 5);
        // Node 100 has both query terms; node 101 only has "graph".
        // With similar doc lengths, multi-term match wins.
        assert_eq!(results[0].0, 100);
    }

    #[test]
    fn test_single_char_tokens_filtered() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "I am a test");
        // "I", "a" should be filtered (single char)
        assert_eq!(idx.term_count(), 2); // "am" and "test"
    }

    #[test]
    fn test_case_insensitive() {
        let mut idx = TextIndex::new();
        idx.index_node(1, "Hello WORLD");
        let results = idx.search("hello world", 10);
        assert_eq!(results.len(), 1);
    }
}
