//! Token budget enforcement using a greedy knapsack algorithm.
//!
//! Chunks are ranked by value density (relevance_score / token_count),
//! greedily included until the budget is exhausted, and then re-sorted
//! by relevance_score for final output ordering.

use weav_core::types::TokenBudget;

use crate::executor::ContextChunk;

/// Result of budget enforcement.
#[derive(Debug)]
pub struct BudgetResult {
    /// Chunks that fit within the budget, sorted by relevance_score descending.
    pub included: Vec<ContextChunk>,
    /// Number of chunks that were excluded.
    pub excluded_count: u32,
    /// Total tokens used by included chunks.
    pub total_tokens: u32,
    /// Utilization ratio: total_tokens / max_tokens.
    pub budget_utilization: f32,
}

/// Enforce a token budget on a list of context chunks.
///
/// Algorithm:
/// 1. Sort chunks by value density: `relevance_score / token_count`
/// 2. Greedily include chunks until budget is exhausted
/// 3. Re-sort included chunks by `relevance_score` descending (for output ordering)
/// 4. Calculate utilization = `total_tokens / max_tokens`
pub fn enforce_budget(chunks: Vec<ContextChunk>, budget: &TokenBudget) -> BudgetResult {
    if chunks.is_empty() {
        return BudgetResult {
            included: Vec::new(),
            excluded_count: 0,
            total_tokens: 0,
            budget_utilization: 0.0,
        };
    }

    let max_tokens = budget.max_tokens;

    if max_tokens == 0 {
        return BudgetResult {
            included: Vec::new(),
            excluded_count: chunks.len() as u32,
            total_tokens: 0,
            budget_utilization: 0.0,
        };
    }

    // Sort by value density (relevance_score / token_count), descending.
    // Chunks with zero tokens get infinite density (include them first).
    let mut ranked: Vec<(usize, f32)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let density = if c.token_count == 0 {
                f32::MAX
            } else {
                c.relevance_score / c.token_count as f32
            };
            (i, density)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut included_indices = Vec::new();
    let mut total_tokens: u32 = 0;

    for (idx, _density) in &ranked {
        let chunk = &chunks[*idx];
        if total_tokens + chunk.token_count <= max_tokens {
            total_tokens += chunk.token_count;
            included_indices.push(*idx);
        }
    }

    let excluded_count = (chunks.len() - included_indices.len()) as u32;

    // Collect included chunks
    // We need to consume the original vec, so convert to indexed access
    let mut chunks_vec: Vec<Option<ContextChunk>> = chunks.into_iter().map(Some).collect();
    let mut included: Vec<ContextChunk> = included_indices
        .iter()
        .filter_map(|&i| chunks_vec[i].take())
        .collect();

    // Re-sort by relevance_score descending for output ordering
    included.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let budget_utilization = if max_tokens > 0 {
        total_tokens as f32 / max_tokens as f32
    } else {
        0.0
    };

    BudgetResult {
        included,
        excluded_count,
        total_tokens,
        budget_utilization,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use weav_core::types::TokenBudget;

    fn make_chunk(node_id: u64, relevance: f32, tokens: u32) -> ContextChunk {
        ContextChunk {
            node_id,
            content: format!("content for node {node_id}"),
            label: "test".to_string(),
            relevance_score: relevance,
            depth: 0,
            token_count: tokens,
            provenance: None,
            relationships: Vec::new(),
        }
    }

    #[test]
    fn test_enforce_budget_empty() {
        let result = enforce_budget(Vec::new(), &TokenBudget::new(100));
        assert!(result.included.is_empty());
        assert_eq!(result.excluded_count, 0);
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.budget_utilization, 0.0);
    }

    #[test]
    fn test_enforce_budget_all_fit() {
        let chunks = vec![
            make_chunk(1, 0.9, 10),
            make_chunk(2, 0.8, 20),
            make_chunk(3, 0.7, 30),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(100));
        assert_eq!(result.included.len(), 3);
        assert_eq!(result.excluded_count, 0);
        assert_eq!(result.total_tokens, 60);
        assert!((result.budget_utilization - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_enforce_budget_partial_fit() {
        let chunks = vec![
            make_chunk(1, 0.9, 50),
            make_chunk(2, 0.8, 50),
            make_chunk(3, 0.7, 50),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(100));
        assert_eq!(result.included.len(), 2);
        assert_eq!(result.excluded_count, 1);
        assert_eq!(result.total_tokens, 100);
        assert!((result.budget_utilization - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_enforce_budget_density_ordering() {
        // Chunk 1: high relevance but high tokens (density = 0.9/100 = 0.009)
        // Chunk 2: moderate relevance, low tokens (density = 0.5/10 = 0.05)
        // Chunk 3: low relevance, low tokens (density = 0.3/10 = 0.03)
        let chunks = vec![
            make_chunk(1, 0.9, 100),
            make_chunk(2, 0.5, 10),
            make_chunk(3, 0.3, 10),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(25));

        // Should include chunk 2 and 3 (density-first), not chunk 1
        assert_eq!(result.included.len(), 2);
        let ids: Vec<u64> = result.included.iter().map(|c| c.node_id).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_enforce_budget_output_sorted_by_relevance() {
        let chunks = vec![
            make_chunk(1, 0.3, 10),
            make_chunk(2, 0.9, 10),
            make_chunk(3, 0.5, 10),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(100));

        // Output should be sorted by relevance descending
        assert_eq!(result.included.len(), 3);
        assert!((result.included[0].relevance_score - 0.9).abs() < 0.001);
        assert!((result.included[1].relevance_score - 0.5).abs() < 0.001);
        assert!((result.included[2].relevance_score - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_enforce_budget_zero_budget() {
        let chunks = vec![make_chunk(1, 0.9, 10)];
        let result = enforce_budget(chunks, &TokenBudget::new(0));
        assert!(result.included.is_empty());
        assert_eq!(result.excluded_count, 1);
    }

    #[test]
    fn test_enforce_budget_zero_token_chunks() {
        let chunks = vec![
            make_chunk(1, 0.9, 0),
            make_chunk(2, 0.8, 10),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(10));

        // Zero-token chunks should always be included
        assert_eq!(result.included.len(), 2);
        assert_eq!(result.total_tokens, 10);
    }

    #[test]
    fn test_enforce_budget_single_chunk_too_large() {
        let chunks = vec![make_chunk(1, 0.9, 200)];
        let result = enforce_budget(chunks, &TokenBudget::new(100));
        assert!(result.included.is_empty());
        assert_eq!(result.excluded_count, 1);
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_enforce_budget_utilization() {
        let chunks = vec![
            make_chunk(1, 0.9, 25),
            make_chunk(2, 0.8, 25),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(100));
        assert!((result.budget_utilization - 0.5).abs() < 0.001);
    }
}
