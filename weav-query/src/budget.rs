//! Token budget enforcement using a greedy knapsack algorithm.
//!
//! Chunks are ranked by value density (relevance_score / token_count),
//! greedily included until the budget is exhausted, and then re-sorted
//! by relevance_score for final output ordering.

use weav_core::types::{NodeId, TokenAllocation, TokenBudget};

use crate::executor::ContextChunk;

/// Result of budget enforcement.
#[derive(Debug)]
pub struct BudgetResult {
    /// Chunks that fit within the budget, sorted by relevance_score descending.
    pub included: Vec<ContextChunk>,
    /// Node IDs of chunks that were excluded.
    pub excluded: Vec<NodeId>,
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
            excluded: Vec::new(),
            total_tokens: 0,
            budget_utilization: 0.0,
        };
    }

    let max_tokens = budget.max_tokens;

    if max_tokens == 0 {
        let excluded: Vec<NodeId> = chunks.iter().map(|c| c.node_id).collect();
        return BudgetResult {
            included: Vec::new(),
            excluded,
            total_tokens: 0,
            budget_utilization: 0.0,
        };
    }

    match &budget.allocation {
        TokenAllocation::Proportional {
            entities_pct,
            relationships_pct,
            text_chunks_pct,
            metadata_pct,
        } => enforce_budget_proportional(
            chunks,
            max_tokens,
            *entities_pct,
            *relationships_pct,
            *text_chunks_pct,
            *metadata_pct,
        ),
        _ => enforce_budget_greedy(chunks, max_tokens),
    }
}

/// Greedy budget enforcement (for Auto and Priority allocations).
fn enforce_budget_greedy(chunks: Vec<ContextChunk>, max_tokens: u32) -> BudgetResult {
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

    // Collect excluded node IDs
    let included_set: std::collections::HashSet<usize> = included_indices.iter().copied().collect();
    let excluded: Vec<NodeId> = chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| !included_set.contains(i))
        .map(|(_, c)| c.node_id)
        .collect();

    // Collect included chunks
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
        excluded,
        total_tokens,
        budget_utilization,
    }
}

/// Proportional budget enforcement: split budget into category pools and fill each separately.
fn enforce_budget_proportional(
    chunks: Vec<ContextChunk>,
    max_tokens: u32,
    entities_pct: f32,
    relationships_pct: f32,
    text_chunks_pct: f32,
    metadata_pct: f32,
) -> BudgetResult {
    let entities_budget = (max_tokens as f32 * entities_pct) as u32;
    let relationships_budget = (max_tokens as f32 * relationships_pct) as u32;
    let text_chunks_budget = (max_tokens as f32 * text_chunks_pct) as u32;
    let metadata_budget = (max_tokens as f32 * metadata_pct) as u32;

    // Categorize chunks by their label
    let mut entities: Vec<(usize, &ContextChunk)> = Vec::new();
    let mut relationships: Vec<(usize, &ContextChunk)> = Vec::new();
    let mut text_chunks: Vec<(usize, &ContextChunk)> = Vec::new();
    let mut metadata: Vec<(usize, &ContextChunk)> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let label_lower = chunk.label.to_lowercase();
        match label_lower.as_str() {
            "relationship" | "edge" | "relation" => relationships.push((i, chunk)),
            "text" | "chunk" | "text_chunk" | "document" => text_chunks.push((i, chunk)),
            "metadata" | "meta" | "config" => metadata.push((i, chunk)),
            _ => entities.push((i, chunk)),
        }
    }

    fn greedy_fill(
        pool: &[(usize, &ContextChunk)],
        pool_budget: u32,
    ) -> (Vec<usize>, u32) {
        let mut ranked: Vec<(usize, f32)> = pool
            .iter()
            .map(|(i, c)| {
                let density = if c.token_count == 0 {
                    f32::MAX
                } else {
                    c.relevance_score / c.token_count as f32
                };
                (*i, density)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut included = Vec::new();
        let mut used: u32 = 0;
        for (idx, _) in &ranked {
            let chunk = pool.iter().find(|(i, _)| i == idx).unwrap().1;
            if used + chunk.token_count <= pool_budget {
                used += chunk.token_count;
                included.push(*idx);
            }
        }
        (included, used)
    }

    let (ent_inc, ent_used) = greedy_fill(&entities, entities_budget);
    let (rel_inc, rel_used) = greedy_fill(&relationships, relationships_budget);
    let (txt_inc, txt_used) = greedy_fill(&text_chunks, text_chunks_budget);
    let (meta_inc, meta_used) = greedy_fill(&metadata, metadata_budget);

    let mut all_included: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for idx in ent_inc.iter().chain(rel_inc.iter()).chain(txt_inc.iter()).chain(meta_inc.iter()) {
        all_included.insert(*idx);
    }

    let total_tokens = ent_used + rel_used + txt_used + meta_used;

    let excluded: Vec<NodeId> = chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| !all_included.contains(i))
        .map(|(_, c)| c.node_id)
        .collect();

    let mut chunks_vec: Vec<Option<ContextChunk>> = chunks.into_iter().map(Some).collect();
    let mut included: Vec<ContextChunk> = all_included
        .iter()
        .filter_map(|&i| chunks_vec[i].take())
        .collect();

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
        excluded,
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
            temporal: None,
        }
    }

    #[test]
    fn test_enforce_budget_empty() {
        let result = enforce_budget(Vec::new(), &TokenBudget::new(100));
        assert!(result.included.is_empty());
        assert_eq!(result.excluded.len(), 0);
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
        assert_eq!(result.excluded.len(), 0);
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
        assert_eq!(result.excluded.len(), 1);
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
        assert_eq!(result.excluded.len(), 1);
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
        assert_eq!(result.excluded.len(), 1);
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

    fn make_chunk_with_label(node_id: u64, relevance: f32, tokens: u32, label: &str) -> ContextChunk {
        ContextChunk {
            node_id,
            content: format!("content for node {node_id}"),
            label: label.to_string(),
            relevance_score: relevance,
            depth: 0,
            token_count: tokens,
            provenance: None,
            relationships: Vec::new(),
            temporal: None,
        }
    }

    #[test]
    fn test_enforce_budget_proportional() {
        let chunks = vec![
            make_chunk_with_label(1, 0.9, 10, "entity"),
            make_chunk_with_label(2, 0.8, 10, "relationship"),
            make_chunk_with_label(3, 0.7, 10, "text_chunk"),
            make_chunk_with_label(4, 0.6, 10, "metadata"),
        ];
        let budget = TokenBudget {
            max_tokens: 100,
            allocation: TokenAllocation::Proportional {
                entities_pct: 0.4,
                relationships_pct: 0.3,
                text_chunks_pct: 0.2,
                metadata_pct: 0.1,
            },
        };
        let result = enforce_budget(chunks, &budget);
        // All chunks fit within their proportional budgets (each is only 10 tokens)
        assert_eq!(result.included.len(), 4);
        assert_eq!(result.excluded.len(), 0);
        assert_eq!(result.total_tokens, 40);
    }

    #[test]
    fn test_enforce_budget_zero_max_tokens() {
        let chunks = vec![
            make_chunk(1, 0.9, 10),
            make_chunk(2, 0.8, 20),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(0));
        assert!(result.included.is_empty());
        assert_eq!(result.excluded.len(), 2);
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.budget_utilization, 0.0);
    }

    #[test]
    fn test_enforce_budget_proportional_single_category() {
        // All chunks have the same label ("entity" category — default catch-all)
        let chunks = vec![
            make_chunk_with_label(1, 0.9, 10, "person"),
            make_chunk_with_label(2, 0.8, 10, "person"),
            make_chunk_with_label(3, 0.7, 10, "person"),
        ];
        let budget = TokenBudget {
            max_tokens: 100,
            allocation: TokenAllocation::Proportional {
                entities_pct: 0.5,
                relationships_pct: 0.2,
                text_chunks_pct: 0.2,
                metadata_pct: 0.1,
            },
        };
        let result = enforce_budget(chunks, &budget);
        // All chunks are in the entities category (50 token budget), each is 10 tokens
        assert_eq!(result.included.len(), 3);
        assert_eq!(result.total_tokens, 30);
    }

    #[test]
    fn test_enforce_budget_all_chunks_zero_tokens() {
        // All chunks have token_count=0. All should be included (infinite density).
        let chunks = vec![
            make_chunk(1, 0.9, 0),
            make_chunk(2, 0.8, 0),
            make_chunk(3, 0.7, 0),
            make_chunk(4, 0.6, 0),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(100));

        assert_eq!(
            result.included.len(),
            4,
            "All zero-token chunks should be included"
        );
        assert_eq!(result.excluded.len(), 0);
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.budget_utilization, 0.0);

        // Output should be sorted by relevance descending
        for i in 1..result.included.len() {
            assert!(
                result.included[i - 1].relevance_score >= result.included[i].relevance_score,
                "Output should be sorted by relevance descending"
            );
        }
    }

    #[test]
    fn test_enforce_budget_single_huge_chunk() {
        // One chunk with 10000 tokens, budget=100. Should be excluded.
        let chunks = vec![make_chunk(1, 0.99, 10000)];
        let result = enforce_budget(chunks, &TokenBudget::new(100));

        assert!(
            result.included.is_empty(),
            "A chunk with 10000 tokens should not fit in a 100-token budget"
        );
        assert_eq!(result.excluded.len(), 1);
        assert_eq!(result.excluded[0], 1);
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.budget_utilization, 0.0);
    }

    #[test]
    fn test_enforce_budget_equal_density_tiebreak() {
        // Multiple chunks with identical relevance/token ratio.
        // All have density = 0.5/10 = 0.05. Budget only fits 2.
        // Verify deterministic behavior (no panic, correct count).
        let chunks = vec![
            make_chunk(1, 0.5, 10),
            make_chunk(2, 0.5, 10),
            make_chunk(3, 0.5, 10),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(20));

        assert_eq!(
            result.included.len(),
            2,
            "Budget=20 should fit exactly 2 chunks of 10 tokens each"
        );
        assert_eq!(result.excluded.len(), 1);
        assert_eq!(result.total_tokens, 20);
        assert!((result.budget_utilization - 1.0).abs() < 0.001);

        // All included chunks should have equal relevance
        for chunk in &result.included {
            assert!(
                (chunk.relevance_score - 0.5).abs() < f32::EPSILON,
                "All included chunks should have relevance 0.5"
            );
        }
    }

    #[test]
    fn test_enforce_budget_one_token() {
        // Budget=1 token. Only chunks with 0 or 1 tokens fit.
        let chunks = vec![
            make_chunk(1, 0.9, 0),  // zero tokens — fits
            make_chunk(2, 0.8, 1),  // exactly 1 token — fits
            make_chunk(3, 0.7, 2),  // 2 tokens — too large
            make_chunk(4, 0.6, 10), // 10 tokens — too large
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(1));

        let included_ids: Vec<u64> = result.included.iter().map(|c| c.node_id).collect();
        assert!(
            included_ids.contains(&1),
            "Zero-token chunk should be included"
        );
        assert!(
            included_ids.contains(&2),
            "One-token chunk should fit in budget=1"
        );
        assert_eq!(
            result.included.len(),
            2,
            "Only chunks with 0 or 1 tokens should be included"
        );
        assert_eq!(result.total_tokens, 1);
        assert!((result.budget_utilization - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_enforce_budget_proportional_zero_category() {
        // Proportional with metadata_pct=0.0. Metadata chunks should be excluded.
        let chunks = vec![
            make_chunk_with_label(1, 0.9, 10, "person"),      // entity
            make_chunk_with_label(2, 0.8, 10, "relationship"), // relationship
            make_chunk_with_label(3, 0.7, 10, "text_chunk"),   // text
            make_chunk_with_label(4, 0.95, 10, "metadata"),    // metadata — budget=0
        ];
        let budget = TokenBudget {
            max_tokens: 100,
            allocation: TokenAllocation::Proportional {
                entities_pct: 0.4,
                relationships_pct: 0.3,
                text_chunks_pct: 0.3,
                metadata_pct: 0.0, // zero allocation for metadata
            },
        };
        let result = enforce_budget(chunks, &budget);

        let included_ids: Vec<u64> = result.included.iter().map(|c| c.node_id).collect();
        assert!(
            !included_ids.contains(&4),
            "Metadata chunk (node 4) should be excluded when metadata_pct=0.0"
        );
        assert!(
            included_ids.contains(&1),
            "Entity chunk should be included"
        );
        assert!(
            included_ids.contains(&2),
            "Relationship chunk should be included"
        );
        assert!(
            included_ids.contains(&3),
            "Text chunk should be included"
        );
        assert_eq!(result.included.len(), 3);
    }

    #[test]
    fn test_enforce_budget_proportional_all_same_label() {
        // All chunks have same label. They compete for one category's budget.
        // "person" maps to entities (catch-all). Budget = 100 * 0.3 = 30 for entities.
        // Each chunk is 15 tokens, so only 2 fit in 30 tokens.
        let chunks = vec![
            make_chunk_with_label(1, 0.9, 15, "person"),
            make_chunk_with_label(2, 0.8, 15, "person"),
            make_chunk_with_label(3, 0.7, 15, "person"),
        ];
        let budget = TokenBudget {
            max_tokens: 100,
            allocation: TokenAllocation::Proportional {
                entities_pct: 0.3,
                relationships_pct: 0.3,
                text_chunks_pct: 0.2,
                metadata_pct: 0.2,
            },
        };
        let result = enforce_budget(chunks, &budget);

        // entities_budget = 100 * 0.3 = 30. Each chunk=15 tokens, so 2 fit.
        assert_eq!(
            result.included.len(),
            2,
            "Only 2 of 3 chunks should fit in entities budget of 30 tokens (each 15)"
        );
        assert_eq!(result.total_tokens, 30);
        assert_eq!(result.excluded.len(), 1);
    }

    #[test]
    fn test_enforce_budget_proportional_rounding() {
        // Proportional with percentages summing to 1.05 (> 1.0).
        // Verify no panic and budget enforcement still works correctly.
        let chunks = vec![
            make_chunk_with_label(1, 0.9, 10, "person"),
            make_chunk_with_label(2, 0.8, 10, "relationship"),
            make_chunk_with_label(3, 0.7, 10, "text_chunk"),
            make_chunk_with_label(4, 0.6, 10, "metadata"),
        ];
        let budget = TokenBudget {
            max_tokens: 100,
            allocation: TokenAllocation::Proportional {
                entities_pct: 0.30,
                relationships_pct: 0.30,
                text_chunks_pct: 0.25,
                metadata_pct: 0.20, // sum = 1.05
            },
        };
        // Should NOT panic
        let result = enforce_budget(chunks, &budget);

        // Each category has enough budget for its single 10-token chunk
        // entities: 30, relationships: 30, text: 25, metadata: 20
        assert_eq!(
            result.included.len(),
            4,
            "All chunks should fit since each category budget >= 10 tokens"
        );
        assert_eq!(result.total_tokens, 40);
    }

    #[test]
    fn test_enforce_budget_negative_relevance() {
        // Chunk with negative relevance_score. Should still be handled correctly.
        let chunks = vec![
            make_chunk(1, -0.5, 10),
            make_chunk(2, 0.8, 10),
            make_chunk(3, -1.0, 5),
        ];
        let result = enforce_budget(chunks, &TokenBudget::new(100));

        // All chunks should fit within the budget
        assert_eq!(
            result.included.len(),
            3,
            "All chunks should be included regardless of negative relevance"
        );
        assert_eq!(result.total_tokens, 25);
        assert_eq!(result.excluded.len(), 0);

        // Output should be sorted by relevance descending
        // Positive (0.8) first, then negatives (-0.5, -1.0)
        assert!(
            (result.included[0].relevance_score - 0.8).abs() < f32::EPSILON,
            "Highest relevance (0.8) should be first"
        );
        assert!(
            (result.included[1].relevance_score - (-0.5)).abs() < f32::EPSILON,
            "Second should be -0.5, got {}",
            result.included[1].relevance_score
        );
        assert!(
            (result.included[2].relevance_score - (-1.0)).abs() < f32::EPSILON,
            "Last should be -1.0, got {}",
            result.included[2].relevance_score
        );
    }
}
