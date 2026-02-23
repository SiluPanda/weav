//! Query planner that converts a `ContextQuery` into an ordered execution plan.

use weav_core::types::{DecayFunction, Direction, Timestamp, TokenBudget};

use crate::parser::{ContextQuery, EdgeFilterConfig, SeedStrategy};

// ─── Plan types ─────────────────────────────────────────────────────────────

/// An ordered list of execution steps produced by the planner.
#[derive(Debug)]
pub struct QueryPlan {
    pub steps: Vec<PlanStep>,
}

/// A single step in the query execution plan.
#[derive(Debug)]
pub enum PlanStep {
    /// Perform a vector similarity search to find seed nodes.
    VectorSearch {
        query_vector: Vec<f32>,
        k: u16,
    },
    /// Look up nodes by their entity keys.
    NodeLookup {
        node_keys: Vec<String>,
    },
    /// Traverse the graph from seed nodes.
    GraphTraversal {
        max_depth: u8,
        direction: Direction,
        edge_filter: Option<EdgeFilterConfig>,
    },
    /// Propagate scores from seeds using flow scoring.
    FlowScore {
        alpha: f32,
        theta: f32,
        max_depth: u8,
    },
    /// Filter results by temporal validity.
    TemporalFilter {
        timestamp: Timestamp,
    },
    /// Apply relevance decay based on age.
    RelevanceScore {
        decay: Option<DecayFunction>,
    },
    /// Enforce the token budget constraint.
    TokenBudgetEnforce {
        budget: TokenBudget,
    },
    /// Format the final context output.
    FormatContext {
        include_provenance: bool,
    },
}

// ─── Planner logic ──────────────────────────────────────────────────────────

/// Convert a `ContextQuery` into an ordered `QueryPlan`.
///
/// The plan steps are ordered as follows:
/// 1. VectorSearch (if seeds have vectors)
/// 2. NodeLookup (if seeds have node keys)
/// 3. GraphTraversal (always)
/// 4. FlowScore (always, with defaults alpha=0.5, theta=0.01)
/// 5. TemporalFilter (if temporal_at is set)
/// 6. RelevanceScore (with decay)
/// 7. TokenBudgetEnforce (if budget is set)
/// 8. FormatContext (always)
pub fn plan_context_query(query: &ContextQuery) -> QueryPlan {
    let mut steps = Vec::new();

    // Step 1: Vector search if seeds contain an embedding
    match &query.seeds {
        SeedStrategy::Vector { embedding, top_k } => {
            steps.push(PlanStep::VectorSearch {
                query_vector: embedding.clone(),
                k: *top_k,
            });
        }
        SeedStrategy::Both {
            embedding, top_k, ..
        } => {
            steps.push(PlanStep::VectorSearch {
                query_vector: embedding.clone(),
                k: *top_k,
            });
        }
        SeedStrategy::Nodes(_) => {}
    }

    // Step 2: Node lookup if seeds contain node keys
    match &query.seeds {
        SeedStrategy::Nodes(keys) if !keys.is_empty() => {
            steps.push(PlanStep::NodeLookup {
                node_keys: keys.clone(),
            });
        }
        SeedStrategy::Both { node_keys, .. } if !node_keys.is_empty() => {
            steps.push(PlanStep::NodeLookup {
                node_keys: node_keys.clone(),
            });
        }
        _ => {}
    }

    // Step 3: Graph traversal (always)
    steps.push(PlanStep::GraphTraversal {
        max_depth: query.max_depth,
        direction: query.direction,
        edge_filter: query.edge_filter.clone(),
    });

    // Step 4: Flow score (always, with defaults)
    steps.push(PlanStep::FlowScore {
        alpha: 0.5,
        theta: 0.01,
        max_depth: query.max_depth,
    });

    // Step 5: Temporal filter (if set)
    if let Some(ts) = query.temporal_at {
        steps.push(PlanStep::TemporalFilter { timestamp: ts });
    }

    // Step 6: Relevance score (with decay)
    steps.push(PlanStep::RelevanceScore {
        decay: query.decay.clone(),
    });

    // Step 7: Token budget enforcement (if set)
    if let Some(ref budget) = query.budget {
        steps.push(PlanStep::TokenBudgetEnforce {
            budget: budget.clone(),
        });
    }

    // Step 8: Format context (always)
    steps.push(PlanStep::FormatContext {
        include_provenance: query.include_provenance,
    });

    QueryPlan { steps }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use weav_core::types::TokenBudget;

    fn make_basic_query() -> ContextQuery {
        ContextQuery {
            query_text: Some("test query".to_string()),
            graph: "test_graph".to_string(),
            budget: None,
            seeds: SeedStrategy::Nodes(vec!["key1".to_string()]),
            max_depth: 2,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
        }
    }

    #[test]
    fn test_plan_basic_node_seeds() {
        let query = make_basic_query();
        let plan = plan_context_query(&query);

        // Should have: NodeLookup, GraphTraversal, FlowScore, RelevanceScore, FormatContext
        assert_eq!(plan.steps.len(), 5);
        assert!(matches!(&plan.steps[0], PlanStep::NodeLookup { .. }));
        assert!(matches!(&plan.steps[1], PlanStep::GraphTraversal { .. }));
        assert!(matches!(&plan.steps[2], PlanStep::FlowScore { .. }));
        assert!(matches!(&plan.steps[3], PlanStep::RelevanceScore { .. }));
        assert!(matches!(&plan.steps[4], PlanStep::FormatContext { .. }));
    }

    #[test]
    fn test_plan_vector_seeds() {
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Vector {
            embedding: vec![1.0, 0.0, 0.0],
            top_k: 5,
        };
        let plan = plan_context_query(&query);

        // Should have: VectorSearch, GraphTraversal, FlowScore, RelevanceScore, FormatContext
        assert_eq!(plan.steps.len(), 5);
        assert!(matches!(&plan.steps[0], PlanStep::VectorSearch { k: 5, .. }));
    }

    #[test]
    fn test_plan_both_seeds() {
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Both {
            embedding: vec![1.0, 0.0],
            top_k: 10,
            node_keys: vec!["k1".to_string()],
        };
        let plan = plan_context_query(&query);

        // Should have: VectorSearch, NodeLookup, GraphTraversal, FlowScore, RelevanceScore, FormatContext
        assert_eq!(plan.steps.len(), 6);
        assert!(matches!(&plan.steps[0], PlanStep::VectorSearch { .. }));
        assert!(matches!(&plan.steps[1], PlanStep::NodeLookup { .. }));
    }

    #[test]
    fn test_plan_with_budget() {
        let mut query = make_basic_query();
        query.budget = Some(TokenBudget::new(4096));
        let plan = plan_context_query(&query);

        // Should include TokenBudgetEnforce step
        let has_budget = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::TokenBudgetEnforce { .. }));
        assert!(has_budget);
    }

    #[test]
    fn test_plan_with_temporal() {
        let mut query = make_basic_query();
        query.temporal_at = Some(1000);
        let plan = plan_context_query(&query);

        let has_temporal = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::TemporalFilter { timestamp: 1000 }));
        assert!(has_temporal);
    }

    #[test]
    fn test_plan_with_decay() {
        let mut query = make_basic_query();
        query.decay = Some(DecayFunction::Exponential {
            half_life_ms: 3600000,
        });
        let plan = plan_context_query(&query);

        let has_decay = plan.steps.iter().any(|s| {
            matches!(
                s,
                PlanStep::RelevanceScore {
                    decay: Some(DecayFunction::Exponential { .. })
                }
            )
        });
        assert!(has_decay);
    }

    #[test]
    fn test_plan_with_provenance() {
        let mut query = make_basic_query();
        query.include_provenance = true;
        let plan = plan_context_query(&query);

        let has_prov = plan.steps.iter().any(|s| {
            matches!(
                s,
                PlanStep::FormatContext {
                    include_provenance: true
                }
            )
        });
        assert!(has_prov);
    }

    #[test]
    fn test_plan_empty_node_seeds() {
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Nodes(vec![]);
        let plan = plan_context_query(&query);

        // Empty node list should NOT produce a NodeLookup step
        let has_lookup = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::NodeLookup { .. }));
        assert!(!has_lookup);
    }

    #[test]
    fn test_plan_flow_score_defaults() {
        let query = make_basic_query();
        let plan = plan_context_query(&query);

        let flow = plan.steps.iter().find_map(|s| match s {
            PlanStep::FlowScore {
                alpha,
                theta,
                max_depth,
            } => Some((*alpha, *theta, *max_depth)),
            _ => None,
        });
        assert_eq!(flow, Some((0.5, 0.01, 2)));
    }

    #[test]
    fn test_plan_full_query() {
        let query = ContextQuery {
            query_text: Some("full test".to_string()),
            graph: "g".to_string(),
            budget: Some(TokenBudget::new(2048)),
            seeds: SeedStrategy::Both {
                embedding: vec![0.1, 0.2],
                top_k: 5,
                node_keys: vec!["k".to_string()],
            },
            max_depth: 3,
            direction: Direction::Outgoing,
            edge_filter: Some(EdgeFilterConfig {
                labels: Some(vec!["knows".to_string()]),
                min_weight: Some(0.5),
                min_confidence: None,
            }),
            decay: Some(DecayFunction::Linear { max_age_ms: 1000 }),
            include_provenance: true,
            temporal_at: Some(5000),
            limit: Some(20),
        };

        let plan = plan_context_query(&query);

        // Should have all 8 steps:
        // VectorSearch, NodeLookup, GraphTraversal, FlowScore,
        // TemporalFilter, RelevanceScore, TokenBudgetEnforce, FormatContext
        assert_eq!(plan.steps.len(), 8);
        assert!(matches!(&plan.steps[0], PlanStep::VectorSearch { .. }));
        assert!(matches!(&plan.steps[1], PlanStep::NodeLookup { .. }));
        assert!(matches!(&plan.steps[2], PlanStep::GraphTraversal { .. }));
        assert!(matches!(&plan.steps[3], PlanStep::FlowScore { .. }));
        assert!(matches!(&plan.steps[4], PlanStep::TemporalFilter { .. }));
        assert!(matches!(&plan.steps[5], PlanStep::RelevanceScore { .. }));
        assert!(matches!(&plan.steps[6], PlanStep::TokenBudgetEnforce { .. }));
        assert!(matches!(&plan.steps[7], PlanStep::FormatContext { .. }));
    }
}
