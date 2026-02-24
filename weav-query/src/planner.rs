//! Query planner that converts a `ContextQuery` into an ordered execution plan.

use weav_core::types::{ConflictPolicy, DecayFunction, Direction, Timestamp, TokenBudget};

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
    /// Extract paths through the graph.
    PathExtraction {
        max_paths: u32,
        max_length: u8,
    },
    /// Detect conflicting information across nodes.
    ConflictDetection {
        policy: ConflictPolicy,
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

    // Step 4b: Path extraction (when multiple node seeds)
    match &query.seeds {
        SeedStrategy::Nodes(keys) if keys.len() >= 2 => {
            steps.push(PlanStep::PathExtraction {
                max_paths: 10,
                max_length: query.max_depth,
            });
        }
        SeedStrategy::Both { node_keys, .. } if node_keys.len() >= 2 => {
            steps.push(PlanStep::PathExtraction {
                max_paths: 10,
                max_length: query.max_depth,
            });
        }
        _ => {}
    }

    // Step 5: Temporal filter (if set)
    if let Some(ts) = query.temporal_at {
        steps.push(PlanStep::TemporalFilter { timestamp: ts });
    }

    // Step 6: Conflict detection (always, with default policy)
    steps.push(PlanStep::ConflictDetection {
        policy: ConflictPolicy::default(),
    });

    // Step 7: Relevance score (with decay)
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
    use crate::parser::{SortDirection, SortField, SortOrder};
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
            sort: None,
        }
    }

    #[test]
    fn test_plan_basic_node_seeds() {
        let query = make_basic_query();
        let plan = plan_context_query(&query);

        // Should have: NodeLookup, GraphTraversal, FlowScore, ConflictDetection, RelevanceScore, FormatContext
        assert_eq!(plan.steps.len(), 6);
        assert!(matches!(&plan.steps[0], PlanStep::NodeLookup { .. }));
        assert!(matches!(&plan.steps[1], PlanStep::GraphTraversal { .. }));
        assert!(matches!(&plan.steps[2], PlanStep::FlowScore { .. }));
        assert!(matches!(&plan.steps[3], PlanStep::ConflictDetection { .. }));
        assert!(matches!(&plan.steps[4], PlanStep::RelevanceScore { .. }));
        assert!(matches!(&plan.steps[5], PlanStep::FormatContext { .. }));
    }

    #[test]
    fn test_plan_vector_seeds() {
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Vector {
            embedding: vec![1.0, 0.0, 0.0],
            top_k: 5,
        };
        let plan = plan_context_query(&query);

        // Should have: VectorSearch, GraphTraversal, FlowScore, ConflictDetection, RelevanceScore, FormatContext
        assert_eq!(plan.steps.len(), 6);
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

        // Should have: VectorSearch, NodeLookup, GraphTraversal, FlowScore, ConflictDetection, RelevanceScore, FormatContext
        assert_eq!(plan.steps.len(), 7);
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
    fn test_plan_step_path_extraction_exists() {
        // Verify PlanStep::PathExtraction can be constructed
        let step = PlanStep::PathExtraction {
            max_paths: 5,
            max_length: 3,
        };
        assert!(matches!(
            step,
            PlanStep::PathExtraction {
                max_paths: 5,
                max_length: 3,
            }
        ));
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
            sort: None,
        };

        let plan = plan_context_query(&query);

        // Should have all 9 steps:
        // VectorSearch, NodeLookup, GraphTraversal, FlowScore,
        // TemporalFilter, ConflictDetection, RelevanceScore, TokenBudgetEnforce, FormatContext
        assert_eq!(plan.steps.len(), 9);
        assert!(matches!(&plan.steps[0], PlanStep::VectorSearch { .. }));
        assert!(matches!(&plan.steps[1], PlanStep::NodeLookup { .. }));
        assert!(matches!(&plan.steps[2], PlanStep::GraphTraversal { .. }));
        assert!(matches!(&plan.steps[3], PlanStep::FlowScore { .. }));
        assert!(matches!(&plan.steps[4], PlanStep::TemporalFilter { .. }));
        assert!(matches!(&plan.steps[5], PlanStep::ConflictDetection { .. }));
        assert!(matches!(&plan.steps[6], PlanStep::RelevanceScore { .. }));
        assert!(matches!(&plan.steps[7], PlanStep::TokenBudgetEnforce { .. }));
        assert!(matches!(&plan.steps[8], PlanStep::FormatContext { .. }));
    }

    #[test]
    fn test_plan_path_extraction_for_multi_seeds() {
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Nodes(vec!["key1".to_string(), "key2".to_string()]);
        let plan = plan_context_query(&query);

        let has_path_extraction = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::PathExtraction { .. }));
        assert!(
            has_path_extraction,
            "Should have PathExtraction for 2+ node seeds"
        );

        // Verify PathExtraction has correct params
        let path_step = plan.steps.iter().find_map(|s| match s {
            PlanStep::PathExtraction {
                max_paths,
                max_length,
            } => Some((*max_paths, *max_length)),
            _ => None,
        });
        assert_eq!(path_step, Some((10, 2)));
    }

    // ── Round 7 edge-case tests ─────────────────────────────────────────────

    #[test]
    fn test_plan_empty_vector_seeds() {
        // Seeds with empty embedding and top_k=0
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Vector {
            embedding: vec![],
            top_k: 0,
        };
        let plan = plan_context_query(&query);

        // Should still include a VectorSearch step even with empty embedding
        let has_vector_search = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::VectorSearch { .. }));
        assert!(
            has_vector_search,
            "VectorSearch step should be present even with empty embedding"
        );

        // Verify the VectorSearch has k=0 and empty query_vector
        let vs = plan.steps.iter().find_map(|s| match s {
            PlanStep::VectorSearch { query_vector, k } => Some((query_vector.clone(), *k)),
            _ => None,
        });
        assert_eq!(vs, Some((vec![], 0)));
    }

    #[test]
    fn test_plan_both_seeds_empty() {
        // Both seeds with empty embedding and empty keys
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Both {
            embedding: vec![],
            top_k: 0,
            node_keys: vec![],
        };
        let plan = plan_context_query(&query);

        // VectorSearch should be present (Both always adds VectorSearch)
        let has_vector_search = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::VectorSearch { .. }));
        assert!(has_vector_search, "VectorSearch should be present for Both seeds");

        // NodeLookup should NOT be present (empty node_keys are skipped)
        let has_node_lookup = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::NodeLookup { .. }));
        assert!(
            !has_node_lookup,
            "NodeLookup should NOT be present when node_keys is empty"
        );

        // GraphTraversal should always be present
        let has_traversal = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::GraphTraversal { .. }));
        assert!(has_traversal, "GraphTraversal should always be present");
    }

    #[test]
    fn test_plan_max_depth_zero() {
        // Query with max_depth=0 should still include GraphTraversal
        let mut query = make_basic_query();
        query.max_depth = 0;
        let plan = plan_context_query(&query);

        let traversal = plan.steps.iter().find_map(|s| match s {
            PlanStep::GraphTraversal { max_depth, .. } => Some(*max_depth),
            _ => None,
        });
        assert_eq!(
            traversal,
            Some(0),
            "GraphTraversal should be present with max_depth=0"
        );

        // FlowScore should also reflect max_depth=0
        let flow = plan.steps.iter().find_map(|s| match s {
            PlanStep::FlowScore { max_depth, .. } => Some(*max_depth),
            _ => None,
        });
        assert_eq!(flow, Some(0), "FlowScore should reflect max_depth=0");
    }

    #[test]
    fn test_plan_all_options_combined() {
        // Query with every option set
        let query = ContextQuery {
            query_text: Some("all options".to_string()),
            graph: "g".to_string(),
            budget: Some(TokenBudget::new(8192)),
            seeds: SeedStrategy::Both {
                embedding: vec![0.1, 0.2, 0.3],
                top_k: 20,
                node_keys: vec!["k1".to_string(), "k2".to_string(), "k3".to_string()],
            },
            max_depth: 5,
            direction: Direction::Outgoing,
            edge_filter: Some(EdgeFilterConfig {
                labels: Some(vec!["related".to_string(), "depends_on".to_string()]),
                min_weight: Some(0.3),
                min_confidence: Some(0.7),
            }),
            decay: Some(DecayFunction::Step { cutoff_ms: 86400000 }),
            include_provenance: true,
            temporal_at: Some(999999),
            limit: Some(100),
            sort: Some(SortOrder {
                field: SortField::Recency,
                direction: SortDirection::Asc,
            }),
        };

        let plan = plan_context_query(&query);

        // Should have all 10 steps:
        // VectorSearch, NodeLookup, GraphTraversal, FlowScore, PathExtraction,
        // TemporalFilter, ConflictDetection, RelevanceScore, TokenBudgetEnforce, FormatContext
        assert_eq!(plan.steps.len(), 10, "all options combined should produce 10 steps");
        assert!(matches!(&plan.steps[0], PlanStep::VectorSearch { k: 20, .. }));
        assert!(matches!(&plan.steps[1], PlanStep::NodeLookup { .. }));
        assert!(matches!(&plan.steps[2], PlanStep::GraphTraversal { max_depth: 5, .. }));
        assert!(matches!(&plan.steps[3], PlanStep::FlowScore { max_depth: 5, .. }));
        assert!(matches!(&plan.steps[4], PlanStep::PathExtraction { max_paths: 10, max_length: 5 }));
        assert!(matches!(&plan.steps[5], PlanStep::TemporalFilter { timestamp: 999999 }));
        assert!(matches!(&plan.steps[6], PlanStep::ConflictDetection { .. }));
        assert!(matches!(&plan.steps[7], PlanStep::RelevanceScore { decay: Some(DecayFunction::Step { .. }) }));
        assert!(matches!(&plan.steps[8], PlanStep::TokenBudgetEnforce { .. }));
        assert!(matches!(&plan.steps[9], PlanStep::FormatContext { include_provenance: true }));
    }

    #[test]
    fn test_plan_single_node_seed_no_path_extraction() {
        // Single node key seed — PathExtraction requires 2+ seeds
        let mut query = make_basic_query();
        query.seeds = SeedStrategy::Nodes(vec!["only_one".to_string()]);
        let plan = plan_context_query(&query);

        let has_path_extraction = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::PathExtraction { .. }));
        assert!(
            !has_path_extraction,
            "PathExtraction should NOT be generated for a single node seed"
        );
    }

    #[test]
    fn test_plan_conflict_detection_always_present() {
        // Verify ConflictDetection is present for various query shapes

        // 1. Basic node seeds
        let plan1 = plan_context_query(&make_basic_query());
        assert!(
            plan1.steps.iter().any(|s| matches!(s, PlanStep::ConflictDetection { .. })),
            "ConflictDetection should be present for basic node seeds"
        );

        // 2. Vector seeds
        let mut q2 = make_basic_query();
        q2.seeds = SeedStrategy::Vector {
            embedding: vec![1.0],
            top_k: 5,
        };
        let plan2 = plan_context_query(&q2);
        assert!(
            plan2.steps.iter().any(|s| matches!(s, PlanStep::ConflictDetection { .. })),
            "ConflictDetection should be present for vector seeds"
        );

        // 3. Empty seeds
        let mut q3 = make_basic_query();
        q3.seeds = SeedStrategy::Nodes(vec![]);
        let plan3 = plan_context_query(&q3);
        assert!(
            plan3.steps.iter().any(|s| matches!(s, PlanStep::ConflictDetection { .. })),
            "ConflictDetection should be present even with empty seeds"
        );

        // 4. With budget and temporal
        let mut q4 = make_basic_query();
        q4.budget = Some(TokenBudget::new(1000));
        q4.temporal_at = Some(5000);
        let plan4 = plan_context_query(&q4);
        assert!(
            plan4.steps.iter().any(|s| matches!(s, PlanStep::ConflictDetection { .. })),
            "ConflictDetection should be present with budget and temporal"
        );
    }
}
