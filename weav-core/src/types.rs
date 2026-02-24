//! Core type definitions for Weav.
//!
//! All fundamental types used across the engine are defined here.

use compact_str::CompactString;
use serde::{Deserialize, Serialize};

// ─── Identifiers ───────────────────────────────────────────────────────────

/// Unique node identifier within a graph (auto-incrementing).
pub type NodeId = u64;

/// Unique edge identifier within a graph.
pub type EdgeId = u64;

/// Graph identifier (max 4B graphs).
pub type GraphId = u32;

/// Shard identifier.
pub type ShardId = u16;

/// Compact label encoding for node/edge types.
pub type LabelId = u16;

/// Compact property key encoding.
pub type PropertyKeyId = u16;

/// Milliseconds since Unix epoch.
pub type Timestamp = u64;

// ─── Bi-Temporal ───────────────────────────────────────────────────────────

/// Bi-temporal metadata for tracking both real-world validity and database
/// transaction time. An open interval uses `u64::MAX` as the sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiTemporal {
    /// When this fact became true in the real world.
    pub valid_from: Timestamp,
    /// When this fact stopped being true (`u64::MAX` = still valid).
    pub valid_until: Timestamp,
    /// When this was written to the database.
    pub tx_from: Timestamp,
    /// When this was superseded in the database (`u64::MAX` = current).
    pub tx_until: Timestamp,
}

impl BiTemporal {
    /// A sentinel value representing "still current / no end".
    pub const OPEN: Timestamp = u64::MAX;

    /// Create a new bi-temporal that is currently valid starting at `now`.
    pub fn new_current(now: Timestamp) -> Self {
        Self {
            valid_from: now,
            valid_until: Self::OPEN,
            tx_from: now,
            tx_until: Self::OPEN,
        }
    }

    /// Returns `true` if this record is valid at the given real-world timestamp.
    pub fn is_valid_at(&self, ts: Timestamp) -> bool {
        self.valid_from <= ts && ts < self.valid_until
    }

    /// Returns `true` if this record is current in the database at the given
    /// transaction timestamp.
    pub fn is_current_at(&self, tx_ts: Timestamp) -> bool {
        self.tx_from <= tx_ts && tx_ts < self.tx_until
    }

    /// Returns `true` if this record is both valid and current right now
    /// (i.e. both `valid_until` and `tx_until` are open).
    pub fn is_active(&self) -> bool {
        self.valid_until == Self::OPEN && self.tx_until == Self::OPEN
    }

    /// Invalidate this record at the given timestamp (sets `valid_until`).
    pub fn invalidate(&mut self, at: Timestamp) {
        self.valid_until = at;
    }

    /// Supersede this record in the database at the given transaction time.
    pub fn supersede(&mut self, tx_ts: Timestamp) {
        self.tx_until = tx_ts;
    }
}

impl Default for BiTemporal {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self::new_current(now)
    }
}

// ─── Value ─────────────────────────────────────────────────────────────────

/// Dynamic value type for node/edge properties.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(CompactString),
    Bytes(Vec<u8>),
    Vector(Vec<f32>),
    List(Vec<Value>),
    Map(Vec<(CompactString, Value)>),
    Timestamp(u64),
}

impl Value {
    /// Returns the type name as a static string.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::Vector(_) => "vector",
            Value::List(_) => "list",
            Value::Map(_) => "map",
            Value::Timestamp(_) => "timestamp",
        }
    }

    /// Try to extract a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to extract a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to extract a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract a vector reference.
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Value::Vector(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// The expected type of values in a property column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Null,
    Bool,
    Int,
    Float,
    String,
    Bytes,
    Vector,
    List,
    Map,
    Timestamp,
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Null => ValueType::Null,
            Value::Bool(_) => ValueType::Bool,
            Value::Int(_) => ValueType::Int,
            Value::Float(_) => ValueType::Float,
            Value::String(_) => ValueType::String,
            Value::Bytes(_) => ValueType::Bytes,
            Value::Vector(_) => ValueType::Vector,
            Value::List(_) => ValueType::List,
            Value::Map(_) => ValueType::Map,
            Value::Timestamp(_) => ValueType::Timestamp,
        }
    }
}

// ─── Provenance ────────────────────────────────────────────────────────────

/// Provenance metadata tracking the origin and confidence of a fact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source identifier (e.g., "gpt-4-turbo", "user-input", "sec-filing-10k").
    pub source: CompactString,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// How this fact was produced.
    pub extraction_method: ExtractionMethod,
    /// Optional reference to the source document.
    pub source_document_id: Option<CompactString>,
    /// Optional byte offset within the source document.
    pub source_chunk_offset: Option<u32>,
}

impl Provenance {
    pub fn new(source: impl Into<CompactString>, confidence: f32) -> Self {
        Self {
            source: source.into(),
            confidence: confidence.clamp(0.0, 1.0),
            extraction_method: ExtractionMethod::UserProvided,
            source_document_id: None,
            source_chunk_offset: None,
        }
    }
}

/// How a fact was extracted / produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionMethod {
    /// Extracted by a large language model.
    LlmExtracted,
    /// Extracted by a traditional NLP pipeline.
    NlpPipeline,
    /// Provided directly by a user.
    UserProvided,
    /// Derived / computed from other facts.
    Derived,
    /// Bulk imported from an external source.
    Imported,
}

// ─── Token Budget ──────────────────────────────────────────────────────────

/// Specification for the maximum token budget of a context query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenBudget {
    pub max_tokens: u32,
    pub allocation: TokenAllocation,
}

impl TokenBudget {
    pub fn new(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            allocation: TokenAllocation::Auto,
        }
    }
}

/// How the token budget is allocated across content categories.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TokenAllocation {
    /// Proportional allocation by percentage.
    Proportional {
        entities_pct: f32,
        relationships_pct: f32,
        text_chunks_pct: f32,
        metadata_pct: f32,
    },
    /// Priority-ordered: fill higher-priority categories first.
    Priority(Vec<ContentPriority>),
    /// Engine decides based on query shape.
    Auto,
}

/// Content categories for priority-based token allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentPriority {
    Entities,
    Relationships,
    TextChunks,
    Metadata,
}

// ─── Decay Functions ───────────────────────────────────────────────────────

/// Relevance decay function applied to scores based on age.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DecayFunction {
    /// No decay — score is unchanged.
    None,
    /// Exponential decay: `score * 0.5^(age / half_life)`.
    Exponential { half_life_ms: u64 },
    /// Linear decay: `score * max(0, 1 - age / max_age)`.
    Linear { max_age_ms: u64 },
    /// Step function: full score if `age < cutoff`, else 0.
    Step { cutoff_ms: u64 },
    /// Piecewise-linear decay defined by breakpoints `(age_ms, multiplier)`.
    Custom { breakpoints: Vec<(u64, f32)> },
}

impl DecayFunction {
    /// Apply the decay function to a score given the current time and the
    /// item's timestamp.
    pub fn apply(&self, score: f32, item_ts: Timestamp, now: Timestamp) -> f32 {
        if now <= item_ts {
            return score;
        }
        let age = now - item_ts;
        match self {
            DecayFunction::None => score,
            DecayFunction::Exponential { half_life_ms } => {
                if *half_life_ms == 0 {
                    return 0.0;
                }
                score * (0.5_f32).powf(age as f32 / *half_life_ms as f32)
            }
            DecayFunction::Linear { max_age_ms } => {
                if *max_age_ms == 0 {
                    return 0.0;
                }
                score * (1.0 - (age as f32 / *max_age_ms as f32)).max(0.0)
            }
            DecayFunction::Step { cutoff_ms } => {
                if age < *cutoff_ms {
                    score
                } else {
                    0.0
                }
            }
            DecayFunction::Custom { breakpoints } => {
                if breakpoints.is_empty() {
                    return score;
                }
                // Find the two breakpoints that bracket `age` and interpolate.
                let mut prev = (0u64, 1.0f32);
                for &(bp_age, bp_mult) in breakpoints {
                    if age <= bp_age {
                        // Interpolate between prev and this breakpoint.
                        if bp_age == prev.0 {
                            return score * bp_mult;
                        }
                        let t = (age - prev.0) as f32 / (bp_age - prev.0) as f32;
                        return score * (prev.1 + t * (bp_mult - prev.1));
                    }
                    prev = (bp_age, bp_mult);
                }
                // Past the last breakpoint: use the last multiplier.
                score * prev.1
            }
        }
    }
}

impl Default for DecayFunction {
    fn default() -> Self {
        DecayFunction::None
    }
}

// ─── Conflict Resolution ───────────────────────────────────────────────────

/// Policy for resolving conflicting writes to the same entity/property.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// Most recent write wins.
    LastWriteWins,
    /// Highest confidence provenance wins.
    HighestConfidence,
    /// Mark old as invalid, keep both (Zep-style).
    TemporalInvalidation,
    /// Attempt to merge properties.
    Merge,
    /// Reject conflicting writes.
    Reject,
}

impl Default for ConflictPolicy {
    fn default() -> Self {
        ConflictPolicy::LastWriteWins
    }
}

// ─── Node / Edge data (for insertion) ──────────────────────────────────────

/// Input data for creating a new node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeData {
    pub label: CompactString,
    pub properties: Vec<(CompactString, Value)>,
    pub embedding: Option<Vec<f32>>,
    pub entity_key: Option<CompactString>,
    pub provenance: Option<Provenance>,
}

/// Input data for creating a new edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeData {
    pub source: NodeId,
    pub target: NodeId,
    pub label: CompactString,
    pub properties: Vec<(CompactString, Value)>,
    pub weight: f32,
    pub provenance: Option<Provenance>,
}

impl Default for EdgeData {
    fn default() -> Self {
        Self {
            source: 0,
            target: 0,
            label: CompactString::default(),
            properties: Vec::new(),
            weight: 1.0,
            provenance: None,
        }
    }
}

// ─── Direction ─────────────────────────────────────────────────────────────

/// Direction for graph traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

// ─── Scored results ────────────────────────────────────────────────────────

/// A node with an associated relevance score.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredNode {
    pub node_id: NodeId,
    pub score: f32,
    pub depth: u8,
}

/// A scored path through the graph.
#[derive(Clone, Debug)]
pub struct ScoredPath {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub reliability: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitemporal_new_current() {
        let bt = BiTemporal::new_current(1000);
        assert_eq!(bt.valid_from, 1000);
        assert_eq!(bt.valid_until, BiTemporal::OPEN);
        assert_eq!(bt.tx_from, 1000);
        assert_eq!(bt.tx_until, BiTemporal::OPEN);
        assert!(bt.is_active());
    }

    #[test]
    fn test_bitemporal_valid_at() {
        let bt = BiTemporal {
            valid_from: 100,
            valid_until: 200,
            tx_from: 100,
            tx_until: BiTemporal::OPEN,
        };
        assert!(bt.is_valid_at(100));
        assert!(bt.is_valid_at(150));
        assert!(!bt.is_valid_at(200));
        assert!(!bt.is_valid_at(50));
    }

    #[test]
    fn test_bitemporal_invalidate() {
        let mut bt = BiTemporal::new_current(100);
        assert!(bt.is_active());
        bt.invalidate(200);
        assert!(!bt.is_active());
        assert!(bt.is_valid_at(150));
        assert!(!bt.is_valid_at(250));
    }

    #[test]
    fn test_value_accessors() {
        assert_eq!(Value::Int(42).as_int(), Some(42));
        assert_eq!(Value::Float(3.14).as_float(), Some(3.14));
        assert_eq!(Value::Int(42).as_float(), Some(42.0));
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::String("hello".into()).as_str(), Some("hello"));
        assert_eq!(Value::Null.as_int(), None);

        let v = Value::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.as_vector(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn test_value_type_name() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Bool(true).type_name(), "bool");
        assert_eq!(Value::Int(0).type_name(), "int");
        assert_eq!(Value::Float(0.0).type_name(), "float");
        assert_eq!(Value::String("".into()).type_name(), "string");
    }

    #[test]
    fn test_provenance_confidence_clamping() {
        let p = Provenance::new("test", 1.5);
        assert_eq!(p.confidence, 1.0);
        let p2 = Provenance::new("test", -0.5);
        assert_eq!(p2.confidence, 0.0);
    }

    #[test]
    fn test_decay_none() {
        let d = DecayFunction::None;
        assert_eq!(d.apply(1.0, 0, 1000), 1.0);
    }

    #[test]
    fn test_decay_exponential() {
        let d = DecayFunction::Exponential {
            half_life_ms: 1000,
        };
        let score = d.apply(1.0, 0, 1000);
        assert!((score - 0.5).abs() < 0.001);

        let score2 = d.apply(1.0, 0, 2000);
        assert!((score2 - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_decay_linear() {
        let d = DecayFunction::Linear { max_age_ms: 1000 };
        assert!((d.apply(1.0, 0, 500) - 0.5).abs() < 0.001);
        assert_eq!(d.apply(1.0, 0, 1000), 0.0);
        assert_eq!(d.apply(1.0, 0, 2000), 0.0);
    }

    #[test]
    fn test_decay_step() {
        let d = DecayFunction::Step { cutoff_ms: 1000 };
        assert_eq!(d.apply(1.0, 0, 500), 1.0);
        assert_eq!(d.apply(1.0, 0, 1000), 0.0);
        assert_eq!(d.apply(1.0, 0, 2000), 0.0);
    }

    #[test]
    fn test_decay_custom() {
        let d = DecayFunction::Custom {
            breakpoints: vec![(500, 0.5), (1000, 0.0)],
        };
        // At age 0: score = 1.0 (before first breakpoint)
        assert_eq!(d.apply(1.0, 1000, 1000), 1.0);
        // At age 250: interpolate (0,1.0) → (500,0.5) → 0.75
        let score = d.apply(1.0, 0, 250);
        assert!((score - 0.75).abs() < 0.001);
        // At age 500: 0.5
        assert!((d.apply(1.0, 0, 500) - 0.5).abs() < 0.001);
        // At age 750: interpolate (500,0.5) → (1000,0.0) → 0.25
        assert!((d.apply(1.0, 0, 750) - 0.25).abs() < 0.001);
        // Past last breakpoint: use last multiplier (0.0)
        assert_eq!(d.apply(1.0, 0, 2000), 0.0);
    }

    #[test]
    fn test_token_budget_default() {
        let budget = TokenBudget::new(4096);
        assert_eq!(budget.max_tokens, 4096);
        assert_eq!(budget.allocation, TokenAllocation::Auto);
    }

    #[test]
    fn test_bitemporal_is_current_at() {
        let bt = BiTemporal {
            valid_from: 100,
            valid_until: BiTemporal::OPEN,
            tx_from: 500,
            tx_until: 1000,
        };
        // Before tx_from
        assert!(!bt.is_current_at(499));
        // At tx_from (inclusive)
        assert!(bt.is_current_at(500));
        // Within range
        assert!(bt.is_current_at(750));
        // At tx_until (exclusive)
        assert!(!bt.is_current_at(1000));
        // After tx_until
        assert!(!bt.is_current_at(1500));

        // A currently-open record
        let bt_open = BiTemporal::new_current(100);
        assert!(bt_open.is_current_at(100));
        assert!(bt_open.is_current_at(u64::MAX - 1));
    }

    #[test]
    fn test_bitemporal_supersede() {
        let mut bt = BiTemporal::new_current(100);
        assert!(bt.is_active());
        assert_eq!(bt.tx_until, BiTemporal::OPEN);

        bt.supersede(500);
        assert_eq!(bt.tx_until, 500);
        assert!(!bt.is_active());
        // Still current at 499
        assert!(bt.is_current_at(499));
        // No longer current at 500
        assert!(!bt.is_current_at(500));
    }

    #[test]
    fn test_value_type_name_all_variants() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Bool(false).type_name(), "bool");
        assert_eq!(Value::Int(0).type_name(), "int");
        assert_eq!(Value::Float(0.0).type_name(), "float");
        assert_eq!(Value::String("x".into()).type_name(), "string");
        assert_eq!(Value::Bytes(vec![1, 2]).type_name(), "bytes");
        assert_eq!(Value::Vector(vec![1.0]).type_name(), "vector");
        assert_eq!(Value::List(vec![Value::Null]).type_name(), "list");
        assert_eq!(
            Value::Map(vec![("k".into(), Value::Int(1))]).type_name(),
            "map"
        );
        assert_eq!(Value::Timestamp(12345).type_name(), "timestamp");
    }

    #[test]
    fn test_value_accessors_none_on_mismatch() {
        let s = Value::String("hello".into());
        assert_eq!(s.as_int(), None);
        assert_eq!(s.as_float(), None);
        assert_eq!(s.as_bool(), None);
        assert_eq!(s.as_vector(), None);

        let b = Value::Bool(true);
        assert_eq!(b.as_int(), None);
        assert_eq!(b.as_float(), None);
        assert_eq!(b.as_str(), None);
        assert_eq!(b.as_vector(), None);

        let v = Value::Vector(vec![1.0]);
        assert_eq!(v.as_int(), None);
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_bool(), None);
        assert_eq!(v.as_str(), None);

        assert_eq!(Value::Null.as_int(), None);
        assert_eq!(Value::Null.as_float(), None);
        assert_eq!(Value::Null.as_bool(), None);
        assert_eq!(Value::Null.as_str(), None);
        assert_eq!(Value::Null.as_vector(), None);
    }

    #[test]
    fn test_value_value_type_all() {
        assert_eq!(Value::Null.value_type(), ValueType::Null);
        assert_eq!(Value::Bool(true).value_type(), ValueType::Bool);
        assert_eq!(Value::Int(42).value_type(), ValueType::Int);
        assert_eq!(Value::Float(3.14).value_type(), ValueType::Float);
        assert_eq!(Value::String("hi".into()).value_type(), ValueType::String);
        assert_eq!(Value::Bytes(vec![0]).value_type(), ValueType::Bytes);
        assert_eq!(Value::Vector(vec![1.0]).value_type(), ValueType::Vector);
        assert_eq!(
            Value::List(vec![Value::Null]).value_type(),
            ValueType::List
        );
        assert_eq!(
            Value::Map(vec![("k".into(), Value::Null)]).value_type(),
            ValueType::Map
        );
        assert_eq!(Value::Timestamp(0).value_type(), ValueType::Timestamp);
    }

    #[test]
    fn test_extraction_method_variants() {
        let _a = ExtractionMethod::LlmExtracted;
        let _b = ExtractionMethod::NlpPipeline;
        let _c = ExtractionMethod::UserProvided;
        let _d = ExtractionMethod::Derived;
        let _e = ExtractionMethod::Imported;

        // Verify they compare correctly
        assert_eq!(ExtractionMethod::LlmExtracted, ExtractionMethod::LlmExtracted);
        assert_ne!(ExtractionMethod::LlmExtracted, ExtractionMethod::Imported);
    }

    #[test]
    fn test_token_allocation_proportional() {
        let alloc = TokenAllocation::Proportional {
            entities_pct: 0.4,
            relationships_pct: 0.3,
            text_chunks_pct: 0.2,
            metadata_pct: 0.1,
        };
        match alloc {
            TokenAllocation::Proportional {
                entities_pct,
                relationships_pct,
                text_chunks_pct,
                metadata_pct,
            } => {
                let sum = entities_pct + relationships_pct + text_chunks_pct + metadata_pct;
                assert!((sum - 1.0).abs() < 0.001);
            }
            _ => panic!("expected Proportional"),
        }
    }

    #[test]
    fn test_token_allocation_priority() {
        let alloc = TokenAllocation::Priority(vec![
            ContentPriority::Entities,
            ContentPriority::Relationships,
            ContentPriority::TextChunks,
            ContentPriority::Metadata,
        ]);
        match alloc {
            TokenAllocation::Priority(priorities) => {
                assert_eq!(priorities.len(), 4);
                assert_eq!(priorities[0], ContentPriority::Entities);
                assert_eq!(priorities[3], ContentPriority::Metadata);
            }
            _ => panic!("expected Priority"),
        }
    }

    #[test]
    fn test_conflict_policy_all_variants() {
        let policies = vec![
            ConflictPolicy::LastWriteWins,
            ConflictPolicy::HighestConfidence,
            ConflictPolicy::TemporalInvalidation,
            ConflictPolicy::Merge,
            ConflictPolicy::Reject,
        ];
        assert_eq!(policies.len(), 5);
        assert_eq!(ConflictPolicy::default(), ConflictPolicy::LastWriteWins);
        assert_ne!(ConflictPolicy::Merge, ConflictPolicy::Reject);
    }

    #[test]
    fn test_decay_exponential_zero_half_life() {
        let d = DecayFunction::Exponential { half_life_ms: 0 };
        // With half_life_ms=0 and age > 0, should return 0.0
        assert_eq!(d.apply(1.0, 0, 1000), 0.0);
        assert_eq!(d.apply(0.5, 0, 1), 0.0);
    }

    #[test]
    fn test_decay_linear_zero_max_age() {
        let d = DecayFunction::Linear { max_age_ms: 0 };
        // With max_age_ms=0 and age > 0, should return 0.0
        assert_eq!(d.apply(1.0, 0, 1000), 0.0);
        assert_eq!(d.apply(0.5, 0, 1), 0.0);
    }

    #[test]
    fn test_decay_custom_empty_breakpoints() {
        let d = DecayFunction::Custom {
            breakpoints: vec![],
        };
        // Empty breakpoints => score unchanged
        assert_eq!(d.apply(1.0, 0, 1000), 1.0);
        assert_eq!(d.apply(0.75, 0, 500), 0.75);
    }

    #[test]
    fn test_decay_future_item() {
        // When now <= item_ts, score should be returned unchanged
        let d_exp = DecayFunction::Exponential { half_life_ms: 1000 };
        assert_eq!(d_exp.apply(0.8, 1000, 500), 0.8); // now < item_ts
        assert_eq!(d_exp.apply(0.8, 1000, 1000), 0.8); // now == item_ts

        let d_lin = DecayFunction::Linear { max_age_ms: 1000 };
        assert_eq!(d_lin.apply(0.5, 2000, 1000), 0.5);

        let d_step = DecayFunction::Step { cutoff_ms: 100 };
        assert_eq!(d_step.apply(0.9, 500, 100), 0.9);
    }

    #[test]
    fn test_node_data_construction() {
        let nd = NodeData {
            label: "Person".into(),
            properties: vec![
                ("name".into(), Value::String("Alice".into())),
                ("age".into(), Value::Int(30)),
            ],
            embedding: Some(vec![0.1, 0.2, 0.3]),
            entity_key: Some("alice-001".into()),
            provenance: Some(Provenance::new("test-source", 0.9)),
        };
        assert_eq!(nd.label.as_str(), "Person");
        assert_eq!(nd.properties.len(), 2);
        assert_eq!(nd.embedding.as_ref().unwrap().len(), 3);
        assert_eq!(nd.entity_key.as_ref().unwrap().as_str(), "alice-001");
        assert!(nd.provenance.is_some());
    }

    #[test]
    fn test_edge_data_default() {
        let ed = EdgeData::default();
        assert_eq!(ed.source, 0);
        assert_eq!(ed.target, 0);
        assert_eq!(ed.label.as_str(), "");
        assert!(ed.properties.is_empty());
        assert_eq!(ed.weight, 1.0);
        assert!(ed.provenance.is_none());
    }

    #[test]
    fn test_direction_variants() {
        let out = Direction::Outgoing;
        let inc = Direction::Incoming;
        let both = Direction::Both;
        assert_eq!(out, Direction::Outgoing);
        assert_eq!(inc, Direction::Incoming);
        assert_eq!(both, Direction::Both);
        assert_ne!(out, inc);
        assert_ne!(out, both);
        assert_ne!(inc, both);
    }

    #[test]
    fn test_scored_path_construction() {
        let sp = ScoredPath {
            nodes: vec![1, 2, 3],
            edges: vec![10, 11],
            reliability: 0.85,
        };
        assert_eq!(sp.nodes, vec![1, 2, 3]);
        assert_eq!(sp.edges, vec![10, 11]);
        assert!((sp.reliability - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_provenance_boundary_confidence() {
        // confidence = 0.0 should remain 0.0
        let p0 = Provenance::new("source-a", 0.0);
        assert_eq!(p0.confidence, 0.0);
        assert_eq!(p0.source.as_str(), "source-a");
        assert_eq!(p0.extraction_method, ExtractionMethod::UserProvided);
        assert!(p0.source_document_id.is_none());
        assert!(p0.source_chunk_offset.is_none());

        // confidence = 1.0 should remain 1.0
        let p1 = Provenance::new("source-b", 1.0);
        assert_eq!(p1.confidence, 1.0);
    }

    // ── Round 1 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_bitemporal_zero_width_window() {
        // valid_from == valid_until => half-open [from, until) is empty
        let bt = BiTemporal {
            valid_from: 500,
            valid_until: 500,
            tx_from: 500,
            tx_until: 500,
        };
        assert!(!bt.is_valid_at(500), "zero-width window should not be valid at boundary");
        assert!(!bt.is_valid_at(499));
        assert!(!bt.is_valid_at(501));
        assert!(!bt.is_current_at(500), "zero-width tx window should not be current");
        assert!(!bt.is_active());
    }

    #[test]
    fn test_bitemporal_max_timestamp_boundary() {
        let bt = BiTemporal {
            valid_from: u64::MAX,
            valid_until: BiTemporal::OPEN,
            tx_from: 0,
            tx_until: BiTemporal::OPEN,
        };
        // valid_from = MAX, valid_until = MAX => [MAX, MAX) is empty
        assert!(!bt.is_valid_at(u64::MAX), "half-open at MAX should be invalid");
        assert!(!bt.is_valid_at(u64::MAX - 1), "below MAX not >= valid_from if MAX-1 < MAX");
        // Actually MAX-1 < MAX so valid_from(MAX) <= MAX-1 is false
    }

    #[test]
    fn test_bitemporal_is_current_at_max() {
        let bt = BiTemporal {
            valid_from: 0,
            valid_until: BiTemporal::OPEN,
            tx_from: u64::MAX,
            tx_until: BiTemporal::OPEN,
        };
        // tx_from = MAX, tx_until = MAX => [MAX, MAX) is empty
        assert!(!bt.is_current_at(u64::MAX));
        assert!(!bt.is_current_at(u64::MAX - 1));
    }

    #[test]
    fn test_decay_custom_unsorted_breakpoints() {
        // Breakpoints out of order: code iterates linearly, so (1000, 0.5) is
        // checked first. For age=750, 750 <= 1000 => interpolates (0,1.0)->(1000,0.5)
        let d = DecayFunction::Custom {
            breakpoints: vec![(1000, 0.5), (500, 0.8)],
        };
        let score = d.apply(1.0, 0, 750);
        // Interpolation: t = 750/1000 = 0.75, result = 1.0 + 0.75*(0.5-1.0) = 0.625
        assert!((score - 0.625).abs() < 0.001);
    }

    #[test]
    fn test_decay_custom_duplicate_ages() {
        // Two breakpoints at the same age: bp_age == prev.0,
        // so the code path `bp_age == prev.0` triggers, returns score * bp_mult
        let d = DecayFunction::Custom {
            breakpoints: vec![(500, 0.5), (500, 0.3)],
        };
        // At age 500: first bp (500, 0.5) matches since age <= 500.
        // prev = (0, 1.0), bp = (500, 0.5), bp_age != prev.0 (500 != 0)
        // t = 500/500 = 1.0, result = 1.0 + 1.0*(0.5-1.0) = 0.5
        let score = d.apply(1.0, 0, 500);
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_decay_custom_single_breakpoint() {
        let d = DecayFunction::Custom {
            breakpoints: vec![(500, 0.0)],
        };
        // At age 250: interpolate (0,1.0) -> (500,0.0), t = 0.5, result = 0.5
        assert!((d.apply(1.0, 0, 250) - 0.5).abs() < 0.001);
        // At age 500: t = 1.0, result = 0.0
        assert!((d.apply(1.0, 0, 500) - 0.0).abs() < 0.001);
        // Past last: use last multiplier (0.0)
        assert_eq!(d.apply(1.0, 0, 1000), 0.0);
    }

    #[test]
    fn test_value_empty_list_and_map() {
        let empty_list = Value::List(vec![]);
        assert_eq!(empty_list.type_name(), "list");
        assert_eq!(empty_list.value_type(), ValueType::List);
        assert_eq!(empty_list.as_int(), None);

        let empty_map = Value::Map(vec![]);
        assert_eq!(empty_map.type_name(), "map");
        assert_eq!(empty_map.value_type(), ValueType::Map);
        assert_eq!(empty_map.as_str(), None);
    }

    #[test]
    fn test_value_nested_structures() {
        let nested = Value::List(vec![
            Value::Map(vec![
                ("inner_list".into(), Value::List(vec![
                    Value::Int(42),
                    Value::String("deep".into()),
                ])),
            ]),
            Value::List(vec![Value::Null]),
        ]);
        assert_eq!(nested.type_name(), "list");
        assert_eq!(nested.value_type(), ValueType::List);
        // Verify inner structure is accessible
        if let Value::List(items) = &nested {
            assert_eq!(items.len(), 2);
            if let Value::Map(entries) = &items[0] {
                assert_eq!(entries[0].0.as_str(), "inner_list");
            } else {
                panic!("expected Map");
            }
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn test_value_float_nan_and_infinity() {
        let nan_val = Value::Float(f64::NAN);
        assert_eq!(nan_val.type_name(), "float");
        let extracted = nan_val.as_float().unwrap();
        assert!(extracted.is_nan());

        let inf_val = Value::Float(f64::INFINITY);
        assert_eq!(inf_val.as_float(), Some(f64::INFINITY));

        let neg_inf = Value::Float(f64::NEG_INFINITY);
        assert_eq!(neg_inf.as_float(), Some(f64::NEG_INFINITY));

        // NaN != NaN (IEEE 754)
        assert_ne!(Value::Float(f64::NAN), Value::Float(f64::NAN));
    }

    #[test]
    fn test_provenance_nan_confidence() {
        let p = Provenance::new("src", f32::NAN);
        // f32::NAN.clamp(0.0, 1.0) returns NaN in Rust (comparisons with NaN are false)
        assert!(p.confidence.is_nan());
    }

    #[test]
    fn test_token_allocation_proportional_not_summing_to_one() {
        // Percentages intentionally sum to 0.5 - no validation at type level
        let alloc = TokenAllocation::Proportional {
            entities_pct: 0.2,
            relationships_pct: 0.1,
            text_chunks_pct: 0.1,
            metadata_pct: 0.1,
        };
        if let TokenAllocation::Proportional { entities_pct, relationships_pct, text_chunks_pct, metadata_pct } = alloc {
            let sum = entities_pct + relationships_pct + text_chunks_pct + metadata_pct;
            assert!((sum - 0.5).abs() < 0.001, "sum should be 0.5, got {sum}");
        }

        // Over 1.0
        let over = TokenAllocation::Proportional {
            entities_pct: 0.5,
            relationships_pct: 0.5,
            text_chunks_pct: 0.5,
            metadata_pct: 0.5,
        };
        if let TokenAllocation::Proportional { entities_pct, relationships_pct, text_chunks_pct, metadata_pct } = over {
            let sum = entities_pct + relationships_pct + text_chunks_pct + metadata_pct;
            assert!((sum - 2.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_token_allocation_priority_with_duplicates() {
        let alloc = TokenAllocation::Priority(vec![
            ContentPriority::Entities,
            ContentPriority::Entities,
            ContentPriority::Metadata,
        ]);
        if let TokenAllocation::Priority(p) = alloc {
            assert_eq!(p.len(), 3);
            assert_eq!(p[0], p[1]); // duplicates are allowed
        }
    }
}
