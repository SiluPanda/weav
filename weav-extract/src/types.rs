//! Core types for the extraction pipeline.

#[cfg(feature = "llm-providers")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use weav_core::types::ResolutionMode;

// ─── Input types ─────────────────────────────────────────────────────────────

/// Supported document formats for ingestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentFormat {
    PlainText,
    Pdf,
    Docx,
    Csv,
}

impl DocumentFormat {
    /// Parse a format string (case-insensitive).
    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" | "plaintext" | "plain_text" | "txt" => Some(Self::PlainText),
            "pdf" => Some(Self::Pdf),
            "docx" => Some(Self::Docx),
            "csv" => Some(Self::Csv),
            _ => None,
        }
    }
}

/// Document content — either UTF-8 text or raw binary bytes.
#[derive(Debug, Clone)]
pub enum DocumentContent {
    Text(String),
    Binary(Vec<u8>),
}

/// An input document to be processed by the extraction pipeline.
#[derive(Debug, Clone)]
pub struct InputDocument {
    pub document_id: String,
    pub format: DocumentFormat,
    pub content: DocumentContent,
}

// ─── Chunk types ─────────────────────────────────────────────────────────────

/// A chunk of text produced by the chunker.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub chunk_index: usize,
    pub text: String,
    pub byte_offset: usize,
    pub token_count: usize,
    pub document_id: String,
}

/// A chunk with its computed embedding vector.
#[derive(Debug, Clone)]
pub struct ChunkWithEmbedding {
    pub chunk: TextChunk,
    pub embedding: Vec<f32>,
}

// ─── LLM output types (with JsonSchema for structured output) ────────────────

/// The JSON structure the LLM returns for extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "llm-providers", derive(JsonSchema))]
pub struct LlmExtractionOutput {
    pub entities: Vec<LlmEntity>,
    pub relationships: Vec<LlmRelationship>,
}

/// An entity as extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "llm-providers", derive(JsonSchema))]
pub struct LlmEntity {
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub properties: serde_json::Value,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

/// A relationship as extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "llm-providers", derive(JsonSchema))]
pub struct LlmRelationship {
    pub source_entity: String,
    pub target_entity: String,
    pub relationship_type: String,
    #[serde(default)]
    pub properties: serde_json::Value,
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

fn default_confidence() -> f64 {
    1.0
}

fn default_weight() -> f64 {
    1.0
}

// ─── Extracted types (mapped from LLM output) ───────────────────────────────

/// A fully mapped extracted entity ready for graph insertion.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub properties: Vec<(String, weav_core::types::Value)>,
    pub confidence: f32,
    pub source_chunks: Vec<usize>,
}

/// A fully mapped extracted relationship ready for graph insertion.
#[derive(Debug, Clone)]
pub struct ExtractedRelationship {
    pub source_entity: String,
    pub target_entity: String,
    pub relationship_type: String,
    pub properties: Vec<(String, weav_core::types::Value)>,
    pub weight: f32,
    pub confidence: f32,
    pub source_chunks: Vec<usize>,
}

/// An entity with its computed embedding vector.
#[derive(Debug, Clone)]
pub struct EntityWithEmbedding {
    pub entity: ExtractedEntity,
    pub embedding: Option<Vec<f32>>,
}

// ─── Pipeline result ─────────────────────────────────────────────────────────

/// Statistics from a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    pub total_chunks: usize,
    pub total_entities: usize,
    pub entities_merged: usize,
    pub entities_resolved: usize,
    pub total_relationships: usize,
    pub relationships_merged: usize,
    pub llm_calls: usize,
    pub embedding_calls: usize,
    pub pipeline_duration_ms: u64,
}

/// The complete result of running the extraction pipeline.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub document_id: String,
    pub chunks: Vec<ChunkWithEmbedding>,
    pub entities: Vec<EntityWithEmbedding>,
    pub relationships: Vec<ExtractedRelationship>,
    pub stats: ExtractionStats,
}

/// Options controlling pipeline behavior.
#[derive(Debug, Clone, Default)]
pub struct IngestOptions {
    pub document_id: Option<String>,
    pub format: Option<DocumentFormat>,
    pub skip_extraction: bool,
    pub skip_dedup: bool,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub entity_types: Option<Vec<String>>,
    pub custom_extraction_prompt: Option<String>,
    pub resolution_mode: Option<ResolutionMode>,
    pub link_existing_entities: Option<bool>,
    pub resolution_candidate_limit: Option<usize>,
    pub custom_resolution_prompt: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_format_from_str() {
        assert_eq!(
            DocumentFormat::from_str_lossy("text"),
            Some(DocumentFormat::PlainText)
        );
        assert_eq!(
            DocumentFormat::from_str_lossy("PDF"),
            Some(DocumentFormat::Pdf)
        );
        assert_eq!(
            DocumentFormat::from_str_lossy("docx"),
            Some(DocumentFormat::Docx)
        );
        assert_eq!(
            DocumentFormat::from_str_lossy("CSV"),
            Some(DocumentFormat::Csv)
        );
        assert_eq!(
            DocumentFormat::from_str_lossy("txt"),
            Some(DocumentFormat::PlainText)
        );
        assert_eq!(DocumentFormat::from_str_lossy("unknown"), None);
    }

    #[test]
    fn test_llm_extraction_output_deserialize() {
        let json = r#"{
            "entities": [{
                "name": "Alice",
                "entity_type": "Person",
                "description": "A person named Alice",
                "properties": {"age": 30},
                "confidence": 0.95
            }],
            "relationships": [{
                "source_entity": "Alice",
                "target_entity": "Bob",
                "relationship_type": "knows",
                "properties": {},
                "weight": 0.8,
                "confidence": 0.9
            }]
        }"#;
        let output: LlmExtractionOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.entities.len(), 1);
        assert_eq!(output.entities[0].name, "Alice");
        assert_eq!(output.relationships.len(), 1);
        assert_eq!(output.relationships[0].relationship_type, "knows");
    }

    #[test]
    fn test_llm_output_defaults() {
        let json = r#"{
            "entities": [{
                "name": "Test",
                "entity_type": "Thing"
            }],
            "relationships": [{
                "source_entity": "A",
                "target_entity": "B",
                "relationship_type": "related"
            }]
        }"#;
        let output: LlmExtractionOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.entities[0].confidence, 1.0);
        assert_eq!(output.entities[0].description, "");
        assert_eq!(output.relationships[0].weight, 1.0);
        assert_eq!(output.relationships[0].confidence, 1.0);
    }

    #[test]
    fn test_ingest_options_default() {
        let opts = IngestOptions::default();
        assert!(!opts.skip_extraction);
        assert!(!opts.skip_dedup);
        assert!(opts.document_id.is_none());
        assert!(opts.format.is_none());
        assert!(opts.entity_types.is_none());
        assert!(opts.resolution_mode.is_none());
        assert!(opts.link_existing_entities.is_none());
        assert!(opts.resolution_candidate_limit.is_none());
        assert!(opts.custom_resolution_prompt.is_none());
    }

    #[test]
    fn test_extraction_stats_default() {
        let stats = ExtractionStats::default();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_entities, 0);
        assert_eq!(stats.entities_merged, 0);
        assert_eq!(stats.entities_resolved, 0);
    }

    #[test]
    #[cfg(feature = "llm-providers")]
    fn test_llm_extraction_output_schema() {
        let schema = schemars::schema_for!(LlmExtractionOutput);
        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("entities"));
        assert!(json.contains("relationships"));
        assert!(json.contains("entity_type"));
    }
}
