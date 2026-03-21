//! Pipeline orchestration: document → chunks → embeddings → entities → graph.

use std::collections::HashMap;
use std::time::Instant;

use tokio::sync::Semaphore;
use weav_core::config::ExtractConfig;
use weav_core::error::WeavResult;

use crate::chunker;
use crate::document;
use crate::extractor;
use crate::llm_client::LlmClient;
use crate::types::*;

/// Run the full extraction pipeline on a document.
///
/// Steps:
/// 1. Parse document to text
/// 2. Chunk text
/// 3. Generate chunk embeddings (batched, concurrent)
/// 4. Extract entities/relationships via LLM (batched by token limit)
/// 5. Deduplicate entities by name
/// 6. Generate entity description embeddings
/// 7. Deduplicate relationships by (source, target, type)
/// 8. Return ExtractionResult
pub async fn run_pipeline(
    doc: InputDocument,
    options: &IngestOptions,
    config: &ExtractConfig,
) -> WeavResult<ExtractionResult> {
    let start = Instant::now();
    let mut stats = ExtractionStats::default();

    // Step 1: Parse document to text.
    let text = document::extract_text(&doc)?;
    tracing::debug!(
        document_id = %doc.document_id,
        text_len = text.len(),
        "extracted text from document"
    );

    // Step 2: Chunk text.
    let effective_config = if options.chunk_size.is_some() || options.chunk_overlap.is_some() {
        let mut cfg = config.clone();
        if let Some(size) = options.chunk_size {
            cfg.chunk_size = size;
        }
        if let Some(overlap) = options.chunk_overlap {
            cfg.chunk_overlap = overlap;
        }
        cfg
    } else {
        config.clone()
    };

    let chunks = chunker::chunk_text(&text, &doc.document_id, &effective_config)?;
    stats.total_chunks = chunks.len();
    tracing::debug!(chunks = chunks.len(), "text chunked");

    // Step 3: Generate chunk embeddings.
    let llm = LlmClient::new(config)?;
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

    let embedding_semaphore = Semaphore::new(config.max_concurrent_embedding_calls);
    let chunk_embeddings = embed_texts_batched(
        &llm,
        &chunk_texts,
        config.embedding_batch_size,
        &embedding_semaphore,
        &mut stats,
    )
    .await?;

    let chunks_with_embeddings: Vec<ChunkWithEmbedding> = chunks
        .into_iter()
        .zip(chunk_embeddings.into_iter())
        .map(|(chunk, embedding)| ChunkWithEmbedding { chunk, embedding })
        .collect();

    // Step 4: Extract entities/relationships via LLM (if not skipped).
    let mut entities: Vec<ExtractedEntity> = Vec::new();
    let mut relationships: Vec<ExtractedRelationship> = Vec::new();

    if !options.skip_extraction {
        let extraction_batches =
            batch_chunks_by_tokens(&chunks_with_embeddings, config.max_extraction_tokens);

        let _llm_semaphore = Semaphore::new(config.max_concurrent_llm_calls);

        for batch in &extraction_batches {
            let batch_chunks: Vec<&TextChunk> =
                batch.iter().map(|cwe| &cwe.chunk).collect();
            let chunk_indices: Vec<usize> =
                batch_chunks.iter().map(|c| c.chunk_index).collect();

            let text_chunks: Vec<TextChunk> =
                batch_chunks.iter().map(|c| (*c).clone()).collect();

            let output = llm
                .extract_entities(
                    &text_chunks,
                    options.entity_types.as_deref(),
                    options.custom_extraction_prompt.as_deref(),
                )
                .await?;
            stats.llm_calls += 1;

            let batch_entities = extractor::map_entities(&output, &chunk_indices);
            let batch_rels = extractor::map_relationships(&output, &chunk_indices);

            entities.extend(batch_entities);
            relationships.extend(batch_rels);
        }
    }

    // Step 5: Deduplicate entities by name.
    if !options.skip_dedup {
        let original_count = entities.len();
        entities = dedup_entities(entities);
        stats.entities_merged = original_count - entities.len();
    }
    stats.total_entities = entities.len();

    // Step 6: Generate entity description embeddings.
    let entity_descriptions: Vec<String> = entities
        .iter()
        .map(|e| {
            if e.description.is_empty() {
                format!("{} ({})", e.name, e.entity_type)
            } else {
                e.description.clone()
            }
        })
        .collect();

    let entity_embeddings = if !entity_descriptions.is_empty() {
        embed_texts_batched(
            &llm,
            &entity_descriptions,
            config.embedding_batch_size,
            &embedding_semaphore,
            &mut stats,
        )
        .await?
    } else {
        Vec::new()
    };

    let entities_with_embeddings: Vec<EntityWithEmbedding> = entities
        .into_iter()
        .zip(
            entity_embeddings
                .into_iter()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .map(|(entity, embedding)| EntityWithEmbedding { entity, embedding })
        .collect();

    // Step 7: Deduplicate relationships by (source, target, type).
    if !options.skip_dedup {
        let original_count = relationships.len();
        relationships = dedup_relationships(relationships);
        stats.relationships_merged = original_count - relationships.len();
    }
    stats.total_relationships = relationships.len();
    stats.pipeline_duration_ms = start.elapsed().as_millis() as u64;

    tracing::info!(
        document_id = %doc.document_id,
        chunks = stats.total_chunks,
        entities = stats.total_entities,
        relationships = stats.total_relationships,
        duration_ms = stats.pipeline_duration_ms,
        "extraction pipeline complete"
    );

    Ok(ExtractionResult {
        document_id: doc.document_id.clone(),
        chunks: chunks_with_embeddings,
        entities: entities_with_embeddings,
        relationships,
        stats,
    })
}

/// Batch texts for embedding and call the LLM client.
async fn embed_texts_batched(
    llm: &LlmClient,
    texts: &[String],
    batch_size: usize,
    _semaphore: &Semaphore,
    stats: &mut ExtractionStats,
) -> WeavResult<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let mut all_embeddings = Vec::with_capacity(texts.len());

    for batch in texts.chunks(batch_size) {
        let batch_vec: Vec<String> = batch.to_vec();
        let embeddings = llm.embed_batch(&batch_vec).await?;
        stats.embedding_calls += 1;
        all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
}

/// Batch chunks by cumulative token count, respecting max_extraction_tokens.
fn batch_chunks_by_tokens(
    chunks: &[ChunkWithEmbedding],
    max_tokens: usize,
) -> Vec<Vec<ChunkWithEmbedding>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_tokens = 0;

    for chunk in chunks {
        if current_tokens + chunk.chunk.token_count > max_tokens && !current_batch.is_empty() {
            batches.push(std::mem::take(&mut current_batch));
            current_tokens = 0;
        }
        current_tokens += chunk.chunk.token_count;
        current_batch.push(chunk.clone());
    }

    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    batches
}

/// Default Jaro-Winkler similarity threshold for entity deduplication.
const DEDUP_FUZZY_THRESHOLD: f64 = 0.85;

/// Deduplicate entities by fuzzy name matching (Jaro-Winkler similarity),
/// merging source_chunks and keeping the higher confidence.
fn dedup_entities(entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
    let mut result: Vec<ExtractedEntity> = Vec::new();

    for entity in entities {
        // Find the best fuzzy match among already-seen entities.
        let match_idx = result.iter().enumerate().find_map(|(idx, existing)| {
            if weav_graph::dedup::fuzzy_name_match(
                &entity.name,
                &existing.name,
                DEDUP_FUZZY_THRESHOLD,
            ) {
                Some(idx)
            } else {
                None
            }
        });

        if let Some(idx) = match_idx {
            // Merge: combine source_chunks, keep higher confidence.
            let existing = &mut result[idx];
            for chunk_idx in &entity.source_chunks {
                if !existing.source_chunks.contains(chunk_idx) {
                    existing.source_chunks.push(*chunk_idx);
                }
            }
            if entity.confidence > existing.confidence {
                existing.confidence = entity.confidence;
                existing.description = entity.description;
            }
        } else {
            result.push(entity);
        }
    }

    result
}

/// Deduplicate relationships by (source, target, type) key.
fn dedup_relationships(relationships: Vec<ExtractedRelationship>) -> Vec<ExtractedRelationship> {
    let mut seen: HashMap<(String, String, String), usize> = HashMap::new();
    let mut result: Vec<ExtractedRelationship> = Vec::new();

    for rel in relationships {
        let key = (
            rel.source_entity.to_lowercase(),
            rel.target_entity.to_lowercase(),
            rel.relationship_type.to_lowercase(),
        );
        if let Some(&idx) = seen.get(&key) {
            let existing = &mut result[idx];
            for chunk_idx in &rel.source_chunks {
                if !existing.source_chunks.contains(chunk_idx) {
                    existing.source_chunks.push(*chunk_idx);
                }
            }
            if rel.confidence > existing.confidence {
                existing.confidence = rel.confidence;
                existing.weight = rel.weight;
            }
        } else {
            seen.insert(key, result.len());
            result.push(rel);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_entities_no_duplicates() {
        let entities = vec![
            ExtractedEntity {
                name: "Alice".into(),
                entity_type: "Person".into(),
                description: "A person".into(),
                properties: vec![],
                confidence: 0.9,
                source_chunks: vec![0],
            },
            ExtractedEntity {
                name: "Bob".into(),
                entity_type: "Person".into(),
                description: "Another person".into(),
                properties: vec![],
                confidence: 0.8,
                source_chunks: vec![1],
            },
        ];
        let result = dedup_entities(entities);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_dedup_entities_with_duplicates() {
        let entities = vec![
            ExtractedEntity {
                name: "Alice".into(),
                entity_type: "Person".into(),
                description: "First mention".into(),
                properties: vec![],
                confidence: 0.7,
                source_chunks: vec![0],
            },
            ExtractedEntity {
                name: "alice".into(), // Case-insensitive duplicate
                entity_type: "Person".into(),
                description: "Second mention with better desc".into(),
                properties: vec![],
                confidence: 0.9,
                source_chunks: vec![1, 2],
            },
        ];
        let result = dedup_entities(entities);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Alice");
        assert_eq!(result[0].confidence, 0.9);
        assert_eq!(result[0].description, "Second mention with better desc");
        assert_eq!(result[0].source_chunks, vec![0, 1, 2]);
    }

    #[test]
    fn test_dedup_relationships_no_duplicates() {
        let rels = vec![
            ExtractedRelationship {
                source_entity: "Alice".into(),
                target_entity: "Bob".into(),
                relationship_type: "knows".into(),
                properties: vec![],
                weight: 0.8,
                confidence: 0.9,
                source_chunks: vec![0],
            },
            ExtractedRelationship {
                source_entity: "Alice".into(),
                target_entity: "Acme".into(),
                relationship_type: "works_at".into(),
                properties: vec![],
                weight: 1.0,
                confidence: 0.95,
                source_chunks: vec![0],
            },
        ];
        let result = dedup_relationships(rels);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_dedup_relationships_with_duplicates() {
        let rels = vec![
            ExtractedRelationship {
                source_entity: "Alice".into(),
                target_entity: "Bob".into(),
                relationship_type: "knows".into(),
                properties: vec![],
                weight: 0.5,
                confidence: 0.7,
                source_chunks: vec![0],
            },
            ExtractedRelationship {
                source_entity: "alice".into(),
                target_entity: "bob".into(),
                relationship_type: "Knows".into(),
                properties: vec![],
                weight: 0.9,
                confidence: 0.95,
                source_chunks: vec![1],
            },
        ];
        let result = dedup_relationships(rels);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].confidence, 0.95);
        assert_eq!(result[0].weight, 0.9);
        assert_eq!(result[0].source_chunks, vec![0, 1]);
    }

    #[test]
    fn test_batch_chunks_by_tokens_single_batch() {
        let chunks = vec![
            ChunkWithEmbedding {
                chunk: TextChunk {
                    chunk_index: 0,
                    text: "hello".into(),
                    byte_offset: 0,
                    token_count: 100,
                    document_id: "doc".into(),
                },
                embedding: vec![],
            },
            ChunkWithEmbedding {
                chunk: TextChunk {
                    chunk_index: 1,
                    text: "world".into(),
                    byte_offset: 5,
                    token_count: 100,
                    document_id: "doc".into(),
                },
                embedding: vec![],
            },
        ];
        let batches = batch_chunks_by_tokens(&chunks, 1000);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
    }

    #[test]
    fn test_batch_chunks_by_tokens_multiple_batches() {
        let chunks = vec![
            ChunkWithEmbedding {
                chunk: TextChunk {
                    chunk_index: 0,
                    text: "a".into(),
                    byte_offset: 0,
                    token_count: 500,
                    document_id: "doc".into(),
                },
                embedding: vec![],
            },
            ChunkWithEmbedding {
                chunk: TextChunk {
                    chunk_index: 1,
                    text: "b".into(),
                    byte_offset: 1,
                    token_count: 500,
                    document_id: "doc".into(),
                },
                embedding: vec![],
            },
            ChunkWithEmbedding {
                chunk: TextChunk {
                    chunk_index: 2,
                    text: "c".into(),
                    byte_offset: 2,
                    token_count: 500,
                    document_id: "doc".into(),
                },
                embedding: vec![],
            },
        ];
        let batches = batch_chunks_by_tokens(&chunks, 800);
        assert_eq!(batches.len(), 3); // Each chunk is its own batch
    }

    #[test]
    fn test_batch_chunks_empty() {
        let batches = batch_chunks_by_tokens(&[], 1000);
        assert!(batches.is_empty());
    }
}
