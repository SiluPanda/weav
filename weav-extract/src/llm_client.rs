//! Multi-provider LLM client for entity extraction and embedding generation.

use llm::builder::LLMBackend;
use llm::chat::ChatMessage;
use weav_core::config::ExtractConfig;
use weav_core::error::{WeavError, WeavResult};
use weav_core::types::ResolutionMode;

use crate::types::{LlmExtractionOutput, TextChunk};

/// Wraps the `llm` crate to provide extraction and embedding capabilities.
pub struct LlmClient {
    config: ExtractConfig,
}

impl LlmClient {
    /// Create a new LLM client from extraction config.
    pub fn new(config: &ExtractConfig) -> WeavResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Extract entities and relationships from text chunks using an LLM.
    pub async fn extract_entities(
        &self,
        chunks: &[TextChunk],
        entity_types: Option<&[String]>,
        custom_prompt: Option<&str>,
        resolution_mode: ResolutionMode,
        custom_resolution_prompt: Option<&str>,
    ) -> WeavResult<LlmExtractionOutput> {
        let schema = serde_json::to_string_pretty(&schemars::schema_for!(LlmExtractionOutput))
            .map_err(|e| WeavError::ExtractionError(format!("failed to generate schema: {e}")))?;

        let system_prompt = build_extraction_prompt(
            &schema,
            entity_types,
            custom_prompt,
            resolution_mode,
            custom_resolution_prompt,
        );

        let combined_text: String = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[Chunk {}]\n{}", i, c.text))
            .collect::<Vec<_>>()
            .join("\n\n");

        let backend = parse_backend(&self.config.llm_backend);
        let api_key = self.config.llm_api_key.as_deref().unwrap_or("");

        let mut builder = llm::builder::LLMBuilder::new()
            .backend(backend)
            .api_key(api_key)
            .model(&self.config.extraction_model)
            .temperature(self.config.extraction_temperature)
            .max_tokens(self.config.max_extraction_tokens as u32)
            .system(&system_prompt);

        if let Some(ref base_url) = self.config.llm_base_url {
            builder = builder.base_url(base_url);
        }

        let llm_instance = builder
            .build()
            .map_err(|e| WeavError::LlmError(format!("failed to build LLM client: {e}")))?;

        let messages = vec![ChatMessage::user().content(&combined_text).build()];

        let response = llm_instance
            .chat(&messages)
            .await
            .map_err(|e| WeavError::LlmError(format!("LLM extraction call failed: {e}")))?;

        let response_text = response
            .text()
            .ok_or_else(|| WeavError::LlmError("LLM returned empty response".into()))?;
        let cleaned = strip_code_fences(&response_text);
        parse_extraction_response(&cleaned)
    }

    /// Generate embeddings for a batch of texts.
    pub async fn embed_batch(&self, texts: &[String]) -> WeavResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let backend = parse_backend(&self.config.embedding_backend);
        let api_key = self
            .config
            .embedding_api_key
            .as_deref()
            .or(self.config.llm_api_key.as_deref())
            .unwrap_or("");

        let mut builder = llm::builder::LLMBuilder::new()
            .backend(backend)
            .api_key(api_key)
            .model(&self.config.embedding_model);

        if let Some(ref base_url) = self.config.llm_base_url {
            builder = builder.base_url(base_url);
        }

        let llm_instance = builder
            .build()
            .map_err(|e| WeavError::LlmError(format!("failed to build embedding client: {e}")))?;

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.config.embedding_batch_size) {
            let batch_vec: Vec<String> = batch.to_vec();
            let embeddings = llm_instance
                .embed(batch_vec)
                .await
                .map_err(|e| WeavError::LlmError(format!("embedding call failed: {e}")))?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Get the configured embedding dimensions.
    pub fn embedding_dimensions(&self) -> u16 {
        self.config.embedding_dimensions
    }
}

/// Map a backend string to an LLMBackend enum value.
pub fn parse_backend(s: &str) -> LLMBackend {
    match s.to_lowercase().as_str() {
        "openai" => LLMBackend::OpenAI,
        "anthropic" => LLMBackend::Anthropic,
        "ollama" => LLMBackend::Ollama,
        "deepseek" => LLMBackend::DeepSeek,
        "xai" => LLMBackend::XAI,
        "phind" => LLMBackend::Phind,
        "groq" => LLMBackend::Groq,
        "google" => LLMBackend::Google,
        _ => LLMBackend::OpenAI, // Default to OpenAI-compatible
    }
}

/// Build the system prompt for entity/relationship extraction.
fn build_extraction_prompt(
    schema: &str,
    entity_types: Option<&[String]>,
    custom_prompt: Option<&str>,
    resolution_mode: ResolutionMode,
    custom_resolution_prompt: Option<&str>,
) -> String {
    let mut prompt = String::from(
        "You are an expert knowledge graph extraction system. \
         Analyze the provided text and extract all entities and relationships.\n\n",
    );

    if let Some(types) = entity_types {
        prompt.push_str(&format!(
            "Focus on extracting these entity types: {}.\n\n",
            types.join(", ")
        ));
    }

    if let Some(custom) = custom_prompt {
        prompt.push_str(&format!("Additional instructions: {custom}\n\n"));
    }

    if matches!(resolution_mode, ResolutionMode::Semantic) {
        prompt.push_str(
            "Alias resolution mode is enabled. When two mentions in the provided text refer \
             to the same entity, add a `canonical_name` string inside that entity's `properties` \
             object and set it to the most canonical mention from the provided text. \
             Omit `canonical_name` when the entity name is already canonical.\n\n",
        );

        if let Some(custom) = custom_resolution_prompt {
            prompt.push_str(&format!("Resolution instructions: {custom}\n\n"));
        }
    }

    prompt.push_str(&format!(
        "Return your response as valid JSON matching this schema:\n```json\n{schema}\n```\n\n\
         Guidelines:\n\
         - Extract all named entities (people, organizations, locations, concepts, etc.)\n\
         - Extract all relationships between entities\n\
         - Use descriptive relationship types (e.g., \"works_at\", \"located_in\", \"authored\")\n\
         - Provide a confidence score between 0.0 and 1.0 for each entity and relationship\n\
         - Include relevant properties for entities when mentioned in the text\n\
         - Return ONLY the JSON object, no additional text"
    ));

    prompt
}

/// Strip markdown code fences from LLM response.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        if let Some(stripped) = rest.strip_suffix("```") {
            return stripped.trim().to_string();
        }
        return rest.trim().to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(stripped) = rest.strip_suffix("```") {
            return stripped.trim().to_string();
        }
        return rest.trim().to_string();
    }
    trimmed.to_string()
}

/// Parse the LLM's JSON response into structured output.
fn parse_extraction_response(text: &str) -> WeavResult<LlmExtractionOutput> {
    serde_json::from_str(text).map_err(|e| {
        WeavError::ExtractionError(format!(
            "failed to parse LLM extraction response: {e}\nResponse was: {}",
            &text[..text.len().min(500)]
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_backend() {
        assert!(matches!(parse_backend("openai"), LLMBackend::OpenAI));
        assert!(matches!(parse_backend("anthropic"), LLMBackend::Anthropic));
        assert!(matches!(parse_backend("ollama"), LLMBackend::Ollama));
        assert!(matches!(parse_backend("OpenAI"), LLMBackend::OpenAI));
        assert!(matches!(parse_backend("unknown"), LLMBackend::OpenAI));
    }

    #[test]
    fn test_strip_code_fences_json() {
        let input = "```json\n{\"entities\": []}\n```";
        assert_eq!(strip_code_fences(input), "{\"entities\": []}");
    }

    #[test]
    fn test_strip_code_fences_plain() {
        let input = "```\n{\"entities\": []}\n```";
        assert_eq!(strip_code_fences(input), "{\"entities\": []}");
    }

    #[test]
    fn test_strip_code_fences_none() {
        let input = "{\"entities\": []}";
        assert_eq!(strip_code_fences(input), "{\"entities\": []}");
    }

    #[test]
    fn test_parse_extraction_response_valid() {
        let json =
            r#"{"entities": [{"name": "Alice", "entity_type": "Person"}], "relationships": []}"#;
        let result = parse_extraction_response(json).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Alice");
    }

    #[test]
    fn test_parse_extraction_response_invalid() {
        let result = parse_extraction_response("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_extraction_prompt_basic() {
        let prompt = build_extraction_prompt("{}", None, None, ResolutionMode::Heuristic, None);
        assert!(prompt.contains("knowledge graph extraction"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_build_extraction_prompt_with_entity_types() {
        let types = vec!["Person".into(), "Organization".into()];
        let prompt =
            build_extraction_prompt("{}", Some(&types), None, ResolutionMode::Heuristic, None);
        assert!(prompt.contains("Person, Organization"));
    }

    #[test]
    fn test_build_extraction_prompt_with_custom() {
        let prompt = build_extraction_prompt(
            "{}",
            None,
            Some("Focus on technical entities"),
            ResolutionMode::Heuristic,
            None,
        );
        assert!(prompt.contains("Focus on technical entities"));
    }

    #[test]
    fn test_build_extraction_prompt_with_semantic_resolution() {
        let prompt = build_extraction_prompt(
            "{}",
            None,
            None,
            ResolutionMode::Semantic,
            Some("Prefer full company names."),
        );
        assert!(prompt.contains("canonical_name"));
        assert!(prompt.contains("Prefer full company names."));
    }

    #[test]
    fn test_llm_client_new() {
        let config = ExtractConfig::default();
        let client = LlmClient::new(&config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_malformed_json_does_not_panic() {
        // Various forms of malformed input that an LLM might return
        let malformed_inputs = vec![
            "",
            "not json at all",
            "{",
            "{\"entities\": [}",
            "null",
            "42",
            "\"just a string\"",
            "[\"array\", \"not\", \"object\"]",
            "{\"entities\": \"not_an_array\"}",
            "{\"wrong_field\": []}",
            "```json\nnot actually json\n```",
            "{\"entities\": [{\"name\": 123}], \"relationships\": []}",
        ];

        for input in &malformed_inputs {
            let cleaned = strip_code_fences(input);
            let result = parse_extraction_response(&cleaned);
            // Must return Err, never panic
            assert!(
                result.is_err(),
                "expected Err for malformed input: {input:?}"
            );
        }
    }

    #[test]
    fn test_strip_code_fences_empty_input() {
        assert_eq!(strip_code_fences(""), "");
    }

    #[test]
    fn test_strip_code_fences_only_backticks() {
        assert_eq!(strip_code_fences("``````"), "");
    }

    #[test]
    fn test_strip_code_fences_nested_backticks() {
        let input = "```json\n{\"key\": \"value with ``` inside\"}\n```";
        let result = strip_code_fences(input);
        assert!(result.contains("key"));
    }
}
