//! Maps LLM extraction output to typed entities and relationships.

use compact_str::CompactString;
use weav_core::types::Value;

use crate::types::{
    ExtractedEntity, ExtractedRelationship, LlmEntity, LlmExtractionOutput, LlmRelationship,
};

/// Map LLM entities to typed ExtractedEntity instances.
pub fn map_entities(
    output: &LlmExtractionOutput,
    source_chunk_indices: &[usize],
) -> Vec<ExtractedEntity> {
    output
        .entities
        .iter()
        .map(|e| map_single_entity(e, source_chunk_indices))
        .collect()
}

/// Map LLM relationships to typed ExtractedRelationship instances.
pub fn map_relationships(
    output: &LlmExtractionOutput,
    source_chunk_indices: &[usize],
) -> Vec<ExtractedRelationship> {
    output
        .relationships
        .iter()
        .map(|r| map_single_relationship(r, source_chunk_indices))
        .collect()
}

fn map_single_entity(entity: &LlmEntity, source_chunks: &[usize]) -> ExtractedEntity {
    let properties = json_value_to_properties(&entity.properties);
    let confidence = entity.confidence.clamp(0.0, 1.0) as f32;

    ExtractedEntity {
        name: entity.name.clone(),
        entity_type: entity.entity_type.clone(),
        description: entity.description.clone(),
        properties,
        confidence,
        source_chunks: source_chunks.to_vec(),
    }
}

fn map_single_relationship(
    rel: &LlmRelationship,
    source_chunks: &[usize],
) -> ExtractedRelationship {
    let properties = json_value_to_properties(&rel.properties);
    let weight = rel.weight.clamp(0.0, 1.0) as f32;
    let confidence = rel.confidence.clamp(0.0, 1.0) as f32;

    ExtractedRelationship {
        source_entity: rel.source_entity.clone(),
        target_entity: rel.target_entity.clone(),
        relationship_type: rel.relationship_type.clone(),
        properties,
        weight,
        confidence,
        source_chunks: source_chunks.to_vec(),
    }
}

/// Convert a serde_json::Value (from LLM properties) to Vec<(String, Value)>.
fn json_value_to_properties(val: &serde_json::Value) -> Vec<(String, Value)> {
    match val {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| (k.clone(), json_to_weav_value(v)))
            .collect(),
        serde_json::Value::Null => Vec::new(),
        _ => Vec::new(),
    }
}

/// Convert a serde_json::Value to a weav Value.
fn json_to_weav_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(CompactString::new(s)),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_weav_value).collect();
            Value::List(items)
        }
        serde_json::Value::Object(map) => {
            let pairs: Vec<(CompactString, Value)> = map
                .iter()
                .map(|(k, v)| (CompactString::new(k), json_to_weav_value(v)))
                .collect();
            Value::Map(pairs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LlmEntity, LlmExtractionOutput, LlmRelationship};

    #[test]
    fn test_map_entities_basic() {
        let output = LlmExtractionOutput {
            entities: vec![
                LlmEntity {
                    name: "Alice".into(),
                    entity_type: "Person".into(),
                    description: "A developer".into(),
                    properties: serde_json::json!({"role": "engineer"}),
                    confidence: 0.95,
                },
                LlmEntity {
                    name: "Acme Corp".into(),
                    entity_type: "Organization".into(),
                    description: "A tech company".into(),
                    properties: serde_json::json!({}),
                    confidence: 0.8,
                },
            ],
            relationships: vec![],
        };

        let entities = map_entities(&output, &[0, 1]);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[0].entity_type, "Person");
        assert_eq!(entities[0].confidence, 0.95);
        assert_eq!(entities[0].source_chunks, vec![0, 1]);
        assert_eq!(entities[0].properties.len(), 1);
        assert_eq!(entities[0].properties[0].0, "role");
    }

    #[test]
    fn test_map_relationships_basic() {
        let output = LlmExtractionOutput {
            entities: vec![],
            relationships: vec![LlmRelationship {
                source_entity: "Alice".into(),
                target_entity: "Acme Corp".into(),
                relationship_type: "works_at".into(),
                properties: serde_json::json!({"since": 2020}),
                weight: 0.9,
                confidence: 0.85,
            }],
        };

        let rels = map_relationships(&output, &[0]);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].source_entity, "Alice");
        assert_eq!(rels[0].target_entity, "Acme Corp");
        assert_eq!(rels[0].relationship_type, "works_at");
        assert_eq!(rels[0].weight, 0.9);
        assert_eq!(rels[0].confidence, 0.85);
    }

    #[test]
    fn test_confidence_clamping() {
        let entity = LlmEntity {
            name: "Test".into(),
            entity_type: "Thing".into(),
            description: "".into(),
            properties: serde_json::Value::Null,
            confidence: 1.5,
        };
        let mapped = map_single_entity(&entity, &[]);
        assert_eq!(mapped.confidence, 1.0);

        let entity2 = LlmEntity {
            confidence: -0.5,
            ..entity
        };
        let mapped2 = map_single_entity(&entity2, &[]);
        assert_eq!(mapped2.confidence, 0.0);
    }

    #[test]
    fn test_json_value_to_properties_null() {
        let props = json_value_to_properties(&serde_json::Value::Null);
        assert!(props.is_empty());
    }

    #[test]
    fn test_json_value_to_properties_object() {
        let val = serde_json::json!({"name": "Alice", "age": 30, "active": true});
        let props = json_value_to_properties(&val);
        assert_eq!(props.len(), 3);
    }

    #[test]
    fn test_json_to_weav_value_types() {
        assert!(matches!(
            json_to_weav_value(&serde_json::json!(null)),
            Value::Null
        ));
        assert!(matches!(
            json_to_weav_value(&serde_json::json!(true)),
            Value::Bool(true)
        ));
        assert!(matches!(
            json_to_weav_value(&serde_json::json!(42)),
            Value::Int(42)
        ));
        assert!(matches!(
            json_to_weav_value(&serde_json::json!(3.14)),
            Value::Float(_)
        ));
        assert!(matches!(
            json_to_weav_value(&serde_json::json!("hello")),
            Value::String(_)
        ));
    }
}
