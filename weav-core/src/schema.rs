//! Schema validation and property type constraints.

use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{WeavError, WeavResult};
use crate::types::{Value, ValueType};

/// A constraint that can be applied to nodes or edges with a specific label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SchemaConstraint {
    /// Property must be of the specified type.
    PropertyType {
        property: CompactString,
        expected_type: ValueType,
    },
    /// Property must exist (cannot be null/missing).
    PropertyRequired { property: CompactString },
    /// Property value must be unique across all nodes/edges with this label.
    PropertyUnique { property: CompactString },
}

/// Schema definition for a specific label (node or edge type).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LabelSchema {
    pub constraints: Vec<SchemaConstraint>,
}

/// Schema registry for a graph, mapping label names to their schemas.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphSchema {
    pub node_schemas: HashMap<CompactString, LabelSchema>,
    pub edge_schemas: HashMap<CompactString, LabelSchema>,
}

impl GraphSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint for a node label.
    pub fn add_node_constraint(
        &mut self,
        label: impl Into<CompactString>,
        constraint: SchemaConstraint,
    ) {
        self.node_schemas
            .entry(label.into())
            .or_default()
            .constraints
            .push(constraint);
    }

    /// Add a constraint for an edge label.
    pub fn add_edge_constraint(
        &mut self,
        label: impl Into<CompactString>,
        constraint: SchemaConstraint,
    ) {
        self.edge_schemas
            .entry(label.into())
            .or_default()
            .constraints
            .push(constraint);
    }

    /// Validate properties against the schema for a node label.
    /// Returns Ok(()) if valid, Err with details if not.
    pub fn validate_node_properties(
        &self,
        label: &str,
        properties: &[(String, Value)],
    ) -> WeavResult<()> {
        if let Some(schema) = self.node_schemas.get(label) {
            validate_properties(&schema.constraints, properties)
        } else {
            Ok(()) // No schema = no constraints
        }
    }

    /// Validate properties against the schema for an edge label.
    pub fn validate_edge_properties(
        &self,
        label: &str,
        properties: &[(String, Value)],
    ) -> WeavResult<()> {
        if let Some(schema) = self.edge_schemas.get(label) {
            validate_properties(&schema.constraints, properties)
        } else {
            Ok(())
        }
    }
}

fn validate_properties(
    constraints: &[SchemaConstraint],
    properties: &[(String, Value)],
) -> WeavResult<()> {
    for constraint in constraints {
        match constraint {
            SchemaConstraint::PropertyType {
                property,
                expected_type,
            } => {
                if let Some((_, value)) = properties.iter().find(|(k, _)| k == property.as_str())
                    && value.value_type() != *expected_type
                    && *value != Value::Null
                {
                    return Err(WeavError::InvalidConfig(format!(
                        "property '{}' expected type {:?}, got {:?}",
                        property,
                        expected_type,
                        value.value_type()
                    )));
                }
            }
            SchemaConstraint::PropertyRequired { property } => {
                match properties.iter().find(|(k, _)| k == property.as_str()) {
                    None => {
                        return Err(WeavError::InvalidConfig(format!(
                            "required property '{}' is missing",
                            property
                        )));
                    }
                    Some((_, Value::Null)) => {
                        return Err(WeavError::InvalidConfig(format!(
                            "required property '{}' is missing",
                            property
                        )));
                    }
                    _ => {}
                }
            }
            SchemaConstraint::PropertyUnique { .. } => {
                // Uniqueness is enforced at the engine level, not here
                // (requires checking all existing nodes)
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_type_correct_passes() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyType {
                property: CompactString::new("age"),
                expected_type: ValueType::Int,
            },
        );
        let props = vec![("age".to_string(), Value::Int(30))];
        assert!(schema.validate_node_properties("Person", &props).is_ok());
    }

    #[test]
    fn test_property_type_wrong_fails() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyType {
                property: CompactString::new("age"),
                expected_type: ValueType::Int,
            },
        );
        let props = vec![(
            "age".to_string(),
            Value::String(CompactString::new("thirty")),
        )];
        let result = schema.validate_node_properties("Person", &props);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::InvalidConfig(msg) => {
                assert!(msg.contains("age"));
                assert!(msg.contains("Int"));
            }
            other => panic!("expected InvalidConfig, got: {other}"),
        }
    }

    #[test]
    fn test_property_type_null_is_allowed() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyType {
                property: CompactString::new("age"),
                expected_type: ValueType::Int,
            },
        );
        let props = vec![("age".to_string(), Value::Null)];
        assert!(schema.validate_node_properties("Person", &props).is_ok());
    }

    #[test]
    fn test_property_type_missing_property_passes() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyType {
                property: CompactString::new("age"),
                expected_type: ValueType::Int,
            },
        );
        let props = vec![("name".to_string(), Value::String(CompactString::new("Bob")))];
        assert!(schema.validate_node_properties("Person", &props).is_ok());
    }

    #[test]
    fn test_property_required_present_passes() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );
        let props = vec![(
            "name".to_string(),
            Value::String(CompactString::new("Alice")),
        )];
        assert!(schema.validate_node_properties("Person", &props).is_ok());
    }

    #[test]
    fn test_property_required_missing_fails() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );
        let props = vec![("age".to_string(), Value::Int(30))];
        let result = schema.validate_node_properties("Person", &props);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::InvalidConfig(msg) => {
                assert!(msg.contains("name"));
                assert!(msg.contains("missing"));
            }
            other => panic!("expected InvalidConfig, got: {other}"),
        }
    }

    #[test]
    fn test_property_required_null_fails() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );
        let props = vec![("name".to_string(), Value::Null)];
        let result = schema.validate_node_properties("Person", &props);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_schema_everything_passes() {
        let schema = GraphSchema::new();
        let props = vec![
            ("anything".to_string(), Value::Int(42)),
            (
                "goes".to_string(),
                Value::String(CompactString::new("here")),
            ),
        ];
        assert!(schema.validate_node_properties("Whatever", &props).is_ok());
        assert!(schema.validate_edge_properties("Whatever", &props).is_ok());
    }

    #[test]
    fn test_multiple_constraints_same_label() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyType {
                property: CompactString::new("age"),
                expected_type: ValueType::Int,
            },
        );

        // Both satisfied
        let props = vec![
            (
                "name".to_string(),
                Value::String(CompactString::new("Alice")),
            ),
            ("age".to_string(), Value::Int(30)),
        ];
        assert!(schema.validate_node_properties("Person", &props).is_ok());

        // Name missing -> fails
        let props2 = vec![("age".to_string(), Value::Int(30))];
        assert!(schema.validate_node_properties("Person", &props2).is_err());

        // Age wrong type -> fails
        let props3 = vec![
            ("name".to_string(), Value::String(CompactString::new("Bob"))),
            (
                "age".to_string(),
                Value::String(CompactString::new("thirty")),
            ),
        ];
        assert!(schema.validate_node_properties("Person", &props3).is_err());
    }

    #[test]
    fn test_node_vs_edge_schema_independence() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );
        schema.add_edge_constraint(
            "KNOWS",
            SchemaConstraint::PropertyType {
                property: CompactString::new("since"),
                expected_type: ValueType::Int,
            },
        );

        // Node schema should not apply to edges
        let edge_props = vec![("since".to_string(), Value::Int(2024))];
        assert!(
            schema
                .validate_edge_properties("KNOWS", &edge_props)
                .is_ok()
        );

        // Edge schema should not apply to nodes
        let node_props = vec![(
            "name".to_string(),
            Value::String(CompactString::new("Alice")),
        )];
        assert!(
            schema
                .validate_node_properties("Person", &node_props)
                .is_ok()
        );

        // Node without required name fails
        let bad_node_props = vec![("age".to_string(), Value::Int(30))];
        assert!(
            schema
                .validate_node_properties("Person", &bad_node_props)
                .is_err()
        );
    }

    #[test]
    fn test_edge_property_type_validation() {
        let mut schema = GraphSchema::new();
        schema.add_edge_constraint(
            "KNOWS",
            SchemaConstraint::PropertyType {
                property: CompactString::new("weight"),
                expected_type: ValueType::Float,
            },
        );

        let good_props = vec![("weight".to_string(), Value::Float(0.95))];
        assert!(
            schema
                .validate_edge_properties("KNOWS", &good_props)
                .is_ok()
        );

        let bad_props = vec![("weight".to_string(), Value::Int(1))];
        assert!(
            schema
                .validate_edge_properties("KNOWS", &bad_props)
                .is_err()
        );
    }

    #[test]
    fn test_schema_unregistered_label_passes() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );

        // "Company" has no schema, so anything passes
        let props = vec![("random".to_string(), Value::Bool(true))];
        assert!(schema.validate_node_properties("Company", &props).is_ok());
    }

    #[test]
    fn test_uniqueness_constraint_is_no_op_at_schema_level() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyUnique {
                property: CompactString::new("email"),
            },
        );

        // Uniqueness is enforced at engine level, schema validation should pass
        let props = vec![(
            "email".to_string(),
            Value::String(CompactString::new("alice@example.com")),
        )];
        assert!(schema.validate_node_properties("Person", &props).is_ok());
    }

    #[test]
    fn test_schema_serialization_roundtrip() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyRequired {
                property: CompactString::new("name"),
            },
        );
        schema.add_node_constraint(
            "Person",
            SchemaConstraint::PropertyType {
                property: CompactString::new("age"),
                expected_type: ValueType::Int,
            },
        );
        schema.add_edge_constraint(
            "KNOWS",
            SchemaConstraint::PropertyUnique {
                property: CompactString::new("id"),
            },
        );

        let json = serde_json::to_string(&schema).unwrap();
        let deserialized: GraphSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, deserialized);
    }

    #[test]
    fn test_property_type_bool() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Setting",
            SchemaConstraint::PropertyType {
                property: CompactString::new("enabled"),
                expected_type: ValueType::Bool,
            },
        );

        let good = vec![("enabled".to_string(), Value::Bool(true))];
        assert!(schema.validate_node_properties("Setting", &good).is_ok());

        let bad = vec![("enabled".to_string(), Value::Int(1))];
        assert!(schema.validate_node_properties("Setting", &bad).is_err());
    }

    #[test]
    fn test_property_type_string() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Document",
            SchemaConstraint::PropertyType {
                property: CompactString::new("title"),
                expected_type: ValueType::String,
            },
        );

        let good = vec![(
            "title".to_string(),
            Value::String(CompactString::new("Hello")),
        )];
        assert!(schema.validate_node_properties("Document", &good).is_ok());

        let bad = vec![("title".to_string(), Value::Int(42))];
        assert!(schema.validate_node_properties("Document", &bad).is_err());
    }

    #[test]
    fn test_property_type_float() {
        let mut schema = GraphSchema::new();
        schema.add_node_constraint(
            "Measurement",
            SchemaConstraint::PropertyType {
                property: CompactString::new("value"),
                expected_type: ValueType::Float,
            },
        );

        let good = vec![("value".to_string(), Value::Float(3.14))];
        assert!(
            schema
                .validate_node_properties("Measurement", &good)
                .is_ok()
        );

        let bad = vec![(
            "value".to_string(),
            Value::String(CompactString::new("3.14")),
        )];
        assert!(
            schema
                .validate_node_properties("Measurement", &bad)
                .is_err()
        );
    }
}
