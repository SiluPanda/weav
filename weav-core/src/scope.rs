//! Scope identifiers and graph-name resolution helpers.

use serde::{Deserialize, Serialize};

use crate::error::{WeavError, WeavResult};

/// Hierarchical request scope used to derive a canonical graph name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl ScopeRef {
    /// Returns `true` when no scope component is set.
    pub fn is_empty(&self) -> bool {
        self.workspace_id.is_none()
            && self.user_id.is_none()
            && self.agent_id.is_none()
            && self.session_id.is_none()
    }

    /// Resolve the scope into its canonical graph name.
    pub fn resolve_graph_name(&self) -> WeavResult<String> {
        let workspace_id = normalized_component(self.workspace_id.as_deref(), "workspace_id")?;
        let user_id = normalized_optional_component(self.user_id.as_deref(), "user_id")?;
        let agent_id = normalized_optional_component(self.agent_id.as_deref(), "agent_id")?;
        let session_id = normalized_optional_component(self.session_id.as_deref(), "session_id")?;

        if agent_id.is_some() && user_id.is_none() {
            return Err(WeavError::InvalidConfig(
                "scope requires 'user_id' before 'agent_id'".to_string(),
            ));
        }

        if session_id.is_some() && agent_id.is_none() {
            return Err(WeavError::InvalidConfig(
                "scope requires 'agent_id' before 'session_id'".to_string(),
            ));
        }

        let mut graph = format!("ws:{workspace_id}");
        if let Some(user_id) = user_id {
            graph.push_str(":user:");
            graph.push_str(&user_id);
        }
        if let Some(agent_id) = agent_id {
            graph.push_str(":agent:");
            graph.push_str(&agent_id);
        }
        if let Some(session_id) = session_id {
            graph.push_str(":session:");
            graph.push_str(&session_id);
        }

        Ok(graph)
    }
}

/// Resolve a graph reference, preferring an explicit graph name over a scope.
pub fn resolve_graph_ref(graph: Option<&str>, scope: Option<&ScopeRef>) -> WeavResult<String> {
    if let Some(graph) = graph.map(str::trim).filter(|graph| !graph.is_empty()) {
        return Ok(graph.to_string());
    }

    match scope {
        Some(scope) if !scope.is_empty() => scope.resolve_graph_name(),
        _ => Err(WeavError::InvalidConfig(
            "either 'graph' or 'scope' must be provided".to_string(),
        )),
    }
}

fn normalized_component(value: Option<&str>, field: &str) -> WeavResult<String> {
    normalized_optional_component(value, field)?
        .ok_or_else(|| WeavError::InvalidConfig(format!("scope requires non-empty '{field}'")))
}

fn normalized_optional_component(value: Option<&str>, field: &str) -> WeavResult<Option<String>> {
    match value {
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Err(WeavError::InvalidConfig(format!(
                    "scope field '{field}' cannot be empty"
                )))
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_workspace_scope() {
        let scope = ScopeRef {
            workspace_id: Some("acme".to_string()),
            user_id: None,
            agent_id: None,
            session_id: None,
        };

        assert_eq!(scope.resolve_graph_name().unwrap(), "ws:acme");
    }

    #[test]
    fn resolves_full_scope_hierarchy() {
        let scope = ScopeRef {
            workspace_id: Some("acme".to_string()),
            user_id: Some("u_123".to_string()),
            agent_id: Some("a_support".to_string()),
            session_id: Some("s_456".to_string()),
        };

        assert_eq!(
            scope.resolve_graph_name().unwrap(),
            "ws:acme:user:u_123:agent:a_support:session:s_456"
        );
    }

    #[test]
    fn rejects_missing_workspace() {
        let err = ScopeRef::default().resolve_graph_name().unwrap_err();
        assert!(matches!(err, WeavError::InvalidConfig(_)));
        assert!(err.to_string().contains("workspace_id"));
    }

    #[test]
    fn rejects_agent_without_user() {
        let err = ScopeRef {
            workspace_id: Some("acme".to_string()),
            user_id: None,
            agent_id: Some("agent".to_string()),
            session_id: None,
        }
        .resolve_graph_name()
        .unwrap_err();

        assert!(err.to_string().contains("user_id"));
    }

    #[test]
    fn rejects_session_without_agent() {
        let err = ScopeRef {
            workspace_id: Some("acme".to_string()),
            user_id: Some("user".to_string()),
            agent_id: None,
            session_id: Some("session".to_string()),
        }
        .resolve_graph_name()
        .unwrap_err();

        assert!(err.to_string().contains("agent_id"));
    }

    #[test]
    fn explicit_graph_wins_over_scope() {
        let scope = ScopeRef {
            workspace_id: Some("acme".to_string()),
            user_id: Some("user".to_string()),
            agent_id: None,
            session_id: None,
        };

        assert_eq!(
            resolve_graph_ref(Some("explicit_graph"), Some(&scope)).unwrap(),
            "explicit_graph"
        );
    }

    #[test]
    fn blank_graph_falls_back_to_scope() {
        let scope = ScopeRef {
            workspace_id: Some("acme".to_string()),
            user_id: Some("user".to_string()),
            agent_id: None,
            session_id: None,
        };

        assert_eq!(
            resolve_graph_ref(Some("   "), Some(&scope)).unwrap(),
            "ws:acme:user:user"
        );
    }
}
