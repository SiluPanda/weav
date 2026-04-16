//! Configuration system for Weav.

use crate::schema::GraphSchema;
use crate::types::{ConflictPolicy, ResolutionMode};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level Weav configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct WeavConfig {
    pub server: ServerConfig,
    pub engine: EngineConfig,
    pub persistence: PersistenceConfig,
    pub memory: MemoryConfig,
    pub auth: AuthConfig,
    pub extract: ExtractConfig,
}

impl WeavConfig {
    /// Load configuration from a TOML file, with environment variable overrides.
    ///
    /// Environment variables use the `WEAV_` prefix and `_` separators.
    /// E.g. `WEAV_SERVER_PORT=6380`.
    pub fn load(path: Option<&str>) -> Result<Self, crate::error::WeavError> {
        let mut config = if let Some(path) = path {
            let contents = std::fs::read_to_string(path).map_err(|e| {
                crate::error::WeavError::InvalidConfig(format!(
                    "failed to read config file '{path}': {e}"
                ))
            })?;
            toml::from_str::<WeavConfig>(&contents).map_err(|e| {
                crate::error::WeavError::InvalidConfig(format!("failed to parse config: {e}"))
            })?
        } else {
            WeavConfig::default()
        };

        // Apply environment variable overrides.
        config.apply_env_overrides();
        config.validate()?;
        Ok(config)
    }

    fn apply_env_overrides(&mut self) {
        macro_rules! env_override {
            ($var:expr, $field:expr) => {
                if let Ok(v) = std::env::var($var) {
                    if let Ok(val) = v.parse() {
                        $field = val;
                    }
                }
            };
        }
        macro_rules! env_override_str {
            ($var:expr, $field:expr) => {
                if let Ok(v) = std::env::var($var) {
                    $field = v.into();
                }
            };
        }
        macro_rules! env_override_opt {
            ($var:expr, $field:expr) => {
                if let Ok(v) = std::env::var($var) {
                    $field = Some(v);
                }
            };
        }
        macro_rules! env_override_opt_parse {
            ($var:expr, $field:expr) => {
                if let Ok(v) = std::env::var($var) {
                    if let Ok(val) = v.parse() {
                        $field = Some(val);
                    }
                }
            };
        }

        env_override!("WEAV_SERVER_PORT", self.server.port);
        env_override_opt_parse!("WEAV_SERVER_HTTP_PORT", self.server.http_port);
        env_override_opt_parse!("WEAV_SERVER_GRPC_PORT", self.server.grpc_port);
        env_override_str!("WEAV_SERVER_BIND_ADDRESS", self.server.bind_address);
        env_override!("WEAV_TLS_ENABLED", self.server.tls_enabled);
        env_override_opt!("WEAV_TLS_CERT", self.server.tls_cert_path);
        env_override_opt!("WEAV_TLS_KEY", self.server.tls_key_path);
        env_override_opt_parse!("WEAV_ENGINE_NUM_SHARDS", self.engine.num_shards);
        env_override!("WEAV_PERSISTENCE_ENABLED", self.persistence.enabled);
        env_override_str!("WEAV_PERSISTENCE_DATA_DIR", self.persistence.data_dir);
        env_override_opt_parse!("WEAV_MEMORY_MAX_MEMORY_MB", self.memory.max_memory_mb);
        env_override!("WEAV_AUTH_ENABLED", self.auth.enabled);
        env_override!("WEAV_AUTH_REQUIRE_AUTH", self.auth.require_auth);
        env_override_opt!("WEAV_AUTH_DEFAULT_PASSWORD", self.auth.default_password);

        // Extract config overrides.
        env_override!("WEAV_EXTRACT_ENABLED", self.extract.enabled);
        env_override_str!("WEAV_EXTRACT_LLM_BACKEND", self.extract.llm_backend);
        env_override_opt!("WEAV_EXTRACT_LLM_API_KEY", self.extract.llm_api_key);
        env_override_str!(
            "WEAV_EXTRACT_EXTRACTION_MODEL",
            self.extract.extraction_model
        );
        env_override_opt!("WEAV_EXTRACT_LLM_BASE_URL", self.extract.llm_base_url);
        env_override_str!(
            "WEAV_EXTRACT_EMBEDDING_BACKEND",
            self.extract.embedding_backend
        );
        env_override_opt!(
            "WEAV_EXTRACT_EMBEDDING_API_KEY",
            self.extract.embedding_api_key
        );
        env_override_str!("WEAV_EXTRACT_EMBEDDING_MODEL", self.extract.embedding_model);
        env_override!(
            "WEAV_EXTRACT_EMBEDDING_DIMENSIONS",
            self.extract.embedding_dimensions
        );
        env_override!("WEAV_EXTRACT_CHUNK_SIZE", self.extract.chunk_size);
        env_override!("WEAV_EXTRACT_CHUNK_OVERLAP", self.extract.chunk_overlap);
        env_override!(
            "WEAV_EXTRACT_MAX_EXTRACTION_TOKENS",
            self.extract.max_extraction_tokens
        );
        if let Ok(v) = std::env::var("WEAV_EXTRACT_RESOLUTION_MODE")
            && let Some(mode) = ResolutionMode::from_str_lossy(&v)
        {
            self.extract.resolution_mode = mode;
        }
        env_override!(
            "WEAV_EXTRACT_LINK_EXISTING_ENTITIES",
            self.extract.link_existing_entities
        );
        env_override!(
            "WEAV_EXTRACT_RESOLUTION_CANDIDATE_LIMIT",
            self.extract.resolution_candidate_limit
        );
        env_override!(
            "WEAV_EXTRACT_MAX_CONCURRENT_LLM_CALLS",
            self.extract.max_concurrent_llm_calls
        );
        env_override!(
            "WEAV_EXTRACT_MAX_CONCURRENT_EMBEDDING_CALLS",
            self.extract.max_concurrent_embedding_calls
        );
        env_override!(
            "WEAV_EXTRACT_EMBEDDING_BATCH_SIZE",
            self.extract.embedding_batch_size
        );
    }

    fn validate(&self) -> Result<(), crate::error::WeavError> {
        if self.server.port == 0 {
            return Err(crate::error::WeavError::InvalidConfig(
                "server.port must be > 0".into(),
            ));
        }
        if let Some(n) = self.engine.num_shards
            && n == 0
        {
            return Err(crate::error::WeavError::InvalidConfig(
                "engine.num_shards must be > 0".into(),
            ));
        }
        if self.server.tls_enabled
            && (self.server.tls_cert_path.is_none() || self.server.tls_key_path.is_none())
        {
            return Err(crate::error::WeavError::InvalidConfig(
                "TLS enabled but tls_cert_path and tls_key_path must be set".into(),
            ));
        }
        if self.engine.default_vector_dimensions == 0 {
            return Err(crate::error::WeavError::InvalidConfig(
                "engine.default_vector_dimensions must be > 0".into(),
            ));
        }
        if self.engine.max_vector_dimensions < self.engine.default_vector_dimensions {
            return Err(crate::error::WeavError::InvalidConfig(
                "engine.max_vector_dimensions must be >= default_vector_dimensions".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub grpc_port: Option<u16>,
    pub http_port: Option<u16>,
    pub max_connections: u32,
    pub tcp_keepalive_secs: u32,
    pub read_timeout_ms: u64,
    /// Enable TLS encryption for all protocols. Requires cert_path and key_path.
    pub tls_enabled: bool,
    /// Path to PEM-encoded TLS certificate file.
    pub tls_cert_path: Option<String>,
    /// Path to PEM-encoded TLS private key file.
    pub tls_key_path: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".into(),
            port: 6380,
            grpc_port: Some(6381),
            http_port: Some(6382),
            max_connections: 10_000,
            tcp_keepalive_secs: 300,
            read_timeout_ms: 30_000,
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub num_shards: Option<u16>,
    pub message_bus_buffer: usize,
    pub default_vector_dimensions: u16,
    pub max_vector_dimensions: u16,
    pub default_hnsw_m: u16,
    pub default_hnsw_ef_construction: u16,
    pub default_hnsw_ef_search: u16,
    pub default_conflict_policy: ConflictPolicy,
    pub enable_temporal: bool,
    pub enable_provenance: bool,
    pub token_counter: TokenCounterType,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_shards: None, // Will default to num_cpus at runtime
            message_bus_buffer: 4096,
            default_vector_dimensions: 1536,
            max_vector_dimensions: 4096,
            default_hnsw_m: 16,
            default_hnsw_ef_construction: 200,
            default_hnsw_ef_search: 50,
            default_conflict_policy: ConflictPolicy::LastWriteWins,
            enable_temporal: true,
            enable_provenance: true,
            token_counter: TokenCounterType::CharDiv4,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub data_dir: PathBuf,
    pub wal_enabled: bool,
    pub wal_sync_mode: WalSyncMode,
    pub snapshot_interval_secs: u64,
    pub max_wal_size_mb: u64,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            data_dir: PathBuf::from("./weav-data"),
            wal_enabled: true,
            wal_sync_mode: WalSyncMode::EverySecond,
            snapshot_interval_secs: 3600,
            max_wal_size_mb: 256,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub enum WalSyncMode {
    Always,
    #[default]
    EverySecond,
    Never,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub max_memory_mb: Option<u64>,
    pub eviction_policy: EvictionPolicy,
    pub arena_size_mb: u16,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            eviction_policy: EvictionPolicy::NoEviction,
            arena_size_mb: 64,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum EvictionPolicy {
    #[default]
    NoEviction,
    LRU,
    RelevanceDecay,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub enum TokenCounterType {
    #[serde(rename = "tiktoken_cl100k")]
    TiktokenCl100k,
    #[serde(rename = "tiktoken_o200k")]
    TiktokenO200k,
    #[default]
    #[serde(rename = "char_div_4")]
    CharDiv4,
    Exact(String),
}

/// Extraction pipeline configuration (LLM-powered ingestion).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ExtractConfig {
    /// Enable the extraction pipeline. When false, INGEST commands are rejected.
    pub enabled: bool,
    /// LLM backend for entity extraction: "openai", "anthropic", "ollama", etc.
    pub llm_backend: String,
    /// API key for the extraction LLM.
    pub llm_api_key: Option<String>,
    /// Model name for entity/relationship extraction.
    pub extraction_model: String,
    /// Custom base URL for the extraction LLM (e.g. local Ollama).
    pub llm_base_url: Option<String>,
    /// LLM backend for embedding generation.
    pub embedding_backend: String,
    /// API key for the embedding model (defaults to llm_api_key if not set).
    pub embedding_api_key: Option<String>,
    /// Model name for embedding generation.
    pub embedding_model: String,
    /// Output dimensions of the embedding model.
    pub embedding_dimensions: u16,
    /// Target chunk size in tokens.
    pub chunk_size: usize,
    /// Overlap between chunks in tokens.
    pub chunk_overlap: usize,
    /// Maximum tokens to send to the LLM per extraction call.
    pub max_extraction_tokens: usize,
    /// Temperature for extraction LLM calls.
    pub extraction_temperature: f32,
    /// Alias resolution mode for extracted entities.
    pub resolution_mode: ResolutionMode,
    /// When true, ingest may link extracted entities to existing graph-local entities.
    pub link_existing_entities: bool,
    /// Maximum number of resolution candidates to consider per entity.
    pub resolution_candidate_limit: usize,
    /// Maximum concurrent LLM extraction calls.
    pub max_concurrent_llm_calls: usize,
    /// Maximum concurrent embedding API calls.
    pub max_concurrent_embedding_calls: usize,
    /// Number of texts per embedding API batch call.
    pub embedding_batch_size: usize,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            llm_backend: "openai".into(),
            llm_api_key: None,
            extraction_model: "gpt-4.1-mini".into(),
            llm_base_url: None,
            embedding_backend: "openai".into(),
            embedding_api_key: None,
            embedding_model: "text-embedding-3-small".into(),
            embedding_dimensions: 1536,
            chunk_size: 512,
            chunk_overlap: 50,
            max_extraction_tokens: 4096,
            extraction_temperature: 0.0,
            resolution_mode: ResolutionMode::Heuristic,
            link_existing_entities: true,
            resolution_candidate_limit: 8,
            max_concurrent_llm_calls: 4,
            max_concurrent_embedding_calls: 8,
            embedding_batch_size: 32,
        }
    }
}

/// Authentication and authorization configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Enable the auth subsystem. When false, all auth checks are skipped.
    pub enabled: bool,
    /// Require authentication for all connections. When false and auth is enabled,
    /// unauthenticated connections get default (full) permissions.
    pub require_auth: bool,
    /// Path to an ACL file for persistent user definitions.
    pub acl_file: Option<PathBuf>,
    /// A default password for Redis-compat single-password AUTH.
    pub default_password: Option<String>,
    /// Statically-defined users from config.
    pub users: Vec<UserConfig>,
}

/// A user definition in the config file.
#[derive(Debug, Clone, Deserialize)]
pub struct UserConfig {
    pub username: String,
    pub password: Option<String>,
    /// Command category permissions, e.g. `["+@read", "+@write", "+@admin"]`
    #[serde(default)]
    pub categories: Vec<String>,
    /// Graph-level access patterns.
    #[serde(default)]
    pub graph_patterns: Vec<GraphPatternConfig>,
    /// API keys for this user.
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Whether the user is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

/// Graph-level access pattern configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphPatternConfig {
    pub pattern: String,
    pub permission: String,
}

/// Per-graph configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphConfig {
    pub default_conflict_policy: ConflictPolicy,
    pub default_decay: crate::types::DecayFunction,
    pub max_nodes: Option<u64>,
    pub max_edges: Option<u64>,
    pub enable_temporal: bool,
    pub enable_provenance: bool,
    pub vector_dimensions: u16,
    pub auto_dedup_threshold: Option<f32>,
    /// Default TTL in milliseconds for nodes/edges created in this graph.
    /// If set, new nodes/edges without an explicit TTL will inherit this value.
    pub default_ttl_ms: Option<u64>,
    /// Schema constraints for property validation on nodes and edges.
    pub schema: GraphSchema,
    /// Eviction policy for this graph when at capacity.
    #[serde(default)]
    pub eviction_policy: EvictionPolicy,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            default_conflict_policy: ConflictPolicy::LastWriteWins,
            default_decay: crate::types::DecayFunction::None,
            max_nodes: None,
            max_edges: None,
            enable_temporal: true,
            enable_provenance: true,
            vector_dimensions: 1536,
            auto_dedup_threshold: None,
            default_ttl_ms: None,
            schema: GraphSchema::new(),
            eviction_policy: EvictionPolicy::NoEviction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WeavConfig::default();
        assert_eq!(config.server.port, 6380);
        assert_eq!(config.engine.default_vector_dimensions, 1536);
        assert!(!config.persistence.enabled);
    }

    #[test]
    fn test_config_validation_zero_port() {
        let mut config = WeavConfig::default();
        config.server.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_shards() {
        let mut config = WeavConfig::default();
        config.engine.num_shards = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_dimension_mismatch() {
        let mut config = WeavConfig::default();
        config.engine.default_vector_dimensions = 2048;
        config.engine.max_vector_dimensions = 1024;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_deserialization() {
        let toml_str = r#"
[server]
port = 7000
bind_address = "127.0.0.1"

[engine]
num_shards = 4
default_vector_dimensions = 768

[persistence]
enabled = true
data_dir = "/tmp/weav-data"

[memory]
max_memory_mb = 8192
"#;
        let config: WeavConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 7000);
        assert_eq!(config.engine.num_shards, Some(4));
        assert_eq!(config.engine.default_vector_dimensions, 768);
        assert!(config.persistence.enabled);
        assert_eq!(config.memory.max_memory_mb, Some(8192));
    }

    #[test]
    fn test_config_load_none_returns_default() {
        let config = WeavConfig::load(None).unwrap();
        assert_eq!(config.server.port, 6380);
        assert_eq!(config.engine.default_vector_dimensions, 1536);
        assert!(!config.persistence.enabled);
    }

    #[test]
    fn test_config_validate_success() {
        let config = WeavConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_dimensions() {
        let mut config = WeavConfig::default();
        config.engine.default_vector_dimensions = 0;
        let result = config.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::error::WeavError::InvalidConfig(msg) => {
                assert!(msg.contains("default_vector_dimensions"));
            }
            other => panic!("expected InvalidConfig, got: {other}"),
        }
    }

    #[test]
    fn test_wal_sync_mode_default() {
        let mode = WalSyncMode::default();
        assert!(matches!(mode, WalSyncMode::EverySecond));
    }

    #[test]
    fn test_eviction_policy_default() {
        let policy = EvictionPolicy::default();
        assert!(matches!(policy, EvictionPolicy::NoEviction));
    }

    #[test]
    fn test_server_config_defaults() {
        let sc = ServerConfig::default();
        assert_eq!(sc.bind_address, "0.0.0.0");
        assert_eq!(sc.port, 6380);
        assert_eq!(sc.grpc_port, Some(6381));
        assert_eq!(sc.http_port, Some(6382));
        assert_eq!(sc.max_connections, 10_000);
        assert_eq!(sc.tcp_keepalive_secs, 300);
        assert_eq!(sc.read_timeout_ms, 30_000);
    }

    #[test]
    fn test_engine_config_defaults() {
        let ec = EngineConfig::default();
        assert_eq!(ec.num_shards, None);
        assert_eq!(ec.message_bus_buffer, 4096);
        assert_eq!(ec.default_vector_dimensions, 1536);
        assert_eq!(ec.max_vector_dimensions, 4096);
        assert_eq!(ec.default_hnsw_m, 16);
        assert_eq!(ec.default_hnsw_ef_construction, 200);
        assert_eq!(ec.default_hnsw_ef_search, 50);
        assert_eq!(ec.default_conflict_policy, ConflictPolicy::LastWriteWins);
        assert!(ec.enable_temporal);
        assert!(ec.enable_provenance);
        assert!(matches!(ec.token_counter, TokenCounterType::CharDiv4));
    }

    #[test]
    fn test_persistence_config_defaults() {
        let pc = PersistenceConfig::default();
        assert!(!pc.enabled);
        assert_eq!(pc.data_dir, PathBuf::from("./weav-data"));
        assert!(pc.wal_enabled);
        assert!(matches!(pc.wal_sync_mode, WalSyncMode::EverySecond));
        assert_eq!(pc.snapshot_interval_secs, 3600);
        assert_eq!(pc.max_wal_size_mb, 256);
    }

    // ── Round 1 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_config_validate_dimensions_equal() {
        let mut config = WeavConfig::default();
        config.engine.default_vector_dimensions = 2048;
        config.engine.max_vector_dimensions = 2048;
        // max >= default, should pass
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_both_dimensions_zero() {
        let mut config = WeavConfig::default();
        config.engine.default_vector_dimensions = 0;
        config.engine.max_vector_dimensions = 0;
        // default = 0 should fail first
        let err = config.validate().unwrap_err();
        match err {
            crate::error::WeavError::InvalidConfig(msg) => {
                assert!(msg.contains("default_vector_dimensions"));
            }
            other => panic!("expected InvalidConfig, got: {other}"),
        }
    }

    #[test]
    fn test_toml_unknown_fields_ignored() {
        let toml_str = r#"
[server]
port = 6380
unknown_field = "should be ignored"

[engine]
nonexistent_setting = 42
"#;
        let config: WeavConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 6380);
    }

    #[test]
    fn test_toml_empty_string_produces_defaults() {
        let config: WeavConfig = toml::from_str("").unwrap();
        assert_eq!(config.server.port, 6380);
        assert_eq!(config.engine.default_vector_dimensions, 1536);
        assert!(!config.persistence.enabled);
    }

    #[test]
    fn test_toml_partial_config() {
        let toml_str = r#"
[server]
port = 9999
"#;
        let config: WeavConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.port, 9999);
        // Everything else should be defaults
        assert_eq!(config.engine.default_vector_dimensions, 1536);
        assert_eq!(config.engine.num_shards, None);
        assert!(!config.persistence.enabled);
        assert_eq!(config.memory.max_memory_mb, None);
    }

    #[test]
    fn test_auth_config_defaults() {
        let ac = AuthConfig::default();
        assert!(!ac.enabled);
        assert!(!ac.require_auth);
        assert!(ac.acl_file.is_none());
        assert!(ac.default_password.is_none());
        assert!(ac.users.is_empty());
    }

    #[test]
    fn test_auth_config_in_weav_config_default() {
        let config = WeavConfig::default();
        assert!(!config.auth.enabled);
        assert!(!config.auth.require_auth);
    }

    #[test]
    fn test_toml_auth_config() {
        let toml_str = r#"
[auth]
enabled = true
require_auth = true
default_password = "mysecret"

[[auth.users]]
username = "admin"
password = "supersecret"
categories = ["+@read", "+@write", "+@admin"]

[[auth.users]]
username = "reader"
password = "readonly123"
categories = ["+@read", "+@connection"]
graph_patterns = [{ pattern = "*", permission = "read" }]
api_keys = ["wk_live_abc123"]
"#;
        let config: WeavConfig = toml::from_str(toml_str).unwrap();
        assert!(config.auth.enabled);
        assert!(config.auth.require_auth);
        assert_eq!(config.auth.default_password, Some("mysecret".into()));
        assert_eq!(config.auth.users.len(), 2);

        let admin = &config.auth.users[0];
        assert_eq!(admin.username, "admin");
        assert_eq!(admin.password, Some("supersecret".into()));
        assert_eq!(admin.categories, vec!["+@read", "+@write", "+@admin"]);
        assert!(admin.graph_patterns.is_empty());
        assert!(admin.enabled);

        let reader = &config.auth.users[1];
        assert_eq!(reader.username, "reader");
        assert_eq!(reader.categories, vec!["+@read", "+@connection"]);
        assert_eq!(reader.graph_patterns.len(), 1);
        assert_eq!(reader.graph_patterns[0].pattern, "*");
        assert_eq!(reader.graph_patterns[0].permission, "read");
        assert_eq!(reader.api_keys, vec!["wk_live_abc123"]);
    }

    #[test]
    fn test_graph_config_defaults() {
        let gc = GraphConfig::default();
        assert_eq!(
            gc.default_conflict_policy,
            crate::types::ConflictPolicy::LastWriteWins
        );
        assert_eq!(gc.default_decay, crate::types::DecayFunction::None);
        assert_eq!(gc.max_nodes, None);
        assert_eq!(gc.max_edges, None);
        assert!(gc.enable_temporal);
        assert!(gc.enable_provenance);
        assert_eq!(gc.vector_dimensions, 1536);
        assert_eq!(gc.auto_dedup_threshold, None);
    }

    #[test]
    fn test_extract_config_defaults() {
        let ec = ExtractConfig::default();
        assert!(!ec.enabled);
        assert_eq!(ec.llm_backend, "openai");
        assert_eq!(ec.llm_api_key, None);
        assert_eq!(ec.extraction_model, "gpt-4.1-mini");
        assert_eq!(ec.llm_base_url, None);
        assert_eq!(ec.embedding_backend, "openai");
        assert_eq!(ec.embedding_api_key, None);
        assert_eq!(ec.embedding_model, "text-embedding-3-small");
        assert_eq!(ec.embedding_dimensions, 1536);
        assert_eq!(ec.chunk_size, 512);
        assert_eq!(ec.chunk_overlap, 50);
        assert_eq!(ec.max_extraction_tokens, 4096);
        assert_eq!(ec.extraction_temperature, 0.0);
        assert_eq!(ec.resolution_mode, ResolutionMode::Heuristic);
        assert!(ec.link_existing_entities);
        assert_eq!(ec.resolution_candidate_limit, 8);
        assert_eq!(ec.max_concurrent_llm_calls, 4);
        assert_eq!(ec.max_concurrent_embedding_calls, 8);
        assert_eq!(ec.embedding_batch_size, 32);
    }

    #[test]
    fn test_extract_config_in_weav_config() {
        let config = WeavConfig::default();
        assert!(!config.extract.enabled);
        assert_eq!(config.extract.llm_backend, "openai");
    }

    #[test]
    fn test_toml_extract_config() {
        let toml_str = r#"
[extract]
enabled = true
llm_backend = "anthropic"
llm_api_key = "sk-test-123"
extraction_model = "claude-sonnet-4-20250514"
embedding_model = "text-embedding-3-large"
embedding_dimensions = 3072
chunk_size = 1024
chunk_overlap = 100
resolution_mode = "semantic"
link_existing_entities = false
resolution_candidate_limit = 12
"#;
        let config: WeavConfig = toml::from_str(toml_str).unwrap();
        assert!(config.extract.enabled);
        assert_eq!(config.extract.llm_backend, "anthropic");
        assert_eq!(config.extract.llm_api_key, Some("sk-test-123".into()));
        assert_eq!(config.extract.extraction_model, "claude-sonnet-4-20250514");
        assert_eq!(config.extract.embedding_model, "text-embedding-3-large");
        assert_eq!(config.extract.embedding_dimensions, 3072);
        assert_eq!(config.extract.chunk_size, 1024);
        assert_eq!(config.extract.chunk_overlap, 100);
        assert_eq!(config.extract.resolution_mode, ResolutionMode::Semantic);
        assert!(!config.extract.link_existing_entities);
        assert_eq!(config.extract.resolution_candidate_limit, 12);
        // Defaults preserved for unset fields
        assert_eq!(config.extract.max_concurrent_llm_calls, 4);
        assert_eq!(config.extract.embedding_batch_size, 32);
    }
}
