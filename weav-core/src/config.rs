//! Configuration system for Weav.

use crate::types::ConflictPolicy;
use serde::Deserialize;
use std::path::PathBuf;

/// Top-level Weav configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WeavConfig {
    pub server: ServerConfig,
    pub engine: EngineConfig,
    pub persistence: PersistenceConfig,
    pub memory: MemoryConfig,
}

impl Default for WeavConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
            persistence: PersistenceConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
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
        if let Ok(v) = std::env::var("WEAV_SERVER_PORT") {
            if let Ok(port) = v.parse() {
                self.server.port = port;
            }
        }
        if let Ok(v) = std::env::var("WEAV_SERVER_BIND_ADDRESS") {
            self.server.bind_address = v;
        }
        if let Ok(v) = std::env::var("WEAV_ENGINE_NUM_SHARDS") {
            if let Ok(n) = v.parse() {
                self.engine.num_shards = Some(n);
            }
        }
        if let Ok(v) = std::env::var("WEAV_PERSISTENCE_ENABLED") {
            if let Ok(b) = v.parse() {
                self.persistence.enabled = b;
            }
        }
        if let Ok(v) = std::env::var("WEAV_PERSISTENCE_DATA_DIR") {
            self.persistence.data_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("WEAV_MEMORY_MAX_MEMORY_MB") {
            if let Ok(n) = v.parse() {
                self.memory.max_memory_mb = Some(n);
            }
        }
    }

    fn validate(&self) -> Result<(), crate::error::WeavError> {
        if self.server.port == 0 {
            return Err(crate::error::WeavError::InvalidConfig(
                "server.port must be > 0".into(),
            ));
        }
        if let Some(n) = self.engine.num_shards {
            if n == 0 {
                return Err(crate::error::WeavError::InvalidConfig(
                    "engine.num_shards must be > 0".into(),
                ));
            }
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

#[derive(Debug, Clone, Deserialize)]
pub enum WalSyncMode {
    Always,
    EverySecond,
    Never,
}

impl Default for WalSyncMode {
    fn default() -> Self {
        WalSyncMode::EverySecond
    }
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

#[derive(Debug, Clone, Deserialize)]
pub enum EvictionPolicy {
    NoEviction,
    LRU,
    RelevanceDecay,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy::NoEviction
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum TokenCounterType {
    #[serde(rename = "tiktoken_cl100k")]
    TiktokenCl100k,
    #[serde(rename = "tiktoken_o200k")]
    TiktokenO200k,
    #[serde(rename = "char_div_4")]
    CharDiv4,
    Exact(String),
}

impl Default for TokenCounterType {
    fn default() -> Self {
        TokenCounterType::CharDiv4
    }
}

/// Per-graph configuration.
#[derive(Debug, Clone, Deserialize)]
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
}
