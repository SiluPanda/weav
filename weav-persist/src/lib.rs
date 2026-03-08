//! Phase 5: Persistence Layer
//!
//! This crate provides durable storage for the Weav context graph database,
//! including a write-ahead log, snapshot engine, and recovery manager.

pub mod wal;
pub mod snapshot;
pub mod recovery;

/// Current time in milliseconds since the Unix epoch.
pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
