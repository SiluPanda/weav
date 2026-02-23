//! Recovery manager for restoring state from snapshots and WAL files.
//!
//! The recovery procedure:
//! 1. Find the latest valid snapshot.
//! 2. Load the snapshot (or start fresh if none exists).
//! 3. Find all WAL files in the data directory.
//! 4. Replay WAL entries that come after the snapshot's `wal_sequence`.
//! 5. Return a `RecoveryResult` with counts and any errors encountered.
//!
//! Note: The actual reconstruction of in-memory data structures (GraphShard,
//! AdjacencyStore, etc.) is handled by a higher layer. This module reads,
//! validates, and counts the entries.

use std::fs;
use std::io;
use std::path::PathBuf;

use crate::snapshot::SnapshotEngine;
use crate::wal::{compute_checksum, WalEntry, WalReader};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of a recovery operation.
#[derive(Debug)]
pub struct RecoveryResult {
    /// Number of snapshots that were successfully loaded (0 or 1).
    pub snapshots_loaded: u32,
    /// Number of WAL entries replayed after the snapshot.
    pub wal_entries_replayed: u64,
    /// Number of graphs found in the loaded snapshot (or 0).
    pub graphs_recovered: u32,
    /// Non-fatal errors encountered during recovery.
    pub errors: Vec<String>,
}

// ─── RecoveryManager ────────────────────────────────────────────────────────

/// Manages recovery by coordinating snapshots and WAL replay.
pub struct RecoveryManager {
    data_dir: PathBuf,
}

impl RecoveryManager {
    /// Create a new recovery manager operating on the given data directory.
    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    /// Run the full recovery procedure:
    ///
    /// 1. Load the latest snapshot (if any).
    /// 2. Collect all WAL files.
    /// 3. Replay WAL entries whose `seq` is greater than the snapshot's
    ///    `wal_sequence` (or all entries if no snapshot exists).
    /// 4. Return a summary.
    pub fn recover(&self) -> io::Result<RecoveryResult> {
        let mut result = RecoveryResult {
            snapshots_loaded: 0,
            wal_entries_replayed: 0,
            graphs_recovered: 0,
            errors: Vec::new(),
        };

        // Step 1: Try to load the latest snapshot.
        let snapshot_engine = SnapshotEngine::new(self.data_dir.clone());
        let wal_sequence_cutoff: u64;

        match snapshot_engine.latest_snapshot()? {
            Some(snap_path) => match snapshot_engine.load_snapshot(&snap_path) {
                Ok(snapshot) => {
                    result.snapshots_loaded = 1;
                    result.graphs_recovered = snapshot.meta.graph_count;
                    wal_sequence_cutoff = snapshot.meta.wal_sequence;
                }
                Err(e) => {
                    result
                        .errors
                        .push(format!("failed to load snapshot {}: {e}", snap_path.display()));
                    wal_sequence_cutoff = 0;
                }
            },
            None => {
                wal_sequence_cutoff = 0;
            }
        }

        // Step 2: Find and replay WAL files.
        let wal_files = self.find_wal_files()?;

        for wal_path in &wal_files {
            let reader = match WalReader::open(wal_path) {
                Ok(r) => r,
                Err(e) => {
                    result
                        .errors
                        .push(format!("failed to open WAL {}: {e}", wal_path.display()));
                    continue;
                }
            };

            for entry_result in reader {
                match entry_result {
                    Ok(entry) => {
                        if entry.seq > wal_sequence_cutoff {
                            if self.validate_wal_entry(&entry) {
                                result.wal_entries_replayed += 1;
                            } else {
                                result.errors.push(format!(
                                    "checksum mismatch for WAL entry seq {}",
                                    entry.seq
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        result.errors.push(format!(
                            "error reading WAL entry from {}: {e}",
                            wal_path.display()
                        ));
                        // Stop reading this WAL file on error (likely truncated).
                        break;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Find all WAL files (matching pattern `wal*`) in the data directory,
    /// sorted by name.
    pub fn find_wal_files(&self) -> io::Result<Vec<PathBuf>> {
        if !self.data_dir.exists() {
            return Ok(Vec::new());
        }

        let mut wal_files = Vec::new();

        for entry in fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            if name.starts_with("wal") {
                wal_files.push(path);
            }
        }

        wal_files.sort();
        Ok(wal_files)
    }

    /// Validate a WAL entry's checksum.
    pub fn validate_wal_entry(&self, entry: &WalEntry) -> bool {
        let op_bytes = match bincode::serialize(&entry.operation) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let expected = compute_checksum(&op_bytes);
        entry.checksum == expected
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::*;
    use crate::wal::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use weav_core::config::WalSyncMode;

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("weav_rec_test_{name}_{}", now_millis()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_recovery_empty_dir() {
        let dir = test_dir("empty");
        let mgr = RecoveryManager::new(dir.clone());
        let result = mgr.recover().unwrap();
        assert_eq!(result.snapshots_loaded, 0);
        assert_eq!(result.wal_entries_replayed, 0);
        assert_eq!(result.graphs_recovered, 0);
        assert!(result.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_recovery_wal_only() {
        let dir = test_dir("wal_only");
        let wal_path = dir.join("wal");

        {
            let mut wal =
                WriteAheadLog::new(wal_path, 1024 * 1024, WalSyncMode::Always).unwrap();
            wal.append(
                0,
                WalOperation::GraphCreate {
                    name: "g1".into(),
                    config_json: "{}".into(),
                },
            )
            .unwrap();
            wal.append(
                0,
                WalOperation::NodeAdd {
                    graph_id: 1,
                    node_id: 1,
                    label: "Person".into(),
                    properties_json: "{}".into(),
                    embedding: None,
                    entity_key: None,
                },
            )
            .unwrap();
        }

        let mgr = RecoveryManager::new(dir.clone());
        let result = mgr.recover().unwrap();
        assert_eq!(result.snapshots_loaded, 0);
        assert_eq!(result.wal_entries_replayed, 2);
        assert_eq!(result.graphs_recovered, 0);
        assert!(result.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_recovery_snapshot_only() {
        let dir = test_dir("snap_only");
        let engine = SnapshotEngine::new(dir.clone());

        let snap = FullSnapshot {
            meta: SnapshotMeta {
                path: PathBuf::new(),
                created_at: now_millis(),
                size_bytes: 0,
                node_count: 5,
                edge_count: 3,
                graph_count: 2,
                wal_sequence: 100,
            },
            graphs: vec![
                GraphSnapshot {
                    graph_id: 1,
                    graph_name: "g1".into(),
                    config_json: "{}".into(),
                    nodes: Vec::new(),
                    edges: Vec::new(),
                },
                GraphSnapshot {
                    graph_id: 2,
                    graph_name: "g2".into(),
                    config_json: "{}".into(),
                    nodes: Vec::new(),
                    edges: Vec::new(),
                },
            ],
        };
        engine.save_snapshot(&snap).unwrap();

        let mgr = RecoveryManager::new(dir.clone());
        let result = mgr.recover().unwrap();
        assert_eq!(result.snapshots_loaded, 1);
        assert_eq!(result.graphs_recovered, 2);
        assert_eq!(result.wal_entries_replayed, 0);
        assert!(result.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_recovery_snapshot_plus_wal() {
        let dir = test_dir("snap_wal");

        // Create a snapshot at wal_sequence = 3.
        let engine = SnapshotEngine::new(dir.clone());
        let snap = FullSnapshot {
            meta: SnapshotMeta {
                path: PathBuf::new(),
                created_at: now_millis(),
                size_bytes: 0,
                node_count: 1,
                edge_count: 0,
                graph_count: 1,
                wal_sequence: 3,
            },
            graphs: vec![GraphSnapshot {
                graph_id: 1,
                graph_name: "g1".into(),
                config_json: "{}".into(),
                nodes: vec![NodeSnapshot {
                    node_id: 1,
                    label: "Person".into(),
                    properties_json: "{}".into(),
                    embedding: None,
                    entity_key: None,
                }],
                edges: Vec::new(),
            }],
        };
        engine.save_snapshot(&snap).unwrap();

        // Create a WAL with 5 entries (seq 1..5).
        // Only entries with seq > 3 should be replayed.
        let wal_path = dir.join("wal");
        {
            let mut wal =
                WriteAheadLog::new(wal_path, 1024 * 1024, WalSyncMode::Always).unwrap();
            for i in 0..5 {
                wal.append(
                    0,
                    WalOperation::NodeAdd {
                        graph_id: 1,
                        node_id: 10 + i,
                        label: "X".into(),
                        properties_json: "{}".into(),
                        embedding: None,
                        entity_key: None,
                    },
                )
                .unwrap();
            }
        }

        let mgr = RecoveryManager::new(dir.clone());
        let result = mgr.recover().unwrap();
        assert_eq!(result.snapshots_loaded, 1);
        assert_eq!(result.graphs_recovered, 1);
        // Entries with seq 4 and 5 should be replayed.
        assert_eq!(result.wal_entries_replayed, 2);
        assert!(result.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_wal_files() {
        let dir = test_dir("find_wal");

        // Create some files with various names.
        std::fs::write(dir.join("wal"), b"data").unwrap();
        std::fs::write(dir.join("wal.12345"), b"data").unwrap();
        std::fs::write(dir.join("wal.99999"), b"data").unwrap();
        std::fs::write(dir.join("snapshot-1.bin"), b"data").unwrap();
        std::fs::write(dir.join("other.txt"), b"data").unwrap();

        let mgr = RecoveryManager::new(dir.clone());
        let files = mgr.find_wal_files().unwrap();
        assert_eq!(files.len(), 3);
        // Should be sorted by name.
        let names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["wal", "wal.12345", "wal.99999"]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_wal_entry() {
        let mgr = RecoveryManager::new(PathBuf::from("/tmp/unused"));

        let op = WalOperation::GraphCreate {
            name: "g".into(),
            config_json: "{}".into(),
        };
        let op_bytes = bincode::serialize(&op).unwrap();
        let checksum = compute_checksum(&op_bytes);

        let good_entry = WalEntry {
            seq: 1,
            timestamp: 1000,
            shard_id: 0,
            operation: op.clone(),
            checksum,
        };
        assert!(mgr.validate_wal_entry(&good_entry));

        let bad_entry = WalEntry {
            seq: 1,
            timestamp: 1000,
            shard_id: 0,
            operation: op,
            checksum: checksum.wrapping_add(1), // Corrupt checksum.
        };
        assert!(!mgr.validate_wal_entry(&bad_entry));
    }

    #[test]
    fn test_find_wal_files_nonexistent_dir() {
        let dir = PathBuf::from("/tmp/weav_nonexistent_dir_test_12345");
        let mgr = RecoveryManager::new(dir);
        let files = mgr.find_wal_files().unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_recovery_with_rotated_wal() {
        let dir = test_dir("rotated_wal");

        // Create a WAL, write some entries, rotate, write more.
        let wal_path = dir.join("wal");
        {
            let mut wal =
                WriteAheadLog::new(wal_path, 50, WalSyncMode::Always).unwrap();
            wal.append(
                0,
                WalOperation::GraphCreate {
                    name: "test-graph-with-a-longer-name".into(),
                    config_json: r#"{"key":"value"}"#.into(),
                },
            )
            .unwrap();

            // Rotate.
            let _rotated = wal.rotate().unwrap();

            // Write another entry to the new WAL.
            wal.append(
                0,
                WalOperation::NodeAdd {
                    graph_id: 1,
                    node_id: 1,
                    label: "Person".into(),
                    properties_json: "{}".into(),
                    embedding: None,
                    entity_key: None,
                },
            )
            .unwrap();
        }

        let mgr = RecoveryManager::new(dir.clone());
        let wal_files = mgr.find_wal_files().unwrap();
        assert_eq!(wal_files.len(), 2);

        let result = mgr.recover().unwrap();
        assert_eq!(result.wal_entries_replayed, 2);
        assert!(result.errors.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }
}
