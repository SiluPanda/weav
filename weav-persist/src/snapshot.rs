//! Snapshot engine for periodic full-state persistence.
//!
//! Snapshots capture the entire state of all graphs at a point in time and are
//! written as bincode-serialized binary files.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use weav_core::types::*;

// ─── SnapshotFormat ─────────────────────────────────────────────────────────

/// Serialization format for snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotFormat {
    Bincode,
    // Rkyv support planned for future versions
}

impl Default for SnapshotFormat {
    fn default() -> Self {
        SnapshotFormat::Bincode
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// Metadata about a snapshot file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub path: PathBuf,
    pub created_at: u64,
    pub size_bytes: u64,
    pub node_count: u64,
    pub edge_count: u64,
    pub graph_count: u32,
    pub wal_sequence: u64,
}

/// Data captured in a snapshot for a single graph.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub graph_id: GraphId,
    pub graph_name: String,
    pub config_json: String,
    pub nodes: Vec<NodeSnapshot>,
    pub edges: Vec<EdgeSnapshot>,
}

/// Snapshot of a single node.
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeSnapshot {
    pub node_id: NodeId,
    pub label: String,
    pub properties_json: String,
    pub embedding: Option<Vec<f32>>,
    pub entity_key: Option<String>,
}

/// Snapshot of a single edge.
#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeSnapshot {
    pub edge_id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub label: String,
    pub weight: f32,
    pub valid_from: Timestamp,
    pub valid_until: Timestamp,
}

/// Full snapshot containing all graphs.
#[derive(Debug, Serialize, Deserialize)]
pub struct FullSnapshot {
    pub meta: SnapshotMeta,
    pub graphs: Vec<GraphSnapshot>,
}

// ─── SnapshotEngine ─────────────────────────────────────────────────────────

/// Engine for creating, loading, listing, and cleaning up snapshots.
pub struct SnapshotEngine {
    data_dir: PathBuf,
    format: SnapshotFormat,
}

impl SnapshotEngine {
    /// Create a new snapshot engine that stores snapshots in `data_dir`
    /// using the default format (Bincode).
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            format: SnapshotFormat::default(),
        }
    }

    /// Create a new snapshot engine with an explicit serialization format.
    pub fn with_format(data_dir: PathBuf, format: SnapshotFormat) -> Self {
        Self { data_dir, format }
    }

    /// Save a snapshot to disk. Returns the path of the written file.
    pub fn save_snapshot(&self, snapshot: &FullSnapshot) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.data_dir)?;

        let filename = format!("snapshot-{}.bin", snapshot.meta.created_at);
        let path = self.data_dir.join(&filename);

        let data = match &self.format {
            SnapshotFormat::Bincode => bincode::serialize(snapshot).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bincode serialize snapshot: {e}"),
                )
            })?,
        };

        let mut file = File::create(&path)?;
        file.write_all(&data)?;
        file.sync_all()?;

        Ok(path)
    }

    /// Load a snapshot from a file on disk.
    pub fn load_snapshot(&self, path: &Path) -> io::Result<FullSnapshot> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let snapshot: FullSnapshot = match &self.format {
            SnapshotFormat::Bincode => bincode::deserialize(&data).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bincode deserialize snapshot: {e}"),
                )
            })?,
        };

        Ok(snapshot)
    }

    /// List all snapshot metadata, sorted by `created_at` descending (newest first).
    pub fn list_snapshots(&self) -> io::Result<Vec<SnapshotMeta>> {
        if !self.data_dir.exists() {
            return Ok(Vec::new());
        }

        let mut snapshots = Vec::new();

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

            if !name.starts_with("snapshot-") || !name.ends_with(".bin") {
                continue;
            }

            // Try to load the snapshot to extract its metadata.
            match self.load_snapshot(&path) {
                Ok(snap) => {
                    let mut meta = snap.meta;
                    // Ensure the path in meta matches the actual on-disk path.
                    meta.path = path;
                    meta.size_bytes = entry.metadata()?.len();
                    snapshots.push(meta);
                }
                Err(_) => {
                    // Skip corrupted snapshot files.
                    continue;
                }
            }
        }

        // Sort newest first.
        snapshots.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(snapshots)
    }

    /// Return the path to the most recent snapshot, if any.
    pub fn latest_snapshot(&self) -> io::Result<Option<PathBuf>> {
        let snapshots = self.list_snapshots()?;
        Ok(snapshots.into_iter().next().map(|m| m.path))
    }

    /// Delete old snapshots, keeping only the `keep` most recent ones.
    /// Returns the number of snapshots deleted.
    pub fn cleanup_old_snapshots(&self, keep: usize) -> io::Result<u32> {
        let snapshots = self.list_snapshots()?;
        let mut deleted = 0u32;

        if snapshots.len() <= keep {
            return Ok(0);
        }

        for meta in snapshots.into_iter().skip(keep) {
            if fs::remove_file(&meta.path).is_ok() {
                deleted += 1;
            }
        }

        Ok(deleted)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Create a `SnapshotMeta` with the given parameters. Convenience for building
/// `FullSnapshot` values.
pub fn make_meta(
    node_count: u64,
    edge_count: u64,
    graph_count: u32,
    wal_sequence: u64,
) -> SnapshotMeta {
    SnapshotMeta {
        path: PathBuf::new(),
        created_at: now_millis(),
        size_bytes: 0,
        node_count,
        edge_count,
        graph_count,
        wal_sequence,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("weav_snap_test_{name}_{}", now_millis()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_snapshot(wal_seq: u64, created_at: u64) -> FullSnapshot {
        FullSnapshot {
            meta: SnapshotMeta {
                path: PathBuf::new(),
                created_at,
                size_bytes: 0,
                node_count: 2,
                edge_count: 1,
                graph_count: 1,
                wal_sequence: wal_seq,
            },
            graphs: vec![GraphSnapshot {
                graph_id: 1,
                graph_name: "test-graph".into(),
                config_json: "{}".into(),
                nodes: vec![
                    NodeSnapshot {
                        node_id: 1,
                        label: "Person".into(),
                        properties_json: r#"{"name":"Alice"}"#.into(),
                        embedding: Some(vec![1.0, 2.0, 3.0]),
                        entity_key: Some("alice".into()),
                    },
                    NodeSnapshot {
                        node_id: 2,
                        label: "Person".into(),
                        properties_json: r#"{"name":"Bob"}"#.into(),
                        embedding: None,
                        entity_key: Some("bob".into()),
                    },
                ],
                edges: vec![EdgeSnapshot {
                    edge_id: 1,
                    source: 1,
                    target: 2,
                    label: "KNOWS".into(),
                    weight: 1.0,
                    valid_from: 1000,
                    valid_until: u64::MAX,
                }],
            }],
        }
    }

    #[test]
    fn test_snapshot_save_and_load() {
        let dir = test_dir("save_load");
        let engine = SnapshotEngine::new(dir.clone());

        let snap = sample_snapshot(10, now_millis());
        let path = engine.save_snapshot(&snap).unwrap();
        assert!(path.exists());

        let loaded = engine.load_snapshot(&path).unwrap();
        assert_eq!(loaded.meta.wal_sequence, 10);
        assert_eq!(loaded.graphs.len(), 1);
        assert_eq!(loaded.graphs[0].nodes.len(), 2);
        assert_eq!(loaded.graphs[0].edges.len(), 1);
        assert_eq!(loaded.graphs[0].graph_name, "test-graph");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_list_and_latest() {
        let dir = test_dir("list_latest");
        let engine = SnapshotEngine::new(dir.clone());

        let ts1 = now_millis();
        let snap1 = sample_snapshot(5, ts1);
        engine.save_snapshot(&snap1).unwrap();

        // Small delay to ensure different timestamp.
        thread::sleep(Duration::from_millis(10));

        let ts2 = now_millis();
        let snap2 = sample_snapshot(15, ts2);
        engine.save_snapshot(&snap2).unwrap();

        let list = engine.list_snapshots().unwrap();
        assert_eq!(list.len(), 2);
        // Newest first.
        assert_eq!(list[0].wal_sequence, 15);
        assert_eq!(list[1].wal_sequence, 5);

        let latest = engine.latest_snapshot().unwrap();
        assert!(latest.is_some());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_cleanup() {
        let dir = test_dir("cleanup");
        let engine = SnapshotEngine::new(dir.clone());

        // Create 4 snapshots with distinct timestamps.
        for i in 0..4 {
            let ts = now_millis() + i;
            let snap = sample_snapshot(i, ts);
            engine.save_snapshot(&snap).unwrap();
            thread::sleep(Duration::from_millis(10));
        }

        let before = engine.list_snapshots().unwrap();
        assert_eq!(before.len(), 4);

        let deleted = engine.cleanup_old_snapshots(2).unwrap();
        assert_eq!(deleted, 2);

        let after = engine.list_snapshots().unwrap();
        assert_eq!(after.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_empty_dir() {
        let dir = test_dir("empty");
        let engine = SnapshotEngine::new(dir.join("nonexistent"));

        let list = engine.list_snapshots().unwrap();
        assert!(list.is_empty());

        let latest = engine.latest_snapshot().unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_snapshot_node_edge_data_roundtrip() {
        let dir = test_dir("roundtrip");
        let engine = SnapshotEngine::new(dir.clone());

        let snap = sample_snapshot(42, now_millis());
        let path = engine.save_snapshot(&snap).unwrap();
        let loaded = engine.load_snapshot(&path).unwrap();

        let node = &loaded.graphs[0].nodes[0];
        assert_eq!(node.node_id, 1);
        assert_eq!(node.label, "Person");
        assert_eq!(node.embedding, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(node.entity_key, Some("alice".into()));

        let edge = &loaded.graphs[0].edges[0];
        assert_eq!(edge.edge_id, 1);
        assert_eq!(edge.source, 1);
        assert_eq!(edge.target, 2);
        assert_eq!(edge.weight, 1.0);
        assert_eq!(edge.valid_until, u64::MAX);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_snapshot_with_format() {
        let dir = test_dir("with_format");
        let engine = SnapshotEngine::with_format(dir.clone(), SnapshotFormat::Bincode);

        let snap = sample_snapshot(7, now_millis());
        let path = engine.save_snapshot(&snap).unwrap();
        assert!(path.exists());

        let loaded = engine.load_snapshot(&path).unwrap();
        assert_eq!(loaded.meta.wal_sequence, 7);
        assert_eq!(loaded.graphs.len(), 1);
        assert_eq!(loaded.graphs[0].nodes.len(), 2);
        assert_eq!(loaded.graphs[0].edges.len(), 1);
        assert_eq!(loaded.graphs[0].graph_name, "test-graph");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cleanup_keep_more_than_exist() {
        let dir = test_dir("cleanup_keep_more");
        let engine = SnapshotEngine::new(dir.clone());

        // Save 2 snapshots with distinct timestamps.
        let ts1 = now_millis();
        engine.save_snapshot(&sample_snapshot(1, ts1)).unwrap();
        thread::sleep(Duration::from_millis(10));
        let ts2 = now_millis();
        engine.save_snapshot(&sample_snapshot(2, ts2)).unwrap();

        let before = engine.list_snapshots().unwrap();
        assert_eq!(before.len(), 2);

        // Ask to keep 5, but only 2 exist -- nothing should be deleted.
        let deleted = engine.cleanup_old_snapshots(5).unwrap();
        assert_eq!(deleted, 0);

        let after = engine.list_snapshots().unwrap();
        assert_eq!(after.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_empty_snapshot_roundtrip() {
        let dir = test_dir("empty_roundtrip");
        let engine = SnapshotEngine::new(dir.clone());

        let snap = FullSnapshot {
            meta: SnapshotMeta {
                path: PathBuf::new(),
                created_at: now_millis(),
                size_bytes: 0,
                node_count: 0,
                edge_count: 0,
                graph_count: 0,
                wal_sequence: 0,
            },
            graphs: Vec::new(),
        };

        let path = engine.save_snapshot(&snap).unwrap();
        assert!(path.exists());

        let loaded = engine.load_snapshot(&path).unwrap();
        assert_eq!(loaded.meta.node_count, 0);
        assert_eq!(loaded.meta.edge_count, 0);
        assert_eq!(loaded.meta.graph_count, 0);
        assert_eq!(loaded.meta.wal_sequence, 0);
        assert!(loaded.graphs.is_empty());

        fs::remove_dir_all(&dir).ok();
    }
}
