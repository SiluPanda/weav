//! Write-Ahead Log (WAL) for durable operation recording.
//!
//! Each WAL entry is stored as a length-prefixed bincode-serialized blob:
//! `[u32 length][bincode WalEntry bytes]`

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use weav_core::config::WalSyncMode;
use weav_core::types::*;

// ─── Types ──────────────────────────────────────────────────────────────────

/// A single entry in the write-ahead log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub seq: u64,
    pub timestamp: u64,
    pub shard_id: ShardId,
    pub operation: WalOperation,
    pub checksum: u32,
}

/// Operations that can be recorded in the WAL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    GraphCreate {
        name: String,
        config_json: String,
    },
    GraphDrop {
        name: String,
    },
    NodeAdd {
        graph_id: GraphId,
        node_id: NodeId,
        label: String,
        properties_json: String,
        embedding: Option<Vec<f32>>,
        entity_key: Option<String>,
    },
    NodeUpdate {
        graph_id: GraphId,
        node_id: NodeId,
        properties_json: String,
    },
    NodeDelete {
        graph_id: GraphId,
        node_id: NodeId,
    },
    EdgeAdd {
        graph_id: GraphId,
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        label: String,
        weight: f32,
    },
    EdgeInvalidate {
        graph_id: GraphId,
        edge_id: EdgeId,
        timestamp: Timestamp,
    },
    EdgeDelete {
        graph_id: GraphId,
        edge_id: EdgeId,
    },
    VectorUpdate {
        graph_id: GraphId,
        node_id: NodeId,
        vector: Vec<f32>,
    },
}

// ─── Checksum ───────────────────────────────────────────────────────────────

/// Compute a simple checksum over a byte slice.
///
/// This uses a basic wrapping-add approach. For production you would use
/// `crc32fast`, but this keeps the dependency list minimal.
pub fn compute_checksum(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        // Rotate left by 5 bits and wrapping-add the byte, mixed with position.
        hash = hash.rotate_left(5).wrapping_add(byte as u32).wrapping_add(i as u32);
    }
    hash
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── WriteAheadLog ──────────────────────────────────────────────────────────

/// Write-ahead log for durable operation recording.
pub struct WriteAheadLog {
    writer: BufWriter<File>,
    path: PathBuf,
    current_size: u64,
    max_size: u64,
    sync_mode: WalSyncMode,
    sequence_number: u64,
}

impl WriteAheadLog {
    /// Open or create a WAL file at `path` for appending.
    pub fn new(path: PathBuf, max_size: u64, sync_mode: WalSyncMode) -> io::Result<Self> {
        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let current_size = file.metadata()?.len();

        // Determine the current sequence number by scanning existing entries.
        let sequence_number = Self::scan_last_seq(&path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            current_size,
            max_size,
            sync_mode,
            sequence_number,
        })
    }

    /// Scan the WAL file to find the last sequence number (0 if empty).
    fn scan_last_seq(path: &Path) -> io::Result<u64> {
        if !path.exists() {
            return Ok(0);
        }
        let meta = fs::metadata(path)?;
        if meta.len() == 0 {
            return Ok(0);
        }
        let reader = WalReader::open(path)?;
        let mut last_seq = 0u64;
        for entry_result in reader {
            match entry_result {
                Ok(entry) => last_seq = entry.seq,
                Err(_) => break, // Stop at first corrupted entry.
            }
        }
        Ok(last_seq)
    }

    /// Append an operation to the WAL. Returns the assigned sequence number.
    pub fn append(&mut self, shard_id: ShardId, operation: WalOperation) -> io::Result<u64> {
        self.sequence_number += 1;
        let seq = self.sequence_number;

        // Compute checksum over the serialized operation bytes.
        let op_bytes = bincode::serialize(&operation).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bincode serialize op: {e}"))
        })?;
        let checksum = compute_checksum(&op_bytes);

        let entry = WalEntry {
            seq,
            timestamp: now_millis(),
            shard_id,
            operation,
            checksum,
        };

        let entry_bytes = bincode::serialize(&entry).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bincode serialize entry: {e}"),
            )
        })?;

        let len = entry_bytes.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&entry_bytes)?;
        self.writer.flush()?;

        self.current_size += 4 + entry_bytes.len() as u64;

        // Sync according to the configured mode.
        match &self.sync_mode {
            WalSyncMode::Always => {
                self.writer.get_ref().sync_all()?;
            }
            WalSyncMode::EverySecond | WalSyncMode::Never => {
                // Caller is responsible for periodic sync or no sync at all.
            }
        }

        Ok(seq)
    }

    /// Force an fsync of the WAL file to disk.
    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
    }

    /// Returns `true` if the current WAL file has reached or exceeded `max_size`.
    pub fn should_rotate(&self) -> bool {
        self.current_size >= self.max_size
    }

    /// Rotate the WAL: rename the current file to `{path}.{timestamp}` and
    /// open a fresh file. Returns the path of the rotated (old) file.
    pub fn rotate(&mut self) -> io::Result<PathBuf> {
        // Flush any pending writes.
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        // Build the rotated filename.
        let ts = now_millis();
        let rotated = PathBuf::from(format!("{}.{}", self.path.display(), ts));

        // Rename the current file.
        fs::rename(&self.path, &rotated)?;

        // Open a new empty file at the original path.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        self.writer = BufWriter::new(file);
        self.current_size = 0;

        Ok(rotated)
    }

    /// The current sequence number (i.e. the seq of the last appended entry).
    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }
}

// ─── WalReader ──────────────────────────────────────────────────────────────

/// Iterator over WAL entries for replay.
pub struct WalReader {
    reader: BufReader<File>,
}

impl WalReader {
    /// Open a WAL file for reading.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
        })
    }
}

impl Iterator for WalReader {
    type Item = io::Result<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        // Read the 4-byte length prefix.
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e)),
        }

        let len = u32::from_le_bytes(len_buf) as usize;

        // Read the entry bytes.
        let mut entry_buf = vec![0u8; len];
        if let Err(e) = self.reader.read_exact(&mut entry_buf) {
            return Some(Err(e));
        }

        // Deserialize.
        let entry: WalEntry = match bincode::deserialize(&entry_buf) {
            Ok(e) => e,
            Err(e) => {
                return Some(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bincode deserialize: {e}"),
                )));
            }
        };

        // Validate checksum.
        let op_bytes = match bincode::serialize(&entry.operation) {
            Ok(b) => b,
            Err(e) => {
                return Some(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bincode re-serialize for checksum: {e}"),
                )));
            }
        };
        let expected = compute_checksum(&op_bytes);
        if entry.checksum != expected {
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "checksum mismatch for seq {}: expected {expected}, got {}",
                    entry.seq, entry.checksum
                ),
            )));
        }

        Some(Ok(entry))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("weav_wal_test_{name}_{}", now_millis()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_wal_write_and_read_back() {
        let dir = test_dir("write_read");
        let wal_path = dir.join("wal");

        // Write entries.
        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
            assert_eq!(wal.sequence_number(), 0);

            let seq1 = wal
                .append(
                    0,
                    WalOperation::GraphCreate {
                        name: "test-graph".into(),
                        config_json: "{}".into(),
                    },
                )
                .unwrap();
            assert_eq!(seq1, 1);

            let seq2 = wal
                .append(
                    0,
                    WalOperation::NodeAdd {
                        graph_id: 1,
                        node_id: 100,
                        label: "Person".into(),
                        properties_json: r#"{"name":"Alice"}"#.into(),
                        embedding: Some(vec![1.0, 2.0, 3.0]),
                        entity_key: Some("alice".into()),
                    },
                )
                .unwrap();
            assert_eq!(seq2, 2);
            assert_eq!(wal.sequence_number(), 2);
        }

        // Read entries back.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[1].seq, 2);

        match &entries[0].operation {
            WalOperation::GraphCreate { name, .. } => assert_eq!(name, "test-graph"),
            other => panic!("unexpected operation: {other:?}"),
        }
        match &entries[1].operation {
            WalOperation::NodeAdd {
                node_id, label, ..
            } => {
                assert_eq!(*node_id, 100);
                assert_eq!(label, "Person");
            }
            other => panic!("unexpected operation: {other:?}"),
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_checksum_validation() {
        let dir = test_dir("checksum");
        let wal_path = dir.join("wal");

        // Write one entry.
        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
            wal.append(
                0,
                WalOperation::GraphDrop {
                    name: "g".into(),
                },
            )
            .unwrap();
        }

        // Verify checksum passes during read.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<_> = reader.collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_ok());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_rotation() {
        let dir = test_dir("rotate");
        let wal_path = dir.join("wal");

        let mut wal = WriteAheadLog::new(wal_path.clone(), 50, WalSyncMode::Always).unwrap();

        // Append enough to exceed the tiny max_size.
        wal.append(
            0,
            WalOperation::GraphCreate {
                name: "big-graph-name-to-fill-up".into(),
                config_json: "{}".into(),
            },
        )
        .unwrap();

        assert!(wal.should_rotate());

        let rotated_path = wal.rotate().unwrap();
        assert!(rotated_path.exists());
        assert!(!wal.should_rotate()); // New file is empty.

        // Can still write to the new file.
        let seq = wal
            .append(
                0,
                WalOperation::GraphDrop {
                    name: "x".into(),
                },
            )
            .unwrap();
        assert_eq!(seq, 2); // Sequence continues.

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_resume_sequence_number() {
        let dir = test_dir("resume_seq");
        let wal_path = dir.join("wal");

        // Write 3 entries.
        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
            wal.append(0, WalOperation::GraphDrop { name: "a".into() }).unwrap();
            wal.append(0, WalOperation::GraphDrop { name: "b".into() }).unwrap();
            wal.append(0, WalOperation::GraphDrop { name: "c".into() }).unwrap();
            assert_eq!(wal.sequence_number(), 3);
        }

        // Re-open; should resume at seq 3.
        {
            let wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
            assert_eq!(wal.sequence_number(), 3);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_all_operation_types() {
        let dir = test_dir("all_ops");
        let wal_path = dir.join("wal");

        let ops = vec![
            WalOperation::GraphCreate {
                name: "g".into(),
                config_json: "{}".into(),
            },
            WalOperation::GraphDrop {
                name: "g".into(),
            },
            WalOperation::NodeAdd {
                graph_id: 1,
                node_id: 1,
                label: "L".into(),
                properties_json: "{}".into(),
                embedding: None,
                entity_key: None,
            },
            WalOperation::NodeUpdate {
                graph_id: 1,
                node_id: 1,
                properties_json: r#"{"x":1}"#.into(),
            },
            WalOperation::NodeDelete {
                graph_id: 1,
                node_id: 1,
            },
            WalOperation::EdgeAdd {
                graph_id: 1,
                edge_id: 1,
                source: 1,
                target: 2,
                label: "KNOWS".into(),
                weight: 0.9,
            },
            WalOperation::EdgeInvalidate {
                graph_id: 1,
                edge_id: 1,
                timestamp: 999,
            },
            WalOperation::EdgeDelete {
                graph_id: 1,
                edge_id: 1,
            },
            WalOperation::VectorUpdate {
                graph_id: 1,
                node_id: 1,
                vector: vec![0.1, 0.2],
            },
        ];

        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Never).unwrap();
            for op in &ops {
                wal.append(0, op.clone()).unwrap();
            }
        }

        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), ops.len());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_compute_checksum_deterministic() {
        let data = b"hello world";
        let c1 = compute_checksum(data);
        let c2 = compute_checksum(data);
        assert_eq!(c1, c2);

        // Different data should give a different checksum (with very high probability).
        let c3 = compute_checksum(b"hello worlD");
        assert_ne!(c1, c3);
    }
}
