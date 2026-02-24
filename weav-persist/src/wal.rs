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

impl WalOperation {
    /// Return the graph_id hint for operations that carry one, or 0.
    pub fn graph_id_hint(&self) -> GraphId {
        match self {
            WalOperation::NodeAdd { graph_id, .. }
            | WalOperation::NodeUpdate { graph_id, .. }
            | WalOperation::NodeDelete { graph_id, .. }
            | WalOperation::EdgeAdd { graph_id, .. }
            | WalOperation::EdgeInvalidate { graph_id, .. }
            | WalOperation::EdgeDelete { graph_id, .. }
            | WalOperation::VectorUpdate { graph_id, .. } => *graph_id,
            WalOperation::GraphCreate { .. } | WalOperation::GraphDrop { .. } => 0,
        }
    }
}

// ─── Checksum ───────────────────────────────────────────────────────────────

/// Compute a CRC32 checksum over a byte slice.
///
/// Uses the `crc32fast` crate which provides hardware-accelerated CRC32C
/// on supported platforms, with a fast software fallback otherwise.
pub fn compute_checksum(data: &[u8]) -> u32 {
    crc32fast::hash(data)
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

    /// Truncate WAL entries with sequence numbers before `seq`.
    ///
    /// Reads all entries from the current WAL file, keeps only those with
    /// `entry.seq >= seq`, and rewrites the file. This is typically called
    /// after a successful snapshot to reclaim WAL space.
    pub fn truncate_before(&mut self, seq: u64) -> io::Result<()> {
        // Flush pending writes before reading the file.
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        // Read all entries that should be kept.
        let reader = WalReader::open(&self.path)?;
        let kept: Vec<WalEntry> = reader
            .filter_map(|r| r.ok())
            .filter(|entry| entry.seq >= seq)
            .collect();

        // Rewrite the WAL file with only the kept entries.
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        let mut writer = BufWriter::new(file);

        let mut new_size = 0u64;
        for entry in &kept {
            let entry_bytes = bincode::serialize(entry).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bincode serialize entry: {e}"),
                )
            })?;
            let len = entry_bytes.len() as u32;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&entry_bytes)?;
            new_size += 4 + entry_bytes.len() as u64;
        }

        writer.flush()?;
        writer.get_ref().sync_all()?;

        // Re-open the file in append mode for future writes.
        let file = OpenOptions::new()
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        self.current_size = new_size;

        Ok(())
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

    #[test]
    fn test_wal_sync_explicit() {
        let dir = test_dir("sync_explicit");
        let wal_path = dir.join("wal");

        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Never).unwrap();
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
                    label: "A".into(),
                    properties_json: "{}".into(),
                    embedding: None,
                    entity_key: None,
                },
            )
            .unwrap();

            // Explicitly sync -- should not panic.
            wal.sync().unwrap();
        }

        // Entries should be readable after explicit sync.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[1].seq, 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_truncate_before() {
        let dir = test_dir("truncate_before");
        let wal_path = dir.join("wal");

        // Write 5 entries (seq 1..=5).
        let mut wal =
            WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
        for _ in 0..5 {
            wal.append(
                0,
                WalOperation::GraphDrop {
                    name: "g".into(),
                },
            )
            .unwrap();
        }
        assert_eq!(wal.sequence_number(), 5);

        // Truncate entries before seq 3 (keep seq 3, 4, 5).
        wal.truncate_before(3).unwrap();

        // Read back and verify only entries with seq >= 3 remain.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].seq, 3);
        assert_eq!(entries[1].seq, 4);
        assert_eq!(entries[2].seq, 5);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_sync_mode_every_second() {
        let dir = test_dir("sync_every_second");
        let wal_path = dir.join("wal");

        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::EverySecond)
                    .unwrap();
            wal.append(
                0,
                WalOperation::GraphCreate {
                    name: "g1".into(),
                    config_json: "{}".into(),
                },
            )
            .unwrap();
            wal.append(
                1,
                WalOperation::NodeAdd {
                    graph_id: 1,
                    node_id: 10,
                    label: "Thing".into(),
                    properties_json: r#"{"k":"v"}"#.into(),
                    embedding: Some(vec![0.5]),
                    entity_key: None,
                },
            )
            .unwrap();
        }

        // Verify entries are readable.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[1].seq, 2);
        assert_eq!(entries[1].shard_id, 1);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_reader_corrupted_entry() {
        let dir = test_dir("corrupted_entry");
        let wal_path = dir.join("wal");

        // Write two valid entries.
        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
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
                WalOperation::GraphDrop {
                    name: "g1".into(),
                },
            )
            .unwrap();
        }

        // Append garbage bytes after the valid entries to simulate corruption.
        {
            use std::io::Write;
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            // Write a plausible length prefix (say 50 bytes) followed by garbage data.
            let fake_len: u32 = 50;
            file.write_all(&fake_len.to_le_bytes()).unwrap();
            file.write_all(&vec![0xDE; 50]).unwrap();
            file.flush().unwrap();
        }

        // Read with WalReader: the first two entries should be OK, then corruption.
        let reader = WalReader::open(&wal_path).unwrap();
        let results: Vec<io::Result<WalEntry>> = reader.collect();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        // Third entry should be an error (deserialization or checksum failure).
        assert!(results[2].is_err());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_checksum_mismatch_detection() {
        let dir = test_dir("checksum_mismatch");
        let wal_path = dir.join("wal");

        // Build a WalEntry with a deliberately wrong checksum and write it manually.
        // This guarantees a checksum mismatch that the reader must detect.
        {
            use std::io::Write;
            let op = WalOperation::NodeAdd {
                graph_id: 1,
                node_id: 42,
                label: "Person".into(),
                properties_json: r#"{"name":"Alice"}"#.into(),
                embedding: Some(vec![1.0, 2.0]),
                entity_key: None,
            };
            let op_bytes = bincode::serialize(&op).unwrap();
            let correct_checksum = compute_checksum(&op_bytes);

            // Create an entry with a corrupted checksum (off by one).
            let entry = WalEntry {
                seq: 1,
                timestamp: now_millis(),
                shard_id: 0,
                operation: op,
                checksum: correct_checksum.wrapping_add(1),
            };

            let entry_bytes = bincode::serialize(&entry).unwrap();
            let len = entry_bytes.len() as u32;

            let mut file = File::create(&wal_path).unwrap();
            file.write_all(&len.to_le_bytes()).unwrap();
            file.write_all(&entry_bytes).unwrap();
            file.flush().unwrap();
        }

        // Reading should yield exactly one entry which is a checksum mismatch error.
        let reader = WalReader::open(&wal_path).unwrap();
        let results: Vec<io::Result<WalEntry>> = reader.collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "Corrupted entry should be detected");
        let err_msg = results[0].as_ref().unwrap_err().to_string();
        assert!(
            err_msg.contains("checksum mismatch"),
            "Error should mention checksum mismatch, got: {err_msg}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_truncated_length_prefix() {
        let dir = test_dir("truncated_len");
        let wal_path = dir.join("wal");

        // Write 2 valid entries.
        {
            let mut wal =
                WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
            wal.append(0, WalOperation::GraphDrop { name: "a".into() })
                .unwrap();
            wal.append(0, WalOperation::GraphDrop { name: "b".into() })
                .unwrap();
        }

        // Append only 2 incomplete bytes (less than a full u32 length prefix).
        {
            use std::io::Write;
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[0x01, 0x02]).unwrap();
            file.flush().unwrap();
        }

        // Reader should return the 2 valid entries and then handle the truncated
        // prefix gracefully (UnexpectedEof causes the iterator to return None).
        let reader = WalReader::open(&wal_path).unwrap();
        let results: Vec<io::Result<WalEntry>> = reader.collect();
        // The 2-byte incomplete length prefix triggers UnexpectedEof on read_exact,
        // which the reader treats as end-of-file (returns None), so we get exactly 2 results.
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_empty_file() {
        let dir = test_dir("empty_file");
        let wal_path = dir.join("wal");

        // Create an empty file (0 bytes).
        fs::write(&wal_path, b"").unwrap();

        // WalReader::open should succeed.
        let reader = WalReader::open(&wal_path).unwrap();

        // Iterating should produce no entries.
        let entries: Vec<io::Result<WalEntry>> = reader.collect();
        assert!(entries.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_graph_id_hint_coverage() {
        // GraphCreate returns 0.
        let op = WalOperation::GraphCreate {
            name: "g".into(),
            config_json: "{}".into(),
        };
        assert_eq!(op.graph_id_hint(), 0);

        // GraphDrop returns 0.
        let op = WalOperation::GraphDrop {
            name: "g".into(),
        };
        assert_eq!(op.graph_id_hint(), 0);

        // NodeAdd returns its graph_id.
        let op = WalOperation::NodeAdd {
            graph_id: 5,
            node_id: 1,
            label: "L".into(),
            properties_json: "{}".into(),
            embedding: None,
            entity_key: None,
        };
        assert_eq!(op.graph_id_hint(), 5);

        // NodeUpdate returns its graph_id.
        let op = WalOperation::NodeUpdate {
            graph_id: 10,
            node_id: 1,
            properties_json: "{}".into(),
        };
        assert_eq!(op.graph_id_hint(), 10);

        // NodeDelete returns its graph_id.
        let op = WalOperation::NodeDelete {
            graph_id: 7,
            node_id: 1,
        };
        assert_eq!(op.graph_id_hint(), 7);

        // EdgeAdd returns its graph_id.
        let op = WalOperation::EdgeAdd {
            graph_id: 3,
            edge_id: 1,
            source: 1,
            target: 2,
            label: "KNOWS".into(),
            weight: 1.0,
        };
        assert_eq!(op.graph_id_hint(), 3);

        // EdgeInvalidate returns its graph_id.
        let op = WalOperation::EdgeInvalidate {
            graph_id: 8,
            edge_id: 1,
            timestamp: 1000,
        };
        assert_eq!(op.graph_id_hint(), 8);

        // EdgeDelete returns its graph_id.
        let op = WalOperation::EdgeDelete {
            graph_id: 12,
            edge_id: 1,
        };
        assert_eq!(op.graph_id_hint(), 12);

        // VectorUpdate returns its graph_id.
        let op = WalOperation::VectorUpdate {
            graph_id: 99,
            node_id: 1,
            vector: vec![0.1],
        };
        assert_eq!(op.graph_id_hint(), 99);
    }

    #[test]
    fn test_wal_truncate_before_all() {
        let dir = test_dir("truncate_all");
        let wal_path = dir.join("wal");

        // Write 5 entries (seq 1..=5).
        let mut wal =
            WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
        for _ in 0..5 {
            wal.append(0, WalOperation::GraphDrop { name: "g".into() })
                .unwrap();
        }
        assert_eq!(wal.sequence_number(), 5);

        // Truncate with seq > last entry: all entries should be removed.
        wal.truncate_before(100).unwrap();

        // Read back: no entries should remain.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert!(entries.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_wal_truncate_before_none() {
        let dir = test_dir("truncate_none");
        let wal_path = dir.join("wal");

        // Write 5 entries (seq 1..=5).
        let mut wal =
            WriteAheadLog::new(wal_path.clone(), 1024 * 1024, WalSyncMode::Always).unwrap();
        for _ in 0..5 {
            wal.append(0, WalOperation::GraphDrop { name: "g".into() })
                .unwrap();
        }
        assert_eq!(wal.sequence_number(), 5);

        // Truncate with seq=1: keeps all entries with seq >= 1 (i.e., all of them).
        wal.truncate_before(1).unwrap();

        // Read back: all 5 entries should remain.
        let reader = WalReader::open(&wal_path).unwrap();
        let entries: Vec<WalEntry> = reader.map(|r| r.unwrap()).collect();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].seq, 1);
        assert_eq!(entries[4].seq, 5);

        fs::remove_dir_all(&dir).ok();
    }
}
