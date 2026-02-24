//! Cross-shard message bus.
//!
//! Provides a bounded channel-based message bus for routing messages between
//! shards in the Weav engine. Uses xxHash for deterministic shard routing.

use std::collections::HashMap;

use crate::error::WeavError;
use crate::types::{EdgeId, GraphId, NodeData, NodeId, ScoredNode, ShardId};

// ─── Messages ──────────────────────────────────────────────────────────────

/// Result of a graph traversal across shards.
#[derive(Clone, Debug)]
pub struct TraversalResult {
    pub visited_nodes: Vec<NodeId>,
    pub visited_edges: Vec<EdgeId>,
    pub depth_map: HashMap<NodeId, u8>,
}

/// A message sent between shards via the [`MessageBus`].
pub enum ShardMessage {
    /// Request a BFS/DFS traversal starting from the given nodes.
    TraverseFrom {
        request_id: u64,
        origin_shard: ShardId,
        node_ids: Vec<NodeId>,
        depth: u8,
        respond_to: crossbeam_channel::Sender<TraversalResult>,
    },
    /// Request a vector similarity search within a graph.
    VectorSearch {
        request_id: u64,
        origin_shard: ShardId,
        graph_id: GraphId,
        query_vector: Vec<f32>,
        k: u16,
        respond_to: crossbeam_channel::Sender<Vec<ScoredNode>>,
    },
    /// Insert a node into a specific graph.
    InsertNode {
        request_id: u64,
        graph_id: GraphId,
        node: NodeData,
        respond_to: crossbeam_channel::Sender<Result<NodeId, WeavError>>,
    },
    /// Request a point-in-time snapshot.
    Snapshot {
        respond_to: crossbeam_channel::Sender<()>,
    },
    /// Gracefully shut down the shard worker.
    Shutdown,
}

// ─── MessageBus ────────────────────────────────────────────────────────────

/// A bounded, multi-producer channel bus that routes [`ShardMessage`]s to the
/// correct shard worker.
pub struct MessageBus {
    senders: Vec<crossbeam_channel::Sender<ShardMessage>>,
    num_shards: u16,
    buffer_size: usize,
}

impl MessageBus {
    /// Create a new message bus with `num_shards` channels, each with the given
    /// `buffer_size`. Returns the bus and a vec of receivers (one per shard).
    pub fn new(
        num_shards: u16,
        buffer_size: usize,
    ) -> (Self, Vec<crossbeam_channel::Receiver<ShardMessage>>) {
        let mut senders = Vec::with_capacity(num_shards as usize);
        let mut receivers = Vec::with_capacity(num_shards as usize);

        for _ in 0..num_shards {
            let (tx, rx) = crossbeam_channel::bounded(buffer_size);
            senders.push(tx);
            receivers.push(rx);
        }

        let bus = Self {
            senders,
            num_shards,
            buffer_size,
        };

        (bus, receivers)
    }

    /// Send a message to a specific shard.
    ///
    /// Returns [`WeavError::ShardUnavailable`] if the shard index is out of
    /// range or the channel is disconnected.
    pub fn send_to_shard(
        &self,
        shard: ShardId,
        msg: ShardMessage,
    ) -> Result<(), WeavError> {
        let sender = self
            .senders
            .get(shard as usize)
            .ok_or(WeavError::ShardUnavailable(shard))?;

        sender
            .send(msg)
            .map_err(|_| WeavError::ShardUnavailable(shard))
    }

    /// Deterministically route a `(graph_id, node_id)` pair to a shard using
    /// xxHash.
    pub fn route_by_key(&self, graph_id: GraphId, node_id: NodeId) -> ShardId {
        let mut buf = [0u8; 12];
        buf[..4].copy_from_slice(&graph_id.to_le_bytes());
        buf[4..12].copy_from_slice(&node_id.to_le_bytes());
        let hash = xxhash_rust::xxh3::xxh3_64(&buf);
        (hash % self.num_shards as u64) as ShardId
    }

    /// Returns the number of shards this bus was created with.
    pub fn num_shards(&self) -> u16 {
        self.num_shards
    }

    /// Returns the configured buffer size per channel.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Broadcast a shutdown message to all shard workers.
    ///
    /// Since [`ShardMessage`] variants like `TraverseFrom` and `VectorSearch`
    /// contain oneshot senders and cannot be cloned, broadcast is limited to
    /// control messages. This method sends [`ShardMessage::Shutdown`] to every
    /// shard. Send failures (e.g. disconnected receivers) are silently ignored.
    pub fn broadcast_shutdown(&self) {
        for sender in &self.senders {
            let _ = sender.send(ShardMessage::Shutdown);
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_correct_number_of_channels() {
        let (bus, receivers) = MessageBus::new(4, 64);
        assert_eq!(bus.num_shards(), 4);
        assert_eq!(bus.buffer_size(), 64);
        assert_eq!(bus.senders.len(), 4);
        assert_eq!(receivers.len(), 4);
    }

    #[test]
    fn test_send_to_shard_and_receive() {
        let (bus, receivers) = MessageBus::new(2, 16);

        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);
        bus.send_to_shard(
            0,
            ShardMessage::Snapshot {
                respond_to: resp_tx,
            },
        )
        .unwrap();

        let msg = receivers[0].try_recv().unwrap();
        match msg {
            ShardMessage::Snapshot { respond_to } => {
                respond_to.send(()).unwrap();
                resp_rx.recv().unwrap();
            }
            _ => panic!("expected Snapshot message"),
        }
    }

    #[test]
    fn test_send_to_invalid_shard() {
        let (bus, _receivers) = MessageBus::new(2, 16);
        let result = bus.send_to_shard(99, ShardMessage::Shutdown);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::ShardUnavailable(id) => assert_eq!(id, 99),
            other => panic!("expected ShardUnavailable, got: {other}"),
        }
    }

    #[test]
    fn test_send_to_disconnected_channel() {
        let (bus, receivers) = MessageBus::new(2, 16);
        // Drop all receivers so the channel is disconnected.
        drop(receivers);
        let result = bus.send_to_shard(0, ShardMessage::Shutdown);
        assert!(result.is_err());
    }

    #[test]
    fn test_route_by_key_deterministic() {
        let (bus, _receivers) = MessageBus::new(8, 16);
        let shard_a = bus.route_by_key(1, 100);
        let shard_b = bus.route_by_key(1, 100);
        assert_eq!(shard_a, shard_b);
    }

    #[test]
    fn test_route_by_key_within_range() {
        let (bus, _receivers) = MessageBus::new(4, 16);
        for graph_id in 0..100 {
            for node_id in 0..100 {
                let shard = bus.route_by_key(graph_id, node_id);
                assert!(shard < 4, "shard {shard} out of range for 4 shards");
            }
        }
    }

    #[test]
    fn test_route_by_key_distributes_across_shards() {
        let (bus, _receivers) = MessageBus::new(4, 16);
        let mut counts = [0u32; 4];
        for node_id in 0..1000 {
            let shard = bus.route_by_key(0, node_id);
            counts[shard as usize] += 1;
        }
        // Each shard should get at least some nodes (xxHash distributes well).
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 100,
                "shard {i} only got {count} of 1000 nodes — poor distribution"
            );
        }
    }

    #[test]
    fn test_traverse_from_roundtrip() {
        let (bus, receivers) = MessageBus::new(1, 16);
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);

        bus.send_to_shard(
            0,
            ShardMessage::TraverseFrom {
                request_id: 42,
                origin_shard: 0,
                node_ids: vec![1, 2, 3],
                depth: 2,
                respond_to: resp_tx,
            },
        )
        .unwrap();

        let msg = receivers[0].try_recv().unwrap();
        match msg {
            ShardMessage::TraverseFrom {
                request_id,
                node_ids,
                depth,
                respond_to,
                ..
            } => {
                assert_eq!(request_id, 42);
                assert_eq!(node_ids, vec![1, 2, 3]);
                assert_eq!(depth, 2);

                let result = TraversalResult {
                    visited_nodes: vec![1, 2, 3, 4],
                    visited_edges: vec![10, 11],
                    depth_map: HashMap::from([(1, 0), (2, 1), (3, 1), (4, 2)]),
                };
                respond_to.send(result).unwrap();
            }
            _ => panic!("expected TraverseFrom"),
        }

        let result = resp_rx.recv().unwrap();
        assert_eq!(result.visited_nodes.len(), 4);
        assert_eq!(result.depth_map[&4], 2);
    }

    #[test]
    fn test_insert_node_roundtrip() {
        let (bus, receivers) = MessageBus::new(1, 16);
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);

        let node = NodeData {
            label: "Person".into(),
            properties: vec![],
            embedding: None,
            entity_key: Some("alice".into()),
            provenance: None,
        };

        bus.send_to_shard(
            0,
            ShardMessage::InsertNode {
                request_id: 1,
                graph_id: 0,
                node,
                respond_to: resp_tx,
            },
        )
        .unwrap();

        let msg = receivers[0].try_recv().unwrap();
        match msg {
            ShardMessage::InsertNode {
                request_id,
                respond_to,
                ..
            } => {
                assert_eq!(request_id, 1);
                respond_to.send(Ok(42)).unwrap();
            }
            _ => panic!("expected InsertNode"),
        }

        let result = resp_rx.recv().unwrap();
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_vector_search_roundtrip() {
        let (bus, receivers) = MessageBus::new(1, 16);
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(1);

        bus.send_to_shard(
            0,
            ShardMessage::VectorSearch {
                request_id: 7,
                origin_shard: 0,
                graph_id: 1,
                query_vector: vec![0.1, 0.2, 0.3],
                k: 5,
                respond_to: resp_tx,
            },
        )
        .unwrap();

        let msg = receivers[0].try_recv().unwrap();
        match msg {
            ShardMessage::VectorSearch {
                request_id,
                k,
                respond_to,
                ..
            } => {
                assert_eq!(request_id, 7);
                assert_eq!(k, 5);
                respond_to
                    .send(vec![ScoredNode {
                        node_id: 10,
                        score: 0.95,
                        depth: 0,
                    }])
                    .unwrap();
            }
            _ => panic!("expected VectorSearch"),
        }

        let results = resp_rx.recv().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 10);
    }

    #[test]
    fn test_shutdown_message() {
        let (bus, receivers) = MessageBus::new(2, 16);
        bus.send_to_shard(0, ShardMessage::Shutdown).unwrap();
        bus.send_to_shard(1, ShardMessage::Shutdown).unwrap();

        for rx in &receivers {
            let msg = rx.try_recv().unwrap();
            assert!(matches!(msg, ShardMessage::Shutdown));
        }
    }

    #[test]
    fn test_broadcast_shutdown() {
        let (bus, receivers) = MessageBus::new(4, 16);
        bus.broadcast_shutdown();

        // Every shard should have received exactly one Shutdown message.
        for (i, rx) in receivers.iter().enumerate() {
            let msg = rx.try_recv().unwrap_or_else(|_| {
                panic!("shard {i} did not receive a Shutdown message")
            });
            assert!(
                matches!(msg, ShardMessage::Shutdown),
                "shard {i} received unexpected message variant"
            );
            // No extra messages should be pending.
            assert!(
                rx.try_recv().is_err(),
                "shard {i} has unexpected extra messages"
            );
        }
    }

    #[test]
    fn test_broadcast_shutdown_with_disconnected_receivers() {
        let (bus, receivers) = MessageBus::new(3, 16);
        // Drop one receiver to simulate a disconnected shard.
        drop(receivers);
        // Should not panic even if all receivers are disconnected.
        bus.broadcast_shutdown();
    }

    #[test]
    fn test_route_by_key_zero_values() {
        let (bus, _receivers) = MessageBus::new(4, 16);
        let shard = bus.route_by_key(0, 0);
        assert!(shard < 4, "shard {shard} out of range for 4 shards");
    }

    #[test]
    fn test_route_by_key_max_values() {
        let (bus, _receivers) = MessageBus::new(4, 16);
        let shard = bus.route_by_key(u32::MAX, u64::MAX);
        assert!(shard < 4, "shard {shard} out of range for 4 shards");
    }

    #[test]
    fn test_broadcast_shutdown_single_shard() {
        let (bus, receivers) = MessageBus::new(1, 16);
        bus.broadcast_shutdown();

        let msg = receivers[0]
            .try_recv()
            .expect("single shard should receive Shutdown");
        assert!(matches!(msg, ShardMessage::Shutdown));
        assert!(
            receivers[0].try_recv().is_err(),
            "should have exactly one message"
        );
    }

    // ── Round 2 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_message_bus_single_shard() {
        let (bus, _receivers) = MessageBus::new(1, 16);
        // All keys route to shard 0
        for node_id in 0..100 {
            assert_eq!(bus.route_by_key(0, node_id), 0);
        }
    }

    #[test]
    fn test_message_bus_route_different_graph_same_node() {
        let (bus, _receivers) = MessageBus::new(8, 16);
        let shard_g1 = bus.route_by_key(1, 100);
        let shard_g2 = bus.route_by_key(2, 100);
        // Different graph_ids with same node_id may route to different shards
        // (or the same - we just verify both are valid)
        assert!(shard_g1 < 8);
        assert!(shard_g2 < 8);
        // They should be deterministic
        assert_eq!(bus.route_by_key(1, 100), shard_g1);
        assert_eq!(bus.route_by_key(2, 100), shard_g2);
    }

    #[test]
    fn test_message_bus_multiple_messages_fifo() {
        let (bus, receivers) = MessageBus::new(1, 128);
        // Send 100 snapshot messages
        for _ in 0..100 {
            let (resp_tx, _resp_rx) = crossbeam_channel::bounded(1);
            bus.send_to_shard(0, ShardMessage::Snapshot { respond_to: resp_tx }).unwrap();
        }
        // Receive all 100
        let mut count = 0;
        while receivers[0].try_recv().is_ok() {
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_message_bus_broadcast_shutdown_no_receivers() {
        let (bus, receivers) = MessageBus::new(4, 16);
        drop(receivers);
        // Should not panic
        bus.broadcast_shutdown();
    }

    #[test]
    fn test_message_bus_num_shards_and_buffer() {
        let (bus, _receivers) = MessageBus::new(16, 256);
        assert_eq!(bus.num_shards(), 16);
        assert_eq!(bus.buffer_size(), 256);
    }

    #[test]
    fn test_message_bus_send_after_partial_receiver_drop() {
        let (bus, mut receivers) = MessageBus::new(3, 16);
        // Drop receiver for shard 1 only
        let _r0 = receivers.remove(0); // keep shard 0
        // receivers now has shard 1 and shard 2 (originally indices 1, 2)
        drop(receivers.remove(0)); // drop shard 1's receiver

        // Send to shard 0 should work
        bus.send_to_shard(0, ShardMessage::Shutdown).unwrap();
        // Send to shard 1 should fail (disconnected)
        assert!(bus.send_to_shard(1, ShardMessage::Shutdown).is_err());
        // Send to shard 2 should work
        bus.send_to_shard(2, ShardMessage::Shutdown).unwrap();
    }
}
