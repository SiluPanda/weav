use prometheus::{
    IntCounter, IntGauge, IntGaugeVec, HistogramVec, IntCounterVec,
    Histogram, Opts, HistogramOpts, Registry,
};
use std::sync::LazyLock;

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

// ─── Existing metrics ──────────────────────────────────────────────────────

pub static NODES_TOTAL: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_nodes_total", "Total number of nodes");
    let gauge = IntGaugeVec::new(opts, &["graph"]).unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

pub static EDGES_TOTAL: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_edges_total", "Total number of edges");
    let gauge = IntGaugeVec::new(opts, &["graph"]).unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

pub static QUERY_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new("weav_query_duration_seconds", "Query duration in seconds");
    let hist = HistogramVec::new(opts, &["type"]).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

pub static QUERY_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_query_total", "Total queries");
    let counter = IntCounterVec::new(opts, &["type", "status"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static CONNECTIONS_ACTIVE: LazyLock<IntGauge> = LazyLock::new(|| {
    let gauge = IntGauge::new("weav_connections_active", "Active connections").unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

// ─── WAL metrics ───────────────────────────────────────────────────────────

pub static WAL_WRITES_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_wal_writes_total", "Total WAL write operations");
    let counter = IntCounterVec::new(opts, &["graph", "operation"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static WAL_BYTES_WRITTEN: LazyLock<IntCounter> = LazyLock::new(|| {
    let counter = IntCounter::new("weav_wal_bytes_written_total", "Total bytes written to WAL").unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

pub static WAL_SYNC_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    let opts = HistogramOpts::new("weav_wal_sync_duration_seconds", "WAL fsync duration in seconds");
    let hist = Histogram::with_opts(opts).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

// ─── Snapshot metrics ──────────────────────────────────────────────────────

pub static SNAPSHOT_DURATION: LazyLock<Histogram> = LazyLock::new(|| {
    let opts = HistogramOpts::new("weav_snapshot_duration_seconds", "Snapshot save duration in seconds");
    let hist = Histogram::with_opts(opts).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

pub static SNAPSHOT_SIZE_BYTES: LazyLock<IntGauge> = LazyLock::new(|| {
    let gauge = IntGauge::new("weav_snapshot_size_bytes", "Size of the last snapshot in bytes").unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

// ─── Vector search metrics ─────────────────────────────────────────────────

pub static VECTOR_SEARCH_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new("weav_vector_search_duration_seconds", "Vector search duration in seconds");
    let hist = HistogramVec::new(opts, &["graph"]).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

pub static VECTOR_INDEX_SIZE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_vector_index_size", "Number of vectors in the index");
    let gauge = IntGaugeVec::new(opts, &["graph"]).unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

// ─── Token budget metrics ──────────────────────────────────────────────────

pub static TOKEN_BUDGET_USAGE: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new(
        "weav_token_budget_usage_ratio",
        "Token budget utilization ratio (0.0 to 1.0)",
    );
    let hist = HistogramVec::new(opts, &["graph", "strategy"]).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

pub static TOKEN_BUDGET_OVERFLOW: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_token_budget_overflow_total", "Token budget overflow events");
    let counter = IntCounterVec::new(opts, &["graph"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

// ─── TTL metrics ───────────────────────────────────────────────────────────

pub static TTL_EXPIRED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_ttl_expired_total", "Total entities expired by TTL sweep");
    let counter = IntCounterVec::new(opts, &["graph", "entity_type"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

// ─── Graph lifecycle metrics ───────────────────────────────────────────────

pub static GRAPHS_TOTAL: LazyLock<IntGauge> = LazyLock::new(|| {
    let gauge = IntGauge::new("weav_graphs_total", "Total number of graphs").unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

// ─── Auth metrics ──────────────────────────────────────────────────────────

pub static AUTH_ATTEMPTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_auth_attempts_total", "Total authentication attempts");
    let counter = IntCounterVec::new(opts, &["result"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

// ─── Ingest metrics ────────────────────────────────────────────────────────

pub static INGEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new("weav_ingest_duration_seconds", "Document ingest duration in seconds");
    let hist = HistogramVec::new(opts, &["graph", "format"]).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

pub static INGEST_DOCUMENTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new("weav_ingest_documents_total", "Total documents ingested");
    let counter = IntCounterVec::new(opts, &["graph"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_metrics_registered() {
        // Force initialization of every metric — each LazyLock registers on first access.
        // Dereference triggers the register() call inside the LazyLock closure.
        // If registration fails (e.g., duplicate name), it would panic here.
        let _ = &*NODES_TOTAL;
        let _ = &*EDGES_TOTAL;
        let _ = &*QUERY_DURATION;
        let _ = &*QUERY_TOTAL;
        let _ = &*CONNECTIONS_ACTIVE;
        let _ = &*WAL_WRITES_TOTAL;
        let _ = &*WAL_BYTES_WRITTEN;
        let _ = &*WAL_SYNC_DURATION;
        let _ = &*SNAPSHOT_DURATION;
        let _ = &*SNAPSHOT_SIZE_BYTES;
        let _ = &*VECTOR_SEARCH_DURATION;
        let _ = &*VECTOR_INDEX_SIZE;
        let _ = &*TOKEN_BUDGET_USAGE;
        let _ = &*TOKEN_BUDGET_OVERFLOW;
        let _ = &*TTL_EXPIRED_TOTAL;
        let _ = &*GRAPHS_TOTAL;
        let _ = &*AUTH_ATTEMPTS_TOTAL;
        let _ = &*INGEST_DURATION;
        let _ = &*INGEST_DOCUMENTS_TOTAL;

        // Touch each labeled metric so gather() returns them.
        // Vec-typed metrics only appear in gather() after with_label_values() is called.
        NODES_TOTAL.with_label_values(&["_init"]).set(0);
        EDGES_TOTAL.with_label_values(&["_init"]).set(0);
        QUERY_DURATION.with_label_values(&["_init"]).observe(0.0);
        QUERY_TOTAL.with_label_values(&["_init", "_init"]).inc();
        WAL_WRITES_TOTAL.with_label_values(&["_init", "_init"]).inc();
        VECTOR_SEARCH_DURATION.with_label_values(&["_init"]).observe(0.0);
        VECTOR_INDEX_SIZE.with_label_values(&["_init"]).set(0);
        TOKEN_BUDGET_USAGE.with_label_values(&["_init", "_init"]).observe(0.0);
        TOKEN_BUDGET_OVERFLOW.with_label_values(&["_init"]).inc();
        TTL_EXPIRED_TOTAL.with_label_values(&["_init", "_init"]).inc();
        AUTH_ATTEMPTS_TOTAL.with_label_values(&["_init"]).inc();
        INGEST_DURATION.with_label_values(&["_init", "_init"]).observe(0.0);
        INGEST_DOCUMENTS_TOTAL.with_label_values(&["_init"]).inc();

        let families = REGISTRY.gather();
        // 19 metrics: 5 existing + 14 new. All should appear as distinct families.
        assert!(
            families.len() >= 19,
            "Expected at least 19 metric families, got {}",
            families.len()
        );
    }

    #[test]
    fn test_counter_increment() {
        WAL_WRITES_TOTAL
            .with_label_values(&["test_graph", "NodeAdd"])
            .inc();
        let val = WAL_WRITES_TOTAL
            .with_label_values(&["test_graph", "NodeAdd"])
            .get();
        assert!(val >= 1, "WAL_WRITES_TOTAL should be >= 1, got {val}");
    }

    #[test]
    fn test_histogram_observation() {
        WAL_SYNC_DURATION.observe(0.005);
        let count = WAL_SYNC_DURATION.get_sample_count();
        assert!(count >= 1, "WAL_SYNC_DURATION should have >= 1 observation, got {count}");
    }

    #[test]
    fn test_gauge_set_get() {
        GRAPHS_TOTAL.set(42);
        assert_eq!(GRAPHS_TOTAL.get(), 42);
        GRAPHS_TOTAL.set(0);
    }

    #[test]
    fn test_int_counter_increment() {
        let before = WAL_BYTES_WRITTEN.get();
        WAL_BYTES_WRITTEN.inc_by(1024);
        assert_eq!(WAL_BYTES_WRITTEN.get(), before + 1024);
    }

    #[test]
    fn test_labeled_histogram_observation() {
        VECTOR_SEARCH_DURATION
            .with_label_values(&["bench_graph"])
            .observe(0.042);
        let count = VECTOR_SEARCH_DURATION
            .with_label_values(&["bench_graph"])
            .get_sample_count();
        assert!(count >= 1, "VECTOR_SEARCH_DURATION should have >= 1 observation");
    }

    #[test]
    fn test_labeled_gauge_set() {
        VECTOR_INDEX_SIZE
            .with_label_values(&["idx_graph"])
            .set(500);
        let val = VECTOR_INDEX_SIZE
            .with_label_values(&["idx_graph"])
            .get();
        assert_eq!(val, 500);
    }

    #[test]
    fn test_ttl_expired_counter() {
        TTL_EXPIRED_TOTAL
            .with_label_values(&["g", "node"])
            .inc_by(5);
        let val = TTL_EXPIRED_TOTAL
            .with_label_values(&["g", "node"])
            .get();
        assert!(val >= 5, "TTL_EXPIRED_TOTAL should be >= 5, got {val}");
    }

    #[test]
    fn test_auth_attempts_counter() {
        AUTH_ATTEMPTS_TOTAL
            .with_label_values(&["success"])
            .inc();
        AUTH_ATTEMPTS_TOTAL
            .with_label_values(&["failure"])
            .inc();
        let success = AUTH_ATTEMPTS_TOTAL
            .with_label_values(&["success"])
            .get();
        let failure = AUTH_ATTEMPTS_TOTAL
            .with_label_values(&["failure"])
            .get();
        assert!(success >= 1);
        assert!(failure >= 1);
    }

    #[test]
    fn test_token_budget_metrics() {
        TOKEN_BUDGET_USAGE
            .with_label_values(&["g", "greedy"])
            .observe(0.85);
        let count = TOKEN_BUDGET_USAGE
            .with_label_values(&["g", "greedy"])
            .get_sample_count();
        assert!(count >= 1);

        TOKEN_BUDGET_OVERFLOW
            .with_label_values(&["g"])
            .inc();
        let val = TOKEN_BUDGET_OVERFLOW
            .with_label_values(&["g"])
            .get();
        assert!(val >= 1);
    }

    #[test]
    fn test_snapshot_metrics() {
        SNAPSHOT_DURATION.observe(1.5);
        assert!(SNAPSHOT_DURATION.get_sample_count() >= 1);

        SNAPSHOT_SIZE_BYTES.set(1_048_576);
        assert_eq!(SNAPSHOT_SIZE_BYTES.get(), 1_048_576);
    }

    #[test]
    fn test_ingest_metrics() {
        INGEST_DURATION
            .with_label_values(&["g", "text"])
            .observe(2.3);
        let count = INGEST_DURATION
            .with_label_values(&["g", "text"])
            .get_sample_count();
        assert!(count >= 1);

        INGEST_DOCUMENTS_TOTAL
            .with_label_values(&["g"])
            .inc();
        let val = INGEST_DOCUMENTS_TOTAL
            .with_label_values(&["g"])
            .get();
        assert!(val >= 1);
    }
}
