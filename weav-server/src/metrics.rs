use prometheus::{Registry, IntGaugeVec, HistogramVec, IntCounterVec, Opts, HistogramOpts};
use std::sync::LazyLock;

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

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

pub static CONNECTIONS_ACTIVE: LazyLock<prometheus::IntGauge> = LazyLock::new(|| {
    let gauge = prometheus::IntGauge::new("weav_connections_active", "Active connections").unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});
