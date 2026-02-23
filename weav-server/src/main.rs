use std::sync::Arc;
use weav_core::config::{WalSyncMode, WeavConfig};
use weav_server::{engine::Engine, grpc_server::WeavGrpcService, http, resp3_server};
use weav_proto::grpc::weav_service_server::WeavServiceServer;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = WeavConfig::load(None).unwrap_or_default();
    let engine = Arc::new(Engine::new(config.clone()));

    // Recovery on startup.
    if config.persistence.enabled {
        let data_dir = config.persistence.data_dir.clone();
        let mgr = weav_persist::recovery::RecoveryManager::new(data_dir);
        match mgr.recover() {
            Ok(result) => {
                tracing::info!(
                    "Recovery: {} snapshots, {} WAL entries, {} graphs, {} errors",
                    result.snapshots_loaded,
                    result.wal_entries_replayed,
                    result.graphs_recovered,
                    result.errors.len(),
                );
                if let Err(e) = engine.recover(result) {
                    tracing::error!("Recovery failed: {e}");
                }
            }
            Err(e) => {
                tracing::error!("Recovery error: {e}");
            }
        }
    }

    // Spawn background WAL sync task if configured for EverySecond mode.
    if config.persistence.enabled {
        if matches!(config.persistence.wal_sync_mode, WalSyncMode::EverySecond) {
            let engine_wal = engine.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    engine_wal.sync_wal();
                }
            });
            tracing::info!("WAL EverySecond sync task started");
        }
    }

    let app = http::build_router(engine.clone());

    let http_addr = format!(
        "{}:{}",
        config.server.bind_address,
        config.server.http_port.unwrap_or(6382)
    );
    let resp3_addr = format!(
        "{}:{}",
        config.server.bind_address,
        config.server.port
    );

    tracing::info!("Weav HTTP server listening on {}", http_addr);
    tracing::info!("Weav RESP3 server listening on {}", resp3_addr);

    let grpc_addr = format!(
        "{}:{}",
        config.server.bind_address,
        config.server.grpc_port.unwrap_or(6381)
    );

    let engine_resp3 = engine.clone();
    let resp3_handle = tokio::spawn(async move {
        resp3_server::run_resp3_server(engine_resp3, &resp3_addr).await;
    });

    let http_listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
    let http_handle = tokio::spawn(async move {
        axum::serve(http_listener, app).await.unwrap();
    });

    let grpc_service = WeavGrpcService {
        engine: engine.clone(),
    };
    tracing::info!("Weav gRPC server listening on {}", grpc_addr);
    let grpc_addr_parsed: std::net::SocketAddr = grpc_addr.parse().unwrap();
    let grpc_handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(WeavServiceServer::new(grpc_service))
            .serve(grpc_addr_parsed)
            .await
            .unwrap();
    });

    tokio::select! {
        _ = http_handle => {}
        _ = resp3_handle => {}
        _ = grpc_handle => {}
    }
}
