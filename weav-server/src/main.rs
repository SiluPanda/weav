use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use weav_core::config::{WalSyncMode, WeavConfig};
use weav_server::{engine::Engine, http};

#[cfg(feature = "grpc")]
use weav_server::grpc_server::WeavGrpcService;
#[cfg(feature = "grpc")]
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

    // Set up graceful shutdown.
    let shutdown_token = CancellationToken::new();
    {
        let token = shutdown_token.clone();
        tokio::spawn(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to register SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = sigterm.recv() => {}
                };
            }
            #[cfg(not(unix))]
            ctrl_c.await.ok();

            tracing::info!("shutdown signal received, draining connections...");
            token.cancel();
        });
    }

    // Spawn background WAL sync task if configured for EverySecond mode.
    if config.persistence.enabled
        && matches!(config.persistence.wal_sync_mode, WalSyncMode::EverySecond)
    {
        let engine_wal = engine.clone();
        let token = shutdown_token.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = engine_wal.sync_wal() {
                            tracing::error!("WAL sync failed: {e}");
                        }
                    }
                    _ = token.cancelled() => {
                        tracing::info!("WAL sync task shutting down");
                        break;
                    }
                }
            }
        });
        tracing::info!("WAL EverySecond sync task started");
    }

    // Spawn background TTL sweep task (every 10 seconds)
    {
        let engine_ttl = engine.clone();
        let token = shutdown_token.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let expired = engine_ttl.sweep_ttl();
                        if expired > 0 {
                            tracing::info!("TTL sweep: removed {expired} expired entities");
                        }
                    }
                    _ = token.cancelled() => {
                        tracing::info!("TTL sweep task shutting down");
                        break;
                    }
                }
            }
        });
        tracing::info!("TTL sweep task started (10s interval)");
    }

    // HTTP server (always enabled).
    let app = http::build_router(engine.clone());
    let http_addr = format!(
        "{}:{}",
        config.server.bind_address,
        config.server.http_port.unwrap_or(6382)
    );
    tracing::info!("Weav HTTP server listening on {}", http_addr);

    let http_listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
    let http_shutdown_token = shutdown_token.clone();
    let http_handle = tokio::spawn(async move {
        axum::serve(http_listener, app)
            .with_graceful_shutdown(async move { http_shutdown_token.cancelled().await })
            .await
            .unwrap();
    });

    // RESP3 server (optional).
    #[cfg(feature = "resp3")]
    let resp3_handle = {
        let resp3_addr = format!(
            "{}:{}",
            config.server.bind_address,
            config.server.port
        );
        tracing::info!("Weav RESP3 server listening on {}", resp3_addr);
        let engine_resp3 = engine.clone();
        tokio::spawn(async move {
            weav_server::resp3_server::run_resp3_server(engine_resp3, &resp3_addr).await;
        })
    };

    // gRPC server (optional).
    #[cfg(feature = "grpc")]
    let grpc_handle = {
        let grpc_addr = format!(
            "{}:{}",
            config.server.bind_address,
            config.server.grpc_port.unwrap_or(6381)
        );
        tracing::info!("Weav gRPC server listening on {}", grpc_addr);
        let grpc_service = WeavGrpcService {
            engine: engine.clone(),
        };
        let grpc_addr_parsed: std::net::SocketAddr = grpc_addr.parse().unwrap();
        let grpc_shutdown_token = shutdown_token.clone();

        #[allow(unused_mut)]
        let mut builder = tonic::transport::Server::builder();

        #[cfg(feature = "tls")]
        if config.server.tls_enabled {
            if let (Some(cert_path), Some(key_path)) = (
                config.server.tls_cert_path.as_deref(),
                config.server.tls_key_path.as_deref(),
            ) {
                if let (Ok(cert), Ok(key)) = (
                    std::fs::read_to_string(cert_path),
                    std::fs::read_to_string(key_path),
                ) {
                    let tls = tonic::transport::ServerTlsConfig::new()
                        .identity(tonic::transport::Identity::from_pem(cert, key));
                    builder = builder.tls_config(tls).expect("invalid gRPC TLS config");
                    tracing::info!("gRPC TLS enabled");
                }
            }
        }

        tokio::spawn(async move {
            builder
                .add_service(WeavServiceServer::new(grpc_service))
                .serve_with_shutdown(grpc_addr_parsed, async move {
                    grpc_shutdown_token.cancelled().await;
                })
                .await
                .unwrap();
        })
    };

    // Wait for servers to finish.
    #[cfg(all(feature = "resp3", feature = "grpc"))]
    tokio::select! {
        _ = http_handle => {}
        _ = resp3_handle => {}
        _ = grpc_handle => {}
    }

    #[cfg(all(feature = "resp3", not(feature = "grpc")))]
    tokio::select! {
        _ = http_handle => {}
        _ = resp3_handle => {}
    }

    #[cfg(all(not(feature = "resp3"), feature = "grpc"))]
    tokio::select! {
        _ = http_handle => {}
        _ = grpc_handle => {}
    }

    #[cfg(all(not(feature = "resp3"), not(feature = "grpc")))]
    http_handle.await.unwrap();
}
