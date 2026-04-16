use std::io;
use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;
use weav_core::config::{WalSyncMode, WeavConfig};
use weav_server::{engine::Engine, http};

#[cfg(feature = "grpc")]
use weav_proto::grpc::weav_service_server::WeavServiceServer;
#[cfg(feature = "grpc")]
use weav_server::grpc_server::WeavGrpcService;
#[cfg(feature = "tls")]
use weav_server::tls::{TlsListener, load_tls_config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = WeavConfig::load(None)?;
    let engine = Arc::new(Engine::new(config.clone()));

    #[cfg(feature = "tls")]
    let tls_config = if config.server.tls_enabled {
        let cert_path = config
            .server
            .tls_cert_path
            .as_deref()
            .ok_or_else(|| io::Error::other("missing tls_cert_path after validation"))?;
        let key_path = config
            .server
            .tls_key_path
            .as_deref()
            .ok_or_else(|| io::Error::other("missing tls_key_path after validation"))?;
        Some(load_tls_config(cert_path, key_path)?)
    } else {
        None
    };

    #[cfg(not(feature = "tls"))]
    if config.server.tls_enabled {
        return Err(io::Error::other(
            "tls_enabled requires building weav-server with the `tls` feature",
        )
        .into());
    }

    // Recovery on startup.
    if config.persistence.enabled {
        let data_dir = config.persistence.data_dir.clone();
        let mgr = weav_persist::recovery::RecoveryManager::new(data_dir);
        let result = mgr.recover()?;
        tracing::info!(
            "Recovery: {} snapshots, {} WAL entries, {} graphs, {} errors",
            result.snapshots_loaded,
            result.wal_entries_replayed,
            result.graphs_recovered,
            result.errors.len(),
        );
        if !result.errors.is_empty() {
            for err in &result.errors {
                tracing::error!("Recovery warning: {}", err);
            }
            return Err(io::Error::other(format!(
                "recovery encountered {} error(s); refusing to start",
                result.errors.len()
            ))
            .into());
        }
        engine.recover(result)?;
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
            let mut interval = tokio::time::interval(Duration::from_secs(1));
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

    // Spawn background snapshots when enabled.
    if config.persistence.enabled && config.persistence.snapshot_interval_secs > 0 {
        let engine_snapshot = engine.clone();
        let token = shutdown_token.clone();
        let snapshot_interval = Duration::from_secs(config.persistence.snapshot_interval_secs);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(snapshot_interval);
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(err) = engine_snapshot.snapshot() {
                            tracing::error!("snapshot task failed: {}", err);
                        }
                    }
                    _ = token.cancelled() => {
                        tracing::info!("snapshot task shutting down");
                        break;
                    }
                }
            }
        });
        tracing::info!(
            "Snapshot task started ({}s interval)",
            config.persistence.snapshot_interval_secs
        );
    }

    // Spawn background TTL sweep task (every 10 seconds)
    {
        let engine_ttl = engine.clone();
        let token = shutdown_token.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
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
    tracing::info!(
        "Weav HTTP server listening on {}{}",
        if config.server.tls_enabled {
            "https://"
        } else {
            "http://"
        },
        http_addr
    );

    let http_listener = tokio::net::TcpListener::bind(&http_addr).await?;
    let http_shutdown_token = shutdown_token.clone();
    #[cfg(feature = "tls")]
    let http_tls_config = tls_config.clone();
    let http_handle = tokio::spawn(async move {
        #[cfg(feature = "tls")]
        if let Some(tls_config) = http_tls_config {
            return axum::serve(TlsListener::new(http_listener, tls_config), app)
                .with_graceful_shutdown(async move { http_shutdown_token.cancelled().await })
                .await;
        }

        axum::serve(http_listener, app)
            .with_graceful_shutdown(async move { http_shutdown_token.cancelled().await })
            .await
    });

    // RESP3 server (optional).
    #[cfg(feature = "resp3")]
    let resp3_handle = {
        let resp3_addr = format!("{}:{}", config.server.bind_address, config.server.port);
        tracing::info!(
            "Weav RESP3 server listening on {}{}",
            if config.server.tls_enabled {
                "tls://"
            } else {
                ""
            },
            resp3_addr
        );
        let engine_resp3 = engine.clone();
        #[cfg(feature = "tls")]
        let resp3_tls_acceptor = tls_config.clone().map(tokio_rustls::TlsAcceptor::from);
        tokio::spawn(async move {
            #[cfg(feature = "tls")]
            {
                weav_server::resp3_server::run_resp3_server(
                    engine_resp3,
                    &resp3_addr,
                    resp3_tls_acceptor,
                )
                .await
            }
            #[cfg(not(feature = "tls"))]
            {
                weav_server::resp3_server::run_resp3_server(engine_resp3, &resp3_addr).await
            }
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
        let grpc_addr_parsed: std::net::SocketAddr = grpc_addr.parse()?;
        let grpc_shutdown_token = shutdown_token.clone();

        #[allow(unused_mut)]
        let mut builder = tonic::transport::Server::builder();

        #[cfg(feature = "tls")]
        if config.server.tls_enabled {
            let cert_path = config
                .server
                .tls_cert_path
                .as_deref()
                .ok_or_else(|| io::Error::other("missing tls_cert_path after validation"))?;
            let key_path = config
                .server
                .tls_key_path
                .as_deref()
                .ok_or_else(|| io::Error::other("missing tls_key_path after validation"))?;
            let cert = std::fs::read_to_string(cert_path)?;
            let key = std::fs::read_to_string(key_path)?;
            let tls = tonic::transport::ServerTlsConfig::new()
                .identity(tonic::transport::Identity::from_pem(cert, key));
            builder = builder.tls_config(tls)?;
            tracing::info!("gRPC TLS enabled");
        }

        tokio::spawn(async move {
            builder
                .add_service(WeavServiceServer::new(grpc_service))
                .serve_with_shutdown(grpc_addr_parsed, async move {
                    grpc_shutdown_token.cancelled().await;
                })
                .await
        })
    };

    // Wait for servers to finish.
    #[cfg(all(feature = "resp3", feature = "grpc"))]
    tokio::select! {
        result = http_handle => { result??; }
        result = resp3_handle => { result??; }
        result = grpc_handle => { result??; }
    }

    #[cfg(all(feature = "resp3", not(feature = "grpc")))]
    tokio::select! {
        result = http_handle => { result??; }
        result = resp3_handle => { result??; }
    }

    #[cfg(all(not(feature = "resp3"), feature = "grpc"))]
    tokio::select! {
        result = http_handle => { result??; }
        result = grpc_handle => { result??; }
    }

    #[cfg(all(not(feature = "resp3"), not(feature = "grpc")))]
    http_handle.await??;

    if config.persistence.enabled {
        engine.sync_wal()?;
    }

    Ok(())
}
