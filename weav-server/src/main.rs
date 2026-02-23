use std::sync::Arc;
use weav_core::config::WeavConfig;
use weav_server::{engine::Engine, http};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config = WeavConfig::default();
    let engine = Arc::new(Engine::new(config));

    let app = http::build_router(engine);

    let addr = "0.0.0.0:6382";
    tracing::info!("Weav HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
