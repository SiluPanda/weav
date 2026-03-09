//! Weav MCP server binary.
//!
//! Runs a Model Context Protocol server over stdio, exposing Weav's
//! graph database operations as MCP tools for AI agent integration.

use std::sync::Arc;

use anyhow::Result;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use weav_core::config::WeavConfig;
use weav_mcp::WeavMcpServer;
use weav_server::engine::Engine;

#[tokio::main]
async fn main() -> Result<()> {
    // Log to stderr so stdout is reserved for MCP JSON-RPC messages.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting Weav MCP server");

    // Create a Weav engine with default (in-memory) configuration.
    let mut config = WeavConfig::default();
    config.persistence.enabled = false; // MCP mode: in-memory only
    let engine = Arc::new(Engine::new(config));

    let server = WeavMcpServer::new(engine);
    let service = server.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("MCP serving error: {:?}", e);
    })?;

    service.waiting().await?;
    Ok(())
}
