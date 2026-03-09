//! MCP (Model Context Protocol) server for Weav.
//!
//! Exposes Weav's context graph capabilities as MCP tools for AI agent
//! integration. Runs over stdio using the standard MCP transport.

pub mod tools;

use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use weav_server::engine::Engine;

/// The Weav MCP server handler.
///
/// Wraps an `Arc<Engine>` and exposes graph CRUD operations and context
/// queries as MCP tools that AI agents can invoke.
#[derive(Clone)]
pub struct WeavMcpServer {
    pub engine: Arc<Engine>,
    tool_router: ToolRouter<WeavMcpServer>,
}
