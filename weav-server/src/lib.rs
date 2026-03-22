// Phase 6: Server Layer
#![allow(
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::unnecessary_unwrap,
    clippy::manual_map,
    clippy::needless_borrows_for_generic_args,
    clippy::field_reassign_with_default
)]

pub mod engine;
pub mod http;
pub mod metrics;

#[cfg(feature = "resp3")]
pub mod resp3_server;

#[cfg(feature = "grpc")]
pub mod grpc_server;

#[cfg(feature = "tls")]
pub mod tls;
