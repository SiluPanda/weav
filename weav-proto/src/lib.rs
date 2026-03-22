// Phase 6: Protocol definitions

pub mod resp3;
pub mod command;

#[cfg(feature = "grpc")]
pub mod grpc {
    tonic::include_proto!("weav.v1");
}
