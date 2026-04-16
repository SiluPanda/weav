// Phase 6: Protocol definitions

pub mod command;
pub mod resp3;

#[cfg(feature = "grpc")]
pub mod grpc {
    tonic::include_proto!("weav.v1");
}
