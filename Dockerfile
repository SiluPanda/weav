# Weav — In-Memory Context Graph Database for AI/LLM Workloads
# Multi-stage build: builder + minimal runtime image

# ─── Builder Stage ────────────────────────────────────────────────────────────

FROM rust:1.85-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake g++ protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY weav-core/Cargo.toml weav-core/Cargo.toml
COPY weav-graph/Cargo.toml weav-graph/Cargo.toml
COPY weav-vector/Cargo.toml weav-vector/Cargo.toml
COPY weav-auth/Cargo.toml weav-auth/Cargo.toml
COPY weav-extract/Cargo.toml weav-extract/Cargo.toml
COPY weav-query/Cargo.toml weav-query/Cargo.toml
COPY weav-persist/Cargo.toml weav-persist/Cargo.toml
COPY weav-proto/Cargo.toml weav-proto/Cargo.toml
COPY weav-server/Cargo.toml weav-server/Cargo.toml
COPY weav-cli/Cargo.toml weav-cli/Cargo.toml
COPY weav-mcp/Cargo.toml weav-mcp/Cargo.toml
COPY benchmarks/Cargo.toml benchmarks/Cargo.toml

# Create stub lib.rs for each crate to cache dependencies
RUN for dir in weav-core weav-graph weav-vector weav-auth weav-extract \
    weav-query weav-persist weav-proto weav-server weav-cli weav-mcp benchmarks; do \
    mkdir -p "$dir/src" && echo "" > "$dir/src/lib.rs"; done && \
    mkdir -p weav-server/src && echo "fn main() {}" > weav-server/src/main.rs && \
    mkdir -p weav-cli/src && echo "fn main() {}" > weav-cli/src/main.rs && \
    mkdir -p weav-mcp/src && echo "fn main() {}" > weav-mcp/src/main.rs

# Cache dependency build
RUN cargo build --release -p weav-server --features weav-server/full 2>/dev/null || true

# Copy actual source
COPY . .

# Build release binary
RUN cargo build --release -p weav-server --features weav-server/full && \
    cargo build --release -p weav-cli && \
    cargo build --release -p weav-mcp

# ─── Runtime Stage ────────────────────────────────────────────────────────────

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r weav && useradd -r -g weav -d /data -s /sbin/nologin weav
RUN mkdir -p /data && chown weav:weav /data

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/target/release/weav-server /app/weav-server
COPY --from=builder /build/target/release/weav-cli /app/weav-cli
COPY --from=builder /build/target/release/weav-mcp /app/weav-mcp

# Expose ports: RESP3, gRPC, HTTP
EXPOSE 6380 6381 6382

# Data directory for WAL + snapshots
VOLUME /data

# Environment defaults
ENV WEAV_SERVER_BIND_ADDRESS=0.0.0.0
ENV WEAV_PERSISTENCE_ENABLED=true
ENV WEAV_PERSISTENCE_DATA_DIR=/data

# Health check via HTTP
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6382/health || exit 1

USER weav

ENTRYPOINT ["/app/weav-server"]
