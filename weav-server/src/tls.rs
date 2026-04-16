//! TLS configuration helper using rustls.

use std::fs;
use std::io::BufReader;
use std::sync::Arc;
use std::time::Duration;

use axum::serve::Listener;
use rustls::ServerConfig;
use tokio::io;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::{TlsAcceptor, server::TlsStream};
use weav_core::error::{WeavError, WeavResult};

/// Load a rustls `ServerConfig` from PEM certificate and key files.
pub fn load_tls_config(cert_path: &str, key_path: &str) -> WeavResult<Arc<ServerConfig>> {
    // Read certificate chain
    let cert_file = fs::File::open(cert_path).map_err(|e| {
        WeavError::InvalidConfig(format!("failed to open TLS cert file '{cert_path}': {e}"))
    })?;
    let mut cert_reader = BufReader::new(cert_file);
    let certs: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                WeavError::InvalidConfig(format!("failed to parse TLS certificates: {e}"))
            })?;

    if certs.is_empty() {
        return Err(WeavError::InvalidConfig(
            "TLS certificate file contains no certificates".into(),
        ));
    }

    // Read private key
    let key_file = fs::File::open(key_path).map_err(|e| {
        WeavError::InvalidConfig(format!("failed to open TLS key file '{key_path}': {e}"))
    })?;
    let mut key_reader = BufReader::new(key_file);
    let key = rustls_pemfile::private_key(&mut key_reader)
        .map_err(|e| WeavError::InvalidConfig(format!("failed to parse TLS private key: {e}")))?
        .ok_or_else(|| WeavError::InvalidConfig("TLS key file contains no private key".into()))?;

    // Build rustls config
    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| WeavError::InvalidConfig(format!("failed to build TLS config: {e}")))?;

    Ok(Arc::new(config))
}

/// Axum-compatible TLS listener for HTTPS serving.
pub struct TlsListener {
    listener: TcpListener,
    acceptor: TlsAcceptor,
}

impl TlsListener {
    pub fn new(listener: TcpListener, config: Arc<ServerConfig>) -> Self {
        Self {
            listener,
            acceptor: TlsAcceptor::from(config),
        }
    }
}

impl Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, addr) = match self.listener.accept().await {
                Ok(conn) => conn,
                Err(err) => {
                    handle_accept_error(err).await;
                    continue;
                }
            };

            match self.acceptor.accept(stream).await {
                Ok(tls_stream) => return (tls_stream, addr),
                Err(err) => {
                    tracing::warn!("TLS handshake failed from {}: {}", addr, err);
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

async fn handle_accept_error(err: io::Error) {
    if is_connection_error(&err) {
        return;
    }

    tracing::error!("accept error: {}", err);
    tokio::time::sleep(Duration::from_secs(1)).await;
}

fn is_connection_error(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tls_config_missing_cert() {
        let result = load_tls_config("/nonexistent/cert.pem", "/nonexistent/key.pem");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cert"), "error should mention cert: {err}");
    }

    #[test]
    fn test_load_tls_config_missing_key() {
        // Create a temp cert file but point to missing key
        let dir = std::env::temp_dir().join("weav_tls_test");
        let _ = std::fs::create_dir_all(&dir);
        let cert_path = dir.join("empty_cert.pem");
        std::fs::write(&cert_path, "").unwrap();

        let result = load_tls_config(cert_path.to_str().unwrap(), "/nonexistent/key.pem");
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
