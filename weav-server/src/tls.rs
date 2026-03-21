//! TLS configuration helper using rustls.

use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use rustls::ServerConfig;
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
        .ok_or_else(|| {
            WeavError::InvalidConfig("TLS key file contains no private key".into())
        })?;

    // Build rustls config
    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| WeavError::InvalidConfig(format!("failed to build TLS config: {e}")))?;

    Ok(Arc::new(config))
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
