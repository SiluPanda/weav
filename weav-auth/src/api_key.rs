//! API key generation and verification.
//!
//! API keys are prefixed with `wk_` for easy identification. The raw key
//! is shown to the user once; the server stores only a SHA-256 hash.

use rand::Rng;
use sha2::{Digest, Sha256};

const API_KEY_PREFIX: &str = "wk_";
const API_KEY_RANDOM_BYTES: usize = 32;

/// Generate a new API key and return (raw_key, sha256_hash).
pub fn generate_api_key() -> (String, String) {
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; API_KEY_RANDOM_BYTES];
    rng.fill(&mut bytes);

    let hex_part: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let raw_key = format!("{API_KEY_PREFIX}{hex_part}");
    let hash = hash_api_key(&raw_key);
    (raw_key, hash)
}

/// Hash an API key using SHA-256. Returns hex-encoded hash.
pub fn hash_api_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

/// Verify an API key against a stored SHA-256 hash.
pub fn verify_api_key(key: &str, stored_hash: &str) -> bool {
    hash_api_key(key) == stored_hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_api_key_format() {
        let (key, _hash) = generate_api_key();
        assert!(key.starts_with("wk_"));
        // wk_ (3) + 64 hex chars = 67
        assert_eq!(key.len(), 3 + API_KEY_RANDOM_BYTES * 2);
    }

    #[test]
    fn test_generate_api_key_unique() {
        let (key1, _) = generate_api_key();
        let (key2, _) = generate_api_key();
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_and_verify_api_key() {
        let (key, hash) = generate_api_key();
        assert!(verify_api_key(&key, &hash));
    }

    #[test]
    fn test_wrong_key_fails() {
        let (_key, hash) = generate_api_key();
        assert!(!verify_api_key("wk_wrong", &hash));
    }

    #[test]
    fn test_hash_api_key_deterministic() {
        let key = "wk_test123";
        let hash1 = hash_api_key(key);
        let hash2 = hash_api_key(key);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_api_key_is_hex() {
        let hash = hash_api_key("wk_test");
        assert_eq!(hash.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_verify_empty_key_against_hash() {
        let hash = hash_api_key("");
        assert!(verify_api_key("", &hash));
        assert!(!verify_api_key("notempty", &hash));
    }
}
