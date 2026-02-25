//! Argon2id password hashing and verification.

use argon2::{
    Argon2,
    PasswordHash,
    PasswordHasher,
    PasswordVerifier,
    password_hash::SaltString,
};
use rand::rngs::OsRng;

/// Hash a plaintext password using Argon2id with a random salt.
pub fn hash_password(password: &str) -> Result<String, String> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| format!("failed to hash password: {e}"))?;
    Ok(hash.to_string())
}

/// Verify a plaintext password against an Argon2id hash.
pub fn verify_password(password: &str, hash: &str) -> bool {
    let parsed = match PasswordHash::new(hash) {
        Ok(h) => h,
        Err(_) => return false,
    };
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_and_verify() {
        let password = "supersecret";
        let hash = hash_password(password).unwrap();
        assert!(verify_password(password, &hash));
    }

    #[test]
    fn test_wrong_password_fails() {
        let hash = hash_password("correct").unwrap();
        assert!(!verify_password("wrong", &hash));
    }

    #[test]
    fn test_different_hashes_for_same_password() {
        let hash1 = hash_password("same").unwrap();
        let hash2 = hash_password("same").unwrap();
        // Different salts should produce different hashes.
        assert_ne!(hash1, hash2);
        // But both should verify.
        assert!(verify_password("same", &hash1));
        assert!(verify_password("same", &hash2));
    }

    #[test]
    fn test_verify_invalid_hash_returns_false() {
        assert!(!verify_password("anything", "not-a-valid-hash"));
    }

    #[test]
    fn test_empty_password() {
        let hash = hash_password("").unwrap();
        assert!(verify_password("", &hash));
        assert!(!verify_password("notempty", &hash));
    }

    #[test]
    fn test_long_password() {
        let password = "a".repeat(1000);
        let hash = hash_password(&password).unwrap();
        assert!(verify_password(&password, &hash));
    }

    #[test]
    fn test_unicode_password() {
        let password = "p@$$w0rd_日本語_🔑";
        let hash = hash_password(password).unwrap();
        assert!(verify_password(password, &hash));
    }
}
