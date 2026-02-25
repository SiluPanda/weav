//! ACL user store — thread-safe user management with CRUD operations.

use std::collections::HashMap;

use parking_lot::RwLock;

use weav_core::config::{AuthConfig, UserConfig};
use weav_core::error::{WeavError, WeavResult};

use crate::api_key;
use crate::identity::{
    CommandCategorySet, GraphAcl, GraphPermission, SessionIdentity, UserPermissions,
};
use crate::password;

/// Stored ACL user definition.
#[derive(Debug, Clone)]
pub struct AclUser {
    pub username: String,
    /// Argon2id password hash. None means no password auth.
    pub password_hash: Option<String>,
    /// Whether the user is enabled.
    pub enabled: bool,
    /// Allowed command categories.
    pub categories: CommandCategorySet,
    /// Graph-level access patterns.
    pub graph_acl: Vec<GraphAcl>,
    /// SHA-256 hashes of API keys.
    pub api_key_hashes: Vec<String>,
}

impl AclUser {
    /// Build a SessionIdentity for this user.
    pub fn to_identity(&self) -> SessionIdentity {
        SessionIdentity {
            username: self.username.clone(),
            permissions: UserPermissions {
                allowed_categories: self.categories.clone(),
                graph_acl: self.graph_acl.clone(),
            },
        }
    }
}

/// Thread-safe ACL user store.
pub struct AclStore {
    users: RwLock<HashMap<String, AclUser>>,
    /// Default password for Redis-compat single-password AUTH.
    default_password: Option<String>,
    /// Whether auth is required for all connections.
    require_auth: bool,
}

impl AclStore {
    /// Create a new AclStore from the auth config.
    pub fn from_config(config: &AuthConfig) -> Self {
        let mut users = HashMap::new();

        // Load statically-defined users from config.
        for user_config in &config.users {
            let user = acl_user_from_config(user_config);
            users.insert(user.username.clone(), user);
        }

        // If there's a default password and no "default" user, create one.
        if config.default_password.is_some() && !users.contains_key("default") {
            let hash = config
                .default_password
                .as_ref()
                .map(|p| password::hash_password(p).unwrap_or_default());
            users.insert(
                "default".into(),
                AclUser {
                    username: "default".into(),
                    password_hash: hash,
                    enabled: true,
                    categories: CommandCategorySet::all(),
                    graph_acl: Vec::new(),
                    api_key_hashes: Vec::new(),
                },
            );
        }

        Self {
            users: RwLock::new(users),
            default_password: config.default_password.clone(),
            require_auth: config.require_auth,
        }
    }

    /// Whether authentication is required.
    pub fn require_auth(&self) -> bool {
        self.require_auth
    }

    /// Authenticate with username + password. Returns a SessionIdentity on success.
    pub fn authenticate(&self, username: &str, password_raw: &str) -> WeavResult<SessionIdentity> {
        let users = self.users.read();

        // Redis-compat: AUTH <password> only (no username) maps to "default" user.
        let user = users.get(username).ok_or_else(|| {
            WeavError::AuthenticationFailed(format!("user '{}' not found", username))
        })?;

        if !user.enabled {
            return Err(WeavError::AuthenticationFailed(format!(
                "user '{}' is disabled",
                username
            )));
        }

        let password_hash = user.password_hash.as_ref().ok_or_else(|| {
            WeavError::AuthenticationFailed("user has no password configured".into())
        })?;

        if !password::verify_password(password_raw, password_hash) {
            return Err(WeavError::AuthenticationFailed("invalid password".into()));
        }

        Ok(user.to_identity())
    }

    /// Authenticate with default password (Redis-compat AUTH <password>).
    pub fn authenticate_default(&self, password_raw: &str) -> WeavResult<SessionIdentity> {
        match &self.default_password {
            Some(default_pw) if default_pw == password_raw => {
                // Return the "default" user identity or full-access identity.
                let users = self.users.read();
                if let Some(user) = users.get("default") {
                    Ok(user.to_identity())
                } else {
                    Ok(SessionIdentity {
                        username: "default".into(),
                        permissions: UserPermissions::full_access(),
                    })
                }
            }
            _ => Err(WeavError::AuthenticationFailed("invalid password".into())),
        }
    }

    /// Authenticate with an API key. Returns a SessionIdentity on success.
    pub fn authenticate_api_key(&self, key: &str) -> WeavResult<SessionIdentity> {
        let key_hash = api_key::hash_api_key(key);
        let users = self.users.read();

        for user in users.values() {
            if !user.enabled {
                continue;
            }
            for stored_hash in &user.api_key_hashes {
                if *stored_hash == key_hash {
                    return Ok(user.to_identity());
                }
            }
        }

        Err(WeavError::AuthenticationFailed("invalid API key".into()))
    }

    /// Get a user by username.
    pub fn get_user(&self, username: &str) -> Option<AclUser> {
        self.users.read().get(username).cloned()
    }

    /// Set (create or update) a user.
    pub fn set_user(&self, user: AclUser) {
        self.users.write().insert(user.username.clone(), user);
    }

    /// Delete a user by username. Returns true if the user existed.
    pub fn delete_user(&self, username: &str) -> bool {
        self.users.write().remove(username).is_some()
    }

    /// List all usernames.
    pub fn list_users(&self) -> Vec<String> {
        self.users.read().keys().cloned().collect()
    }

    /// Serialize all users to ACL file format (one user per line).
    pub fn serialize_acl(&self) -> String {
        let users = self.users.read();
        let mut lines = Vec::new();
        for user in users.values() {
            let mut parts = vec![format!("user {}", user.username)];
            if user.enabled {
                parts.push("on".into());
            } else {
                parts.push("off".into());
            }
            // We can't serialize the raw password from a hash, so just mark it.
            if user.password_hash.is_some() {
                parts.push("#<hashed>".into());
            }
            for acl in &user.graph_acl {
                parts.push(format!("~{}:{}", acl.pattern, match acl.permission {
                    GraphPermission::Read => "read",
                    GraphPermission::ReadWrite => "readwrite",
                    GraphPermission::Admin => "admin",
                    GraphPermission::None => "none",
                }));
            }
            lines.push(parts.join(" "));
        }
        lines.join("\n")
    }

    /// Load users from ACL file content (overrides by username).
    pub fn load_acl(&self, content: &str) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(user) = parse_acl_line(line) {
                self.set_user(user);
            }
        }
    }
}

/// Build an AclUser from a UserConfig.
fn acl_user_from_config(config: &UserConfig) -> AclUser {
    let password_hash = config
        .password
        .as_ref()
        .map(|p| password::hash_password(p).unwrap_or_default());

    let categories = if config.categories.is_empty() {
        CommandCategorySet::all()
    } else {
        CommandCategorySet::from_acl_strings(&config.categories)
    };

    let graph_acl: Vec<GraphAcl> = config
        .graph_patterns
        .iter()
        .map(|gp| GraphAcl {
            pattern: gp.pattern.clone(),
            permission: GraphPermission::from_str(&gp.permission),
        })
        .collect();

    let api_key_hashes: Vec<String> = config
        .api_keys
        .iter()
        .map(|k| api_key::hash_api_key(k))
        .collect();

    AclUser {
        username: config.username.clone(),
        password_hash,
        enabled: config.enabled,
        categories,
        graph_acl,
        api_key_hashes,
    }
}

/// Parse a single ACL file line into an AclUser.
fn parse_acl_line(line: &str) -> Option<AclUser> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 || parts[0] != "user" {
        return None;
    }

    let username = parts[1].to_string();
    let mut enabled = true;
    let mut password_hash = None;
    let mut graph_acl = Vec::new();

    for part in &parts[2..] {
        match *part {
            "on" => enabled = true,
            "off" => enabled = false,
            p if p.starts_with('#') => {
                // Hashed password marker (we preserve the hash).
                if p != "#<hashed>" {
                    password_hash = Some(p[1..].to_string());
                }
            }
            p if p.starts_with('>') => {
                // Plaintext password — hash it.
                password_hash = Some(
                    password::hash_password(&p[1..]).unwrap_or_default(),
                );
            }
            p if p.starts_with('~') => {
                // Graph pattern: ~pattern:permission
                let inner = &p[1..];
                if let Some((pat, perm)) = inner.rsplit_once(':') {
                    graph_acl.push(GraphAcl {
                        pattern: pat.to_string(),
                        permission: GraphPermission::from_str(perm),
                    });
                }
            }
            _ => {}
        }
    }

    Some(AclUser {
        username,
        password_hash,
        enabled,
        categories: CommandCategorySet::all(),
        graph_acl,
        api_key_hashes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::CommandCategory;

    fn make_config(users: Vec<UserConfig>) -> AuthConfig {
        AuthConfig {
            enabled: true,
            require_auth: true,
            acl_file: None,
            default_password: None,
            users,
        }
    }

    fn make_user_config(username: &str, password: &str) -> UserConfig {
        UserConfig {
            username: username.into(),
            password: Some(password.into()),
            categories: vec!["+@read".into(), "+@write".into(), "+@connection".into()],
            graph_patterns: Vec::new(),
            api_keys: Vec::new(),
            enabled: true,
        }
    }

    #[test]
    fn test_acl_store_from_config_empty() {
        let config = make_config(vec![]);
        let store = AclStore::from_config(&config);
        assert!(store.list_users().is_empty());
    }

    #[test]
    fn test_acl_store_authenticate_success() {
        let config = make_config(vec![make_user_config("alice", "secret123")]);
        let store = AclStore::from_config(&config);

        let identity = store.authenticate("alice", "secret123").unwrap();
        assert_eq!(identity.username, "alice");
        assert!(identity.permissions.has_category(CommandCategory::Read));
    }

    #[test]
    fn test_acl_store_authenticate_wrong_password() {
        let config = make_config(vec![make_user_config("alice", "secret123")]);
        let store = AclStore::from_config(&config);

        let result = store.authenticate("alice", "wrong");
        assert!(result.is_err());
    }

    #[test]
    fn test_acl_store_authenticate_unknown_user() {
        let config = make_config(vec![make_user_config("alice", "secret123")]);
        let store = AclStore::from_config(&config);

        let result = store.authenticate("bob", "anything");
        assert!(result.is_err());
    }

    #[test]
    fn test_acl_store_authenticate_disabled_user() {
        let mut uc = make_user_config("alice", "secret123");
        uc.enabled = false;
        let config = make_config(vec![uc]);
        let store = AclStore::from_config(&config);

        let result = store.authenticate("alice", "secret123");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[test]
    fn test_acl_store_default_password() {
        let config = AuthConfig {
            enabled: true,
            require_auth: true,
            acl_file: None,
            default_password: Some("defaultpass".into()),
            users: vec![],
        };
        let store = AclStore::from_config(&config);

        let identity = store.authenticate_default("defaultpass").unwrap();
        assert_eq!(identity.username, "default");

        let result = store.authenticate_default("wrong");
        assert!(result.is_err());
    }

    #[test]
    fn test_acl_store_api_key_auth() {
        let api_key = "wk_testapikey12345";
        let uc = UserConfig {
            username: "apiuser".into(),
            password: None,
            categories: vec!["+@read".into()],
            graph_patterns: Vec::new(),
            api_keys: vec![api_key.into()],
            enabled: true,
        };
        let config = make_config(vec![uc]);
        let store = AclStore::from_config(&config);

        let identity = store.authenticate_api_key(api_key).unwrap();
        assert_eq!(identity.username, "apiuser");

        let result = store.authenticate_api_key("wk_wrong");
        assert!(result.is_err());
    }

    #[test]
    fn test_acl_store_crud() {
        let config = make_config(vec![]);
        let store = AclStore::from_config(&config);

        // Create.
        let user = AclUser {
            username: "bob".into(),
            password_hash: Some(password::hash_password("pass").unwrap()),
            enabled: true,
            categories: CommandCategorySet::all(),
            graph_acl: Vec::new(),
            api_key_hashes: Vec::new(),
        };
        store.set_user(user);

        // Read.
        let bob = store.get_user("bob").unwrap();
        assert_eq!(bob.username, "bob");

        // List.
        let users = store.list_users();
        assert_eq!(users, vec!["bob"]);

        // Authenticate.
        let identity = store.authenticate("bob", "pass").unwrap();
        assert_eq!(identity.username, "bob");

        // Delete.
        assert!(store.delete_user("bob"));
        assert!(store.get_user("bob").is_none());
        assert!(!store.delete_user("bob")); // already deleted
    }

    #[test]
    fn test_acl_store_require_auth() {
        let config = AuthConfig {
            enabled: true,
            require_auth: true,
            ..Default::default()
        };
        let store = AclStore::from_config(&config);
        assert!(store.require_auth());

        let config2 = AuthConfig {
            enabled: true,
            require_auth: false,
            ..Default::default()
        };
        let store2 = AclStore::from_config(&config2);
        assert!(!store2.require_auth());
    }

    #[test]
    fn test_acl_store_graph_acl_permissions() {
        let uc = UserConfig {
            username: "reader".into(),
            password: Some("pass".into()),
            categories: vec!["+@read".into(), "+@connection".into()],
            graph_patterns: vec![
                weav_core::config::GraphPatternConfig {
                    pattern: "app:*".into(),
                    permission: "read".into(),
                },
            ],
            api_keys: Vec::new(),
            enabled: true,
        };
        let config = make_config(vec![uc]);
        let store = AclStore::from_config(&config);

        let identity = store.authenticate("reader", "pass").unwrap();
        assert!(identity.permissions.can_read_graph("app:users"));
        assert!(!identity.permissions.can_write_graph("app:users"));
        assert!(!identity.permissions.can_read_graph("other"));
    }

    #[test]
    fn test_parse_acl_line_basic() {
        let user = parse_acl_line("user alice on >secret ~*:readwrite").unwrap();
        assert_eq!(user.username, "alice");
        assert!(user.enabled);
        assert!(user.password_hash.is_some());
        assert_eq!(user.graph_acl.len(), 1);
        assert_eq!(user.graph_acl[0].pattern, "*");
    }

    #[test]
    fn test_parse_acl_line_disabled() {
        let user = parse_acl_line("user bob off").unwrap();
        assert_eq!(user.username, "bob");
        assert!(!user.enabled);
    }

    #[test]
    fn test_parse_acl_line_invalid() {
        assert!(parse_acl_line("not a user line").is_none());
        assert!(parse_acl_line("").is_none());
    }

    #[test]
    fn test_load_acl() {
        let config = make_config(vec![]);
        let store = AclStore::from_config(&config);

        let acl_content = "# comment\nuser alice on >pass123\nuser bob off\n";
        store.load_acl(acl_content);

        let users = store.list_users();
        assert_eq!(users.len(), 2);
        assert!(store.get_user("alice").unwrap().enabled);
        assert!(!store.get_user("bob").unwrap().enabled);
    }

    #[test]
    fn test_serialize_acl() {
        let config = make_config(vec![make_user_config("alice", "pass")]);
        let store = AclStore::from_config(&config);

        let serialized = store.serialize_acl();
        assert!(serialized.contains("user alice"));
        assert!(serialized.contains("on"));
    }

    #[test]
    fn test_acl_user_to_identity() {
        let user = AclUser {
            username: "test".into(),
            password_hash: None,
            enabled: true,
            categories: CommandCategorySet::all(),
            graph_acl: Vec::new(),
            api_key_hashes: Vec::new(),
        };
        let identity = user.to_identity();
        assert_eq!(identity.username, "test");
        assert!(identity.permissions.has_category(CommandCategory::Read));
    }

    #[test]
    fn test_acl_store_no_password_user() {
        let uc = UserConfig {
            username: "nopass".into(),
            password: None,
            categories: vec!["+@read".into()],
            graph_patterns: Vec::new(),
            api_keys: Vec::new(),
            enabled: true,
        };
        let config = make_config(vec![uc]);
        let store = AclStore::from_config(&config);

        let result = store.authenticate("nopass", "anything");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no password"));
    }
}
