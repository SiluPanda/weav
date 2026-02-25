//! Session identity and permission types for Weav authentication.

use std::collections::HashSet;

/// Represents the authenticated identity for a connection/request.
#[derive(Debug, Clone)]
pub struct SessionIdentity {
    pub username: String,
    pub permissions: UserPermissions,
}

/// Permissions associated with a user.
#[derive(Debug, Clone)]
pub struct UserPermissions {
    /// Allowed command categories.
    pub allowed_categories: CommandCategorySet,
    /// Graph-level access control rules. Empty means all graphs allowed.
    pub graph_acl: Vec<GraphAcl>,
}

impl UserPermissions {
    /// Full-access permissions (all categories, all graphs).
    pub fn full_access() -> Self {
        Self {
            allowed_categories: CommandCategorySet::all(),
            graph_acl: Vec::new(),
        }
    }

    /// Check if the user has permission for the given command category.
    pub fn has_category(&self, category: CommandCategory) -> bool {
        self.allowed_categories.contains(category)
    }

    /// Check if the user has read access to the given graph.
    pub fn can_read_graph(&self, graph_name: &str) -> bool {
        self.check_graph_permission(graph_name, GraphPermission::Read)
    }

    /// Check if the user has write access to the given graph.
    pub fn can_write_graph(&self, graph_name: &str) -> bool {
        self.check_graph_permission(graph_name, GraphPermission::ReadWrite)
    }

    /// Check if the user has admin access to the given graph.
    pub fn can_admin_graph(&self, graph_name: &str) -> bool {
        self.check_graph_permission(graph_name, GraphPermission::Admin)
    }

    fn check_graph_permission(&self, graph_name: &str, required: GraphPermission) -> bool {
        // Empty ACL means full access to all graphs.
        if self.graph_acl.is_empty() {
            return true;
        }

        for acl in &self.graph_acl {
            if acl.matches(graph_name) {
                return acl.permission >= required;
            }
        }

        // No matching pattern means no access.
        false
    }
}

/// Command categories for authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandCategory {
    Connection,
    Read,
    Write,
    Admin,
}

impl CommandCategory {
    /// Parse a category from a string like "+@read" or "-@write".
    pub fn from_acl_str(s: &str) -> Option<(bool, Self)> {
        let s = s.trim();
        let (grant, name) = if let Some(rest) = s.strip_prefix("+@") {
            (true, rest)
        } else if let Some(rest) = s.strip_prefix("-@") {
            (false, rest)
        } else {
            return None;
        };

        let cat = match name.to_lowercase().as_str() {
            "connection" => CommandCategory::Connection,
            "read" => CommandCategory::Read,
            "write" => CommandCategory::Write,
            "admin" => CommandCategory::Admin,
            "all" => return Some((grant, CommandCategory::Admin)), // handled specially
            _ => return None,
        };
        Some((grant, cat))
    }
}

/// A set of command categories, used for efficient permission checks.
#[derive(Debug, Clone)]
pub struct CommandCategorySet {
    categories: HashSet<CommandCategory>,
}

impl CommandCategorySet {
    /// Empty set (no permissions).
    pub fn empty() -> Self {
        Self {
            categories: HashSet::new(),
        }
    }

    /// Full set (all permissions).
    pub fn all() -> Self {
        let mut categories = HashSet::new();
        categories.insert(CommandCategory::Connection);
        categories.insert(CommandCategory::Read);
        categories.insert(CommandCategory::Write);
        categories.insert(CommandCategory::Admin);
        Self { categories }
    }

    /// Build from ACL category strings like "+@read", "+@write", "-@admin".
    pub fn from_acl_strings(strings: &[String]) -> Self {
        let mut set = Self::empty();
        for s in strings {
            if s == "+@all" {
                set = Self::all();
                continue;
            }
            if s == "-@all" {
                set = Self::empty();
                continue;
            }
            if let Some((grant, cat)) = CommandCategory::from_acl_str(s) {
                if grant {
                    set.categories.insert(cat);
                } else {
                    set.categories.remove(&cat);
                }
            }
        }
        set
    }

    /// Check if the set contains a category.
    pub fn contains(&self, category: CommandCategory) -> bool {
        self.categories.contains(&category)
    }

    pub fn insert(&mut self, category: CommandCategory) {
        self.categories.insert(category);
    }

    pub fn remove(&mut self, category: CommandCategory) {
        self.categories.remove(&category);
    }
}

/// Graph-level access control entry.
#[derive(Debug, Clone)]
pub struct GraphAcl {
    /// Glob pattern for graph names (e.g., "app:*", "*", "shared").
    pub pattern: String,
    /// Permission level for matching graphs.
    pub permission: GraphPermission,
}

impl GraphAcl {
    /// Check if a graph name matches this ACL pattern.
    pub fn matches(&self, graph_name: &str) -> bool {
        glob_match::glob_match(&self.pattern, graph_name)
    }
}

/// Permission level for graph access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GraphPermission {
    /// No access.
    None = 0,
    /// Read-only access.
    Read = 1,
    /// Read and write access.
    ReadWrite = 2,
    /// Full admin access (create, drop, etc.).
    Admin = 3,
}

impl GraphPermission {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "read" | "r" => GraphPermission::Read,
            "readwrite" | "rw" | "write" => GraphPermission::ReadWrite,
            "admin" | "all" => GraphPermission::Admin,
            _ => GraphPermission::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_identity_creation() {
        let identity = SessionIdentity {
            username: "alice".into(),
            permissions: UserPermissions::full_access(),
        };
        assert_eq!(identity.username, "alice");
        assert!(identity.permissions.has_category(CommandCategory::Read));
        assert!(identity.permissions.has_category(CommandCategory::Write));
        assert!(identity.permissions.has_category(CommandCategory::Admin));
        assert!(identity.permissions.has_category(CommandCategory::Connection));
    }

    #[test]
    fn test_command_category_set_empty() {
        let set = CommandCategorySet::empty();
        assert!(!set.contains(CommandCategory::Read));
        assert!(!set.contains(CommandCategory::Write));
        assert!(!set.contains(CommandCategory::Admin));
        assert!(!set.contains(CommandCategory::Connection));
    }

    #[test]
    fn test_command_category_set_all() {
        let set = CommandCategorySet::all();
        assert!(set.contains(CommandCategory::Read));
        assert!(set.contains(CommandCategory::Write));
        assert!(set.contains(CommandCategory::Admin));
        assert!(set.contains(CommandCategory::Connection));
    }

    #[test]
    fn test_command_category_set_from_acl_strings() {
        let strings = vec!["+@read".into(), "+@connection".into()];
        let set = CommandCategorySet::from_acl_strings(&strings);
        assert!(set.contains(CommandCategory::Read));
        assert!(set.contains(CommandCategory::Connection));
        assert!(!set.contains(CommandCategory::Write));
        assert!(!set.contains(CommandCategory::Admin));
    }

    #[test]
    fn test_command_category_set_plus_all() {
        let strings = vec!["+@all".into()];
        let set = CommandCategorySet::from_acl_strings(&strings);
        assert!(set.contains(CommandCategory::Read));
        assert!(set.contains(CommandCategory::Write));
        assert!(set.contains(CommandCategory::Admin));
        assert!(set.contains(CommandCategory::Connection));
    }

    #[test]
    fn test_command_category_set_add_then_remove() {
        let strings = vec!["+@all".into(), "-@admin".into()];
        let set = CommandCategorySet::from_acl_strings(&strings);
        assert!(set.contains(CommandCategory::Read));
        assert!(set.contains(CommandCategory::Write));
        assert!(!set.contains(CommandCategory::Admin));
    }

    #[test]
    fn test_command_category_from_acl_str() {
        assert_eq!(
            CommandCategory::from_acl_str("+@read"),
            Some((true, CommandCategory::Read))
        );
        assert_eq!(
            CommandCategory::from_acl_str("-@write"),
            Some((false, CommandCategory::Write))
        );
        assert_eq!(
            CommandCategory::from_acl_str("+@admin"),
            Some((true, CommandCategory::Admin))
        );
        assert_eq!(
            CommandCategory::from_acl_str("+@connection"),
            Some((true, CommandCategory::Connection))
        );
        assert_eq!(CommandCategory::from_acl_str("invalid"), None);
        assert_eq!(CommandCategory::from_acl_str("+@unknown"), None);
    }

    #[test]
    fn test_graph_acl_matches_exact() {
        let acl = GraphAcl {
            pattern: "mydb".into(),
            permission: GraphPermission::Read,
        };
        assert!(acl.matches("mydb"));
        assert!(!acl.matches("other"));
    }

    #[test]
    fn test_graph_acl_matches_wildcard() {
        let acl = GraphAcl {
            pattern: "app:*".into(),
            permission: GraphPermission::ReadWrite,
        };
        assert!(acl.matches("app:users"));
        assert!(acl.matches("app:orders"));
        assert!(!acl.matches("shared"));
    }

    #[test]
    fn test_graph_acl_matches_star() {
        let acl = GraphAcl {
            pattern: "*".into(),
            permission: GraphPermission::Admin,
        };
        assert!(acl.matches("anything"));
        assert!(acl.matches("app:stuff"));
    }

    #[test]
    fn test_graph_permission_ordering() {
        assert!(GraphPermission::None < GraphPermission::Read);
        assert!(GraphPermission::Read < GraphPermission::ReadWrite);
        assert!(GraphPermission::ReadWrite < GraphPermission::Admin);
    }

    #[test]
    fn test_graph_permission_from_str() {
        assert_eq!(GraphPermission::from_str("read"), GraphPermission::Read);
        assert_eq!(GraphPermission::from_str("r"), GraphPermission::Read);
        assert_eq!(GraphPermission::from_str("readwrite"), GraphPermission::ReadWrite);
        assert_eq!(GraphPermission::from_str("rw"), GraphPermission::ReadWrite);
        assert_eq!(GraphPermission::from_str("write"), GraphPermission::ReadWrite);
        assert_eq!(GraphPermission::from_str("admin"), GraphPermission::Admin);
        assert_eq!(GraphPermission::from_str("all"), GraphPermission::Admin);
        assert_eq!(GraphPermission::from_str("bogus"), GraphPermission::None);
    }

    #[test]
    fn test_user_permissions_full_access_graph() {
        let perms = UserPermissions::full_access();
        assert!(perms.can_read_graph("any"));
        assert!(perms.can_write_graph("any"));
        assert!(perms.can_admin_graph("any"));
    }

    #[test]
    fn test_user_permissions_graph_acl_read_only() {
        let perms = UserPermissions {
            allowed_categories: CommandCategorySet::all(),
            graph_acl: vec![GraphAcl {
                pattern: "shared".into(),
                permission: GraphPermission::Read,
            }],
        };
        assert!(perms.can_read_graph("shared"));
        assert!(!perms.can_write_graph("shared"));
        assert!(!perms.can_admin_graph("shared"));
        // No match for other graphs.
        assert!(!perms.can_read_graph("other"));
    }

    #[test]
    fn test_user_permissions_graph_acl_pattern() {
        let perms = UserPermissions {
            allowed_categories: CommandCategorySet::all(),
            graph_acl: vec![
                GraphAcl {
                    pattern: "app:*".into(),
                    permission: GraphPermission::ReadWrite,
                },
                GraphAcl {
                    pattern: "shared".into(),
                    permission: GraphPermission::Read,
                },
            ],
        };
        assert!(perms.can_read_graph("app:users"));
        assert!(perms.can_write_graph("app:users"));
        assert!(!perms.can_admin_graph("app:users"));
        assert!(perms.can_read_graph("shared"));
        assert!(!perms.can_write_graph("shared"));
    }

    #[test]
    fn test_user_permissions_empty_graph_acl_is_full_access() {
        let perms = UserPermissions {
            allowed_categories: CommandCategorySet::empty(),
            graph_acl: Vec::new(),
        };
        assert!(perms.can_read_graph("anything"));
        assert!(perms.can_write_graph("anything"));
        assert!(perms.can_admin_graph("anything"));
    }
}
