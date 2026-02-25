//! Command classification for authorization.
//!
//! Maps command names to categories (Connection, Read, Write, Admin)
//! and extracts graph names from commands for graph-level ACL checks.

use crate::identity::CommandCategory;

/// Classify a command string (the first token, uppercased) into a category.
///
/// Takes the full command name like "PING", "NODE.ADD", "GRAPH.CREATE", etc.
pub fn command_category(cmd_name: &str) -> CommandCategory {
    match cmd_name.to_uppercase().as_str() {
        // Connection commands — always allowed for authenticated users.
        "PING" | "INFO" | "AUTH" => CommandCategory::Connection,

        // Read commands.
        "NODE.GET" | "EDGE.GET" | "GRAPH.INFO" | "GRAPH.LIST" | "STATS"
        | "CONTEXT" | "CONFIG.GET" | "ACL WHOAMI" => CommandCategory::Read,

        // Write commands.
        "NODE.ADD" | "NODE.UPDATE" | "NODE.DELETE"
        | "EDGE.ADD" | "EDGE.DELETE" | "EDGE.INVALIDATE"
        | "BULK.INSERT.NODES" | "BULK.INSERT.EDGES" => CommandCategory::Write,

        // Admin commands.
        "GRAPH.CREATE" | "GRAPH.DROP" | "SNAPSHOT" | "CONFIG.SET"
        | "ACL SETUSER" | "ACL DELUSER" | "ACL LIST" | "ACL GETUSER"
        | "ACL SAVE" | "ACL LOAD" => CommandCategory::Admin,

        // Default: treat unknown as Admin (principle of least privilege).
        _ => CommandCategory::Admin,
    }
}

/// Classify a parsed Command enum variant into its category.
pub fn classify_command(cmd: &str) -> CommandCategory {
    // Normalize the command string: first two tokens form the command key.
    let upper = cmd.to_uppercase();
    let parts: Vec<&str> = upper.split_whitespace().collect();

    match parts.as_slice() {
        ["PING"] | ["INFO"] | ["AUTH", ..] => CommandCategory::Connection,

        ["NODE", "GET", ..] | ["EDGE", "GET", ..] | ["GRAPH", "INFO", ..]
        | ["GRAPH", "LIST", ..] | ["STATS", ..] | ["CONTEXT", ..]
        | ["CONFIG", "GET", ..] | ["ACL", "WHOAMI"] => CommandCategory::Read,

        ["NODE", "ADD", ..] | ["NODE", "UPDATE", ..] | ["NODE", "DELETE", ..]
        | ["EDGE", "ADD", ..] | ["EDGE", "DELETE", ..] | ["EDGE", "INVALIDATE", ..]
        | ["BULK", ..] => CommandCategory::Write,

        ["GRAPH", "CREATE", ..] | ["GRAPH", "DROP", ..]
        | ["SNAPSHOT"] | ["CONFIG", "SET", ..]
        | ["ACL", "SETUSER", ..] | ["ACL", "DELUSER", ..]
        | ["ACL", "LIST"] | ["ACL", "GETUSER", ..]
        | ["ACL", "SAVE"] | ["ACL", "LOAD"] => CommandCategory::Admin,

        _ => CommandCategory::Admin,
    }
}

/// Extract the graph name from a command string, if present.
/// Returns None for commands that don't operate on a specific graph.
pub fn extract_graph_name(cmd: &str) -> Option<String> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let upper: Vec<String> = parts.iter().map(|p| p.to_uppercase()).collect();

    match upper.first().map(|s| s.as_str()) {
        // Commands with graph name directly after the operation.
        Some("GRAPH") => {
            // GRAPH CREATE "name", GRAPH DROP "name", GRAPH INFO "name"
            if upper.len() >= 3 {
                Some(unquote(parts[2]))
            } else {
                None
            }
        }
        Some("CONTEXT") => {
            // CONTEXT "query" FROM "graph" ...
            if let Some(pos) = upper.iter().position(|s| s == "FROM") {
                if pos + 1 < parts.len() {
                    return Some(unquote(parts[pos + 1]));
                }
            }
            None
        }
        Some("NODE") | Some("EDGE") | Some("BULK") => {
            // Look for "TO" keyword to find graph name.
            if let Some(pos) = upper.iter().position(|s| s == "TO") {
                if pos + 1 < parts.len() {
                    return Some(unquote(parts[pos + 1]));
                }
            }
            // NODE GET "graph" <id>
            if upper.len() >= 3
                && (upper[1] == "GET" || upper[1] == "DELETE")
            {
                return Some(unquote(parts[2]));
            }
            None
        }
        _ => None,
    }
}

/// Remove surrounding double quotes from a string.
fn unquote(s: &str) -> String {
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_category_connection() {
        assert_eq!(command_category("PING"), CommandCategory::Connection);
        assert_eq!(command_category("INFO"), CommandCategory::Connection);
        assert_eq!(command_category("AUTH"), CommandCategory::Connection);
    }

    #[test]
    fn test_command_category_read() {
        assert_eq!(command_category("NODE.GET"), CommandCategory::Read);
        assert_eq!(command_category("EDGE.GET"), CommandCategory::Read);
        assert_eq!(command_category("GRAPH.INFO"), CommandCategory::Read);
        assert_eq!(command_category("GRAPH.LIST"), CommandCategory::Read);
        assert_eq!(command_category("STATS"), CommandCategory::Read);
        assert_eq!(command_category("CONTEXT"), CommandCategory::Read);
        assert_eq!(command_category("CONFIG.GET"), CommandCategory::Read);
    }

    #[test]
    fn test_command_category_write() {
        assert_eq!(command_category("NODE.ADD"), CommandCategory::Write);
        assert_eq!(command_category("NODE.UPDATE"), CommandCategory::Write);
        assert_eq!(command_category("NODE.DELETE"), CommandCategory::Write);
        assert_eq!(command_category("EDGE.ADD"), CommandCategory::Write);
        assert_eq!(command_category("EDGE.DELETE"), CommandCategory::Write);
        assert_eq!(command_category("EDGE.INVALIDATE"), CommandCategory::Write);
    }

    #[test]
    fn test_command_category_admin() {
        assert_eq!(command_category("GRAPH.CREATE"), CommandCategory::Admin);
        assert_eq!(command_category("GRAPH.DROP"), CommandCategory::Admin);
        assert_eq!(command_category("SNAPSHOT"), CommandCategory::Admin);
        assert_eq!(command_category("CONFIG.SET"), CommandCategory::Admin);
    }

    #[test]
    fn test_command_category_unknown() {
        assert_eq!(command_category("UNKNOWN"), CommandCategory::Admin);
    }

    #[test]
    fn test_classify_command_ping() {
        assert_eq!(classify_command("PING"), CommandCategory::Connection);
    }

    #[test]
    fn test_classify_command_node_add() {
        assert_eq!(
            classify_command("NODE ADD TO \"g\" LABEL \"l\""),
            CommandCategory::Write
        );
    }

    #[test]
    fn test_classify_command_graph_create() {
        assert_eq!(
            classify_command("GRAPH CREATE \"mydb\""),
            CommandCategory::Admin
        );
    }

    #[test]
    fn test_classify_command_context() {
        assert_eq!(
            classify_command("CONTEXT \"query\" FROM \"g\""),
            CommandCategory::Read
        );
    }

    #[test]
    fn test_classify_command_acl_whoami() {
        assert_eq!(classify_command("ACL WHOAMI"), CommandCategory::Read);
    }

    #[test]
    fn test_classify_command_acl_setuser() {
        assert_eq!(
            classify_command("ACL SETUSER alice >pass on +@read"),
            CommandCategory::Admin
        );
    }

    #[test]
    fn test_extract_graph_name_context() {
        assert_eq!(
            extract_graph_name("CONTEXT \"query\" FROM \"mydb\" BUDGET 4096"),
            Some("mydb".into())
        );
    }

    #[test]
    fn test_extract_graph_name_node_add() {
        assert_eq!(
            extract_graph_name("NODE ADD TO \"mydb\" LABEL \"person\""),
            Some("mydb".into())
        );
    }

    #[test]
    fn test_extract_graph_name_graph_create() {
        assert_eq!(
            extract_graph_name("GRAPH CREATE \"mydb\""),
            Some("mydb".into())
        );
    }

    #[test]
    fn test_extract_graph_name_ping() {
        assert_eq!(extract_graph_name("PING"), None);
    }

    #[test]
    fn test_extract_graph_name_node_get() {
        assert_eq!(
            extract_graph_name("NODE GET \"testdb\" 42"),
            Some("testdb".into())
        );
    }

    #[test]
    fn test_unquote() {
        assert_eq!(unquote("\"hello\""), "hello");
        assert_eq!(unquote("hello"), "hello");
        assert_eq!(unquote("\"\""), "");
        assert_eq!(unquote("\""), "\"");
    }
}
