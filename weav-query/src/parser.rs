//! Command parser for the Weav query language.
//!
//! Parses a simplified command language into a `Command` enum using
//! simple string-based parsing (split/match).

use compact_str::CompactString;

use weav_core::config::GraphConfig;
use weav_core::error::WeavError;
use weav_core::types::{DecayFunction, Direction, Timestamp, TokenBudget, Value};

// ─── Command types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Command {
    /// Context retrieval query.
    Context(ContextQuery),
    /// Add a node.
    NodeAdd(NodeAddCmd),
    /// Get a node.
    NodeGet(NodeGetCmd),
    /// Delete a node.
    NodeDelete(NodeDeleteCmd),
    /// Add an edge.
    EdgeAdd(EdgeAddCmd),
    /// Invalidate an edge (temporal close).
    EdgeInvalidate(EdgeInvalidateCmd),
    /// Create a new graph.
    GraphCreate(GraphCreateCmd),
    /// Drop a graph.
    GraphDrop(String),
    /// Get info about a graph.
    GraphInfo(String),
    /// List all graphs.
    GraphList,
    /// Health-check ping.
    Ping,
    /// Server info.
    Info,
    /// Statistics (optionally for a specific graph).
    Stats(Option<String>),
    /// Update a node's properties/embedding.
    NodeUpdate(NodeUpdateCmd),
    /// Bulk insert nodes.
    BulkInsertNodes(BulkInsertNodesCmd),
    /// Bulk insert edges.
    BulkInsertEdges(BulkInsertEdgesCmd),
    /// Trigger a snapshot.
    Snapshot,
    /// Delete an edge.
    EdgeDelete(EdgeDeleteCmd),
    /// Get an edge.
    EdgeGet(EdgeGetCmd),
    /// Set a config key.
    ConfigSet(String, String),
    /// Get a config key.
    ConfigGet(String),
    /// Authenticate: AUTH [username] password
    Auth {
        username: Option<String>,
        password: String,
    },
    /// ACL SETUSER username [>password] [on|off] [+@cat|-@cat] [~pattern]
    AclSetUser(AclSetUserCmd),
    /// ACL DELUSER username
    AclDelUser(String),
    /// ACL LIST
    AclList,
    /// ACL GETUSER username
    AclGetUser(String),
    /// ACL WHOAMI
    AclWhoAmI,
    /// ACL SAVE
    AclSave,
    /// ACL LOAD
    AclLoad,
    /// Ingest a document (extract entities/relationships and build graph).
    Ingest(IngestCmd),
    /// Search nodes by property key/value.
    Search(SearchCmd),
    /// Get neighbors of a node.
    Neighbors(NeighborsCmd),
}

impl Command {
    /// Returns a short, stable string label for this command variant (for metrics/logging).
    pub fn type_name(&self) -> &'static str {
        match self {
            Command::Context(_) => "context",
            Command::NodeAdd(_) => "node_add",
            Command::NodeGet(_) => "node_get",
            Command::NodeDelete(_) => "node_delete",
            Command::NodeUpdate(_) => "node_update",
            Command::EdgeAdd(_) => "edge_add",
            Command::EdgeInvalidate(_) => "edge_invalidate",
            Command::EdgeDelete(_) => "edge_delete",
            Command::EdgeGet(_) => "edge_get",
            Command::GraphCreate(_) => "graph_create",
            Command::GraphDrop(_) => "graph_drop",
            Command::GraphInfo(_) => "graph_info",
            Command::GraphList => "graph_list",
            Command::BulkInsertNodes(_) => "bulk_insert_nodes",
            Command::BulkInsertEdges(_) => "bulk_insert_edges",
            Command::Ping => "ping",
            Command::Info => "info",
            Command::Stats(_) => "stats",
            Command::Snapshot => "snapshot",
            Command::ConfigSet(_, _) => "config_set",
            Command::ConfigGet(_) => "config_get",
            Command::Auth { .. } => "auth",
            Command::AclSetUser(_) => "acl_set_user",
            Command::AclDelUser(_) => "acl_del_user",
            Command::AclList => "acl_list",
            Command::AclGetUser(_) => "acl_get_user",
            Command::AclWhoAmI => "acl_whoami",
            Command::AclSave => "acl_save",
            Command::AclLoad => "acl_load",
            Command::Ingest(_) => "ingest",
            Command::Search(_) => "search",
            Command::Neighbors(_) => "neighbors",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AclSetUserCmd {
    pub username: String,
    /// Plaintext password to set (from ">password" token).
    pub password: Option<String>,
    /// Whether to enable the user.
    pub enabled: Option<bool>,
    /// Category grants/revocations like "+@read", "-@admin".
    pub categories: Vec<String>,
    /// Graph patterns like "~pattern:permission".
    pub graph_patterns: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct ContextQuery {
    pub query_text: Option<String>,
    pub graph: String,
    pub budget: Option<TokenBudget>,
    pub seeds: SeedStrategy,
    pub max_depth: u8,
    pub direction: Direction,
    pub edge_filter: Option<EdgeFilterConfig>,
    pub decay: Option<DecayFunction>,
    pub include_provenance: bool,
    pub temporal_at: Option<Timestamp>,
    pub limit: Option<u32>,
    pub sort: Option<SortOrder>,
}

#[derive(Debug, Clone)]
pub enum SeedStrategy {
    Vector { embedding: Vec<f32>, top_k: u16 },
    Nodes(Vec<String>),
    Both {
        embedding: Vec<f32>,
        top_k: u16,
        node_keys: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct EdgeFilterConfig {
    pub labels: Option<Vec<String>>,
    pub min_weight: Option<f32>,
    pub min_confidence: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct NodeAddCmd {
    pub graph: String,
    pub label: String,
    pub properties: Vec<(String, Value)>,
    pub embedding: Option<Vec<f32>>,
    pub entity_key: Option<String>,
    /// Time-to-live in milliseconds. If set, the node will be auto-expired after this duration.
    pub ttl_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct NodeGetCmd {
    pub graph: String,
    pub node_id: Option<u64>,
    pub entity_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NodeDeleteCmd {
    pub graph: String,
    pub node_id: u64,
}

#[derive(Debug, Clone)]
pub struct EdgeAddCmd {
    pub graph: String,
    pub source: u64,
    pub target: u64,
    pub label: String,
    pub weight: f32,
    pub properties: Vec<(String, Value)>,
    /// Time-to-live in milliseconds. If set, the edge will be auto-expired after this duration.
    pub ttl_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct EdgeInvalidateCmd {
    pub graph: String,
    pub edge_id: u64,
}

#[derive(Debug, Clone)]
pub struct NodeUpdateCmd {
    pub graph: String,
    pub node_id: u64,
    pub properties: Vec<(String, Value)>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct BulkInsertNodesCmd {
    pub graph: String,
    pub nodes: Vec<NodeAddCmd>,
}

#[derive(Debug, Clone)]
pub struct BulkInsertEdgesCmd {
    pub graph: String,
    pub edges: Vec<EdgeAddCmd>,
}

#[derive(Debug, Clone)]
pub struct GraphCreateCmd {
    pub name: String,
    pub config: Option<GraphConfig>,
}

#[derive(Debug, Clone)]
pub struct EdgeDeleteCmd {
    pub graph: String,
    pub edge_id: u64,
}

#[derive(Debug, Clone)]
pub struct EdgeGetCmd {
    pub graph: String,
    pub edge_id: u64,
}

/// Ingest a document into a graph via the extraction pipeline.
#[derive(Debug, Clone)]
pub struct IngestCmd {
    pub graph: String,
    pub content: String,
    pub format: Option<String>,
    pub document_id: Option<String>,
    pub skip_extraction: bool,
    pub skip_dedup: bool,
    pub chunk_size: Option<usize>,
    pub entity_types: Option<Vec<String>>,
}

/// Search nodes by property key/value.
#[derive(Debug, Clone)]
pub struct SearchCmd {
    pub graph: String,
    pub key: String,
    pub value: String,
    pub limit: Option<u32>,
}

/// Get neighbors of a node.
#[derive(Debug, Clone)]
pub struct NeighborsCmd {
    pub graph: String,
    pub node_id: u64,
    pub label: Option<String>,
    pub direction: Direction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SortField {
    Relevance,
    Recency,
    Confidence,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SortOrder {
    pub field: SortField,
    pub direction: SortDirection,
}

// ─── Tokenizer helpers ──────────────────────────────────────────────────────

/// Tokenize input respecting double-quoted strings.
/// Quoted strings are returned without the surrounding quotes.
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '"' {
            chars.next(); // consume opening quote
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c == '"' {
                    chars.next(); // consume closing quote
                    break;
                }
                if c == '\\' {
                    chars.next();
                    if let Some(&escaped) = chars.peek() {
                        s.push(escaped);
                        chars.next();
                    }
                } else {
                    s.push(c);
                    chars.next();
                }
            }
            tokens.push(format!("\"{}\"", s));
        } else if ch == '{' || ch == '[' {
            // Consume a JSON-like block, tracking bracket depth.
            let open = ch;
            let close = if ch == '{' { '}' } else { ']' };
            let mut depth = 0;
            let mut s = String::new();
            let mut in_string = false;
            while let Some(&c) = chars.peek() {
                s.push(c);
                chars.next();
                if in_string {
                    if c == '\\' {
                        // consume next char as part of escape
                        if let Some(&esc) = chars.peek() {
                            s.push(esc);
                            chars.next();
                        }
                    } else if c == '"' {
                        in_string = false;
                    }
                } else {
                    if c == '"' {
                        in_string = true;
                    } else if c == open {
                        depth += 1;
                    } else if c == close {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
            tokens.push(s);
        } else {
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == '"' || c == '{' || c == '[' {
                    break;
                }
                s.push(c);
                chars.next();
            }
            tokens.push(s);
        }
    }
    tokens
}

/// Strip surrounding double quotes from a token if present.
fn unquote(s: &str) -> &str {
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn parse_err(msg: impl Into<String>) -> WeavError {
    WeavError::QueryParseError(msg.into())
}

/// Parse a JSON-like object string into a list of (key, Value) pairs.
fn parse_properties_json(json_str: &str) -> Result<Vec<(String, Value)>, WeavError> {
    let val: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| parse_err(format!("invalid JSON properties: {e}")))?;
    let obj = val
        .as_object()
        .ok_or_else(|| parse_err("properties must be a JSON object"))?;
    let mut props = Vec::new();
    for (k, v) in obj {
        props.push((k.clone(), json_to_value(v)));
    }
    Ok(props)
}

/// Convert a serde_json::Value to our Value type.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(CompactString::new(s)),
        serde_json::Value::Array(arr) => {
            // Check if it looks like a vector (all numbers)
            if arr.iter().all(|v| v.is_number()) {
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                Value::Vector(floats)
            } else {
                let items: Vec<Value> = arr.iter().map(json_to_value).collect();
                Value::List(items)
            }
        }
        serde_json::Value::Object(map) => {
            let pairs: Vec<(CompactString, Value)> = map
                .iter()
                .map(|(k, v)| (CompactString::new(k), json_to_value(v)))
                .collect();
            Value::Map(pairs)
        }
    }
}

/// Parse a JSON array string into a Vec<f32>.
fn parse_f32_array(s: &str) -> Result<Vec<f32>, WeavError> {
    let val: serde_json::Value =
        serde_json::from_str(s).map_err(|e| parse_err(format!("invalid embedding array: {e}")))?;
    let arr = val
        .as_array()
        .ok_or_else(|| parse_err("embedding must be a JSON array"))?;
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| parse_err("embedding array elements must be numbers"))
        })
        .collect()
}

// ─── Main parser ────────────────────────────────────────────────────────────

/// Parse a command string into a `Command`.
pub fn parse_command(input: &str) -> Result<Command, WeavError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(parse_err("empty command"));
    }

    let tokens = tokenize(input);
    if tokens.is_empty() {
        return Err(parse_err("empty command"));
    }

    let first = tokens[0].to_uppercase();
    match first.as_str() {
        "PING" => Ok(Command::Ping),
        "INFO" => Ok(Command::Info),
        "SNAPSHOT" => Ok(Command::Snapshot),
        "STATS" => {
            if tokens.len() > 1 {
                Ok(Command::Stats(Some(unquote(&tokens[1]).to_string())))
            } else {
                Ok(Command::Stats(None))
            }
        }
        "GRAPH" => parse_graph_command(&tokens),
        "NODE" => parse_node_command(&tokens),
        "EDGE" => parse_edge_command(&tokens),
        "CONTEXT" => parse_context_command(&tokens),
        "BULK" => parse_bulk_command(&tokens),
        "CONFIG" => parse_config_command(&tokens),
        "AUTH" => parse_auth_command(&tokens),
        "ACL" => parse_acl_command(&tokens),
        "INGEST" => parse_ingest_command(&tokens),
        "SEARCH" => parse_search_command(&tokens),
        "NEIGHBORS" => parse_neighbors_command(&tokens),
        _ => Err(parse_err(format!("unknown command: {}", &tokens[0]))),
    }
}

fn parse_graph_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("GRAPH requires a subcommand (CREATE, DROP, INFO, LIST)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "CREATE" => {
            if tokens.len() < 3 {
                return Err(parse_err("GRAPH CREATE requires a name"));
            }
            let name = unquote(&tokens[2]).to_string();
            // Optional config JSON after name
            let config = if tokens.len() > 3 {
                let json_str = &tokens[3];
                let cfg: GraphConfig = serde_json::from_str(json_str)
                    .map_err(|e| parse_err(format!("invalid graph config JSON: {e}")))?;
                Some(cfg)
            } else {
                None
            };
            Ok(Command::GraphCreate(GraphCreateCmd { name, config }))
        }
        "DROP" => {
            if tokens.len() < 3 {
                return Err(parse_err("GRAPH DROP requires a name"));
            }
            Ok(Command::GraphDrop(unquote(&tokens[2]).to_string()))
        }
        "INFO" => {
            if tokens.len() < 3 {
                return Err(parse_err("GRAPH INFO requires a name"));
            }
            Ok(Command::GraphInfo(unquote(&tokens[2]).to_string()))
        }
        "LIST" => Ok(Command::GraphList),
        _ => Err(parse_err(format!("unknown GRAPH subcommand: {}", &tokens[1]))),
    }
}

fn parse_node_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("NODE requires a subcommand (ADD, GET, UPDATE, DELETE)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "ADD" => parse_node_add(tokens),
        "GET" => parse_node_get(tokens),
        "UPDATE" => parse_node_update(tokens),
        "DELETE" => parse_node_delete(tokens),
        _ => Err(parse_err(format!("unknown NODE subcommand: {}", &tokens[1]))),
    }
}

/// Parse: NODE ADD TO "graph" LABEL "label" PROPERTIES {...} [EMBEDDING [...]] [KEY "key"]
fn parse_node_add(tokens: &[String]) -> Result<Command, WeavError> {
    // Find "TO" keyword
    let to_pos = find_keyword(tokens, "TO")
        .ok_or_else(|| parse_err("NODE ADD requires TO \"graph\""))?;
    if to_pos + 1 >= tokens.len() {
        return Err(parse_err("NODE ADD TO requires a graph name"));
    }
    let graph = unquote(&tokens[to_pos + 1]).to_string();

    // Find "LABEL" keyword
    let label_pos = find_keyword(tokens, "LABEL")
        .ok_or_else(|| parse_err("NODE ADD requires LABEL \"label\""))?;
    if label_pos + 1 >= tokens.len() {
        return Err(parse_err("NODE ADD LABEL requires a label value"));
    }
    let label = unquote(&tokens[label_pos + 1]).to_string();

    // Find "PROPERTIES" keyword (optional)
    let properties = if let Some(props_pos) = find_keyword(tokens, "PROPERTIES") {
        if props_pos + 1 >= tokens.len() {
            return Err(parse_err("PROPERTIES requires a JSON object"));
        }
        parse_properties_json(&tokens[props_pos + 1])?
    } else {
        Vec::new()
    };

    // Find "EMBEDDING" keyword (optional)
    let embedding = if let Some(emb_pos) = find_keyword(tokens, "EMBEDDING") {
        if emb_pos + 1 >= tokens.len() {
            return Err(parse_err("EMBEDDING requires a JSON array"));
        }
        Some(parse_f32_array(&tokens[emb_pos + 1])?)
    } else {
        None
    };

    // Find "KEY" keyword (optional)
    let entity_key = if let Some(key_pos) = find_keyword(tokens, "KEY") {
        if key_pos + 1 >= tokens.len() {
            return Err(parse_err("KEY requires a value"));
        }
        Some(unquote(&tokens[key_pos + 1]).to_string())
    } else {
        None
    };

    // Find "TTL" keyword (optional) — value in milliseconds
    let ttl_ms = if let Some(ttl_pos) = find_keyword(tokens, "TTL") {
        if ttl_pos + 1 >= tokens.len() {
            return Err(parse_err("TTL requires a value in milliseconds"));
        }
        Some(tokens[ttl_pos + 1].parse::<u64>().map_err(|_| {
            parse_err("TTL value must be a positive integer (milliseconds)")
        })?)
    } else {
        None
    };

    Ok(Command::NodeAdd(NodeAddCmd {
        graph,
        label,
        properties,
        embedding,
        entity_key,
        ttl_ms,
    }))
}

/// Parse: NODE GET "graph" <id> or NODE GET "graph" WHERE entity_key = "key"
fn parse_node_get(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 3 {
        return Err(parse_err("NODE GET requires a graph name"));
    }
    let graph = unquote(&tokens[2]).to_string();

    if tokens.len() < 4 {
        return Err(parse_err("NODE GET requires an id or WHERE clause"));
    }

    let upper = tokens[3].to_uppercase();
    if upper == "WHERE" {
        // NODE GET "graph" WHERE entity_key = "key"
        if tokens.len() < 7 {
            return Err(parse_err("NODE GET WHERE requires: entity_key = \"key\""));
        }
        let field = &tokens[4];
        if field.to_lowercase() != "entity_key" {
            return Err(parse_err(format!("NODE GET WHERE only supports entity_key, got: {field}")));
        }
        // tokens[5] should be "="
        let key = unquote(&tokens[6]).to_string();
        Ok(Command::NodeGet(NodeGetCmd {
            graph,
            node_id: None,
            entity_key: Some(key),
        }))
    } else {
        // NODE GET "graph" <id>
        let id: u64 = tokens[3]
            .parse()
            .map_err(|_| parse_err(format!("invalid node id: {}", &tokens[3])))?;
        Ok(Command::NodeGet(NodeGetCmd {
            graph,
            node_id: Some(id),
            entity_key: None,
        }))
    }
}

/// Parse: NODE DELETE "graph" <id>
fn parse_node_delete(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 4 {
        return Err(parse_err("NODE DELETE requires a graph name and node id"));
    }
    let graph = unquote(&tokens[2]).to_string();
    let node_id: u64 = tokens[3]
        .parse()
        .map_err(|_| parse_err(format!("invalid node id: {}", &tokens[3])))?;
    Ok(Command::NodeDelete(NodeDeleteCmd { graph, node_id }))
}

fn parse_edge_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("EDGE requires a subcommand (ADD, GET, DELETE, INVALIDATE)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "ADD" => parse_edge_add(tokens),
        "INVALIDATE" => parse_edge_invalidate(tokens),
        "DELETE" => parse_edge_delete(tokens),
        "GET" => parse_edge_get(tokens),
        _ => Err(parse_err(format!("unknown EDGE subcommand: {}", &tokens[1]))),
    }
}

/// Parse: EDGE ADD TO "graph" FROM <src> TO <tgt> LABEL "label" [WEIGHT <f>] [PROPERTIES {...}]
fn parse_edge_add(tokens: &[String]) -> Result<Command, WeavError> {
    // Find first "TO" after EDGE ADD (the graph name)
    let to_graph_pos = find_keyword_after(tokens, "TO", 2)
        .ok_or_else(|| parse_err("EDGE ADD requires TO \"graph\""))?;
    if to_graph_pos + 1 >= tokens.len() {
        return Err(parse_err("EDGE ADD TO requires a graph name"));
    }
    let graph = unquote(&tokens[to_graph_pos + 1]).to_string();

    // Find FROM <src>
    let from_pos = find_keyword(tokens, "FROM")
        .ok_or_else(|| parse_err("EDGE ADD requires FROM <source_id>"))?;
    if from_pos + 1 >= tokens.len() {
        return Err(parse_err("EDGE ADD FROM requires a source id"));
    }
    let source: u64 = tokens[from_pos + 1]
        .parse()
        .map_err(|_| parse_err(format!("invalid source id: {}", &tokens[from_pos + 1])))?;

    // Find second "TO" after FROM (the target node)
    let to_tgt_pos = find_keyword_after(tokens, "TO", from_pos + 1)
        .ok_or_else(|| parse_err("EDGE ADD requires TO <target_id> after FROM"))?;
    if to_tgt_pos + 1 >= tokens.len() {
        return Err(parse_err("EDGE ADD TO requires a target id"));
    }
    let target: u64 = tokens[to_tgt_pos + 1]
        .parse()
        .map_err(|_| parse_err(format!("invalid target id: {}", &tokens[to_tgt_pos + 1])))?;

    // Find LABEL
    let label_pos = find_keyword(tokens, "LABEL")
        .ok_or_else(|| parse_err("EDGE ADD requires LABEL \"label\""))?;
    if label_pos + 1 >= tokens.len() {
        return Err(parse_err("EDGE ADD LABEL requires a label value"));
    }
    let label = unquote(&tokens[label_pos + 1]).to_string();

    // Find WEIGHT (optional, default 1.0)
    let weight = if let Some(w_pos) = find_keyword(tokens, "WEIGHT") {
        if w_pos + 1 >= tokens.len() {
            return Err(parse_err("WEIGHT requires a float value"));
        }
        tokens[w_pos + 1]
            .parse::<f32>()
            .map_err(|_| parse_err(format!("invalid weight: {}", &tokens[w_pos + 1])))?
    } else {
        1.0
    };

    // Find PROPERTIES (optional)
    let properties = if let Some(props_pos) = find_keyword(tokens, "PROPERTIES") {
        if props_pos + 1 >= tokens.len() {
            return Err(parse_err("PROPERTIES requires a JSON object"));
        }
        parse_properties_json(&tokens[props_pos + 1])?
    } else {
        Vec::new()
    };

    // Find TTL (optional) — value in milliseconds
    let ttl_ms = if let Some(ttl_pos) = find_keyword(tokens, "TTL") {
        if ttl_pos + 1 >= tokens.len() {
            return Err(parse_err("TTL requires a value in milliseconds"));
        }
        Some(tokens[ttl_pos + 1].parse::<u64>().map_err(|_| {
            parse_err("TTL value must be a positive integer (milliseconds)")
        })?)
    } else {
        None
    };

    Ok(Command::EdgeAdd(EdgeAddCmd {
        graph,
        source,
        target,
        label,
        weight,
        properties,
        ttl_ms,
    }))
}

/// Parse: EDGE INVALIDATE "graph" <edge_id>
fn parse_edge_invalidate(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 4 {
        return Err(parse_err(
            "EDGE INVALIDATE requires a graph name and edge id",
        ));
    }
    let graph = unquote(&tokens[2]).to_string();
    let edge_id: u64 = tokens[3]
        .parse()
        .map_err(|_| parse_err(format!("invalid edge id: {}", &tokens[3])))?;
    Ok(Command::EdgeInvalidate(EdgeInvalidateCmd { graph, edge_id }))
}

/// Parse: CONTEXT "query" FROM "graph" [BUDGET <n> TOKENS] [SEEDS VECTOR [...] TOP <k>]
///        [SEEDS NODES [...]] [DEPTH <d>] [DIRECTION OUT|IN|BOTH]
///        [FILTER LABELS [...] MIN_WEIGHT <f> MIN_CONFIDENCE <f>]
///        [DECAY EXPONENTIAL <ms> | LINEAR <ms> | STEP <ms> | NONE]
///        [PROVENANCE] [AT <timestamp>] [LIMIT <n>]
fn parse_context_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("CONTEXT requires a query text"));
    }

    let query_text = Some(unquote(&tokens[1]).to_string());

    // Find FROM "graph"
    let from_pos = find_keyword(tokens, "FROM")
        .ok_or_else(|| parse_err("CONTEXT requires FROM \"graph\""))?;
    if from_pos + 1 >= tokens.len() {
        return Err(parse_err("CONTEXT FROM requires a graph name"));
    }
    let graph = unquote(&tokens[from_pos + 1]).to_string();

    // Parse BUDGET <n> TOKENS
    let budget = if let Some(b_pos) = find_keyword(tokens, "BUDGET") {
        if b_pos + 1 >= tokens.len() {
            return Err(parse_err("BUDGET requires a token count"));
        }
        let max_tokens: u32 = tokens[b_pos + 1]
            .parse()
            .map_err(|_| parse_err(format!("invalid budget: {}", &tokens[b_pos + 1])))?;
        // Skip optional "TOKENS" keyword
        Some(TokenBudget::new(max_tokens))
    } else {
        None
    };

    // Parse SEEDS — can be VECTOR, NODES, or both
    let mut embedding: Option<Vec<f32>> = None;
    let mut top_k: u16 = 10;
    let mut node_keys: Vec<String> = Vec::new();

    // Find all SEEDS keywords
    let mut idx = 0;
    while idx < tokens.len() {
        if tokens[idx].to_uppercase() == "SEEDS" && idx + 1 < tokens.len() {
            let seed_type = tokens[idx + 1].to_uppercase();
            match seed_type.as_str() {
                "VECTOR" => {
                    if idx + 2 < tokens.len() {
                        embedding = Some(parse_f32_array(&tokens[idx + 2])?);
                        // Check for TOP <k>
                        if idx + 4 < tokens.len() && tokens[idx + 3].to_uppercase() == "TOP" {
                            top_k = tokens[idx + 4]
                                .parse()
                                .map_err(|_| parse_err(format!("invalid top_k: {}", &tokens[idx + 4])))?;
                        }
                    }
                }
                "NODES" => {
                    if idx + 2 < tokens.len() {
                        // Parse node keys as JSON array of strings
                        let arr_str = &tokens[idx + 2];
                        let val: serde_json::Value = serde_json::from_str(arr_str)
                            .map_err(|e| parse_err(format!("invalid nodes array: {e}")))?;
                        if let Some(arr) = val.as_array() {
                            for v in arr {
                                if let Some(s) = v.as_str() {
                                    node_keys.push(s.to_string());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        idx += 1;
    }

    let seeds = if embedding.is_some() && !node_keys.is_empty() {
        SeedStrategy::Both {
            embedding: embedding.unwrap(),
            top_k,
            node_keys,
        }
    } else if let Some(emb) = embedding {
        SeedStrategy::Vector {
            embedding: emb,
            top_k,
        }
    } else if !node_keys.is_empty() {
        SeedStrategy::Nodes(node_keys)
    } else {
        // Default: vector search from query text embedding (query_text is required)
        SeedStrategy::Nodes(Vec::new())
    };

    // Parse DEPTH <d>
    let max_depth = if let Some(d_pos) = find_keyword(tokens, "DEPTH") {
        if d_pos + 1 >= tokens.len() {
            return Err(parse_err("DEPTH requires a value"));
        }
        tokens[d_pos + 1]
            .parse()
            .map_err(|_| parse_err(format!("invalid depth: {}", &tokens[d_pos + 1])))?
    } else {
        2
    };

    // Parse DIRECTION
    let direction = if let Some(dir_pos) = find_keyword(tokens, "DIRECTION") {
        if dir_pos + 1 >= tokens.len() {
            return Err(parse_err("DIRECTION requires OUT, IN, or BOTH"));
        }
        match tokens[dir_pos + 1].to_uppercase().as_str() {
            "OUT" | "OUTGOING" => Direction::Outgoing,
            "IN" | "INCOMING" => Direction::Incoming,
            "BOTH" => Direction::Both,
            other => return Err(parse_err(format!("invalid direction: {other}"))),
        }
    } else {
        Direction::Both
    };

    // Parse FILTER LABELS [...] MIN_WEIGHT <f> MIN_CONFIDENCE <f>
    let edge_filter = if let Some(f_pos) = find_keyword(tokens, "FILTER") {
        let mut labels = None;
        let mut min_weight = None;
        let mut min_confidence = None;

        let mut fi = f_pos + 1;
        while fi < tokens.len() {
            let upper = tokens[fi].to_uppercase();
            match upper.as_str() {
                "LABELS" => {
                    if fi + 1 < tokens.len() {
                        let arr_str = &tokens[fi + 1];
                        let val: serde_json::Value = serde_json::from_str(arr_str).unwrap_or(serde_json::Value::Null);
                        if let Some(arr) = val.as_array() {
                            let mut lbls = Vec::new();
                            for v in arr {
                                if let Some(s) = v.as_str() {
                                    lbls.push(s.to_string());
                                }
                            }
                            labels = Some(lbls);
                        }
                        fi += 1;
                    }
                }
                "MIN_WEIGHT" => {
                    if fi + 1 < tokens.len() {
                        min_weight = tokens[fi + 1].parse().ok();
                        fi += 1;
                    }
                }
                "MIN_CONFIDENCE" => {
                    if fi + 1 < tokens.len() {
                        min_confidence = tokens[fi + 1].parse().ok();
                        fi += 1;
                    }
                }
                _ => break,
            }
            fi += 1;
        }

        Some(EdgeFilterConfig {
            labels,
            min_weight,
            min_confidence,
        })
    } else {
        None
    };

    // Parse DECAY
    let decay = if let Some(d_pos) = find_keyword(tokens, "DECAY") {
        if d_pos + 1 >= tokens.len() {
            return Err(parse_err("DECAY requires a type"));
        }
        let dtype = tokens[d_pos + 1].to_uppercase();
        match dtype.as_str() {
            "NONE" => Some(DecayFunction::None),
            "EXPONENTIAL" => {
                if d_pos + 2 >= tokens.len() {
                    return Err(parse_err("DECAY EXPONENTIAL requires half_life_ms"));
                }
                let ms: u64 = tokens[d_pos + 2]
                    .parse()
                    .map_err(|_| parse_err("invalid half_life_ms"))?;
                Some(DecayFunction::Exponential { half_life_ms: ms })
            }
            "LINEAR" => {
                if d_pos + 2 >= tokens.len() {
                    return Err(parse_err("DECAY LINEAR requires max_age_ms"));
                }
                let ms: u64 = tokens[d_pos + 2]
                    .parse()
                    .map_err(|_| parse_err("invalid max_age_ms"))?;
                Some(DecayFunction::Linear { max_age_ms: ms })
            }
            "STEP" => {
                if d_pos + 2 >= tokens.len() {
                    return Err(parse_err("DECAY STEP requires cutoff_ms"));
                }
                let ms: u64 = tokens[d_pos + 2]
                    .parse()
                    .map_err(|_| parse_err("invalid cutoff_ms"))?;
                Some(DecayFunction::Step { cutoff_ms: ms })
            }
            other => return Err(parse_err(format!("unknown decay type: {other}"))),
        }
    } else {
        None
    };

    // Parse PROVENANCE flag
    let include_provenance = find_keyword(tokens, "PROVENANCE").is_some();

    // Parse AT <timestamp>
    let temporal_at = if let Some(at_pos) = find_keyword(tokens, "AT") {
        if at_pos + 1 >= tokens.len() {
            return Err(parse_err("AT requires a timestamp"));
        }
        let ts: Timestamp = tokens[at_pos + 1]
            .parse()
            .map_err(|_| parse_err(format!("invalid timestamp: {}", &tokens[at_pos + 1])))?;
        Some(ts)
    } else {
        None
    };

    // Parse LIMIT <n>
    let limit = if let Some(l_pos) = find_keyword(tokens, "LIMIT") {
        if l_pos + 1 >= tokens.len() {
            return Err(parse_err("LIMIT requires a value"));
        }
        let n: u32 = tokens[l_pos + 1]
            .parse()
            .map_err(|_| parse_err(format!("invalid limit: {}", &tokens[l_pos + 1])))?;
        Some(n)
    } else {
        None
    };

    // Parse SCORE BY <field> <ASC|DESC>
    let sort = if let Some(score_pos) = find_keyword(tokens, "SCORE") {
        if score_pos + 3 < tokens.len() && tokens[score_pos + 1].to_uppercase() == "BY" {
            let field = match tokens[score_pos + 2].to_uppercase().as_str() {
                "RELEVANCE" => SortField::Relevance,
                "RECENCY" => SortField::Recency,
                "CONFIDENCE" => SortField::Confidence,
                other => return Err(parse_err(format!("unknown sort field: {other}"))),
            };
            let direction = match tokens[score_pos + 3].to_uppercase().as_str() {
                "ASC" => SortDirection::Asc,
                "DESC" => SortDirection::Desc,
                other => return Err(parse_err(format!("unknown sort direction: {other}"))),
            };
            Some(SortOrder { field, direction })
        } else {
            Some(SortOrder {
                field: SortField::Relevance,
                direction: SortDirection::Desc,
            })
        }
    } else {
        None
    };

    Ok(Command::Context(ContextQuery {
        query_text,
        graph,
        budget,
        seeds,
        max_depth,
        direction,
        edge_filter,
        decay,
        include_provenance,
        temporal_at,
        limit,
        sort,
    }))
}

/// Parse: NODE UPDATE "graph" <id> PROPERTIES {...} [EMBEDDING [...]]
fn parse_node_update(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 4 {
        return Err(parse_err("NODE UPDATE requires a graph name and node id"));
    }
    let graph = unquote(&tokens[2]).to_string();
    let node_id: u64 = tokens[3]
        .parse()
        .map_err(|_| parse_err(format!("invalid node id: {}", &tokens[3])))?;

    // Find "PROPERTIES" keyword (optional)
    let properties = if let Some(props_pos) = find_keyword(tokens, "PROPERTIES") {
        if props_pos + 1 >= tokens.len() {
            return Err(parse_err("PROPERTIES requires a JSON object"));
        }
        parse_properties_json(&tokens[props_pos + 1])?
    } else {
        Vec::new()
    };

    // Find "EMBEDDING" keyword (optional)
    let embedding = if let Some(emb_pos) = find_keyword(tokens, "EMBEDDING") {
        if emb_pos + 1 >= tokens.len() {
            return Err(parse_err("EMBEDDING requires a JSON array"));
        }
        Some(parse_f32_array(&tokens[emb_pos + 1])?)
    } else {
        None
    };

    Ok(Command::NodeUpdate(NodeUpdateCmd {
        graph,
        node_id,
        properties,
        embedding,
    }))
}

/// Parse: BULK NODES TO "graph" DATA [<json_array>]
///        BULK EDGES TO "graph" DATA [<json_array>]
fn parse_bulk_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("BULK requires a subcommand (NODES, EDGES)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "NODES" => parse_bulk_nodes(tokens),
        "EDGES" => parse_bulk_edges(tokens),
        _ => Err(parse_err(format!("unknown BULK subcommand: {}", &tokens[1]))),
    }
}

/// Parse: BULK NODES TO "graph" DATA [<json_array>]
fn parse_bulk_nodes(tokens: &[String]) -> Result<Command, WeavError> {
    let to_pos = find_keyword(tokens, "TO")
        .ok_or_else(|| parse_err("BULK NODES requires TO \"graph\""))?;
    if to_pos + 1 >= tokens.len() {
        return Err(parse_err("BULK NODES TO requires a graph name"));
    }
    let graph = unquote(&tokens[to_pos + 1]).to_string();

    let data_pos = find_keyword(tokens, "DATA")
        .ok_or_else(|| parse_err("BULK NODES requires DATA [...]"))?;
    if data_pos + 1 >= tokens.len() {
        return Err(parse_err("BULK NODES DATA requires a JSON array"));
    }

    let val: serde_json::Value = serde_json::from_str(&tokens[data_pos + 1])
        .map_err(|e| parse_err(format!("invalid bulk nodes JSON: {e}")))?;
    let arr = val
        .as_array()
        .ok_or_else(|| parse_err("BULK NODES DATA must be a JSON array"))?;

    let mut nodes = Vec::new();
    for item in arr {
        let obj = item
            .as_object()
            .ok_or_else(|| parse_err("each bulk node must be a JSON object"))?;
        let label = obj
            .get("label")
            .and_then(|v| v.as_str())
            .ok_or_else(|| parse_err("each bulk node requires a \"label\" field"))?
            .to_string();
        let properties = if let Some(props) = obj.get("properties") {
            if let Some(pobj) = props.as_object() {
                pobj.iter()
                    .map(|(k, v)| (k.clone(), json_to_value(v)))
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let embedding = obj.get("embedding").and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
        });
        let entity_key = obj
            .get("entity_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        nodes.push(NodeAddCmd {
            graph: graph.clone(),
            label,
            properties,
            embedding,
            entity_key,
            ttl_ms: None,
        });
    }

    Ok(Command::BulkInsertNodes(BulkInsertNodesCmd { graph, nodes }))
}

/// Parse: BULK EDGES TO "graph" DATA [<json_array>]
fn parse_bulk_edges(tokens: &[String]) -> Result<Command, WeavError> {
    let to_pos = find_keyword(tokens, "TO")
        .ok_or_else(|| parse_err("BULK EDGES requires TO \"graph\""))?;
    if to_pos + 1 >= tokens.len() {
        return Err(parse_err("BULK EDGES TO requires a graph name"));
    }
    let graph = unquote(&tokens[to_pos + 1]).to_string();

    let data_pos = find_keyword(tokens, "DATA")
        .ok_or_else(|| parse_err("BULK EDGES requires DATA [...]"))?;
    if data_pos + 1 >= tokens.len() {
        return Err(parse_err("BULK EDGES DATA requires a JSON array"));
    }

    let val: serde_json::Value = serde_json::from_str(&tokens[data_pos + 1])
        .map_err(|e| parse_err(format!("invalid bulk edges JSON: {e}")))?;
    let arr = val
        .as_array()
        .ok_or_else(|| parse_err("BULK EDGES DATA must be a JSON array"))?;

    let mut edges = Vec::new();
    for item in arr {
        let obj = item
            .as_object()
            .ok_or_else(|| parse_err("each bulk edge must be a JSON object"))?;
        let source = obj
            .get("source")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| parse_err("each bulk edge requires a \"source\" field"))?;
        let target = obj
            .get("target")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| parse_err("each bulk edge requires a \"target\" field"))?;
        let label = obj
            .get("label")
            .and_then(|v| v.as_str())
            .ok_or_else(|| parse_err("each bulk edge requires a \"label\" field"))?
            .to_string();
        let weight = obj
            .get("weight")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);
        let properties = if let Some(props) = obj.get("properties") {
            if let Some(pobj) = props.as_object() {
                pobj.iter()
                    .map(|(k, v)| (k.clone(), json_to_value(v)))
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        edges.push(EdgeAddCmd {
            graph: graph.clone(),
            source,
            target,
            label,
            weight,
            properties,
            ttl_ms: None,
        });
    }

    Ok(Command::BulkInsertEdges(BulkInsertEdgesCmd { graph, edges }))
}

/// Parse: EDGE DELETE "graph" <edge_id>
fn parse_edge_delete(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 4 {
        return Err(parse_err("EDGE DELETE requires a graph name and edge id"));
    }
    let graph = unquote(&tokens[2]).to_string();
    let edge_id: u64 = tokens[3]
        .parse()
        .map_err(|_| parse_err(format!("invalid edge id: {}", &tokens[3])))?;
    Ok(Command::EdgeDelete(EdgeDeleteCmd { graph, edge_id }))
}

/// Parse: EDGE GET "graph" <edge_id>
fn parse_edge_get(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 4 {
        return Err(parse_err("EDGE GET requires a graph name and edge id"));
    }
    let graph = unquote(&tokens[2]).to_string();
    let edge_id: u64 = tokens[3]
        .parse()
        .map_err(|_| parse_err(format!("invalid edge id: {}", &tokens[3])))?;
    Ok(Command::EdgeGet(EdgeGetCmd { graph, edge_id }))
}

/// Parse: CONFIG SET <key> <value> or CONFIG GET <key>
fn parse_config_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("CONFIG requires a subcommand (SET, GET)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "SET" => {
            if tokens.len() < 4 {
                return Err(parse_err("CONFIG SET requires <key> <value>"));
            }
            let key = unquote(&tokens[2]).to_string();
            let value = unquote(&tokens[3]).to_string();
            Ok(Command::ConfigSet(key, value))
        }
        "GET" => {
            if tokens.len() < 3 {
                return Err(parse_err("CONFIG GET requires <key>"));
            }
            let key = unquote(&tokens[2]).to_string();
            Ok(Command::ConfigGet(key))
        }
        _ => Err(parse_err(format!("unknown CONFIG subcommand: {}", &tokens[1]))),
    }
}

// ─── AUTH / ACL commands ─────────────────────────────────────────────────────

fn parse_auth_command(tokens: &[String]) -> Result<Command, WeavError> {
    // AUTH [username] password
    match tokens.len() {
        2 => {
            // AUTH <password> — Redis-compat single-password form.
            Ok(Command::Auth {
                username: None,
                password: unquote(&tokens[1]).to_string(),
            })
        }
        3 => {
            // AUTH <username> <password>
            Ok(Command::Auth {
                username: Some(unquote(&tokens[1]).to_string()),
                password: unquote(&tokens[2]).to_string(),
            })
        }
        _ => Err(parse_err("AUTH requires 1 or 2 arguments: AUTH [username] password")),
    }
}

fn parse_acl_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 2 {
        return Err(parse_err("ACL requires a subcommand (SETUSER, DELUSER, LIST, GETUSER, WHOAMI, SAVE, LOAD)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "SETUSER" => {
            if tokens.len() < 3 {
                return Err(parse_err("ACL SETUSER requires a username"));
            }
            let username = unquote(&tokens[2]).to_string();
            let mut password = None;
            let mut enabled = None;
            let mut categories = Vec::new();
            let mut graph_patterns = Vec::new();

            for token in &tokens[3..] {
                let t = token.as_str();
                if let Some(pw) = t.strip_prefix('>') {
                    password = Some(pw.to_string());
                } else if t.eq_ignore_ascii_case("on") {
                    enabled = Some(true);
                } else if t.eq_ignore_ascii_case("off") {
                    enabled = Some(false);
                } else if t.starts_with('+') || t.starts_with('-') {
                    categories.push(t.to_string());
                } else if let Some(pat) = t.strip_prefix('~') {
                    // ~pattern:permission
                    if let Some((p, perm)) = pat.rsplit_once(':') {
                        graph_patterns.push((p.to_string(), perm.to_string()));
                    }
                }
            }

            Ok(Command::AclSetUser(AclSetUserCmd {
                username,
                password,
                enabled,
                categories,
                graph_patterns,
            }))
        }
        "DELUSER" => {
            if tokens.len() < 3 {
                return Err(parse_err("ACL DELUSER requires a username"));
            }
            Ok(Command::AclDelUser(unquote(&tokens[2]).to_string()))
        }
        "LIST" => Ok(Command::AclList),
        "GETUSER" => {
            if tokens.len() < 3 {
                return Err(parse_err("ACL GETUSER requires a username"));
            }
            Ok(Command::AclGetUser(unquote(&tokens[2]).to_string()))
        }
        "WHOAMI" => Ok(Command::AclWhoAmI),
        "SAVE" => Ok(Command::AclSave),
        "LOAD" => Ok(Command::AclLoad),
        _ => Err(parse_err(format!("unknown ACL subcommand: {}", &tokens[1]))),
    }
}

/// Parse: INGEST "graph" "content" [FORMAT "pdf"|"text"|"docx"|"csv"]
///        [DOCID "id"] [SKIP_EXTRACTION] [SKIP_DEDUP] [CHUNK_SIZE 512]
///        [ENTITY_TYPES "Person,Organization"]
fn parse_ingest_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 3 {
        return Err(parse_err(
            "INGEST requires at least a graph name and content",
        ));
    }
    let graph = unquote(&tokens[1]).to_string();
    let content = unquote(&tokens[2]).to_string();

    let mut format = None;
    let mut document_id = None;
    let mut skip_extraction = false;
    let mut skip_dedup = false;
    let mut chunk_size = None;
    let mut entity_types = None;

    let mut i = 3;
    while i < tokens.len() {
        let upper = tokens[i].to_uppercase();
        match upper.as_str() {
            "FORMAT" => {
                i += 1;
                if i < tokens.len() {
                    format = Some(unquote(&tokens[i]).to_string());
                }
            }
            "DOCID" => {
                i += 1;
                if i < tokens.len() {
                    document_id = Some(unquote(&tokens[i]).to_string());
                }
            }
            "SKIP_EXTRACTION" => {
                skip_extraction = true;
            }
            "SKIP_DEDUP" => {
                skip_dedup = true;
            }
            "CHUNK_SIZE" => {
                i += 1;
                if i < tokens.len() {
                    chunk_size = Some(tokens[i].parse::<usize>().map_err(|_| {
                        parse_err("CHUNK_SIZE value must be a number")
                    })?);
                }
            }
            "ENTITY_TYPES" => {
                i += 1;
                if i < tokens.len() {
                    let types_str = unquote(&tokens[i]);
                    entity_types = Some(
                        types_str
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                }
            }
            _ => {}
        }
        i += 1;
    }

    Ok(Command::Ingest(IngestCmd {
        graph,
        content,
        format,
        document_id,
        skip_extraction,
        skip_dedup,
        chunk_size,
        entity_types,
    }))
}

/// Find a keyword (case-insensitive) in the token list, starting from index 0.
fn find_keyword(tokens: &[String], keyword: &str) -> Option<usize> {
    find_keyword_after(tokens, keyword, 0)
}

/// Find a keyword (case-insensitive) in the token list, starting from a given index.
fn find_keyword_after(tokens: &[String], keyword: &str, start: usize) -> Option<usize> {
    let upper = keyword.to_uppercase();
    for (i, tok) in tokens.iter().enumerate() {
        if i >= start && tok.to_uppercase() == upper {
            return Some(i);
        }
    }
    None
}

/// Parse: SEARCH "graph" WHERE key = "value" [LIMIT n]
fn parse_search_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 6 {
        return Err(parse_err("SEARCH requires: SEARCH \"graph\" WHERE key = \"value\" [LIMIT n]"));
    }
    let graph = unquote(&tokens[1]).to_string();

    // Find WHERE keyword
    let where_pos = find_keyword(tokens, "WHERE")
        .ok_or_else(|| parse_err("SEARCH requires WHERE clause"))?;
    if where_pos + 3 >= tokens.len() {
        return Err(parse_err("WHERE requires: key = \"value\""));
    }
    let key = tokens[where_pos + 1].clone();
    // tokens[where_pos + 2] should be "="
    let value = unquote(&tokens[where_pos + 3]).to_string();

    let limit = if let Some(lim_pos) = find_keyword(tokens, "LIMIT") {
        if lim_pos + 1 >= tokens.len() {
            return Err(parse_err("LIMIT requires a number"));
        }
        Some(tokens[lim_pos + 1].parse::<u32>().map_err(|_| {
            parse_err("LIMIT value must be a positive integer")
        })?)
    } else {
        None
    };

    Ok(Command::Search(SearchCmd {
        graph,
        key,
        value,
        limit,
    }))
}

/// Parse: NEIGHBORS "graph" <node_id> [LABEL "label"] [DIRECTION OUT|IN|BOTH]
fn parse_neighbors_command(tokens: &[String]) -> Result<Command, WeavError> {
    if tokens.len() < 3 {
        return Err(parse_err("NEIGHBORS requires: NEIGHBORS \"graph\" <node_id>"));
    }
    let graph = unquote(&tokens[1]).to_string();
    let node_id: u64 = tokens[2].parse()
        .map_err(|_| parse_err(format!("invalid node_id: {}", &tokens[2])))?;

    let label = if let Some(lbl_pos) = find_keyword(tokens, "LABEL") {
        if lbl_pos + 1 >= tokens.len() {
            return Err(parse_err("LABEL requires a value"));
        }
        Some(unquote(&tokens[lbl_pos + 1]).to_string())
    } else {
        None
    };

    let direction = if let Some(dir_pos) = find_keyword(tokens, "DIRECTION") {
        if dir_pos + 1 >= tokens.len() {
            return Err(parse_err("DIRECTION requires OUT, IN, or BOTH"));
        }
        match tokens[dir_pos + 1].to_uppercase().as_str() {
            "OUT" => Direction::Outgoing,
            "IN" => Direction::Incoming,
            "BOTH" => Direction::Both,
            _ => return Err(parse_err("DIRECTION must be OUT, IN, or BOTH")),
        }
    } else {
        Direction::Both
    };

    Ok(Command::Neighbors(NeighborsCmd {
        graph,
        node_id,
        label,
        direction,
    }))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ping() {
        let cmd = parse_command("PING").unwrap();
        assert!(matches!(cmd, Command::Ping));
    }

    #[test]
    fn test_parse_info() {
        let cmd = parse_command("INFO").unwrap();
        assert!(matches!(cmd, Command::Info));
    }

    #[test]
    fn test_parse_snapshot() {
        let cmd = parse_command("SNAPSHOT").unwrap();
        assert!(matches!(cmd, Command::Snapshot));
    }

    #[test]
    fn test_parse_stats_no_graph() {
        let cmd = parse_command("STATS").unwrap();
        match cmd {
            Command::Stats(None) => {}
            _ => panic!("expected Stats(None)"),
        }
    }

    #[test]
    fn test_parse_stats_with_graph() {
        let cmd = parse_command("STATS \"my_graph\"").unwrap();
        match cmd {
            Command::Stats(Some(name)) => assert_eq!(name, "my_graph"),
            _ => panic!("expected Stats(Some)"),
        }
    }

    #[test]
    fn test_parse_graph_create() {
        let cmd = parse_command("GRAPH CREATE \"my_graph\"").unwrap();
        match cmd {
            Command::GraphCreate(gc) => {
                assert_eq!(gc.name, "my_graph");
                assert!(gc.config.is_none());
            }
            _ => panic!("expected GraphCreate"),
        }
    }

    #[test]
    fn test_parse_graph_drop() {
        let cmd = parse_command("GRAPH DROP \"my_graph\"").unwrap();
        match cmd {
            Command::GraphDrop(name) => assert_eq!(name, "my_graph"),
            _ => panic!("expected GraphDrop"),
        }
    }

    #[test]
    fn test_parse_graph_info() {
        let cmd = parse_command("GRAPH INFO \"my_graph\"").unwrap();
        match cmd {
            Command::GraphInfo(name) => assert_eq!(name, "my_graph"),
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_parse_graph_list() {
        let cmd = parse_command("GRAPH LIST").unwrap();
        assert!(matches!(cmd, Command::GraphList));
    }

    #[test]
    fn test_parse_node_add() {
        let cmd = parse_command(
            r#"NODE ADD TO "test_graph" LABEL "person" PROPERTIES {"name": "Alice", "age": 30} KEY "alice-001""#,
        )
        .unwrap();
        match cmd {
            Command::NodeAdd(na) => {
                assert_eq!(na.graph, "test_graph");
                assert_eq!(na.label, "person");
                assert_eq!(na.properties.len(), 2);
                assert_eq!(na.entity_key, Some("alice-001".to_string()));
            }
            _ => panic!("expected NodeAdd"),
        }
    }

    #[test]
    fn test_parse_node_get_by_id() {
        let cmd = parse_command("NODE GET \"test_graph\" 42").unwrap();
        match cmd {
            Command::NodeGet(ng) => {
                assert_eq!(ng.graph, "test_graph");
                assert_eq!(ng.node_id, Some(42));
                assert!(ng.entity_key.is_none());
            }
            _ => panic!("expected NodeGet"),
        }
    }

    #[test]
    fn test_parse_node_get_by_key() {
        let cmd =
            parse_command("NODE GET \"test_graph\" WHERE entity_key = \"alice-001\"").unwrap();
        match cmd {
            Command::NodeGet(ng) => {
                assert_eq!(ng.graph, "test_graph");
                assert!(ng.node_id.is_none());
                assert_eq!(ng.entity_key, Some("alice-001".to_string()));
            }
            _ => panic!("expected NodeGet"),
        }
    }

    #[test]
    fn test_parse_node_delete() {
        let cmd = parse_command("NODE DELETE \"test_graph\" 42").unwrap();
        match cmd {
            Command::NodeDelete(nd) => {
                assert_eq!(nd.graph, "test_graph");
                assert_eq!(nd.node_id, 42);
            }
            _ => panic!("expected NodeDelete"),
        }
    }

    #[test]
    fn test_parse_edge_add() {
        let cmd = parse_command(
            r#"EDGE ADD TO "test_graph" FROM 1 TO 2 LABEL "knows" WEIGHT 0.9"#,
        )
        .unwrap();
        match cmd {
            Command::EdgeAdd(ea) => {
                assert_eq!(ea.graph, "test_graph");
                assert_eq!(ea.source, 1);
                assert_eq!(ea.target, 2);
                assert_eq!(ea.label, "knows");
                assert!((ea.weight - 0.9).abs() < 0.001);
            }
            _ => panic!("expected EdgeAdd"),
        }
    }

    #[test]
    fn test_parse_edge_add_default_weight() {
        let cmd = parse_command(
            r#"EDGE ADD TO "test_graph" FROM 1 TO 2 LABEL "knows""#,
        )
        .unwrap();
        match cmd {
            Command::EdgeAdd(ea) => {
                assert!((ea.weight - 1.0).abs() < 0.001);
            }
            _ => panic!("expected EdgeAdd"),
        }
    }

    #[test]
    fn test_parse_edge_invalidate() {
        let cmd = parse_command("EDGE INVALIDATE \"test_graph\" 5").unwrap();
        match cmd {
            Command::EdgeInvalidate(ei) => {
                assert_eq!(ei.graph, "test_graph");
                assert_eq!(ei.edge_id, 5);
            }
            _ => panic!("expected EdgeInvalidate"),
        }
    }

    #[test]
    fn test_parse_context_basic() {
        let cmd = parse_command(
            r#"CONTEXT "what is rust" FROM "knowledge" BUDGET 4096 TOKENS DEPTH 3 DIRECTION BOTH"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                assert_eq!(cq.query_text.as_deref(), Some("what is rust"));
                assert_eq!(cq.graph, "knowledge");
                assert_eq!(cq.budget.as_ref().unwrap().max_tokens, 4096);
                assert_eq!(cq.max_depth, 3);
                assert_eq!(cq.direction, Direction::Both);
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_context_with_seeds() {
        let cmd = parse_command(
            r#"CONTEXT "test" FROM "g" SEEDS NODES ["key1", "key2"]"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                match &cq.seeds {
                    SeedStrategy::Nodes(keys) => {
                        assert_eq!(keys.len(), 2);
                        assert_eq!(keys[0], "key1");
                        assert_eq!(keys[1], "key2");
                    }
                    _ => panic!("expected Nodes seed strategy"),
                }
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_context_with_decay() {
        let cmd = parse_command(
            r#"CONTEXT "test" FROM "g" DECAY EXPONENTIAL 3600000"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                match &cq.decay {
                    Some(DecayFunction::Exponential { half_life_ms }) => {
                        assert_eq!(*half_life_ms, 3600000);
                    }
                    _ => panic!("expected Exponential decay"),
                }
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_context_with_provenance() {
        let cmd = parse_command(
            r#"CONTEXT "test" FROM "g" PROVENANCE"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                assert!(cq.include_provenance);
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_context_with_limit() {
        let cmd = parse_command(
            r#"CONTEXT "test" FROM "g" LIMIT 50"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                assert_eq!(cq.limit, Some(50));
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_empty_input() {
        assert!(parse_command("").is_err());
        assert!(parse_command("   ").is_err());
    }

    #[test]
    fn test_parse_unknown_command() {
        assert!(parse_command("FOOBAR").is_err());
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize(r#"GRAPH CREATE "my graph""#);
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "GRAPH");
        assert_eq!(tokens[1], "CREATE");
        assert_eq!(tokens[2], "\"my graph\"");
    }

    #[test]
    fn test_tokenize_json() {
        let tokens = tokenize(r#"NODE ADD PROPERTIES {"name": "Alice"}"#);
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[3], "{\"name\": \"Alice\"}");
    }

    #[test]
    fn test_parse_context_at_timestamp() {
        let cmd = parse_command(
            r#"CONTEXT "test" FROM "g" AT 1000000"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                assert_eq!(cq.temporal_at, Some(1000000));
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_node_update() {
        let cmd = parse_command(
            r#"NODE UPDATE "test_graph" 42 PROPERTIES {"name": "Bob", "age": 25}"#,
        )
        .unwrap();
        match cmd {
            Command::NodeUpdate(nu) => {
                assert_eq!(nu.graph, "test_graph");
                assert_eq!(nu.node_id, 42);
                assert_eq!(nu.properties.len(), 2);
                assert!(nu.embedding.is_none());
            }
            _ => panic!("expected NodeUpdate"),
        }
    }

    #[test]
    fn test_parse_node_update_with_embedding() {
        let cmd = parse_command(
            r#"NODE UPDATE "g" 1 PROPERTIES {"x": 1} EMBEDDING [0.1, 0.2, 0.3]"#,
        )
        .unwrap();
        match cmd {
            Command::NodeUpdate(nu) => {
                assert_eq!(nu.node_id, 1);
                assert!(nu.embedding.is_some());
                assert_eq!(nu.embedding.unwrap().len(), 3);
            }
            _ => panic!("expected NodeUpdate"),
        }
    }

    #[test]
    fn test_parse_node_update_no_properties() {
        let cmd = parse_command(r#"NODE UPDATE "g" 5 EMBEDDING [1.0, 2.0]"#).unwrap();
        match cmd {
            Command::NodeUpdate(nu) => {
                assert_eq!(nu.node_id, 5);
                assert!(nu.properties.is_empty());
                assert!(nu.embedding.is_some());
            }
            _ => panic!("expected NodeUpdate"),
        }
    }

    #[test]
    fn test_parse_bulk_nodes() {
        let cmd = parse_command(
            r#"BULK NODES TO "g" DATA [{"label": "person", "properties": {"name": "A"}}, {"label": "topic", "entity_key": "t1"}]"#,
        )
        .unwrap();
        match cmd {
            Command::BulkInsertNodes(bulk) => {
                assert_eq!(bulk.graph, "g");
                assert_eq!(bulk.nodes.len(), 2);
                assert_eq!(bulk.nodes[0].label, "person");
                assert_eq!(bulk.nodes[1].label, "topic");
                assert_eq!(bulk.nodes[1].entity_key, Some("t1".to_string()));
            }
            _ => panic!("expected BulkInsertNodes"),
        }
    }

    #[test]
    fn test_parse_bulk_edges() {
        let cmd = parse_command(
            r#"BULK EDGES TO "g" DATA [{"source": 1, "target": 2, "label": "knows", "weight": 0.9}]"#,
        )
        .unwrap();
        match cmd {
            Command::BulkInsertEdges(bulk) => {
                assert_eq!(bulk.graph, "g");
                assert_eq!(bulk.edges.len(), 1);
                assert_eq!(bulk.edges[0].source, 1);
                assert_eq!(bulk.edges[0].target, 2);
                assert_eq!(bulk.edges[0].label, "knows");
                assert!((bulk.edges[0].weight - 0.9).abs() < 0.001);
            }
            _ => panic!("expected BulkInsertEdges"),
        }
    }

    #[test]
    fn test_parse_bulk_edges_default_weight() {
        let cmd = parse_command(
            r#"BULK EDGES TO "g" DATA [{"source": 1, "target": 2, "label": "rel"}]"#,
        )
        .unwrap();
        match cmd {
            Command::BulkInsertEdges(bulk) => {
                assert!((bulk.edges[0].weight - 1.0).abs() < 0.001);
            }
            _ => panic!("expected BulkInsertEdges"),
        }
    }

    #[test]
    fn test_case_insensitive() {
        assert!(matches!(parse_command("ping").unwrap(), Command::Ping));
        assert!(matches!(parse_command("Ping").unwrap(), Command::Ping));
        assert!(matches!(parse_command("info").unwrap(), Command::Info));
    }

    #[test]
    fn test_parse_graph_unknown_subcommand() {
        assert!(parse_command("GRAPH UNKNOWN").is_err());
    }

    #[test]
    fn test_parse_config_unknown_subcommand() {
        assert!(parse_command("CONFIG UNKNOWN").is_err());
    }

    #[test]
    fn test_parse_node_delete_non_numeric_id() {
        assert!(parse_command("NODE DELETE \"g\" abc").is_err());
    }

    #[test]
    fn test_parse_edge_invalidate_non_numeric_id() {
        assert!(parse_command("EDGE INVALIDATE \"g\" abc").is_err());
    }

    #[test]
    fn test_parse_edge_add_non_numeric_source() {
        assert!(parse_command("EDGE ADD TO \"g\" FROM abc TO 2 LABEL \"knows\"").is_err());
    }

    #[test]
    fn test_parse_edge_add_non_numeric_target() {
        assert!(parse_command("EDGE ADD TO \"g\" FROM 1 TO abc LABEL \"knows\"").is_err());
    }

    #[test]
    fn test_parse_node_add_invalid_properties_json() {
        assert!(parse_command("NODE ADD TO \"g\" LABEL \"person\" PROPERTIES {invalid}").is_err());
    }

    #[test]
    fn test_parse_node_add_invalid_embedding() {
        assert!(parse_command("NODE ADD TO \"g\" LABEL \"person\" EMBEDDING [not,numbers]").is_err());
    }

    #[test]
    fn test_parse_context_invalid_budget() {
        assert!(parse_command("CONTEXT \"q\" FROM \"g\" BUDGET abc").is_err());
    }

    #[test]
    fn test_parse_context_invalid_depth() {
        assert!(parse_command("CONTEXT \"q\" FROM \"g\" DEPTH abc").is_err());
    }

    #[test]
    fn test_parse_context_unknown_direction() {
        assert!(parse_command("CONTEXT \"q\" FROM \"g\" DIRECTION SIDEWAYS").is_err());
    }

    #[test]
    fn test_parse_bulk_nodes_invalid_json() {
        assert!(parse_command("BULK NODES TO \"g\" DATA {invalid}").is_err());
    }

    #[test]
    fn test_parse_bulk_edges_invalid_json() {
        assert!(parse_command("BULK EDGES TO \"g\" DATA {invalid}").is_err());
    }

    #[test]
    fn test_parse_edge_delete_non_numeric() {
        assert!(parse_command("EDGE DELETE \"g\" abc").is_err());
    }

    #[test]
    fn test_parse_edge_get_non_numeric() {
        assert!(parse_command("EDGE GET \"g\" abc").is_err());
    }

    #[test]
    fn test_parse_node_update_non_numeric() {
        assert!(parse_command("NODE UPDATE \"g\" abc PROPERTIES {\"name\":\"test\"}").is_err());
    }

    // ── Round 7 edge-case tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_empty_string() {
        let result = parse_command("");
        assert!(result.is_err(), "empty string should return error");
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = parse_command("   \t  ");
        assert!(result.is_err(), "whitespace-only input should return error");
    }

    #[test]
    fn test_parse_unknown_command_with_args() {
        let result = parse_command("FOOBAR something");
        assert!(result.is_err(), "unknown command with args should return error");
    }

    #[test]
    fn test_parse_sql_injection_attempt() {
        // Should fail parsing gracefully, not panic or execute anything dangerous
        let result = parse_command("NODE ADD; DROP TABLE users; --");
        assert!(result.is_err(), "SQL injection-like input should fail gracefully");
    }

    #[test]
    fn test_parse_unclosed_quote() {
        // The tokenizer consumes unclosed quotes until EOF and wraps them.
        // Depending on how the resulting tokens are interpreted, the command
        // may parse or error. Either way it must not panic.
        let result = parse_command("GRAPH CREATE \"unclosed");
        // The tokenizer treats the rest of input as the quoted string content,
        // so this actually parses as GraphCreate with name "unclosed".
        // We just assert it does not panic; either Ok or Err is acceptable.
        match result {
            Ok(Command::GraphCreate(gc)) => {
                assert_eq!(gc.name, "unclosed");
            }
            Err(_) => { /* also acceptable */ }
            other => panic!("unexpected parse result: {:?}", other),
        }
    }

    #[test]
    fn test_parse_escaped_quotes_in_name() {
        // Escaped quotes inside a quoted string — tokenizer supports backslash escapes
        let result = parse_command(r#"GRAPH CREATE "my \"graph\" name""#);
        match result {
            Ok(Command::GraphCreate(gc)) => {
                assert_eq!(gc.name, "my \"graph\" name");
            }
            Err(_) => { /* also acceptable if parser rejects embedded quotes */ }
            other => panic!("unexpected parse result: {:?}", other),
        }
    }

    #[test]
    fn test_parse_negative_node_id() {
        // NodeId is u64, cannot be negative — parser should error
        let result = parse_command("NODE GET \"g\" -1");
        assert!(result.is_err(), "negative node id should return error (u64 cannot be negative)");
    }

    #[test]
    fn test_parse_node_id_overflow() {
        // Number larger than u64::MAX should fail to parse
        let result = parse_command("NODE GET \"g\" 99999999999999999999");
        assert!(result.is_err(), "node id overflow (> u64::MAX) should return error");
    }

    #[test]
    fn test_parse_context_zero_budget() {
        // BUDGET 0 TOKENS should parse successfully (0 is a valid u32)
        let cmd = parse_command(r#"CONTEXT "q" FROM "g" BUDGET 0 TOKENS"#).unwrap();
        match cmd {
            Command::Context(cq) => {
                assert_eq!(cq.budget.as_ref().unwrap().max_tokens, 0);
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_context_huge_budget() {
        // u32::MAX = 4294967295, should parse successfully
        let cmd = parse_command(r#"CONTEXT "q" FROM "g" BUDGET 4294967295 TOKENS"#).unwrap();
        match cmd {
            Command::Context(cq) => {
                assert_eq!(cq.budget.as_ref().unwrap().max_tokens, 4294967295);
            }
            _ => panic!("expected Context"),
        }
    }

    #[test]
    fn test_parse_node_add_empty_properties() {
        // Empty JSON object {} should parse with empty property list
        let cmd = parse_command(
            r#"NODE ADD TO "g" LABEL "L" PROPERTIES {}"#,
        )
        .unwrap();
        match cmd {
            Command::NodeAdd(na) => {
                assert_eq!(na.graph, "g");
                assert_eq!(na.label, "L");
                assert!(na.properties.is_empty(), "empty JSON object should yield empty props");
            }
            _ => panic!("expected NodeAdd"),
        }
    }

    #[test]
    fn test_parse_context_empty_embedding() {
        // SEEDS VECTOR [] — empty vector should parse
        let cmd = parse_command(
            r#"CONTEXT "q" FROM "g" SEEDS VECTOR [] TOP 5"#,
        )
        .unwrap();
        match cmd {
            Command::Context(cq) => {
                match &cq.seeds {
                    SeedStrategy::Vector { embedding, top_k } => {
                        assert!(embedding.is_empty(), "empty embedding array should parse to empty vec");
                        assert_eq!(*top_k, 5);
                    }
                    _ => panic!("expected Vector seed strategy"),
                }
            }
            _ => panic!("expected Context"),
        }
    }

    // ── AUTH / ACL command tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_auth_password_only() {
        let cmd = parse_command("AUTH mysecret").unwrap();
        match cmd {
            Command::Auth { username, password } => {
                assert!(username.is_none());
                assert_eq!(password, "mysecret");
            }
            _ => panic!("expected Auth"),
        }
    }

    #[test]
    fn test_parse_auth_username_password() {
        let cmd = parse_command("AUTH alice secret123").unwrap();
        match cmd {
            Command::Auth { username, password } => {
                assert_eq!(username, Some("alice".into()));
                assert_eq!(password, "secret123");
            }
            _ => panic!("expected Auth"),
        }
    }

    #[test]
    fn test_parse_auth_no_args() {
        let result = parse_command("AUTH");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_acl_whoami() {
        let cmd = parse_command("ACL WHOAMI").unwrap();
        assert!(matches!(cmd, Command::AclWhoAmI));
    }

    #[test]
    fn test_parse_acl_list() {
        let cmd = parse_command("ACL LIST").unwrap();
        assert!(matches!(cmd, Command::AclList));
    }

    #[test]
    fn test_parse_acl_save() {
        let cmd = parse_command("ACL SAVE").unwrap();
        assert!(matches!(cmd, Command::AclSave));
    }

    #[test]
    fn test_parse_acl_load() {
        let cmd = parse_command("ACL LOAD").unwrap();
        assert!(matches!(cmd, Command::AclLoad));
    }

    #[test]
    fn test_parse_acl_getuser() {
        let cmd = parse_command("ACL GETUSER alice").unwrap();
        match cmd {
            Command::AclGetUser(name) => assert_eq!(name, "alice"),
            _ => panic!("expected AclGetUser"),
        }
    }

    #[test]
    fn test_parse_acl_deluser() {
        let cmd = parse_command("ACL DELUSER bob").unwrap();
        match cmd {
            Command::AclDelUser(name) => assert_eq!(name, "bob"),
            _ => panic!("expected AclDelUser"),
        }
    }

    #[test]
    fn test_parse_acl_setuser_basic() {
        let cmd = parse_command("ACL SETUSER alice >pass123 on +@read +@write").unwrap();
        match cmd {
            Command::AclSetUser(cmd) => {
                assert_eq!(cmd.username, "alice");
                assert_eq!(cmd.password, Some("pass123".into()));
                assert_eq!(cmd.enabled, Some(true));
                assert_eq!(cmd.categories, vec!["+@read", "+@write"]);
            }
            _ => panic!("expected AclSetUser"),
        }
    }

    #[test]
    fn test_parse_acl_setuser_with_graph_pattern() {
        let cmd = parse_command("ACL SETUSER bob on +@read ~app:*:readwrite").unwrap();
        match cmd {
            Command::AclSetUser(cmd) => {
                assert_eq!(cmd.username, "bob");
                assert_eq!(cmd.graph_patterns.len(), 1);
                assert_eq!(cmd.graph_patterns[0].0, "app:*");
                assert_eq!(cmd.graph_patterns[0].1, "readwrite");
            }
            _ => panic!("expected AclSetUser"),
        }
    }

    #[test]
    fn test_parse_acl_setuser_disabled() {
        let cmd = parse_command("ACL SETUSER charlie off").unwrap();
        match cmd {
            Command::AclSetUser(cmd) => {
                assert_eq!(cmd.username, "charlie");
                assert_eq!(cmd.enabled, Some(false));
            }
            _ => panic!("expected AclSetUser"),
        }
    }

    #[test]
    fn test_parse_acl_unknown_subcommand() {
        let result = parse_command("ACL UNKNOWN");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_acl_no_subcommand() {
        let result = parse_command("ACL");
        assert!(result.is_err());
    }

    // ── Ingest command tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_ingest_basic() {
        let cmd = parse_command(r#"INGEST "mygraph" "Hello world text""#).unwrap();
        if let Command::Ingest(c) = cmd {
            assert_eq!(c.graph, "mygraph");
            assert_eq!(c.content, "Hello world text");
            assert!(c.format.is_none());
            assert!(c.document_id.is_none());
            assert!(!c.skip_extraction);
            assert!(!c.skip_dedup);
            assert!(c.chunk_size.is_none());
            assert!(c.entity_types.is_none());
        } else {
            panic!("expected Ingest command");
        }
    }

    #[test]
    fn test_parse_ingest_with_options() {
        let cmd = parse_command(
            r#"INGEST "mygraph" "some content" FORMAT "pdf" DOCID "doc123" SKIP_EXTRACTION CHUNK_SIZE 1024 ENTITY_TYPES "Person,Organization""#,
        )
        .unwrap();
        if let Command::Ingest(c) = cmd {
            assert_eq!(c.graph, "mygraph");
            assert_eq!(c.content, "some content");
            assert_eq!(c.format, Some("pdf".into()));
            assert_eq!(c.document_id, Some("doc123".into()));
            assert!(c.skip_extraction);
            assert!(!c.skip_dedup);
            assert_eq!(c.chunk_size, Some(1024));
            assert_eq!(
                c.entity_types,
                Some(vec!["Person".into(), "Organization".into()])
            );
        } else {
            panic!("expected Ingest command");
        }
    }

    #[test]
    fn test_parse_ingest_skip_dedup() {
        let cmd = parse_command(r#"INGEST "g" "text" SKIP_DEDUP"#).unwrap();
        if let Command::Ingest(c) = cmd {
            assert!(c.skip_dedup);
            assert!(!c.skip_extraction);
        } else {
            panic!("expected Ingest command");
        }
    }

    #[test]
    fn test_parse_ingest_missing_args() {
        let result = parse_command("INGEST");
        assert!(result.is_err());
        let result = parse_command(r#"INGEST "mygraph""#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_node_add_with_ttl() {
        let cmd = parse_command(
            r#"NODE ADD TO "g" LABEL "temp" PROPERTIES {"name": "ephemeral"} TTL 60000"#,
        ).unwrap();
        match cmd {
            Command::NodeAdd(c) => {
                assert_eq!(c.graph, "g");
                assert_eq!(c.label, "temp");
                assert_eq!(c.ttl_ms, Some(60000));
            }
            _ => panic!("expected NodeAdd"),
        }
    }

    #[test]
    fn test_parse_node_add_without_ttl() {
        let cmd = parse_command(
            r#"NODE ADD TO "g" LABEL "perm" PROPERTIES {"name": "permanent"}"#,
        ).unwrap();
        match cmd {
            Command::NodeAdd(c) => {
                assert_eq!(c.ttl_ms, None);
            }
            _ => panic!("expected NodeAdd"),
        }
    }

    #[test]
    fn test_parse_edge_add_with_ttl() {
        let cmd = parse_command(
            r#"EDGE ADD TO "g" FROM 1 TO 2 LABEL "link" TTL 30000"#,
        ).unwrap();
        match cmd {
            Command::EdgeAdd(c) => {
                assert_eq!(c.source, 1);
                assert_eq!(c.target, 2);
                assert_eq!(c.ttl_ms, Some(30000));
            }
            _ => panic!("expected EdgeAdd"),
        }
    }

    #[test]
    fn test_parse_ttl_invalid_value() {
        let result = parse_command(
            r#"NODE ADD TO "g" LABEL "x" TTL abc"#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_search() {
        let cmd = parse_command(
            r#"SEARCH "mygraph" WHERE name = "Alice""#,
        ).unwrap();
        match cmd {
            Command::Search(c) => {
                assert_eq!(c.graph, "mygraph");
                assert_eq!(c.key, "name");
                assert_eq!(c.value, "Alice");
                assert_eq!(c.limit, None);
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn test_parse_search_with_limit() {
        let cmd = parse_command(
            r#"SEARCH "g" WHERE age = "30" LIMIT 10"#,
        ).unwrap();
        match cmd {
            Command::Search(c) => {
                assert_eq!(c.key, "age");
                assert_eq!(c.value, "30");
                assert_eq!(c.limit, Some(10));
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn test_parse_neighbors() {
        let cmd = parse_command(
            r#"NEIGHBORS "mygraph" 42"#,
        ).unwrap();
        match cmd {
            Command::Neighbors(c) => {
                assert_eq!(c.graph, "mygraph");
                assert_eq!(c.node_id, 42);
                assert!(c.label.is_none());
                assert_eq!(c.direction, Direction::Both);
            }
            _ => panic!("expected Neighbors"),
        }
    }

    #[test]
    fn test_parse_neighbors_with_direction() {
        let cmd = parse_command(
            r#"NEIGHBORS "g" 1 DIRECTION OUT"#,
        ).unwrap();
        match cmd {
            Command::Neighbors(c) => {
                assert_eq!(c.node_id, 1);
                assert_eq!(c.direction, Direction::Outgoing);
            }
            _ => panic!("expected Neighbors"),
        }
    }

    #[test]
    fn test_parse_neighbors_with_label() {
        let cmd = parse_command(
            r#"NEIGHBORS "g" 5 LABEL "KNOWS" DIRECTION IN"#,
        ).unwrap();
        match cmd {
            Command::Neighbors(c) => {
                assert_eq!(c.node_id, 5);
                assert_eq!(c.label, Some("KNOWS".to_string()));
                assert_eq!(c.direction, Direction::Incoming);
            }
            _ => panic!("expected Neighbors"),
        }
    }

    #[test]
    fn test_parse_search_missing_where() {
        let result = parse_command(r#"SEARCH "g""#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_search_with_equals_sign() {
        let cmd = parse_command(
            r#"SEARCH "g" WHERE email = "a@b.com""#,
        ).unwrap();
        match cmd {
            Command::Search(c) => {
                assert_eq!(c.graph, "g");
                assert_eq!(c.key, "email");
                assert_eq!(c.value, "a@b.com");
                assert_eq!(c.limit, None);
            }
            _ => panic!("expected Search"),
        }
    }

    #[test]
    fn test_parse_neighbors_invalid_id() {
        let result = parse_command(r#"NEIGHBORS "g" abc"#);
        assert!(result.is_err(), "non-numeric node_id should fail parsing");
    }
}
