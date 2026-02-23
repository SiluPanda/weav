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
    /// Trigger a snapshot.
    Snapshot,
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
}

#[derive(Debug, Clone)]
pub struct EdgeInvalidateCmd {
    pub graph: String,
    pub edge_id: u64,
}

#[derive(Debug, Clone)]
pub struct GraphCreateCmd {
    pub name: String,
    pub config: Option<GraphConfig>,
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
        return Err(parse_err("NODE requires a subcommand (ADD, GET, DELETE)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "ADD" => parse_node_add(tokens),
        "GET" => parse_node_get(tokens),
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

    Ok(Command::NodeAdd(NodeAddCmd {
        graph,
        label,
        properties,
        embedding,
        entity_key,
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
        return Err(parse_err("EDGE requires a subcommand (ADD, INVALIDATE)"));
    }
    let sub = tokens[1].to_uppercase();
    match sub.as_str() {
        "ADD" => parse_edge_add(tokens),
        "INVALIDATE" => parse_edge_invalidate(tokens),
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

    Ok(Command::EdgeAdd(EdgeAddCmd {
        graph,
        source,
        target,
        label,
        weight,
        properties,
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
    fn test_case_insensitive() {
        assert!(matches!(parse_command("ping").unwrap(), Command::Ping));
        assert!(matches!(parse_command("Ping").unwrap(), Command::Ping));
        assert!(matches!(parse_command("info").unwrap(), Command::Info));
    }
}
