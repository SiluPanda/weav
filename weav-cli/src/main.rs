use clap::Parser;
use futures::{SinkExt, StreamExt};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use tokio::net::TcpStream;
use tokio_util::codec::Framed;
use weav_proto::resp3::{Resp3Codec, Resp3Value};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "weav-cli", about = "Interactive CLI for Weav database")]
struct Cli {
    /// Server host
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Server port (RESP3)
    #[arg(short, long, default_value_t = 6380)]
    port: u16,

    /// Execute a single command and exit
    #[arg(short, long)]
    command: Option<String>,

    /// Username for authentication
    #[arg(short = 'u', long)]
    user: Option<String>,

    /// Password for authentication
    #[arg(short = 'a', long)]
    password: Option<String>,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Split input into tokens respecting double-quoted strings.
///
/// `GRAPH CREATE "my graph"` becomes `["GRAPH", "CREATE", "my graph"]`.
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut has_token = false; // tracks whether we have started building a token
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '\\' {
                // Handle escaped characters inside quotes.
                if let Some(&next) = chars.peek() {
                    match next {
                        '"' | '\\' => {
                            current.push(next);
                            chars.next();
                        }
                        _ => {
                            current.push(ch);
                        }
                    }
                } else {
                    current.push(ch);
                }
            } else if ch == '"' {
                in_quotes = false;
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
            has_token = true;
        } else if ch.is_whitespace() {
            if has_token {
                tokens.push(std::mem::take(&mut current));
                has_token = false;
            }
        } else {
            current.push(ch);
            has_token = true;
        }
    }

    if has_token {
        tokens.push(current);
    }

    tokens
}

// ---------------------------------------------------------------------------
// RESP3 response formatter
// ---------------------------------------------------------------------------

/// Format a `Resp3Value` for human-readable display.
fn format_resp3(value: &Resp3Value) -> String {
    match value {
        Resp3Value::SimpleString(s) => s.clone(),
        Resp3Value::BlobString(b) => String::from_utf8_lossy(b).to_string(),
        Resp3Value::SimpleError(e) => format!("(error) {}", e),
        Resp3Value::Number(n) => format!("(integer) {}", n),
        Resp3Value::Double(d) => format!("(double) {}", d),
        Resp3Value::Boolean(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Resp3Value::Null => "(nil)".to_string(),
        Resp3Value::Array(items) => {
            if items.is_empty() {
                return "(empty array)".to_string();
            }
            items
                .iter()
                .enumerate()
                .map(|(i, v)| format!("{}) {}", i + 1, format_resp3(v)))
                .collect::<Vec<_>>()
                .join("\n")
        }
        Resp3Value::Map(pairs) => {
            if pairs.is_empty() {
                return "(empty map)".to_string();
            }
            pairs
                .iter()
                .enumerate()
                .map(|(i, (k, v))| format!("{}) {} => {}", i + 1, format_resp3(k), format_resp3(v)))
                .collect::<Vec<_>>()
                .join("\n")
        }
        Resp3Value::BigNumber(n) => format!("(big number) {}", n),
    }
}

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------

const HELP_TEXT: &str = "\
Available commands:
  PING                              - Test connectivity
  INFO                              - Server information
  GRAPH CREATE \"<name>\"             - Create a new graph
  GRAPH LIST                        - List all graphs
  GRAPH INFO \"<name>\"               - Show graph details
  GRAPH DROP \"<name>\"               - Delete a graph
  NODE ADD TO \"<graph>\" LABEL \"<l>\" PROPERTIES {...} - Add a node
  NODE GET \"<graph>\" <id>           - Get a node by ID
  NODE UPDATE \"<graph>\" <id> PROPERTIES {...} [EMBEDDING [...]] - Update a node
  NODE DELETE \"<graph>\" <id>        - Delete a node
  EDGE ADD TO \"<graph>\" FROM <s> TO <t> LABEL \"<l>\" - Add an edge
  BULK NODES TO \"<graph>\" DATA [{...}, ...] - Bulk insert nodes
  BULK EDGES TO \"<graph>\" DATA [{...}, ...] - Bulk insert edges
  CONTEXT \"<query>\" FROM \"<graph>\" BUDGET <n> TOKENS - Context retrieval
  STATS                             - Show statistics
  SNAPSHOT                          - Trigger snapshot
  help                              - Show this help
  quit / exit                       - Exit the CLI";

// ---------------------------------------------------------------------------
// Networking: send a command and receive the response
// ---------------------------------------------------------------------------

async fn send_command(
    host: &str,
    port: u16,
    input: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let tokens = tokenize(input);
    if tokens.is_empty() {
        return Err("empty command".into());
    }

    // Build RESP3 array of BlobStrings from the tokens.
    let array = Resp3Value::Array(
        tokens
            .into_iter()
            .map(|t| Resp3Value::BlobString(t.into_bytes()))
            .collect(),
    );

    // Connect to the server.
    let addr = format!("{}:{}", host, port);
    let stream = TcpStream::connect(&addr).await?;
    let mut framed = Framed::new(stream, Resp3Codec::new());

    // Send the command.
    framed.send(array).await?;

    // Read the response.
    match framed.next().await {
        Some(Ok(response)) => Ok(format_resp3(&response)),
        Some(Err(e)) => Err(Box::new(e)),
        None => Err("connection closed without response".into()),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Single-command mode: execute and exit.
    if let Some(ref cmd) = cli.command {
        match send_command(&cli.host, cli.port, cmd).await {
            Ok(output) => println!("{}", output),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    // Auto-authenticate if credentials are provided.
    if cli.password.is_some() {
        let auth_cmd = match &cli.user {
            Some(u) => format!("AUTH {} {}", u, cli.password.as_ref().unwrap()),
            None => format!("AUTH {}", cli.password.as_ref().unwrap()),
        };
        match send_command(&cli.host, cli.port, &auth_cmd).await {
            Ok(output) => {
                if output.starts_with("(error)") {
                    eprintln!("Authentication failed: {}", output);
                    std::process::exit(1);
                }
                if cli.command.is_none() {
                    println!("Authenticated: {}", output);
                }
            }
            Err(e) => {
                eprintln!("Authentication error: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Interactive REPL mode.
    println!(
        "weav-cli — connected to {}:{} (type 'help' for commands, 'quit' to exit)",
        cli.host, cli.port
    );

    let mut rl = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(e) => {
            eprintln!("Failed to initialize readline: {}", e);
            std::process::exit(1);
        }
    };

    loop {
        match rl.readline("weav> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Add to history.
                let _ = rl.add_history_entry(trimmed);

                // Handle built-in commands.
                let lower = trimmed.to_lowercase();
                if lower == "quit" || lower == "exit" {
                    println!("Bye!");
                    break;
                }
                if lower == "help" {
                    println!("{}", HELP_TEXT);
                    continue;
                }

                // Send the command to the server.
                match send_command(&cli.host, cli.port, trimmed).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C: print hint and continue.
                println!("(Use 'quit' or 'exit' to leave)");
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D: exit.
                println!("Bye!");
                break;
            }
            Err(e) => {
                eprintln!("Readline error: {}", e);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- tokenize -----------------------------------------------------------

    #[test]
    fn test_tokenize_simple() {
        assert_eq!(tokenize("PING"), vec!["PING"]);
    }

    #[test]
    fn test_tokenize_multiple_words() {
        assert_eq!(tokenize("GRAPH LIST"), vec!["GRAPH", "LIST"]);
    }

    #[test]
    fn test_tokenize_quoted_string() {
        assert_eq!(
            tokenize("GRAPH CREATE \"my graph\""),
            vec!["GRAPH", "CREATE", "my graph"]
        );
    }

    #[test]
    fn test_tokenize_multiple_quoted_strings() {
        assert_eq!(
            tokenize("CONTEXT \"hello world\" FROM \"my graph\""),
            vec!["CONTEXT", "hello world", "FROM", "my graph"]
        );
    }

    #[test]
    fn test_tokenize_escaped_quote_in_string() {
        assert_eq!(
            tokenize(r#"NODE ADD TO "test" LABEL "say \"hi\"""#),
            vec!["NODE", "ADD", "TO", "test", "LABEL", "say \"hi\""]
        );
    }

    #[test]
    fn test_tokenize_empty_input() {
        let result: Vec<String> = vec![];
        assert_eq!(tokenize(""), result);
    }

    #[test]
    fn test_tokenize_only_whitespace() {
        let result: Vec<String> = vec![];
        assert_eq!(tokenize("   "), result);
    }

    #[test]
    fn test_tokenize_extra_whitespace() {
        assert_eq!(tokenize("  GRAPH   LIST  "), vec!["GRAPH", "LIST"]);
    }

    #[test]
    fn test_tokenize_empty_quoted_string() {
        assert_eq!(
            tokenize("GRAPH CREATE \"\""),
            vec!["GRAPH", "CREATE", ""]
        );
    }

    #[test]
    fn test_tokenize_json_properties() {
        assert_eq!(
            tokenize(r#"NODE ADD TO "g" LABEL "person" PROPERTIES "{\"name\": \"Alice\"}""#),
            vec![
                "NODE",
                "ADD",
                "TO",
                "g",
                "LABEL",
                "person",
                "PROPERTIES",
                "{\"name\": \"Alice\"}"
            ]
        );
    }

    // -- format_resp3 -------------------------------------------------------

    #[test]
    fn test_format_simple_string() {
        let val = Resp3Value::SimpleString("OK".to_string());
        assert_eq!(format_resp3(&val), "OK");
    }

    #[test]
    fn test_format_blob_string() {
        let val = Resp3Value::BlobString(b"hello world".to_vec());
        assert_eq!(format_resp3(&val), "hello world");
    }

    #[test]
    fn test_format_simple_error() {
        let val = Resp3Value::SimpleError("ERR unknown command".to_string());
        assert_eq!(format_resp3(&val), "(error) ERR unknown command");
    }

    #[test]
    fn test_format_number() {
        let val = Resp3Value::Number(42);
        assert_eq!(format_resp3(&val), "(integer) 42");
    }

    #[test]
    fn test_format_negative_number() {
        let val = Resp3Value::Number(-7);
        assert_eq!(format_resp3(&val), "(integer) -7");
    }

    #[test]
    fn test_format_double() {
        let val = Resp3Value::Double(3.14);
        assert_eq!(format_resp3(&val), "(double) 3.14");
    }

    #[test]
    fn test_format_boolean_true() {
        let val = Resp3Value::Boolean(true);
        assert_eq!(format_resp3(&val), "true");
    }

    #[test]
    fn test_format_boolean_false() {
        let val = Resp3Value::Boolean(false);
        assert_eq!(format_resp3(&val), "false");
    }

    #[test]
    fn test_format_null() {
        let val = Resp3Value::Null;
        assert_eq!(format_resp3(&val), "(nil)");
    }

    #[test]
    fn test_format_array() {
        let val = Resp3Value::Array(vec![
            Resp3Value::SimpleString("hello".to_string()),
            Resp3Value::Number(42),
            Resp3Value::Null,
        ]);
        assert_eq!(format_resp3(&val), "1) hello\n2) (integer) 42\n3) (nil)");
    }

    #[test]
    fn test_format_empty_array() {
        let val = Resp3Value::Array(vec![]);
        assert_eq!(format_resp3(&val), "(empty array)");
    }

    #[test]
    fn test_format_map() {
        let val = Resp3Value::Map(vec![
            (
                Resp3Value::SimpleString("name".to_string()),
                Resp3Value::BlobString(b"Alice".to_vec()),
            ),
            (
                Resp3Value::SimpleString("age".to_string()),
                Resp3Value::Number(30),
            ),
        ]);
        assert_eq!(
            format_resp3(&val),
            "1) name => Alice\n2) age => (integer) 30"
        );
    }

    #[test]
    fn test_format_empty_map() {
        let val = Resp3Value::Map(vec![]);
        assert_eq!(format_resp3(&val), "(empty map)");
    }

    #[test]
    fn test_format_nested_array() {
        let val = Resp3Value::Array(vec![
            Resp3Value::Array(vec![Resp3Value::Number(1), Resp3Value::Number(2)]),
            Resp3Value::SimpleString("done".to_string()),
        ]);
        let formatted = format_resp3(&val);
        assert_eq!(formatted, "1) 1) (integer) 1\n2) (integer) 2\n2) done");
    }

    // -- tokenize edge cases ------------------------------------------------

    #[test]
    fn test_tokenize_unterminated_quote() {
        // Unterminated quote: the quoted token runs to end of input without panic.
        let tokens = tokenize("GRAPH CREATE \"unclosed");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "GRAPH");
        assert_eq!(tokens[1], "CREATE");
        assert_eq!(tokens[2], "unclosed");
    }

    #[test]
    fn test_tokenize_backslash_at_eof() {
        // Backslash at end of quoted string should not panic.
        let tokens = tokenize("\"test\\");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], "test\\");
    }

    #[test]
    fn test_tokenize_tab_character() {
        // Tab is whitespace, so it should split tokens.
        let tokens = tokenize("hello\tworld");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "hello");
        assert_eq!(tokens[1], "world");
    }

    #[test]
    fn test_tokenize_single_quote_in_unquoted() {
        // Single quote (apostrophe) has no special meaning outside double quotes.
        let tokens = tokenize("can't stop");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "can't");
        assert_eq!(tokens[1], "stop");
    }

    // -- format_resp3 edge cases --------------------------------------------

    #[test]
    fn test_format_resp3_big_number() {
        let val = Resp3Value::BigNumber("12345".to_string());
        let formatted = format_resp3(&val);
        assert!(formatted.contains("12345"));
    }

    #[test]
    fn test_format_resp3_number_zero() {
        let val = Resp3Value::Number(0);
        assert_eq!(format_resp3(&val), "(integer) 0");
    }

    #[test]
    fn test_format_resp3_number_min() {
        let val = Resp3Value::Number(i64::MIN);
        let formatted = format_resp3(&val);
        assert!(formatted.contains(&i64::MIN.to_string()));
        assert!(formatted.starts_with("(integer) "));
    }

    #[test]
    fn test_format_resp3_number_max() {
        let val = Resp3Value::Number(i64::MAX);
        let formatted = format_resp3(&val);
        assert!(formatted.contains(&i64::MAX.to_string()));
        assert!(formatted.starts_with("(integer) "));
    }

    #[test]
    fn test_format_resp3_double_nan() {
        let val = Resp3Value::Double(f64::NAN);
        let formatted = format_resp3(&val);
        assert!(formatted.contains("NaN"));
    }

    #[test]
    fn test_format_resp3_double_infinity() {
        let val = Resp3Value::Double(f64::INFINITY);
        let formatted = format_resp3(&val);
        assert!(formatted.contains("inf"));
    }

    #[test]
    fn test_format_resp3_double_zero() {
        let val = Resp3Value::Double(0.0);
        assert_eq!(format_resp3(&val), "(double) 0");
    }

    #[test]
    fn test_format_resp3_empty_simple_string() {
        let val = Resp3Value::SimpleString("".into());
        let formatted = format_resp3(&val);
        assert_eq!(formatted, "");
    }

    #[test]
    fn test_format_resp3_empty_blob_string() {
        let val = Resp3Value::BlobString(vec![]);
        let formatted = format_resp3(&val);
        assert_eq!(formatted, "");
    }

    #[test]
    fn test_format_resp3_single_item_array() {
        let val = Resp3Value::Array(vec![Resp3Value::Number(1)]);
        let formatted = format_resp3(&val);
        assert_eq!(formatted, "1) (integer) 1");
    }

    #[test]
    fn test_format_resp3_map_with_nested_values() {
        let val = Resp3Value::Map(vec![
            (
                Resp3Value::SimpleString("items".to_string()),
                Resp3Value::Array(vec![
                    Resp3Value::Number(10),
                    Resp3Value::Number(20),
                ]),
            ),
            (
                Resp3Value::SimpleString("count".to_string()),
                Resp3Value::Number(2),
            ),
        ]);
        let formatted = format_resp3(&val);
        assert!(formatted.contains("items => 1) (integer) 10\n2) (integer) 20"));
        assert!(formatted.contains("count => (integer) 2"));
    }
}
