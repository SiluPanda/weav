//! RESP3 TCP server for handling commands over the Redis-like protocol.

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_util::codec::Framed;

use weav_proto::command::{
    context_result_to_resp3, error_to_resp3, node_id_to_resp3, ok_resp3,
    resp3_to_command_string, string_list_to_resp3,
};
use weav_proto::resp3::{Resp3Codec, Resp3Value};
use weav_query::parser::parse_command;

use crate::engine::{CommandResponse, Engine};

/// Convert a `CommandResponse` to a `Resp3Value` for sending back over the wire.
fn command_response_to_resp3(resp: CommandResponse) -> Resp3Value {
    match resp {
        CommandResponse::Ok => ok_resp3(),
        CommandResponse::Pong => Resp3Value::SimpleString("PONG".to_string()),
        CommandResponse::Integer(n) => node_id_to_resp3(n),
        CommandResponse::IntegerList(ids) => {
            Resp3Value::Array(ids.into_iter().map(|id| Resp3Value::Number(id as i64)).collect())
        }
        CommandResponse::Text(s) => Resp3Value::BlobString(s.into_bytes()),
        CommandResponse::StringList(items) => string_list_to_resp3(items),
        CommandResponse::Context(result) => context_result_to_resp3(&result),
        CommandResponse::NodeInfo(info) => {
            let mut pairs = vec![
                (
                    Resp3Value::SimpleString("node_id".to_string()),
                    Resp3Value::Number(info.node_id as i64),
                ),
                (
                    Resp3Value::SimpleString("label".to_string()),
                    Resp3Value::SimpleString(info.label),
                ),
            ];
            let prop_pairs: Vec<(Resp3Value, Resp3Value)> = info
                .properties
                .into_iter()
                .map(|(k, v)| {
                    (
                        Resp3Value::SimpleString(k),
                        Resp3Value::BlobString(format!("{v:?}").into_bytes()),
                    )
                })
                .collect();
            pairs.push((
                Resp3Value::SimpleString("properties".to_string()),
                Resp3Value::Map(prop_pairs),
            ));
            Resp3Value::Map(pairs)
        }
        CommandResponse::GraphInfo(info) => Resp3Value::Map(vec![
            (
                Resp3Value::SimpleString("name".to_string()),
                Resp3Value::SimpleString(info.name),
            ),
            (
                Resp3Value::SimpleString("node_count".to_string()),
                Resp3Value::Number(info.node_count as i64),
            ),
            (
                Resp3Value::SimpleString("edge_count".to_string()),
                Resp3Value::Number(info.edge_count as i64),
            ),
        ]),
        CommandResponse::EdgeInfo(info) => Resp3Value::Map(vec![
            (
                Resp3Value::SimpleString("edge_id".to_string()),
                Resp3Value::Number(info.edge_id as i64),
            ),
            (
                Resp3Value::SimpleString("source".to_string()),
                Resp3Value::Number(info.source as i64),
            ),
            (
                Resp3Value::SimpleString("target".to_string()),
                Resp3Value::Number(info.target as i64),
            ),
            (
                Resp3Value::SimpleString("label".to_string()),
                Resp3Value::SimpleString(info.label),
            ),
            (
                Resp3Value::SimpleString("weight".to_string()),
                Resp3Value::Double(info.weight as f64),
            ),
        ]),
        CommandResponse::Null => Resp3Value::Null,
        CommandResponse::Error(msg) => Resp3Value::SimpleError(msg),
    }
}

/// Run the RESP3 TCP server, accepting connections and processing commands.
pub async fn run_resp3_server(engine: Arc<Engine>, addr: &str) {
    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind RESP3 server on {}: {}", addr, e);
            return;
        }
    };

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::warn!("Failed to accept RESP3 connection: {}", e);
                continue;
            }
        };

        let engine = engine.clone();
        tokio::spawn(async move {
            tracing::debug!("RESP3 connection from {}", peer);
            let mut framed = Framed::new(stream, Resp3Codec::new());

            // Per-connection identity (None until AUTH succeeds).
            let mut identity: Option<weav_auth::identity::SessionIdentity> = None;

            while let Some(frame_result) = framed.next().await {
                let frame = match frame_result {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::debug!("RESP3 decode error from {}: {}", peer, e);
                        break;
                    }
                };

                // Convert RESP3 frame to command string.
                let cmd_str = match resp3_to_command_string(&frame) {
                    Ok(s) => s,
                    Err(e) => {
                        let resp = error_to_resp3(&e);
                        if framed.send(resp).await.is_err() {
                            break;
                        }
                        continue;
                    }
                };

                // Parse the command.
                let cmd = match parse_command(&cmd_str) {
                    Ok(cmd) => cmd,
                    Err(e) => {
                        if framed.send(error_to_resp3(&e)).await.is_err() {
                            break;
                        }
                        continue;
                    }
                };

                // Handle AUTH specially: authenticate and store the identity.
                if let weav_query::parser::Command::Auth { ref username, ref password } = cmd {
                    if engine.is_auth_enabled() {
                        let auth_result = match username {
                            Some(u) => engine.authenticate(u, password),
                            None => engine.authenticate_default(password),
                        };
                        let response = match auth_result {
                            Ok(id) => {
                                let resp_text = format!("OK (user: {})", id.username);
                                identity = Some(id);
                                command_response_to_resp3(CommandResponse::Text(resp_text))
                            }
                            Err(e) => error_to_resp3(&e),
                        };
                        if framed.send(response).await.is_err() {
                            break;
                        }
                        continue;
                    }
                }

                // Execute command with current identity.
                let response = match engine.execute_command(cmd, identity.as_ref()) {
                    Ok(resp) => command_response_to_resp3(resp),
                    Err(e) => error_to_resp3(&e),
                };

                if framed.send(response).await.is_err() {
                    break;
                }
            }

            tracing::debug!("RESP3 connection closed from {}", peer);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use weav_core::config::WeavConfig;

    #[test]
    fn test_command_response_to_resp3_ok() {
        let resp = command_response_to_resp3(CommandResponse::Ok);
        match resp {
            Resp3Value::SimpleString(s) => assert_eq!(s, "OK"),
            _ => panic!("expected SimpleString OK"),
        }
    }

    #[test]
    fn test_command_response_to_resp3_text() {
        let resp = command_response_to_resp3(CommandResponse::Text("hello".into()));
        match resp {
            Resp3Value::BlobString(bytes) => assert_eq!(bytes, b"hello"),
            _ => panic!("expected BlobString"),
        }
    }

    #[test]
    fn test_command_response_to_resp3_integer() {
        let resp = command_response_to_resp3(CommandResponse::Integer(42));
        match resp {
            Resp3Value::Number(n) => assert_eq!(n, 42),
            _ => panic!("expected Number(42)"),
        }
    }

    #[test]
    fn test_command_response_to_resp3_string_list() {
        let resp = command_response_to_resp3(CommandResponse::StringList(vec![
            "a".into(),
            "b".into(),
        ]));
        match resp {
            Resp3Value::Array(items) => {
                assert_eq!(items.len(), 2);
                match &items[0] {
                    Resp3Value::BlobString(b) => assert_eq!(b, b"a"),
                    _ => panic!("expected BlobString for first item"),
                }
                match &items[1] {
                    Resp3Value::BlobString(b) => assert_eq!(b, b"b"),
                    _ => panic!("expected BlobString for second item"),
                }
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_command_response_to_resp3_null() {
        let resp = command_response_to_resp3(CommandResponse::Null);
        assert!(matches!(resp, Resp3Value::Null));
    }

    #[test]
    fn test_command_response_to_resp3_error() {
        let resp = command_response_to_resp3(CommandResponse::Error("oops".into()));
        match resp {
            Resp3Value::SimpleError(msg) => assert_eq!(msg, "oops"),
            _ => panic!("expected SimpleError"),
        }
    }

    #[tokio::test]
    async fn test_resp3_ping_pong() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn server.
        let engine_clone = engine.clone();
        let server_handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut framed = Framed::new(stream, Resp3Codec::new());
            if let Some(Ok(frame)) = framed.next().await {
                let cmd_str = resp3_to_command_string(&frame).unwrap();
                let cmd = parse_command(&cmd_str).unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                let resp3 = command_response_to_resp3(resp);
                framed.send(resp3).await.unwrap();
            }
        });

        // Connect as client.
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();

        // Send RESP3-encoded PING: *1\r\n$4\r\nPING\r\n
        let ping_bytes = b"*1\r\n$4\r\nPING\r\n";
        stream.write_all(ping_bytes).await.unwrap();

        // Read response.
        let mut buf = vec![0u8; 64];
        let n = stream.read(&mut buf).await.unwrap();
        let response = std::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(response, "+PONG\r\n");

        server_handle.await.unwrap();
    }
}
