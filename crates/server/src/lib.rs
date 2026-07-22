#![deny(unsafe_code)]

//! Narrow local HTTP serving boundary for Codex.
//!
//! Inferno intentionally implements only the streaming Responses API contract
//! needed by Codex. Model loading and inference remain owned by the CLI so the
//! server crate has no dependency on Metal or the model runtime.

mod http;
mod responses;

use std::{
    net::{TcpListener, TcpStream},
    sync::mpsc::{self, Sender},
    thread,
};

use common::{Error, Result};
use tracing::{info, warn};

pub use responses::{ResponseUsage, ResponsesRequest, ResponsesStream};

use http::{read_request, write_json_error, write_json_response};

pub const CODEX_MODEL_ID: &str = "glm-5.2-q2";
const CODEX_MODEL_CATALOG: &str = include_str!("../../../examples/inferno.models.json");

/// Executes one Responses request using an already-loaded model runtime.
///
/// The server processes requests serially. This matches Codex's turn loop and
/// prevents concurrent requests from competing for Inferno's Metal buffers and
/// routed-expert cache.
pub trait ResponsesHandler {
    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> Result<ResponseUsage>;
}

/// Serves local Codex requests until the process is terminated.
pub fn serve(listener: TcpListener, handler: &mut impl ResponsesHandler) -> Result<()> {
    let bound_addr = listener.local_addr().map_err(|source| Error::Io {
        path: "inferno-listener".into(),
        source,
    })?;
    info!(addr = %bound_addr, "Inferno Responses server ready");

    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("inferno-http".to_string())
        .spawn(move || accept_connections(listener, sender))
        .map_err(|source| Error::Io {
            path: "inferno-http-thread".into(),
            source,
        })?;

    while let Ok(work) = receiver.recv() {
        match work {
            InferenceWork::Request {
                mut socket,
                request,
            } => {
                let peer = socket.peer_addr().ok();
                if let Err(error) = handle_responses(&mut socket, handler, request) {
                    warn!(?peer, %error, "Inferno Responses request failed");
                }
            }
            InferenceWork::ListenerFailed(message) => return Err(Error::runtime(message)),
        }
    }
    Err(Error::runtime("Inferno HTTP listener stopped"))
}

enum InferenceWork {
    Request {
        socket: TcpStream,
        request: ResponsesRequest,
    },
    ListenerFailed(String),
}

fn accept_connections(listener: TcpListener, sender: Sender<InferenceWork>) {
    for connection in listener.incoming() {
        match connection {
            Ok(socket) => {
                let peer = socket.peer_addr().ok();
                if let Err(error) = route_connection(socket, &sender) {
                    warn!(?peer, %error, "Inferno Responses request failed");
                }
            }
            Err(source) => {
                let _ = sender.send(InferenceWork::ListenerFailed(format!(
                    "Inferno HTTP listener failed: {source}"
                )));
                return;
            }
        }
    }
}

fn route_connection(mut socket: TcpStream, sender: &Sender<InferenceWork>) -> Result<()> {
    let request = match read_request(&mut socket) {
        Ok(request) => request,
        Err(error) => {
            write_json_error(&mut socket, 400, "invalid_request", &error.to_string())?;
            return Err(error);
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => write_json_response(
            &mut socket,
            200,
            &serde_json::json!({"status": "ok", "service": "inferno"}),
        ),
        ("GET", "/v1/models") => write_json_response(&mut socket, 200, &codex_model_catalog()),
        ("POST", "/v1/responses") => {
            let request = match ResponsesRequest::parse(&request.body) {
                Ok(request) => request,
                Err(error) => {
                    write_json_error(&mut socket, 400, "invalid_request", &error.to_string())?;
                    return Err(error);
                }
            };
            sender
                .send(InferenceWork::Request { socket, request })
                .map_err(|_| Error::runtime("Inferno inference loop stopped"))
        }
        ("POST", _) | ("GET", _) => write_json_error(
            &mut socket,
            404,
            "route_not_found",
            "Inferno supports GET /health, GET /v1/models, and POST /v1/responses",
        ),
        _ => write_json_error(
            &mut socket,
            405,
            "method_not_allowed",
            "unsupported HTTP method",
        ),
    }
}

fn handle_responses(
    socket: &mut TcpStream,
    handler: &mut impl ResponsesHandler,
    request: ResponsesRequest,
) -> Result<()> {
    let mut stream = ResponsesStream::begin(socket, &request.model)?;
    match handler.generate(request, &mut stream) {
        Ok(usage) => stream.completed(usage),
        Err(error) => {
            stream.failed(&error.to_string())?;
            Err(error)
        }
    }
}

fn codex_model_catalog() -> serde_json::Value {
    serde_json::from_str(CODEX_MODEL_CATALOG)
        .expect("checked-in Inferno Codex model catalog must be valid JSON")
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use super::*;

    struct FixedHandler;

    impl ResponsesHandler for FixedHandler {
        fn generate(
            &mut self,
            _request: ResponsesRequest,
            stream: &mut ResponsesStream<'_>,
        ) -> Result<ResponseUsage> {
            stream.text_delta("hello")?;
            stream.message_done("hello")?;
            Ok(ResponseUsage {
                input_tokens: 3,
                output_tokens: 1,
            })
        }
    }

    #[test]
    fn serves_a_complete_codex_responses_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            route_connection(stream, &sender).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let body = serde_json::json!({
            "model": "glm-5.2-q2",
            "instructions": "Be concise.",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "tools": [],
            "stream": true
        })
        .to_string();
        write!(
            client,
            "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
        client.flush().unwrap();
        let InferenceWork::Request {
            mut socket,
            request,
        } = receiver.recv().unwrap()
        else {
            panic!("expected inference request");
        };
        handle_responses(&mut socket, &mut FixedHandler, request).unwrap();
        drop(socket);
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("event: response.created"));
        assert!(response.contains("event: response.output_item.added"));
        assert!(response.contains("event: response.output_text.delta"));
        assert!(response.contains("event: response.output_item.done"));
        assert!(response.contains("event: response.completed"));
        assert!(response.contains("\"input_tokens\":3"));
    }

    #[test]
    fn model_catalog_describes_only_infernos_codex_model() {
        let catalog = codex_model_catalog();
        let models = catalog["models"].as_array().unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["slug"], "glm-5.2-q2");
        assert_eq!(models[0]["shell_type"], "unified_exec");
        assert_eq!(models[0]["supports_parallel_tool_calls"], false);
        assert_eq!(models[0]["supports_reasoning_summaries"], false);
        assert_eq!(models[0]["context_window"], 32768);
    }
}
