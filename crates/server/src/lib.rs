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
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender, SyncSender, TrySendError},
        Arc,
    },
    thread,
    time::Duration,
};

use common::{Error, Result};
use tracing::{info, warn};

pub use responses::{ResponseUsage, ResponsesRequest, ResponsesStream};

use http::{read_request, write_json_error, write_json_response};

pub const GLM_CODEX_MODEL_ID: &str = "glm-5.2-q2";
pub const LAGUNA_CODEX_MODEL_ID: &str = "laguna-s-2.1-int4";
pub const LAGUNA_GGUF_CODEX_MODEL_ID: &str = "laguna-s-2.1-gguf";
pub const LAGUNA_XS_GGUF_CODEX_MODEL_ID: &str = "laguna-xs-2.1-gguf";
const CODEX_MODEL_CATALOG: &str = include_str!("../../../examples/inferno.models.json");
const EXCLUSIVE_CONNECTION_IO_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ServerAdmission {
    #[default]
    Queue,
    RejectWhenBusy,
}

/// Executes one Responses request using an already-loaded model runtime.
///
/// The server processes requests serially. This matches Codex's turn loop and
/// prevents concurrent requests from competing for Inferno's Metal buffers and
/// routed-expert cache.
pub trait ResponsesHandler {
    fn admission(&self) -> ServerAdmission {
        ServerAdmission::Queue
    }

    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> Result<ResponseUsage>;
}

/// Serves local Codex requests until the process is terminated.
pub fn serve(
    listener: TcpListener,
    handler: &mut impl ResponsesHandler,
    model_id: &str,
) -> Result<()> {
    let bound_addr = listener.local_addr().map_err(|source| Error::Io {
        path: "inferno-listener".into(),
        source,
    })?;
    let model_catalog = codex_model_catalog(model_id)?;
    info!(addr = %bound_addr, "Inferno Responses server ready");

    let admission = handler.admission();
    let (sender, receiver, exclusive_busy) = match admission {
        ServerAdmission::Queue => {
            let (sender, receiver) = mpsc::channel();
            (WorkSender::Queue(sender), receiver, None)
        }
        ServerAdmission::RejectWhenBusy => {
            let (sender, receiver) = mpsc::sync_channel(1);
            let busy = Arc::new(AtomicBool::new(false));
            (
                WorkSender::Exclusive {
                    sender,
                    busy: Arc::clone(&busy),
                },
                receiver,
                Some(busy),
            )
        }
    };
    thread::Builder::new()
        .name("inferno-http".to_string())
        .spawn(move || accept_connections(listener, sender, model_catalog))
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
                let result = handle_responses(&mut socket, handler, request);
                if let Some(busy) = &exclusive_busy {
                    busy.store(false, Ordering::Release);
                }
                if let Err(error) = result {
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

enum WorkSender {
    Queue(Sender<InferenceWork>),
    Exclusive {
        sender: SyncSender<InferenceWork>,
        busy: Arc<AtomicBool>,
    },
}

impl WorkSender {
    fn configure_socket(&self, socket: &TcpStream) -> Result<()> {
        if !matches!(self, Self::Exclusive { .. }) {
            return Ok(());
        }
        socket
            .set_read_timeout(Some(EXCLUSIVE_CONNECTION_IO_TIMEOUT))
            .and_then(|()| socket.set_write_timeout(Some(EXCLUSIVE_CONNECTION_IO_TIMEOUT)))
            .and_then(|()| socket.set_nodelay(true))
            .map_err(|source| Error::Io {
                path: "inferno-http-socket".into(),
                source,
            })
    }

    fn send_request(&self, socket: TcpStream, request: ResponsesRequest) -> Result<()> {
        let work = InferenceWork::Request { socket, request };
        match self {
            Self::Queue(sender) => sender
                .send(work)
                .map_err(|_| Error::runtime("Inferno inference loop stopped")),
            Self::Exclusive { sender, busy } => {
                if busy
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    let InferenceWork::Request {
                        mut socket,
                        request: _,
                    } = work
                    else {
                        unreachable!("request submission cannot contain a listener failure")
                    };
                    return write_busy_response(&mut socket);
                }
                match sender.try_send(work) {
                    Ok(()) => Ok(()),
                    Err(TrySendError::Full(InferenceWork::Request {
                        mut socket,
                        request: _,
                    })) => {
                        busy.store(false, Ordering::Release);
                        write_busy_response(&mut socket)
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        busy.store(false, Ordering::Release);
                        Err(Error::runtime("Inferno inference loop stopped"))
                    }
                    Err(TrySendError::Full(InferenceWork::ListenerFailed(_))) => {
                        unreachable!("request submission cannot contain a listener failure")
                    }
                }
            }
        }
    }

    fn send_listener_failure(&self, message: String) {
        let work = InferenceWork::ListenerFailed(message);
        match self {
            Self::Queue(sender) => {
                let _ = sender.send(work);
            }
            Self::Exclusive { sender, .. } => {
                let _ = sender.send(work);
            }
        }
    }
}

fn write_busy_response(socket: &mut TcpStream) -> Result<()> {
    write_json_error(
        socket,
        429,
        "server_busy",
        "Inferno is already processing a Laguna GGUF request",
    )
}

fn accept_connections(listener: TcpListener, sender: WorkSender, model_catalog: serde_json::Value) {
    for connection in listener.incoming() {
        match connection {
            Ok(socket) => {
                let peer = socket.peer_addr().ok();
                if let Err(error) = sender.configure_socket(&socket) {
                    warn!(?peer, %error, "Inferno could not configure HTTP connection");
                    continue;
                }
                if let Err(error) = route_connection(socket, &sender, &model_catalog) {
                    warn!(?peer, %error, "Inferno Responses request failed");
                }
            }
            Err(source) => {
                sender.send_listener_failure(format!("Inferno HTTP listener failed: {source}"));
                return;
            }
        }
    }
}

fn route_connection(
    mut socket: TcpStream,
    sender: &WorkSender,
    model_catalog: &serde_json::Value,
) -> Result<()> {
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
        ("GET", "/v1/models") => write_json_response(&mut socket, 200, model_catalog),
        ("POST", "/v1/responses") => {
            let request = match ResponsesRequest::parse(&request.body) {
                Ok(request) => request,
                Err(error) => {
                    write_json_error(&mut socket, 400, "invalid_request", &error.to_string())?;
                    return Err(error);
                }
            };
            sender.send_request(socket, request)
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

fn codex_model_catalog(model_id: &str) -> Result<serde_json::Value> {
    let mut catalog: serde_json::Value = serde_json::from_str(CODEX_MODEL_CATALOG)?;
    let models = catalog
        .get_mut("models")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| Error::runtime("Inferno model catalog is missing a models array"))?;
    models.retain(|model| model.get("slug").and_then(serde_json::Value::as_str) == Some(model_id));
    if models.len() != 1 {
        return Err(Error::runtime(format!(
            "Inferno model catalog does not contain exactly one entry for {model_id:?}"
        )));
    }
    Ok(catalog)
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
        let sender = WorkSender::Queue(sender);
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let catalog = codex_model_catalog(GLM_CODEX_MODEL_ID).unwrap();
            route_connection(stream, &sender, &catalog).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let body = serde_json::json!({
            "model": GLM_CODEX_MODEL_ID,
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
    fn exclusive_admission_rejects_requests_while_inference_is_busy() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (sender, _receiver) = mpsc::sync_channel(1);
        let busy = Arc::new(AtomicBool::new(true));
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let catalog = codex_model_catalog(LAGUNA_GGUF_CODEX_MODEL_ID).unwrap();
            route_connection(stream, &WorkSender::Exclusive { sender, busy }, &catalog).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let body = serde_json::json!({
            "model": LAGUNA_GGUF_CODEX_MODEL_ID,
            "instructions": "Be concise.",
            "input": [],
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

        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();

        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests"));
        assert!(response.contains("\"code\":\"server_busy\""));
    }

    #[test]
    fn exclusive_admission_accepts_the_first_request_before_receiver_waits() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        let busy = Arc::new(AtomicBool::new(false));
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let catalog = codex_model_catalog(LAGUNA_GGUF_CODEX_MODEL_ID).unwrap();
            route_connection(stream, &WorkSender::Exclusive { sender, busy }, &catalog).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let body = serde_json::json!({
            "model": LAGUNA_GGUF_CODEX_MODEL_ID,
            "instructions": "Be concise.",
            "input": [],
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
        server.join().unwrap();

        let InferenceWork::Request { request, .. } = receiver.recv().unwrap() else {
            panic!("expected inference request");
        };
        assert_eq!(request.model, LAGUNA_GGUF_CODEX_MODEL_ID);
    }

    #[test]
    fn model_catalog_describes_only_the_loaded_model() {
        for model_id in [
            GLM_CODEX_MODEL_ID,
            LAGUNA_CODEX_MODEL_ID,
            LAGUNA_GGUF_CODEX_MODEL_ID,
            LAGUNA_XS_GGUF_CODEX_MODEL_ID,
        ] {
            let catalog = codex_model_catalog(model_id).unwrap();
            let models = catalog["models"].as_array().unwrap();

            assert_eq!(models.len(), 1);
            assert_eq!(models[0]["slug"], model_id);
            assert_eq!(models[0]["shell_type"], "unified_exec");
            assert_eq!(models[0]["supports_parallel_tool_calls"], false);
            assert_eq!(models[0]["supports_reasoning_summaries"], false);
            assert_eq!(models[0]["context_window"], 32768);
        }
    }

    #[test]
    fn model_catalog_rejects_unknown_runtime_model() {
        let error = codex_model_catalog("unknown").unwrap_err();
        assert!(error.to_string().contains("unknown"));
    }
}
