#![deny(unsafe_code)]

//! Future production serving layer for health, metrics, and streaming APIs.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use common::{Error, Result};
use config::{Config, ConfigSource};
use serde::Serialize;
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerState {
    pub service_name: String,
    pub model_id: String,
    pub model_type: String,
    pub config_source: String,
    pub runtime_mode: String,
    pub backend: String,
    pub device: String,
    pub custom_kernels: bool,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dense_layers: usize,
    pub sparse_moe_layers: usize,
    pub vocab_size: usize,
    pub max_context: usize,
}

impl ServerState {
    pub fn from_config(config: &Config, source: &ConfigSource) -> Self {
        Self {
            service_name: "inferno".to_string(),
            model_id: "glm-5.2-like".to_string(),
            model_type: config.model_type.clone(),
            config_source: format_config_source(source),
            runtime_mode: "q2_runtime_pending".to_string(),
            backend: "metal".to_string(),
            device: default_device_label().to_string(),
            custom_kernels: true,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            dense_layers: config.dense_layers,
            sparse_moe_layers: config.sparse_moe_layers(),
            vocab_size: config.vocab_size,
            max_context: config.max_context,
        }
    }
}

#[derive(Debug, Default)]
pub struct ServerMetrics {
    requests_total: AtomicU64,
    health_requests_total: AtomicU64,
    model_info_requests_total: AtomicU64,
    openai_models_requests_total: AtomicU64,
    openai_chat_completions_requests_total: AtomicU64,
    openai_completions_requests_total: AtomicU64,
    openai_responses_requests_total: AtomicU64,
    metrics_requests_total: AtomicU64,
    stream_requests_total: AtomicU64,
    not_found_total: AtomicU64,
}

impl ServerMetrics {
    pub fn snapshot(&self) -> ServerMetricsSnapshot {
        ServerMetricsSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            health_requests_total: self.health_requests_total.load(Ordering::Relaxed),
            model_info_requests_total: self.model_info_requests_total.load(Ordering::Relaxed),
            openai_models_requests_total: self.openai_models_requests_total.load(Ordering::Relaxed),
            openai_chat_completions_requests_total: self
                .openai_chat_completions_requests_total
                .load(Ordering::Relaxed),
            openai_completions_requests_total: self
                .openai_completions_requests_total
                .load(Ordering::Relaxed),
            openai_responses_requests_total: self
                .openai_responses_requests_total
                .load(Ordering::Relaxed),
            metrics_requests_total: self.metrics_requests_total.load(Ordering::Relaxed),
            stream_requests_total: self.stream_requests_total.load(Ordering::Relaxed),
            not_found_total: self.not_found_total.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerMetricsSnapshot {
    pub requests_total: u64,
    pub health_requests_total: u64,
    pub model_info_requests_total: u64,
    pub openai_models_requests_total: u64,
    pub openai_chat_completions_requests_total: u64,
    pub openai_completions_requests_total: u64,
    pub openai_responses_requests_total: u64,
    pub metrics_requests_total: u64,
    pub stream_requests_total: u64,
    pub not_found_total: u64,
}

#[derive(Debug, Clone)]
pub struct ServerApp {
    state: Arc<ServerState>,
    metrics: Arc<ServerMetrics>,
}

impl ServerApp {
    pub fn new(state: ServerState) -> Self {
        Self {
            state: Arc::new(state),
            metrics: Arc::new(ServerMetrics::default()),
        }
    }

    pub fn state(&self) -> &ServerState {
        &self.state
    }

    pub fn metrics(&self) -> ServerMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn handle(&self, request: &HttpRequest) -> HttpResponse {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/health") => {
                self.metrics
                    .health_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(200, health_body(&self.state))
            }
            ("GET", "/v1/model") => {
                self.metrics
                    .model_info_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(200, model_info_body(&self.state))
            }
            ("GET", "/v1/models") => {
                self.metrics
                    .openai_models_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(200, openai_models_body(&self.state))
            }
            ("GET", "/metrics") => {
                self.metrics
                    .metrics_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                text_response(200, metrics_body(&self.metrics.snapshot()))
            }
            ("POST", "/v1/chat/completions") => {
                self.metrics
                    .openai_chat_completions_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(
                    501,
                    openai_generation_not_implemented_body("chat.completions"),
                )
            }
            ("POST", "/v1/completions") => {
                self.metrics
                    .openai_completions_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(501, openai_generation_not_implemented_body("completions"))
            }
            ("POST", "/v1/responses") => {
                self.metrics
                    .openai_responses_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(501, openai_generation_not_implemented_body("responses"))
            }
            ("POST", "/v1/generate/stream") => {
                self.metrics
                    .stream_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                json_response(501, stream_placeholder_body())
            }
            _ => {
                self.metrics.not_found_total.fetch_add(1, Ordering::Relaxed);
                json_response(404, error_body(404, "route not found"))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
}

impl HttpRequest {
    pub fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            body: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub content_type: String,
    pub body: String,
}

impl HttpResponse {
    pub fn to_http_bytes(&self) -> Vec<u8> {
        let reason = reason_phrase(self.status);
        format!(
            "HTTP/1.1 {} {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            self.status,
            reason,
            self.content_type,
            self.body.len(),
            self.body
        )
        .into_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerCheckEndpointReport {
    pub method: String,
    pub path: String,
    pub status: u16,
    pub content_type: String,
    pub body_contains: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerCheckReport {
    pub state: ServerState,
    pub endpoints: Vec<ServerCheckEndpointReport>,
    pub metrics: ServerMetricsSnapshot,
    pub limitations: Vec<String>,
}

pub fn run_server_check(config: &Config, source: &ConfigSource) -> Result<ServerCheckReport> {
    let state = ServerState::from_config(config, source);
    let app = ServerApp::new(state.clone());
    let checks = [
        HttpRequest::new("GET", "/health"),
        HttpRequest::new("GET", "/v1/model"),
        HttpRequest::new("GET", "/v1/models"),
        HttpRequest::new("GET", "/metrics"),
        HttpRequest::new("POST", "/v1/chat/completions"),
        HttpRequest::new("POST", "/v1/completions"),
        HttpRequest::new("POST", "/v1/responses"),
        HttpRequest::new("POST", "/v1/generate/stream"),
        HttpRequest::new("GET", "/missing"),
    ];

    let mut endpoints = Vec::with_capacity(checks.len());
    for request in checks {
        let response = app.handle(&request);
        let body_contains = expected_body_markers(&request.path, response.status);
        for marker in &body_contains {
            if !response.body.contains(marker) {
                return Err(Error::runtime(format!(
                    "server-check response for {} {} did not contain marker {marker:?}",
                    request.method, request.path
                )));
            }
        }
        endpoints.push(ServerCheckEndpointReport {
            method: request.method,
            path: request.path,
            status: response.status,
            content_type: response.content_type,
            body_contains,
        });
    }

    Ok(ServerCheckReport {
        state,
        endpoints,
        metrics: app.metrics(),
        limitations: vec![
            "server uses a minimal std TCP HTTP scaffold, not a production async HTTP stack".to_string(),
            "streaming generation endpoint returns 501 until real streaming runtime is integrated".to_string(),
            "OpenAI-compatible generation routes return OpenAI-shaped 501 errors until real generation is integrated".to_string(),
            "authentication, request limits, TLS, and full OpenAI protocol coverage are not implemented".to_string(),
            "model info reports configured architecture and runtime mode, not loaded production weights".to_string(),
        ],
    })
}

#[derive(Debug, Clone)]
pub struct ShutdownFlag {
    inner: Arc<AtomicBool>,
}

impl ShutdownFlag {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn request_shutdown(&self) {
        self.inner.store(true, Ordering::SeqCst);
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }
}

impl Default for ShutdownFlag {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeReport {
    pub bound_addr: SocketAddr,
    pub handled_connections: u64,
    pub shutdown_requested: bool,
}

pub fn serve_until_shutdown(
    listener: TcpListener,
    app: ServerApp,
    shutdown: ShutdownFlag,
) -> Result<ServeReport> {
    listener.set_nonblocking(true).map_err(|source| Error::Io {
        path: "tcp-listener".into(),
        source,
    })?;
    let bound_addr = listener.local_addr().map_err(|source| Error::Io {
        path: "tcp-listener".into(),
        source,
    })?;
    let mut handled_connections = 0_u64;

    info!(addr = %bound_addr, "starting glm server scaffold");
    while !shutdown.is_shutdown_requested() {
        match listener.accept() {
            Ok((stream, peer_addr)) => {
                if let Err(error) = handle_stream(stream, &app) {
                    warn!(%peer_addr, %error, "failed to handle HTTP connection");
                }
                handled_connections += 1;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(25));
            }
            Err(source) => {
                return Err(Error::Io {
                    path: "tcp-listener".into(),
                    source,
                });
            }
        }
    }

    info!(
        addr = %bound_addr,
        handled_connections,
        "server scaffold shutdown complete"
    );
    Ok(ServeReport {
        bound_addr,
        handled_connections,
        shutdown_requested: true,
    })
}

fn handle_stream(mut stream: TcpStream, app: &ServerApp) -> Result<()> {
    let mut buffer = [0_u8; 8192];
    let bytes_read = stream.read(&mut buffer).map_err(|source| Error::Io {
        path: "tcp-stream".into(),
        source,
    })?;
    let request = parse_http_request(&buffer[..bytes_read])?;
    let response = app.handle(&request);
    stream
        .write_all(&response.to_http_bytes())
        .map_err(|source| Error::Io {
            path: "tcp-stream".into(),
            source,
        })?;
    Ok(())
}

fn parse_http_request(bytes: &[u8]) -> Result<HttpRequest> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| Error::runtime(format!("HTTP request was not UTF-8: {error}")))?;
    let mut sections = text.split("\r\n\r\n");
    let head = sections.next().unwrap_or_default();
    let body = sections.next().unwrap_or_default().to_string();
    let request_line = head
        .lines()
        .next()
        .ok_or_else(|| Error::runtime("HTTP request line is missing"))?;
    let parts = request_line.split_whitespace().collect::<Vec<_>>();
    if parts.len() < 2 {
        return Err(Error::runtime(format!(
            "invalid HTTP request line: {request_line:?}"
        )));
    }
    Ok(HttpRequest {
        method: parts[0].to_string(),
        path: parts[1].to_string(),
        body,
    })
}

#[derive(Debug, Serialize)]
struct HealthBody<'a> {
    status: &'a str,
    service: &'a str,
    ready: bool,
    runtime_mode: &'a str,
}

#[derive(Debug, Serialize)]
struct ModelInfoBody<'a> {
    service: &'a str,
    model_id: &'a str,
    model_type: &'a str,
    config_source: &'a str,
    runtime_mode: &'a str,
    backend: &'a str,
    device: &'a str,
    custom_kernels: bool,
    hidden_size: usize,
    num_layers: usize,
    dense_layers: usize,
    sparse_moe_layers: usize,
    vocab_size: usize,
    max_context: usize,
    real_weights_loaded: bool,
}

#[derive(Debug, Serialize)]
struct OpenAiModelListBody<'a> {
    object: &'a str,
    data: Vec<OpenAiModelBody<'a>>,
}

#[derive(Debug, Serialize)]
struct OpenAiModelBody<'a> {
    id: &'a str,
    object: &'a str,
    created: u64,
    owned_by: &'a str,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorResponse<'a> {
    error: OpenAiErrorBody<'a>,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorBody<'a> {
    message: String,
    #[serde(rename = "type")]
    error_type: &'a str,
    param: Option<&'a str>,
    code: &'a str,
}

#[derive(Debug, Serialize)]
struct PlaceholderBody<'a> {
    error: &'a str,
    message: &'a str,
    supported_later: Vec<&'a str>,
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
    status: u16,
    message: &'a str,
}

fn health_body(state: &ServerState) -> String {
    to_json(&HealthBody {
        status: "ok",
        service: &state.service_name,
        ready: true,
        runtime_mode: &state.runtime_mode,
    })
}

fn model_info_body(state: &ServerState) -> String {
    to_json(&ModelInfoBody {
        service: &state.service_name,
        model_id: &state.model_id,
        model_type: &state.model_type,
        config_source: &state.config_source,
        runtime_mode: &state.runtime_mode,
        backend: &state.backend,
        device: &state.device,
        custom_kernels: state.custom_kernels,
        hidden_size: state.hidden_size,
        num_layers: state.num_layers,
        dense_layers: state.dense_layers,
        sparse_moe_layers: state.sparse_moe_layers,
        vocab_size: state.vocab_size,
        max_context: state.max_context,
        real_weights_loaded: false,
    })
}

fn openai_models_body(state: &ServerState) -> String {
    to_json(&OpenAiModelListBody {
        object: "list",
        data: vec![OpenAiModelBody {
            id: &state.model_id,
            object: "model",
            created: 0,
            owned_by: &state.service_name,
        }],
    })
}

fn openai_generation_not_implemented_body(endpoint: &str) -> String {
    to_json(&OpenAiErrorResponse {
        error: OpenAiErrorBody {
            message: format!(
                "OpenAI-compatible {endpoint} route is present, but real GLM generation is not yet integrated into the server"
            ),
            error_type: "server_error",
            param: None,
            code: "generation_not_implemented",
        },
    })
}

fn stream_placeholder_body() -> String {
    to_json(&PlaceholderBody {
        error: "streaming_generation_not_implemented",
        message: "streaming endpoint is reserved until real runtime streaming is integrated",
        supported_later: vec![
            "server-sent events",
            "cancellation",
            "token text deltas",
            "runtime metrics",
        ],
    })
}

fn error_body(status: u16, message: &'static str) -> String {
    to_json(&ErrorBody {
        error: "http_error",
        status,
        message,
    })
}

fn metrics_body(metrics: &ServerMetricsSnapshot) -> String {
    format!(
        concat!(
            "# TYPE server_requests_total counter\n",
            "server_requests_total {}\n",
            "# TYPE server_health_requests_total counter\n",
            "server_health_requests_total {}\n",
            "# TYPE server_model_info_requests_total counter\n",
            "server_model_info_requests_total {}\n",
            "# TYPE server_openai_models_requests_total counter\n",
            "server_openai_models_requests_total {}\n",
            "# TYPE server_openai_chat_completions_requests_total counter\n",
            "server_openai_chat_completions_requests_total {}\n",
            "# TYPE server_openai_completions_requests_total counter\n",
            "server_openai_completions_requests_total {}\n",
            "# TYPE server_openai_responses_requests_total counter\n",
            "server_openai_responses_requests_total {}\n",
            "# TYPE server_metrics_requests_total counter\n",
            "server_metrics_requests_total {}\n",
            "# TYPE server_stream_requests_total counter\n",
            "server_stream_requests_total {}\n",
            "# TYPE server_not_found_total counter\n",
            "server_not_found_total {}\n",
        ),
        metrics.requests_total,
        metrics.health_requests_total,
        metrics.model_info_requests_total,
        metrics.openai_models_requests_total,
        metrics.openai_chat_completions_requests_total,
        metrics.openai_completions_requests_total,
        metrics.openai_responses_requests_total,
        metrics.metrics_requests_total,
        metrics.stream_requests_total,
        metrics.not_found_total
    )
}

fn json_response(status: u16, body: String) -> HttpResponse {
    HttpResponse {
        status,
        content_type: "application/json".to_string(),
        body,
    }
}

fn text_response(status: u16, body: String) -> HttpResponse {
    HttpResponse {
        status,
        content_type: "text/plain; version=0.0.4".to_string(),
        body,
    }
}

fn to_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).expect("server response serialization should not fail")
}

fn expected_body_markers(path: &str, status: u16) -> Vec<String> {
    match (path, status) {
        ("/health", 200) => vec!["\"status\": \"ok\"".to_string()],
        ("/v1/model", 200) => vec!["\"model_type\"".to_string(), "\"runtime_mode\"".to_string()],
        ("/v1/models", 200) => vec![
            "\"object\": \"list\"".to_string(),
            "\"id\": \"glm-5.2-like\"".to_string(),
        ],
        ("/metrics", 200) => vec!["server_requests_total".to_string()],
        ("/v1/chat/completions", 501) | ("/v1/completions", 501) | ("/v1/responses", 501) => vec![
            "\"type\": \"server_error\"".to_string(),
            "\"code\": \"generation_not_implemented\"".to_string(),
        ],
        ("/v1/generate/stream", 501) => {
            vec!["streaming_generation_not_implemented".to_string()]
        }
        (_, 404) => vec!["route not found".to_string()],
        _ => Vec::new(),
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        404 => "Not Found",
        501 => "Not Implemented",
        _ => "Unknown",
    }
}

fn format_config_source(source: &ConfigSource) -> String {
    match source {
        ConfigSource::EmbeddedInfernoDefaults => "embedded Inferno defaults".to_string(),
        ConfigSource::External(path) => path.display().to_string(),
    }
}

fn default_device_label() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "metal"
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use config::load_embedded_config;

    use super::*;

    #[test]
    fn server_routes_report_expected_statuses() {
        let config = load_embedded_config().unwrap();
        let source = ConfigSource::EmbeddedInfernoDefaults;
        let report = run_server_check(&config, &source).unwrap();

        assert_eq!(report.endpoints.len(), 9);
        assert!(report
            .endpoints
            .iter()
            .any(|endpoint| endpoint.path == "/health" && endpoint.status == 200));
        assert!(report
            .endpoints
            .iter()
            .any(|endpoint| endpoint.path == "/v1/models" && endpoint.status == 200));
        assert!(report
            .endpoints
            .iter()
            .any(|endpoint| { endpoint.path == "/v1/chat/completions" && endpoint.status == 501 }));
        assert!(report
            .endpoints
            .iter()
            .any(|endpoint| endpoint.path == "/v1/generate/stream" && endpoint.status == 501));
        assert_eq!(report.metrics.requests_total, 9);
        assert_eq!(report.metrics.openai_models_requests_total, 1);
        assert_eq!(report.metrics.openai_chat_completions_requests_total, 1);
        assert_eq!(report.metrics.openai_completions_requests_total, 1);
        assert_eq!(report.metrics.openai_responses_requests_total, 1);
        assert_eq!(report.metrics.not_found_total, 1);
    }

    #[test]
    fn openai_routes_use_expected_schema_markers() {
        let config = load_embedded_config().unwrap();
        let state = ServerState::from_config(&config, &ConfigSource::EmbeddedInfernoDefaults);
        let app = ServerApp::new(state);

        let models = app.handle(&HttpRequest::new("GET", "/v1/models"));
        assert_eq!(models.status, 200);
        assert!(models.body.contains("\"object\": \"list\""));
        assert!(models.body.contains("\"id\": \"glm-5.2-like\""));

        let chat = app.handle(&HttpRequest::new("POST", "/v1/chat/completions"));
        assert_eq!(chat.status, 501);
        assert!(chat.body.contains("\"error\""));
        assert!(chat.body.contains("\"type\": \"server_error\""));
        assert!(chat
            .body
            .contains("\"code\": \"generation_not_implemented\""));
    }

    #[test]
    fn http_response_contains_content_length_and_body() {
        let response = json_response(200, "{\"ok\":true}".to_string());
        let rendered = String::from_utf8(response.to_http_bytes()).unwrap();

        assert!(rendered.starts_with("HTTP/1.1 200 OK"));
        assert!(rendered.contains("content-type: application/json"));
        assert!(rendered.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn parses_basic_http_request() {
        let request =
            parse_http_request(b"GET /health HTTP/1.1\r\nhost: localhost\r\n\r\n").unwrap();

        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/health");
    }

    #[test]
    fn shutdown_flag_tracks_requested_shutdown() {
        let shutdown = ShutdownFlag::new();
        assert!(!shutdown.is_shutdown_requested());
        shutdown.request_shutdown();
        assert!(shutdown.is_shutdown_requested());
    }
}
