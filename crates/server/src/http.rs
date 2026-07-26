use std::io::{Read, Write};

use common::{Error, Result};
use serde::Serialize;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
}

pub(crate) fn read_request(reader: &mut impl Read) -> Result<HttpRequest> {
    let mut bytes = Vec::with_capacity(8 * 1024);
    let header_end = loop {
        if let Some(index) = find_header_end(&bytes) {
            break index;
        }
        if bytes.len() >= MAX_HEADER_BYTES {
            return Err(Error::runtime("HTTP request headers exceed 64 KiB"));
        }
        let mut chunk = [0_u8; 8 * 1024];
        let count = reader.read(&mut chunk).map_err(|source| Error::Io {
            path: "inferno-http-request".into(),
            source,
        })?;
        if count == 0 {
            return Err(Error::runtime(
                "HTTP connection closed before request headers completed",
            ));
        }
        bytes.extend_from_slice(&chunk[..count]);
    };

    let (method, path, content_length) = {
        let header_text = std::str::from_utf8(&bytes[..header_end])
            .map_err(|error| Error::runtime(format!("HTTP headers are not UTF-8: {error}")))?;
        let mut lines = header_text.split("\r\n");
        let request_line = lines
            .next()
            .ok_or_else(|| Error::runtime("HTTP request line is missing"))?;
        let mut request_parts = request_line.split_whitespace();
        let method = request_parts
            .next()
            .ok_or_else(|| Error::runtime("HTTP method is missing"))?;
        let target = request_parts
            .next()
            .ok_or_else(|| Error::runtime("HTTP target is missing"))?;
        let version = request_parts
            .next()
            .ok_or_else(|| Error::runtime("HTTP version is missing"))?;
        if request_parts.next().is_some() || version != "HTTP/1.1" {
            return Err(Error::runtime(format!(
                "unsupported HTTP request line {request_line:?}"
            )));
        }

        let mut content_length = None;
        for line in lines {
            let (name, value) = line
                .split_once(':')
                .ok_or_else(|| Error::runtime(format!("invalid HTTP header {line:?}")))?;
            if name.eq_ignore_ascii_case("content-length") {
                let parsed = value.trim().parse::<usize>().map_err(|error| {
                    Error::runtime(format!("invalid content-length header: {error}"))
                })?;
                content_length = Some(parsed);
            }
        }
        (
            method.to_string(),
            target.split('?').next().unwrap_or(target).to_string(),
            content_length,
        )
    };

    let content_length = match method.as_str() {
        "POST" => {
            content_length.ok_or_else(|| Error::runtime("POST request requires content-length"))?
        }
        _ => content_length.unwrap_or(0),
    };
    if content_length > MAX_BODY_BYTES {
        return Err(Error::runtime("HTTP request body exceeds 16 MiB"));
    }

    let body_start = header_end + 4;
    let expected_len = body_start
        .checked_add(content_length)
        .ok_or_else(|| Error::runtime("HTTP request size overflow"))?;
    while bytes.len() < expected_len {
        let remaining = expected_len - bytes.len();
        let mut chunk = [0_u8; 8 * 1024];
        let read_len = remaining.min(chunk.len());
        let count = reader
            .read(&mut chunk[..read_len])
            .map_err(|source| Error::Io {
                path: "inferno-http-request".into(),
                source,
            })?;
        if count == 0 {
            return Err(Error::runtime(
                "HTTP connection closed before request body completed",
            ));
        }
        bytes.extend_from_slice(&chunk[..count]);
    }

    let body = std::str::from_utf8(&bytes[body_start..expected_len])
        .map_err(|error| Error::runtime(format!("HTTP body is not UTF-8: {error}")))?
        .to_string();
    Ok(HttpRequest { method, path, body })
}

pub(crate) fn write_json_response(
    writer: &mut impl Write,
    status: u16,
    body: &impl Serialize,
) -> Result<()> {
    let body = serde_json::to_vec(body)?;
    let head = format!(
        "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        reason_phrase(status),
        body.len()
    );
    writer
        .write_all(head.as_bytes())
        .and_then(|()| writer.write_all(&body))
        .and_then(|()| writer.flush())
        .map_err(|source| Error::Io {
            path: "inferno-http-response".into(),
            source,
        })
}

pub(crate) fn write_json_error(
    writer: &mut impl Write,
    status: u16,
    code: &str,
    message: &str,
) -> Result<()> {
    write_json_response(
        writer,
        status,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": code
            }
        }),
    )
}

pub(crate) fn write_sse_headers(writer: &mut (impl Write + ?Sized)) -> Result<()> {
    writer
        .write_all(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\nx-accel-buffering: no\r\n\r\n",
        )
        .and_then(|()| writer.flush())
        .map_err(|source| Error::Io {
            path: "inferno-sse-response".into(),
            source,
        })
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        429 => "Too Many Requests",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn reads_body_across_multiple_reads() {
        let body = r#"{"model":"glm-5.2-q2"}"#;
        let request = format!(
            "POST /v1/responses?trace=1 HTTP/1.1\r\nhost: localhost\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let mut reader = Cursor::new(request.into_bytes());

        let parsed = read_request(&mut reader).unwrap();

        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/v1/responses");
        assert_eq!(parsed.body, body);
    }

    #[test]
    fn rejects_post_without_content_length() {
        let mut reader = Cursor::new(b"POST /v1/responses HTTP/1.1\r\n\r\n".to_vec());

        let error = read_request(&mut reader).unwrap_err();

        assert!(error.to_string().contains("content-length"));
    }
}
