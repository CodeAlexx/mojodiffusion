// Blocking HTTP helpers for the serenity web trainer supervisor API.
// All calls run on worker threads, never on the egui UI thread.

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::io::Read;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(10);

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout(TIMEOUT)
        .build()
}

/// GET a JSON endpoint. `path` starts with '/'.
pub fn get_json(base: &str, path: &str) -> Result<Value> {
    let url = format!("{base}{path}");
    match agent().get(&url).call() {
        Ok(resp) => Ok(serde_json::from_str(&resp.into_string()?)?),
        Err(ureq::Error::Status(code, resp)) => {
            let body = resp.into_string().unwrap_or_default();
            Err(anyhow!("GET {path} -> {code}: {body}"))
        }
        Err(e) => Err(anyhow!("GET {path}: {e}")),
    }
}

/// POST a JSON body; returns (status, body). 4xx/5xx are returned, not errors,
/// so the UI can show the server's error JSON verbatim.
pub fn post_json(base: &str, path: &str, body: &Value) -> Result<(u16, Value)> {
    send_json(base, path, body, "POST")
}

pub fn put_json(base: &str, path: &str, body: &Value) -> Result<(u16, Value)> {
    send_json(base, path, body, "PUT")
}

fn send_json(base: &str, path: &str, body: &Value, method: &str) -> Result<(u16, Value)> {
    let url = format!("{base}{path}");
    let req = agent().request(method, &url).set("Content-Type", "application/json");
    match req.send_string(&body.to_string()) {
        Ok(resp) => {
            let code = resp.status();
            let text = resp.into_string()?;
            Ok((code, serde_json::from_str(&text).unwrap_or(Value::String(text))))
        }
        Err(ureq::Error::Status(code, resp)) => {
            let text = resp.into_string().unwrap_or_default();
            Ok((code, serde_json::from_str(&text).unwrap_or(Value::String(text))))
        }
        Err(e) => Err(anyhow!("{method} {path}: {e}")),
    }
}

/// Fetch raw bytes (sample/dataset images via /files/...). 16 MB cap.
pub fn get_bytes(base: &str, path: &str) -> Result<Vec<u8>> {
    let url = format!("{base}{path}");
    match agent().get(&url).call() {
        Ok(resp) => {
            let mut buf = Vec::new();
            resp.into_reader().take(16 * 1024 * 1024).read_to_end(&mut buf)?;
            Ok(buf)
        }
        Err(e) => Err(anyhow!("GET {path}: {e}")),
    }
}

/// Open the SSE stream reader (/api/events). Caller reads lines until EOF/error.
pub fn open_sse(base: &str) -> Result<Box<dyn Read + Send>> {
    let url = format!("{base}/api/events");
    // No read timeout: SSE is a long-lived stream; axum keep-alives arrive every 15s.
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(3)).build();
    match agent.get(&url).call() {
        Ok(resp) => Ok(Box::new(resp.into_reader())),
        Err(e) => Err(anyhow!("SSE connect: {e}")),
    }
}
