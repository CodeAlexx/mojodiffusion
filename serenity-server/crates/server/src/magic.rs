//! GET /v1/llms + POST /v1/magic_prompt — Ideogram-4 prompt generator.
//!
//! `/v1/llms` lists local GGUF LLMs (for the Settings "prompt generator" selector).
//! `/v1/magic_prompt {idea, aspect, model}` expands a short natural-language idea
//! into the structured Ideogram-4 caption JSON, by shelling out to
//! `scripts/magic_prompt.sh`, which spins up an EPHEMERAL llama-server (GPU,
//! spawn→generate→kill) so the LLM never holds VRAM while the image worker runs.
//! The system prompt is Ideogram's own magic_prompt v1.txt.

use std::path::Path;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::AppState;

/// Where local LLM GGUFs live (Gemma, Qwen3, …). One level of subdirs is scanned.
const MODELS_DIR: &str = "/home/alex/models";
const MAGIC_SCRIPT: &str = "/home/alex/mojodiffusion/serenity-server/scripts/magic_prompt.sh";

/// GET /v1/llms — `{"llms":[{id,name,path,size}]}` of local GGUF models.
pub async fn get_llms(State(_st): State<AppState>) -> Response {
    let mut out: Vec<Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(MODELS_DIR) {
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&p) {
                    for s in sub.flatten() {
                        let sp = s.path();
                        if sp.extension().map(|e| e == "gguf").unwrap_or(false) {
                            push_llm(&mut out, &sp);
                        }
                    }
                }
            } else if p.extension().map(|e| e == "gguf").unwrap_or(false) {
                push_llm(&mut out, &p);
            }
        }
    }
    out.sort_by(|a, b| {
        a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or(""))
    });
    Json(json!({ "llms": out })).into_response()
}

fn push_llm(out: &mut Vec<Value>, path: &Path) {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    out.push(json!({
        "id": path.to_string_lossy(),
        "name": name,
        "path": path.to_string_lossy(),
        "size": size,
    }));
}

/// POST /v1/magic_prompt {idea, aspect, model} -> {"caption": "<single-line JSON>"}.
pub async fn post_magic_prompt(State(_st): State<AppState>, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let idea = v["idea"].as_str().unwrap_or("").trim().to_string();
    if idea.is_empty() {
        return err(StatusCode::BAD_REQUEST, "idea required");
    }
    let aspect = v["aspect"].as_str().unwrap_or("1:1").to_string();
    let model = v["model"].as_str().unwrap_or("").to_string();
    if model.is_empty() || !Path::new(&model).exists() {
        return err(StatusCode::BAD_REQUEST, "valid model gguf path required");
    }
    // Shell the ephemeral-llama-server script (blocking; model load + gen ~15-40s).
    let out = std::process::Command::new("bash")
        .arg(MAGIC_SCRIPT)
        .arg(&model)
        .arg(&aspect)
        .arg(&idea)
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let caption = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if caption.is_empty() {
                return err(StatusCode::INTERNAL_SERVER_ERROR, "empty caption");
            }
            Json(json!({ "caption": caption })).into_response()
        }
        Ok(o) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!(
                "magic_prompt failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            ),
        ),
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("spawn failed: {e}"),
        ),
    }
}

fn err(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({ "error": msg }))).into_response()
}
