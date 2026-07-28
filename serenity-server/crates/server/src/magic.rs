//! GET /v1/llms + POST /v1/magic_prompt — Ideogram-4 prompt generator.
//!
//! `/v1/llms` lists local GGUF LLMs (for the Settings "prompt generator" selector).
//! `/v1/magic_prompt {idea, aspect, model}` expands a short natural-language idea
//! into the structured Ideogram-4 caption JSON, by shelling out to
//! `scripts/magic_prompt.sh`, which spins up an EPHEMERAL llama-server (GPU,
//! spawn→generate→kill) so the LLM never holds VRAM while the image worker runs.
//! The system prompt is Ideogram's own magic_prompt v1.txt.

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::AppState;

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("repository root")
        .to_path_buf()
}

fn llm_root() -> PathBuf {
    std::env::var_os("SERENITY_LLM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("SERENITY_MODEL_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join(".serenity/models")
                })
                .join("llms")
        })
}

fn mojo_library_path(root: &Path) -> std::ffi::OsString {
    let mut paths = vec![root.join(".pixi/envs/default/lib")];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).unwrap_or_default()
}

/// GET /v1/llms — `{"llms":[{id,name,path,size}]}` of local GGUF models.
pub async fn get_llms(State(_st): State<AppState>) -> Response {
    let mut out: Vec<Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(llm_root()) {
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
        a["name"]
            .as_str()
            .unwrap_or("")
            .cmp(b["name"].as_str().unwrap_or(""))
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
pub async fn post_magic_prompt(State(st): State<AppState>, body: String) -> Response {
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
    // Cross-path single-GPU lease (audit L3): the LLM subprocess must not
    // co-run with a generate/video/caption job on a 16GB card. Held (RAII)
    // across the blocking spawn below; 409 if the GPU is busy.
    let gpu_tag = crate::gpu_lock::next_tag("magic");
    let _gpu = match crate::gpu_lock::try_acquire(&st.gpu_owner, "magic", &gpu_tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                Json(crate::gpu_lock::gpu_busy_conflict_report("magic", &cur)),
            )
                .into_response();
        }
    };
    // Shell the ephemeral-llama-server script (blocking; model load + gen ~15-40s).
    let out = std::process::Command::new("bash")
        .arg(repository_root().join("serenity-server/scripts/magic_prompt.sh"))
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

/// Pure-Mojo magic prompt: shells output/bin/ideogram4_magic (Qwen3-8B +
/// disk prefix cache, ~23s warm incl. model load; first-ever run builds the
/// ~1GB prefix cache once, ~4.5min). Parses the JSON between the
/// "=== MAGIC PROMPT JSON ===" / "=== END ===" markers.
fn run_mojo_magic(idea: &str, aspect: &str) -> Result<String, String> {
    let root = repository_root();
    let mojo_magic_bin = root.join("output/bin/ideogram4_magic");
    if !mojo_magic_bin.exists() {
        return Err(format!(
            "missing {}; run the ideogram4_magic build",
            mojo_magic_bin.display()
        ));
    }
    let out = std::process::Command::new(&mojo_magic_bin)
        .arg(idea)
        .arg(aspect)
        .env("LD_LIBRARY_PATH", mojo_library_path(&root))
        .output()
        .map_err(|e| format!("spawn failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "mojo magic failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let start = stdout.find("=== MAGIC PROMPT JSON ===");
    let end = stdout.find("=== END ===");
    match (start, end) {
        (Some(a), Some(b)) if b > a => {
            let caption = stdout[a + "=== MAGIC PROMPT JSON ===".len()..b]
                .trim()
                .to_string();
            if caption.is_empty() {
                Err("empty caption".to_string())
            } else {
                Ok(caption)
            }
        }
        _ => Err(format!(
            "no caption markers in output; tail: {}",
            &stdout[stdout.len().saturating_sub(200)..]
        )),
    }
}

/// POST /enhance_prompt {prompt, arch?, aspect?} -> {"prompt": "<enhanced>"}.
/// Simple-mode Enhance button (simple.js). Backend = the pure-Mojo magic
/// prompt (structured-JSON caption — ideogram-schema; other arch templates
/// later). Falls back to a clear error the frontend's local enhancer handles.
pub async fn post_enhance_prompt(State(st): State<AppState>, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return err(StatusCode::BAD_REQUEST, "prompt required");
    }
    let aspect = v["aspect"].as_str().unwrap_or("1:1").to_string();
    // Cross-path single-GPU lease (audit L3): the LLM subprocess must not
    // co-run with a generate/video/caption job on a 16GB card. Held (RAII)
    // across the blocking spawn below; 409 if the GPU is busy.
    let gpu_tag = crate::gpu_lock::next_tag("magic");
    let _gpu = match crate::gpu_lock::try_acquire(&st.gpu_owner, "magic", &gpu_tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                Json(crate::gpu_lock::gpu_busy_conflict_report("magic", &cur)),
            )
                .into_response();
        }
    };
    match run_mojo_magic(&prompt, &aspect) {
        Ok(caption) => Json(json!({ "prompt": caption, "engine": "mojo-magic" })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e),
    }
}
