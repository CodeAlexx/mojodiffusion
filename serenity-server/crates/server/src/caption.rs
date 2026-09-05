//! POST /v1/caption — one-shot image captioner.
//!
//! Shells the pure-Mojo Qwen3-VL captioner (`output/bin/qwen3vl_caption`), which
//! takes `<image> [prompt] [max_new]` and prints the caption between the
//! `=== CAPTION ===` / `=== END ===` markers. This is GPU-heavy and one-shot
//! (~9-120s/image by bucket; the binary loads the model per call), so the request
//! runs synchronously — the caller (a context-menu action) awaits it.
//!
//! Mirrors magic.rs's binary-shelling pattern (LD_LIBRARY_PATH env + marker parse).

use std::path::{Path, PathBuf};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::AppState;
use crate::blocking::CommandDeadline;

const CAPTION_START: &str = "=== CAPTION ===";
const CAPTION_END: &str = "=== END ===";

fn repository_root() -> PathBuf {
    crate::repository_root_path()
}

fn mojo_library_path(root: &Path) -> std::ffi::OsString {
    let mut paths = vec![
        root.join(".pixi/envs/default/lib"),
        root.join("serenitymojo/ops/cshim/lib"),
    ];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).unwrap_or_default()
}

fn err(code: StatusCode, msg: &str) -> Response {
    (code, axum::Json(json!({ "error": msg }))).into_response()
}

/// The generation lease excludes an active image job, but an idle Mojo worker
/// can still retain almost the entire GPU (Krea2 measures about 20 GiB after a
/// 1024px render). Reap it before the one-shot vision model starts so automatic
/// Canvas captioning does not fail with an empty CUDA/OOM error.
fn evict_idle_image_worker(st: &AppState) -> Result<(), Response> {
    let (evict_tx, evict_rx) = std::sync::mpsc::channel();
    if st.ctl.send(crate::DriverCtl::EvictIdle(evict_tx)).is_err() {
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "image worker driver unavailable before captioning",
        ));
    }
    match evict_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(true) => Ok(()),
        Ok(false) => Err(err(
            StatusCode::CONFLICT,
            "image worker became active before captioning",
        )),
        Err(_) => Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "timed out evicting idle image worker before captioning",
        )),
    }
}

/// Validate an image path against the server output and optional dataset root.
fn validate_image_path(p: &str, out_dir: &Path) -> Result<(), String> {
    if p.is_empty() {
        return Err("image_path is required".into());
    }
    let resolved = std::fs::canonicalize(p).map_err(|_| format!("image not found: {p}"))?;
    if !resolved.is_file() {
        return Err(format!("image not found: {p}"));
    }
    let mut roots = Vec::new();
    if let Ok(root) = std::fs::canonicalize(out_dir) {
        roots.push(root);
    }
    if let Some(root) =
        std::env::var_os("SERENITY_DATA_ROOT").and_then(|p| std::fs::canonicalize(p).ok())
    {
        roots.push(root);
    }
    if !roots.iter().any(|root| resolved.starts_with(root)) {
        return Err("image_path is outside server output and SERENITY_DATA_ROOT".into());
    }
    Ok(())
}

/// POST /v1/caption `{ image_path|gallery_id, prompt?, max_new? }` -> `{ "caption": <text> }`.
///
/// `gallery_id` (a `job-XXXX` id) resolves to `<out_dir>/<id>.png`; otherwise
/// `image_path` (absolute) is used. GPU/one-shot: runs synchronously.
/// Async entry: the body is pure blocking work (a GPU subprocess held for
/// seconds to minutes), so it runs on the blocking pool rather than parking a
/// runtime worker, and a panic inside it answers 500 instead of resetting the
/// connection. See blocking.rs.
pub async fn post_caption(State(st): State<AppState>, body: String) -> Response {
    crate::blocking::offload(move || post_caption_blocking(st, body)).await
}

fn post_caption_blocking(st: AppState, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };

    // Resolve the target image: gallery_id (job-XXXX) → <out_dir>/<id>.png, else image_path.
    let image_path = if let Some(gid) = v["gallery_id"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if !gid.starts_with("job-") || gid.contains('/') || gid.contains("..") {
            return err(StatusCode::BAD_REQUEST, "invalid gallery_id");
        }
        st.out_dir
            .join(format!("{gid}.png"))
            .to_string_lossy()
            .into_owned()
    } else {
        v["image_path"].as_str().unwrap_or("").trim().to_string()
    };

    if let Err(e) = validate_image_path(&image_path, st.out_dir.as_path()) {
        return err(StatusCode::BAD_REQUEST, &e);
    }

    let root = repository_root();
    let caption_bin = root.join("output/bin/qwen3vl_caption");
    if !caption_bin.exists() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!(
                "missing {}; build the qwen3vl_caption binary",
                caption_bin.display()
            ),
        );
    }

    // Cross-path single-GPU lease (audit L3): the captioner is a GPU subprocess;
    // it must not co-run with a generate/video/magic job on a 16GB card. Held
    // (RAII) across the spawn+wait below; 409 if the GPU is busy.
    let gpu_tag = crate::gpu_lock::next_tag("caption");
    let _gpu = match crate::gpu_lock::try_acquire(&st.gpu_owner, "caption", &gpu_tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                axum::Json(crate::gpu_lock::gpu_busy_conflict_report("caption", &cur)),
            )
                .into_response();
        }
    };
    if let Err(response) = evict_idle_image_worker(&st) {
        return response;
    }

    let prompt = v["prompt"]
        .as_str()
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    // Clamp max_new to a sane range so a bad client can't drive a runaway decode.
    let max_new = v["max_new"].as_u64().map(|n| n.clamp(1, 1024));

    let mut cmd = std::process::Command::new(&caption_bin);
    cmd.arg(&image_path);
    // The binary is positional: <image> [prompt] [max_new]. Only pass a prompt if
    // non-empty; only pass max_new when a prompt is also present (positional).
    if !prompt.is_empty() {
        cmd.arg(&prompt);
        if let Some(n) = max_new {
            cmd.arg(n.to_string());
        }
    }
    cmd.env("LD_LIBRARY_PATH", mojo_library_path(&root));

    let out = match cmd.output_with_deadline(crate::blocking::subprocess_deadline()) {
        Ok(o) => o,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("subprocess failed: {e}"),
            );
        }
    };
    if !out.status.success() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!(
                "caption failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    match extract_caption(&stdout) {
        Some(caption) => axum::Json(json!({ "caption": caption })).into_response(),
        None => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!(
                "no caption markers in output; tail: {}",
                &stdout[stdout.len().saturating_sub(200)..]
            ),
        ),
    }
}

/// POST /v1/h3/director `{ system_prompt, prompt, image_path?, max_new? }` ->
/// `{ "text": <model output>, "json": <parsed object or null>, "elapsed_ms": n }`.
///
/// Runs the H3 captioner (pure-Mojo Qwen3-VL, `output/bin/qwen3vl_caption`) in
/// its flag mode: text-only unless `image_path` is given. Used by the H3 Studio
/// Director pass with the `serenity.h3.caption.v2` system prompt. Same GPU
/// lease / idle-image-worker eviction as `/v1/caption`; synchronous.
/// Async entry: the body is pure blocking work (a GPU subprocess held for
/// seconds to minutes), so it runs on the blocking pool rather than parking a
/// runtime worker, and a panic inside it answers 500 instead of resetting the
/// connection. See blocking.rs.
pub async fn post_h3_director(State(st): State<AppState>, body: String) -> Response {
    crate::blocking::offload(move || post_h3_director_blocking(st, body)).await
}

fn post_h3_director_blocking(st: AppState, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let system_prompt = v["system_prompt"].as_str().unwrap_or("").trim().to_string();
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return err(StatusCode::BAD_REQUEST, "prompt is required");
    }
    let image_path = v["image_path"].as_str().unwrap_or("").trim().to_string();
    if !image_path.is_empty() {
        if let Err(e) = validate_image_path(&image_path, st.out_dir.as_path()) {
            return err(StatusCode::BAD_REQUEST, &e);
        }
    }
    let max_new = v["max_new"].as_u64().unwrap_or(1500).clamp(1, 4096);
    let root = repository_root();
    let caption_bin = root.join("output/bin/qwen3vl_caption");
    if !caption_bin.exists() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("missing {}; build the qwen3vl_caption binary", caption_bin.display()),
        );
    }
    // Stage the prompts as files (long system prompts, exact bytes).
    let stage_dir = st.out_dir.join(".h3_director");
    if let Err(e) = std::fs::create_dir_all(&stage_dir) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, &format!("stage dir: {e}"));
    }
    let tag = crate::gpu_lock::next_tag("h3_director");
    let prompt_path = stage_dir.join(format!("{tag}.prompt.txt"));
    let system_path = stage_dir.join(format!("{tag}.system.txt"));
    if let Err(e) = std::fs::write(&prompt_path, &prompt) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, &format!("stage prompt: {e}"));
    }
    if !system_prompt.is_empty() {
        if let Err(e) = std::fs::write(&system_path, &system_prompt) {
            return err(StatusCode::INTERNAL_SERVER_ERROR, &format!("stage system: {e}"));
        }
    }
    let _gpu = match crate::gpu_lock::try_acquire(&st.gpu_owner, "h3_director", &tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                axum::Json(crate::gpu_lock::gpu_busy_conflict_report("h3_director", &cur)),
            )
                .into_response();
        }
    };
    if let Err(response) = evict_idle_image_worker(&st) {
        return response;
    }
    let started = std::time::Instant::now();
    let mut cmd = std::process::Command::new(&caption_bin);
    cmd.arg(format!("--prompt-file={}", prompt_path.display()));
    if !system_prompt.is_empty() {
        cmd.arg(format!("--system-file={}", system_path.display()));
    }
    if !image_path.is_empty() {
        cmd.arg(format!("--image={image_path}"));
    }
    cmd.arg(format!("--max-new={max_new}"));
    cmd.env("LD_LIBRARY_PATH", mojo_library_path(&root));
    let out = match cmd.output_with_deadline(crate::blocking::subprocess_deadline()) {
        Ok(o) => o,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, &format!("subprocess failed: {e}")),
    };
    if !out.status.success() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!(
                "director pass failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let text = match extract_caption(&stdout) {
        Some(t) => t,
        None => return err(StatusCode::INTERNAL_SERVER_ERROR, "director pass produced no output"),
    };
    let parsed = parse_director_json(&text);
    (
        StatusCode::OK,
        axum::Json(json!({
            "schema": "serenity.h3.director.run.v1",
            "text": text,
            "json": parsed,
            "elapsed_ms": started.elapsed().as_millis() as u64,
            "max_new": max_new,
            "image_path": image_path,
        })),
    )
        .into_response()
}

/// Best-effort JSON extraction from a model reply: strips Markdown fences and
/// falls back to the outermost `{...}` span.
fn parse_director_json(text: &str) -> Value {
    let trimmed = text.trim();
    let candidates: [&str; 1] = [trimmed];
    for c in candidates {
        let mut body = c;
        if let Some(rest) = body.strip_prefix("```json") { body = rest; }
        else if let Some(rest) = body.strip_prefix("```") { body = rest; }
        if let Some(rest) = body.strip_suffix("```") { body = rest; }
        if let Ok(v) = serde_json::from_str::<Value>(body.trim()) {
            if v.is_object() { return v; }
        }
        if let (Some(start), Some(end)) = (body.find('{'), body.rfind('}')) {
            if end > start {
                if let Ok(v) = serde_json::from_str::<Value>(&body[start..=end]) {
                    if v.is_object() { return v; }
                }
            }
        }
    }
    Value::Null
}

/// Pull the text between the `=== CAPTION ===` and `=== END ===` marker lines.
fn extract_caption(stdout: &str) -> Option<String> {
    let start = stdout.find(CAPTION_START)?;
    let after = start + CAPTION_START.len();
    let end_rel = stdout[after..].find(CAPTION_END)?;
    let text = stdout[after..after + end_rel].trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_between_markers() {
        let s = "[caption] bucket grid: 32 x 32\n=== CAPTION ===\na red car on a road\n=== END ===\ntiming: 8.6s\n";
        assert_eq!(extract_caption(s).as_deref(), Some("a red car on a road"));
    }

    #[test]
    fn director_json_strips_fences_and_prose() {
        let v = parse_director_json("```json\n{\"a\": 1}\n```");
        assert_eq!(v["a"], 1);
        let v = parse_director_json("Here you go: {\"shots\": []} thanks");
        assert!(v["shots"].is_array());
        assert!(parse_director_json("no json at all").is_null());
    }

    #[test]
    fn extract_missing_markers_is_none() {
        assert!(extract_caption("no markers here").is_none());
        assert!(extract_caption("=== CAPTION ===\n\n=== END ===").is_none());
    }

    #[test]
    fn validate_rejects_traversal_and_outside() {
        let root = Path::new("/portable/output");
        assert!(validate_image_path("", root).is_err());
        assert!(validate_image_path("/portable/../etc/passwd", root).is_err());
        assert!(validate_image_path("/etc/passwd", root).is_err());
        assert!(validate_image_path("/portable/does-not-exist-xyz.png", root).is_err());
    }
}
