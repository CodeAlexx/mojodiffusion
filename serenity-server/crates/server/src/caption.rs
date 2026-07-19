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
use serde_json::{json, Value};

use crate::AppState;

const CAPTION_START: &str = "=== CAPTION ===";
const CAPTION_END: &str = "=== END ===";

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("repository root")
        .to_path_buf()
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
pub async fn post_caption(State(st): State<AppState>, body: String) -> Response {
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
                .into_response()
        }
    };

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

    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("spawn failed: {e}"),
            )
        }
    };
    if !out.status.success() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!(
                "caption failed: {}",
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
