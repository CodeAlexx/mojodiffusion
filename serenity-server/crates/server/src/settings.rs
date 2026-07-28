//! Settings / gallery completeness routes the Konva frontend calls but the server
//! didn't yet answer (measured 404s, 2026-07-11). All cheap + non-GPU.
//!
//!   GET  /templates                    — workflow template list (bare array)
//!   GET  /folder_paths                  — watched model/output dirs, grouped by category
//!   POST /folder_paths/add {path}       — append an extra watched dir (persisted)
//!   GET  /stagehand_settings            — offload knobs blob
//!   POST /stagehand_settings {...}      — persist offload knobs blob
//!   POST /video_edit/resolve_view_path  — {filename,subfolder,type} → {path} (mirrors /view)
//!   GET  /output_files                  — files directly under out_dir (name/size/mtime)
//!   DELETE /output_files/{name}         — delete one output file (guarded)
//!   POST /open_output_dir               — {path} of the output dir (no shelling)
//!
//! folder_paths + stagehand knobs persist in `<out_dir>/state/ui_settings.json`.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use axum::extract::{Path as AxPath, Query, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::AppState;

/// Canonical serenity model roots (same as models.rs's scan roots). Reported under
/// /folder_paths so the settings "Model Search Directories" list is non-empty.
fn model_root() -> PathBuf {
    std::env::var_os("SERENITY_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".serenity/models")
        })
}

fn json_ok(doc: &Value) -> Response {
    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}

fn err(status: StatusCode, msg: &str) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&json!({ "error": msg })).unwrap(),
    )
        .into_response()
}

// ── persisted UI settings blob (`<out_dir>/state/ui_settings.json`) ──────────────

fn settings_path(out: &Path) -> std::path::PathBuf {
    out.join("state").join("ui_settings.json")
}

fn load_settings(out: &Path) -> Value {
    std::fs::read_to_string(settings_path(out))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({ "folder_paths": [], "stagehand": {} }))
}

fn save_settings(out: &Path, doc: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(out.join("state"))?;
    std::fs::write(settings_path(out), serde_json::to_string(doc).unwrap())
}

/// The persisted extra watched dirs (["..."]); empty when unset.
fn extra_folder_paths(settings: &Value) -> Vec<String> {
    settings
        .get("folder_paths")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn configured_extra_model_roots(out: &Path) -> Vec<PathBuf> {
    extra_folder_paths(&load_settings(out))
        .into_iter()
        .map(PathBuf::from)
        .collect()
}

// ── GET /templates ───────────────────────────────────────────────────────────────

const BUILTIN_TEMPLATES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../canvas/workflows");

fn collect_template_dir(entries: &mut BTreeMap<String, Value>, dir: &Path, url_prefix: &str) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in rd.flatten() {
        let path = ent.path();
        let filename = ent.file_name().to_string_lossy().into_owned();
        if !path.is_file() || !filename.to_ascii_lowercase().ends_with(".json") {
            continue;
        }
        let name = filename
            .strip_suffix(".json")
            .unwrap_or(&filename)
            .to_string();
        entries.insert(
            name.clone(),
            json!({
                "name": name,
                "file": filename,
                "path": path.to_string_lossy(),
                "url": format!("{url_prefix}/{filename}"),
            }),
        );
    }
}

fn template_entries(builtin_dir: &Path, user_dir: &Path) -> Vec<Value> {
    let mut entries = BTreeMap::new();
    collect_template_dir(&mut entries, builtin_dir, "/workflows");
    // A user-saved file with the same stem intentionally overrides its bundled
    // counterpart without hiding any other built-in template.
    collect_template_dir(&mut entries, user_dir, "/out/templates");
    entries.into_values().collect()
}

/// Workflow templates bundled with the web canvas plus user templates under
/// `<out_dir>/templates/`. Returns a BARE ARRAY because shell.js consumes it
/// directly. The bundled files remain browser-fetchable through the canvas
/// fallback; user files use the guarded `/out/templates/*` route.
pub async fn get_templates(State(st): State<AppState>) -> Response {
    let user_dir = st.out_dir.join("templates");
    let _ = std::fs::create_dir_all(&user_dir);
    let out = template_entries(Path::new(BUILTIN_TEMPLATES_DIR), &user_dir);
    json_ok(&Value::Array(out))
}

// ── /folder_paths ────────────────────────────────────────────────────────────────

/// GET /folder_paths — `{ <category>: { paths: [...] } }`. The frontend iterates
/// `Object.keys(data)` and reads `data[cat].paths`.
pub async fn get_folder_paths(State(st): State<AppState>) -> Response {
    let settings = load_settings(st.out_dir.as_path());
    let extra = extra_folder_paths(&settings);
    let out = st.out_dir.to_string_lossy().into_owned();
    let models = model_root();
    let checkpoints = models.join("checkpoints").to_string_lossy().into_owned();
    let loras = models.join("loras").to_string_lossy().into_owned();
    let doc = json!({
        "checkpoints": { "paths": [checkpoints] },
        "loras": { "paths": [loras] },
        "output": { "paths": [out] },
        "extra": { "paths": extra },
    });
    json_ok(&doc)
}

/// POST /folder_paths/add `{path}` — append an extra watched dir and persist.
/// Returns `{added, paths}`. Frontend only checks for a `.error` key.
pub async fn post_folder_paths_add(State(st): State<AppState>, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let path = v["path"].as_str().unwrap_or("").trim().to_string();
    if path.is_empty() {
        return err(StatusCode::BAD_REQUEST, "'path' is required");
    }
    let candidate = PathBuf::from(&path);
    if !candidate.is_dir() {
        return err(
            StatusCode::BAD_REQUEST,
            "'path' must be an existing directory",
        );
    }
    let path = std::fs::canonicalize(&candidate)
        .unwrap_or(candidate)
        .to_string_lossy()
        .into_owned();
    let out = st.out_dir.as_path();
    let mut settings = load_settings(out);
    let mut paths = extra_folder_paths(&settings);
    if !paths.iter().any(|p| p == &path) {
        paths.push(path.clone());
    }
    if let Some(m) = settings.as_object_mut() {
        m.insert("folder_paths".into(), json!(paths));
    }
    if save_settings(out, &settings).is_err() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "cannot persist folder paths",
        );
    }
    crate::models::set_extra_model_roots(paths.iter().map(PathBuf::from).collect());
    json_ok(&json!({ "added": path, "paths": paths }))
}

// ── /stagehand_settings ──────────────────────────────────────────────────────────

/// GET /stagehand_settings — the persisted offload knobs blob (`{}` when unset).
pub async fn get_stagehand_settings(State(st): State<AppState>) -> Response {
    let settings = load_settings(st.out_dir.as_path());
    let blob = settings
        .get("stagehand")
        .cloned()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({}));
    json_ok(&blob)
}

/// POST /stagehand_settings — persist the offload knobs blob (the whole JSON object
/// body is stored under `stagehand`). Returns the stored blob.
pub async fn post_stagehand_settings(State(st): State<AppState>, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) if v.is_object() => v,
        Ok(_) => return err(StatusCode::BAD_REQUEST, "body must be a JSON object"),
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let out = st.out_dir.as_path();
    let mut settings = load_settings(out);
    if let Some(m) = settings.as_object_mut() {
        m.insert("stagehand".into(), v.clone());
    }
    if save_settings(out, &settings).is_err() {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "cannot persist stagehand settings",
        );
    }
    json_ok(&v)
}

// ── /video_edit/resolve_view_path ────────────────────────────────────────────────

/// Resolve a `{filename, subfolder, type}` (the /view query shape the frontend
/// already holds) to an absolute on-disk path under out_dir. Returns `{path}` when
/// the file exists, else 404. Mirrors comfy::get_view's path assembly + `..` guard.
fn resolve_under_out(out: &Path, filename: &str, subfolder: &str) -> Option<String> {
    if filename.is_empty() || filename.contains("..") || subfolder.contains("..") {
        return None;
    }
    let mut p = out.to_path_buf();
    for seg in subfolder.split('/').filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    for seg in filename.split('/').filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    if p.is_file() {
        Some(p.to_string_lossy().into_owned())
    } else {
        None
    }
}

/// POST /video_edit/resolve_view_path `{filename, subfolder?, type?}` → `{path}`.
pub async fn post_resolve_view_path(State(st): State<AppState>, body: String) -> Response {
    let v: Value = match serde_json::from_str::<Value>(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let filename = v["filename"].as_str().unwrap_or("").to_string();
    let subfolder = v["subfolder"].as_str().unwrap_or("").to_string();
    match resolve_under_out(st.out_dir.as_path(), &filename, &subfolder) {
        Some(path) => json_ok(&json!({ "path": path })),
        None => err(StatusCode::NOT_FOUND, "view path not found"),
    }
}

/// GET /video_edit/resolve_view_path?filename=&subfolder= — query variant.
pub async fn get_resolve_view_path(
    State(st): State<AppState>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let filename = q.get("filename").cloned().unwrap_or_default();
    let subfolder = q.get("subfolder").cloned().unwrap_or_default();
    match resolve_under_out(st.out_dir.as_path(), &filename, &subfolder) {
        Some(path) => json_ok(&json!({ "path": path })),
        None => err(StatusCode::NOT_FOUND, "view path not found"),
    }
}

// ── /output_files ────────────────────────────────────────────────────────────────

/// GET /output_files — files DIRECTLY under out_dir (skips subdirs + dotfiles).
/// BARE ARRAY of `{name, size_bytes, modified}` (mtime seconds); the frontend reads
/// `files.length`, `f.name`, `f.size_bytes`, `f.modified`.
pub async fn get_output_files(State(st): State<AppState>) -> Response {
    let out = st.out_dir.as_path();
    let mut files: Vec<Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(out) {
        for ent in rd.flatten() {
            let name = ent.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let meta = match ent.metadata() {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            files.push(json!({
                "name": name,
                "size_bytes": meta.len(),
                "modified": modified,
            }));
        }
    }
    // newest first (largest mtime)
    files.sort_by(|a, b| {
        b["modified"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&a["modified"].as_u64().unwrap_or(0))
    });
    json_ok(&Value::Array(files))
}

/// DELETE /output_files/{name} — delete a single file directly under out_dir.
/// Guarded: no `/`, no `..`, must be an existing regular file in out_dir. The UI
/// wires this to the per-file trash button.
pub async fn delete_output_file(
    State(st): State<AppState>,
    AxPath(name): AxPath<String>,
) -> Response {
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return err(StatusCode::BAD_REQUEST, "invalid file name");
    }
    let p = st.out_dir.join(&name);
    match std::fs::metadata(&p) {
        Ok(m) if m.is_file() => {}
        _ => return err(StatusCode::NOT_FOUND, "output file not found"),
    }
    match std::fs::remove_file(&p) {
        Ok(_) => json_ok(&json!({ "deleted": name })),
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("delete failed: {e}"),
        ),
    }
}

/// GET|POST /open_output_dir — return `{path}` of the output dir. A server MUST NOT
/// shell `xdg-open` (headless); the frontend only needs the path (and ignores the
/// body on its POST click handler).
pub async fn open_output_dir(State(st): State<AppState>) -> Response {
    json_ok(&json!({ "path": st.out_dir.to_string_lossy() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_rejects_traversal() {
        let out = Path::new("/tmp/serenity_settings_test_root");
        assert!(resolve_under_out(out, "../etc/passwd", "").is_none());
        assert!(resolve_under_out(out, "x.png", "../..").is_none());
        assert!(resolve_under_out(out, "", "").is_none());
    }

    #[test]
    fn resolve_finds_existing_file() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("serenity_resolve_{}_{}", std::process::id(), nonce));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("job-0001.png");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(
            resolve_under_out(&dir, "job-0001.png", ""),
            Some(f.to_string_lossy().into_owned())
        );
        assert!(resolve_under_out(&dir, "nope.png", "").is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn folder_paths_roundtrip() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("serenity_fp_{}_{}", std::process::id(), nonce));
        std::fs::create_dir_all(&dir).unwrap();
        let mut settings = load_settings(&dir);
        assert!(extra_folder_paths(&settings).is_empty());
        settings
            .as_object_mut()
            .unwrap()
            .insert("folder_paths".into(), json!(["/a/b", "/c/d"]));
        save_settings(&dir, &settings).unwrap();
        let reloaded = load_settings(&dir);
        assert_eq!(extra_folder_paths(&reloaded), vec!["/a/b", "/c/d"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn template_entries_merge_builtins_and_user_overrides() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "serenity_templates_{}_{}",
            std::process::id(),
            nonce
        ));
        let builtin = root.join("builtin");
        let user = root.join("user");
        std::fs::create_dir_all(&builtin).unwrap();
        std::fs::create_dir_all(&user).unwrap();
        std::fs::write(builtin.join("alpha.json"), b"{}").unwrap();
        std::fs::write(builtin.join("beta.json"), b"{}").unwrap();
        std::fs::write(user.join("beta.json"), b"{\"user\":true}").unwrap();
        std::fs::write(user.join("notes.txt"), b"ignored").unwrap();

        let entries = template_entries(&builtin, &user);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["name"], "alpha");
        assert_eq!(entries[0]["url"], "/workflows/alpha.json");
        assert_eq!(entries[1]["name"], "beta");
        assert_eq!(entries[1]["url"], "/out/templates/beta.json");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
