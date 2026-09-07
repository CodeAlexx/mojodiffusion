//! `/v1/h3/projects` — server-backed movie library for H3 Studio.
//!
//! Scoped ENTIRELY to `<out_dir>/movies/<id>/project.json`, the same per-movie
//! folders the H3 render pipeline writes takes into. This is what makes a movie
//! (and its rendered shots) visible in ANY browser instead of trapped in one
//! browser's localStorage. It never touches the Generate gallery, History, or
//! Queue — a movie's data stays inside its own folder.

use axum::{
    extract::{Path, State},
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{json, Value};

use crate::AppState;

fn movies_root(st: &AppState) -> std::path::PathBuf {
    st.out_dir.join("movies")
}

/// Keep only filesystem-safe id segments (matches the render pipeline's rule).
fn safe_id(id: &str) -> String {
    id.trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('.')
        .to_string()
}

fn json_ok(v: Value) -> Response {
    (
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()),
    )
        .into_response()
}

/// GET /v1/h3/projects — list every movie on disk as `{id, project}` pairs.
pub async fn get_projects(State(st): State<AppState>) -> Response {
    let root = movies_root(&st);
    let mut projects: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let id = entry.file_name().to_string_lossy().to_string();
            let pj = entry.path().join("project.json");
            if let Ok(text) = std::fs::read_to_string(&pj) {
                if let Ok(value) = serde_json::from_str::<Value>(&text) {
                    projects.push(json!({ "id": id, "project": value }));
                }
            }
        }
    }
    json_ok(json!({ "schema": "serenity.h3.library.v1", "projects": projects }))
}

fn shot_has_take(shot: &Value) -> bool {
    shot.get("take_output_paths")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .any(|p| p.as_str().map(|s| !s.is_empty()).unwrap_or(false))
        })
        .unwrap_or(false)
}

/// Never let an incoming (possibly stale) project wipe takes already recorded on
/// disk: for each incoming shot that carries no take, copy the take fields from
/// the matching existing shot (matched by `id`, else by position).
fn merge_preserve_takes(incoming: &mut Value, existing: &Value) {
    let ex_shots = match existing.get("shots").and_then(|v| v.as_array()) {
        Some(s) => s.clone(),
        None => return,
    };
    let in_shots = match incoming.get_mut("shots").and_then(|v| v.as_array_mut()) {
        Some(s) => s,
        None => return,
    };
    for (idx, ishot) in in_shots.iter_mut().enumerate() {
        if shot_has_take(ishot) {
            continue;
        }
        let iid = ishot.get("id").cloned();
        let ex = ex_shots
            .iter()
            .find(|e| iid.is_some() && e.get("id") == iid.as_ref())
            .or_else(|| ex_shots.get(idx));
        if let (Some(ex), Some(obj)) = (ex, ishot.as_object_mut()) {
            if shot_has_take(ex) {
                for k in [
                    "take_job_ids",
                    "take_states",
                    "take_output_paths",
                    "selected_take",
                    "status",
                    "output_path",
                ] {
                    if let Some(val) = ex.get(k) {
                        obj.insert(k.to_string(), val.clone());
                    }
                }
            }
        }
    }
}

/// PUT /v1/h3/projects/:id — persist one movie's project.json under its folder.
pub async fn put_project(
    State(st): State<AppState>,
    Path(id): Path<String>,
    body: String,
) -> Response {
    let safe = safe_id(&id);
    if safe.is_empty() || safe.contains("..") {
        return (StatusCode::BAD_REQUEST, "invalid project id").into_response();
    }
    let mut value: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, format!("invalid project json: {error}"))
                .into_response()
        }
    };
    let dir = movies_root(&st).join(&safe);
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create movie folder: {error}"),
        )
            .into_response();
    }
    let path = dir.join("project.json");
    // Preserve already-recorded takes against a stale-tab overwrite.
    if let Ok(text) = std::fs::read_to_string(&path) {
        if let Ok(existing) = serde_json::from_str::<Value>(&text) {
            merge_preserve_takes(&mut value, &existing);
        }
    }
    match serde_json::to_vec_pretty(&value) {
        Ok(bytes) => match std::fs::write(&path, bytes) {
            Ok(_) => json_ok(json!({ "ok": true, "id": safe })),
            Err(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("cannot write project.json: {error}"),
            )
                .into_response(),
        },
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot serialize project: {error}"),
        )
            .into_response(),
    }
}

/// DELETE /v1/h3/projects/:id — remove one movie folder (and its takes) entirely.
pub async fn delete_project(State(st): State<AppState>, Path(id): Path<String>) -> Response {
    let safe = safe_id(&id);
    if safe.is_empty() || safe.contains("..") {
        return (StatusCode::BAD_REQUEST, "invalid project id").into_response();
    }
    let dir = movies_root(&st).join(&safe);
    let _ = std::fs::remove_dir_all(&dir);
    json_ok(json!({ "ok": true, "id": safe }))
}
