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

// ---------------------------------------------------------------------------
// Movie assembly
//
// H3 Studio could plan, render and export a delivery MANIFEST, but nothing in
// either stack ever turned the ordered takes into a film -- the existing
// THE_WALLET_movie.mp4 was produced by a hand-run ffmpeg. This is that step,
// inside the app: normalise each selected take to one container spec, then
// stream-copy them together.
//
// Normalisation is not optional. The takes are individually encoded and a raw
// concat of differently-shaped streams desynchronises audio; re-encoding each
// segment to a common spec first (and only then copying) is what the manual
// run did, and it is why the norm_NN.mp4 intermediates exist next to the film.

/// Container spec every segment is normalised to before concatenation.
const MOVIE_FPS: &str = "24";
const MOVIE_AUDIO_RATE: &str = "48000";

fn take_path_for_shot(movie_dir: &std::path::Path, out_root: &std::path::Path, shot: &Value) -> Option<std::path::PathBuf> {
    let selected = shot.get("selected_take").and_then(Value::as_i64).unwrap_or(-1);
    let paths = shot.get("take_output_paths").and_then(Value::as_array)?;
    let raw = if selected >= 0 {
        paths.get(selected as usize).and_then(Value::as_str)
    } else {
        None
    }
    .or_else(|| shot.get("output_path").and_then(Value::as_str))
    .map(str::trim)
    .filter(|value| !value.is_empty())?;

    // Stored paths are the browser's URLs ("/out/<rel>" or
    // "/out/movies/<id>/video-NNNN/video.mp4"). Resolve against this server's
    // own roots and never outside them.
    let relative = raw.strip_prefix("/out/").unwrap_or(raw.trim_start_matches('/'));
    for candidate in [out_root.join(relative), movie_dir.join(relative)] {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    // A bare job id recorded without a path still resolves inside this movie.
    if let Some(job) = shot.get("take_job_ids").and_then(Value::as_array) {
        if selected >= 0 {
            if let Some(id) = job.get(selected as usize).and_then(Value::as_str) {
                for candidate in [movie_dir.join(id).join("video.mp4"), out_root.join(id).join("video.mp4")] {
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }
        }
    }
    None
}

fn movie_file_name(title: &str) -> String {
    let cleaned: String = title
        .trim()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
        .collect();
    let trimmed = cleaned.trim_matches('_').to_string();
    format!("{}_movie.mp4", if trimmed.is_empty() { "UNTITLED".to_string() } else { trimmed })
}

/// POST /v1/h3/projects/:id/movie — assemble the ordered selected takes into
/// one delivery file next to the project, and report exactly what was used.
pub async fn post_assemble_movie(State(st): State<AppState>, Path(id): Path<String>) -> Response {
    let safe = safe_id(&id);
    if safe.is_empty() || safe.contains("..") {
        return (StatusCode::BAD_REQUEST, "invalid project id").into_response();
    }
    let movie_dir = movies_root(&st).join(&safe);
    let project_path = movie_dir.join("project.json");
    let project: Value = match std::fs::read_to_string(&project_path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
    {
        Some(value) => value,
        None => {
            return (
                StatusCode::NOT_FOUND,
                "no project.json for this movie; save the project first",
            )
                .into_response()
        }
    };
    let shots = match project.get("shots").and_then(Value::as_array) {
        Some(shots) if !shots.is_empty() => shots.clone(),
        _ => return (StatusCode::BAD_REQUEST, "project has no shots").into_response(),
    };

    let mut segments: Vec<std::path::PathBuf> = Vec::new();
    let mut used: Vec<Value> = Vec::new();
    let mut skipped: Vec<Value> = Vec::new();
    for (index, shot) in shots.iter().enumerate() {
        let title = shot.get("title").and_then(Value::as_str).unwrap_or("").to_string();
        match take_path_for_shot(&movie_dir, &st.out_dir, shot) {
            Some(source) => {
                let normalised = movie_dir.join(format!("norm_{index:02}.mp4"));
                let status = std::process::Command::new("ffmpeg")
                    .args(["-y", "-v", "error", "-i"])
                    .arg(&source)
                    .args([
                        "-r", MOVIE_FPS,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
                        "-c:a", "aac", "-ar", MOVIE_AUDIO_RATE, "-ac", "2",
                    ])
                    .arg(&normalised)
                    .status();
                match status {
                    Ok(code) if code.success() && normalised.is_file() => {
                        segments.push(normalised);
                        used.push(json!({ "index": index, "title": title, "source": source.to_string_lossy() }));
                    }
                    _ => skipped.push(json!({ "index": index, "title": title, "reason": "segment could not be normalised" })),
                }
            }
            None => skipped.push(json!({ "index": index, "title": title, "reason": "no rendered take on disk" })),
        }
    }
    if segments.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            "no shot in this movie has a rendered take on disk",
        )
            .into_response();
    }

    let list_path = movie_dir.join("concat.txt");
    let list = segments
        .iter()
        .map(|path| format!("file '{}'\n", path.to_string_lossy()))
        .collect::<String>();
    if let Err(error) = std::fs::write(&list_path, list) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot write concat list: {error}"),
        )
            .into_response();
    }
    let title = project.get("title").and_then(Value::as_str).unwrap_or("Untitled");
    let movie_path = movie_dir.join(movie_file_name(title));
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "concat", "-safe", "0", "-i"])
        .arg(&list_path)
        .args(["-c", "copy"])
        .arg(&movie_path)
        .status();
    match status {
        Ok(code) if code.success() && movie_path.is_file() => {
            let bytes = std::fs::metadata(&movie_path).map(|m| m.len()).unwrap_or(0);
            json_ok(json!({
                "schema": "serenity.h3.movie.assembled.v1",
                "ok": true,
                "id": safe,
                "title": title,
                "path": movie_path.to_string_lossy(),
                "url": format!("/out/movies/{safe}/{}", movie_file_name(title)),
                "bytes": bytes,
                "shots_used": used,
                "shots_skipped": skipped,
            }))
        }
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "ffmpeg could not concatenate the normalised segments",
        )
            .into_response(),
    }
}
