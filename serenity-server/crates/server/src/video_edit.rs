//! Serenity browser Video-tab backend powered by the vendored Genesis engine.
//!
//! Boundary:
//! - this module owns HTTP/project persistence and adapts the browser timeline;
//! - `genesis-web` owns the headless Rust project model and worker protocol;
//! - `genesis-gcompose` is a separate Rust/C/FFmpeg/OpenCL process;
//! - no Mojo code or native Genesis window participates in this path.

use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Multipart, Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use genesis_web::model::{
    Clip as GenesisClip, Project as GenesisProject, Subtitle as GenesisSubtitle,
    Track as GenesisTrack, TrackKind as GenesisTrackKind, Transition as GenesisTransition,
};
use image::{DynamicImage, ImageFormat, RgbaImage};
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};
use tokio::io::AsyncWriteExt;

use crate::AppState;

const GENESIS_PREVIEW_WIDTH: u32 = genesis_web::worker::PVW as u32;
const GENESIS_PREVIEW_HEIGHT: u32 = genesis_web::worker::PVH as u32;
const MAX_THUMBNAILS: usize = 120;

#[derive(Clone, serde::Serialize)]
struct ExportStatus {
    export_id: String,
    state: String,
    output_path: Option<String>,
    output_url: Option<String>,
    error: Option<String>,
}

static EXPORTS: OnceLock<Mutex<HashMap<String, ExportStatus>>> = OnceLock::new();
static GENESIS_ENV_READY: OnceLock<()> = OnceLock::new();

fn exports() -> &'static Mutex<HashMap<String, ExportStatus>> {
    EXPORTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn safe_component(raw: &str) -> Option<String> {
    let value: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect();
    if value.is_empty() || value == "." || value == ".." {
        None
    } else {
        Some(value)
    }
}

fn project_root(state: &AppState) -> PathBuf {
    state.out_dir.join("video_projects")
}

fn project_dir(state: &AppState, id: &str) -> Result<PathBuf, String> {
    let id = safe_component(id).ok_or_else(|| "invalid project id".to_string())?;
    Ok(project_root(state).join(id))
}

fn project_path(state: &AppState, id: &str) -> Result<PathBuf, String> {
    Ok(project_dir(state, id)?.join("project.json"))
}

fn read_project(state: &AppState, id: &str) -> Result<JsonValue, String> {
    let path = project_path(state, id)?;
    let bytes =
        fs::read(&path).map_err(|e| format!("read video project {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse video project {}: {e}", path.display()))
}

fn write_project(state: &AppState, id: &str, mut value: JsonValue) -> Result<JsonValue, String> {
    let dir = project_dir(state, id)?;
    fs::create_dir_all(dir.join("media"))
        .map_err(|e| format!("create video project directory: {e}"))?;
    if !value.is_object() {
        return Err("project body must be a JSON object".to_string());
    }
    value["id"] = JsonValue::String(id.to_string());
    let bytes =
        serde_json::to_vec_pretty(&value).map_err(|e| format!("serialize video project: {e}"))?;
    let final_path = dir.join("project.json");
    let temp_path = dir.join("project.json.tmp");
    fs::write(&temp_path, bytes).map_err(|e| format!("write video project: {e}"))?;
    fs::rename(&temp_path, &final_path).map_err(|e| format!("commit video project: {e}"))?;
    Ok(value)
}

fn is_legacy_demo_clip(clip: &JsonValue) -> bool {
    if clip
        .get("source_path")
        .and_then(JsonValue::as_str)
        .is_some_and(|source| !source.is_empty())
    {
        return false;
    }
    matches!(
        (
            clip.get("id").and_then(JsonValue::as_str),
            clip.get("label").and_then(JsonValue::as_str)
        ),
        (Some("clip-1"), Some("Intro"))
            | (Some("clip-2"), Some("Scene 1"))
            | (Some("clip-3"), Some("Overlay"))
            | (Some("clip-4"), Some("Music.mp3"))
            | (Some("clip-5"), Some("Hello world"))
            | (Some("clip-6"), Some("Second line"))
    )
}

fn migrate_legacy_demo_project(
    state: &AppState,
    id: &str,
    mut project: JsonValue,
) -> Result<(JsonValue, bool), String> {
    let has_legacy_demo = project
        .get("tracks")
        .and_then(JsonValue::as_array)
        .is_some_and(|tracks| {
            tracks.iter().any(|track| {
                track
                    .get("clips")
                    .and_then(JsonValue::as_array)
                    .is_some_and(|clips| clips.iter().any(is_legacy_demo_clip))
            })
        });
    if !has_legacy_demo {
        return Ok((project, false));
    }

    let project_fps = project
        .get("fps")
        .and_then(JsonValue::as_f64)
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .unwrap_or(30.0)
        .clamp(1.0, 120.0);
    let mut first_video_size = None;
    if let Some(tracks) = project.get_mut("tracks").and_then(JsonValue::as_array_mut) {
        for track in tracks.iter_mut() {
            let track_type = track
                .get("type")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            let Some(clips) = track.get_mut("clips").and_then(JsonValue::as_array_mut) else {
                continue;
            };
            clips.retain(|clip| !is_legacy_demo_clip(clip));

            let mut cursor = 0_i64;
            for clip in clips.iter_mut() {
                let Some(source) = clip
                    .get("source_path")
                    .and_then(JsonValue::as_str)
                    .map(str::to_string)
                else {
                    continue;
                };
                let resolved = resolve_source_path(state, Some(id), &source)?;
                let probe = crate::video::probe_video_path(&resolved.to_string_lossy())?;
                let has_video = probe
                    .get("has_video")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let has_audio = probe
                    .get("has_audio")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let duration = probe
                    .get("duration")
                    .and_then(JsonValue::as_f64)
                    .filter(|duration| duration.is_finite() && *duration > 0.0)
                    .or_else(|| {
                        probe
                            .get("audio_duration")
                            .and_then(JsonValue::as_f64)
                            .filter(|duration| duration.is_finite() && *duration > 0.0)
                    })
                    .unwrap_or(1.0 / project_fps);
                let source_fps = if has_video {
                    probe
                        .get("fps")
                        .and_then(JsonValue::as_f64)
                        .filter(|fps| fps.is_finite() && *fps > 0.0)
                        .unwrap_or(project_fps)
                } else {
                    project_fps
                };
                let source_frames = if has_video {
                    probe
                        .get("frame_count")
                        .and_then(JsonValue::as_i64)
                        .filter(|frames| *frames > 0)
                        .unwrap_or_else(|| (duration * source_fps).round() as i64)
                } else {
                    (duration * source_fps).round() as i64
                }
                .max(1);
                let timeline_frames = (duration * project_fps).round().max(1.0) as i64;

                clip["startFrame"] = json!(cursor);
                clip["endFrame"] = json!(cursor + timeline_frames);
                clip["source_fps"] = json!(source_fps);
                clip["source_frames"] = json!(source_frames);
                clip["duration_seconds"] = json!(duration);
                clip["has_audio"] = json!(has_audio);
                clip["media_type"] = json!(if track_type == "audio" || !has_video {
                    "audio"
                } else {
                    "video"
                });
                cursor += timeline_frames;

                if has_video && first_video_size.is_none() {
                    let width = probe.get("width").and_then(JsonValue::as_i64).unwrap_or(0);
                    let height = probe.get("height").and_then(JsonValue::as_i64).unwrap_or(0);
                    if width > 0 && height > 0 {
                        first_video_size = Some((width, height));
                    }
                }
            }
        }

        let mut kept_video = false;
        let mut kept_audio = false;
        tracks.retain(|track| {
            let track_type = track.get("type").and_then(JsonValue::as_str).unwrap_or("");
            let has_clips = track
                .get("clips")
                .and_then(JsonValue::as_array)
                .is_some_and(|clips| !clips.is_empty());
            match track_type {
                "video" => {
                    if has_clips || !kept_video {
                        kept_video = true;
                        true
                    } else {
                        false
                    }
                }
                "audio" => {
                    if has_clips || !kept_audio {
                        kept_audio = true;
                        true
                    } else {
                        false
                    }
                }
                _ => has_clips,
            }
        });
    }
    if let Some((width, height)) = first_video_size {
        project["width"] = json!(width);
        project["height"] = json!(height);
    }
    project["editor_schema_version"] = json!(2);
    project["legacy_demo_removed"] = json!(true);
    Ok((project, true))
}

fn ensure_genesis_environment(state: &AppState) {
    GENESIS_ENV_READY.get_or_init(|| {
        if std::env::var_os("SERENITY_GENESIS_WORKER").is_none() {
            if let Some(output_root) = state.out_dir.parent() {
                let worker = output_root.join("bin").join("genesis-gcompose");
                std::env::set_var("SERENITY_GENESIS_WORKER", worker);
            }
        }
        if std::env::var_os("SERENITY_GENESIS_ASSETS").is_none() {
            if let Some(output_root) = state.out_dir.parent() {
                if let Some(repo_root) = output_root.parent() {
                    std::env::set_var(
                        "SERENITY_GENESIS_ASSETS",
                        repo_root.join("vendor/genesis/web/assets"),
                    );
                }
            }
        }
    });
}

pub async fn post_projects(
    State(state): State<AppState>,
    Json(mut body): Json<JsonValue>,
) -> Response {
    let seq = state
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let id = format!("project-{}-{seq}", now_millis());
    if body.get("name").is_none() {
        body["name"] = JsonValue::String("Untitled Project".to_string());
    }
    if body.get("fps").is_none() {
        body["fps"] = json!(30);
    }
    match write_project(&state, &id, body) {
        Ok(project) => Json(project).into_response(),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

pub async fn get_project(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match read_project(&state, &id) {
        Ok(project) => match migrate_legacy_demo_project(&state, &id, project) {
            Ok((project, true)) => match write_project(&state, &id, project) {
                Ok(project) => Json(project).into_response(),
                Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e),
            },
            Ok((project, false)) => Json(project).into_response(),
            Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e),
        },
        Err(e) => json_error(StatusCode::NOT_FOUND, e),
    }
}

pub async fn put_project(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<JsonValue>,
) -> Response {
    match write_project(&state, &id, body) {
        Ok(project) => Json(project).into_response(),
        Err(e) => json_error(StatusCode::BAD_REQUEST, e),
    }
}

fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_value(url: &str, key: &str) -> Option<String> {
    let query = url.split_once('?')?.1;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k == key {
            return Some(percent_decode(v));
        }
    }
    None
}

fn resolve_source_path(
    state: &AppState,
    project_id: Option<&str>,
    source: &str,
) -> Result<PathBuf, String> {
    let source = percent_decode(source.trim());
    if source.is_empty() {
        return Err("empty media source".to_string());
    }

    let direct = PathBuf::from(&source);
    if direct.is_absolute() && direct.is_file() {
        return direct
            .canonicalize()
            .map_err(|e| format!("resolve media path: {e}"));
    }

    if let Some(rest) = source.strip_prefix("/out/") {
        let path = state.out_dir.join(rest);
        if path.is_file() {
            return path
                .canonicalize()
                .map_err(|e| format!("resolve output media: {e}"));
        }
    }

    if source.starts_with("/view?") || source.contains("/view?") {
        if let Some(filename) = query_value(&source, "filename") {
            let subfolder = query_value(&source, "subfolder").unwrap_or_default();
            let path = state.out_dir.join(subfolder).join(filename);
            if path.is_file() {
                return path
                    .canonicalize()
                    .map_err(|e| format!("resolve viewed media: {e}"));
            }
        }
    }

    let relative = source
        .strip_prefix("/video_edit/media/")
        .unwrap_or(&source)
        .trim_start_matches('/');
    let root_candidate = project_root(state).join(relative);
    if root_candidate.is_file() {
        return root_candidate
            .canonicalize()
            .map_err(|e| format!("resolve imported media: {e}"));
    }
    if let Some(id) = project_id {
        let candidate = project_dir(state, id)?.join(relative);
        if candidate.is_file() {
            return candidate
                .canonicalize()
                .map_err(|e| format!("resolve project media: {e}"));
        }
    }
    let output_candidate = state.out_dir.join(relative);
    if output_candidate.is_file() {
        return output_candidate
            .canonicalize()
            .map_err(|e| format!("resolve Serenity media: {e}"));
    }
    Err(format!("media not found: {source}"))
}

fn value_f32(value: Option<&JsonValue>, default: f32) -> f32 {
    value.and_then(JsonValue::as_f64).unwrap_or(default as f64) as f32
}

fn value_i64(value: Option<&JsonValue>, default: i64) -> i64 {
    value.and_then(JsonValue::as_i64).unwrap_or(default)
}

fn transition_kind(name: &str) -> Option<i32> {
    match name {
        "fade" | "crossfade" => Some(0),
        "wipeleft" => Some(1),
        "wiperight" => Some(2),
        "wipeup" => Some(3),
        "wipedown" => Some(4),
        "slideleft" | "slideright" => Some(5),
        "circleopen" | "circleclose" => Some(6),
        "dissolve" | "fadeblack" | "fadewhite" => Some(7),
        _ => None,
    }
}

fn apply_web_effects(clip: &mut GenesisClip, value: &JsonValue) {
    if let Some(path) = value.get("lut_path").and_then(JsonValue::as_str) {
        if !path.is_empty() {
            clip.look = 2;
            clip.look_amt = 1.0;
            clip.lut = path.to_string();
        }
    }
    let Some(effects) = value.get("effects").and_then(JsonValue::as_array) else {
        return;
    };
    for effect in effects {
        if effect
            .get("enabled")
            .and_then(JsonValue::as_bool)
            .is_some_and(|enabled| !enabled)
        {
            continue;
        }
        let kind = effect.get("type").and_then(JsonValue::as_str).unwrap_or("");
        let params = effect.get("params").unwrap_or(&JsonValue::Null);
        match kind {
            "brightness" => clip.bright = value_f32(params.get("value"), 0.0),
            "contrast" => clip.contrast = value_f32(params.get("value"), 1.0),
            "saturation" => clip.sat = value_f32(params.get("value"), 1.0),
            "hue" => clip.hsl[0] = value_f32(params.get("degrees"), 0.0),
            "gamma" => {
                let gamma = value_f32(params.get("value"), 1.0).max(0.01);
                clip.gamma = [gamma, gamma, gamma];
            }
            "blur" => clip.blur = value_f32(params.get("radius"), 0.0),
            "sharpen" => clip.sharpen = value_f32(params.get("amount"), 0.0),
            "denoise" => {
                clip.denoise = (value_f32(params.get("strength"), 0.0) / 15.0).clamp(0.0, 1.0)
            }
            "glow" => clip.glow_amt = (value_f32(params.get("radius"), 0.0) / 40.0).clamp(0.0, 1.0),
            "vignette" => clip.vignette = value_f32(params.get("angle"), 0.0).clamp(0.0, 1.0),
            "speed" => clip.speed = value_f32(params.get("rate"), 1.0).max(0.01),
            "flip_h" => clip.flip |= 1,
            "flip_v" => clip.flip |= 2,
            _ => {}
        }
    }
}

/// Overlay the browser's native Genesis property payload onto a freshly-built
/// Genesis clip. The browser is allowed to edit render/effect fields, but the
/// timeline adapter remains authoritative for media identity and placement.
///
/// Starting from the serialized `GenesisClip::video` value preserves every
/// non-zero identity default in the upstream model and means newly-added
/// Genesis fields automatically remain backwards compatible.
fn apply_web_genesis_properties(clip: &mut GenesisClip, value: &JsonValue) {
    let Some(properties) = value.get("genesis").and_then(JsonValue::as_object) else {
        return;
    };
    let Ok(mut merged) = serde_json::to_value(&*clip) else {
        return;
    };
    let Some(target) = merged.as_object_mut() else {
        return;
    };
    for (key, field) in properties {
        if matches!(
            key.as_str(),
            "media" | "src_in" | "len" | "t0" | "track" | "seq" | "group"
        ) {
            continue;
        }
        target.insert(key.clone(), field.clone());
    }
    if let Ok(updated) = serde_json::from_value::<GenesisClip>(merged) {
        *clip = updated;
    }
}

fn apply_web_genesis_project(project: &mut GenesisProject, web: &JsonValue) {
    let Some(properties) = web.get("genesis").and_then(JsonValue::as_object) else {
        return;
    };
    let Ok(mut merged) = serde_json::to_value(&*project) else {
        return;
    };
    let Some(target) = merged.as_object_mut() else {
        return;
    };
    for key in [
        "bright",
        "contrast",
        "sat",
        "bright_kf",
        "contrast_kf",
        "sat_kf",
        "opacity_kf",
        "gain_kf",
        "pip_kf",
        "export",
        "kf_interp",
        "export_in",
        "export_out",
    ] {
        if let Some(field) = properties.get(key) {
            target.insert(key.to_string(), field.clone());
        }
    }
    if let Ok(updated) = serde_json::from_value::<GenesisProject>(merged) {
        *project = updated;
    }
}

fn web_project_to_genesis(
    state: &AppState,
    project_id: Option<&str>,
    web: &JsonValue,
) -> Result<GenesisProject, String> {
    let web_tracks = web
        .get("tracks")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "video project has no tracks array".to_string())?;

    // `demo` establishes every non-zero identity default in the upstream model.
    let mut project = GenesisProject::demo(String::new());
    project.media.clear();
    project.names.clear();
    project.media_bin.clear();
    project.clips.clear();
    project.transitions.clear();
    project.subtitles.clear();
    project.tracks.clear();
    project.bright = 0.0;
    project.contrast = 1.0;
    project.sat = 1.0;

    let mut track_map: Vec<Option<u8>> = vec![None; web_tracks.len()];
    let web_fps = web
        .get("fps")
        .and_then(JsonValue::as_f64)
        .unwrap_or(30.0)
        .clamp(1.0, 120.0) as f32;
    // Browser tracks are top-to-bottom; Genesis tracks are bottom-to-top.
    for (web_index, track) in web_tracks.iter().enumerate().rev() {
        let kind = match track.get("type").and_then(JsonValue::as_str) {
            Some("video") => Some(GenesisTrackKind::Video),
            Some("audio") => Some(GenesisTrackKind::Audio),
            _ => None,
        };
        if let Some(kind) = kind {
            let name = track.get("name").and_then(JsonValue::as_str).unwrap_or(
                if kind == GenesisTrackKind::Video {
                    "Video"
                } else {
                    "Audio"
                },
            );
            let index = project.tracks.len();
            if index > u8::MAX as usize {
                return Err("Genesis supports at most 256 tracks".to_string());
            }
            let mut genesis_track = GenesisTrack::new(kind, name);
            genesis_track.hidden = track
                .get("hidden")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            genesis_track.muted = track
                .get("muted")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            genesis_track.locked = track
                .get("locked")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            if let Some(mixer) = track.get("genesis") {
                genesis_track.gain = value_f32(mixer.get("gain"), 1.0).clamp(0.0, 2.0);
                genesis_track.pan = value_f32(mixer.get("pan"), 0.0).clamp(-1.0, 1.0);
                genesis_track.solo = mixer
                    .get("solo")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
            }
            project.tracks.push(genesis_track);
            track_map[web_index] = Some(index as u8);
        }
    }

    let mut media_index: HashMap<String, usize> = HashMap::new();
    for (web_track_index, track) in web_tracks.iter().enumerate() {
        let track_type = track.get("type").and_then(JsonValue::as_str).unwrap_or("");
        let clips = track
            .get("clips")
            .and_then(JsonValue::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if track_type == "text" {
            for clip in clips {
                let start = value_i64(clip.get("startFrame"), 0);
                let end = value_i64(clip.get("endFrame"), start + 1);
                let text = clip
                    .get("label")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("")
                    .to_string();
                if end > start && !text.is_empty() {
                    project.subtitles.push(GenesisSubtitle { start, end, text });
                }
            }
            continue;
        }
        let Some(genesis_track) = track_map[web_track_index] else {
            continue;
        };
        for clip in clips {
            let source = clip
                .get("source_path")
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            if source.is_empty() {
                continue;
            }
            let resolved = resolve_source_path(state, project_id, source)?;
            let resolved = resolved.to_string_lossy().into_owned();
            let media = if let Some(index) = media_index.get(&resolved) {
                *index
            } else {
                let index = project.media.len();
                project.media.push(resolved.clone());
                project.names.push(
                    clip.get("label")
                        .and_then(JsonValue::as_str)
                        .unwrap_or("clip")
                        .to_string(),
                );
                project.media_bin.push(0);
                media_index.insert(resolved, index);
                index
            };
            let start = value_i64(clip.get("startFrame"), 0);
            let end = value_i64(clip.get("endFrame"), start + 1);
            if end <= start {
                continue;
            }
            let label = clip
                .get("label")
                .and_then(JsonValue::as_str)
                .unwrap_or("clip");
            let mut genesis_clip =
                GenesisClip::video(media, start, end - start, genesis_track, label);
            genesis_clip.src_in = value_i64(clip.get("source_start"), 0);
            genesis_clip.fade_in = value_i64(clip.get("fade_in"), 0).max(0);
            genesis_clip.fade_out = value_i64(clip.get("fade_out"), 0).max(0);
            apply_web_effects(&mut genesis_clip, clip);
            apply_web_genesis_properties(&mut genesis_clip, clip);
            let source_fps = value_f32(clip.get("source_fps"), web_fps).clamp(1.0, 120.0);
            genesis_clip.speed *= source_fps / web_fps;
            project.clips.push(genesis_clip);

            if let Some(transition) = clip.get("transition_in") {
                if let Some(kind) = transition
                    .get("type")
                    .and_then(JsonValue::as_str)
                    .and_then(transition_kind)
                {
                    let duration = value_i64(transition.get("duration"), 15).max(2);
                    project.transitions.push(GenesisTransition {
                        track: genesis_track,
                        center: start,
                        dur: duration,
                        kind,
                    });
                }
            }
        }
    }

    let fps = web.get("fps").and_then(JsonValue::as_u64).unwrap_or(30);
    let width = web
        .get("width")
        .and_then(JsonValue::as_u64)
        .unwrap_or(1280)
        .clamp(16, u32::MAX as u64) as u32;
    let height = web
        .get("height")
        .and_then(JsonValue::as_u64)
        .unwrap_or(720)
        .clamp(16, u32::MAX as u64) as u32;
    project.export.out_w = width;
    project.export.out_h = height;
    project.export.fps_num = fps.clamp(1, u32::MAX as u64) as u32;
    project.export.fps_den = 1;
    apply_web_genesis_project(&mut project, web);
    Ok(project)
}

fn png_response(bytes: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(bytes));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

fn encode_rgba_png(width: u32, height: u32, rgba: Vec<u8>) -> Result<Vec<u8>, String> {
    let image = RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| format!("Genesis returned invalid RGBA size for {width}x{height}"))?;
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut out, ImageFormat::Png)
        .map_err(|e| format!("encode Genesis preview PNG: {e}"))?;
    Ok(out.into_inner())
}

#[derive(Deserialize)]
pub struct PreviewRequest {
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    project: Option<JsonValue>,
    frame: i64,
}

pub async fn post_preview(
    State(state): State<AppState>,
    Json(request): Json<PreviewRequest>,
) -> Response {
    ensure_genesis_environment(&state);
    let web = match request.project {
        Some(value) => value,
        None => {
            let Some(id) = request.project_id.as_deref() else {
                return json_error(StatusCode::BAD_REQUEST, "project or project_id is required");
            };
            match read_project(&state, id) {
                Ok(value) => value,
                Err(e) => return json_error(StatusCode::NOT_FOUND, e),
            }
        }
    };
    let project = match web_project_to_genesis(&state, request.project_id.as_deref(), &web) {
        Ok(project) => project,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, e),
    };
    if project.clips.is_empty() {
        let mut empty = vec![0; (GENESIS_PREVIEW_WIDTH * GENESIS_PREVIEW_HEIGHT * 4) as usize];
        for alpha in empty.iter_mut().skip(3).step_by(4) {
            *alpha = 255;
        }
        return match encode_rgba_png(GENESIS_PREVIEW_WIDTH, GENESIS_PREVIEW_HEIGHT, empty) {
            Ok(bytes) => png_response(bytes),
            Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }
    let frame = request.frame.max(0);
    let rendered =
        tokio::task::spawn_blocking(move || genesis_web::worker::request_frame(&project, frame))
            .await;
    match rendered {
        Ok(Some(rgba)) => {
            match encode_rgba_png(GENESIS_PREVIEW_WIDTH, GENESIS_PREVIEW_HEIGHT, rgba) {
                Ok(bytes) => png_response(bytes),
                Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e),
            }
        }
        Ok(None) => json_error(
            StatusCode::BAD_GATEWAY,
            "Genesis gcompose could not render the requested frame",
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Genesis preview task failed: {e}"),
        ),
    }
}

#[derive(Deserialize)]
pub struct EffectPreviewRequest {
    source_path: String,
    #[serde(default)]
    effects: Vec<JsonValue>,
    #[serde(default)]
    lut_path: Option<String>,
    #[serde(default)]
    seek_sec: f64,
}

async fn render_effect_preview(state: AppState, request: EffectPreviewRequest) -> Response {
    ensure_genesis_environment(&state);
    let source_frame = (request.seek_sec.max(0.0) * 30.0).round() as i64;
    let mut clip = json!({
        "id": "preview-clip",
        "startFrame": 0,
        "endFrame": 1,
        "source_start": source_frame,
        "source_path": request.source_path,
        "label": "Effect preview",
        "effects": request.effects
    });
    if let Some(path) = request.lut_path {
        clip["lut_path"] = JsonValue::String(path);
    }
    let web = json!({
        "name": "Effect preview",
        "fps": 30,
        "width": GENESIS_PREVIEW_WIDTH,
        "height": GENESIS_PREVIEW_HEIGHT,
        "tracks": [{
            "id": "preview-video",
            "name": "Video",
            "type": "video",
            "clips": [clip]
        }]
    });
    let project = match web_project_to_genesis(&state, None, &web) {
        Ok(project) => project,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, e),
    };
    match tokio::task::spawn_blocking(move || genesis_web::worker::request_frame(&project, 0)).await
    {
        Ok(Some(rgba)) => {
            match encode_rgba_png(GENESIS_PREVIEW_WIDTH, GENESIS_PREVIEW_HEIGHT, rgba) {
                Ok(bytes) => png_response(bytes),
                Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e),
            }
        }
        Ok(None) => json_error(
            StatusCode::BAD_GATEWAY,
            "Genesis could not render the effect preview",
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("effect preview task failed: {e}"),
        ),
    }
}

pub async fn post_preview_effect(
    State(state): State<AppState>,
    Json(request): Json<EffectPreviewRequest>,
) -> Response {
    render_effect_preview(state, request).await
}

pub async fn post_lut_preview(
    State(state): State<AppState>,
    Json(request): Json<EffectPreviewRequest>,
) -> Response {
    render_effect_preview(state, request).await
}

fn lut_dir(state: &AppState) -> PathBuf {
    state.out_dir.join("video_luts")
}

pub async fn get_luts(State(state): State<AppState>) -> Response {
    let dir = lut_dir(&state);
    if let Err(e) = fs::create_dir_all(&dir) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("create LUT directory: {e}"),
        );
    }
    let mut luts = Vec::new();
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("read LUT directory: {e}"),
            )
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("cube") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("LUT");
        luts.push(json!({
            "name": name,
            "path": path.to_string_lossy()
        }));
    }
    luts.sort_by(|a, b| {
        a.get("name")
            .and_then(JsonValue::as_str)
            .cmp(&b.get("name").and_then(JsonValue::as_str))
    });
    Json(luts).into_response()
}

pub async fn post_lut_upload(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let dir = lut_dir(&state);
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("create LUT directory: {e}"),
        );
    }
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("multipart: {e}")),
        };
        if field.name() != Some("file") {
            continue;
        }
        let raw_name = field.file_name().unwrap_or("uploaded.cube");
        let Some(mut name) = safe_component(raw_name) else {
            return json_error(StatusCode::BAD_REQUEST, "invalid LUT filename");
        };
        if !name.to_ascii_lowercase().ends_with(".cube") {
            return json_error(StatusCode::BAD_REQUEST, "LUT must be a .cube file");
        }
        if !name.ends_with(".cube") {
            name.truncate(name.len() - 5);
            name.push_str(".cube");
        }
        let path = dir.join(&name);
        let bytes = match field.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("read LUT upload: {e}")),
        };
        if bytes.is_empty() || bytes.len() > 32 * 1024 * 1024 {
            return json_error(StatusCode::BAD_REQUEST, "invalid LUT file size");
        }
        if let Err(e) = tokio::fs::write(&path, bytes).await {
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, format!("write LUT: {e}"));
        }
        let display_name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("LUT");
        return Json(json!({
            "name": display_name,
            "path": path.to_string_lossy()
        }))
        .into_response();
    }
    json_error(StatusCode::BAD_REQUEST, "multipart LUT file is required")
}

fn content_type_for(path: &FsPath) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/mp4",
        "aac" => "audio/aac",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

fn serve_file(path: &FsPath, headers: &HeaderMap) -> Response {
    let metadata = match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => metadata,
        _ => return json_error(StatusCode::NOT_FOUND, "media file not found"),
    };
    let total = metadata.len();
    let range = headers
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes="))
        .and_then(|value| value.split(',').next())
        .and_then(|value| value.split_once('-'));

    let (start, end, partial) = if let Some((start, end)) = range {
        let start = start.parse::<u64>().ok();
        let end = if end.is_empty() {
            None
        } else {
            end.parse::<u64>().ok()
        };
        match start {
            Some(start) if start < total => {
                let end = end
                    .unwrap_or(total.saturating_sub(1))
                    .min(total.saturating_sub(1));
                if end < start {
                    return json_error(StatusCode::RANGE_NOT_SATISFIABLE, "invalid media range");
                }
                (start, end, true)
            }
            _ => {
                let mut response = json_error(
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    "media range outside file",
                );
                if let Ok(value) = HeaderValue::from_str(&format!("bytes */{total}")) {
                    response.headers_mut().insert(header::CONTENT_RANGE, value);
                }
                return response;
            }
        }
    } else {
        (0, total.saturating_sub(1), false)
    };

    let len = if total == 0 { 0 } else { end - start + 1 };
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("open media: {e}"),
            )
        }
    };
    if let Err(e) = file.seek(SeekFrom::Start(start)) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("seek media: {e}"),
        );
    }
    let mut body = vec![0u8; len as usize];
    if let Err(e) = file.read_exact(&mut body) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read media: {e}"),
        );
    }

    let mut response = Response::new(Body::from(body));
    *response.status_mut() = if partial {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(content_type_for(path)),
    );
    response
        .headers_mut()
        .insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if let Ok(value) = HeaderValue::from_str(&len.to_string()) {
        response.headers_mut().insert(header::CONTENT_LENGTH, value);
    }
    if partial {
        if let Ok(value) = HeaderValue::from_str(&format!("bytes {start}-{end}/{total}")) {
            response.headers_mut().insert(header::CONTENT_RANGE, value);
        }
    }
    response
}

pub async fn get_media(
    State(state): State<AppState>,
    Path(path): Path<String>,
    headers: HeaderMap,
) -> Response {
    match resolve_source_path(&state, None, &path) {
        Ok(path) => serve_file(&path, &headers),
        Err(e) => json_error(StatusCode::NOT_FOUND, e),
    }
}

pub async fn get_cache(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response {
    let Some(name) = safe_component(&name) else {
        return json_error(StatusCode::BAD_REQUEST, "invalid cache filename");
    };
    let path = state.out_dir.join("video_editor_cache").join(name);
    serve_file(&path, &headers)
}

pub async fn post_import_clip(
    State(state): State<AppState>,
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> Response {
    ensure_genesis_environment(&state);
    let dir = match project_dir(&state, &id) {
        Ok(dir) => dir.join("media"),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("create project media directory: {e}"),
        );
    }

    let mut saved: Option<(PathBuf, String)> = None;
    let mut project_fps_hint: Option<f64> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("multipart: {e}")),
        };
        if field.name() == Some("project_fps") {
            project_fps_hint = field
                .text()
                .await
                .ok()
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|fps| fps.is_finite() && *fps > 0.0)
                .map(|fps| fps.clamp(1.0, 120.0));
            continue;
        }
        if field.name() != Some("file") {
            continue;
        }
        let raw_name = field.file_name().unwrap_or("clip.bin");
        let base_name = safe_component(raw_name).unwrap_or_else(|| "clip.bin".to_string());
        let stored_name = format!("{}-{base_name}", now_millis());
        let path = dir.join(&stored_name);
        let mut output = match tokio::fs::File::create(&path).await {
            Ok(output) => output,
            Err(e) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("create imported clip: {e}"),
                )
            }
        };
        let mut field = field;
        loop {
            match field.chunk().await {
                Ok(Some(chunk)) => {
                    if let Err(e) = output.write_all(&chunk).await {
                        return json_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("write imported clip: {e}"),
                        );
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    return json_error(StatusCode::BAD_REQUEST, format!("read imported clip: {e}"))
                }
            }
        }
        if let Err(e) = output.flush().await {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("flush imported clip: {e}"),
            );
        }
        saved = Some((path, stored_name));
        break;
    }

    let Some((path, stored_name)) = saved else {
        return json_error(StatusCode::BAD_REQUEST, "multipart file field is required");
    };
    let project_fps = project_fps_hint.unwrap_or_else(|| {
        read_project(&state, &id)
            .ok()
            .and_then(|project| project.get("fps").and_then(JsonValue::as_f64))
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .unwrap_or(30.0)
            .clamp(1.0, 120.0)
    });
    let probe_path = path.to_string_lossy().into_owned();
    let (duration_frames, source_fps, media_type, width, height, has_audio, duration_seconds) =
        tokio::task::spawn_blocking(move || {
            let probe = crate::video::probe_video_path(&probe_path).ok();
            let has_video = probe
                .as_ref()
                .and_then(|value| value.get("has_video"))
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let has_audio = probe
                .as_ref()
                .and_then(|value| value.get("has_audio"))
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            let duration = probe
                .as_ref()
                .and_then(|value| value.get("duration"))
                .and_then(JsonValue::as_f64)
                .filter(|duration| duration.is_finite() && *duration > 0.0)
                .or_else(|| {
                    probe
                        .as_ref()
                        .and_then(|value| value.get("audio_duration"))
                        .and_then(JsonValue::as_f64)
                        .filter(|duration| duration.is_finite() && *duration > 0.0)
                })
                .unwrap_or(1.0 / project_fps);
            let fps = if has_video {
                probe
                    .as_ref()
                    .and_then(|value| value.get("fps"))
                    .and_then(JsonValue::as_f64)
                    .filter(|fps| fps.is_finite() && *fps > 0.0)
                    .unwrap_or(project_fps)
            } else {
                project_fps
            };
            let frames = if has_video {
                probe
                    .as_ref()
                    .and_then(|value| value.get("frame_count"))
                    .and_then(JsonValue::as_i64)
                    .filter(|frames| *frames > 0)
                    .or_else(|| genesis_web::worker::media_frames(&probe_path))
                    .unwrap_or_else(|| (duration * fps).round() as i64)
            } else {
                (duration * fps).round() as i64
            }
            .max(1);
            let width = probe
                .as_ref()
                .and_then(|value| value.get("width"))
                .and_then(JsonValue::as_i64)
                .unwrap_or(0);
            let height = probe
                .as_ref()
                .and_then(|value| value.get("height"))
                .and_then(JsonValue::as_i64)
                .unwrap_or(0);
            (
                frames,
                fps,
                if has_video { "video" } else { "audio" },
                width,
                height,
                has_audio,
                duration,
            )
        })
        .await
        .unwrap_or((1, project_fps, "audio", 0, 0, false, 1.0 / project_fps));
    Json(json!({
        "clip_id": format!("clip-{}", now_millis()),
        "source_path": format!("{id}/media/{stored_name}"),
        "duration_frames": duration_frames.max(1),
        "duration_seconds": duration_seconds,
        "source_fps": source_fps,
        "media_type": media_type,
        "width": width,
        "height": height,
        "has_audio": has_audio
    }))
    .into_response()
}

#[derive(Deserialize)]
pub struct ThumbnailRequest {
    source_path: String,
    #[serde(default = "default_thumbnail_height")]
    height: u32,
    #[serde(default = "default_media_fps")]
    fps: f64,
}

fn default_thumbnail_height() -> u32 {
    36
}

fn default_media_fps() -> f64 {
    30.0
}

pub async fn post_thumbnails(
    State(state): State<AppState>,
    Json(request): Json<ThumbnailRequest>,
) -> Response {
    ensure_genesis_environment(&state);
    let path = match resolve_source_path(&state, None, &request.source_path) {
        Ok(path) => path,
        Err(e) => return json_error(StatusCode::NOT_FOUND, e),
    };
    let height = request.height.clamp(16, 120);
    let fps = request.fps.clamp(1.0, 120.0);
    let width = ((height as f32) * 16.0 / 9.0).round().max(16.0) as u32;
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    height.hash(&mut hasher);
    fps.to_bits().hash(&mut hasher);
    let cache_name = format!("thumbs-{:016x}.png", hasher.finish());
    let cache_dir = state.out_dir.join("video_editor_cache");
    let cache_path = cache_dir.join(&cache_name);
    if cache_path.is_file() {
        let frame_count = image::image_dimensions(&cache_path)
            .ok()
            .map(|(sprite_width, _)| (sprite_width / width).max(1))
            .unwrap_or(1);
        return Json(json!({
            "sprite_url": format!("/video_edit/cache/{cache_name}"),
            "thumb_width": width,
            "thumb_height": height,
            "frame_count": frame_count
        }))
        .into_response();
    }
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("create thumbnail cache: {e}"),
        );
    }

    let render_path = path.to_string_lossy().into_owned();
    let result = tokio::task::spawn_blocking(move || {
        let total_frames = genesis_web::worker::media_frames(&render_path).unwrap_or(1);
        let seconds = ((total_frames as f64 / fps).ceil() as usize).max(1);
        let count = seconds.min(MAX_THUMBNAILS);
        let mut sprite = RgbaImage::new(width * count as u32, height);
        for index in 0..count {
            let source_frame =
                ((index as f64 * fps).round() as i64).min(total_frames.saturating_sub(1));
            let rgba = genesis_web::worker::thumbnail(
                &render_path,
                source_frame,
                width as usize,
                height as usize,
            )?;
            let thumb = RgbaImage::from_raw(width, height, rgba)?;
            image::imageops::replace(&mut sprite, &thumb, (index as u32 * width) as i64, 0);
        }
        Some((sprite, count))
    })
    .await;

    let (sprite, count) = match result {
        Ok(Some(value)) => value,
        Ok(None) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                "Genesis could not decode thumbnail frames",
            )
        }
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("thumbnail task failed: {e}"),
            )
        }
    };
    if let Err(e) = DynamicImage::ImageRgba8(sprite).save_with_format(&cache_path, ImageFormat::Png)
    {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("write thumbnail sprite: {e}"),
        );
    }
    Json(json!({
        "sprite_url": format!("/video_edit/cache/{cache_name}"),
        "thumb_width": width,
        "thumb_height": height,
        "frame_count": count
    }))
    .into_response()
}

#[derive(Deserialize)]
pub struct WaveformRequest {
    source_path: String,
    #[serde(default = "default_samples_per_second")]
    samples_per_second: usize,
    #[serde(default = "default_media_fps")]
    fps: f64,
}

fn default_samples_per_second() -> usize {
    30
}

pub async fn post_waveform(
    State(state): State<AppState>,
    Json(request): Json<WaveformRequest>,
) -> Response {
    ensure_genesis_environment(&state);
    let path = match resolve_source_path(&state, None, &request.source_path) {
        Ok(path) => path,
        Err(e) => return json_error(StatusCode::NOT_FOUND, e),
    };
    let samples_per_second = request.samples_per_second.clamp(1, 120);
    let fps = request.fps.clamp(1.0, 120.0);
    let render_path = path.to_string_lossy().into_owned();
    let result = tokio::task::spawn_blocking(move || {
        let probe = crate::video::probe_video_path(&render_path).ok();
        let duration_seconds = probe
            .as_ref()
            .and_then(|value| value.get("duration"))
            .and_then(JsonValue::as_f64)
            .filter(|duration| duration.is_finite() && *duration > 0.0)
            .or_else(|| {
                probe
                    .as_ref()
                    .and_then(|value| value.get("audio_duration"))
                    .and_then(JsonValue::as_f64)
                    .filter(|duration| duration.is_finite() && *duration > 0.0)
            })
            .unwrap_or_else(|| {
                genesis_web::worker::media_frames(&render_path).unwrap_or(1) as f64 / fps
            });
        let buckets =
            ((duration_seconds * samples_per_second as f64).ceil() as usize).clamp(1, 100_000);
        let peaks = genesis_web::worker::audio_envelope(&render_path, buckets)
            .unwrap_or_else(|| vec![0.0; buckets]);
        (peaks, duration_seconds)
    })
    .await;
    match result {
        Ok((peaks, duration_seconds)) => Json(json!({
            "peaks": peaks,
            "sample_rate": samples_per_second,
            "duration_seconds": duration_seconds
        }))
        .into_response(),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("waveform task failed: {e}"),
        ),
    }
}

#[derive(Deserialize)]
pub struct ExportRequest {
    project_id: String,
    #[serde(default = "default_export_format")]
    format: String,
    #[serde(default = "default_export_width")]
    width: u32,
    #[serde(default = "default_export_height")]
    height: u32,
    #[serde(default = "default_export_fps")]
    fps: u32,
    #[serde(default = "default_export_quality")]
    quality: String,
    #[serde(default = "default_true")]
    include_audio: bool,
    output_filename: String,
    #[serde(default)]
    range_start_frame: Option<i64>,
    #[serde(default)]
    range_end_frame: Option<i64>,
    #[serde(default)]
    lut_path: Option<String>,
    #[serde(default)]
    genesis_export: Option<JsonValue>,
}

fn default_export_format() -> String {
    "h264".to_string()
}
fn default_export_width() -> u32 {
    1280
}
fn default_export_height() -> u32 {
    720
}
fn default_export_fps() -> u32 {
    30
}
fn default_export_quality() -> String {
    "high".to_string()
}
fn default_true() -> bool {
    true
}

pub async fn post_export(
    State(state): State<AppState>,
    Json(request): Json<ExportRequest>,
) -> Response {
    ensure_genesis_environment(&state);
    let web = match read_project(&state, &request.project_id) {
        Ok(project) => project,
        Err(e) => return json_error(StatusCode::NOT_FOUND, e),
    };
    let mut project = match web_project_to_genesis(&state, Some(&request.project_id), &web) {
        Ok(project) => project,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, e),
    };
    project.export.out_w = request.width.clamp(16, 8192);
    project.export.out_h = request.height.clamp(16, 8192);
    project.export.fps_num = request.fps.clamp(1, 120);
    project.export.fps_den = 1;
    project.export.rate_mode = 1;
    project.export.crf = match request.quality.as_str() {
        "lossless" => 0,
        "high" => 18,
        "medium" => 23,
        "low" => 28,
        _ => 20,
    };
    project.export.rate_value = project.export.crf;
    project.export.vcodec = match request.format.as_str() {
        "prores" => "prores_ks",
        "vp9" => "libvpx-vp9",
        "h264" => "libx264",
        _ => "mpeg4",
    }
    .to_string();
    if let Some(native) = request
        .genesis_export
        .as_ref()
        .and_then(JsonValue::as_object)
    {
        if let Ok(mut merged) = serde_json::to_value(&project.export) {
            if let Some(target) = merged.as_object_mut() {
                for (key, value) in native {
                    target.insert(key.clone(), value.clone());
                }
                if let Ok(updated) = serde_json::from_value(merged) {
                    project.export = updated;
                }
            }
        }
    }
    if let Some(lut_path) = request.lut_path.as_deref().filter(|path| !path.is_empty()) {
        for clip in &mut project.clips {
            clip.look = 2;
            clip.look_amt = 1.0;
            clip.lut = lut_path.to_string();
        }
    }
    if let (Some(start), Some(end)) = (request.range_start_frame, request.range_end_frame) {
        if end > start {
            project.export_in = start.max(0);
            project.export_out = end;
        }
    }
    if !request.include_audio {
        // Genesis treats an unavailable audio encoder as a supported video-only export.
        // `none` deliberately selects no FFmpeg encoder while keeping the video encoder live.
        project.export.acodec = "none".to_string();
        let audio_tracks: Vec<bool> = project
            .tracks
            .iter()
            .map(|track| track.kind == GenesisTrackKind::Audio)
            .collect();
        project.clips.retain(|clip| {
            !audio_tracks
                .get(clip.track as usize)
                .copied()
                .unwrap_or(false)
        });
    }

    let Some(filename) = safe_component(&request.output_filename) else {
        return json_error(StatusCode::BAD_REQUEST, "invalid export filename");
    };
    let export_dir = match project_dir(&state, &request.project_id) {
        Ok(dir) => dir.join("exports"),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = fs::create_dir_all(&export_dir) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("create export directory: {e}"),
        );
    }
    let output_path = export_dir.join(&filename);
    let output_string = output_path.to_string_lossy().into_owned();
    let export_id = format!("export-{}", now_millis());
    let output_url = format!(
        "/video_edit/media/{}/exports/{}",
        request.project_id, filename
    );
    let status = ExportStatus {
        export_id: export_id.clone(),
        state: "running".to_string(),
        output_path: None,
        output_url: None,
        error: None,
    };
    exports()
        .lock()
        .expect("video export state poisoned")
        .insert(export_id.clone(), status);

    let task_id = export_id.clone();
    let task_output = output_string.clone();
    tokio::task::spawn_blocking(move || {
        let ok = genesis_web::worker::render_program(&project, &task_output);
        let mut map = exports().lock().expect("video export state poisoned");
        if let Some(status) = map.get_mut(&task_id) {
            if ok {
                status.state = "complete".to_string();
                status.output_path = Some(task_output);
                status.output_url = Some(output_url);
            } else {
                status.state = "failed".to_string();
                status.error = Some("Genesis gcompose export failed".to_string());
            }
        }
    });

    Json(json!({
        "export_id": export_id,
        "state": "running",
        "status_url": format!("/video_edit/export/{export_id}")
    }))
    .into_response()
}

pub async fn get_export(Path(id): Path<String>) -> Response {
    let map = exports().lock().expect("video export state poisoned");
    match map.get(&id) {
        Some(status) => Json(status).into_response(),
        None => json_error(StatusCode::NOT_FOUND, "video export not found"),
    }
}

pub async fn get_status(State(state): State<AppState>) -> Response {
    ensure_genesis_environment(&state);
    let worker = std::env::var_os("SERENITY_GENESIS_WORKER")
        .map(PathBuf::from)
        .unwrap_or_default();
    Json(json!({
        "engine": "Genesis gcompose",
        "transport": "Rust/C/FFmpeg/OpenCL sidecar",
        "native_ui": false,
        "mojo": false,
        "worker_path": worker,
        "worker_ready": worker.is_file()
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_browser_effects_do_not_reach_genesis() {
        let mut clip = GenesisClip::video(0, 0, 1, 0, "test");
        apply_web_effects(
            &mut clip,
            &json!({
                "effects": [
                    {
                        "type": "saturation",
                        "enabled": false,
                        "params": { "value": 0.0 }
                    },
                    {
                        "type": "contrast",
                        "enabled": true,
                        "params": { "value": 1.25 }
                    }
                ]
            }),
        );

        assert_eq!(clip.sat, 1.0);
        assert_eq!(clip.contrast, 1.25);
    }

    #[test]
    fn native_properties_reach_genesis_without_retargeting_clip() {
        let mut clip = GenesisClip::video(7, 12, 90, 2, "test");
        apply_web_genesis_properties(
            &mut clip,
            &json!({
                "genesis": {
                    "media": 99,
                    "t0": 999,
                    "track": 8,
                    "sat": 0.25,
                    "rot": 14.0,
                    "curve": [0.0, 0.2, 0.55, 0.8, 1.0],
                    "audio_fx": {
                        "eq_low_db": 3.0,
                        "eq_mid_db": 0.0,
                        "eq_high_db": -2.0,
                        "pan": -0.2,
                        "compress": true,
                        "gate": false,
                        "normalize": false,
                        "reverb": 0.0,
                        "delay_ms": 0.0,
                        "delay_decay": 0.5,
                        "pitch": 0.0,
                        "lowpass_hz": 0.0,
                        "highpass_hz": 0.0,
                        "tremolo": 0.0,
                        "bass_db": 0.0,
                        "treble_db": 0.0,
                        "notch_hz": 0.0,
                        "chorus": 0.0,
                        "flanger": 0.0,
                        "phaser": 0.0,
                        "limiter": 0.0,
                        "geq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    }
                }
            }),
        );

        assert_eq!(clip.media, 7);
        assert_eq!(clip.t0, 12);
        assert_eq!(clip.track, 2);
        assert_eq!(clip.sat, 0.25);
        assert_eq!(clip.rot, 14.0);
        assert_eq!(clip.curve[2], 0.55);
        assert_eq!(clip.audio_fx.eq_low_db, 3.0);
        assert!(clip.audio_fx.compress);
    }

    #[test]
    fn legacy_demo_detection_never_removes_real_media() {
        assert!(is_legacy_demo_clip(&json!({
            "id": "clip-4",
            "label": "Music.mp3"
        })));
        assert!(!is_legacy_demo_clip(&json!({
            "id": "clip-4",
            "label": "Music.mp3",
            "source_path": "project-1/media/real-music.mp3"
        })));
        assert!(!is_legacy_demo_clip(&json!({
            "id": "clip-user",
            "label": "Intro"
        })));
    }
}
