//! /v1/video — faithful port of the daemon's video endpoints (video_api.mojo):
//! GET /v1/video (readiness), POST /v1/video (bounded LTX2 staged smoke),
//! GET /v1/video/probe?path= (ffprobe wrapper).
//!
//! GET readiness is mostly static + a `test -x` on the LTX2 smoke runner; the
//! backend/model/resident identity fields echo the Rust server's identity
//! (the daemon echoed its own — same convention as /v1/health). POST runs the
//! external runner (not built by default -> "missing executable" error). Probe
//! shells `ffprobe` and reshapes its JSON.

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::AppState;

const RUNNER: &str = "output/bin/ltx2_video_smoke_runner";
const BACKEND_NAME: &str = "isolated-rust";

fn json_resp(status: StatusCode, doc: &Value) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}
fn err_detail(status: StatusCode, detail: &str) -> Response {
    json_resp(status, &json!({ "detail": detail }))
}

fn runner_available() -> bool {
    // `test -x`: exists + executable bit set.
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(RUNNER)
        .map(|m| m.is_file() && (m.permissions().mode() & 0o111) != 0)
        .unwrap_or(false)
}

fn readiness_doc() -> Value {
    let ready = runner_available();
    let state = if ready { "bounded_smoke_ready" } else { "runner_missing" };
    let runners = json!([
        {
            "model": "lance_t2v",
            "status": "smoke_only",
            "runner": "serenitymojo/pipeline/lance_t2v_256_9f_dense_probe.mojo",
            "limit": "standalone pipeline artifact gate; not daemon job-backed",
        },
        {
            "model": "ltx2_t2v_av",
            "status": state,
            "runner": RUNNER,
            "mode": "staged lora resident noaudio nonag",
            "default_steps": 1,
            "default_weight_mode": "resident",
            "default_audio_mode": "noaudio",
            "target_width": 768,
            "target_height": 512,
            "target_frame_count": 121,
            "target_fps": 24,
            "limit": "bounded daemon smoke only; artifact acceptance requires a successful MP4/A-V probe with timings and VRAM evidence; full video parity remains separate",
        },
    ]);
    json!({
        "schema": "serenity.video_status.v1",
        "endpoint": "/v1/video",
        "state": state,
        "readiness_label": if ready { "bounded_daemon_smoke" } else { "build_required" },
        "accepted": false,
        "backend": BACKEND_NAME,
        "model": "",
        "resident": "",
        "mp4": "",
        "frame_count": 0,
        "duration": 0.0,
        "audio": false,
        "non_acceptance_reason": "bounded smoke wiring is not full SwarmUI video parity; artifact acceptance requires frame_count, duration, muxing, audio behavior, timings, and VRAM evidence",
        "probe_endpoint": "/v1/video/probe?path=<mp4>",
        "candidate_runners": runners,
    })
}

/// GET /v1/video — readiness contract.
pub async fn get_video() -> Response {
    json_resp(StatusCode::OK, &readiness_doc())
}

/// POST /v1/video — bounded LTX2 staged smoke (requires the runner to be built).
pub async fn post_video(State(st): State<AppState>, body: String) -> Response {
    let b: Value = serde_json::from_str::<Value>(&body).ok().filter(|v| v.is_object()).unwrap_or_else(|| json!({}));
    let s = |k: &str, d: &str| b.get(k).and_then(|v| v.as_str()).unwrap_or(d).to_string();
    let runner = s("runner", "ltx2_staged_dev_smoke");
    if runner != "ltx2_staged_dev_smoke" {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("unsupported video runner '{runner}'; supported runner is ltx2_staged_dev_smoke"));
    }
    let steps = b.get("steps").and_then(|v| v.as_i64()).unwrap_or(1);
    if !(1..=3).contains(&steps) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'steps' out of range [1..3]");
    }
    let weight_mode = s("weight_mode", "resident");
    if weight_mode != "resident" && weight_mode != "stream" {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("unsupported weight_mode '{weight_mode}'; use resident or stream"));
    }
    let mut audio_mode = s("audio_mode", "noaudio");
    if audio_mode == "video" {
        audio_mode = "noaudio".to_string();
    }
    if audio_mode != "audio" && audio_mode != "noaudio" {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("unsupported audio_mode '{audio_mode}'; use audio or noaudio"));
    }
    if !runner_available() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("missing executable {RUNNER}; run `pixi run build-video-smoke` first"));
    }
    // Runner present: run the bounded staged smoke (timing is non-deterministic, so
    // this path is not byte-verifiable). video_id from the shared counter.
    let n = st.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    let _ = std::fs::create_dir_all(&out_dir);
    let mut mp4 = out_dir.join("ltx2_t2v_stage2_dev_smoke.mp4");
    let mut wav = String::new();
    if audio_mode == "audio" {
        mp4 = out_dir.join("ltx2_t2v_av_stage2_dev_smoke.mp4");
        wav = out_dir.join("dev_audio.wav").to_string_lossy().into_owned();
    }
    let log_path = out_dir.join("ltx2_video_runner.log");
    let t0 = std::time::Instant::now();
    let rc = std::process::Command::new(RUNNER)
        .args(["staged", "lora", &weight_mode, &audio_mode, "nonag", &out_dir.to_string_lossy(), &steps.to_string()])
        .output()
        .map(|o| o.status.code().unwrap_or(-1))
        .unwrap_or(-1);
    let wall = t0.elapsed().as_secs_f64();
    let mut o = json!({
        "schema": "serenity.video_result.v1", "video_id": video_id, "runner": runner,
        "backend": BACKEND_NAME, "model": "", "resident": "",
        "readiness_label": "bounded_daemon_smoke",
        "accepted_video_artifact": false, "accepted_av_artifact": false,
        "accepted_video_parity": false, "accepted_sampler_parity": false,
        "steps": steps, "mode": format!("staged lora {weight_mode} {audio_mode} nonag"),
        "weight_mode": weight_mode, "audio_mode": audio_mode, "exit_code": rc,
        "out_dir": out_dir.to_string_lossy(), "mp4": mp4.to_string_lossy(), "wav": wav,
        "log_path": log_path.to_string_lossy(),
        "result_path": out_dir.join("ltx2_video_result.json").to_string_lossy(),
        "runner_timing_path": out_dir.join("ltx2_runner_timings.json").to_string_lossy(),
        "total_wall_seconds": wall,
        "note": "Daemon-backed LTX2 staged dev smoke. This proves product wiring only when exit_code is zero and probe.muxing is probe_ok; it does not claim full video parity.",
    });
    if rc != 0 {
        if let Some(m) = o.as_object_mut() {
            m.insert("state".into(), json!("failed"));
            m.insert("error".into(), json!("LTX2 staged smoke runner failed; inspect log_path"));
        }
        return json_resp(StatusCode::INTERNAL_SERVER_ERROR, &o);
    }
    json_resp(StatusCode::OK, &o)
}

fn fps_from_rate(rate: &str) -> f64 {
    // "num/den"
    if let Some((n, d)) = rate.split_once('/') {
        let (n, d) = (n.parse::<f64>().unwrap_or(0.0), d.parse::<f64>().unwrap_or(0.0));
        if d != 0.0 {
            return n / d;
        }
    }
    rate.parse::<f64>().unwrap_or(0.0)
}

/// GET /v1/video/probe?path=<mp4> — ffprobe wrapper reshaped to serenity.video_probe.v1.
pub async fn get_video_probe(Query(q): Query<HashMap<String, String>>) -> Response {
    let mp4 = q.get("path").cloned().unwrap_or_default();
    if mp4.is_empty() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'path' query parameter is required");
    }
    if mp4.contains('\n') || mp4.contains('\r') {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot probe MP4: invalid video path");
    }
    if std::process::Command::new("sh").args(["-c", "command -v ffprobe >/dev/null 2>&1"]).status().map(|s| !s.success()).unwrap_or(true) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot probe MP4: ffprobe is not available on PATH");
    }
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v", "error", "-count_frames", "-show_entries",
            "stream=index,codec_type,codec_name,width,height,nb_frames,nb_read_frames,duration,avg_frame_rate",
            "-show_entries", "format=duration,format_name", "-of", "json", &mp4,
        ])
        .output();
    let out = match out {
        Ok(o) if o.status.success() => o.stdout,
        Ok(o) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("cannot probe MP4: {}", String::from_utf8_lossy(&o.stderr).trim())),
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("cannot probe MP4: {e}")),
    };
    let probe: Value = serde_json::from_slice(&out).unwrap_or_else(|_| json!({}));
    let fstr = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let ffloat = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| x.as_f64())).unwrap_or(0.0);
    let fint = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str().and_then(|s| s.parse::<i64>().ok()).or_else(|| x.as_i64())).unwrap_or(0);
    let (mut fmt_dur, mut fmt_name) = (0.0, String::new());
    if let Some(f) = probe.get("format") {
        fmt_dur = ffloat(f, "duration");
        fmt_name = fstr(f, "format_name");
    }
    let (mut has_video, mut has_audio) = (false, false);
    let (mut w, mut h, mut frames, mut dur, mut fps) = (0i64, 0i64, 0i64, 0.0f64, 0.0f64);
    let (mut vcodec, mut acodec, mut adur) = (String::new(), String::new(), 0.0f64);
    let mut stream_count = 0;
    if let Some(streams) = probe.get("streams").and_then(|s| s.as_array()) {
        stream_count = streams.len() as i64;
        for s in streams {
            match fstr(s, "codec_type").as_str() {
                "video" if !has_video => {
                    has_video = true;
                    w = fint(s, "width"); h = fint(s, "height"); vcodec = fstr(s, "codec_name");
                    dur = ffloat(s, "duration"); fps = fps_from_rate(&fstr(s, "avg_frame_rate"));
                    frames = fint(s, "nb_read_frames");
                    if frames <= 0 { frames = fint(s, "nb_frames"); }
                    if frames <= 0 && dur > 0.0 && fps > 0.0 { frames = (dur * fps + 0.5) as i64; }
                }
                "audio" if !has_audio => {
                    has_audio = true; acodec = fstr(s, "codec_name"); adur = ffloat(s, "duration");
                }
                _ => {}
            }
        }
    }
    if dur <= 0.0 { dur = fmt_dur; }
    let doc = json!({
        "schema": "serenity.video_probe.v1", "mp4": mp4, "format_name": fmt_name,
        "stream_count": stream_count, "has_video": has_video, "has_audio": has_audio, "audio": has_audio,
        "width": w, "height": h, "frame_count": frames, "duration": dur, "fps": fps,
        "video_codec": vcodec, "audio_codec": acodec, "audio_duration": adur,
        "muxing": if has_video && frames > 0 && dur > 0.0 { "probe_ok" } else { "incomplete_probe" },
        "audio_behavior": if has_audio { "audio_stream_present" } else { "video_only_no_audio_stream" },
    });
    json_resp(StatusCode::OK, &doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_shape() {
        let d = readiness_doc();
        assert_eq!(d.get("schema").unwrap(), "serenity.video_status.v1");
        assert_eq!(d.get("endpoint").unwrap(), "/v1/video");
        // runner not built in tests -> runner_missing
        assert_eq!(d.get("state").unwrap(), "runner_missing");
        assert_eq!(d.get("readiness_label").unwrap(), "build_required");
        assert_eq!(d.get("accepted").unwrap(), false);
        let runners = d.get("candidate_runners").unwrap().as_array().unwrap();
        assert_eq!(runners.len(), 2);
        assert_eq!(runners[1].get("model").unwrap(), "ltx2_t2v_av");
        assert_eq!(runners[1].get("target_frame_count").unwrap(), 121);
    }

    #[test]
    fn fps_parse() {
        assert_eq!(fps_from_rate("24/1"), 24.0);
        assert_eq!(fps_from_rate("30000/1001"), 30000.0 / 1001.0);
        assert_eq!(fps_from_rate("0/0"), 0.0);
    }
}
