//! Video artifact probing and profile verification.
//!
//! This module owns the ffprobe boundary shared by every video backend. Keeping
//! it independent prevents model-specific orchestration from growing another
//! copy of the same parsing and validation logic.

use super::*;

pub(super) fn probe_matches_video_profile(
    probe: &Value,
    width: i64,
    height: i64,
    frames: i64,
    fps: i64,
    has_audio: bool,
) -> bool {
    probe.get("muxing").and_then(Value::as_str) == Some("probe_ok")
        && probe.get("width").and_then(Value::as_i64) == Some(width)
        && probe.get("height").and_then(Value::as_i64) == Some(height)
        && probe.get("frame_count").and_then(Value::as_i64) == Some(frames)
        && probe.get("fps").and_then(Value::as_f64) == Some(fps as f64)
        && probe.get("has_audio").and_then(Value::as_bool) == Some(has_audio)
}

pub(super) fn fps_from_rate(rate: &str) -> f64 {
    if let Some((numerator, denominator)) = rate.split_once('/') {
        let numerator = numerator.parse::<f64>().unwrap_or(0.0);
        let denominator = denominator.parse::<f64>().unwrap_or(0.0);
        if denominator != 0.0 {
            return numerator / denominator;
        }
    }
    rate.parse::<f64>().unwrap_or(0.0)
}

pub(crate) fn probe_video_path(mp4: &str) -> Result<Value, String> {
    if mp4.contains('\n') || mp4.contains('\r') {
        return Err("cannot probe MP4: invalid video path".to_string());
    }
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v", "error", "-count_frames", "-show_entries",
            "stream=index,codec_type,codec_name,width,height,nb_frames,nb_read_frames,duration,avg_frame_rate,sample_rate,channels",
            "-show_entries", "format=duration,format_name", "-of", "json", mp4,
        ])
        .output()
        .map_err(|error| format!("cannot probe MP4: {error}"))?;
    if !out.status.success() {
        return Err(format!(
            "cannot probe MP4: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let probe: Value = serde_json::from_slice(&out.stdout)
        .map_err(|error| format!("cannot parse ffprobe response: {error}"))?;
    let string_field = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let float_field = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(|field| {
                field
                    .as_str()
                    .and_then(|text| text.parse::<f64>().ok())
                    .or_else(|| field.as_f64())
            })
            .unwrap_or(0.0)
    };
    let integer_field = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(|field| {
                field
                    .as_str()
                    .and_then(|text| text.parse::<i64>().ok())
                    .or_else(|| field.as_i64())
            })
            .unwrap_or(0)
    };
    let (mut format_duration, mut format_name) = (0.0, String::new());
    if let Some(format) = probe.get("format") {
        format_duration = float_field(format, "duration");
        format_name = string_field(format, "format_name");
    }
    let (mut has_video, mut has_audio) = (false, false);
    let (mut width, mut height, mut frames, mut duration, mut fps) =
        (0_i64, 0_i64, 0_i64, 0.0_f64, 0.0_f64);
    let (mut video_codec, mut audio_codec, mut audio_duration) =
        (String::new(), String::new(), 0.0_f64);
    let (mut audio_sample_rate, mut audio_channels) = (0_i64, 0_i64);
    let mut stream_count = 0;
    if let Some(streams) = probe.get("streams").and_then(Value::as_array) {
        stream_count = streams.len() as i64;
        for stream in streams {
            match string_field(stream, "codec_type").as_str() {
                "video" if !has_video => {
                    has_video = true;
                    width = integer_field(stream, "width");
                    height = integer_field(stream, "height");
                    video_codec = string_field(stream, "codec_name");
                    duration = float_field(stream, "duration");
                    fps = fps_from_rate(&string_field(stream, "avg_frame_rate"));
                    frames = integer_field(stream, "nb_read_frames");
                    if frames <= 0 {
                        frames = integer_field(stream, "nb_frames");
                    }
                    if frames <= 0 && duration > 0.0 && fps > 0.0 {
                        frames = (duration * fps + 0.5) as i64;
                    }
                }
                "audio" if !has_audio => {
                    has_audio = true;
                    audio_codec = string_field(stream, "codec_name");
                    audio_duration = float_field(stream, "duration");
                    audio_sample_rate = integer_field(stream, "sample_rate");
                    audio_channels = integer_field(stream, "channels");
                }
                _ => {}
            }
        }
    }
    if duration <= 0.0 {
        duration = format_duration;
    }
    Ok(json!({
        "schema": "serenity.video_probe.v1", "mp4": mp4, "format_name": format_name,
        "stream_count": stream_count, "has_video": has_video, "has_audio": has_audio, "audio": has_audio,
        "width": width, "height": height, "frame_count": frames, "duration": duration, "fps": fps,
        "video_codec": video_codec, "audio_codec": audio_codec, "audio_duration": audio_duration,
        "audio_sample_rate": audio_sample_rate, "audio_channels": audio_channels,
        "muxing": if has_video && frames > 0 && duration > 0.0 { "probe_ok" } else { "incomplete_probe" },
        "audio_behavior": if has_audio { "audio_stream_present" } else { "video_only_no_audio_stream" },
    }))
}

/// GET /v1/video/probe?path=<mp4> — ffprobe wrapper reshaped to
/// `serenity.video_probe.v1`.
pub(crate) async fn get_video_probe(Query(query): Query<HashMap<String, String>>) -> Response {
    let mp4 = query.get("path").cloned().unwrap_or_default();
    if mp4.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'path' query parameter is required",
        );
    }
    match probe_video_path(&mp4) {
        Ok(document) => json_resp(StatusCode::OK, &document),
        Err(error) => err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    }
}
