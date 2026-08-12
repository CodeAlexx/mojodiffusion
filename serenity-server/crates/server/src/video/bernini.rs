//! Bernini-R video profile identity, prerequisites, and product admission.
//!
//! The executor intentionally reuses shared Wan conditioning and process
//! helpers from the parent orchestration module; model-specific policy and
//! sequencing stay here.

use super::*;

/// Bernini-R reuses the admitted UMT5 producer, then runs bounded high/low
/// A14B expert streams and the existing standard-Wan VAE in three isolated
/// Mojo processes. Product exposure remains fail-closed on the local evidence
/// report; Rust is orchestration/readiness only.
pub(super) const BERNINI_T2V: &str = "output/bin/bernini_t2v";
pub(super) const BERNINI_DECODE: &str = "output/bin/bernini_decode";
pub(super) const BERNINI_MODEL_ROOT: &str = "checkpoints/Bernini-R-Diffusers";
pub(super) const BERNINI_ARTIFACT_MANIFEST: &str =
    "checkpoints/Bernini-R-Diffusers/serenity_bernini_r_manifest.json";
pub(super) const BERNINI_HIGH_CACHE: &str =
    "checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/high";
pub(super) const BERNINI_LOW_CACHE: &str =
    "checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/low";
pub(super) const BERNINI_VAE: &str =
    "checkpoints/Bernini-R-Diffusers/vae/diffusion_pytorch_model.safetensors";
pub(super) const BERNINI_PRODUCT_GATE: &str = "output/checks/bernini_r/product_gate.json";
pub(super) const BERNINI_HF_REVISION: &str = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c";
pub(super) const BERNINI_CREATOR_REVISION: &str = "2d2b4591ac053ec25c6371b01a5a6746679e5793";
pub(super) const BERNINI_WIDTH: i64 = 848;
pub(super) const BERNINI_HEIGHT: i64 = 480;
pub(super) const BERNINI_FRAMES: i64 = 81;
pub(super) const BERNINI_FPS: i64 = 16;
pub(super) const BERNINI_DEFAULT_STEPS: i64 = 40;
pub(super) const BERNINI_DEFAULT_GUIDANCE: f64 = 4.0;

fn cache_complete(dir: &str) -> bool {
    let root = std::path::Path::new(dir);
    if !nonempty_file(&root.join("shared.safetensors"))
        || !nonempty_file(&root.join("serenity_cache_manifest.json"))
    {
        return false;
    }
    (0..40).all(|index| nonempty_file(&root.join(format!("block_{index:02}.safetensors"))))
}

pub(super) fn bernini_missing() -> Vec<String> {
    let mut missing = Vec::new();
    for binary in [WAN22_ENCODE, BERNINI_T2V, BERNINI_DECODE] {
        if !bin_x(binary) {
            missing.push(binary.to_string());
        }
    }
    for path in [BERNINI_ARTIFACT_MANIFEST, BERNINI_VAE] {
        let resolved = model_path(path);
        if !nonempty_file(&resolved) {
            missing.push(resolved.to_string_lossy().into_owned());
        }
    }
    let cudnn = serenity_path(MOJO_CUDNN_RUNTIME);
    if !nonempty_file(&cudnn) {
        missing.push(cudnn.to_string_lossy().into_owned());
    }
    for cache in [BERNINI_HIGH_CACHE, BERNINI_LOW_CACHE] {
        let resolved = model_path(cache);
        if !cache_complete(&resolved.to_string_lossy()) {
            missing.push(resolved.to_string_lossy().into_owned());
        }
    }
    missing
}

fn cache_aggregate(dir: &str) -> Option<String> {
    let bytes =
        std::fs::read(std::path::Path::new(dir).join("serenity_cache_manifest.json")).ok()?;
    let document = serde_json::from_slice::<Value>(&bytes).ok()?;
    if document.get("schema").and_then(Value::as_str) != Some("serenity.bernini_r.fp8_cache.v1")
        || document.get("passed").and_then(Value::as_bool) != Some(true)
        || document.get("revision").and_then(Value::as_str) != Some(BERNINI_HF_REVISION)
    {
        return None;
    }
    document
        .get("cache_aggregate_sha256")
        .and_then(Value::as_str)
        .filter(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(str::to_string)
}

pub(crate) fn bernini_product_gate_passed() -> bool {
    let Ok(bytes) = std::fs::read(repo_path(BERNINI_PRODUCT_GATE)) else {
        return false;
    };
    let Ok(document) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    let Some(high_cache) = cache_aggregate(&model_path(BERNINI_HIGH_CACHE).to_string_lossy())
    else {
        return false;
    };
    let Some(low_cache) = cache_aggregate(&model_path(BERNINI_LOW_CACHE).to_string_lossy()) else {
        return false;
    };
    let Some(encode_runner) = sha256sum(&repo_path(WAN22_ENCODE)) else {
        return false;
    };
    let Some(denoise_runner) = sha256sum(&repo_path(BERNINI_T2V)) else {
        return false;
    };
    let Some(decode_runner) = sha256sum(&repo_path(BERNINI_DECODE)) else {
        return false;
    };
    let Some(cudnn_runtime) = sha256sum(&serenity_path(MOJO_CUDNN_RUNTIME)) else {
        return false;
    };
    document.get("schema").and_then(Value::as_str) == Some("serenity.bernini_r.product_gate.v1")
        && document.get("passed").and_then(Value::as_bool) == Some(true)
        && document
            .pointer("/pins/hf_revision")
            .and_then(Value::as_str)
            == Some(BERNINI_HF_REVISION)
        && document
            .pointer("/pins/creator_revision")
            .and_then(Value::as_str)
            == Some(BERNINI_CREATOR_REVISION)
        && document
            .pointer("/pins/high_cache_aggregate_sha256")
            .and_then(Value::as_str)
            == Some(high_cache.as_str())
        && document
            .pointer("/pins/low_cache_aggregate_sha256")
            .and_then(Value::as_str)
            == Some(low_cache.as_str())
        && document
            .pointer("/pins/encode_runner_sha256")
            .and_then(Value::as_str)
            == Some(encode_runner.as_str())
        && document
            .pointer("/pins/denoise_runner_sha256")
            .and_then(Value::as_str)
            == Some(denoise_runner.as_str())
        && document
            .pointer("/pins/decode_runner_sha256")
            .and_then(Value::as_str)
            == Some(decode_runner.as_str())
        && document
            .pointer("/pins/cudnn_runtime_sha256")
            .and_then(Value::as_str)
            == Some(cudnn_runtime.as_str())
        && document.pointer("/profile/width").and_then(Value::as_i64) == Some(BERNINI_WIDTH)
        && document.pointer("/profile/height").and_then(Value::as_i64) == Some(BERNINI_HEIGHT)
        && document.pointer("/profile/frames").and_then(Value::as_i64) == Some(BERNINI_FRAMES)
        && document.pointer("/profile/fps").and_then(Value::as_i64) == Some(BERNINI_FPS)
        && document.pointer("/profile/steps").and_then(Value::as_i64) == Some(BERNINI_DEFAULT_STEPS)
        && document
            .pointer("/performance/requires_isolated_process_stages")
            .and_then(Value::as_bool)
            == Some(true)
}

/// Bernini-R production arm. Rust only validates and sequences the existing
/// Mojo stages; all model, sampler, APG, and VAE math remains Mojo-owned.
pub(super) fn post_video_bernini(st: &AppState, b: &Value) -> Response {
    let prompt = b
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if prompt.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "bernini: 'prompt' is required",
        );
    }
    let requested_negative = b
        .get("negative_prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let negative_prompt = if requested_negative.is_empty() {
        WAN22_DEFAULT_NEGATIVE.to_string()
    } else {
        requested_negative
    };
    let quant = b.get("quant").and_then(Value::as_str).unwrap_or("fp8");
    if quant != "fp8" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Bernini-R requires its bounded FP8 E4M3 expert caches; quant '{quant}' is unsupported"
            ),
        );
    }
    let width = b
        .get("width")
        .and_then(Value::as_i64)
        .unwrap_or(BERNINI_WIDTH);
    let height = b
        .get("height")
        .and_then(Value::as_i64)
        .unwrap_or(BERNINI_HEIGHT);
    let frames = b
        .get("frames")
        .and_then(Value::as_i64)
        .unwrap_or(BERNINI_FRAMES);
    let fps = b.get("fps").and_then(Value::as_i64).unwrap_or(BERNINI_FPS);
    let steps = b
        .get("steps")
        .and_then(Value::as_i64)
        .unwrap_or(BERNINI_DEFAULT_STEPS);
    let guidance = b
        .get("guidance")
        .or_else(|| b.get("cfg"))
        .and_then(Value::as_f64)
        .unwrap_or(BERNINI_DEFAULT_GUIDANCE);
    if width != BERNINI_WIDTH
        || height != BERNINI_HEIGHT
        || frames != BERNINI_FRAMES
        || fps != BERNINI_FPS
        || steps != BERNINI_DEFAULT_STEPS
        || (guidance - BERNINI_DEFAULT_GUIDANCE).abs() > f64::EPSILON
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Bernini-R creator-gated profile is exactly {BERNINI_WIDTH}x{BERNINI_HEIGHT}, {BERNINI_FRAMES} frames, {BERNINI_FPS} fps, {BERNINI_DEFAULT_STEPS} steps, APG {BERNINI_DEFAULT_GUIDANCE}; requested {width}x{height}, {frames} frames, {fps} fps, {steps} steps, guidance {guidance}"
            ),
        );
    }
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };
    let missing = bernini_missing();
    if !missing.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Bernini-R prerequisites missing: {}", missing.join(", ")),
        );
    }
    if !bernini_product_gate_passed() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Bernini-R is installed but its pinned creator-parity and representative-render product gate has not passed",
        );
    }

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    if let Err(error) = std::fs::create_dir_all(&out_dir) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot create Bernini-R output directory: {error}"),
        );
    }
    let conds = out_dir.join("bernini_conds.safetensors");
    let latent = out_dir.join("bernini_latent.safetensors");
    let mp4 = out_dir.join("bernini_r_t2v.mp4");
    let encode_log = out_dir.join("bernini_encode.log");
    let denoise_log = out_dir.join("bernini_denoise.log");
    let decode_log = out_dir.join("bernini_decode.log");
    let result_path = out_dir.join("bernini_video_result.json");

    let encode_started = std::time::Instant::now();
    let mut encode = wan22_command(&repo_path(WAN22_ENCODE));
    encode.arg(&prompt).arg(&negative_prompt).arg(&conds);
    let (encode_rc, encode_peak_vram_mib) = match run_logged_with_gpu_peak(&mut encode, &encode_log)
    {
        Ok(measured) => measured,
        Err(error) => {
            let _ = std::fs::write(&encode_log, format!("spawn failed: {error}"));
            (-1, None)
        }
    };
    let encode_seconds = encode_started.elapsed().as_secs_f64();
    if encode_rc != 0 || !nonempty_file(&conds) {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id,
                "model": "bernini", "state": "failed", "failed_step": "encode",
                "encode_exit_code": encode_rc, "encode_log": encode_log.to_string_lossy(),
                "encode_seconds": encode_seconds, "encode_peak_vram_mib": encode_peak_vram_mib,
                "error": "pinned UMT5 conditioning producer failed",
            }),
        );
    }

    let denoise_started = std::time::Instant::now();
    let mut denoise = wan22_command(&repo_path(BERNINI_T2V));
    let high_cache = model_path(BERNINI_HIGH_CACHE);
    let low_cache = model_path(BERNINI_LOW_CACHE);
    denoise
        .arg(&conds)
        .arg(&high_cache)
        .arg(&low_cache)
        .arg(&latent)
        .arg(steps.to_string())
        .arg(seed.to_string());
    let (denoise_rc, denoise_peak_vram_mib) =
        match run_logged_with_gpu_peak(&mut denoise, &denoise_log) {
            Ok(measured) => measured,
            Err(error) => {
                let _ = std::fs::write(&denoise_log, format!("spawn failed: {error}"));
                (-1, None)
            }
        };
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    if denoise_rc != 0 || !nonempty_file(&latent) {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id,
                "model": "bernini", "state": "failed", "failed_step": "denoise",
                "encode_exit_code": encode_rc, "denoise_exit_code": denoise_rc,
                "conds": conds.to_string_lossy(), "latent": latent.to_string_lossy(),
                "encode_log": encode_log.to_string_lossy(), "denoise_log": denoise_log.to_string_lossy(),
                "encode_seconds": encode_seconds, "denoise_seconds": denoise_seconds,
                "encode_peak_vram_mib": encode_peak_vram_mib,
                "denoise_peak_vram_mib": denoise_peak_vram_mib,
                "error": "Bernini-R dual-expert Mojo denoise failed",
            }),
        );
    }

    let decode_started = std::time::Instant::now();
    let mut decode = wan22_command(&repo_path(BERNINI_DECODE));
    decode
        .arg(&latent)
        .arg(model_path(BERNINI_VAE))
        .arg(&out_dir);
    let (decode_rc, decode_peak_vram_mib) = match run_logged_with_gpu_peak(&mut decode, &decode_log)
    {
        Ok(measured) => measured,
        Err(error) => {
            let _ = std::fs::write(&decode_log, format!("spawn failed: {error}"));
            (-1, None)
        }
    };
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let probe = if decode_rc == 0 && mp4.is_file() {
        probe_video_path(&mp4.to_string_lossy())
            .unwrap_or_else(|error| json!({ "muxing": "probe_failed", "error": error }))
    } else {
        json!({ "muxing": "artifact_missing" })
    };
    let frames_written = count_wan22_frames(&out_dir);
    let artifact_ok = decode_rc == 0
        && mp4.is_file()
        && frames_written == BERNINI_FRAMES as usize
        && probe.get("muxing").and_then(Value::as_str) == Some("probe_ok")
        && probe.get("width").and_then(Value::as_i64) == Some(BERNINI_WIDTH)
        && probe.get("height").and_then(Value::as_i64) == Some(BERNINI_HEIGHT)
        && probe.get("frame_count").and_then(Value::as_i64) == Some(BERNINI_FRAMES)
        && (probe.get("fps").and_then(Value::as_f64).unwrap_or(0.0) - BERNINI_FPS as f64).abs()
            < 0.01
        && probe.get("has_audio").and_then(Value::as_bool) == Some(false);
    let total_wall_seconds = encode_seconds + denoise_seconds + decode_seconds;
    let result = json!({
        "schema": "serenity.video_result.v1", "video_id": video_id,
        "backend": BACKEND_NAME, "control_plane": "serenity-server",
        "model": "bernini", "runner": "bernini_r_t2v",
        "state": if artifact_ok { "done" } else { "failed" },
        "readiness_label": "quality_profile_ready",
        "accepted_video_artifact": artifact_ok,
        "accepted_video_parity": bernini_product_gate_passed(),
        "prompt": prompt, "negative_prompt": negative_prompt,
        "negative_prompt_source": if b.get("negative_prompt").and_then(Value::as_str).is_some_and(|value| !value.trim().is_empty()) { "request" } else { "creator_default" },
        "width": width, "height": height, "frames": frames,
        "frames_written": frames_written, "fps": fps, "steps": steps,
        "seed": seed, "guidance": "t2v_apg_4.0_then_3.2", "quant": quant,
        "processes": ["encode", "dual_expert_denoise", "vae_decode_mux"],
        "process_separated_decode": true,
        "encode_exit_code": encode_rc, "denoise_exit_code": denoise_rc,
        "decode_exit_code": decode_rc,
        "out_dir": out_dir.to_string_lossy(), "conds": conds.to_string_lossy(),
        "latent": latent.to_string_lossy(), "mp4": mp4.to_string_lossy(),
        "mp4_url": if artifact_ok { format!("/out/{video_id}/bernini_r_t2v.mp4") } else { String::new() },
        "probe": probe,
        "encode_log": encode_log.to_string_lossy(),
        "denoise_log": denoise_log.to_string_lossy(),
        "decode_log": decode_log.to_string_lossy(),
        "result_path": result_path.to_string_lossy(),
        "encode_seconds": encode_seconds, "denoise_seconds": denoise_seconds,
        "decode_seconds": decode_seconds, "total_wall_seconds": total_wall_seconds,
        "encode_peak_vram_mib": encode_peak_vram_mib,
        "denoise_peak_vram_mib": denoise_peak_vram_mib,
        "decode_peak_vram_mib": decode_peak_vram_mib,
        "note": "Pinned ByteDance Bernini-R creator profile: exact UMT5 conditioning, high/low A14B streamed Mojo experts, creator UniPC/APG, and fresh-process standard-Wan temporal VAE decode.",
    });
    let _ = std::fs::write(
        &result_path,
        serde_json::to_vec_pretty(&result).unwrap_or_default(),
    );
    json_resp(
        if artifact_ok {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &result,
    )
}
