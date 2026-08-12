//! SCAIL-2 video admission and isolated Mojo stage orchestration.

use super::*;

/// SCAIL-2 single-segment character animation. Every conditioning artifact is
/// produced automatically per request; only the installed model/cache and four
/// user media inputs are prerequisites.
pub(super) const SCAIL2_STAGE: &str = "output/bin/scail2_stage_inputs";
pub(super) const SCAIL2_ENCODE_PROMPT: &str = "output/bin/scail2_encode_prompt";
pub(super) const SCAIL2_ENCODE_CLIP: &str = "output/bin/scail2_encode_clip";
pub(super) const SCAIL2_ENCODE_VAE: &str = "output/bin/scail2_encode_vae";
pub(super) const SCAIL2_PREPARE_CACHE: &str = "output/bin/scail2_prepare_fp8_cache";
pub(super) const SCAIL2_ANIMATION: &str = "output/bin/scail2_animation";
pub(super) const SCAIL2_DECODE: &str = "output/bin/scail2_decode";
pub(super) const SCAIL2_OFFICIAL_ROOT: &str = "checkpoints/SCAIL-2";
pub(super) const SCAIL2_MOJO_ROOT: &str = "checkpoints/SCAIL-2-Mojo";
pub(super) const SCAIL2_UMT5: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5";
pub(super) const SCAIL2_TOKENIZER: &str = "checkpoints/SCAIL-2/umt5-xxl/tokenizer.json";
pub(super) const SCAIL2_CLIP: &str = "checkpoints/SCAIL-2-Mojo/clip_visual/model.safetensors";
pub(super) const SCAIL2_FP8_CACHE: &str = "checkpoints/SCAIL-2-Mojo/transformer_fp8";
pub(super) const SCAIL2_VAE: &str = BERNINI_VAE;
pub(super) const SCAIL2_PRODUCT_GATE: &str = "output/checks/scail2/product_gate.json";
pub(super) const SCAIL2_SOURCE_COMMIT: &str = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5";
pub(super) const SCAIL2_MODEL_REVISION: &str = "150cc0ca4e98e50e60b9295dacde39442fdccab2";
pub(super) const SCAIL2_CHECKPOINT_SHA256: &str =
    "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f";
pub(super) const SCAIL2_WIDTH: i64 = 896;
pub(super) const SCAIL2_HEIGHT: i64 = 512;
pub(super) const SCAIL2_FRAMES: i64 = 65;
pub(super) const SCAIL2_FPS: i64 = 16;
pub(super) const SCAIL2_STEPS: i64 = 40;
pub(super) const SCAIL2_GUIDANCE: f64 = 5.0;

pub(super) fn scail2_cache_complete() -> bool {
    let root = model_path(SCAIL2_FP8_CACHE);
    if !nonempty_file(&root.join("shared.safetensors"))
        || !nonempty_file(&root.join("shared.safetensors.sha256"))
        || !nonempty_file(&root.join("source_provenance.sha256"))
    {
        return false;
    }
    let Ok(provenance) = std::fs::read_to_string(root.join("source_provenance.sha256")) else {
        return false;
    };
    if !provenance.contains(&format!("source_commit={SCAIL2_SOURCE_COMMIT}"))
        || !provenance.contains(&format!("model_revision={SCAIL2_MODEL_REVISION}"))
        || !provenance.contains(&format!("checkpoint_sha256={SCAIL2_CHECKPOINT_SHA256}"))
    {
        return false;
    }
    (0..40).all(|index| {
        nonempty_file(&root.join(format!("block_{index:02}.safetensors")))
            && nonempty_file(&root.join(format!("block_{index:02}.safetensors.sha256")))
    })
}

pub(super) fn scail2_missing() -> Vec<String> {
    let mut missing = Vec::new();
    for binary in [
        SCAIL2_STAGE,
        SCAIL2_ENCODE_PROMPT,
        SCAIL2_ENCODE_CLIP,
        SCAIL2_ENCODE_VAE,
        SCAIL2_PREPARE_CACHE,
        SCAIL2_ANIMATION,
        SCAIL2_DECODE,
    ] {
        if !bin_x(binary) {
            missing.push(binary.to_string());
        }
    }
    for path in [SCAIL2_TOKENIZER, SCAIL2_CLIP, SCAIL2_VAE] {
        let resolved = model_path(path);
        if !nonempty_file(&resolved) {
            missing.push(resolved.to_string_lossy().into_owned());
        }
    }
    let umt5 = model_path(SCAIL2_UMT5);
    if !nonempty_file(&umt5.join("model.safetensors.index.json")) {
        missing.push(umt5.to_string_lossy().into_owned());
    }
    if !scail2_cache_complete() {
        missing.push(model_path(SCAIL2_FP8_CACHE).to_string_lossy().into_owned());
    }
    let cudnn = serenity_path(MOJO_CUDNN_RUNTIME);
    if !nonempty_file(&cudnn) {
        missing.push(cudnn.to_string_lossy().into_owned());
    }
    missing
}

pub(crate) fn scail2_product_gate_passed() -> bool {
    let Ok(bytes) = std::fs::read(repo_path(SCAIL2_PRODUCT_GATE)) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    let binaries = [
        ("stage_runner_sha256", SCAIL2_STAGE),
        ("prompt_runner_sha256", SCAIL2_ENCODE_PROMPT),
        ("clip_runner_sha256", SCAIL2_ENCODE_CLIP),
        ("vae_runner_sha256", SCAIL2_ENCODE_VAE),
        ("cache_runner_sha256", SCAIL2_PREPARE_CACHE),
        ("denoise_runner_sha256", SCAIL2_ANIMATION),
        ("decode_runner_sha256", SCAIL2_DECODE),
    ];
    if binaries.iter().any(|(key, path)| {
        let Some(actual) = sha256sum(&repo_path(path)) else {
            return false;
        };
        doc.pointer(&format!("/pins/{key}")).and_then(Value::as_str) != Some(actual.as_str())
    }) {
        return false;
    }
    doc.get("schema").and_then(Value::as_str) == Some("serenity.scail2.product_gate.v1")
        && doc.get("passed").and_then(Value::as_bool) == Some(true)
        && doc.pointer("/pins/source_commit").and_then(Value::as_str) == Some(SCAIL2_SOURCE_COMMIT)
        && doc.pointer("/pins/model_revision").and_then(Value::as_str)
            == Some(SCAIL2_MODEL_REVISION)
        && doc
            .pointer("/pins/checkpoint_sha256")
            .and_then(Value::as_str)
            == Some(SCAIL2_CHECKPOINT_SHA256)
        && doc.pointer("/profile/width").and_then(Value::as_i64) == Some(SCAIL2_WIDTH)
        && doc.pointer("/profile/height").and_then(Value::as_i64) == Some(SCAIL2_HEIGHT)
        && doc.pointer("/profile/frames").and_then(Value::as_i64) == Some(SCAIL2_FRAMES)
        && doc.pointer("/profile/fps").and_then(Value::as_i64) == Some(SCAIL2_FPS)
        && doc.pointer("/profile/steps").and_then(Value::as_i64) == Some(SCAIL2_STEPS)
        && doc.pointer("/profile/guidance").and_then(Value::as_f64) == Some(SCAIL2_GUIDANCE)
        && doc
            .pointer("/evidence/visual_inspection_passed")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/performance/requires_isolated_process_stages")
            .and_then(Value::as_bool)
            == Some(true)
}

fn run_scail2_stage(
    binary: &str,
    args: &[&std::ffi::OsStr],
    log_path: &std::path::Path,
) -> (i32, Option<u64>, f64) {
    let started = std::time::Instant::now();
    let mut command = wan22_command(&repo_path(binary));
    command.args(args);
    let measured = run_logged_with_gpu_peak(&mut command, log_path);
    let seconds = started.elapsed().as_secs_f64();
    match measured {
        Ok((code, peak)) => (code, peak, seconds),
        Err(error) => {
            let _ = std::fs::write(log_path, format!("spawn failed: {error}"));
            (-1, None, seconds)
        }
    }
}

fn scail2_failed(
    video_id: &str,
    out_dir: &std::path::Path,
    stage: &str,
    code: i32,
    log_path: &std::path::Path,
    seconds: f64,
    peak_vram_mib: Option<u64>,
) -> Response {
    json_resp(
        StatusCode::INTERNAL_SERVER_ERROR,
        &json!({
            "schema": "serenity.video_result.v1",
            "video_id": video_id,
            "model": "scail2",
            "state": "failed",
            "failed_step": stage,
            "exit_code": code,
            "log": log_path.to_string_lossy(),
            "stage_seconds": seconds,
            "stage_peak_vram_mib": peak_vram_mib,
            "out_dir": out_dir.to_string_lossy(),
            "error": format!("SCAIL-2 {stage} failed; inspect the stage log"),
        }),
    )
}

pub(super) struct Scail2RunPaths {
    pub(super) run_dir: std::path::PathBuf,
    pub(super) decode_dir: std::path::PathBuf,
    pub(super) public_dir: std::path::PathBuf,
    pub(super) decoded_mp4: std::path::PathBuf,
    pub(super) public_mp4: std::path::PathBuf,
}

pub(super) fn scail2_run_paths(
    runtime_root: &std::path::Path,
    output_root: &std::path::Path,
    video_id: &str,
) -> Scail2RunPaths {
    let run_dir = runtime_root.join("runs").join("scail2").join(video_id);
    let decode_dir = run_dir.join("decode");
    let public_dir = output_root.join(video_id);
    Scail2RunPaths {
        decoded_mp4: decode_dir.join("scail2_animation.mp4"),
        public_mp4: public_dir.join("scail2_animation.mp4"),
        run_dir,
        decode_dir,
        public_dir,
    }
}

pub(super) fn publish_scail2_mp4(paths: &Scail2RunPaths) -> Result<(), String> {
    std::fs::create_dir_all(&paths.public_dir)
        .map_err(|error| format!("cannot create SCAIL-2 output directory: {error}"))?;
    let partial = paths.public_dir.join(".scail2_animation.mp4.part");
    if let Err(error) = std::fs::copy(&paths.decoded_mp4, &partial) {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("cannot publish SCAIL-2 MP4: {error}"));
    }
    if let Err(error) = std::fs::rename(&partial, &paths.public_mp4) {
        let _ = std::fs::remove_file(&partial);
        return Err(format!("cannot finalize SCAIL-2 MP4: {error}"));
    }
    let _ = std::fs::remove_file(&paths.decoded_mp4);
    Ok(())
}

/// SCAIL-2 production arm. The request supplies media and text only. Serenity
/// creates every stage/cache-consumer artifact, conditions the model, denoises,
/// decodes, and preserves driving-video audio without assistant intervention.
pub(super) fn post_video_scail2(st: &AppState, b: &Value) -> Response {
    let prompt = b
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if prompt.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "scail2: 'prompt' is required",
        );
    }
    let negative_prompt = b
        .get("negative_prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let mode = b
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("animation")
        .trim();
    if mode != "animation" && mode != "replacement" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "scail2: 'mode' must be animation or replacement",
        );
    }
    let width = b
        .get("width")
        .and_then(Value::as_i64)
        .unwrap_or(SCAIL2_WIDTH);
    let height = b
        .get("height")
        .and_then(Value::as_i64)
        .unwrap_or(SCAIL2_HEIGHT);
    let frames = b
        .get("frames")
        .and_then(Value::as_i64)
        .unwrap_or(SCAIL2_FRAMES);
    let fps = b.get("fps").and_then(Value::as_i64).unwrap_or(SCAIL2_FPS);
    let steps = b
        .get("steps")
        .and_then(Value::as_i64)
        .unwrap_or(SCAIL2_STEPS);
    let guidance = b
        .get("guidance")
        .or_else(|| b.get("cfg"))
        .and_then(Value::as_f64)
        .unwrap_or(SCAIL2_GUIDANCE);
    if width != SCAIL2_WIDTH
        || height != SCAIL2_HEIGHT
        || frames != SCAIL2_FRAMES
        || fps != SCAIL2_FPS
        || steps != SCAIL2_STEPS
        || (guidance - SCAIL2_GUIDANCE).abs() > f64::EPSILON
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "SCAIL-2 admitted profile is exactly {SCAIL2_WIDTH}x{SCAIL2_HEIGHT}, {SCAIL2_FRAMES} frames, {SCAIL2_FPS} fps, {SCAIL2_STEPS} steps, CFG {SCAIL2_GUIDANCE}; requested {width}x{height}, {frames} frames, {fps} fps, {steps} steps, CFG {guidance}"
            ),
        );
    }
    let quant = b.get("quant").and_then(Value::as_str).unwrap_or("fp8");
    if quant != "fp8" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("SCAIL-2 requires its bounded FP8 E4M3 cache; quant '{quant}' is unsupported"),
        );
    }
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };

    let aliases = [
        ("reference_image", "reference_image"),
        ("reference_mask", "reference_mask"),
        ("driving_video", "driving_video"),
        ("driving_mask_video", "driving_mask_video"),
    ];
    let mut media = HashMap::<&str, std::path::PathBuf>::new();
    for (key, label) in aliases {
        let raw = b.get(key).and_then(Value::as_str).unwrap_or("").trim();
        if raw.is_empty() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!("scail2: '{label}' is required"),
            );
        }
        if raw.contains('\n') || raw.contains('\r') {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!("scail2: '{label}' contains an invalid path"),
            );
        }
        let requested = std::path::PathBuf::from(raw);
        let resolved = if requested.is_absolute() {
            requested
        } else {
            repo_path(raw)
        };
        if !nonempty_file(&resolved) {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "scail2: '{label}' is missing or empty: {}",
                    resolved.display()
                ),
            );
        }
        media.insert(key, resolved);
    }
    let reference_image = &media["reference_image"];
    let reference_mask = &media["reference_mask"];
    let driving_video = &media["driving_video"];
    let driving_mask_video = &media["driving_mask_video"];
    let additional_image_values = b
        .get("additional_reference_images")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let additional_mask_values = b
        .get("additional_reference_masks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if additional_image_values.len() != additional_mask_values.len()
        || additional_image_values.len() > 3
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "scail2: additional reference images and masks must be paired (maximum 3)",
        );
    }
    let resolve_additional = |values: &[Value], label: &str| {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let raw = value
                    .as_str()
                    .ok_or_else(|| format!("scail2: {label}[{index}] must be a path string"))?
                    .trim();
                if raw.is_empty() || raw.contains('\n') || raw.contains('\r') {
                    return Err(format!("scail2: {label}[{index}] is invalid"));
                }
                let requested = std::path::PathBuf::from(raw);
                let resolved = if requested.is_absolute() {
                    requested
                } else {
                    repo_path(raw)
                };
                if !nonempty_file(&resolved) {
                    return Err(format!(
                        "scail2: {label}[{index}] is missing or empty: {}",
                        resolved.display()
                    ));
                }
                Ok(resolved)
            })
            .collect::<Result<Vec<_>, String>>()
    };
    let additional_images =
        match resolve_additional(&additional_image_values, "additional_reference_images") {
            Ok(paths) => paths,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
    let additional_masks =
        match resolve_additional(&additional_mask_values, "additional_reference_masks") {
            Ok(paths) => paths,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
    let input_probe = match probe_video_path(&driving_video.to_string_lossy()) {
        Ok(probe) if probe.get("has_video").and_then(Value::as_bool) == Some(true) => probe,
        Ok(_) => {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "scail2: driving_video has no readable video stream",
            );
        }
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    if probe_video_path(&driving_mask_video.to_string_lossy())
        .ok()
        .and_then(|probe| probe.get("has_video").and_then(Value::as_bool))
        != Some(true)
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "scail2: driving_mask_video has no readable video stream",
        );
    }

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let paths = scail2_run_paths(&serenity_root(), &st.out_dir, &video_id);
    let out_dir = &paths.run_dir;
    if let Err(error) = std::fs::create_dir_all(out_dir) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot create SCAIL-2 internal run directory: {error}"),
        );
    }
    let stage = out_dir.join("scail2_stage.safetensors");
    let text = out_dir.join("scail2_text_context.safetensors");
    let clip = out_dir.join("scail2_clip_context.safetensors");
    let reference_latent = out_dir.join("scail2_reference_latent.safetensors");
    let pose_latent = out_dir.join("scail2_pose_latent.safetensors");
    let additional_latent = out_dir.join("scail2_additional_reference_latent.safetensors");
    let latent = out_dir.join("scail2_latent.safetensors");
    let result_dir = &paths.decode_dir;
    let decoded_mp4 = &paths.decoded_mp4;
    let mp4 = &paths.public_mp4;
    let result_path = out_dir.join("scail2_video_result.json");
    let umt5_path = model_path(SCAIL2_UMT5);
    let tokenizer_path = model_path(SCAIL2_TOKENIZER);
    let clip_model_path = model_path(SCAIL2_CLIP);
    let vae_path = model_path(SCAIL2_VAE);
    let fp8_cache_path = model_path(SCAIL2_FP8_CACHE);
    let log = |name: &str| out_dir.join(format!("scail2_{name}.log"));

    let mut timings = serde_json::Map::new();
    let mut peaks = serde_json::Map::new();
    let mut exits = serde_json::Map::new();
    let mut logs = serde_json::Map::new();
    let mut run =
        |name: &str, binary: &str, args: &[&std::ffi::OsStr], expected: &std::path::Path| {
            let log_path = log(name);
            logs.insert(name.to_string(), json!(log_path.to_string_lossy()));
            let (code, peak, seconds) = run_scail2_stage(binary, args, &log_path);
            timings.insert(name.to_string(), json!(seconds));
            peaks.insert(name.to_string(), json!(peak));
            exits.insert(name.to_string(), json!(code));
            if code != 0 || !nonempty_file(expected) {
                Some(scail2_failed(
                    &video_id, &out_dir, name, code, &log_path, seconds, peak,
                ))
            } else {
                None
            }
        };

    let height_arg = height.to_string();
    let width_arg = width.to_string();
    let frames_arg = frames.to_string();
    let mut stage_args = vec![
        reference_image.as_os_str(),
        reference_mask.as_os_str(),
        driving_video.as_os_str(),
        driving_mask_video.as_os_str(),
        stage.as_os_str(),
        std::ffi::OsStr::new(&height_arg),
        std::ffi::OsStr::new(&width_arg),
        std::ffi::OsStr::new(&frames_arg),
    ];
    for (image, mask) in additional_images.iter().zip(&additional_masks) {
        stage_args.push(image.as_os_str());
        stage_args.push(mask.as_os_str());
    }
    if let Some(response) = run("stage_inputs", SCAIL2_STAGE, &stage_args, &stage) {
        return response;
    }
    if let Some(response) = run(
        "encode_prompt",
        SCAIL2_ENCODE_PROMPT,
        &[
            umt5_path.as_os_str(),
            tokenizer_path.as_os_str(),
            std::ffi::OsStr::new(&prompt),
            std::ffi::OsStr::new(&negative_prompt),
            text.as_os_str(),
        ],
        &text,
    ) {
        return response;
    }
    if let Some(response) = run(
        "encode_clip",
        SCAIL2_ENCODE_CLIP,
        &[
            clip_model_path.as_os_str(),
            stage.as_os_str(),
            clip.as_os_str(),
        ],
        &clip,
    ) {
        return response;
    }
    if let Some(response) = run(
        "encode_reference",
        SCAIL2_ENCODE_VAE,
        &[
            std::ffi::OsStr::new("ref"),
            stage.as_os_str(),
            vae_path.as_os_str(),
            reference_latent.as_os_str(),
        ],
        &reference_latent,
    ) {
        return response;
    }
    if let Some(response) = run(
        "encode_pose",
        SCAIL2_ENCODE_VAE,
        &[
            std::ffi::OsStr::new("pose"),
            stage.as_os_str(),
            vae_path.as_os_str(),
            pose_latent.as_os_str(),
        ],
        &pose_latent,
    ) {
        return response;
    }
    if !additional_images.is_empty() {
        if let Some(response) = run(
            "encode_additional",
            SCAIL2_ENCODE_VAE,
            &[
                std::ffi::OsStr::new("additional"),
                stage.as_os_str(),
                vae_path.as_os_str(),
                additional_latent.as_os_str(),
            ],
            &additional_latent,
        ) {
            return response;
        }
    }
    let steps_arg = steps.to_string();
    let seed_arg = seed.to_string();
    let additional_arg = if additional_images.is_empty() {
        std::ffi::OsStr::new("-")
    } else {
        additional_latent.as_os_str()
    };
    if let Some(response) = run(
        "denoise",
        SCAIL2_ANIMATION,
        &[
            stage.as_os_str(),
            reference_latent.as_os_str(),
            pose_latent.as_os_str(),
            text.as_os_str(),
            clip.as_os_str(),
            fp8_cache_path.as_os_str(),
            latent.as_os_str(),
            std::ffi::OsStr::new(&steps_arg),
            std::ffi::OsStr::new(&seed_arg),
            std::ffi::OsStr::new(mode),
            additional_arg,
        ],
        &latent,
    ) {
        return response;
    }
    if let Some(response) = run(
        "decode",
        SCAIL2_DECODE,
        &[
            latent.as_os_str(),
            vae_path.as_os_str(),
            result_dir.as_os_str(),
            driving_video.as_os_str(),
        ],
        decoded_mp4,
    ) {
        return response;
    }

    let output_probe = probe_video_path(&decoded_mp4.to_string_lossy())
        .unwrap_or_else(|error| json!({ "muxing": "probe_failed", "error": error }));
    let input_has_audio = input_probe.get("has_audio").and_then(Value::as_bool) == Some(true);
    let artifact_ok = output_probe.get("muxing").and_then(Value::as_str) == Some("probe_ok")
        && output_probe.get("width").and_then(Value::as_i64) == Some(SCAIL2_WIDTH)
        && output_probe.get("height").and_then(Value::as_i64) == Some(SCAIL2_HEIGHT)
        && output_probe.get("frame_count").and_then(Value::as_i64) == Some(SCAIL2_FRAMES)
        && (output_probe
            .get("fps")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            - SCAIL2_FPS as f64)
            .abs()
            < 0.01
        && output_probe.get("has_audio").and_then(Value::as_bool) == Some(input_has_audio);
    if artifact_ok {
        if let Err(error) = publish_scail2_mp4(&paths) {
            return err_detail(StatusCode::INTERNAL_SERVER_ERROR, &error);
        }
    }
    let total_wall_seconds = timings.values().filter_map(Value::as_f64).sum::<f64>();
    let result = json!({
        "schema": "serenity.video_result.v1",
        "video_id": video_id,
        "backend": BACKEND_NAME,
        "control_plane": "serenity-server",
        "model": "scail2",
        "runner": "scail2_animation",
        "state": if artifact_ok { "done" } else { "failed" },
        "readiness_label": "quality_profile_ready",
        "accepted_video_artifact": artifact_ok,
        "accepted_video_parity": scail2_product_gate_passed(),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "mode": mode,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "steps": steps,
        "seed": seed,
        "guidance": guidance,
        "quant": quant,
        "scheduler": "creator_unipc_order2_shift3",
        "processes": if additional_images.is_empty() {
            json!(["stage_inputs", "encode_prompt", "encode_clip", "encode_reference", "encode_pose", "denoise", "decode_audio_mux"])
        } else {
            json!(["stage_inputs", "encode_prompt", "encode_clip", "encode_reference", "encode_pose", "encode_additional", "denoise", "decode_audio_mux"])
        },
        "automatic_conditioning": true,
        "process_separated_decode": true,
        "audio_source": driving_video.to_string_lossy(),
        "input_has_audio": input_has_audio,
        "output_has_audio": output_probe.get("has_audio").and_then(Value::as_bool).unwrap_or(false),
        "out_dir": paths.public_dir.to_string_lossy(),
        "internal_run_dir": out_dir.to_string_lossy(),
        "stage": stage.to_string_lossy(),
        "text_context": text.to_string_lossy(),
        "clip_context": clip.to_string_lossy(),
        "reference_latent": reference_latent.to_string_lossy(),
        "pose_latent": pose_latent.to_string_lossy(),
        "additional_reference_count": additional_images.len(),
        "additional_reference_latent": if additional_images.is_empty() { String::new() } else { additional_latent.to_string_lossy().into_owned() },
        "latent": latent.to_string_lossy(),
        "mp4": mp4.to_string_lossy(),
        "mp4_url": if artifact_ok { format!("/out/{video_id}/scail2_animation.mp4") } else { String::new() },
        "input_probe": input_probe,
        "probe": output_probe,
        "exit_codes": exits,
        "stage_logs": logs,
        "stage_seconds": timings,
        "stage_peak_vram_mib": peaks,
        "total_wall_seconds": total_wall_seconds,
        "result_path": result_path.to_string_lossy(),
        "note": "SCAIL-2 media staging, UMT5, CLIP, reference/pose VAE encoding, FP8 denoise, VAE decode, MP4 mux, and optional driving-video audio preservation ran automatically.",
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
