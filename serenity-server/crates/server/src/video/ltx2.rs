//! LTX-2 video profiles, conditioning, validation, and Mojo orchestration.

use super::*;

pub(super) const RUNNER: &str = "output/bin/ltx2_video_smoke_runner";
// One request-driven runner. Geometry is request data, never encoded in the
// executable name or selected through a per-profile fallback.
pub(super) const LTX2_MOJO_REQUEST_RUNNER: &str = "output/bin/ltx2_serenity_runtime";
pub(super) const REALESRGAN_X4_RUNNER: &str = "output/bin/serenity_realesrgan_x4";
pub(super) const REALESRGAN_X4_WEIGHTS: &str =
    "upscalers/realesrgan-x4plus/RealESRGAN_x4.safetensors";
pub(super) const REALESRGAN_FAST_X4_WEIGHTS: &str =
    "upscalers/realesrgan-fast-x4v3/realesr-general-x4v3.safetensors";
pub(super) const SEEDVR2_PRODUCT_RUNNER: &str = "output/bin/seedvr2_upscale_video";
pub(super) const SEEDVR2_WEIGHTS: [&str; 3] = [
    "upscalers/seedvr2-3b/seedvr2_vae.safetensors",
    "upscalers/seedvr2-3b/seedvr2_dit.safetensors",
    "upscalers/seedvr2-3b/seedvr2_text_emb.safetensors",
];
pub(super) const LTX2_REQUEST_PROFILES_JSON: &str =
    include_str!("../../../../../serenitymojo/configs/ltx2_request_profiles.json");
pub(super) const LTX2_CHECKPOINT_WORKFLOWS_JSON: &str =
    include_str!("../../../../../serenitymojo/configs/ltx2_checkpoint_workflows.json");
pub(super) const LTX2_MOJO_CONDITIONER: &str = "output/bin/ltx2_encode_prompt";
pub(super) const LTX2_MOJO_CONTEXT_SCHEMA: &str = "serenity.ltx2.mojo_gemma3_context_cache.v1";
pub(super) const LTX2_GEMMA_FP8: &str =
    "text_encoders/gemma-3-12b-it-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors";
pub(super) const LTX2_GEMMA_TOKENIZER: &str =
    "text_encoders/gemma-3-12b-it-standalone/tokenizer.json";
pub(super) const LTX2_CONDITIONING_CHECKPOINT: &str =
    "checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
pub(super) const LTX2_CSHIM: &str = "serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so";
pub(super) const LTX2_CONTEXT_PYTHON: &str = ".local/share/LTXDesktop/python/bin/python3";
pub(super) const LTX2_CONTEXT_SCRIPT: &str = "scripts/ltx2_refhq_contexts.py";
pub(super) const LTX2_CREATOR_AUDIO_DECODER: &str = "scripts/ltx2_decode_source_audio.py";
pub(super) const LTX2_CONTEXT_SCHEMA: &str = "serenity.ltx2.refhq_context_cache.v1";
pub(super) const LTX2_CREATOR_REVISION: &str = "780984275fd47128b02bef9b5c085404276866ee";
pub(super) const LTX2_REFHQ_CHECKPOINT: &str = "ltx-2.3-22b-dev-fp8";
pub(super) const LTX2_REFHQ_BF16_CHECKPOINT: &str = "ltx-2.3-22b-dev-fp8-dequant-bf16";
pub(super) const LTX2_REFHQ_DISTILLATION_ADAPTER: &str =
    "checkpoints/ltx-2.3-22b-distilled-lora-384-1.1.safetensors";
// Persistent DISK path, deliberately not /dev/shm: the CUDA JIT cache is the
// only thing standing between a freshly deployed binary and a ~10-20 minute
// first-job compile tax (MJ-1135 measured 1306 s cold vs 142 s warm on the
// same request). tmpfs lost it on every reboot and charged it against RAM.
pub(super) const LTX2_CUDA_CACHE: &str = "output/cuda_cache_video";
pub(super) const LTX2_SAMPLER_PARITY_REPORT: &str = "output/checks/ltx2_sampler_parity.json";
pub(super) const LTX2_VAE_PARITY_REPORT: &str = "output/checks/ltx2_vae_frame_parity.json";
pub(super) const LTX2_AUDIO_PARITY_REPORT: &str = "output/checks/ltx2_audio_parity.json";
pub(super) const LTX2_CREATOR_CUDNN_LIB_CANDIDATES: [&str; 2] = [
    "LTX-Desktop/backend/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ".local/share/LTXDesktop/python/lib/python3.13/site-packages/nvidia/cudnn/lib",
];

/// LTX Creator audio parity is cuDNN-version sensitive. Decode is already a
/// fresh process, so pin only that process to Creator's measured 9.10.2
/// runtime; denoising and SDPA continue to use the general Mojo runtime.
pub(super) fn ltx2_decode_cudnn_lib() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("LTX2_CREATOR_CUDNN_LIB") {
        return std::path::PathBuf::from(path);
    }
    LTX2_CREATOR_CUDNN_LIB_CANDIDATES
        .iter()
        .map(|path| home_path(path))
        .find(|path| nonempty_file(&path.join("libcudnn.so.9")))
        .unwrap_or_else(|| home_path(LTX2_CREATOR_CUDNN_LIB_CANDIDATES[0]))
}

pub(super) fn ltx2_decode_runtime_available() -> bool {
    nonempty_file(&ltx2_decode_cudnn_lib().join("libcudnn.so.9"))
}

pub(super) fn ltx2_decode_ld_path() -> std::ffi::OsString {
    let root = repo_root();
    let mut parts = vec![
        // This must precede Pixi's general cuDNN. The audio decoder's cshim is
        // parity-gated to Creator's 9.10.2 runtime and the dynamic loader keeps
        // the first libcudnn.so.9 loaded for the whole fresh decode process.
        ltx2_decode_cudnn_lib(),
        root.join(".pixi/envs/default/lib"),
        root.join("serenitymojo/ops/cshim/lib"),
    ];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        parts.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(parts).unwrap_or_default()
}

/// Retake and Extend encode source audio in the main request process before
/// denoising. Keep that whole temporal-edit process on Creator's pinned cuDNN
/// runtime; once a process loads libcudnn.so.9, changing the path for a later
/// phase cannot replace it.
pub(super) fn ltx2_request_ld_path(edit_mode: &str) -> std::ffi::OsString {
    if edit_mode == "standard" {
        mojo_ld_path()
    } else {
        ltx2_decode_ld_path()
    }
}
/// svdint4 slab matching the distilled-fp8 base the LTX2 runner streams
/// (`CKPT_FP8` in ltx2_t2v_av_hq.mojo). Selected via `LTX2_INT4_SLAB` for the
/// int4 W4A16-resident path. (Verified present on this box, 2026-07-11.)
pub(super) const LTX2_INT4_SLAB: &str = "checkpoints/ltx-2.3-22b-distilled-svdint4-r32.safetensors";
/// W4A16 resident slab reconstructed from the LTX-2.3 dev checkpoint. The
/// request runner applies the official support LoRA and authored LoRAs on top,
/// so it must not use the already-distilled slab above.
pub(super) const LTX2_REFHQ_INT4_SLAB: &str = "checkpoints/ltx-2.3-22b-svdint4-r32.safetensors";
/// Full dequantized BF16 LTX-2.3 dev transformer used by the request runner
/// when Canvas selects BF16 precision. Activations remain BF16 and reductions
/// remain F32; this is a real storage-mode switch, not an FP8 label alias.
pub(super) const LTX2_REFHQ_BF16: &str = "checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors";

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct Ltx2RequestProfileRegistry {
    pub(super) schema: String,
    pub(super) checkpoint: String,
    pub(super) checkpoints: Vec<Ltx2CheckpointProfile>,
    pub(super) guidance_modes: Value,
    pub(super) profile_groups: Vec<Ltx2RequestProfileGroup>,
    pub(super) post_upscalers: Value,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct Ltx2CheckpointProfile {
    pub(super) id: String,
    pub(super) label: String,
    pub(super) path: String,
    pub(super) aliases: Vec<String>,
    pub(super) support_lora: String,
    pub(super) guidance_modes: Vec<String>,
    pub(super) quant_modes: Vec<String>,
    pub(super) readiness_label: String,
    pub(super) source: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct Ltx2RequestProfileGroup {
    pub(super) id: String,
    pub(super) label: String,
    pub(super) width: i64,
    pub(super) height: i64,
    #[serde(default)]
    pub(super) conditioning_width: Option<i64>,
    #[serde(default)]
    pub(super) conditioning_height: Option<i64>,
    pub(super) fps: i64,
    pub(super) modes: Vec<String>,
    pub(super) durations: Vec<f64>,
    pub(super) frames: Vec<i64>,
    pub(super) source: String,
}

#[derive(Clone, Debug)]
pub(super) struct Ltx2ResolvedRequestProfile {
    pub(super) group_id: String,
    pub(super) label: String,
    pub(super) width: i64,
    pub(super) height: i64,
    pub(super) conditioning_width: i64,
    pub(super) conditioning_height: i64,
    pub(super) frames: i64,
    pub(super) fps: i64,
    pub(super) modes: Vec<String>,
    pub(super) duration: f64,
    pub(super) source: String,
}

pub(super) fn ltx2_checkpoint_workflow_registry() -> &'static Value {
    static REGISTRY: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry: Value = serde_json::from_str(LTX2_CHECKPOINT_WORKFLOWS_JSON)
            .expect("embedded LTX2 checkpoint workflow registry must be valid JSON");
        assert_eq!(
            registry.get("schema").and_then(Value::as_str),
            Some("serenity.ltx2.checkpoint_workflows.v1"),
            "embedded LTX2 checkpoint workflow registry schema mismatch"
        );
        registry
    })
}

pub(super) fn ltx2_checkpoint_workflow(checkpoint: &str) -> Option<&'static Value> {
    let checkpoint = checkpoint
        .strip_suffix(".safetensors")
        .unwrap_or(checkpoint)
        .to_ascii_lowercase();
    ltx2_checkpoint_workflow_registry()
        .get("profiles")
        .and_then(Value::as_array)
        .and_then(|profiles| {
            profiles.iter().find(|profile| {
                profile
                    .get("checkpoints")
                    .and_then(Value::as_array)
                    .is_some_and(|names| {
                        names.iter().filter_map(Value::as_str).any(|name| {
                            name.strip_suffix(".safetensors")
                                .unwrap_or(name)
                                .eq_ignore_ascii_case(&checkpoint)
                        })
                    })
            })
        })
}

pub(super) fn ltx2_checkpoint_workflow_documents() -> Value {
    let profiles = ltx2_checkpoint_workflow_registry()
        .get("profiles")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|mut profile| {
            let adapter_path = profile
                .pointer("/distillation_adapter/path")
                .and_then(Value::as_str)
                .map(model_path);
            let enhancer_weights = profile
                .pointer("/prompt_enhancer/weights")
                .and_then(Value::as_str)
                .map(model_path);
            let enhancer_mmproj = profile
                .pointer("/prompt_enhancer/mmproj")
                .and_then(Value::as_str)
                .map(model_path);
            let enhancer_files_available = enhancer_weights.as_deref().is_some_and(nonempty_file)
                && enhancer_mmproj.as_deref().is_some_and(nonempty_file);
            if let Some(object) = profile.as_object_mut() {
                object.insert(
                    "adapter_available".to_string(),
                    json!(adapter_path.as_deref().is_some_and(nonempty_file)),
                );
                object.insert(
                    "prompt_enhancer_files_available".to_string(),
                    json!(enhancer_files_available),
                );
                // Installed files are not runtime support. Keep this false
                // until the creator's llama.cpp vision route exists and has
                // product evidence.
                object.insert("prompt_enhancer_available".to_string(), json!(false));
                object.insert(
                    "prompt_enhancer_runtime".to_string(),
                    json!("not_implemented"),
                );
            }
            profile
        })
        .collect::<Vec<_>>();
    Value::Array(profiles)
}

pub(super) fn ltx2_request_profile_registry() -> &'static Ltx2RequestProfileRegistry {
    static REGISTRY: std::sync::OnceLock<Ltx2RequestProfileRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry: Ltx2RequestProfileRegistry = serde_json::from_str(LTX2_REQUEST_PROFILES_JSON)
            .expect("embedded LTX2 request profile registry must be valid JSON");
        assert_eq!(
            registry.schema, "serenity.ltx2.request_profiles.v1",
            "embedded LTX2 request profile registry schema mismatch"
        );
        assert!(
            registry.profile_groups.iter().all(|group| {
                !group.frames.is_empty()
                    && group.frames.len() == group.durations.len()
                    && !group.modes.is_empty()
            }),
            "each embedded LTX2 profile group must pair frames with durations and declare modes"
        );
        assert!(
            registry
                .checkpoints
                .iter()
                .any(|profile| profile.id == registry.checkpoint),
            "embedded LTX2 default checkpoint must name a checkpoint profile"
        );
        assert!(
            registry.checkpoints.iter().all(|profile| {
                !profile.id.is_empty()
                    && !profile.path.is_empty()
                    && matches!(profile.support_lora.as_str(), "official" | "baked")
                    && !profile.guidance_modes.is_empty()
                    && !profile.quant_modes.is_empty()
            }),
            "embedded LTX2 checkpoint profiles must be complete"
        );
        registry
    })
}

pub(super) const LTX2_REQUEST_RUNNER_BUILD_INPUTS: &[&str] = &[
    "serenitymojo/configs/ltx2_request_profiles.json",
    // Checkpoint workflow aliases and authored defaults are embedded in this
    // Rust server, which normalizes the request before launching Mojo. The AOT
    // runner never reads that registry, so changing an alias must not
    // invalidate every geometry-specialized executable.
    "serenitymojo/sampling/ltx2_request_cli.mojo",
    "serenitymojo/sampling/ltx2_sampling.mojo",
    "serenitymojo/pipeline/ltx2_t2v_av_hq.mojo",
    "serenitymojo/serve/proc_ipc.mojo",
    "serenitymojo/models/vae/ltx2_tiled_decode.mojo",
    "serenitymojo/models/vae/ltx2_vae_encoder.mojo",
    "serenitymojo/models/vae/conv3d.mojo",
    "serenitymojo/image/png.mojo",
    "serenitymojo/serve/image_io.mojo",
    "serenitymojo/image/decode.mojo",
];

pub(super) fn ltx2_runner_mtime_covers_inputs(
    runner_modified: std::time::SystemTime,
    input_modified: &[std::time::SystemTime],
) -> bool {
    input_modified
        .iter()
        .all(|modified| *modified <= runner_modified)
}

/// Fail closed when the single runtime-geometry runner is missing or stale.
pub(super) fn ltx2_request_runner_current(path: &str) -> bool {
    if !bin_x(path) {
        return false;
    }
    let Ok(runner_modified) = std::fs::metadata(repo_path(path)).and_then(|meta| meta.modified())
    else {
        return false;
    };
    let mut input_modified = Vec::with_capacity(LTX2_REQUEST_RUNNER_BUILD_INPUTS.len());
    for input in LTX2_REQUEST_RUNNER_BUILD_INPUTS {
        let Ok(modified) = std::fs::metadata(repo_path(input)).and_then(|meta| meta.modified())
        else {
            return false;
        };
        input_modified.push(modified);
    }
    ltx2_runner_mtime_covers_inputs(runner_modified, &input_modified)
}

pub(super) fn ltx2_checkpoint_profile(checkpoint: &str) -> Option<&'static Ltx2CheckpointProfile> {
    let checkpoint = checkpoint.trim();
    ltx2_request_profile_registry()
        .checkpoints
        .iter()
        .find(|profile| {
            checkpoint == profile.id
                || checkpoint == format!("{}.safetensors", profile.id)
                || profile.aliases.iter().any(|alias| alias == checkpoint)
        })
}

pub(super) fn ltx2_checkpoint_document(profile: &Ltx2CheckpointProfile) -> Value {
    let path = model_path(&profile.path);
    json!({
        "id": profile.id,
        "label": profile.label,
        "path": profile.path,
        "installed": nonempty_file(&path),
        "support_lora": profile.support_lora,
        "guidance_modes": profile.guidance_modes,
        "quant_modes": profile.quant_modes,
        "readiness_label": profile.readiness_label,
        "source": profile.source,
    })
}

pub(super) fn ltx2_checkpoint_documents() -> Value {
    Value::Array(
        ltx2_request_profile_registry()
            .checkpoints
            .iter()
            .map(ltx2_checkpoint_document)
            .collect(),
    )
}

pub(super) fn ltx2_resolved_profiles() -> Vec<Ltx2ResolvedRequestProfile> {
    ltx2_request_profile_registry()
        .profile_groups
        .iter()
        .flat_map(|group| {
            group
                .frames
                .iter()
                .copied()
                .zip(group.durations.iter().copied())
                .map(|(frames, duration)| Ltx2ResolvedRequestProfile {
                    group_id: group.id.clone(),
                    label: group.label.clone(),
                    width: group.width,
                    height: group.height,
                    conditioning_width: group.conditioning_width.unwrap_or(group.width),
                    conditioning_height: group.conditioning_height.unwrap_or(group.height),
                    frames,
                    fps: group.fps,
                    modes: group.modes.clone(),
                    duration,
                    source: group.source.clone(),
                })
        })
        .collect()
}

pub(super) fn ltx2_profile_runner_available(_profile: &Ltx2ResolvedRequestProfile) -> bool {
    ltx2_request_runner_current(LTX2_MOJO_REQUEST_RUNNER)
}

pub(super) fn ltx2_request_profile_for_mode(
    width: i64,
    height: i64,
    frames: i64,
    fps: f64,
    mode: &str,
) -> Option<Ltx2ResolvedRequestProfile> {
    ltx2_resolved_profiles().into_iter().find(|profile| {
        profile.width == width
            && profile.height == height
            && profile.frames == frames
            && (profile.fps as f64 - fps).abs() <= f64::EPSILON
            && profile.modes.iter().any(|candidate| candidate == mode)
    })
}

pub(super) fn ltx2_profile_document(profile: &Ltx2ResolvedRequestProfile) -> Value {
    json!({
        "id": profile.group_id,
        "label": profile.label,
        "checkpoint": ltx2_request_profile_registry().checkpoint,
        "width": profile.width,
        "height": profile.height,
        "conditioning_width": profile.conditioning_width,
        "conditioning_height": profile.conditioning_height,
        "frames": profile.frames,
        "fps": profile.fps,
        "modes": profile.modes,
        "duration": profile.duration,
        "source": profile.source,
        "runner": LTX2_MOJO_REQUEST_RUNNER,
        "available": ltx2_profile_runner_available(profile),
        "output_format": "mp4",
        "guidance_modes": ltx2_request_profile_registry().guidance_modes,
    })
}

pub(super) fn stage_ltx2_creator_i2v_source(
    source_path: &str,
    profile: &Ltx2ResolvedRequestProfile,
    out_dir: &std::path::Path,
    stem: &str,
) -> Result<std::path::PathBuf, String> {
    // LTX Desktop first converts to RGB, Lanczos-resizes with fill, and
    // center-crops at the UI-authored size. The fast pipeline then performs a
    // one-frame libx264 CRF-33 round trip before its per-stage bilinear resize.
    let source = image::open(source_path)
        .map_err(|error| format!("cannot decode LTX2 I2V source '{source_path}': {error}"))?
        .to_rgb8();
    let source_width = source.width();
    let source_height = source.height();
    let target_width = u32::try_from(profile.conditioning_width)
        .map_err(|_| "invalid LTX2 conditioning width".to_string())?;
    let target_height = u32::try_from(profile.conditioning_height)
        .map_err(|_| "invalid LTX2 conditioning height".to_string())?;
    if source_width == 0 || source_height == 0 || target_width == 0 || target_height == 0 {
        return Err("LTX2 I2V source and conditioning dimensions must be positive".to_string());
    }

    let source_is_wider = u64::from(source_width) * u64::from(target_height)
        > u64::from(target_width) * u64::from(source_height);
    let (resized_width, resized_height) = if source_is_wider {
        (
            u32::try_from(
                u64::from(source_width) * u64::from(target_height) / u64::from(source_height),
            )
            .map_err(|_| "LTX2 I2V resized width overflow".to_string())?,
            target_height,
        )
    } else {
        (
            target_width,
            u32::try_from(
                u64::from(source_height) * u64::from(target_width) / u64::from(source_width),
            )
            .map_err(|_| "LTX2 I2V resized height overflow".to_string())?,
        )
    };
    let resized = image::imageops::resize(
        &source,
        resized_width,
        resized_height,
        image::imageops::FilterType::Lanczos3,
    );
    let left = (resized_width - target_width) / 2;
    let top = (resized_height - target_height) / 2;
    let prepared =
        image::imageops::crop_imm(&resized, left, top, target_width, target_height).to_image();

    let prepared_png = out_dir.join(format!("{stem}_prepared.png"));
    prepared
        .save_with_format(&prepared_png, image::ImageFormat::Png)
        .map_err(|error| format!("cannot save creator-prepared I2V source: {error}"))?;
    let roundtrip_png = out_dir.join(format!("{stem}_conditioning.png"));
    let python = repo_path(".pixi/envs/default/bin/python3");
    let creator_preprocess = repo_path("scripts/ltx2_creator_image_preprocess.py");
    if !python.is_file() || !creator_preprocess.is_file() {
        return Err(format!(
            "creator I2V preprocessing runtime is missing: {}, {}",
            python.display(),
            creator_preprocess.display()
        ));
    }
    let preprocess = std::process::Command::new(&python)
        .arg(&creator_preprocess)
        .arg(&prepared_png)
        .arg(&roundtrip_png)
        .args(["--crf", "33"])
        .output()
        .map_err(|error| format!("cannot launch creator PyAV I2V preprocessing: {error}"))?;
    if !preprocess.status.success() || !nonempty_file(&roundtrip_png) {
        return Err(format!(
            "creator PyAV I2V preprocessing failed: {}",
            String::from_utf8_lossy(&preprocess.stderr).trim()
        ));
    }
    Ok(roundtrip_png)
}

pub(super) fn ltx2_post_upscaler_documents() -> Value {
    let Some(rows) = ltx2_request_profile_registry().post_upscalers.as_array() else {
        return json!([]);
    };
    Value::Array(
        rows.iter()
            .map(|row| {
                let mut doc = row.clone();
                let id = row.get("id").and_then(Value::as_str).unwrap_or("");
                let (available, missing) = match id {
                    "realesrgan-x4plus" => {
                        let mut missing = Vec::new();
                        if !bin_x(REALESRGAN_X4_RUNNER) {
                            missing.push(REALESRGAN_X4_RUNNER.to_string());
                        }
                        if !nonempty_file(&model_path(REALESRGAN_X4_WEIGHTS)) {
                            missing.push(
                                model_path(REALESRGAN_X4_WEIGHTS)
                                    .to_string_lossy()
                                    .into_owned(),
                            );
                        }
                        (missing.is_empty(), missing)
                    }
                    "realesrgan-fast-x4v3" => {
                        let mut missing = Vec::new();
                        if !bin_x(REALESRGAN_X4_RUNNER) {
                            missing.push(REALESRGAN_X4_RUNNER.to_string());
                        }
                        if !nonempty_file(&model_path(REALESRGAN_FAST_X4_WEIGHTS)) {
                            missing.push(
                                model_path(REALESRGAN_FAST_X4_WEIGHTS)
                                    .to_string_lossy()
                                    .into_owned(),
                            );
                        }
                        (missing.is_empty(), missing)
                    }
                    "seedvr2-3b" => {
                        let mut missing = Vec::new();
                        if !bin_x(SEEDVR2_PRODUCT_RUNNER) {
                            missing.push(SEEDVR2_PRODUCT_RUNNER.to_string());
                        }
                        for weight in SEEDVR2_WEIGHTS {
                            if !nonempty_file(&model_path(weight)) {
                                missing.push(model_path(weight).to_string_lossy().into_owned());
                            }
                        }
                        // The checked-in GitHub CLI is still a fixture/demo
                        // runner. Even if somebody drops weights next to it,
                        // do not advertise a user-video route until the
                        // dedicated product binary exists.
                        missing.push("product user-video adapter is not implemented".to_string());
                        (false, missing)
                    }
                    _ => (false, vec!["unknown post-upscaler".to_string()]),
                };
                if let Some(object) = doc.as_object_mut() {
                    object.insert("available".to_string(), json!(available));
                    let status = if id == "seedvr2-3b" {
                        "source_only"
                    } else if id == "realesrgan-x4plus" && available {
                        "experimental_slow"
                    } else if available {
                        "ready"
                    } else {
                        "prerequisites_missing"
                    };
                    object.insert("status".to_string(), json!(status));
                    object.insert("missing".to_string(), json!(missing));
                }
                doc
            })
            .collect(),
    )
}

pub(super) fn runner_available() -> bool {
    bin_x(RUNNER)
}

pub(super) fn verify_ltx2_creator_revision() -> Result<(), String> {
    let root = std::env::var_os("LTX2_CREATOR_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo_root().join("../LTX-2"));
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .map_err(|e| format!("ltx2_refhq: cannot inspect Creator checkout: {e}"))?;
    let actual = String::from_utf8_lossy(&head.stdout).trim().to_string();
    if !head.status.success() || actual != LTX2_CREATOR_REVISION {
        return Err(format!(
            "ltx2_refhq: Creator revision mismatch at {}: expected {}, got {}",
            root.display(),
            LTX2_CREATOR_REVISION,
            if actual.is_empty() {
                "<unreadable>"
            } else {
                &actual
            }
        ));
    }
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(&root)
        .output()
        .map_err(|e| format!("ltx2_refhq: cannot inspect Creator worktree: {e}"))?;
    if !status.status.success() || !status.stdout.is_empty() {
        return Err(format!(
            "ltx2_refhq: Creator checkout at {} is dirty; oracle execution requires the clean pinned revision",
            root.display()
        ));
    }
    Ok(())
}

pub(super) fn tensor_header_matches(doc: &Value, key: &str, dtype: &str, shape: &[u64]) -> bool {
    let Some(entry) = doc.get(key) else {
        return false;
    };
    if entry.get("dtype").and_then(Value::as_str) != Some(dtype) {
        return false;
    }
    let Some(actual) = entry.get("shape").and_then(Value::as_array) else {
        return false;
    };
    actual.len() == shape.len()
        && actual
            .iter()
            .zip(shape)
            .all(|(seen, expected)| seen.as_u64() == Some(*expected))
}

pub(super) fn final_latents_are_bf16(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "video", "BF16", &[1, 128, 16, 34, 60])
        && tensor_header_matches(&doc, "audio", "BF16", &[1, 8, 501, 16])
}

pub(super) fn stage1_cache_is_bf16(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "video", "BF16", &[1, 8160, 128])
        && tensor_header_matches(&doc, "audio", "BF16", &[1, 126, 128])
}

pub(super) fn upscaler_cache_is_f32(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "out", "F32", &[1, 128, 16, 34, 60])
}

pub(super) fn ltx2_context_cache_valid(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    let revision_ok = doc
        .get("__metadata__")
        .and_then(|meta| meta.get("creator_revision"))
        .and_then(Value::as_str)
        == Some(LTX2_CREATOR_REVISION);
    revision_ok
        && tensor_header_matches(&doc, "video_context", "BF16", &[1, 1024, 4096])
        && tensor_header_matches(&doc, "audio_context", "BF16", &[1, 1024, 2048])
        && tensor_header_matches(&doc, "neg_video_context", "BF16", &[1, 1024, 4096])
        && tensor_header_matches(&doc, "neg_audio_context", "BF16", &[1, 1024, 2048])
        && tensor_header_matches(&doc, "video_len", "F32", &[1])
        && tensor_header_matches(&doc, "neg_video_len", "F32", &[1])
}

pub(super) fn ltx2_mojo_context_tensor_valid(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "video_context", "BF16", &[1, 1024, 4096])
        && tensor_header_matches(&doc, "audio_context", "BF16", &[1, 1024, 2048])
        && tensor_header_matches(&doc, "neg_video_context", "BF16", &[1, 1024, 4096])
        && tensor_header_matches(&doc, "neg_audio_context", "BF16", &[1, 1024, 2048])
        && tensor_header_matches(&doc, "video_len", "F32", &[1])
        && tensor_header_matches(&doc, "neg_video_len", "F32", &[1])
}

pub(super) fn ltx2_mojo_conditioning_missing() -> Vec<String> {
    let candidates = [
        repo_path(LTX2_MOJO_CONDITIONER),
        model_path(LTX2_GEMMA_FP8),
        model_path(LTX2_GEMMA_TOKENIZER),
        model_path(LTX2_CONDITIONING_CHECKPOINT),
    ];
    candidates
        .iter()
        .filter(|path| !nonempty_file(path))
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

/// Accept an oracle claim only from the measured report produced by the local
/// parity gate at the pinned Creator revision and the production 0.999 bar.
/// Output evidence is machine-local by design; a clean checkout without a
/// completed gate must report false rather than inheriting a source-code claim.
pub(super) fn ltx2_parity_report_passed(
    path: &str,
    schema: &str,
    require_shared_latent: bool,
) -> bool {
    ltx2_parity_report_matches(
        &repo_path(path),
        &repo_path(RUNNER),
        schema,
        require_shared_latent,
    )
}

pub(super) fn ltx2_parity_report_matches(
    report_path: &std::path::Path,
    runner_path: &std::path::Path,
    schema: &str,
    require_shared_latent: bool,
) -> bool {
    let Ok(bytes) = std::fs::read(report_path) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    let Some(current_runner_sha256) = sha256sum(runner_path) else {
        return false;
    };
    if doc.get("schema").and_then(Value::as_str) != Some(schema)
        || doc.get("creator_revision").and_then(Value::as_str) != Some(LTX2_CREATOR_REVISION)
        || doc.get("mojo_runner_sha256").and_then(Value::as_str)
            != Some(current_runner_sha256.as_str())
        || doc.get("passed").and_then(Value::as_bool) != Some(true)
        || doc.get("bar").and_then(Value::as_f64).unwrap_or(0.0) < 0.999
    {
        return false;
    }
    if require_shared_latent {
        let Some(current_cshim_sha256) = sha256sum(&repo_path(LTX2_CSHIM)) else {
            return false;
        };
        let shared_latent_bound = doc
            .get("shared_latents_sha256")
            .and_then(Value::as_str)
            .is_some_and(|digest| {
                digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
            });
        let cshim_bound = doc.get("mojo_cshim_sha256").and_then(Value::as_str)
            == Some(current_cshim_sha256.as_str());
        return shared_latent_bound && cshim_bound;
    }
    true
}

pub(super) fn ltx2_context_key(prompt: &str, negative_prompt: &str) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for part in [
        LTX2_CONTEXT_SCHEMA,
        LTX2_CREATOR_REVISION,
        prompt,
        negative_prompt,
    ] {
        for byte in part.as_bytes().iter().copied().chain(std::iter::once(0xff)) {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

pub(super) fn ltx2_seed(body: &Value) -> Result<u64, &'static str> {
    const ERROR: &str = "ltx2_refhq 'seed' must be an integer from 0 through 4294967295";
    match body.get("seed") {
        None => Ok(42),
        Some(value) => value
            .as_u64()
            .filter(|seed| *seed <= u64::from(u32::MAX))
            .ok_or(ERROR),
    }
}

pub(super) struct Ltx2ContextCache {
    pub(super) path: std::path::PathBuf,
    pub(super) key: String,
    pub(super) hit: bool,
    pub(super) encoder_seconds: f64,
    pub(super) log_path: std::path::PathBuf,
    pub(super) manifest_path: std::path::PathBuf,
}

/// Run the existing Creator-backed Gemma producer before the Mojo video model
/// is loaded. The completed single-file context is reused by prompt/negative
/// key; the producer exits before refhq denoising begins.
pub(super) fn prepare_ltx2_refhq_context(
    out_root: &std::path::Path,
    prompt: &str,
    negative_prompt: &str,
) -> Result<Ltx2ContextCache, String> {
    if prompt.trim().is_empty() {
        return Err("ltx2_refhq: 'prompt' is required".to_string());
    }
    verify_ltx2_creator_revision()?;
    let effective_negative = if negative_prompt.trim().is_empty() {
        "<creator-default-negative>"
    } else {
        negative_prompt
    };
    let key = ltx2_context_key(prompt, effective_negative);
    let cache_dir = out_root
        .join("conditioning_cache")
        .join("ltx2")
        .join("creator-refhq-v1");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("ltx2_refhq: create conditioning cache: {e}"))?;
    let path = cache_dir.join(format!("{key}.safetensors"));
    let log_path = cache_dir.join(format!("{key}.log"));
    let manifest_path = cache_dir.join(format!("{key}.json"));
    if ltx2_context_cache_valid(&path) {
        return Ok(Ltx2ContextCache {
            path,
            key,
            hit: true,
            encoder_seconds: 0.0,
            log_path,
            manifest_path,
        });
    }

    let context_python = std::env::var_os("LTX2_CONTEXT_PYTHON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| home_path(LTX2_CONTEXT_PYTHON));
    if !context_python.is_file() {
        return Err(format!(
            "ltx2_refhq: conditioning interpreter missing: {}",
            context_python.display()
        ));
    }
    let script = repo_path(LTX2_CONTEXT_SCRIPT);
    if !script.is_file() {
        return Err(format!(
            "ltx2_refhq: conditioning producer missing: {}",
            script.display()
        ));
    }

    let mut command = std::process::Command::new(context_python);
    command
        .current_dir(repo_root())
        .arg(&script)
        .args(["--prompt", prompt, "--out"])
        .arg(&path);
    if negative_prompt.trim().is_empty() {
        command.arg("--neg-official");
    } else {
        command.args(["--neg", negative_prompt]);
    }
    let started = std::time::Instant::now();
    let output = command
        .output()
        .map_err(|e| format!("ltx2_refhq: start conditioning producer: {e}"))?;
    let encoder_seconds = started.elapsed().as_secs_f64();
    let mut log = output.stdout;
    log.extend_from_slice(&output.stderr);
    let _ = std::fs::write(&log_path, &log);
    if !output.status.success() || !ltx2_context_cache_valid(&path) {
        let _ = std::fs::remove_file(&path);
        return Err(format!(
            "ltx2_refhq: conditioning producer failed with {:?}; inspect {}",
            output.status.code(),
            log_path.to_string_lossy()
        ));
    }
    let manifest = json!({
        "schema": LTX2_CONTEXT_SCHEMA,
        "cache_key": key,
        "creator_revision": LTX2_CREATOR_REVISION,
        "producer": LTX2_CONTEXT_SCRIPT,
        "prompt": prompt,
        "negative_prompt": effective_negative,
        "path": path.to_string_lossy(),
        "log_path": log_path.to_string_lossy(),
        "encoder_seconds": encoder_seconds,
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    )
    .map_err(|e| format!("ltx2_refhq: write conditioning manifest: {e}"))?;
    Ok(Ltx2ContextCache {
        path,
        key,
        hit: false,
        encoder_seconds,
        log_path,
        manifest_path,
    })
}

pub(super) fn ltx2_mojo_context_key(
    prompt: &str,
    negative_prompt: &str,
    conditioner_sha256: &str,
) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for part in [
        LTX2_MOJO_CONTEXT_SCHEMA,
        conditioner_sha256,
        prompt,
        negative_prompt,
    ] {
        for byte in part.as_bytes().iter().copied().chain(std::iter::once(0xff)) {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("{h:016x}")
}

pub(super) fn write_ltx2_job_status(
    out_dir: &std::path::Path,
    state: &str,
    phase: &str,
    step: i64,
    total: i64,
    message: &str,
) -> Result<(), String> {
    let path = out_dir.join("status.json");
    let tmp = out_dir.join("status.json.tmp");
    let body = json!({
        "schema": "serenity.ltx2.status.v1",
        "state": state,
        "phase": phase,
        "step": step,
        "total": total,
        "message": message,
    });
    let bytes = serde_json::to_vec_pretty(&body)
        .map_err(|error| format!("serialize LTX2 status: {error}"))?;
    std::fs::write(&tmp, bytes).map_err(|error| format!("write LTX2 status: {error}"))?;
    std::fs::rename(&tmp, &path).map_err(|error| format!("publish LTX2 status: {error}"))
}

pub(super) fn ltx2_mojo_cache_manifest_valid(
    path: &std::path::Path,
    key: &str,
    conditioner_sha256: &str,
) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    doc.get("schema").and_then(Value::as_str) == Some(LTX2_MOJO_CONTEXT_SCHEMA)
        && doc.get("cache_key").and_then(Value::as_str) == Some(key)
        && doc.get("conditioner_sha256").and_then(Value::as_str) == Some(conditioner_sha256)
}

/// Run the Mojo-native Gemma-3 conditioner before loading the LTX denoiser.
/// Both prompts share one layer stream; the resulting pre-connector contexts
/// are cached by prompt, negative prompt, and conditioner binary digest.
pub(super) fn prepare_ltx2_mojo_context<F>(
    out_root: &std::path::Path,
    job_out_dir: &std::path::Path,
    prompt: &str,
    negative_prompt: &str,
    publish: &F,
) -> Result<Ltx2ContextCache, String>
where
    F: Fn(WorkerEvent),
{
    if prompt.trim().is_empty() {
        return Err("LTX2 prompt is required".to_string());
    }
    let missing = ltx2_mojo_conditioning_missing();
    if !missing.is_empty() {
        return Err(format!(
            "LTX2 automatic prompt conditioning is unavailable; missing: {}",
            missing.join(", ")
        ));
    }
    let conditioner = repo_path(LTX2_MOJO_CONDITIONER);
    let conditioner_sha256 =
        sha256sum(&conditioner).ok_or_else(|| "cannot hash LTX2 Mojo conditioner".to_string())?;
    let key = ltx2_mojo_context_key(prompt, negative_prompt, &conditioner_sha256);
    let cache_dir = out_root
        .join("conditioning_cache")
        .join("ltx2")
        .join("mojo-gemma3-v1");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|error| format!("create LTX2 Mojo conditioning cache: {error}"))?;
    let path = cache_dir.join(format!("{key}.safetensors"));
    let log_path = cache_dir.join(format!("{key}.log"));
    let manifest_path = cache_dir.join(format!("{key}.json"));
    if ltx2_mojo_context_tensor_valid(&path)
        && ltx2_mojo_cache_manifest_valid(&manifest_path, &key, &conditioner_sha256)
    {
        let message = "Prompt conditioning cache hit";
        let _ = write_ltx2_job_status(job_out_dir, "running", "conditioning", 48, 48, message);
        publish(WorkerEvent::Progress {
            step: 48,
            total: 48,
            phase: message.to_string(),
            preview: String::new(),
        });
        return Ok(Ltx2ContextCache {
            path,
            key,
            hit: true,
            encoder_seconds: 0.0,
            log_path,
            manifest_path,
        });
    }

    let started = std::time::Instant::now();
    let first_message = "Tokenizing LTX2 prompt";
    let _ = write_ltx2_job_status(job_out_dir, "running", "conditioning", 0, 48, first_message);
    publish(WorkerEvent::Progress {
        step: 0,
        total: 48,
        phase: first_message.to_string(),
        preview: String::new(),
    });

    let mut log = std::fs::File::create(&log_path)
        .map_err(|error| format!("create LTX2 conditioner log: {error}"))?;
    let stderr = log
        .try_clone()
        .map_err(|error| format!("clone LTX2 conditioner log: {error}"))?;
    let mut command = std::process::Command::new(&conditioner);
    command
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .arg(model_path(LTX2_GEMMA_FP8))
        .arg(model_path(LTX2_GEMMA_TOKENIZER))
        .arg(model_path(LTX2_CONDITIONING_CHECKPOINT))
        .arg(&path)
        .arg(prompt)
        .arg(negative_prompt)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::from(stderr));
    let mut child = command
        .spawn()
        .map_err(|error| format!("start LTX2 Mojo conditioner: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "capture LTX2 conditioner output".to_string())?;
    let reader = std::io::BufReader::new(stdout);
    for line in std::io::BufRead::lines(reader) {
        let line = line.map_err(|error| format!("read LTX2 conditioner output: {error}"))?;
        let _ = std::io::Write::write_all(&mut log, format!("{line}\n").as_bytes());
        let Some(activity) = line.strip_prefix("LTX2_ACTIVITY ") else {
            continue;
        };
        let (step, total, message) = if let Some(progress) =
            activity.strip_prefix("encoding Gemma layer ")
        {
            let (step, total) = progress
                .split_once('/')
                .and_then(|(step, total)| {
                    Some((
                        step.trim().parse::<i64>().ok()?,
                        total.trim().parse::<i64>().ok()?,
                    ))
                })
                .unwrap_or((0, 48));
            (
                step,
                total,
                format!("Encoding LTX2 prompt · Gemma layer {step} / {total}"),
            )
        } else {
            match activity {
                "tokenizing prompt" => (0, 48, "Tokenizing LTX2 prompt".to_string()),
                "loading Gemma text encoder" => (0, 48, "Loading Gemma text encoder".to_string()),
                "projecting video and audio conditioning" => (
                    48,
                    48,
                    "Projecting LTX2 video/audio conditioning".to_string(),
                ),
                "saving prompt conditioning" => {
                    (48, 48, "Saving LTX2 prompt conditioning".to_string())
                }
                "conditioning complete" => {
                    (48, 48, "LTX2 prompt conditioning complete".to_string())
                }
                other => (0, 48, other.to_string()),
            }
        };
        let _ = write_ltx2_job_status(
            job_out_dir,
            "running",
            "conditioning",
            step,
            total,
            &message,
        );
        publish(WorkerEvent::Progress {
            step,
            total,
            phase: message,
            preview: String::new(),
        });
    }
    let status = child
        .wait()
        .map_err(|error| format!("wait for LTX2 Mojo conditioner: {error}"))?;
    let encoder_seconds = started.elapsed().as_secs_f64();
    if !status.success() || !ltx2_mojo_context_tensor_valid(&path) {
        let _ = std::fs::remove_file(&path);
        return Err(format!(
            "LTX2 Mojo prompt conditioning failed with {:?}; inspect {}",
            status.code(),
            log_path.display()
        ));
    }
    let manifest = json!({
        "schema": LTX2_MOJO_CONTEXT_SCHEMA,
        "cache_key": key,
        "conditioner": LTX2_MOJO_CONDITIONER,
        "conditioner_sha256": conditioner_sha256,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "path": path.to_string_lossy(),
        "log_path": log_path.to_string_lossy(),
        "encoder_seconds": encoder_seconds,
        "backend": "mojo",
        "conditioning_stage": "pre_connector",
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    )
    .map_err(|error| format!("write LTX2 Mojo conditioning manifest: {error}"))?;
    Ok(Ltx2ContextCache {
        path,
        key,
        hit: false,
        encoder_seconds,
        log_path,
        manifest_path,
    })
}

pub(super) fn normalize_ltx2_prompt_fields(body: &Value) -> Value {
    let mut normalized = body.clone();
    let Some(object) = normalized.as_object_mut() else {
        return normalized;
    };
    for key in ["prompt", "negative"] {
        let Some(text) = object.get(key).and_then(Value::as_str) else {
            continue;
        };
        object.insert(key.to_string(), json!(text.trim()));
    }
    normalized
}

#[derive(Clone, Copy)]
pub(super) struct Ltx2CameraMotionSpec {
    pub(super) id: &'static str,
    pub(super) label: &'static str,
    pub(super) prompt_suffix: &'static str,
    pub(super) adapter_filename: Option<&'static str>,
}

pub(super) const LTX2_CAMERA_MOTIONS: [Ltx2CameraMotionSpec; 9] = [
    Ltx2CameraMotionSpec {
        id: "none",
        label: "None",
        prompt_suffix: "",
        adapter_filename: None,
    },
    Ltx2CameraMotionSpec {
        id: "static",
        label: "Static",
        prompt_suffix: ", static camera, locked off shot, no camera movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-static.safetensors"),
    },
    Ltx2CameraMotionSpec {
        id: "focus_shift",
        label: "Focus shift (prompt)",
        prompt_suffix: ", focus shift, rack focus, changing focal point",
        adapter_filename: None,
    },
    Ltx2CameraMotionSpec {
        id: "dolly_in",
        label: "Dolly in",
        prompt_suffix: ", dolly in, camera pushing forward, smooth forward movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-dolly-in.safetensors"),
    },
    Ltx2CameraMotionSpec {
        id: "dolly_out",
        label: "Dolly out",
        prompt_suffix: ", dolly out, camera pulling back, smooth backward movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-dolly-out.safetensors"),
    },
    Ltx2CameraMotionSpec {
        id: "dolly_left",
        label: "Dolly left",
        prompt_suffix: ", dolly left, camera tracking left, lateral movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-dolly-left.safetensors"),
    },
    Ltx2CameraMotionSpec {
        id: "dolly_right",
        label: "Dolly right",
        prompt_suffix: ", dolly right, camera tracking right, lateral movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-dolly-right.safetensors"),
    },
    Ltx2CameraMotionSpec {
        id: "jib_up",
        label: "Jib up",
        prompt_suffix: ", jib up, camera rising up, upward crane movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-jib-up.safetensors"),
    },
    Ltx2CameraMotionSpec {
        id: "jib_down",
        label: "Jib down",
        prompt_suffix: ", jib down, camera lowering down, downward crane movement",
        adapter_filename: Some("ltx-2-19b-lora-camera-control-jib-down.safetensors"),
    },
];

pub(super) fn ltx2_camera_motion_spec(id: &str) -> Option<Ltx2CameraMotionSpec> {
    LTX2_CAMERA_MOTIONS
        .iter()
        .copied()
        .find(|candidate| candidate.id == id)
}

pub(super) fn ltx2_camera_motion_documents() -> Vec<Value> {
    LTX2_CAMERA_MOTIONS
        .iter()
        .map(|motion| {
            let adapter_path = motion
                .adapter_filename
                .map(|filename| model_path(&format!("loras/{filename}")));
            let adapter_available = adapter_path.as_deref().is_some_and(nonempty_file);
            json!({
                "id": motion.id,
                "label": motion.label,
                "prompt_suffix": motion.prompt_suffix,
                "control": if motion.adapter_filename.is_some() { "lora" } else if motion.id == "none" { "none" } else { "prompt" },
                "adapter": motion.adapter_filename,
                "available": motion.adapter_filename.is_none() || adapter_available,
            })
        })
        .collect()
}

pub(super) fn normalized_ltx2_camera_motion_request(body: &Value) -> Result<Value, String> {
    let mut normalized = body.clone();
    let motion = body
        .get("camera_motion")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .trim();
    let spec = ltx2_camera_motion_spec(motion)
        .ok_or_else(|| format!("unsupported LTX camera_motion '{motion}'"))?;
    let object = normalized
        .as_object_mut()
        .ok_or_else(|| "LTX2 request must be an object".to_string())?;
    object.insert("camera_motion".to_string(), json!(motion));
    if !spec.prompt_suffix.is_empty()
        && object
            .get("creator_camera_motion_applied")
            .and_then(Value::as_bool)
            != Some(true)
    {
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        object.insert("creator_prompt".to_string(), json!(prompt));
        object.insert(
            "prompt".to_string(),
            json!(format!("{prompt}{}", spec.prompt_suffix)),
        );
        object.insert(
            "creator_camera_motion_suffix".to_string(),
            json!(spec.prompt_suffix),
        );
        object.insert("creator_camera_motion_applied".to_string(), json!(true));
    }
    if let Some(filename) = spec.adapter_filename {
        let adapter_path = model_path(&format!("loras/{filename}"));
        let rows = object
            .entry("lora".to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| "LTX2 'lora' must be an array".to_string())?;
        rows.retain(|row| row.get("source").and_then(Value::as_str) != Some("camera_control"));
        rows.push(json!({
            "name": filename,
            "weight": 1.0,
            "role": "overlay",
            "source": "camera_control",
            "camera_motion": motion,
        }));
        object.insert(
            "creator_camera_motion_adapter".to_string(),
            json!(adapter_path.to_string_lossy()),
        );
    }
    Ok(normalized)
}

pub(super) fn normalized_ltx2_checkpoint_workflow_request(body: &Value) -> Result<Value, String> {
    let mut normalized = body.clone();
    let checkpoint = body.get("checkpoint").and_then(Value::as_str).unwrap_or("");
    let Some(profile) = ltx2_checkpoint_workflow(checkpoint) else {
        return Ok(normalized);
    };
    let requested_profile = body
        .get("workflow_profile")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let profile_id = profile.get("id").and_then(Value::as_str).unwrap_or("");
    if let Some(requested) = requested_profile {
        if requested != profile_id {
            return Err(format!(
                "LTX2 checkpoint '{checkpoint}' does not provide workflow_profile '{requested}'; its creator profile is '{profile_id}'"
            ));
        }
    }
    let object = normalized
        .as_object_mut()
        .ok_or_else(|| "LTX2 request must be an object".to_string())?;
    for key in ["guidance_mode", "sampler", "scheduler", "steps"] {
        let value = profile
            .get(key)
            .cloned()
            .ok_or_else(|| format!("LTX2 creator profile '{profile_id}' is missing '{key}'"))?;
        object.insert(key.to_string(), value);
    }
    object.insert("workflow_profile".to_string(), json!(profile_id));
    object.insert(
        "creator_workflow_source".to_string(),
        profile.get("source").cloned().unwrap_or(Value::Null),
    );
    if object
        .get("negative")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        if let Some(default_negative) = profile.get("default_negative").and_then(Value::as_str) {
            object.insert("negative".to_string(), json!(default_negative));
        }
    }
    Ok(normalized)
}

pub(super) fn ltx2_feature_request(body: &Value) -> Result<Option<Value>, String> {
    let feature_id = match body.get("feature_id") {
        None => return Ok(None),
        Some(Value::String(value)) if value.trim().is_empty() || value == "standard" => {
            return Ok(None);
        }
        Some(Value::String(value)) => value.trim(),
        Some(_) => return Err("LTX2 feature_id must be a string".to_string()),
    };
    let mut feature = crate::models::ltx2_feature_document(feature_id)
        .ok_or_else(|| format!("unknown LTX2 feature workflow '{feature_id}'"))?;
    let status = feature
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("feature_runner_required");
    if !matches!(status, "overlay_admitted" | "v2a_admitted") {
        return Err(format!(
            "LTX2 feature workflow '{feature_id}' is installed but not runtime-admitted; status={status}"
        ));
    }
    let filename = feature
        .get("filename")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("LTX2 feature workflow '{feature_id}' has no adapter filename"))?;
    let path = model_path(&format!("loras/{filename}"));
    if !nonempty_file(&path) {
        return Err(format!(
            "LTX2 feature workflow '{feature_id}' is unavailable; missing {}",
            path.display()
        ));
    }
    let weight = body
        .get("feature_weight")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            format!(
                "LTX2 feature workflow '{feature_id}' requires an authored numeric feature_weight"
            )
        })?;
    let min_weight = feature
        .get("weight_min")
        .and_then(Value::as_f64)
        .unwrap_or(-10.0);
    let max_weight = feature
        .get("weight_max")
        .and_then(Value::as_f64)
        .unwrap_or(10.0);
    if !weight.is_finite() || !(min_weight..=max_weight).contains(&weight) {
        return Err(format!(
            "LTX2 feature workflow '{feature_id}' requires feature_weight in [{min_weight}, {max_weight}]; got {weight}"
        ));
    }
    if body
        .get("lora")
        .and_then(Value::as_array)
        .is_some_and(|rows| {
            rows.iter().any(|row| {
                row.get("name").and_then(Value::as_str).is_some_and(|name| {
                    std::path::Path::new(name)
                        .file_name()
                        .and_then(|value| value.to_str())
                        == Some(filename)
                })
            })
        })
    {
        return Err(format!(
            "LTX2 feature workflow '{feature_id}' already owns {filename}; remove the duplicate ordinary LoRA row"
        ));
    }
    let prompt = body.get("prompt").and_then(Value::as_str).unwrap_or("");
    let trigger = feature.get("trigger").and_then(Value::as_str).unwrap_or("");
    if !trigger.is_empty() && !prompt.contains(trigger) {
        return Err(format!(
            "LTX2 feature workflow '{feature_id}' requires the exact prompt trigger {trigger}"
        ));
    }
    match feature_id {
        "cinemagraph" => {
            if body
                .get("image_path")
                .and_then(Value::as_str)
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err("LTX2 Cinemagraph requires a loaded I2V source image".to_string());
            }
        }
        "foley-v2a" => {
            let video_path = body
                .get("video_path")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if video_path.is_empty() {
                return Err("LTX2 Foley requires a loaded V2V source video".to_string());
            }
            if body
                .get("video_mask_path")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
            {
                return Err(
                    "LTX2 Foley freezes the complete source video and does not accept a painted video mask"
                        .to_string(),
                );
            }
            if body.get("audio_policy").and_then(Value::as_str) != Some("generate")
                || body.get("include_audio").and_then(Value::as_bool) != Some(true)
            {
                return Err(
                    "LTX2 Foley requires Audio=Generate so only the audio stream is synthesized"
                        .to_string(),
                );
            }
            let video_strength = body
                .get("video_strength")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            if (video_strength - 1.0).abs() > f64::EPSILON {
                return Err(
                    "LTX2 Foley requires Source strength=1.0 to freeze every video token"
                        .to_string(),
                );
            }
            let probe = probe_video_path(video_path)?;
            let requested_width = body.get("width").and_then(Value::as_i64).unwrap_or(0);
            let requested_height = body.get("height").and_then(Value::as_i64).unwrap_or(0);
            let requested_frames = body.get("frames").and_then(Value::as_i64).unwrap_or(0);
            let requested_fps = body.get("fps").and_then(Value::as_f64).unwrap_or(0.0);
            let actual_fps = probe.get("fps").and_then(Value::as_f64).unwrap_or(0.0);
            if probe.get("width").and_then(Value::as_i64) != Some(requested_width)
                || probe.get("height").and_then(Value::as_i64) != Some(requested_height)
                || probe.get("frame_count").and_then(Value::as_i64) != Some(requested_frames)
                || (actual_fps - requested_fps).abs() > 0.01
            {
                return Err(format!(
                    "LTX2 Foley preserves the original video bitstream, so the source must exactly match {requested_width}x{requested_height}, {requested_frames} frames at {requested_fps} FPS; probe={probe}"
                ));
            }
        }
        _ => {
            return Err(format!(
                "LTX2 feature workflow '{feature_id}' has no admitted request contract"
            ));
        }
    }
    if let Some(object) = feature.as_object_mut() {
        object.insert("path".to_string(), json!(path.to_string_lossy()));
        object.insert("weight".to_string(), json!(weight));
    }
    Ok(Some(feature))
}

pub(super) fn normalized_ltx2_feature_request(body: &Value) -> Result<Value, String> {
    let Some(feature) = ltx2_feature_request(body)? else {
        return Ok(body.clone());
    };
    let mut normalized = body.clone();
    let request = normalized
        .as_object_mut()
        .ok_or_else(|| "LTX2 request must be a JSON object".to_string())?;
    let rows = request
        .entry("lora".to_string())
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| "LTX2 'lora' must be an array".to_string())?;
    let mut row = serde_json::Map::new();
    row.insert(
        "name".to_string(),
        feature.get("filename").cloned().unwrap_or(Value::Null),
    );
    row.insert(
        "weight".to_string(),
        feature.get("weight").cloned().unwrap_or(Value::Null),
    );
    row.insert(
        "feature_id".to_string(),
        feature.get("id").cloned().unwrap_or(Value::Null),
    );
    row.insert(
        "feature_usage".to_string(),
        feature.get("usage").cloned().unwrap_or(Value::Null),
    );
    if let Some(streams) = feature.get("stream_strengths").and_then(Value::as_object) {
        for key in [
            "video",
            "video_to_audio",
            "audio",
            "audio_to_video",
            "other",
        ] {
            if let Some(value) = streams.get(key) {
                row.insert(key.to_string(), value.clone());
            }
        }
    }
    rows.push(Value::Object(row));
    request.insert("feature_adapter".to_string(), feature);
    Ok(normalized)
}

pub(super) fn normalized_ltx2_video_edit_request(body: &Value) -> Result<Value, String> {
    let mode = body
        .get("video_edit_mode")
        .and_then(Value::as_str)
        .unwrap_or("standard")
        .trim();
    if !matches!(mode, "standard" | "retake" | "extend_start" | "extend_end") {
        return Err(format!(
            "LTX2 video_edit_mode must be standard, retake, extend_start, or extend_end; got '{mode}'"
        ));
    }
    if mode == "standard" {
        return Ok(body.clone());
    }
    let video_path = body
        .get("video_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if video_path.is_empty() {
        return Err(format!("LTX2 {mode} requires video_path"));
    }
    if body
        .get("image_path")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return Err(format!("LTX2 {mode} cannot also use image_path"));
    }
    if body
        .get("video_mask_path")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return Err(format!(
            "LTX2 {mode} uses a temporal edit mask and cannot also use video_mask_path"
        ));
    }
    let requested_width = body.get("width").and_then(Value::as_i64).unwrap_or(0);
    let requested_height = body.get("height").and_then(Value::as_i64).unwrap_or(0);
    let requested_frames = body.get("frames").and_then(Value::as_i64).unwrap_or(0);
    let requested_fps = body.get("fps").and_then(Value::as_f64).unwrap_or(0.0);
    if requested_width <= 0
        || requested_height <= 0
        || requested_frames <= 1
        || requested_fps <= 0.0
    {
        return Err(
            "LTX2 temporal edit requires valid target width, height, frames, and FPS".to_string(),
        );
    }
    let probe = probe_video_path(video_path)?;
    let source_width = probe.get("width").and_then(Value::as_i64).unwrap_or(0);
    let source_height = probe.get("height").and_then(Value::as_i64).unwrap_or(0);
    let source_frames = probe
        .get("frame_count")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let source_fps = probe.get("fps").and_then(Value::as_f64).unwrap_or(0.0);
    if probe.get("muxing").and_then(Value::as_str) != Some("probe_ok") || source_frames <= 1 {
        return Err(format!("LTX2 {mode} source probe is incomplete: {probe}"));
    }
    if source_width != requested_width
        || source_height != requested_height
        || (source_fps - requested_fps).abs() > 0.01
    {
        return Err(format!(
            "LTX2 {mode} preserves native source geometry and FPS; source is {source_width}x{source_height}, {source_frames} frames at {source_fps} FPS but target is {requested_width}x{requested_height}, {requested_frames} frames at {requested_fps} FPS"
        ));
    }
    let mut normalized = body.clone();
    let request = normalized
        .as_object_mut()
        .ok_or_else(|| "LTX2 request must be a JSON object".to_string())?;
    request.insert("video_edit_mode".to_string(), json!(mode));
    request.insert("video_source_frames".to_string(), json!(source_frames));
    if mode == "retake" {
        if source_frames != requested_frames {
            return Err(format!(
                "LTX2 Retake requires an exact compiled source profile; source has {source_frames} frames but target has {requested_frames}"
            ));
        }
        let start = body
            .get("video_edit_start")
            .and_then(Value::as_f64)
            .ok_or_else(|| "LTX2 Retake requires numeric video_edit_start".to_string())?;
        let end = body
            .get("video_edit_end")
            .and_then(Value::as_f64)
            .ok_or_else(|| "LTX2 Retake requires numeric video_edit_end".to_string())?;
        let source_duration = (source_frames - 1) as f64 / source_fps;
        if start < 0.0 || end <= start || end - start < 2.0 || end > source_duration + 0.001 {
            return Err(format!(
                "LTX2 Retake window must be at least 2 seconds and within 0..{source_duration:.3}; got {start:.3}..{end:.3}"
            ));
        }
        request.insert("video_edit_start".to_string(), json!(start));
        request.insert("video_edit_end".to_string(), json!(end));
        // LTX Desktop's TemporalRegionMask is binary. Retake has no soft
        // preservation-strength blend: the selected interval is regenerated
        // at mask=1 and every other token is frozen at mask=0.
        request.insert("video_strength".to_string(), json!(0.0));
    } else {
        // LTX Desktop zero-pads the extension region in latent space; the
        // temporal mask fully denoises that region plus the 0.5-second seam.
        request.insert("video_strength".to_string(), json!(0.0));
        if source_frames >= requested_frames {
            return Err(format!(
                "LTX2 Extend target must be longer than its source; source has {source_frames} frames and target has {requested_frames}"
            ));
        }
        let extension_frames = requested_frames - source_frames;
        if extension_frames % 8 != 0 {
            return Err(format!(
                "LTX2 Extend requires an 8-frame-aligned extension; target-source delta is {extension_frames}"
            ));
        }
        let seam_frames = (0.5 * requested_fps).round() as i64;
        let (start, end) = if mode == "extend_start" {
            (
                0.0,
                (extension_frames + seam_frames).min(requested_frames - 1) as f64 / requested_fps,
            )
        } else {
            (
                (source_frames - 1 - seam_frames).max(0) as f64 / requested_fps,
                (requested_frames - 1) as f64 / requested_fps,
            )
        };
        request.insert("video_edit_start".to_string(), json!(start));
        request.insert("video_edit_end".to_string(), json!(end));
        request.insert("video_extend_frames".to_string(), json!(extension_frames));
        request.insert(
            "video_extend_seconds".to_string(),
            json!(extension_frames as f64 / requested_fps),
        );
        request.insert("video_seam_seconds".to_string(), json!(0.5));
    }
    Ok(normalized)
}

pub(super) fn resolve_ltx2_request_checkpoint(
    body: &Value,
) -> Result<crate::models::ResolvedCheckpoint, String> {
    let selection = body
        .get("checkpoint")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let checkpoint = crate::models::resolve_checkpoint(selection).ok_or_else(|| {
        format!(
            "LTX2 checkpoint '{selection}' was not found in the scanned Serenity model registry"
        )
    })?;
    if checkpoint.arch != "ltx2" {
        return Err(format!(
            "checkpoint '{}' is classified as '{}', not LTX2; set its Model Type to LTX 2 / 2.3 if this is a compatible full finetune",
            checkpoint.name, checkpoint.arch
        ));
    }
    let checkpoint_id = checkpoint
        .name
        .strip_suffix(".safetensors")
        .unwrap_or(&checkpoint.name);
    let known_ltx23_single_file = matches!(
        checkpoint_id,
        "ltx-2.3-22b-dev-fp8"
            | "ltx-2.3-22b-dev-fp8-dequant-bf16"
            | "ltx-2.3-22b-distilled-fp8"
            | "ltx-2.3-22b-distilled-fp8-dequant-bf16"
    );
    if !matches!(
        checkpoint.format.as_str(),
        "diffusion_model" | "full_checkpoint"
    ) {
        return Err(format!(
            "LTX2 selected-checkpoint loading requires one complete safetensors file; '{}' has format '{}'",
            checkpoint.name, checkpoint.format
        ));
    }
    if checkpoint.arch_source == "filename"
        && checkpoint.arch_override.is_empty()
        && !known_ltx23_single_file
    {
        return Err(format!(
            "checkpoint '{}' only resembles LTX by filename; confirm it is an LTX 2.3-compatible full checkpoint by setting Model Type to LTX 2 / 2.3",
            checkpoint.name
        ));
    }
    if !nonempty_file(&checkpoint.path) {
        return Err(format!(
            "selected LTX2 checkpoint is missing or empty: {}",
            checkpoint.path.display()
        ));
    }
    Ok(checkpoint)
}

#[derive(Clone, Debug)]
pub(super) struct Ltx2DistillationAdapter {
    pub(super) name: String,
    pub(super) path: std::path::PathBuf,
    pub(super) weight: f64,
    pub(super) stage1_weight: Option<f64>,
    pub(super) stage2_weight: Option<f64>,
    pub(super) source: &'static str,
}

pub(super) fn is_official_ltx2_dev_checkpoint_name(name: &str) -> bool {
    matches!(
        name.strip_suffix(".safetensors").unwrap_or(name),
        "ltx-2.3-22b-dev" | LTX2_REFHQ_CHECKPOINT | LTX2_REFHQ_BF16_CHECKPOINT
    )
}

pub(super) fn is_official_ltx2_dev_checkpoint(
    checkpoint: &crate::models::ResolvedCheckpoint,
) -> bool {
    is_official_ltx2_dev_checkpoint_name(&checkpoint.name)
}

/// Resolve the sampling adapter that belongs to the selected checkpoint.
///
/// Ordinary authored LoRAs remain in `lora[]`. A row explicitly marked
/// `role=distillation` replaces the official creator adapter. The official
/// adapter is only implicit for the two official dev checkpoints; arbitrary
/// finetunes never inherit it by accident, and directly distilled full
/// checkpoints never receive a second distillation delta.
pub(super) fn resolve_ltx2_distillation_adapter(
    body: &Value,
    checkpoint: &crate::models::ResolvedCheckpoint,
) -> Result<Option<Ltx2DistillationAdapter>, String> {
    let mut explicit: Option<Ltx2DistillationAdapter> = None;
    if let Some(rows) = body.get("lora").and_then(Value::as_array) {
        for (index, row) in rows.iter().enumerate() {
            let role = row.get("role").and_then(Value::as_str).unwrap_or("overlay");
            if role != "distillation" {
                continue;
            }
            if explicit.is_some() {
                return Err("LTX2 requests admit at most one role=distillation adapter".to_string());
            }
            let name = row.get("name").and_then(Value::as_str).unwrap_or("");
            let weight = row.get("weight").and_then(Value::as_f64).unwrap_or(1.0);
            let Some((path, arch)) = crate::models::lora_path_and_arch(name) else {
                return Err(format!(
                    "LTX2 lora[{index}] distillation adapter not found in the model registry: {name}"
                ));
            };
            if arch != "ltx2" {
                return Err(format!(
                    "LTX2 lora[{index}] distillation adapter '{name}' targets '{arch}', not ltx2"
                ));
            }
            explicit = Some(Ltx2DistillationAdapter {
                name: name.to_string(),
                path,
                weight,
                stage1_weight: None,
                stage2_weight: None,
                source: "user",
            });
        }
    }

    let guidance_mode = body
        .get("guidance_mode")
        .and_then(Value::as_str)
        .unwrap_or("");
    let edit_mode = body
        .get("video_edit_mode")
        .and_then(Value::as_str)
        .unwrap_or("standard");
    if edit_mode != "standard" {
        let checkpoint_id = checkpoint
            .name
            .strip_suffix(".safetensors")
            .unwrap_or(&checkpoint.name)
            .to_ascii_lowercase();
        if checkpoint_id.contains("distill") {
            if explicit.is_some() {
                return Err(
                    "LTX2 Retake/Extend selected a directly distilled full checkpoint; remove the separate role=distillation adapter"
                        .to_string(),
                );
            }
            return Ok(None);
        }
        return Ok(explicit);
    }
    if guidance_mode == "dev" {
        if explicit.is_some() {
            return Err(
                "role=distillation requires Distilled guidance; remove it for pristine Dev CFG sampling"
                    .to_string(),
            );
        }
        return Ok(None);
    }

    let checkpoint_id = checkpoint
        .name
        .strip_suffix(".safetensors")
        .unwrap_or(&checkpoint.name);
    if let Some(workflow) = ltx2_checkpoint_workflow(checkpoint_id) {
        let workflow_id = workflow.get("id").and_then(Value::as_str).unwrap_or("");
        let requested_workflow = body
            .get("workflow_profile")
            .and_then(Value::as_str)
            .unwrap_or("");
        if requested_workflow == workflow_id {
            if explicit.is_some() {
                return Err(format!(
                    "checkpoint '{}' already uses creator workflow '{}'; remove the manually selected Distillation LoRA",
                    checkpoint.name, workflow_id
                ));
            }
            let adapter = workflow.get("distillation_adapter").ok_or_else(|| {
                format!("creator workflow '{workflow_id}' has no distillation adapter")
            })?;
            let relative_path = adapter.get("path").and_then(Value::as_str).unwrap_or("");
            let path = model_path(relative_path);
            if !nonempty_file(&path) {
                return Err(format!(
                    "creator workflow '{workflow_id}' requires its distillation adapter: {}",
                    path.display()
                ));
            }
            return Ok(Some(Ltx2DistillationAdapter {
                name: adapter
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("creator-distillation-adapter")
                    .to_string(),
                path,
                weight: 1.0,
                stage1_weight: adapter.get("stage1_weight").and_then(Value::as_f64),
                stage2_weight: adapter.get("stage2_weight").and_then(Value::as_f64),
                source: "checkpoint_creator_profile",
            }));
        }
    }
    let direct_distilled = checkpoint_id.to_ascii_lowercase().contains("distill");
    if direct_distilled {
        if explicit.is_some() {
            return Err(format!(
                "checkpoint '{}' is already a directly distilled full checkpoint; remove the separate role=distillation adapter",
                checkpoint.name
            ));
        }
        return Ok(None);
    }
    if let Some(adapter) = explicit {
        return Ok(Some(adapter));
    }
    if is_official_ltx2_dev_checkpoint(checkpoint) {
        let path = model_path(LTX2_REFHQ_DISTILLATION_ADAPTER);
        if !nonempty_file(&path) {
            return Err(format!(
                "official LTX2 dev fast sampling requires its creator distillation adapter: {}",
                path.display()
            ));
        }
        return Ok(Some(Ltx2DistillationAdapter {
            name: std::path::Path::new(LTX2_REFHQ_DISTILLATION_ADAPTER)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("official-ltx2-distillation-adapter")
                .to_string(),
            path,
            weight: 1.0,
            stage1_weight: None,
            stage2_weight: None,
            source: "official_creator_default",
        }));
    }
    Err(format!(
        "checkpoint '{}' is an arbitrary dev/full finetune. Distilled guidance requires its matching LoRA marked Distillation in the LoRA row; choose Dev CFG to run the checkpoint pristine",
        checkpoint.name
    ))
}

pub(super) fn resolve_ltx2_retake_checkpoint(
    selected: crate::models::ResolvedCheckpoint,
    has_distillation_adapter: bool,
) -> Result<crate::models::ResolvedCheckpoint, String> {
    let selected_id = selected
        .name
        .strip_suffix(".safetensors")
        .unwrap_or(&selected.name);
    let selected_is_distilled = selected_id.to_ascii_lowercase().contains("distill");
    if selected_is_distilled {
        if !ltx2_checkpoint_has_creator_edit_components(&selected.path) {
            return Err(format!(
                "LTX2 Retake requires the creator's complete checkpoint (transformer plus video/audio VAE encoders); '{}' is a partial diffusion-only artifact",
                selected.name
            ));
        }
        return Ok(selected);
    }
    if is_official_ltx2_dev_checkpoint(&selected) {
        for direct_distilled in ["ltx-2.3-22b-distilled-1.1", "ltx-2.3-22b-distilled"] {
            if let Some(candidate) = crate::models::resolve_checkpoint(direct_distilled) {
                if ltx2_checkpoint_has_creator_edit_components(&candidate.path) {
                    return Ok(candidate);
                }
            }
        }
        return Err(
            "LTX2 Retake follows the creator's one-stage BF16 topology. Install/select the complete ltx-2.3-22b-distilled-1.1 checkpoint; the partial FP8 diffusion files do not contain the source audio/video encoders"
                .to_string(),
        );
    }
    if has_distillation_adapter {
        if !ltx2_checkpoint_has_creator_edit_components(&selected.path) {
            return Err(format!(
                "LTX2 Retake cannot use '{}' because it lacks the creator video/audio VAE encoder weights",
                selected.name
            ));
        }
        return Ok(selected);
    }
    Err(format!(
        "LTX2 Retake requires either a directly distilled complete checkpoint or the selected dev/full checkpoint's creator-specified role=distillation adapter; '{}' has neither",
        selected.name
    ))
}

pub(super) fn ltx2_checkpoint_has_creator_edit_components(path: &std::path::Path) -> bool {
    let Some(header) = safetensors_header(path) else {
        return false;
    };
    header
        .get("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight")
        .is_some()
        && header.get("vae.encoder.conv_in.conv.weight").is_some()
        && header
            .get("audio_vae.encoder.conv_in.conv.weight")
            .is_some()
}

pub(super) fn validate_ltx2_mojo_request(body: &Value) -> Result<(), String> {
    let required_strings = [
        "checkpoint",
        "prompt",
        "sampler",
        "scheduler",
        "guidance_mode",
    ];
    for key in required_strings {
        let value = body.get(key).and_then(Value::as_str).unwrap_or("").trim();
        if value.is_empty() {
            return Err(format!("LTX2 Mojo request requires non-empty '{key}'"));
        }
    }
    let quant = body.get("quant").and_then(Value::as_str).unwrap_or("");
    if !matches!(quant, "bf16" | "fp8" | "int4") {
        return Err(format!(
            "LTX2 Mojo request requires quant 'bf16', 'fp8', or 'int4'; got '{quant}'"
        ));
    }
    let checkpoint = body["checkpoint"].as_str().unwrap_or("");
    let checkpoint_profile = ltx2_checkpoint_profile(checkpoint).ok_or_else(|| {
        let admitted = ltx2_request_profile_registry()
            .checkpoints
            .iter()
            .map(|profile| profile.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "LTX2 checkpoint '{checkpoint}' is not registered; admitted checkpoints: {admitted}"
        )
    })?;
    if !checkpoint_profile
        .quant_modes
        .iter()
        .any(|mode| mode == quant)
    {
        return Err(format!(
            "LTX2 checkpoint '{}' does not admit quant '{}'; admitted modes: {}",
            checkpoint_profile.id,
            quant,
            checkpoint_profile.quant_modes.join(", ")
        ));
    }
    if quant == "int4" {
        let checkpoint_name = checkpoint_profile.id.as_str();
        if checkpoint_name != LTX2_REFHQ_CHECKPOINT {
            return Err(format!(
                "LTX2 int4 uses a checkpoint-specific slab and is only available for '{LTX2_REFHQ_CHECKPOINT}'; select fp8 or bf16 for '{}'",
                checkpoint_profile.id
            ));
        }
    }
    let guidance_mode = body["guidance_mode"].as_str().unwrap_or("");
    if !matches!(guidance_mode, "distilled" | "dev") {
        return Err(format!(
            "LTX2 guidance_mode must be 'distilled' or 'dev'; got '{guidance_mode}'"
        ));
    }
    if !checkpoint_profile
        .guidance_modes
        .iter()
        .any(|mode| mode == guidance_mode)
    {
        return Err(format!(
            "LTX2 checkpoint '{}' does not admit guidance_mode '{}'; admitted modes: {}",
            checkpoint_profile.id,
            guidance_mode,
            checkpoint_profile.guidance_modes.join(", ")
        ));
    }
    let sampler = body["sampler"].as_str().unwrap_or("");
    let scheduler = body["scheduler"].as_str().unwrap_or("");
    let steps = body.get("steps").and_then(Value::as_i64).unwrap_or(0);
    let workflow_profile = body
        .get("workflow_profile")
        .and_then(Value::as_str)
        .unwrap_or("");
    let selected_workflow = ltx2_checkpoint_workflow(&checkpoint_profile.id)
        .filter(|profile| profile.get("id").and_then(Value::as_str) == Some(workflow_profile));
    if !workflow_profile.is_empty() && selected_workflow.is_none() {
        return Err(format!(
            "LTX2 workflow_profile '{workflow_profile}' is not registered for checkpoint '{}'",
            checkpoint_profile.id
        ));
    }
    if let Some(workflow) = selected_workflow {
        let expected_guidance = workflow
            .get("guidance_mode")
            .and_then(Value::as_str)
            .unwrap_or("");
        let expected_sampler = workflow
            .get("sampler")
            .and_then(Value::as_str)
            .unwrap_or("");
        let expected_scheduler = workflow
            .get("scheduler")
            .and_then(Value::as_str)
            .unwrap_or("");
        let expected_steps = workflow.get("steps").and_then(Value::as_i64).unwrap_or(0);
        if guidance_mode != expected_guidance
            || sampler != expected_sampler
            || scheduler != expected_scheduler
            || steps != expected_steps
        {
            return Err(format!(
                "LTX2 creator workflow '{workflow_profile}' requires guidance_mode={expected_guidance}, sampler={expected_sampler}, scheduler={expected_scheduler}, and steps={expected_steps}; got guidance_mode={guidance_mode}, sampler={sampler}, scheduler={scheduler}, steps={steps}"
            ));
        }
    } else {
        match guidance_mode {
            "distilled" if sampler != "euler" || scheduler != "ltx2_distilled" || steps != 8 => {
                return Err(format!(
                "LTX2 distilled mode requires sampler=euler, scheduler=ltx2_distilled, and steps=8; got sampler={sampler}, scheduler={scheduler}, steps={steps}"
            ));
            }
            "dev" if sampler != "res2s" || scheduler != "ltx2" || !(1..=20).contains(&steps) => {
                return Err(format!(
                "LTX2 dev mode requires sampler=res2s, scheduler=ltx2, and steps in [1, 20]; got sampler={sampler}, scheduler={scheduler}, steps={steps}"
            ));
            }
            _ => {}
        }
    }
    let prompt_enhancer = body
        .get("prompt_enhancer")
        .and_then(Value::as_str)
        .unwrap_or("none");
    if prompt_enhancer != "none" {
        let Some(workflow) = selected_workflow else {
            return Err(format!(
                "LTX2 prompt_enhancer '{prompt_enhancer}' requires a registered checkpoint creator workflow"
            ));
        };
        let enhancer = workflow.get("prompt_enhancer").ok_or_else(|| {
            format!("LTX2 creator workflow '{workflow_profile}' has no prompt enhancer")
        })?;
        let expected = enhancer.get("id").and_then(Value::as_str).unwrap_or("");
        if prompt_enhancer != expected {
            return Err(format!(
                "LTX2 creator workflow '{workflow_profile}' provides prompt_enhancer '{expected}', not '{prompt_enhancer}'"
            ));
        }
        let missing = ["weights", "mmproj"]
            .into_iter()
            .filter_map(|key| {
                let path = enhancer.get(key).and_then(Value::as_str)?;
                let resolved = model_path(path);
                (!nonempty_file(&resolved)).then(|| resolved.display().to_string())
            })
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "LTX2 prompt_enhancer '{prompt_enhancer}' is unavailable; missing creator files: {}",
                missing.join(", ")
            ));
        }
        return Err(format!(
            "LTX2 prompt_enhancer '{prompt_enhancer}' files are installed, but its ephemeral llama.cpp execution route is not implemented; use prompt_enhancer='none' so Serenity cannot claim enhancement that did not run"
        ));
    }
    for key in ["width", "height", "frames", "steps", "seed"] {
        if body.get(key).and_then(Value::as_i64).is_none() {
            return Err(format!("LTX2 Mojo request requires integer '{key}'"));
        }
    }
    if !body.get("fps").map(Value::is_number).unwrap_or(false) {
        return Err("LTX2 Mojo request requires numeric 'fps'".to_string());
    }
    let width = body["width"].as_i64().unwrap_or(0);
    let height = body["height"].as_i64().unwrap_or(0);
    let frames = body["frames"].as_i64().unwrap_or(0);
    let fps = body["fps"].as_f64().unwrap_or(0.0);
    let edit_mode = body
        .get("video_edit_mode")
        .and_then(Value::as_str)
        .unwrap_or("standard");
    if ltx2_request_profile_for_mode(width, height, frames, fps, edit_mode).is_none() {
        let supported = ltx2_resolved_profiles()
            .iter()
            .filter(|profile| profile.modes.iter().any(|candidate| candidate == edit_mode))
            .map(|profile| {
                format!(
                    "{}x{} {}f@{}",
                    profile.width, profile.height, profile.frames, profile.fps
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "unsupported LTX2 {edit_mode} native profile {width}x{height}, {frames} frames at {fps} FPS; admitted profiles: {supported}"
        ));
    }
    if !body
        .get("include_audio")
        .map(Value::is_boolean)
        .unwrap_or(false)
    {
        return Err("LTX2 Mojo request requires boolean 'include_audio'".to_string());
    }
    let include_audio = body["include_audio"].as_bool().unwrap_or(false);
    let audio_policy = body
        .get("audio_policy")
        .and_then(Value::as_str)
        .unwrap_or(if include_audio { "generate" } else { "none" });
    if !matches!(audio_policy, "none" | "generate" | "preserve") {
        return Err(format!(
            "LTX2 audio_policy must be 'none', 'generate', or 'preserve'; got '{audio_policy}'"
        ));
    }
    if (audio_policy == "generate") != include_audio {
        return Err(format!(
            "LTX2 audio_policy '{audio_policy}' conflicts with include_audio={include_audio}"
        ));
    }
    let _ = ltx2_requested_post_upscale(body)?;
    let caps = body
        .get("caps_positive")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if caps.is_empty() {
        let missing = ltx2_mojo_conditioning_missing();
        if !missing.is_empty() {
            return Err(format!(
                "LTX2 automatic prompt conditioning is unavailable; missing: {}",
                missing.join(", ")
            ));
        }
    } else if !std::path::Path::new(caps).is_file() {
        return Err(format!("LTX2 conditioning artifact not found: {caps}"));
    }
    if let Some(caps_negative) = body.get("caps_negative").and_then(Value::as_str) {
        if !caps_negative.trim().is_empty() && !std::path::Path::new(caps_negative.trim()).is_file()
        {
            return Err(format!(
                "LTX2 negative conditioning artifact not found: {}",
                caps_negative.trim()
            ));
        }
    }
    if let Some(noise) = body.get("noise_fixture").and_then(Value::as_str) {
        if !noise.is_empty() && !std::path::Path::new(noise).is_file() {
            return Err(format!("LTX2 noise fixture not found: {noise}"));
        }
    }
    let image_path = body
        .get("image_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !image_path.is_empty() && !std::path::Path::new(image_path).is_file() {
        return Err(format!("LTX2 I2V source image not found: {image_path}"));
    }
    let image_strength = body
        .get("image_strength")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if body
        .get("image_strength")
        .is_some_and(|value| !value.is_number())
    {
        return Err("LTX2 image_strength must be numeric".to_string());
    }
    if !(0.0..=1.0).contains(&image_strength) {
        return Err("LTX2 image_strength must be in [0, 1]".to_string());
    }
    if image_path.is_empty() && body.get("image_strength").is_some() {
        return Err("LTX2 image_strength requires image_path".to_string());
    }
    let last_image_path = body
        .get("last_image_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !last_image_path.is_empty() && !std::path::Path::new(last_image_path).is_file() {
        return Err(format!(
            "LTX2 last-frame source image not found: {last_image_path}"
        ));
    }
    let last_image_strength = body
        .get("last_image_strength")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if body
        .get("last_image_strength")
        .is_some_and(|value| !value.is_number())
    {
        return Err("LTX2 last_image_strength must be numeric".to_string());
    }
    if !(0.0..=1.0).contains(&last_image_strength) {
        return Err("LTX2 last_image_strength must be in [0, 1]".to_string());
    }
    if last_image_path.is_empty() && body.get("last_image_strength").is_some() {
        return Err("LTX2 last_image_strength requires last_image_path".to_string());
    }
    if !last_image_path.is_empty() && edit_mode != "standard" {
        return Err("LTX2 last-frame keyframe interpolation cannot use Retake/Extend".to_string());
    }
    let video_path = body
        .get("video_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !video_path.is_empty() && !std::path::Path::new(video_path).is_file() {
        return Err(format!("LTX2 V2V source video not found: {video_path}"));
    }
    let video_strength = body
        .get("video_strength")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if body
        .get("video_strength")
        .is_some_and(|value| !value.is_number())
    {
        return Err("LTX2 video_strength must be numeric".to_string());
    }
    if !(0.0..=1.0).contains(&video_strength) {
        return Err("LTX2 video_strength must be in [0, 1]".to_string());
    }
    if video_path.is_empty() && body.get("video_strength").is_some() {
        return Err("LTX2 video_strength requires video_path".to_string());
    }
    let video_mask_path = body
        .get("video_mask_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if !video_mask_path.is_empty() && !std::path::Path::new(video_mask_path).is_file() {
        return Err(format!("LTX2 V2V mask image not found: {video_mask_path}"));
    }
    if !video_mask_path.is_empty() && video_path.is_empty() {
        return Err("LTX2 video_mask_path requires video_path".to_string());
    }
    if !image_path.is_empty() && !video_path.is_empty() {
        return Err("LTX2 image_path and video_path are mutually exclusive".to_string());
    }
    if !last_image_path.is_empty() && !video_path.is_empty() {
        return Err("LTX2 last_image_path and video_path are mutually exclusive".to_string());
    }
    if audio_policy == "preserve" {
        if video_path.is_empty() {
            return Err("LTX2 audio_policy 'preserve' requires video_path".to_string());
        }
        let source_probe = probe_video_path(video_path)?;
        if source_probe.get("has_audio").and_then(Value::as_bool) != Some(true) {
            return Err(
                "LTX2 audio_policy 'preserve' requires an audio stream in the source video"
                    .to_string(),
            );
        }
    }
    match body.get("lora") {
        Some(Value::Array(rows)) => {
            for (index, row) in rows.iter().enumerate() {
                let Some(obj) = row.as_object() else {
                    return Err(format!("LTX2 lora[{index}] must be an object"));
                };
                let name = obj.get("name").and_then(Value::as_str).unwrap_or("");
                if name.is_empty() {
                    return Err(format!("LTX2 lora[{index}].name is required"));
                }
                let role = obj.get("role").and_then(Value::as_str).unwrap_or("overlay");
                if !matches!(role, "overlay" | "distillation") {
                    return Err(format!(
                        "LTX2 lora[{index}].role must be 'overlay' or 'distillation'; got '{role}'"
                    ));
                }
                let weight = obj.get("weight").and_then(Value::as_f64).unwrap_or(1.0);
                if obj.get("weight").is_some_and(|v| !v.is_number()) {
                    return Err(format!("LTX2 lora[{index}].weight must be numeric"));
                }
                if !(-10.0..=10.0).contains(&weight) {
                    return Err(format!("LTX2 lora[{index}].weight must be in [-10, 10]"));
                }
                let usage = crate::models::lora_usage(name);
                if usage != "overlay" {
                    return Err(format!(
                        "LTX2 lora[{index}] '{name}' is classified as '{usage}' and requires the dedicated LTX2 feature workflow"
                    ));
                }
                let Some((path, arch)) = crate::models::lora_path_and_arch(name) else {
                    return Err(format!(
                        "LTX2 lora[{index}] not found in the model registry: {name}"
                    ));
                };
                if !path.is_file() {
                    return Err(format!(
                        "LTX2 lora[{index}] registry path is missing: {}",
                        path.display()
                    ));
                }
                if arch != "ltx2" {
                    return Err(format!(
                        "LTX2 lora[{index}] '{name}' targets '{arch}', not ltx2"
                    ));
                }
            }
        }
        Some(_) => return Err("LTX2 'lora' must be an array".to_string()),
        None => return Err("LTX2 Mojo request requires the authored 'lora' array".to_string()),
    }
    let _ = ltx2_feature_request(body)?;
    Ok(())
}

/// Validate installed files only after the complete authored request contract
/// has passed. Keeping this separate makes semantic preflight deterministic on
/// every machine while the HTTP route still rejects missing runtime artifacts
/// before acquiring the GPU lease.
pub(super) fn validate_ltx2_runtime_artifacts(body: &Value) -> Result<(), String> {
    let selection = body
        .get("checkpoint")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let quant = body.get("quant").and_then(Value::as_str).unwrap_or("");
    if quant == "bf16" && selection == LTX2_REFHQ_BF16_CHECKPOINT {
        let path = model_path(LTX2_REFHQ_BF16);
        if !nonempty_file(&path) {
            return Err(format!(
                "LTX2 dequantized dev checkpoint is missing: {}",
                path.display()
            ));
        }
    }
    let checkpoint = resolve_ltx2_request_checkpoint(body)?;
    if quant == "int4" {
        let slab = model_path(LTX2_REFHQ_INT4_SLAB);
        if !nonempty_file(&slab) {
            return Err(format!(
                "LTX2 request selected int4 but the dev-model slab is missing: {}",
                slab.display()
            ));
        }
    }
    let _ = resolve_ltx2_distillation_adapter(body, &checkpoint)?;
    Ok(())
}

pub(super) fn ltx2_requested_post_upscale(body: &Value) -> Result<Option<(String, i64)>, String> {
    let Some(value) = body.get("post_upscale") else {
        return Ok(None);
    };
    let Some(object) = value.as_object() else {
        return Err("LTX2 post_upscale must be an object".to_string());
    };
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .trim();
    if id.is_empty() || id == "none" {
        return Ok(None);
    }
    let factor = object
        .get("factor")
        .and_then(Value::as_i64)
        .ok_or_else(|| "LTX2 post_upscale.factor must be an integer".to_string())?;
    if !matches!(factor, 2 | 4) {
        return Err(format!(
            "LTX2 post-upscale factor must be 2 or 4; got {factor}"
        ));
    }
    if matches!(id, "realesrgan-x4plus" | "realesrgan-fast-x4v3") {
        let weights = if id == "realesrgan-fast-x4v3" {
            REALESRGAN_FAST_X4_WEIGHTS
        } else {
            REALESRGAN_X4_WEIGHTS
        };
        let mut missing = Vec::new();
        if !bin_x(REALESRGAN_X4_RUNNER) {
            missing.push(repo_path(REALESRGAN_X4_RUNNER));
        }
        if !nonempty_file(&model_path(weights)) {
            missing.push(model_path(weights));
        }
        if !missing.is_empty() {
            return Err(format!(
                "Real-ESRGAN post-upscale is unavailable; missing: {}",
                missing
                    .iter()
                    .map(|path| path.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        return Ok(Some((id.to_string(), factor)));
    }
    if id == "seedvr2-3b" {
        return Err(
            "SeedVR2 source is present, but its GitHub CLI is still a fixture/demo \
             runner and is not admitted for user-video post-upscale"
                .to_string(),
        );
    }
    Err(format!("unsupported LTX2 post-upscaler '{id}'"))
}

pub(super) fn run_realesrgan_video_post_upscale<F>(
    upscaler_id: &str,
    native_artifact: &std::path::Path,
    out_dir: &std::path::Path,
    width: i64,
    height: i64,
    frames: i64,
    fps: i64,
    factor: i64,
    mut progress: F,
) -> Result<(std::path::PathBuf, Value), String>
where
    F: FnMut(i64, i64, &str),
{
    if !native_artifact.is_file() {
        return Err(format!(
            "native LTX2 artifact is missing: {}",
            native_artifact.display()
        ));
    }
    if !matches!(factor, 2 | 4) {
        return Err(format!("unsupported Real-ESRGAN scale factor {factor}"));
    }
    let (mode, weights_path) = match upscaler_id {
        "realesrgan-x4plus" => ("frames", model_path(REALESRGAN_X4_WEIGHTS)),
        "realesrgan-fast-x4v3" => ("frames-fast", model_path(REALESRGAN_FAST_X4_WEIGHTS)),
        _ => {
            return Err(format!(
                "unsupported Real-ESRGAN product runner '{upscaler_id}'"
            ));
        }
    };
    let stage_root = out_dir.join(upscaler_id);
    let input_root = stage_root.join("input");
    let output_root = stage_root.join("x4");
    std::fs::create_dir_all(&input_root)
        .and_then(|_| std::fs::create_dir_all(&output_root))
        .map_err(|error| format!("cannot create Real-ESRGAN frame directories: {error}"))?;

    let input_prefix = input_root.join("frame_");
    let output_prefix = output_root.join("frame_");
    let input_pattern = input_root.join("frame_%06d.png");
    let output_pattern = output_root.join("frame_%06d.png");
    progress(0, frames, "Extracting native frames for Real-ESRGAN");
    let extract = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &native_artifact.to_string_lossy(),
            "-vsync",
            "0",
            "-start_number",
            "0",
            "-frames:v",
            &frames.to_string(),
            &input_pattern.to_string_lossy(),
        ])
        .output()
        .map_err(|error| format!("cannot launch ffmpeg for post-upscale staging: {error}"))?;
    if !extract.status.success() {
        return Err(format!(
            "ffmpeg could not extract native LTX2 frames: {}",
            String::from_utf8_lossy(&extract.stderr).trim()
        ));
    }

    let log_path = stage_root.join("runner.log");
    let log = std::fs::File::create(&log_path)
        .map_err(|error| format!("cannot create Real-ESRGAN log: {error}"))?;
    let stderr = log
        .try_clone()
        .map_err(|error| format!("cannot clone Real-ESRGAN log: {error}"))?;
    progress(0, frames, &format!("Loading pure-Mojo {upscaler_id}"));
    let mut child = std::process::Command::new(repo_path(REALESRGAN_X4_RUNNER))
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .args([
            mode,
            &input_prefix.to_string_lossy(),
            &output_prefix.to_string_lossy(),
            &frames.to_string(),
            &weights_path.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(stderr))
        .spawn()
        .map_err(|error| format!("cannot launch pure-Mojo Real-ESRGAN: {error}"))?;
    let mut completed = 0i64;
    let status = loop {
        let observed = std::fs::read_dir(&output_root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.file_name().to_string_lossy().starts_with("frame_")
                    && entry.path().extension().and_then(|value| value.to_str()) == Some("png")
            })
            .count() as i64;
        if observed != completed {
            completed = observed;
            progress(
                completed,
                frames,
                &format!("{upscaler_id} post-upscale frame {completed} of {frames}"),
            );
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(500)),
            Err(error) => return Err(format!("cannot monitor pure-Mojo Real-ESRGAN: {error}")),
        }
    };
    if !status.success() {
        return Err(format!(
            "pure-Mojo Real-ESRGAN failed; inspect {}",
            log_path.display()
        ));
    }

    let target_width = width * factor;
    let target_height = height * factor;
    let artifact = out_dir.join(format!("ltx2_t2v_hq_{factor}x.mp4"));
    progress(frames, frames, "Muxing post-upscaled LTX2 video");
    let mut mux = std::process::Command::new("ffmpeg");
    mux.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        &fps.to_string(),
        "-start_number",
        "0",
        "-i",
        &output_pattern.to_string_lossy(),
        "-i",
        &native_artifact.to_string_lossy(),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
    ]);
    if factor == 2 {
        mux.args([
            "-vf",
            &format!("scale={target_width}:{target_height}:flags=lanczos"),
        ]);
    }
    let mux_output = mux
        .args([
            "-frames:v",
            &frames.to_string(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            &artifact.to_string_lossy(),
        ])
        .output()
        .map_err(|error| format!("cannot launch ffmpeg for post-upscale mux: {error}"))?;
    if !mux_output.status.success() {
        return Err(format!(
            "ffmpeg could not mux the post-upscaled LTX2 video: {}",
            String::from_utf8_lossy(&mux_output.stderr).trim()
        ));
    }
    let probe = probe_video_path(&artifact.to_string_lossy())?;
    if probe.get("muxing").and_then(Value::as_str) != Some("probe_ok")
        || probe.get("width").and_then(Value::as_i64) != Some(target_width)
        || probe.get("height").and_then(Value::as_i64) != Some(target_height)
        || probe.get("frame_count").and_then(Value::as_i64) != Some(frames)
    {
        return Err(format!(
            "post-upscaled LTX2 artifact failed geometry/frame probe: {probe}"
        ));
    }
    Ok((artifact, probe))
}

pub(super) fn start_ltx2_mojo_request(
    st: &AppState,
    body: &Value,
    gpu: crate::gpu_lock::GpuGuard,
) -> Response {
    let normalized_feature_body = match normalized_ltx2_feature_request(body) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    let normalized_body = match normalized_ltx2_video_edit_request(&normalized_feature_body) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    let body = &normalized_body;
    let selected_checkpoint = match resolve_ltx2_request_checkpoint(body) {
        Ok(checkpoint) => checkpoint,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    let edit_mode = body
        .get("video_edit_mode")
        .and_then(Value::as_str)
        .unwrap_or("standard");
    let checkpoint_profile = match body
        .get("checkpoint")
        .and_then(Value::as_str)
        .and_then(ltx2_checkpoint_profile)
    {
        Some(profile) => profile,
        None => {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "LTX2 request checkpoint is not registered",
            )
        }
    };
    let checkpoint_path = model_path(&checkpoint_profile.path);
    let checkpoint_id = checkpoint_profile.id.clone();
    let support_lora = checkpoint_profile.support_lora.clone();
    let profile = match ltx2_request_profile_for_mode(
        body.get("width").and_then(Value::as_i64).unwrap_or(0),
        body.get("height").and_then(Value::as_i64).unwrap_or(0),
        body.get("frames").and_then(Value::as_i64).unwrap_or(0),
        body.get("fps").and_then(Value::as_f64).unwrap_or(0.0),
        edit_mode,
    ) {
        Some(profile) if ltx2_profile_runner_available(&profile) => profile,
        Some(_profile) => {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "LTX2 runtime-geometry Mojo runner is unavailable: {}",
                    LTX2_MOJO_REQUEST_RUNNER
                ),
            );
        }
        None => {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "LTX2 request does not match an admitted native profile",
            );
        }
    };
    let request_runner = LTX2_MOJO_REQUEST_RUNNER.to_string();
    let quant = body
        .get("quant")
        .and_then(Value::as_str)
        .unwrap_or("fp8")
        .to_string();
    let include_audio = body
        .get("include_audio")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let audio_policy = body
        .get("audio_policy")
        .and_then(Value::as_str)
        .unwrap_or(if include_audio { "generate" } else { "none" })
        .to_string();
    let source_video = body
        .get("video_path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let feature_id = body
        .get("feature_id")
        .and_then(Value::as_str)
        .unwrap_or("standard")
        .to_string();
    let requested_post_upscale = match ltx2_requested_post_upscale(body) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    let int4_slab = (quant == "int4").then(|| model_path(LTX2_REFHQ_INT4_SLAB));
    if edit_mode != "standard" && quant != "bf16" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "LTX2 Retake/Extend follows the creator's complete BF16 checkpoint path; select BF16 precision",
        );
    }
    let distillation_adapter = match resolve_ltx2_distillation_adapter(body, &selected_checkpoint) {
        Ok(adapter) => adapter,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    let effective_checkpoint = if edit_mode != "standard" {
        match resolve_ltx2_retake_checkpoint(selected_checkpoint, distillation_adapter.is_some()) {
            Ok(checkpoint) => checkpoint,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        }
    } else {
        selected_checkpoint
    };
    let request_checkpoint_path = effective_checkpoint.path.clone();
    let effective_checkpoint_name = effective_checkpoint
        .name
        .strip_suffix(".safetensors")
        .unwrap_or(&effective_checkpoint.name)
        .to_string();
    let request_checkpoint_env = if quant == "bf16" {
        "LTX2_REFHQ_CKPT_BF16"
    } else {
        "LTX2_REFHQ_CKPT_FP8"
    };
    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    if let Err(error) = std::fs::create_dir_all(&out_dir) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot create LTX2 output directory: {error}"),
        );
    }
    let mut request_document = body.clone();
    if edit_mode == "standard" {
        if let Some(image_path) = body
            .get("image_path")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            let staged_path = match stage_ltx2_creator_i2v_source(
                image_path,
                &profile,
                &out_dir,
                "creator_i2v",
            ) {
                Ok(path) => path,
                Err(error) => {
                    return err_detail(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        &format!("LTX2 creator I2V preprocessing failed: {error}"),
                    );
                }
            };
            if let Some(request) = request_document.as_object_mut() {
                let image_strength = body
                    .get("image_strength")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                request.insert("creator_source_image_path".to_string(), json!(image_path));
                request.insert(
                    "image_path".to_string(),
                    json!(staged_path.to_string_lossy().to_string()),
                );
                request.insert("image_strength".to_string(), json!(image_strength));
                request.insert(
                    "creator_conditioning_width".to_string(),
                    json!(profile.conditioning_width),
                );
                request.insert(
                    "creator_conditioning_height".to_string(),
                    json!(profile.conditioning_height),
                );
                request.insert("creator_image_crf".to_string(), json!(33));
            }
        }
        if let Some(last_image_path) = body
            .get("last_image_path")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            let staged_path = match stage_ltx2_creator_i2v_source(
                last_image_path,
                &profile,
                &out_dir,
                "creator_last_frame",
            ) {
                Ok(path) => path,
                Err(error) => {
                    return err_detail(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        &format!("LTX2 creator last-frame preprocessing failed: {error}"),
                    );
                }
            };
            if let Some(request) = request_document.as_object_mut() {
                let last_image_strength = body
                    .get("last_image_strength")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                request.insert(
                    "creator_last_image_path".to_string(),
                    json!(last_image_path),
                );
                request.insert(
                    "last_image_path".to_string(),
                    json!(staged_path.to_string_lossy().to_string()),
                );
                request.insert(
                    "last_image_strength".to_string(),
                    json!(last_image_strength),
                );
                request.insert(
                    "creator_last_frame_index".to_string(),
                    json!(profile.frames - 1),
                );
                request.insert("creator_last_image_crf".to_string(), json!(33));
            }
        }
    }
    let request_path = out_dir.join("request.json");
    let request_bytes = match serde_json::to_vec_pretty(&request_document) {
        Ok(bytes) => bytes,
        Err(error) => {
            return err_detail(
                StatusCode::BAD_REQUEST,
                &format!("cannot serialize LTX2 request: {error}"),
            );
        }
    };
    if let Err(error) = std::fs::write(&request_path, request_bytes) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot write LTX2 request: {error}"),
        );
    }

    let bus = st.comfy_ws.clone();
    let prompt_id = video_id.clone();
    let thread_video_id = video_id.clone();
    let thread_out_dir = out_dir.clone();
    let thread_request_path = request_path.clone();
    let thread_out_root = st.out_dir.clone();
    let thread_request_runner = request_runner.clone();
    let thread_width = profile.width;
    let thread_height = profile.height;
    let thread_frames = profile.frames;
    let thread_fps = profile.fps;
    let thread_post_upscale = requested_post_upscale.clone();
    let thread_audio_policy = audio_policy.clone();
    let thread_source_video = source_video.clone();
    let thread_source_frames = body
        .get("video_source_frames")
        .and_then(Value::as_i64)
        .unwrap_or(profile.frames);
    let thread_feature_id = feature_id.clone();
    let thread_edit_mode = edit_mode.to_string();
    let thread_checkpoint_path = checkpoint_path.clone();
    let thread_checkpoint_id = checkpoint_id.clone();
    let thread_support_lora = support_lora.clone();
    let mut thread_request = request_document.clone();
    if let Some(request) = thread_request.as_object_mut() {
        // The public API accepts registered aliases, but the runner receives
        // one canonical identity paired with the exact server-resolved path.
        request.insert("checkpoint".to_string(), json!(checkpoint_id));
        // The denoiser always publishes exact final latents, exits, and lets a
        // fresh invocation of the same profile binary own VAE decode. This is
        // required for 720p+ on 24 GB and also prevents allocator fragmentation
        // from making lower-resolution behavior depend on prior jobs.
        request.insert("defer_decode".to_string(), json!(true));
        request.insert(
            "include_audio".to_string(),
            json!(thread_audio_policy == "generate" || thread_edit_mode != "standard"),
        );
        request.insert(
            "audio_policy".to_string(),
            json!(thread_audio_policy.clone()),
        );
        request.insert("checkpoint".to_string(), json!(effective_checkpoint_name));
        request.insert(
            "checkpoint_path".to_string(),
            json!(request_checkpoint_path.to_string_lossy()),
        );
        let overlay_loras = body
            .get("lora")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .filter(|row| row.get("role").and_then(Value::as_str) != Some("distillation"))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        request.insert("lora".to_string(), Value::Array(overlay_loras));
        if let Some(adapter) = distillation_adapter.as_ref() {
            let mut adapter_document = json!({
                "name": adapter.name,
                "path": adapter.path.to_string_lossy(),
                "weight": adapter.weight,
                "source": adapter.source,
            });
            if let Some(value) = adapter.stage1_weight {
                adapter_document
                    .as_object_mut()
                    .expect("adapter document is an object")
                    .insert("stage1_weight".to_string(), json!(value));
            }
            if let Some(value) = adapter.stage2_weight {
                adapter_document
                    .as_object_mut()
                    .expect("adapter document is an object")
                    .insert("stage2_weight".to_string(), json!(value));
            }
            request.insert("distillation_adapter".to_string(), adapter_document);
        } else {
            request.remove("distillation_adapter");
        }
    }
    std::thread::spawn(move || {
        let _gpu = gpu;
        let publish = |event: WorkerEvent| {
            let _ = bus.send((thread_video_id.clone(), event));
        };
        publish(WorkerEvent::Progress {
            step: 0,
            total: 0,
            phase: "Starting LTX2 Mojo runner".to_string(),
            preview: String::new(),
        });

        let authored_caps = thread_request
            .get("caps_positive")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if authored_caps.is_empty() {
            let prompt = thread_request
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let negative = thread_request
                .get("negative")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            match prepare_ltx2_mojo_context(
                &thread_out_root,
                &thread_out_dir,
                &prompt,
                &negative,
                &publish,
            ) {
                Ok(cache) => {
                    let path = cache.path.to_string_lossy().into_owned();
                    if let Some(request) = thread_request.as_object_mut() {
                        request.insert("caps_positive".to_string(), json!(path));
                        request.insert("caps_negative".to_string(), json!(path));
                        request.insert(
                            "conditioning_cache".to_string(),
                            json!({
                                "backend": "mojo",
                                "key": cache.key,
                                "hit": cache.hit,
                                "encoder_seconds": cache.encoder_seconds,
                                "manifest": cache.manifest_path.to_string_lossy(),
                            }),
                        );
                    }
                }
                Err(error) => {
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "failed",
                        "conditioning",
                        0,
                        48,
                        &error,
                    );
                    publish(WorkerEvent::Failed { error });
                    return;
                }
            }
        } else {
            let message = "Using authored LTX2 prompt conditioning";
            let _ =
                write_ltx2_job_status(&thread_out_dir, "running", "conditioning", 0, 0, message);
            publish(WorkerEvent::Progress {
                step: 0,
                total: 0,
                phase: message.to_string(),
                preview: String::new(),
            });
        }
        if thread_edit_mode != "standard" {
            let message = "Decoding source audio with the LTX Desktop PyAV contract";
            let _ = write_ltx2_job_status(
                &thread_out_dir,
                "running",
                "encoding_source_audio",
                0,
                0,
                message,
            );
            publish(WorkerEvent::Progress {
                step: 0,
                total: 0,
                phase: message.to_string(),
                preview: String::new(),
            });
            match stage_ltx2_creator_source_audio(
                std::path::Path::new(&thread_source_video),
                &thread_out_dir,
                thread_source_frames,
                thread_fps as f64,
            ) {
                Ok(Some(audio)) => {
                    if let Some(request) = thread_request.as_object_mut() {
                        request.insert(
                            "source_audio_path".to_string(),
                            audio.get("path").cloned().unwrap_or(Value::Null),
                        );
                        request.insert(
                            "source_audio_sample_rate".to_string(),
                            audio.get("sample_rate").cloned().unwrap_or(Value::Null),
                        );
                        request.insert(
                            "source_audio_channels".to_string(),
                            audio.get("channels").cloned().unwrap_or(Value::Null),
                        );
                        request.insert(
                            "source_audio_samples".to_string(),
                            audio
                                .get("samples_per_channel")
                                .cloned()
                                .unwrap_or(Value::Null),
                        );
                        request.insert("source_audio_decode".to_string(), audio);
                    }
                }
                Ok(None) => {
                    if let Some(request) = thread_request.as_object_mut() {
                        request.insert("source_audio_path".to_string(), json!(""));
                        request.insert("source_audio_sample_rate".to_string(), json!(0));
                        request.insert("source_audio_channels".to_string(), json!(0));
                        request.insert("source_audio_samples".to_string(), json!(0));
                    }
                }
                Err(error) => {
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "failed",
                        "encoding_source_audio",
                        0,
                        0,
                        &error,
                    );
                    publish(WorkerEvent::Failed { error });
                    return;
                }
            }
        }
        let request_bytes = match serde_json::to_vec_pretty(&thread_request) {
            Ok(bytes) => bytes,
            Err(error) => {
                let error = format!("cannot serialize resolved LTX2 request: {error}");
                let _ =
                    write_ltx2_job_status(&thread_out_dir, "failed", "conditioning", 0, 0, &error);
                publish(WorkerEvent::Failed { error });
                return;
            }
        };
        if let Err(error) = std::fs::write(&thread_request_path, request_bytes) {
            let error = format!("cannot write resolved LTX2 request: {error}");
            let _ = write_ltx2_job_status(&thread_out_dir, "failed", "conditioning", 0, 0, &error);
            publish(WorkerEvent::Failed { error });
            return;
        }

        let log_path = thread_out_dir.join("runner.log");
        let log = match std::fs::File::create(&log_path) {
            Ok(file) => file,
            Err(error) => {
                let error = format!("cannot create LTX2 runner log: {error}");
                let _ =
                    write_ltx2_job_status(&thread_out_dir, "failed", "runner_start", 0, 0, &error);
                publish(WorkerEvent::Failed { error });
                return;
            }
        };
        let stderr = match log.try_clone() {
            Ok(file) => file,
            Err(error) => {
                let error = format!("cannot clone LTX2 runner log handle: {error}");
                let _ =
                    write_ltx2_job_status(&thread_out_dir, "failed", "runner_start", 0, 0, &error);
                publish(WorkerEvent::Failed { error });
                return;
            }
        };
        let mut command = std::process::Command::new(repo_path(&thread_request_runner));
        let request_ld_path = ltx2_request_ld_path(&thread_edit_mode);
        let ram_cache_ready = std::fs::create_dir_all(LTX2_CUDA_CACHE).is_ok();
        command
            .current_dir(repo_root())
            .env("LD_LIBRARY_PATH", request_ld_path)
            .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
            .env("LTX2_REFHQ_CKPT_FP8", &thread_checkpoint_path)
            .env("LTX2_REFHQ_CHECKPOINT_ID", &thread_checkpoint_id)
            .env("LTX2_REFHQ_SUPPORT_LORA", &thread_support_lora)
            .arg(&thread_request_path)
            .arg(&thread_out_dir)
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(stderr));
        if !ram_cache_ready {
            command.env("CUDA_CACHE_DISABLE", "1");
        }
        if let Some(path) = int4_slab {
            command.env("LTX2_INT4_SLAB", path);
        }
        command.env(request_checkpoint_env, &request_checkpoint_path);
        if thread_edit_mode != "standard" {
            command.env("LTX2_AUDIO_VAE_CKPT", &request_checkpoint_path);
            command.env("LTX2_VIDEO_VAE_CKPT", &request_checkpoint_path);
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                let error = format!("cannot start LTX2 Mojo runner: {error}");
                let _ =
                    write_ltx2_job_status(&thread_out_dir, "failed", "runner_start", 0, 0, &error);
                publish(WorkerEvent::Failed { error });
                return;
            }
        };

        let status_path = thread_out_dir.join("status.json");
        let mut last_status = String::new();
        let exit_status = loop {
            if let Ok(text) = std::fs::read_to_string(&status_path) {
                if text != last_status {
                    if let Ok(status) = serde_json::from_str::<Value>(&text) {
                        let step = status.get("step").and_then(Value::as_i64).unwrap_or(0);
                        let total = status.get("total").and_then(Value::as_i64).unwrap_or(0);
                        let message = status
                            .get("message")
                            .and_then(Value::as_str)
                            .or_else(|| status.get("phase").and_then(Value::as_str))
                            .unwrap_or("LTX2 running")
                            .to_string();
                        publish(WorkerEvent::Progress {
                            step,
                            total,
                            phase: message,
                            preview: String::new(),
                        });
                    }
                    last_status = text;
                }
            }
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(250)),
                Err(error) => {
                    let error = format!("cannot monitor LTX2 Mojo runner: {error}");
                    let _ =
                        write_ltx2_job_status(&thread_out_dir, "failed", "runner", 0, 0, &error);
                    publish(WorkerEvent::Failed { error });
                    break None;
                }
            }
        };
        let mut final_exit_status = exit_status;
        let decode_handoff = thread_out_dir.join("decode_handoff.json");
        let final_latents = thread_out_dir.join("final_latents.safetensors");
        if final_exit_status
            .as_ref()
            .is_some_and(std::process::ExitStatus::success)
            && decode_handoff.is_file()
            && final_latents.is_file()
        {
            publish(WorkerEvent::Progress {
                step: 0,
                total: 0,
                phase: "Releasing denoiser and starting fresh LTX2 decode".to_string(),
                preview: String::new(),
            });
            let decode_log_path = thread_out_dir.join("decode.log");
            let decode_log = match std::fs::File::create(&decode_log_path) {
                Ok(file) => file,
                Err(error) => {
                    let error = format!("cannot create LTX2 decode log: {error}");
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "failed",
                        "decode_start",
                        0,
                        0,
                        &error,
                    );
                    publish(WorkerEvent::Failed { error });
                    return;
                }
            };
            let decode_stderr = match decode_log.try_clone() {
                Ok(file) => file,
                Err(error) => {
                    let error = format!("cannot clone LTX2 decode log handle: {error}");
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "failed",
                        "decode_start",
                        0,
                        0,
                        &error,
                    );
                    publish(WorkerEvent::Failed { error });
                    return;
                }
            };
            let mut decode = std::process::Command::new(repo_path(&thread_request_runner));
            decode
                .current_dir(repo_root())
                .env("LD_LIBRARY_PATH", ltx2_decode_ld_path())
                .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
                .env("LTX2_REFHQ_CKPT_FP8", &thread_checkpoint_path)
                .env("LTX2_REFHQ_CHECKPOINT_ID", &thread_checkpoint_id)
                .env("LTX2_REFHQ_SUPPORT_LORA", &thread_support_lora)
                .arg("decode")
                .arg(&thread_request_path)
                .arg(&thread_out_dir)
                .stdout(std::process::Stdio::from(decode_log))
                .stderr(std::process::Stdio::from(decode_stderr));
            if !ram_cache_ready {
                decode.env("CUDA_CACHE_DISABLE", "1");
            }
            let mut decode_child = match decode.spawn() {
                Ok(child) => child,
                Err(error) => {
                    let error = format!("cannot start fresh LTX2 Mojo decode: {error}");
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "failed",
                        "decode_start",
                        0,
                        0,
                        &error,
                    );
                    publish(WorkerEvent::Failed { error });
                    return;
                }
            };
            final_exit_status = loop {
                if let Ok(text) = std::fs::read_to_string(&status_path) {
                    if text != last_status {
                        if let Ok(status) = serde_json::from_str::<Value>(&text) {
                            let step = status.get("step").and_then(Value::as_i64).unwrap_or(0);
                            let total = status.get("total").and_then(Value::as_i64).unwrap_or(0);
                            let message = status
                                .get("message")
                                .and_then(Value::as_str)
                                .or_else(|| status.get("phase").and_then(Value::as_str))
                                .unwrap_or("LTX2 fresh decode running")
                                .to_string();
                            publish(WorkerEvent::Progress {
                                step,
                                total,
                                phase: message,
                                preview: String::new(),
                            });
                        }
                        last_status = text;
                    }
                }
                match decode_child.try_wait() {
                    Ok(Some(status)) => break Some(status),
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(250)),
                    Err(error) => {
                        let error = format!("cannot monitor fresh LTX2 Mojo decode: {error}");
                        let _ = write_ltx2_job_status(
                            &thread_out_dir,
                            "failed",
                            "decode",
                            0,
                            0,
                            &error,
                        );
                        publish(WorkerEvent::Failed { error });
                        break None;
                    }
                }
            };
        }

        let result_path = thread_out_dir.join("result.json");
        let mut result = std::fs::read_to_string(&result_path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        let succeeded = final_exit_status
            .as_ref()
            .map(std::process::ExitStatus::success)
            .unwrap_or(false)
            && result
                .as_ref()
                .and_then(|doc| doc.get("state"))
                .and_then(Value::as_str)
                == Some("done");
        if succeeded {
            let authored = result
                .as_ref()
                .and_then(|doc| doc.get("artifact_path"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let mut artifact = if std::path::Path::new(authored).is_absolute() {
                std::path::PathBuf::from(authored)
            } else {
                repo_root().join(authored)
            };
            if artifact.is_file() {
                if thread_feature_id == "foley-v2a" {
                    let message = "Assembling generated Foley audio with the exact source video";
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "running",
                        "assembling_foley_audio",
                        thread_frames,
                        thread_frames,
                        message,
                    );
                    publish(WorkerEvent::Progress {
                        step: thread_frames,
                        total: thread_frames,
                        phase: message.to_string(),
                        preview: String::new(),
                    });
                    match remux_ltx2_generated_audio(
                        std::path::Path::new(&thread_source_video),
                        &artifact,
                        &thread_out_dir,
                        thread_frames,
                    ) {
                        Ok((foley_artifact, probe)) => {
                            if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                                doc.insert(
                                    "native_generated_av_artifact_path".to_string(),
                                    json!(artifact.to_string_lossy()),
                                );
                                doc.insert(
                                    "artifact_path".to_string(),
                                    json!(foley_artifact.to_string_lossy()),
                                );
                                doc.insert(
                                    "mp4_url".to_string(),
                                    json!(format!(
                                        "/out/{}/{}",
                                        thread_video_id,
                                        foley_artifact
                                            .file_name()
                                            .and_then(|value| value.to_str())
                                            .unwrap_or("ltx2_foley_v2a.mp4")
                                    )),
                                );
                                doc.insert("feature_id".to_string(), json!("foley-v2a"));
                                doc.insert("audio_policy".to_string(), json!("generate"));
                                doc.insert("audio_probe".to_string(), probe);
                            }
                            artifact = foley_artifact;
                            if let Some(doc) = result.as_ref() {
                                if let Ok(bytes) = serde_json::to_vec_pretty(doc) {
                                    let _ = std::fs::write(&result_path, bytes);
                                }
                            }
                        }
                        Err(error) => {
                            if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                                doc.insert("state".to_string(), json!("failed"));
                                doc.insert(
                                    "failed_step".to_string(),
                                    json!("assembling_foley_audio"),
                                );
                                doc.insert("error".to_string(), json!(error));
                            }
                            if let Some(doc) = result.as_ref() {
                                if let Ok(bytes) = serde_json::to_vec_pretty(doc) {
                                    let _ = std::fs::write(&result_path, bytes);
                                }
                            }
                            let error = result
                                .as_ref()
                                .and_then(|doc| doc.get("error"))
                                .and_then(Value::as_str)
                                .unwrap_or("LTX2 Foley assembly failed")
                                .to_string();
                            let _ = write_ltx2_job_status(
                                &thread_out_dir,
                                "failed",
                                "assembling_foley_audio",
                                thread_frames,
                                thread_frames,
                                &error,
                            );
                            publish(WorkerEvent::Failed { error });
                            return;
                        }
                    }
                } else if thread_audio_policy == "preserve" && thread_edit_mode == "standard" {
                    let message = "Preserving audio from the V2V source";
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "running",
                        "preserving_source_audio",
                        thread_frames,
                        thread_frames,
                        message,
                    );
                    publish(WorkerEvent::Progress {
                        step: thread_frames,
                        total: thread_frames,
                        phase: message.to_string(),
                        preview: String::new(),
                    });
                    match remux_ltx2_source_audio(
                        &artifact,
                        std::path::Path::new(&thread_source_video),
                        &thread_out_dir,
                        thread_frames,
                    ) {
                        Ok((source_audio_artifact, probe)) => {
                            if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                                doc.insert(
                                    "native_video_only_artifact_path".to_string(),
                                    json!(artifact.to_string_lossy()),
                                );
                                doc.insert(
                                    "artifact_path".to_string(),
                                    json!(source_audio_artifact.to_string_lossy()),
                                );
                                doc.insert(
                                    "mp4_url".to_string(),
                                    json!(format!(
                                        "/out/{}/{}",
                                        thread_video_id,
                                        source_audio_artifact
                                            .file_name()
                                            .and_then(|value| value.to_str())
                                            .unwrap_or("ltx2_source_audio.mp4")
                                    )),
                                );
                                doc.insert("audio_policy".to_string(), json!("preserve"));
                                doc.insert("audio_probe".to_string(), probe);
                            }
                            artifact = source_audio_artifact;
                            if let Some(doc) = result.as_ref() {
                                if let Ok(bytes) = serde_json::to_vec_pretty(doc) {
                                    let _ = std::fs::write(&result_path, bytes);
                                }
                            }
                        }
                        Err(error) => {
                            if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                                doc.insert("state".to_string(), json!("failed"));
                                doc.insert(
                                    "failed_step".to_string(),
                                    json!("preserving_source_audio"),
                                );
                                doc.insert("error".to_string(), json!(error));
                            }
                            if let Some(doc) = result.as_ref() {
                                if let Ok(bytes) = serde_json::to_vec_pretty(doc) {
                                    let _ = std::fs::write(&result_path, bytes);
                                }
                            }
                            let error = result
                                .as_ref()
                                .and_then(|doc| doc.get("error"))
                                .and_then(Value::as_str)
                                .unwrap_or("LTX2 source-audio preservation failed")
                                .to_string();
                            let _ = write_ltx2_job_status(
                                &thread_out_dir,
                                "failed",
                                "preserving_source_audio",
                                thread_frames,
                                thread_frames,
                                &error,
                            );
                            publish(WorkerEvent::Failed { error });
                            return;
                        }
                    }
                } else if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                    doc.insert(
                        "audio_policy".to_string(),
                        json!(thread_audio_policy.clone()),
                    );
                    if thread_feature_id != "standard" {
                        doc.insert("feature_id".to_string(), json!(thread_feature_id.clone()));
                    }
                    if let Ok(bytes) = serde_json::to_vec_pretty(&*doc) {
                        let _ = std::fs::write(&result_path, bytes);
                    }
                }
                if let Some((upscaler, factor)) = thread_post_upscale.as_ref() {
                    let message = format!("Starting {upscaler} {factor}x post-upscale");
                    let _ = write_ltx2_job_status(
                        &thread_out_dir,
                        "running",
                        "post_upscale",
                        0,
                        thread_frames,
                        &message,
                    );
                    publish(WorkerEvent::Progress {
                        step: 0,
                        total: thread_frames,
                        phase: message,
                        preview: String::new(),
                    });
                    let started = std::time::Instant::now();
                    match run_realesrgan_video_post_upscale(
                        upscaler,
                        &artifact,
                        &thread_out_dir,
                        thread_width,
                        thread_height,
                        thread_frames,
                        thread_fps,
                        *factor,
                        |step, total, message| {
                            let _ = write_ltx2_job_status(
                                &thread_out_dir,
                                "running",
                                "post_upscale",
                                step,
                                total,
                                message,
                            );
                            publish(WorkerEvent::Progress {
                                step,
                                total,
                                phase: message.to_string(),
                                preview: String::new(),
                            });
                        },
                    ) {
                        Ok((upscaled_artifact, probe)) => {
                            if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                                doc.insert(
                                    "native_artifact_path".to_string(),
                                    json!(artifact.to_string_lossy()),
                                );
                                doc.insert(
                                    "artifact_path".to_string(),
                                    json!(upscaled_artifact.to_string_lossy()),
                                );
                                doc.insert(
                                    "mp4_url".to_string(),
                                    json!(format!(
                                        "/out/{}/{}",
                                        thread_video_id,
                                        upscaled_artifact
                                            .file_name()
                                            .and_then(|value| value.to_str())
                                            .unwrap_or("ltx2_post_upscale.mp4")
                                    )),
                                );
                                doc.insert("width".to_string(), json!(thread_width * *factor));
                                doc.insert("height".to_string(), json!(thread_height * *factor));
                                doc.insert(
                                    "post_upscale".to_string(),
                                    json!({
                                        "id": upscaler,
                                        "factor": factor,
                                        "backend": "mojo",
                                        "runner": REALESRGAN_X4_RUNNER,
                                        "seconds": started.elapsed().as_secs_f64(),
                                        "probe": probe,
                                    }),
                                );
                            }
                            if let Some(doc) = result.as_ref() {
                                if let Ok(bytes) = serde_json::to_vec_pretty(doc) {
                                    let _ = std::fs::write(&result_path, bytes);
                                }
                            }
                            let done_message =
                                format!("{upscaler} {factor}x post-upscale complete");
                            let _ = write_ltx2_job_status(
                                &thread_out_dir,
                                "done",
                                "done",
                                thread_frames,
                                thread_frames,
                                &done_message,
                            );
                            publish(WorkerEvent::Done {
                                output_path: upscaled_artifact.to_string_lossy().into_owned(),
                            });
                            return;
                        }
                        Err(error) => {
                            if let Some(doc) = result.as_mut().and_then(Value::as_object_mut) {
                                doc.insert("state".to_string(), json!("failed"));
                                doc.insert("failed_step".to_string(), json!("post_upscale"));
                                doc.insert("error".to_string(), json!(error));
                            }
                            if let Some(doc) = result.as_ref() {
                                if let Ok(bytes) = serde_json::to_vec_pretty(doc) {
                                    let _ = std::fs::write(&result_path, bytes);
                                }
                            }
                            let error = result
                                .as_ref()
                                .and_then(|doc| doc.get("error"))
                                .and_then(Value::as_str)
                                .unwrap_or("LTX2 post-upscale failed")
                                .to_string();
                            let _ = write_ltx2_job_status(
                                &thread_out_dir,
                                "failed",
                                "post_upscale",
                                0,
                                thread_frames,
                                &error,
                            );
                            publish(WorkerEvent::Failed { error });
                            return;
                        }
                    }
                }
                let done_message = if thread_feature_id == "foley-v2a" {
                    "Foley audio and source video ready"
                } else if thread_audio_policy == "preserve" {
                    "Video and preserved source audio ready"
                } else if thread_audio_policy == "generate" {
                    "Video and generated audio ready"
                } else {
                    "Video ready"
                };
                let _ = write_ltx2_job_status(
                    &thread_out_dir,
                    "done",
                    "done",
                    thread_frames,
                    thread_frames,
                    done_message,
                );
                publish(WorkerEvent::Done {
                    output_path: artifact.to_string_lossy().into_owned(),
                });
                return;
            }
        }
        let status_error = std::fs::read_to_string(&status_path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|doc| {
                doc.get("message")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
        let error = status_error.unwrap_or_else(|| {
            format!(
                "LTX2 Mojo runner failed; inspect {}",
                log_path.to_string_lossy()
            )
        });
        let _ = write_ltx2_job_status(&thread_out_dir, "failed", "runner", 0, 0, &error);
        publish(WorkerEvent::Failed { error });
    });

    json_resp(
        StatusCode::ACCEPTED,
        &json!({
            "schema": "serenity.video_job.v1",
            "video_id": video_id,
            "prompt_id": prompt_id,
            "model": "ltx2",
            "runner": "ltx2_mojo_request",
            "profile_runner": request_runner,
            "profile": ltx2_profile_document(&profile),
            "backend": "mojo",
            "quant": quant,
            "audio_policy": audio_policy,
            "feature_id": feature_id,
            "state": "queued",
            "status_url": format!("/out/{video_id}/status.json"),
            "result_url": format!("/out/{video_id}/result.json"),
            "request_url": format!("/out/{video_id}/request.json"),
        }),
    )
}

/// LTX2 staged-smoke arm (behavior intact from the original handler; adds the
/// additive `quant` knob — see below). Requires the runner to be built.
pub(super) fn post_video_ltx2(st: &AppState, b: &Value) -> Response {
    let s = |k: &str, d: &str| b.get(k).and_then(|v| v.as_str()).unwrap_or(d).to_string();
    let runner = s("runner", "ltx2_staged_dev_smoke");
    if runner == "ltx2_refhq" {
        return post_video_ltx2_refhq(st, b);
    }
    if runner != "ltx2_staged_dev_smoke" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "unsupported video runner '{runner}'; use ltx2_staged_dev_smoke or ltx2_refhq"
            ),
        );
    }
    // steps=0 = the FULL stage-1 schedule (HQ_STEPS) INCLUDING the final
    // terminal-sigma denoise — the runner skips that final denoise whenever
    // the schedule is capped (n1 != num_steps), which leaves a half-denoised
    // garble (measured video-0089 at steps=15 vs HQ_STEPS=20).
    let steps = b.get("steps").and_then(|v| v.as_i64()).unwrap_or(0);
    if !(0..=20).contains(&steps) {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'steps' out of range [0..20]; 0 = full schedule (recommended)",
        );
    }
    let weight_mode = s("weight_mode", "resident");
    if weight_mode != "resident" && weight_mode != "stream" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("unsupported weight_mode '{weight_mode}'; use resident or stream"),
        );
    }
    let mut audio_mode = s("audio_mode", "noaudio");
    if audio_mode == "video" {
        audio_mode = "noaudio".to_string();
    }
    if audio_mode != "audio" && audio_mode != "noaudio" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("unsupported audio_mode '{audio_mode}'; use audio or noaudio"),
        );
    }
    // ── quant knob (LTX2). The runner's base is distilled-fp8 (CKPT_FP8); the only
    //    executable alternative is int4 W4A16-resident, selected via LTX2_INT4_SLAB
    //    (+ resident weight_mode). W4A4 int4-COMPUTE is NOT integrated. ──
    let quant = s("quant", "bf16");
    if !matches!(quant.as_str(), "bf16" | "fp8" | "int4") {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("unsupported quant '{quant}'; use bf16, fp8, or int4"),
        );
    }
    let mut int4_env: Option<(&str, String)> = None;
    let quant_note: String = match quant.as_str() {
        "int4" => {
            if weight_mode != "resident" {
                return err_detail(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "quant int4 requires weight_mode=resident (int4-resident W4A16); streamed int4 is not wired",
                );
            }
            let int4_slab = model_path(LTX2_INT4_SLAB);
            if !int4_slab.exists() {
                return err_detail(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &format!("quant int4 requested but the svdint4 slab is missing: {}", int4_slab.display()),
                );
            }
            int4_env = Some(("LTX2_INT4_SLAB", int4_slab.to_string_lossy().into_owned()));
            "int4 = W4A16 int4-resident (svdint4 slab, dequant4+low-rank -> bf16 reconstruct on-use); W4A4 int4-compute is NOT integrated".to_string()
        }
        "bf16" => "LTX2 distilled base is fp8; 'bf16' maps to the native fp8-resident path (no separate bf16 weights)".to_string(),
        _ => String::new(), // fp8: native default, no note
    };
    if !runner_available() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("missing executable {RUNNER}; run `pixi run build-video-smoke` first"),
        );
    }
    // Runner present: run the bounded staged smoke (timing is non-deterministic, so
    // this path is not byte-verifiable). video_id from the shared counter.
    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
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
    let mut cmd = std::process::Command::new(repo_path(RUNNER));
    cmd.current_dir(repo_root());
    // Clean contract (mesh root-cause 2026-07-16, MAP.md): stage-1 IS the
    // product — `s1out` skips the corrupting spatial-upsampler leg, and `base`
    // skips fusing the distilled LoRA onto the already-distilled checkpoint.
    cmd.args([
        "staged",
        "base",
        &weight_mode,
        &audio_mode,
        "nonag",
        &out_dir.to_string_lossy(),
        &steps.to_string(),
        "s1out",
    ]);
    if let Some((k, v)) = int4_env {
        cmd.env(k, v);
    }
    let rc = cmd
        .output()
        .map(|o| o.status.code().unwrap_or(-1))
        .unwrap_or(-1);
    let wall = t0.elapsed().as_secs_f64();
    let mut o = json!({
        "schema": "serenity.video_result.v1", "video_id": video_id, "runner": runner,
        "backend": BACKEND_NAME, "control_plane": "serenity-server", "model": "ltx2", "resident": "",
        "readiness_label": "bounded_daemon_smoke",
        "accepted_video_artifact": false, "accepted_av_artifact": false,
        "accepted_video_parity": false, "accepted_sampler_parity": false,
        "steps": steps, "mode": format!("staged lora {weight_mode} {audio_mode} nonag"),
        "weight_mode": weight_mode, "audio_mode": audio_mode, "quant": quant, "quant_note": quant_note,
        "exit_code": rc,
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
            m.insert(
                "error".into(),
                json!("LTX2 staged smoke runner failed; inspect log_path"),
            );
        }
        return json_resp(StatusCode::INTERNAL_SERVER_ERROR, &o);
    }
    json_resp(StatusCode::OK, &o)
}

/// LTX2 ref-HQ product arm. This extends the existing refhq runner rather than
/// introducing another inference backend: phase 1 writes the clean stage-1
/// latents and exits, then phase 2 starts a fresh Mojo process for the Desktop-
/// tiled VAE/audio/mux tail. On a prompt cache miss, the existing Creator-backed
/// CPU producer runs and exits before the 22B Mojo video process starts.
pub(super) fn post_video_ltx2_refhq(st: &AppState, b: &Value) -> Response {
    let checkpoint = b
        .get("checkpoint")
        .and_then(Value::as_str)
        .unwrap_or(LTX2_REFHQ_CHECKPOINT);
    if checkpoint != LTX2_REFHQ_CHECKPOINT {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "ltx2_refhq executes {LTX2_REFHQ_CHECKPOINT}; requested checkpoint '{checkpoint}' is not admitted"
            ),
        );
    }
    let explicit_contexts = b
        .get("contexts_path")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| std::env::var("LTX2_CTX_DUMP").ok())
        .unwrap_or_default();
    let prompt = b
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let negative_prompt = b
        .get("negative_prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let noises = b
        .get("noises_path")
        .and_then(Value::as_str)
        .unwrap_or("-")
        .to_string();
    if noises != "-" && !std::path::Path::new(&noises).is_file() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("ltx2_refhq noise fixture not found: {noises}"),
        );
    }
    let steps = b.get("steps").and_then(Value::as_i64).unwrap_or(15);
    if steps != 15 {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "ltx2_refhq is Creator-gated at exactly 15 steps",
        );
    }
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };
    if !runner_available() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("missing executable {RUNNER}; run `pixi run build-video-smoke` first"),
        );
    }
    if !ltx2_decode_runtime_available() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "ltx2_refhq Creator audio decode runtime missing: {}/libcudnn.so.9",
                ltx2_decode_cudnn_lib().display()
            ),
        );
    }
    let context_cache = if explicit_contexts.is_empty() {
        if prompt.is_empty() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "ltx2_refhq requires 'prompt' when no contexts_path is supplied",
            );
        }
        match prepare_ltx2_refhq_context(&st.out_dir, &prompt, &negative_prompt) {
            Ok(cache) => cache,
            Err(e) => return err_detail(StatusCode::INTERNAL_SERVER_ERROR, &e),
        }
    } else {
        let path = std::path::PathBuf::from(&explicit_contexts);
        if !ltx2_context_cache_valid(&path) {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "ltx2_refhq conditioning cache is missing or does not match the pinned Creator BF16 contract: {explicit_contexts}"
                ),
            );
        }
        Ltx2ContextCache {
            path,
            key: String::new(),
            hit: true,
            encoder_seconds: 0.0,
            log_path: std::path::PathBuf::new(),
            manifest_path: std::path::PathBuf::new(),
        }
    };
    let requested_contexts = context_cache.path.to_string_lossy().into_owned();

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot create LTX2 output directory: {e}"),
        );
    }
    let stage1_log = out_dir.join("ltx2_refhq_stage1.log");
    let upscale_log = out_dir.join("ltx2_refhq_upscale.log");
    let stage2_log = out_dir.join("ltx2_refhq_stage2.log");
    let decode_log = out_dir.join("ltx2_refhq_decode.log");
    let stage1_cache = out_dir.join("stage1_final.safetensors");
    let upscaler_cache = out_dir.join("upsampler.safetensors");
    let final_latents = out_dir.join("final_latents.safetensors");
    let mp4 = out_dir.join("ltx2_refhq.mp4");
    let wav = out_dir.join("refhq_audio.wav");
    let t0 = std::time::Instant::now();
    let seed_arg = seed.to_string();

    let mut stage1 = std::process::Command::new(repo_path(RUNNER));
    stage1
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .args([
            "refhq",
            &requested_contexts,
            &noises,
            &out_dir.to_string_lossy(),
            "15",
            "seed",
            &seed_arg,
            "lora",
            "full",
            "stage1-only",
        ]);
    let (stage1_rc, stage1_peak_vram_mib) = match run_logged_with_gpu_peak(&mut stage1, &stage1_log)
    {
        Ok(result) => result,
        Err(e) => {
            let _ = std::fs::write(&stage1_log, &e);
            (-1, None)
        }
    };
    let stage1_seconds = t0.elapsed().as_secs_f64();
    if stage1_rc != 0 || !stage1_cache_is_bf16(&stage1_cache) {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id,
                "model": "ltx2", "runner": "ltx2_refhq", "state": "failed",
                "failed_step": "stage1", "stage1_exit_code": stage1_rc,
                "seed": seed,
                "conditioning_cache": requested_contexts,
                "conditioning_cache_key": context_cache.key,
                "conditioning_cache_hit": context_cache.hit,
                "conditioning_encoder_seconds": context_cache.encoder_seconds,
                "conditioning_log": context_cache.log_path.to_string_lossy(),
                "conditioning_manifest": context_cache.manifest_path.to_string_lossy(),
                "out_dir": out_dir.to_string_lossy(),
                "stage1_log": stage1_log.to_string_lossy(),
                "stage1_seconds": stage1_seconds,
                "stage1_peak_vram_mib": stage1_peak_vram_mib,
                "stage1_cache": stage1_cache.to_string_lossy(),
                "error": "LTX2 refhq stage 1 failed or did not write the BF16 stage1_final cache",
            }),
        );
    }

    let tu = std::time::Instant::now();
    let mut upscale = std::process::Command::new(repo_path(RUNNER));
    upscale
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .args([
            "refhq-upscale",
            &stage1_cache.to_string_lossy(),
            &out_dir.to_string_lossy(),
        ]);
    let (upscale_rc, upscale_peak_vram_mib) =
        match run_logged_with_gpu_peak(&mut upscale, &upscale_log) {
            Ok(result) => result,
            Err(e) => {
                let _ = std::fs::write(&upscale_log, &e);
                (-1, None)
            }
        };
    let upscale_seconds = tu.elapsed().as_secs_f64();
    if upscale_rc != 0 || !upscaler_cache_is_f32(&upscaler_cache) {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id,
                "model": "ltx2", "runner": "ltx2_refhq", "state": "failed",
                "failed_step": "upscaler", "upscale_exit_code": upscale_rc,
                "seed": seed, "out_dir": out_dir.to_string_lossy(),
                "stage1_cache": stage1_cache.to_string_lossy(),
                "upscaler_cache": upscaler_cache.to_string_lossy(),
                "upscale_log": upscale_log.to_string_lossy(),
                "upscale_seconds": upscale_seconds,
                "upscale_peak_vram_mib": upscale_peak_vram_mib,
                "error": "LTX2 refhq upscaler failed or did not write the F32 official x2 cache",
            }),
        );
    }

    let ts2 = std::time::Instant::now();
    let mut stage2 = std::process::Command::new(repo_path(RUNNER));
    stage2
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .args([
            "refhq",
            &requested_contexts,
            &noises,
            &out_dir.to_string_lossy(),
            "15",
            "seed",
            &seed_arg,
            "lora",
            "full",
            "denoise-only",
            "stage2-only",
            "stage1-cache",
            &stage1_cache.to_string_lossy(),
            "upscaler-cache",
            &upscaler_cache.to_string_lossy(),
        ]);
    let (stage2_rc, stage2_peak_vram_mib) = match run_logged_with_gpu_peak(&mut stage2, &stage2_log)
    {
        Ok(result) => result,
        Err(e) => {
            let _ = std::fs::write(&stage2_log, &e);
            (-1, None)
        }
    };
    let stage2_seconds = ts2.elapsed().as_secs_f64();
    let final_latents_bf16 = final_latents_are_bf16(&final_latents);
    if stage2_rc != 0 || !final_latents_bf16 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id,
                "model": "ltx2", "runner": "ltx2_refhq", "state": "failed",
                "failed_step": "stage2", "stage2_exit_code": stage2_rc,
                "seed": seed, "out_dir": out_dir.to_string_lossy(),
                "stage1_cache": stage1_cache.to_string_lossy(),
                "upscaler_cache": upscaler_cache.to_string_lossy(),
                "final_latents": final_latents.to_string_lossy(),
                "stage2_log": stage2_log.to_string_lossy(),
                "stage2_seconds": stage2_seconds,
                "stage2_peak_vram_mib": stage2_peak_vram_mib,
                "error": "LTX2 refhq stage 2 failed or did not write the BF16 full-resolution final latent cache",
            }),
        );
    }

    let td = std::time::Instant::now();
    let mut decode = std::process::Command::new(repo_path(RUNNER));
    decode
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", ltx2_decode_ld_path())
        .args([
            "refhq-decode",
            &final_latents.to_string_lossy(),
            &out_dir.to_string_lossy(),
        ]);
    let (decode_rc, decode_peak_vram_mib) = match run_logged_with_gpu_peak(&mut decode, &decode_log)
    {
        Ok(result) => result,
        Err(e) => {
            let _ = std::fs::write(&decode_log, &e);
            (-1, None)
        }
    };
    let decode_seconds = td.elapsed().as_secs_f64();
    let probe = if decode_rc == 0 && mp4.is_file() {
        probe_video_path(&mp4.to_string_lossy())
            .unwrap_or_else(|error| json!({ "muxing": "probe_failed", "error": error }))
    } else {
        json!({ "muxing": "artifact_missing" })
    };
    let artifact_ok = decode_rc == 0
        && mp4.is_file()
        && wav.is_file()
        && probe.get("muxing").and_then(Value::as_str) == Some("probe_ok")
        && probe.get("width").and_then(Value::as_i64) == Some(1920)
        && probe.get("height").and_then(Value::as_i64) == Some(1088)
        && probe.get("frame_count").and_then(Value::as_i64) == Some(121)
        && (probe.get("fps").and_then(Value::as_f64).unwrap_or(0.0) - 24.0).abs() < 0.01
        && probe.get("video_codec").and_then(Value::as_str) == Some("h264")
        && probe.get("audio_codec").and_then(Value::as_str) == Some("aac")
        && probe.get("audio_sample_rate").and_then(Value::as_i64) == Some(48_000)
        && probe.get("audio_channels").and_then(Value::as_i64) == Some(2)
        && (5.03..=5.06).contains(&probe.get("duration").and_then(Value::as_f64).unwrap_or(0.0))
        && (5.00..=5.06).contains(
            &probe
                .get("audio_duration")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
        );
    let accepted_sampler_parity = ltx2_parity_report_passed(
        LTX2_SAMPLER_PARITY_REPORT,
        "serenity.ltx2.sampler_parity.v1",
        false,
    );
    let accepted_vae_parity = ltx2_parity_report_passed(
        LTX2_VAE_PARITY_REPORT,
        "serenity.ltx2.vae_frame_parity.v1",
        true,
    );
    let accepted_audio_parity = ltx2_parity_report_passed(
        LTX2_AUDIO_PARITY_REPORT,
        "serenity.ltx2.audio_parity.v1",
        true,
    );
    let result_path = out_dir.join("ltx2_video_result.json");
    let result = json!({
        "schema": "serenity.video_result.v1", "video_id": video_id,
        "backend": "mojo", "control_plane": "serenity-server",
        "model": "ltx2", "runner": "ltx2_refhq",
        "state": if artifact_ok { "done" } else { "failed" },
        "readiness_label": "bounded_product_route",
        "accepted_video_artifact": artifact_ok,
        "accepted_av_artifact": artifact_ok,
        "accepted_vae_parity": accepted_vae_parity,
        "accepted_audio_parity": accepted_audio_parity,
        "accepted_sampler_parity": accepted_sampler_parity,
        "sampler_parity_report": repo_path(LTX2_SAMPLER_PARITY_REPORT).to_string_lossy(),
        "vae_parity_report": repo_path(LTX2_VAE_PARITY_REPORT).to_string_lossy(),
        "audio_parity_report": repo_path(LTX2_AUDIO_PARITY_REPORT).to_string_lossy(),
        "checkpoint": LTX2_REFHQ_CHECKPOINT,
        "prompt": prompt, "negative_prompt": negative_prompt, "seed": seed,
        "conditioning_cache": requested_contexts,
        "conditioning_cache_key": context_cache.key,
        "conditioning_cache_hit": context_cache.hit,
        "conditioning_encoder_seconds": context_cache.encoder_seconds,
        "conditioning_log": context_cache.log_path.to_string_lossy(),
        "conditioning_manifest": context_cache.manifest_path.to_string_lossy(),
        "cache_before_video_model": true,
        "processes": ["stage1", "upscaler", "stage2", "decode"],
        "process_separated_decode": true,
        "creator_cudnn_decode_runtime": "9.10.2",
        "final_latent_dtype": "BF16",
        "accepted_final_latent_handoff": final_latents_bf16,
        "steps": steps, "stage1_steps": steps, "stage2_steps": 3,
        "width": 1920, "height": 1088,
        "frames": 121, "fps": 24, "audio_sample_rate": 48000,
        "tile_contract": { "spatial": [512, 64], "temporal": [64, 24] },
        "stage1_exit_code": stage1_rc,
        "upscale_exit_code": upscale_rc,
        "stage2_exit_code": stage2_rc,
        "decode_exit_code": decode_rc,
        "out_dir": out_dir.to_string_lossy(),
        "stage1_cache": stage1_cache.to_string_lossy(),
        "upscaler_cache": upscaler_cache.to_string_lossy(),
        "final_latents": final_latents.to_string_lossy(),
        "mp4": mp4.to_string_lossy(), "wav": wav.to_string_lossy(),
        "probe": probe,
        "mp4_url": format!("/out/{video_id}/ltx2_refhq.mp4"),
        "wav_url": format!("/out/{video_id}/refhq_audio.wav"),
        "stage1_log": stage1_log.to_string_lossy(),
        "upscale_log": upscale_log.to_string_lossy(),
        "stage2_log": stage2_log.to_string_lossy(),
        "decode_log": decode_log.to_string_lossy(),
        "result_path": result_path.to_string_lossy(),
        "stage1_seconds": stage1_seconds,
        "upscale_seconds": upscale_seconds,
        "stage2_seconds": stage2_seconds,
        "decode_seconds": decode_seconds,
        "stage1_peak_vram_mib": stage1_peak_vram_mib,
        "upscale_peak_vram_mib": upscale_peak_vram_mib,
        "stage2_peak_vram_mib": stage2_peak_vram_mib,
        "decode_peak_vram_mib": decode_peak_vram_mib,
        "total_wall_seconds": context_cache.encoder_seconds + stage1_seconds + upscale_seconds + stage2_seconds + decode_seconds,
        "note": "Creator-gated refhq cache pipeline: stage1 -> official x2 upscaler -> stage2 -> fresh-process Desktop-tiled Mojo A/V decode. Parity claims are read only from current pinned-Creator reports at the 0.999 bar.",
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

/// Construct one Wan process with the active checkout and Mojo runtime. The
/// caller owns logging and peak-VRAM measurement through
/// run_logged_with_gpu_peak.
pub(super) fn remux_ltx2_source_audio(
    video_artifact: &std::path::Path,
    source_video: &std::path::Path,
    out_dir: &std::path::Path,
    expected_frames: i64,
) -> Result<(std::path::PathBuf, Value), String> {
    let artifact = out_dir.join("ltx2_source_audio.mp4");
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &video_artifact.to_string_lossy(),
            "-i",
            &source_video.to_string_lossy(),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            &artifact.to_string_lossy(),
        ])
        .output()
        .map_err(|error| format!("cannot launch ffmpeg to preserve source audio: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg could not preserve the LTX2 source audio: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let probe = probe_video_path(&artifact.to_string_lossy())?;
    if probe.get("muxing").and_then(Value::as_str) != Some("probe_ok")
        || probe.get("has_audio").and_then(Value::as_bool) != Some(true)
        || probe.get("frame_count").and_then(Value::as_i64) != Some(expected_frames)
    {
        return Err(format!(
            "LTX2 source-audio artifact failed stream/frame probe: {probe}"
        ));
    }
    Ok((artifact, probe))
}

/// Stage the source-audio samples with the exact PyAV media contract used by
/// LTX Desktop. Mojo performs the creator resample, log-mel transform, and
/// learned AudioVAE encode; this helper only preserves PyAV's decoded samples.
pub(super) fn stage_ltx2_creator_source_audio(
    source_video: &std::path::Path,
    out_dir: &std::path::Path,
    frames: i64,
    fps: f64,
) -> Result<Option<Value>, String> {
    let probe = probe_video_path(&source_video.to_string_lossy())?;
    if probe.get("has_audio").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let python = repo_path(".pixi/envs/default/bin/python");
    let script = repo_path(LTX2_CREATOR_AUDIO_DECODER);
    if !python.is_file() || !script.is_file() {
        return Err(format!(
            "creator audio staging is unavailable; missing {} or {}",
            python.display(),
            script.display()
        ));
    }
    let raw = out_dir.join("ltx2_creator_source_audio.f32le");
    let duration = frames as f64 / fps;
    let output = std::process::Command::new(&python)
        .current_dir(repo_root())
        .arg(&script)
        .args(["--input"])
        .arg(source_video)
        .args(["--output"])
        .arg(&raw)
        .args(["--max-duration", &format!("{duration:.17}")])
        .output()
        .map_err(|error| format!("cannot launch creator PyAV audio decoder: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "creator PyAV audio decode failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata = stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok())
        .ok_or_else(|| "creator PyAV audio decoder returned no metadata".to_string())?;
    let sample_rate = metadata
        .get("sample_rate")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let channels = metadata
        .get("channels")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let samples = metadata
        .get("samples_per_channel")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let expected_bytes = samples.saturating_mul(channels).saturating_mul(4);
    let actual_bytes = std::fs::metadata(&raw)
        .map(|value| value.len() as i64)
        .unwrap_or(0);
    if sample_rate <= 0 || channels != 2 || samples <= 0 || actual_bytes != expected_bytes {
        return Err(format!(
            "creator source-audio staging contract failed: metadata={metadata}, bytes={actual_bytes}"
        ));
    }
    Ok(Some(json!({
        "path": raw.to_string_lossy(),
        "sample_rate": sample_rate,
        "channels": channels,
        "samples_per_channel": samples,
        "decoder": "pyav-16.1.0-creator-contract",
    })))
}

/// Foley/V2A freezes the source video and synthesizes only audio. Preserve the
/// source video stream byte-for-byte and replace its audio with the Mojo
/// runner's generated audio stream.
pub(super) fn remux_ltx2_generated_audio(
    source_video: &std::path::Path,
    generated_artifact: &std::path::Path,
    out_dir: &std::path::Path,
    expected_frames: i64,
) -> Result<(std::path::PathBuf, Value), String> {
    let artifact = out_dir.join("ltx2_foley_v2a.mp4");
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &source_video.to_string_lossy(),
            "-i",
            &generated_artifact.to_string_lossy(),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            &artifact.to_string_lossy(),
        ])
        .output()
        .map_err(|error| format!("cannot launch ffmpeg for LTX2 Foley output: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg could not assemble the LTX2 Foley output: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let probe = probe_video_path(&artifact.to_string_lossy())?;
    if probe.get("muxing").and_then(Value::as_str) != Some("probe_ok")
        || probe.get("has_video").and_then(Value::as_bool) != Some(true)
        || probe.get("has_audio").and_then(Value::as_bool) != Some(true)
        || probe.get("frame_count").and_then(Value::as_i64) != Some(expected_frames)
    {
        return Err(format!(
            "LTX2 Foley artifact failed stream/frame probe: {probe}"
        ));
    }
    Ok((artifact, probe))
}
