//! /v1/video — faithful port of the daemon's video endpoints (video_api.mojo):
//! GET /v1/video (readiness), POST /v1/video (bounded LTX2 staged smoke),
//! GET /v1/video/probe?path= (ffprobe wrapper).
//!
//! GET readiness is mostly static + a `test -x` on the LTX2 smoke runner; the
//! backend/model/resident identity fields identify Mojo as the inference
//! backend and Serenity Server as its control plane. POST runs the external
//! Mojo runner (not built by default -> "missing executable" error). Probe
//! shells `ffprobe` and reshapes its JSON.

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use serenity_wire::WorkerEvent;

use crate::AppState;

const RUNNER: &str = "output/bin/ltx2_video_smoke_runner";
// One request-driven runner. Geometry is request data, never encoded in the
// executable name or selected through a per-profile fallback.
const LTX2_MOJO_REQUEST_RUNNER: &str = "output/bin/ltx2_serenity_runtime";
const REALESRGAN_X4_RUNNER: &str = "output/bin/serenity_realesrgan_x4";
const REALESRGAN_X4_WEIGHTS: &str = "upscalers/realesrgan-x4plus/RealESRGAN_x4.safetensors";
const REALESRGAN_FAST_X4_WEIGHTS: &str =
    "upscalers/realesrgan-fast-x4v3/realesr-general-x4v3.safetensors";
const SEEDVR2_PRODUCT_RUNNER: &str = "output/bin/seedvr2_upscale_video";
const SEEDVR2_WEIGHTS: [&str; 3] = [
    "upscalers/seedvr2-3b/seedvr2_vae.safetensors",
    "upscalers/seedvr2-3b/seedvr2_dit.safetensors",
    "upscalers/seedvr2-3b/seedvr2_text_emb.safetensors",
];
const LTX2_REQUEST_PROFILES_JSON: &str =
    include_str!("../../../../serenitymojo/configs/ltx2_request_profiles.json");
const LTX2_CHECKPOINT_WORKFLOWS_JSON: &str =
    include_str!("../../../../serenitymojo/configs/ltx2_checkpoint_workflows.json");
const LTX2_MOJO_CONDITIONER: &str = "output/bin/ltx2_encode_prompt";
const LTX2_MOJO_CONTEXT_SCHEMA: &str = "serenity.ltx2.mojo_gemma3_context_cache.v1";
const LTX2_GEMMA_FP8: &str =
    "text_encoders/gemma-3-12b-it-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors";
const LTX2_GEMMA_TOKENIZER: &str = "text_encoders/gemma-3-12b-it-standalone/tokenizer.json";
const LTX2_CONDITIONING_CHECKPOINT: &str = "checkpoints/ltx-2.3-22b-distilled-fp8.safetensors";
const LTX2_CSHIM: &str = "serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so";
const LTX2_CONTEXT_PYTHON: &str = ".local/share/LTXDesktop/python/bin/python3";
const LTX2_CONTEXT_SCRIPT: &str = "scripts/ltx2_refhq_contexts.py";
const LTX2_CREATOR_AUDIO_DECODER: &str = "scripts/ltx2_decode_source_audio.py";
const LTX2_CONTEXT_SCHEMA: &str = "serenity.ltx2.refhq_context_cache.v1";
const LTX2_CREATOR_REVISION: &str = "780984275fd47128b02bef9b5c085404276866ee";
const LTX2_REFHQ_CHECKPOINT: &str = "ltx-2.3-22b-dev-fp8";
const LTX2_REFHQ_BF16_CHECKPOINT: &str = "ltx-2.3-22b-dev-fp8-dequant-bf16";
const LTX2_REFHQ_DISTILLATION_ADAPTER: &str =
    "checkpoints/ltx-2.3-22b-distilled-lora-384-1.1.safetensors";
const LTX2_CUDA_CACHE: &str = "/dev/shm/serenity-ltx2-cuda-cache";
const LTX2_SAMPLER_PARITY_REPORT: &str = "output/checks/ltx2_sampler_parity.json";
const LTX2_VAE_PARITY_REPORT: &str = "output/checks/ltx2_vae_frame_parity.json";
const LTX2_AUDIO_PARITY_REPORT: &str = "output/checks/ltx2_audio_parity.json";
const LTX2_CREATOR_CUDNN_LIB_CANDIDATES: [&str; 2] = [
    "LTX-Desktop/backend/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
    ".local/share/LTXDesktop/python/lib/python3.13/site-packages/nvidia/cudnn/lib",
];
const BACKEND_NAME: &str = "mojo";

/// Resolve the checkout that built this server, with an explicit override for
/// packaged/two-machine deployments.  Never silently jump to a different local
/// clone: the Mojo runner, scripts, and product UI must come from one codebase.
fn repo_root() -> std::path::PathBuf {
    crate::repository_root_path()
}

fn repo_path(path: &str) -> std::path::PathBuf {
    if std::path::Path::new(path).is_absolute() {
        std::path::PathBuf::from(path)
    } else {
        repo_root().join(path)
    }
}

fn user_home() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

fn home_path(path: &str) -> std::path::PathBuf {
    user_home().join(path)
}

fn serenity_root() -> std::path::PathBuf {
    std::env::var_os("SERENITY_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| user_home().join(".serenity"))
}

fn serenity_model_root() -> std::path::PathBuf {
    std::env::var_os("SERENITY_MODEL_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| serenity_root().join("models"))
}

fn serenity_path(path: &str) -> std::path::PathBuf {
    serenity_root().join(path)
}

fn model_path(path: &str) -> std::path::PathBuf {
    serenity_model_root().join(path)
}
/// Wan2.2 two-process video: encode umt5 conds, then T2V (both under output/bin/).
const WAN22_ENCODE: &str = "output/bin/wan22_encode_prompt";
const WAN22_T2V: &str = "output/bin/wan22_t2v_1280x704";
const WAN22_T2V_PORTRAIT: &str = "output/bin/wan22_t2v_704x1280";
const WAN22_I2V_LANDSCAPE: &str = "output/bin/wan22_t2v_1248x704";
const WAN22_I2V_PORTRAIT: &str = "output/bin/wan22_t2v_704x1248";
const WAN22_FIRST_FRAME_LANDSCAPE: &str =
    "output/bin/wan22_encode_first_frame_1280x704";
const WAN22_FIRST_FRAME_PORTRAIT: &str =
    "output/bin/wan22_encode_first_frame_704x1280";
const WAN22_FIRST_FRAME_I2V_LANDSCAPE: &str =
    "output/bin/wan22_encode_first_frame_1248x704";
const WAN22_FIRST_FRAME_I2V_PORTRAIT: &str =
    "output/bin/wan22_encode_first_frame_704x1248";
const WAN22_A14B_LORA_T2V: &str = "output/bin/wan22_a14b_lora_t2v";
const WAN22_A14B_HIGH: &str = "checkpoints/wan2.2_t2v_a14b_fp8_e4m3/high";
const WAN22_A14B_LOW: &str = "checkpoints/wan2.2_t2v_a14b_fp8_e4m3/low";
const WAN22_A14B_VAE: &str = "lingbot-video-moe/vae/diffusion_pytorch_model.safetensors";
const WAN22_MODEL_ROOT: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo";
const WAN22_ARTIFACT_MANIFEST: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/serenity_wan22_manifest.json";
const WAN22_TRANSFORMER_SHARD_1: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/diffusion_pytorch_model-00001-of-00003.safetensors";
const WAN22_TRANSFORMER_SHARD_2: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/diffusion_pytorch_model-00002-of-00003.safetensors";
const WAN22_TRANSFORMER_SHARD_3: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/diffusion_pytorch_model-00003-of-00003.safetensors";
const WAN22_UMT5_FILE: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5/model.safetensors";
const WAN22_TOKENIZER: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/tokenizer.json";
const WAN22_SPIECE: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/spiece.model";
const WAN22_VAE: &str = "vaes/wan2.2_vae.safetensors";
const WAN22_PRODUCT_GATE: &str = "output/checks/wan22_product_gate.json";
const WAN22_HF_REVISION: &str = "installed-official-native";
const WAN22_CREATOR_REVISION: &str = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc";
const WAN22_TRANSFORMER_INDEX_SHA256: &str =
    "cd769dd8bddb0825ffb3516a39d64fc2ac3a5946fb93337f8594af926d6a0f56";
const WAN22_LOCAL_TRANSFORMER_INDEX_SHA256: &str =
    "ff3fe4b6936ac924f881863bcaeda0e5e1e54c8b7e2202b2990aba8fcf18ce47";
const WAN22_TRANSFORMER_SHARD_SHA256: [&str; 3] = [
    "07cddfa20368c5e0884ee6660ed82b29d7ac97a9207b31fb630e4557c5308eb7",
    "38b79f68c95618f5341d4deae5ab364f9c74f10e8e903326499d0cb95353f1ff",
    "8d76abc71dee3e61a59ccc3a2e40889bb52ec9697acebfa7110de73f2a510452",
];
const WAN22_VAE_SHA256: &str =
    "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156";
const WAN22_RUNNER_SOURCE_BUNDLE_SHA256: &str =
    "ea317b6ae0914c4828d85489c1e5a2d0952d2ca3880a122e56287771f65d24fe";
const WAN22_DEFAULT_NEGATIVE: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
const WAN22_CUDA_CACHE: &str = "/dev/shm/serenity-wan22-cuda-cache";
const MOJO_CUDNN_RUNTIME: &str = "cudnn/lib/libcudnn.so.9";
/// Pixi runtime libs + cshim (cuDNN/int4) shims — required by the Mojo binaries.
pub(crate) fn mojo_ld_path() -> std::ffi::OsString {
    let root = repo_root();
    let mut parts = vec![
        root.join(".pixi/envs/default/lib"),
        root.join("serenitymojo/ops/cshim/lib"),
        serenity_path("cudnn/lib"),
    ];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        parts.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(parts).unwrap_or_default()
}

/// LTX Creator audio parity is cuDNN-version sensitive. Decode is already a
/// fresh process, so pin only that process to Creator's measured 9.10.2
/// runtime; denoising and SDPA continue to use the general Mojo runtime.
fn ltx2_decode_cudnn_lib() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("LTX2_CREATOR_CUDNN_LIB") {
        return std::path::PathBuf::from(path);
    }
    LTX2_CREATOR_CUDNN_LIB_CANDIDATES
        .iter()
        .map(|path| home_path(path))
        .find(|path| nonempty_file(&path.join("libcudnn.so.9")))
        .unwrap_or_else(|| home_path(LTX2_CREATOR_CUDNN_LIB_CANDIDATES[0]))
}

fn ltx2_decode_runtime_available() -> bool {
    nonempty_file(&ltx2_decode_cudnn_lib().join("libcudnn.so.9"))
}

fn ltx2_decode_ld_path() -> std::ffi::OsString {
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
fn ltx2_request_ld_path(edit_mode: &str) -> std::ffi::OsString {
    if edit_mode == "standard" {
        mojo_ld_path()
    } else {
        ltx2_decode_ld_path()
    }
}
/// svdint4 slab matching the distilled-fp8 base the LTX2 runner streams
/// (`CKPT_FP8` in ltx2_t2v_av_hq.mojo). Selected via `LTX2_INT4_SLAB` for the
/// int4 W4A16-resident path. (Verified present on this box, 2026-07-11.)
const LTX2_INT4_SLAB: &str = "checkpoints/ltx-2.3-22b-distilled-svdint4-r32.safetensors";
/// W4A16 resident slab reconstructed from the LTX-2.3 dev checkpoint. The
/// request runner applies the official support LoRA and authored LoRAs on top,
/// so it must not use the already-distilled slab above.
const LTX2_REFHQ_INT4_SLAB: &str = "checkpoints/ltx-2.3-22b-svdint4-r32.safetensors";
/// Full dequantized BF16 LTX-2.3 dev transformer used by the request runner
/// when Canvas selects BF16 precision. Activations remain BF16 and reductions
/// remain F32; this is a real storage-mode switch, not an FP8 label alias.
const LTX2_REFHQ_BF16: &str = "checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors";
/// wan22_t2v compiled (comptime) geometry — MUST match the binary. It raises on
/// any `frames != WAN22_FRAMES`, so the server rejects a mismatch up front rather
/// than burning a multi-minute render on a guaranteed failure.
const WAN22_FRAMES: i64 = 121;
const WAN22_WIDTH: i64 = 1280;
const WAN22_HEIGHT: i64 = 704;
const WAN22_PORTRAIT_WIDTH: i64 = 704;
const WAN22_PORTRAIT_HEIGHT: i64 = 1280;
const WAN22_I2V_LANDSCAPE_WIDTH: i64 = 1248;
const WAN22_I2V_LANDSCAPE_HEIGHT: i64 = 704;
const WAN22_I2V_PORTRAIT_WIDTH: i64 = 704;
const WAN22_I2V_PORTRAIT_HEIGHT: i64 = 1248;
const WAN22_FPS: i64 = 24;
const WAN22_DEFAULT_STEPS: i64 = 50;
const WAN22_I2V_STEPS: i64 = 50;
const WAN22_DEFAULT_GUIDANCE: f64 = 5.0;
const WAN22_A14B_FRAMES: i64 = 81;
const WAN22_A14B_WIDTH: i64 = 832;
const WAN22_A14B_HEIGHT: i64 = 480;
const WAN22_A14B_FPS: i64 = 16;
const WAN22_A14B_STEPS: i64 = 40;
const WAN22_A14B_GUIDANCE: f64 = 3.0;

/// Bernini-R reuses the admitted UMT5 producer, then runs bounded high/low
/// A14B expert streams and the existing standard-Wan VAE in three isolated
/// Mojo processes. Product exposure remains fail-closed on the local evidence
/// report; Rust is orchestration/readiness only.
const BERNINI_T2V: &str = "output/bin/bernini_t2v";
const BERNINI_DECODE: &str = "output/bin/bernini_decode";
const BERNINI_MODEL_ROOT: &str = "checkpoints/Bernini-R-Diffusers";
const BERNINI_ARTIFACT_MANIFEST: &str =
    "checkpoints/Bernini-R-Diffusers/serenity_bernini_r_manifest.json";
const BERNINI_HIGH_CACHE: &str = "checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/high";
const BERNINI_LOW_CACHE: &str = "checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/low";
const BERNINI_VAE: &str = "checkpoints/Bernini-R-Diffusers/vae/diffusion_pytorch_model.safetensors";
const BERNINI_PRODUCT_GATE: &str = "output/checks/bernini_r/product_gate.json";
const BERNINI_HF_REVISION: &str = "de8c4621d3ac75cc33efe3db8deaed2023e9ac8c";
const BERNINI_CREATOR_REVISION: &str = "2d2b4591ac053ec25c6371b01a5a6746679e5793";
const BERNINI_WIDTH: i64 = 848;
const BERNINI_HEIGHT: i64 = 480;
const BERNINI_FRAMES: i64 = 81;
const BERNINI_FPS: i64 = 16;
const BERNINI_DEFAULT_STEPS: i64 = 40;
const BERNINI_DEFAULT_GUIDANCE: f64 = 4.0;

/// SCAIL-2 single-segment character animation. Every conditioning artifact is
/// produced automatically per request; only the installed model/cache and four
/// user media inputs are prerequisites.
const SCAIL2_STAGE: &str = "output/bin/scail2_stage_inputs";
const SCAIL2_ENCODE_PROMPT: &str = "output/bin/scail2_encode_prompt";
const SCAIL2_ENCODE_CLIP: &str = "output/bin/scail2_encode_clip";
const SCAIL2_ENCODE_VAE: &str = "output/bin/scail2_encode_vae";
const SCAIL2_PREPARE_CACHE: &str = "output/bin/scail2_prepare_fp8_cache";
const SCAIL2_ANIMATION: &str = "output/bin/scail2_animation";
const SCAIL2_DECODE: &str = "output/bin/scail2_decode";
const SCAIL2_OFFICIAL_ROOT: &str = "checkpoints/SCAIL-2";
const SCAIL2_MOJO_ROOT: &str = "checkpoints/SCAIL-2-Mojo";
const SCAIL2_UMT5: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5";
const SCAIL2_TOKENIZER: &str = "checkpoints/SCAIL-2/umt5-xxl/tokenizer.json";
const SCAIL2_CLIP: &str = "checkpoints/SCAIL-2-Mojo/clip_visual/model.safetensors";
const SCAIL2_FP8_CACHE: &str = "checkpoints/SCAIL-2-Mojo/transformer_fp8";
const SCAIL2_VAE: &str = BERNINI_VAE;
const SCAIL2_PRODUCT_GATE: &str = "output/checks/scail2/product_gate.json";
const SCAIL2_SOURCE_COMMIT: &str = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5";
const SCAIL2_MODEL_REVISION: &str = "150cc0ca4e98e50e60b9295dacde39442fdccab2";
const SCAIL2_CHECKPOINT_SHA256: &str =
    "d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f";
const SCAIL2_WIDTH: i64 = 896;
const SCAIL2_HEIGHT: i64 = 512;
const SCAIL2_FRAMES: i64 = 65;
const SCAIL2_FPS: i64 = 16;
const SCAIL2_STEPS: i64 = 40;
const SCAIL2_GUIDANCE: f64 = 5.0;

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

/// `test -x`: exists + executable bit set. Relative paths resolve against
/// the resolved repo root so the server works regardless of its launch cwd.
fn bin_x(path: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let abs = repo_path(path);
    std::fs::metadata(abs)
        .map(|m| m.is_file() && (m.permissions().mode() & 0o111) != 0)
        .unwrap_or(false)
}

#[derive(Clone, Debug, serde::Deserialize)]
struct Ltx2RequestProfileRegistry {
    schema: String,
    checkpoint: String,
    checkpoints: Vec<Ltx2CheckpointProfile>,
    guidance_modes: Value,
    profile_groups: Vec<Ltx2RequestProfileGroup>,
    post_upscalers: Value,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct Ltx2CheckpointProfile {
    id: String,
    label: String,
    path: String,
    aliases: Vec<String>,
    support_lora: String,
    guidance_modes: Vec<String>,
    quant_modes: Vec<String>,
    readiness_label: String,
    source: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct Ltx2RequestProfileGroup {
    id: String,
    label: String,
    width: i64,
    height: i64,
    #[serde(default)]
    conditioning_width: Option<i64>,
    #[serde(default)]
    conditioning_height: Option<i64>,
    fps: i64,
    modes: Vec<String>,
    durations: Vec<f64>,
    frames: Vec<i64>,
    source: String,
}

#[derive(Clone, Debug)]
struct Ltx2ResolvedRequestProfile {
    group_id: String,
    label: String,
    width: i64,
    height: i64,
    conditioning_width: i64,
    conditioning_height: i64,
    frames: i64,
    fps: i64,
    modes: Vec<String>,
    duration: f64,
    source: String,
}

fn ltx2_checkpoint_workflow_registry() -> &'static Value {
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

fn ltx2_checkpoint_workflow(checkpoint: &str) -> Option<&'static Value> {
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

fn ltx2_checkpoint_workflow_documents() -> Value {
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
            let enhancer_files_available =
                enhancer_weights.as_deref().is_some_and(nonempty_file)
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
                object.insert(
                    "prompt_enhancer_available".to_string(),
                    json!(false),
                );
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

fn ltx2_request_profile_registry() -> &'static Ltx2RequestProfileRegistry {
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

const LTX2_REQUEST_RUNNER_BUILD_INPUTS: &[&str] = &[
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

fn ltx2_runner_mtime_covers_inputs(
    runner_modified: std::time::SystemTime,
    input_modified: &[std::time::SystemTime],
) -> bool {
    input_modified
        .iter()
        .all(|modified| *modified <= runner_modified)
}

/// Fail closed when the single runtime-geometry runner is missing or stale.
fn ltx2_request_runner_current(path: &str) -> bool {
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

fn ltx2_checkpoint_profile(checkpoint: &str) -> Option<&'static Ltx2CheckpointProfile> {
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

fn ltx2_checkpoint_document(profile: &Ltx2CheckpointProfile) -> Value {
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

fn ltx2_checkpoint_documents() -> Value {
    Value::Array(
        ltx2_request_profile_registry()
            .checkpoints
            .iter()
            .map(ltx2_checkpoint_document)
            .collect(),
    )
}

fn ltx2_resolved_profiles() -> Vec<Ltx2ResolvedRequestProfile> {
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

fn ltx2_profile_runner_available(_profile: &Ltx2ResolvedRequestProfile) -> bool {
    ltx2_request_runner_current(LTX2_MOJO_REQUEST_RUNNER)
}

fn ltx2_request_profile_for_mode(
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

fn ltx2_profile_document(profile: &Ltx2ResolvedRequestProfile) -> Value {
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

fn stage_ltx2_creator_i2v_source(
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

fn ltx2_post_upscaler_documents() -> Value {
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

fn runner_available() -> bool {
    bin_x(RUNNER)
}

fn nonempty_file(path: &std::path::Path) -> bool {
    path.metadata()
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false)
}

fn sha256sum(path: &std::path::Path) -> Option<String> {
    let output = std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let digest = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()?
        .to_ascii_lowercase();
    (digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(digest)
}

fn verify_ltx2_creator_revision() -> Result<(), String> {
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

fn gpu_memory_mib_for_pid(pid: u32) -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (seen_pid, memory) = line.split_once(',')?;
            (seen_pid.trim().parse::<u32>().ok()? == pid)
                .then(|| memory.trim().parse::<u64>().ok())?
        })
        .max()
}

fn run_logged_with_gpu_peak(
    command: &mut std::process::Command,
    log_path: &std::path::Path,
) -> Result<(i32, Option<u64>), String> {
    let log = std::fs::File::create(log_path)
        .map_err(|e| format!("cannot create {}: {e}", log_path.display()))?;
    let log_err = log
        .try_clone()
        .map_err(|e| format!("cannot clone {}: {e}", log_path.display()))?;
    let mut child = command
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err))
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;
    let pid = child.id();
    let mut peak_mib = None;
    loop {
        if let Some(memory) = gpu_memory_mib_for_pid(pid) {
            peak_mib = Some(peak_mib.map_or(memory, |peak: u64| peak.max(memory)));
        }
        match child.try_wait().map_err(|e| format!("wait failed: {e}"))? {
            Some(status) => return Ok((status.code().unwrap_or(-1), peak_mib)),
            // Wan's short dequant/attention transients were missed by a 1 s
            // poll during the measured 16 GiB acceptance run. 100 ms captures
            // the real process peak without altering inference execution.
            None => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
}

fn safetensors_header(path: &std::path::Path) -> Option<Value> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length).ok()?;
    let header_len = u64::from_le_bytes(length) as usize;
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return None;
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header).ok()?;
    serde_json::from_slice::<Value>(&header).ok()
}

fn tensor_header_matches(doc: &Value, key: &str, dtype: &str, shape: &[u64]) -> bool {
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

fn final_latents_are_bf16(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "video", "BF16", &[1, 128, 16, 34, 60])
        && tensor_header_matches(&doc, "audio", "BF16", &[1, 8, 501, 16])
}

fn stage1_cache_is_bf16(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "video", "BF16", &[1, 8160, 128])
        && tensor_header_matches(&doc, "audio", "BF16", &[1, 126, 128])
}

fn upscaler_cache_is_f32(path: &std::path::Path) -> bool {
    let Some(doc) = safetensors_header(path) else {
        return false;
    };
    tensor_header_matches(&doc, "out", "F32", &[1, 128, 16, 34, 60])
}

fn ltx2_context_cache_valid(path: &std::path::Path) -> bool {
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

fn ltx2_mojo_context_tensor_valid(path: &std::path::Path) -> bool {
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

fn ltx2_mojo_conditioning_missing() -> Vec<String> {
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
fn ltx2_parity_report_passed(path: &str, schema: &str, require_shared_latent: bool) -> bool {
    ltx2_parity_report_matches(
        &repo_path(path),
        &repo_path(RUNNER),
        schema,
        require_shared_latent,
    )
}

fn ltx2_parity_report_matches(
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

fn ltx2_context_key(prompt: &str, negative_prompt: &str) -> String {
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

fn ltx2_seed(body: &Value) -> Result<u64, &'static str> {
    const ERROR: &str = "ltx2_refhq 'seed' must be an integer from 0 through 4294967295";
    match body.get("seed") {
        None => Ok(42),
        Some(value) => value
            .as_u64()
            .filter(|seed| *seed <= u64::from(u32::MAX))
            .ok_or(ERROR),
    }
}

struct Ltx2ContextCache {
    path: std::path::PathBuf,
    key: String,
    hit: bool,
    encoder_seconds: f64,
    log_path: std::path::PathBuf,
    manifest_path: std::path::PathBuf,
}

/// Run the existing Creator-backed Gemma producer before the Mojo video model
/// is loaded. The completed single-file context is reused by prompt/negative
/// key; the producer exits before refhq denoising begins.
fn prepare_ltx2_refhq_context(
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

fn ltx2_mojo_context_key(prompt: &str, negative_prompt: &str, conditioner_sha256: &str) -> String {
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

fn write_ltx2_job_status(
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

fn ltx2_mojo_cache_manifest_valid(
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
fn prepare_ltx2_mojo_context<F>(
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

/// Which Wan2.2 runtime prerequisites are absent (empty = the arm is runnable).
/// Checking only the two executables produced a false-ready response on a clean
/// machine: both binaries existed while two DiT shards and the converted UMT5
/// cache did not. Keep this list aligned with the paths compiled into the Mojo
/// runners above; Rust remains readiness/control-plane only.
fn wan22_missing() -> Vec<String> {
    let mut m = Vec::new();
    if !bin_x(WAN22_ENCODE) {
        m.push(WAN22_ENCODE.to_string());
    }
    if !bin_x(WAN22_T2V) {
        m.push(WAN22_T2V.to_string());
    }
    if !bin_x(WAN22_T2V_PORTRAIT) {
        m.push(WAN22_T2V_PORTRAIT.to_string());
    }
    if !bin_x(WAN22_I2V_LANDSCAPE) {
        m.push(WAN22_I2V_LANDSCAPE.to_string());
    }
    if !bin_x(WAN22_I2V_PORTRAIT) {
        m.push(WAN22_I2V_PORTRAIT.to_string());
    }
    for binary in [
        WAN22_FIRST_FRAME_LANDSCAPE,
        WAN22_FIRST_FRAME_PORTRAIT,
        WAN22_FIRST_FRAME_I2V_LANDSCAPE,
        WAN22_FIRST_FRAME_I2V_PORTRAIT,
    ] {
        if !bin_x(binary) {
            m.push(binary.to_string());
        }
    }
    for path in [
        WAN22_ARTIFACT_MANIFEST,
        WAN22_TRANSFORMER_SHARD_1,
        WAN22_TRANSFORMER_SHARD_2,
        WAN22_TRANSFORMER_SHARD_3,
        WAN22_UMT5_FILE,
        WAN22_TOKENIZER,
        WAN22_SPIECE,
        WAN22_VAE,
    ] {
        let resolved = model_path(path);
        if !nonempty_file(&resolved) {
            m.push(resolved.to_string_lossy().into_owned());
        }
    }
    m
}

fn wan22_a14b_cache_complete(path: &std::path::Path) -> bool {
    nonempty_file(&path.join("shared.safetensors"))
        && (0..40).all(|index| nonempty_file(&path.join(format!("block_{index:02}.safetensors"))))
}

fn wan22_a14b_missing() -> Vec<String> {
    let mut missing = Vec::new();
    for binary in [WAN22_ENCODE, WAN22_A14B_LORA_T2V] {
        if !bin_x(binary) {
            missing.push(binary.to_string());
        }
    }
    for cache in [WAN22_A14B_HIGH, WAN22_A14B_LOW] {
        let resolved = model_path(cache);
        if !wan22_a14b_cache_complete(&resolved) {
            missing.push(resolved.to_string_lossy().into_owned());
        }
    }
    let vae = model_path(WAN22_A14B_VAE);
    if !nonempty_file(&vae) {
        missing.push(vae.to_string_lossy().into_owned());
    }
    missing
}

fn wan22_a14b_lora(body: &Value) -> Result<(std::path::PathBuf, f64, String), String> {
    let rows = body
        .get("lora")
        .or_else(|| body.get("loras"))
        .and_then(Value::as_array)
        .ok_or_else(|| "Wan2.2 A14B preview requires exactly one authored LoRA".to_string())?;
    if rows.len() != 1 {
        return Err("Wan2.2 A14B preview requires exactly one authored LoRA".to_string());
    }
    let row = rows[0]
        .as_object()
        .ok_or_else(|| "Wan2.2 A14B lora[0] must be an object".to_string())?;
    let name = row.get("name").and_then(Value::as_str).unwrap_or("").trim();
    if name.is_empty() {
        return Err("Wan2.2 A14B lora[0].name is required".to_string());
    }
    let weight = row
        .get("weight")
        .or_else(|| row.get("strength"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if !weight.is_finite() || !(-10.0..=10.0).contains(&weight) {
        return Err("Wan2.2 A14B lora[0] weight must be finite in [-10, 10]".to_string());
    }
    let Some((path, arch)) = crate::models::lora_path_and_arch(name) else {
        return Err(format!(
            "Wan2.2 A14B LoRA not found in the model registry: {name}"
        ));
    };
    if !path.is_file() {
        return Err(format!(
            "Wan2.2 A14B LoRA path is missing: {}",
            path.display()
        ));
    }
    if arch != "wan2.2" {
        return Err(format!(
            "Wan2.2 A14B LoRA '{name}' targets '{arch}', not wan2.2"
        ));
    }
    Ok((path, weight, name.to_string()))
}

fn wan22_ti2v5b_lora_header(doc: &Value) -> Result<usize, String> {
    let Some(tensors) = doc.as_object() else {
        return Err("invalid safetensors header".to_string());
    };
    let mut pairs = 0;
    for (key, entry) in tensors {
        if key == "__metadata__" || !key.ends_with(".lora_A.weight") {
            continue;
        }
        if !key.starts_with("diffusion_model.blocks.") {
            return Err(format!(
                "unsupported Wan LoRA key '{key}'; TI2V-5B admits AI Toolkit/DiffusionModel block adapters"
            ));
        }
        let prefix = key.trim_end_matches(".lora_A.weight");
        let b_key = format!("{prefix}.lora_B.weight");
        let Some(b_entry) = tensors.get(&b_key) else {
            return Err(format!("Wan LoRA is missing pair tensor '{b_key}'"));
        };
        let a_shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Wan LoRA tensor '{key}' has no shape"))?;
        let b_shape = b_entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Wan LoRA tensor '{b_key}' has no shape"))?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!("Wan TI2V-5B LoRA tensor '{key}' must be rank-2"));
        }
        let rank = a_shape[0]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{key}' has invalid rank"))?;
        let input = a_shape[1]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{key}' has invalid input size"))?;
        let output = b_shape[0]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{b_key}' has invalid output size"))?;
        let b_rank = b_shape[1]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{b_key}' has invalid rank"))?;
        if rank == 0 || b_rank != rank {
            return Err(format!("Wan LoRA A/B rank mismatch for '{prefix}'"));
        }
        let module = prefix
            .split_once(".blocks.")
            .and_then(|(_, tail)| tail.split_once('.').map(|(_, module)| module))
            .unwrap_or("");
        let expected = match module {
            "self_attn.q" | "self_attn.k" | "self_attn.v" | "self_attn.o"
            | "cross_attn.q" | "cross_attn.k" | "cross_attn.v" | "cross_attn.o" => {
                (3_072, 3_072)
            }
            "ffn.0" => (14_336, 3_072),
            "ffn.2" => (3_072, 14_336),
            _ => {
                return Err(format!(
                    "Wan TI2V-5B LoRA module '{module}' is not an admitted block linear"
                ));
            }
        };
        if (output, input) != expected {
            return Err(format!(
                "Wan LoRA module '{module}' has [{output},{input}], expected TI2V-5B [{},{}]; this is probably a 14B adapter",
                expected.0, expected.1
            ));
        }
        for (tensor_key, tensor_entry) in [(key.as_str(), entry), (b_key.as_str(), b_entry)] {
            let dtype = tensor_entry
                .get("dtype")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !matches!(dtype, "BF16" | "F16" | "F32") {
                return Err(format!(
                    "Wan LoRA tensor '{tensor_key}' uses unsupported dtype '{dtype}'"
                ));
            }
        }
        pairs += 1;
    }
    if pairs == 0 {
        return Err(
            "no AI Toolkit/DiffusionModel Wan TI2V-5B LoRA A/B pairs were found".to_string(),
        );
    }
    Ok(pairs)
}

fn wan22_ti2v5b_lora(
    body: &Value,
) -> Result<Option<(std::path::PathBuf, f64, String, usize)>, String> {
    let Some(rows) = body
        .get("lora")
        .or_else(|| body.get("loras"))
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() != 1 {
        return Err(
            "Wan2.2-TI2V-5B currently accepts one resident LoRA per render".to_string(),
        );
    }
    let row = rows[0]
        .as_object()
        .ok_or_else(|| "Wan2.2-TI2V-5B lora[0] must be an object".to_string())?;
    let name = row
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return Err("Wan2.2-TI2V-5B lora[0].name is required".to_string());
    }
    let weight = row
        .get("weight")
        .or_else(|| row.get("strength"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if !weight.is_finite() || !(-10.0..=10.0).contains(&weight) {
        return Err(
            "Wan2.2-TI2V-5B lora[0] weight must be finite in [-10, 10]".to_string(),
        );
    }
    if crate::models::lora_usage(name) != "overlay" {
        return Err(format!(
            "Wan2.2-TI2V-5B LoRA '{name}' is a feature adapter, not a model overlay"
        ));
    }
    let Some((path, arch)) = crate::models::lora_path_and_arch(name) else {
        return Err(format!(
            "Wan2.2-TI2V-5B LoRA not found in the model registry: {name}"
        ));
    };
    if arch != "wan2.2" {
        return Err(format!(
            "Wan2.2-TI2V-5B LoRA '{name}' targets '{arch}', not wan2.2"
        ));
    }
    let header = safetensors_header(&path)
        .ok_or_else(|| format!("cannot read Wan LoRA safetensors header: {}", path.display()))?;
    let pairs = wan22_ti2v5b_lora_header(&header)?;
    Ok(Some((path, weight, name.to_string(), pairs)))
}

/// Read acceptance only from the machine-local evidence gate. The report is
/// regenerated by scripts/check_wan22_product_gate.py after verifying the
/// pinned native BF16 shards/VAE, runtime parity, representative frame
/// bytes, muxed T2V/I2V artifacts, visual inspection, wall time, and peak VRAM.
fn wan22_product_gate_passed() -> bool {
    let Ok(bytes) = std::fs::read(repo_path(WAN22_PRODUCT_GATE)) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    doc.get("schema").and_then(Value::as_str) == Some("serenity.wan22.product_gate.v3")
        && doc.get("passed").and_then(Value::as_bool) == Some(true)
        && doc.pointer("/pins/hf_revision").and_then(Value::as_str) == Some(WAN22_HF_REVISION)
        && doc
            .pointer("/pins/creator_revision")
            .and_then(Value::as_str)
            == Some(WAN22_CREATOR_REVISION)
        && doc
            .pointer("/pins/source_transformer_index_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_TRANSFORMER_INDEX_SHA256)
        && doc
            .pointer("/pins/local_transformer_index_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_LOCAL_TRANSFORMER_INDEX_SHA256)
        && (0..WAN22_TRANSFORMER_SHARD_SHA256.len()).all(|index| {
            doc.pointer(&format!(
                "/pins/bf16_transformer_shard_sha256/{index}"
            ))
            .and_then(Value::as_str)
                == Some(WAN22_TRANSFORMER_SHARD_SHA256[index])
        })
        && doc
            .pointer("/pins/bf16_vae_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_VAE_SHA256)
        && doc
            .pointer("/pins/runner_source_bundle_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_RUNNER_SOURCE_BUNDLE_SHA256)
        && doc.pointer("/profile/width").and_then(Value::as_i64) == Some(WAN22_WIDTH)
        && doc.pointer("/profile/height").and_then(Value::as_i64) == Some(WAN22_HEIGHT)
        && doc.pointer("/profile/frames").and_then(Value::as_i64) == Some(WAN22_FRAMES)
        && doc.pointer("/profile/steps").and_then(Value::as_i64) == Some(WAN22_DEFAULT_STEPS)
        && doc.pointer("/profile/guidance").and_then(Value::as_f64) == Some(WAN22_DEFAULT_GUIDANCE)
        && doc.pointer("/profile/shift").and_then(Value::as_f64) == Some(5.0)
        && doc.pointer("/profile/quant").and_then(Value::as_str) == Some("bf16")
        && doc.pointer("/i2v_profile/steps").and_then(Value::as_i64)
            == Some(WAN22_I2V_STEPS)
        && doc.pointer("/i2v_profile/shift").and_then(Value::as_f64) == Some(5.0)
        && doc.pointer("/i2v_profile/quant").and_then(Value::as_str) == Some("bf16")
        && doc.pointer("/i2v_profile/width").and_then(Value::as_i64)
            == Some(WAN22_I2V_PORTRAIT_WIDTH)
        && doc.pointer("/i2v_profile/height").and_then(Value::as_i64)
            == Some(WAN22_I2V_PORTRAIT_HEIGHT)
        && doc
            .pointer("/checks/i2v_first_frame_identity")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/checks/vae_encoder_mojo_parity")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/checks/creator_prompt_extension")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/checks/transformer_bf16_stream_parity")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/performance/requires_isolated_gpu_worker")
            .and_then(Value::as_bool)
            == Some(true)
}

fn bernini_cache_complete(dir: &str) -> bool {
    let root = std::path::Path::new(dir);
    if !nonempty_file(&root.join("shared.safetensors"))
        || !nonempty_file(&root.join("serenity_cache_manifest.json"))
    {
        return false;
    }
    (0..40).all(|index| nonempty_file(&root.join(format!("block_{index:02}.safetensors"))))
}

fn bernini_missing() -> Vec<String> {
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
        if !bernini_cache_complete(&resolved.to_string_lossy()) {
            missing.push(resolved.to_string_lossy().into_owned());
        }
    }
    missing
}

fn bernini_cache_aggregate(dir: &str) -> Option<String> {
    let bytes =
        std::fs::read(std::path::Path::new(dir).join("serenity_cache_manifest.json")).ok()?;
    let doc = serde_json::from_slice::<Value>(&bytes).ok()?;
    if doc.get("schema").and_then(Value::as_str) != Some("serenity.bernini_r.fp8_cache.v1")
        || doc.get("passed").and_then(Value::as_bool) != Some(true)
        || doc.get("revision").and_then(Value::as_str) != Some(BERNINI_HF_REVISION)
    {
        return None;
    }
    doc.get("cache_aggregate_sha256")
        .and_then(Value::as_str)
        .filter(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(str::to_string)
}

pub(crate) fn bernini_product_gate_passed() -> bool {
    let Ok(bytes) = std::fs::read(repo_path(BERNINI_PRODUCT_GATE)) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    let Some(high_cache) =
        bernini_cache_aggregate(&model_path(BERNINI_HIGH_CACHE).to_string_lossy())
    else {
        return false;
    };
    let Some(low_cache) = bernini_cache_aggregate(&model_path(BERNINI_LOW_CACHE).to_string_lossy())
    else {
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
    doc.get("schema").and_then(Value::as_str) == Some("serenity.bernini_r.product_gate.v1")
        && doc.get("passed").and_then(Value::as_bool) == Some(true)
        && doc.pointer("/pins/hf_revision").and_then(Value::as_str) == Some(BERNINI_HF_REVISION)
        && doc
            .pointer("/pins/creator_revision")
            .and_then(Value::as_str)
            == Some(BERNINI_CREATOR_REVISION)
        && doc
            .pointer("/pins/high_cache_aggregate_sha256")
            .and_then(Value::as_str)
            == Some(high_cache.as_str())
        && doc
            .pointer("/pins/low_cache_aggregate_sha256")
            .and_then(Value::as_str)
            == Some(low_cache.as_str())
        && doc
            .pointer("/pins/encode_runner_sha256")
            .and_then(Value::as_str)
            == Some(encode_runner.as_str())
        && doc
            .pointer("/pins/denoise_runner_sha256")
            .and_then(Value::as_str)
            == Some(denoise_runner.as_str())
        && doc
            .pointer("/pins/decode_runner_sha256")
            .and_then(Value::as_str)
            == Some(decode_runner.as_str())
        && doc
            .pointer("/pins/cudnn_runtime_sha256")
            .and_then(Value::as_str)
            == Some(cudnn_runtime.as_str())
        && doc.pointer("/profile/width").and_then(Value::as_i64) == Some(BERNINI_WIDTH)
        && doc.pointer("/profile/height").and_then(Value::as_i64) == Some(BERNINI_HEIGHT)
        && doc.pointer("/profile/frames").and_then(Value::as_i64) == Some(BERNINI_FRAMES)
        && doc.pointer("/profile/fps").and_then(Value::as_i64) == Some(BERNINI_FPS)
        && doc.pointer("/profile/steps").and_then(Value::as_i64) == Some(BERNINI_DEFAULT_STEPS)
        && doc
            .pointer("/performance/requires_isolated_process_stages")
            .and_then(Value::as_bool)
            == Some(true)
}

fn scail2_cache_complete() -> bool {
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

fn scail2_missing() -> Vec<String> {
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
            return true;
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

fn readiness_doc() -> Value {
    let ltx2_runner_ready = runner_available();
    let ltx2_decode_ready = ltx2_decode_runtime_available();
    let ltx2_legacy_ready = ltx2_runner_ready && ltx2_decode_ready;
    let ltx2_profiles = ltx2_resolved_profiles();
    let ltx2_request_ready = ltx2_profiles.iter().any(ltx2_profile_runner_available);
    let ltx2_profile_documents = ltx2_profiles
        .iter()
        .map(ltx2_profile_document)
        .collect::<Vec<_>>();
    let ltx2_bf16_checkpoint = model_path(LTX2_REFHQ_BF16);
    let ltx2_bf16_available = nonempty_file(&ltx2_bf16_checkpoint);
    let ltx2_default_profile = ltx2_profiles
        .iter()
        .find(|profile| ltx2_profile_runner_available(profile))
        .or_else(|| ltx2_profiles.first())
        .map(ltx2_profile_document)
        .unwrap_or(Value::Null);
    let ltx2_conditioning_missing = ltx2_mojo_conditioning_missing();
    let ltx2_auto_conditioning_ready = ltx2_conditioning_missing.is_empty();
    let ltx2_ready = ltx2_request_ready || ltx2_legacy_ready;
    let ltx2_sampler_parity = ltx2_parity_report_passed(
        LTX2_SAMPLER_PARITY_REPORT,
        "serenity.ltx2.sampler_parity.v1",
        false,
    );
    let ltx2_vae_parity = ltx2_parity_report_passed(
        LTX2_VAE_PARITY_REPORT,
        "serenity.ltx2.vae_frame_parity.v1",
        true,
    );
    let ltx2_audio_parity = ltx2_parity_report_passed(
        LTX2_AUDIO_PARITY_REPORT,
        "serenity.ltx2.audio_parity.v1",
        true,
    );
    // Wan2.2 is runnable only when its binaries and every compiled-in model
    // asset are present. Runtime/VRAM acceptance remains a separate gate.
    let wan22_absent = wan22_missing();
    let wan22_ready = wan22_absent.is_empty();
    let wan22_product_accepted = wan22_ready && wan22_product_gate_passed();
    let bernini_absent = bernini_missing();
    let bernini_ready = bernini_absent.is_empty();
    let bernini_product_accepted = bernini_ready && bernini_product_gate_passed();
    let scail2_absent = scail2_missing();
    let scail2_ready = scail2_absent.is_empty();
    let scail2_product_accepted = scail2_ready && scail2_product_gate_passed();
    // top-level state reflects whether ANY arm is runnable.
    let any_ready = ltx2_ready || wan22_ready || bernini_ready || scail2_ready;
    let state = if ltx2_request_ready {
        "request_runner_ready"
    } else if any_ready {
        "bounded_smoke_ready"
    } else {
        "runner_missing"
    };
    let ltx2_status = if ltx2_request_ready {
        "mojo_request_ready"
    } else if ltx2_legacy_ready {
        "refhq_ready"
    } else if !ltx2_runner_ready {
        "runner_missing"
    } else {
        "creator_decode_runtime_missing"
    };
    let runners = json!([
        {
            "model": "lance_t2v",
            "status": "smoke_only",
            "runner": "serenitymojo/pipeline/lance_t2v_256_9f_dense_probe.mojo",
            "limit": "standalone pipeline artifact gate; not daemon job-backed",
        },
        {
            "model": "ltx2_t2v_av",
            "status": ltx2_status,
            "runner": RUNNER,
            "target_width": 1920,
            "target_height": 1088,
            "target_frame_count": 121,
            "modes": {
                "ltx2_staged_dev_smoke": {
                    "target_width": 384,
                    "target_height": 256,
                    "target_frame_count": 121,
                    "product": "stage-1 s1out",
                },
                "ltx2_refhq": {
                    "checkpoint": LTX2_REFHQ_CHECKPOINT,
                    "target_width": 1920,
                    "target_height": 1088,
                    "target_frame_count": 121,
                    "stage1_steps": 15,
                    "stage2_steps": 3,
                    "prompt_driven": true,
                    "processes": ["stage1", "upscaler", "stage2", "decode"],
                    "process_separated_decode": true,
                    "creator_cudnn_decode_runtime": "9.10.2",
                    "creator_cudnn_runtime_available": ltx2_decode_runtime_available(),
                    "conditioning_cache": {
                        "schema": LTX2_CONTEXT_SCHEMA,
                        "producer": LTX2_CONTEXT_SCRIPT,
                        "creator_revision": LTX2_CREATOR_REVISION,
                        "cache_before_video_model": true,
                    },
                    "tile_contract": { "spatial": [512, 64], "temporal": [64, 24] },
                    "accepted_sampler_parity": ltx2_sampler_parity,
                    "accepted_vae_parity": ltx2_vae_parity,
                    "accepted_audio_parity": ltx2_audio_parity,
                    "sampler_parity_report": LTX2_SAMPLER_PARITY_REPORT,
                    "vae_parity_report": LTX2_VAE_PARITY_REPORT,
                    "audio_parity_report": LTX2_AUDIO_PARITY_REPORT,
                },
                "ltx2_mojo_request": {
                    "runner": LTX2_MOJO_REQUEST_RUNNER,
                    "conditioning_runner": LTX2_MOJO_CONDITIONER,
                    "request_schema": "serenity.genparams.v1",
                    "status_schema": "serenity.ltx2.status.v1",
                    "result_schema": "serenity.ltx2.result.v1",
                    "asynchronous": true,
                    "ui_progress": true,
                    "authored_fields": [
                        "prompt", "negative", "width", "height", "frames",
                        "steps", "seed", "fps", "sampler", "scheduler",
                        "guidance_mode",
                        "caps_positive", "caps_negative", "noise_fixture",
                        "image_path", "image_strength",
                        "last_image_path", "last_image_strength",
                        "camera_motion", "video_path",
                        "video_strength", "video_mask_path",
                        "video_edit_mode", "video_edit_start",
                        "video_edit_end", "video_source_frames",
                        "include_audio", "audio_policy", "lora", "quant",
                        "post_upscale", "feature_id", "feature_weight",
                        "workflow_profile", "prompt_enhancer"
                    ],
                    "requires_authored_conditioning": false,
                    "automatic_conditioning": {
                        "available": ltx2_auto_conditioning_ready,
                        "backend": "mojo",
                        "missing": ltx2_conditioning_missing,
                        "manual_caps_override_supported": true,
                    },
                    "compiled_profile": ltx2_default_profile,
                    "supported_profiles": ltx2_profile_documents,
                    "checkpoint_workflows": ltx2_checkpoint_workflow_documents(),
                    "camera_motions": ltx2_camera_motion_documents(),
                    "quant_modes": [
                        {
                            "id": "bf16",
                            "label": "BF16",
                            "available": ltx2_bf16_available,
                            "checkpoint": ltx2_bf16_checkpoint,
                            "dtype_contract": "bf16_transformer_weights_bf16_activations_f32_reductions",
                        },
                        {
                            "id": "fp8",
                            "label": "FP8",
                            "available": true,
                            "dtype_contract": "fp8_transformer_bf16_activations_f32_reductions",
                        },
                        {
                            "id": "int4",
                            "label": "INT4",
                            "available": nonempty_file(&model_path(LTX2_REFHQ_INT4_SLAB)),
                            "dtype_contract": "int4_resident_w4a16_bf16_activations_f32_reductions",
                        }
                    ],
                    "default_checkpoint": ltx2_request_profile_registry().checkpoint,
                    "checkpoints": ltx2_checkpoint_documents(),
                    "post_upscalers": ltx2_post_upscaler_documents(),
                    "feature_adapters": crate::models::ltx2_feature_documents(),
                    "available": ltx2_request_ready,
                },
            },
            "target_fps": 24,
            "quant_modes": ["bf16", "fp8", "int4"],
            "quant_note": "request runner: dequantized dev BF16, native dev FP8, or W4A16 int4-resident. All execute BF16 activations with F32 reductions; W4A4 int4-compute is NOT integrated.",
            "limit": "staged is bounded smoke; refhq parity claims are admitted only from current machine-local Creator reports at the 0.999 bar",
        },
        {
            "model": "wan22_t2v",
            "status": if wan22_product_accepted { "quality_profile_ready" } else if wan22_ready { "gate_required" } else { "prerequisites_missing" },
            "runner": WAN22_T2V,
            "encode_runner": WAN22_ENCODE,
            "first_frame_encode_runners": [
                WAN22_FIRST_FRAME_LANDSCAPE,
                WAN22_FIRST_FRAME_PORTRAIT,
                WAN22_FIRST_FRAME_I2V_LANDSCAPE,
                WAN22_FIRST_FRAME_I2V_PORTRAIT,
            ],
            "missing": wan22_absent,
            "mode": "phase-isolated: wan22_encode_prompt -> optional wan22_encode_first_frame -> wan22_t2v",
            "artifact_root": model_path(WAN22_MODEL_ROOT).to_string_lossy(),
            "artifact_manifest": model_path(WAN22_ARTIFACT_MANIFEST).to_string_lossy(),
            "hf_revision": WAN22_HF_REVISION,
            "creator_revision": WAN22_CREATOR_REVISION,
            "product_gate": WAN22_PRODUCT_GATE,
            "accepted_video_parity": wan22_product_accepted,
            "target_width": WAN22_WIDTH,
            "target_height": WAN22_HEIGHT,
            "target_frame_count": WAN22_FRAMES,
            "native_profiles": [
                {
                    "width": WAN22_WIDTH,
                    "height": WAN22_HEIGHT,
                    "mode": "t2v",
                    "label": "T2V landscape",
                },
                {
                    "width": WAN22_PORTRAIT_WIDTH,
                    "height": WAN22_PORTRAIT_HEIGHT,
                    "mode": "t2v",
                    "label": "T2V portrait",
                },
                {
                    "width": WAN22_I2V_LANDSCAPE_WIDTH,
                    "height": WAN22_I2V_LANDSCAPE_HEIGHT,
                    "mode": "i2v_first_frame",
                    "label": "I2V 16:9 source-derived",
                },
                {
                    "width": WAN22_I2V_PORTRAIT_WIDTH,
                    "height": WAN22_I2V_PORTRAIT_HEIGHT,
                    "mode": "i2v_first_frame",
                    "label": "I2V 9:16 source-derived",
                }
            ],
            "default_steps": WAN22_DEFAULT_STEPS,
            "i2v_steps": WAN22_I2V_STEPS,
            "default_guidance": WAN22_DEFAULT_GUIDANCE,
            "modes": {
                "t2v": {
                    "available": wan22_product_accepted,
                    "steps": WAN22_DEFAULT_STEPS,
                    "shift": 5.0,
                },
                "i2v_first_frame": {
                    "available": wan22_product_accepted,
                    "source_images": 1,
                    "steps": WAN22_I2V_STEPS,
                    "shift": 5.0,
                    "size_policy": "creator max-area aspect-preserving 32-aligned output",
                    "conditioning": "process-isolated creator VAE encode, clean-frame replacement, and per-token zero timestep",
                },
                "last_frame": {
                    "available": false,
                    "reason": "official TI2V-5B accepts one input image; FLF2V requires different weights",
                },
                "vace_control": {
                    "available": false,
                    "reason": "VACE/control weights are not installed",
                },
                "lora": {
                    "available": wan22_product_accepted,
                    "max_count": 1,
                    "base_model": "Wan-AI/Wan2.2-TI2V-5B",
                    "format": "AI Toolkit/DiffusionModel block LoRA",
                    "merge": "BF16: additive delta on each RAM-staged block; FP8: one-time resident dequant-add-requant",
                }
            },
            "camera_motions": [
                { "id": "none", "label": "None", "prompt_suffix": "" },
                { "id": "static", "label": "Static", "prompt_suffix": ", static camera, locked off shot, no camera movement" },
                { "id": "focus_shift", "label": "Focus shift", "prompt_suffix": ", focus shift, rack focus, changing focal point" },
                { "id": "dolly_in", "label": "Dolly in", "prompt_suffix": ", dolly in, camera pushing forward, smooth forward movement" },
                { "id": "dolly_out", "label": "Dolly out", "prompt_suffix": ", dolly out, camera pulling back, smooth backward movement" },
                { "id": "dolly_left", "label": "Dolly left", "prompt_suffix": ", dolly left, camera tracking left, lateral movement" },
                { "id": "dolly_right", "label": "Dolly right", "prompt_suffix": ", dolly right, camera tracking right, lateral movement" },
                { "id": "jib_up", "label": "Jib up", "prompt_suffix": ", jib up, camera rising up, upward crane movement" },
                { "id": "jib_down", "label": "Jib down", "prompt_suffix": ", jib down, camera lowering down, downward crane movement" }
            ],
            "sampler": "Flow-UniPC order 2, predict_x0; creator-native T2V/I2V shift 5",
            "quant_modes": ["bf16", "fp8"],
            "quant_note": "BF16 copies the official converted transformer shards into one complete pinned-host store before sampling and is the admitted quality default; FP8 uses the persistent row-scaled E4M3 cache with BF16 on-use compute.",
            "note": "Creator-native TI2V-5B: T2V uses the 1280x704/704x1280 native shapes; I2V treats that as a max-area bucket and preserves source aspect on a 32-pixel grid. The common 1248x704/704x1248 I2V shapes are precompiled. All profiles use 121 frames, 24 fps, and CFG 5. The visual product gate uses Wan's recommended local-Qwen prompt extension; raw prompts remain valid but may be less detailed.",
            "limit": "requires the existing isolated GPU lease; machine-local product acceptance requires the pinned parity, visual, mux, timing, and peak-VRAM gate",
        },
        {
            "model": "bernini_r_t2v",
            "status": if bernini_product_accepted { "quality_profile_ready" } else if bernini_ready { "gate_required" } else { "prerequisites_missing" },
            "runner": BERNINI_T2V,
            "encode_runner": WAN22_ENCODE,
            "decode_runner": BERNINI_DECODE,
            "missing": bernini_absent,
            "mode": "three-process: wan22_encode_prompt -> bernini_t2v -> bernini_decode",
            "artifact_root": model_path(BERNINI_MODEL_ROOT).to_string_lossy(),
            "artifact_manifest": model_path(BERNINI_ARTIFACT_MANIFEST).to_string_lossy(),
            "hf_revision": BERNINI_HF_REVISION,
            "creator_revision": BERNINI_CREATOR_REVISION,
            "product_gate": BERNINI_PRODUCT_GATE,
            "accepted_video_parity": bernini_product_accepted,
            "target_width": BERNINI_WIDTH,
            "target_height": BERNINI_HEIGHT,
            "target_frame_count": BERNINI_FRAMES,
            "target_fps": BERNINI_FPS,
            "default_steps": BERNINI_DEFAULT_STEPS,
            "default_guidance": BERNINI_DEFAULT_GUIDANCE,
            "sampler": "Creator UniPC bh2 flow, shift 5; APG 4.0/3.2",
            "quant_modes": ["fp8"],
            "quant_note": "persistent per-row FP8 E4M3 dual-expert caches; exactly one block resident",
            "note": "Creator profile: 848x480, 81 frames, 16 fps, 40 steps. High and low A14B experts and the standard-Wan VAE execute in isolated Mojo residency phases.",
            "limit": "fail-closed until pinned creator parity, cache provenance, VAE reuse, visual quality, mux, timing, and peak-VRAM evidence all pass",
        },
        {
            "model": "scail2_animation",
            "status": if scail2_product_accepted { "quality_profile_ready" } else if scail2_ready { "gate_required" } else { "prerequisites_missing" },
            "runner": SCAIL2_ANIMATION,
            "stage_runner": SCAIL2_STAGE,
            "prompt_runner": SCAIL2_ENCODE_PROMPT,
            "clip_runner": SCAIL2_ENCODE_CLIP,
            "vae_runner": SCAIL2_ENCODE_VAE,
            "decode_runner": SCAIL2_DECODE,
            "cache_runner": SCAIL2_PREPARE_CACHE,
            "missing": scail2_absent,
            "mode": "seven-process automatic conditioning -> denoise -> decode/audio mux",
            "artifact_root": model_path(SCAIL2_MOJO_ROOT).to_string_lossy(),
            "official_root": model_path(SCAIL2_OFFICIAL_ROOT).to_string_lossy(),
            "source_commit": SCAIL2_SOURCE_COMMIT,
            "model_revision": SCAIL2_MODEL_REVISION,
            "product_gate": SCAIL2_PRODUCT_GATE,
            "accepted_video_parity": scail2_product_accepted,
            "target_width": SCAIL2_WIDTH,
            "target_height": SCAIL2_HEIGHT,
            "target_frame_count": SCAIL2_FRAMES,
            "target_fps": SCAIL2_FPS,
            "default_steps": SCAIL2_STEPS,
            "default_guidance": SCAIL2_GUIDANCE,
            "sampler": "Creator UniPC order 2, shift 3, CFG 5",
            "quant_modes": ["fp8"],
            "audio": "preserves audio from the driving video when present",
            "inputs": ["reference_image", "reference_mask", "driving_video", "driving_mask_video"],
            "note": "All stage, UMT5, CLIP, VAE encode, denoise, VAE decode, MP4, and optional audio artifacts are created automatically by the server.",
            "limit": "single creator-gated 896x512x65 segment; inputs must be supplied by the user",
        },
    ]);
    json!({
        "schema": "serenity.video_status.v1",
        "endpoint": "/v1/video",
        "state": state,
        "readiness_label": if ltx2_request_ready { "experimental_request_runner_ready" } else if any_ready { "bounded_daemon_smoke" } else { "build_required" },
        "accepted": false,
        "backend": BACKEND_NAME,
        "control_plane": "serenity-server",
        "model": "",
        "resident": "",
        "mp4": "",
        "frame_count": 0,
        "duration": 0.0,
        "audio": false,
        "arms_ready": {
            "ltx2_t2v_av": ltx2_ready,
            "wan22_t2v": wan22_ready,
            "bernini_r_t2v": bernini_product_accepted,
            "scail2_animation": scail2_product_accepted,
        },
        "non_acceptance_reason": "bounded smoke wiring is not full reference UI video parity; artifact acceptance requires frame_count, duration, muxing, audio behavior, timings, and VRAM evidence",
        "probe_endpoint": "/v1/video/probe?path=<mp4>",
        "candidate_runners": runners,
    })
}

/// GET /v1/video — readiness contract.
pub async fn get_video() -> Response {
    json_resp(StatusCode::OK, &readiness_doc())
}

fn normalize_ltx2_prompt_fields(body: &Value) -> Value {
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
struct Ltx2CameraMotionSpec {
    id: &'static str,
    label: &'static str,
    prompt_suffix: &'static str,
    adapter_filename: Option<&'static str>,
}

const LTX2_CAMERA_MOTIONS: [Ltx2CameraMotionSpec; 9] = [
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

fn ltx2_camera_motion_spec(id: &str) -> Option<Ltx2CameraMotionSpec> {
    LTX2_CAMERA_MOTIONS
        .iter()
        .copied()
        .find(|candidate| candidate.id == id)
}

fn ltx2_camera_motion_documents() -> Vec<Value> {
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

fn normalized_ltx2_camera_motion_request(body: &Value) -> Result<Value, String> {
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
        object.insert(
            "creator_camera_motion_applied".to_string(),
            json!(true),
        );
    }
    if let Some(filename) = spec.adapter_filename {
        let adapter_path = model_path(&format!("loras/{filename}"));
        let rows = object
            .entry("lora".to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| "LTX2 'lora' must be an array".to_string())?;
        rows.retain(|row| {
            row.get("source").and_then(Value::as_str) != Some("camera_control")
        });
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

fn normalized_wan22_camera_motion_request(body: &Value) -> Result<Value, String> {
    let mut normalized = body.clone();
    let motion = body
        .get("camera_motion")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .trim();
    let suffix = match motion {
        "none" => "",
        "static" => ", static camera, locked off shot, no camera movement",
        "focus_shift" => ", focus shift, rack focus, changing focal point",
        "dolly_in" => ", dolly in, camera pushing forward, smooth forward movement",
        "dolly_out" => ", dolly out, camera pulling back, smooth backward movement",
        "dolly_left" => ", dolly left, camera tracking left, lateral movement",
        "dolly_right" => ", dolly right, camera tracking right, lateral movement",
        "jib_up" => ", jib up, camera rising up, upward crane movement",
        "jib_down" => ", jib down, camera lowering down, downward crane movement",
        other => return Err(format!("unsupported Wan camera_motion '{other}'")),
    };
    let object = normalized
        .as_object_mut()
        .ok_or_else(|| "Wan request must be an object".to_string())?;
    object.insert("camera_motion".to_string(), json!(motion));
    if !suffix.is_empty()
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
        object.insert("prompt".to_string(), json!(format!("{prompt}{suffix}")));
        object.insert(
            "creator_camera_motion_suffix".to_string(),
            json!(suffix),
        );
        object.insert(
            "creator_camera_motion_applied".to_string(),
            json!(true),
        );
    }
    Ok(normalized)
}

fn normalized_ltx2_checkpoint_workflow_request(body: &Value) -> Result<Value, String> {
    let mut normalized = body.clone();
    let checkpoint = body
        .get("checkpoint")
        .and_then(Value::as_str)
        .unwrap_or("");
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

/// POST /v1/video — dispatch on `model`: `"ltx2"` (default) = the bounded LTX2
/// staged smoke; `"wan22"` = Wan2.2 5B; `"wan22_a14b"` = the bounded A14B
/// LoRA preview; `"bernini"` = the gated Bernini-R;
/// `"scail2"` = automatic character-animation orchestration.
pub async fn post_video(State(st): State<AppState>, body: String) -> Response {
    let mut b: Value = serde_json::from_str::<Value>(&body)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({}));
    let model = b
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("ltx2")
        .to_string();
    if model != "ltx2"
        && model != "wan22"
        && model != "wan22_a14b"
        && model != "bernini"
        && model != "scail2"
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "unsupported video model '{model}'; use ltx2, wan22, wan22_a14b, bernini, or scail2"
            ),
        );
    }
    let is_ltx2_mojo_request =
        model == "ltx2" && b.get("runner").and_then(Value::as_str) == Some("ltx2_mojo_request");
    if is_ltx2_mojo_request {
        b = normalize_ltx2_prompt_fields(&b);
        b = match normalized_ltx2_camera_motion_request(&b) {
            Ok(value) => value,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
        b = match normalized_ltx2_checkpoint_workflow_request(&b) {
            Ok(value) => value,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
        b = match normalized_ltx2_video_edit_request(&b) {
            Ok(value) => value,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
        if let Err(error) = validate_ltx2_mojo_request(&b) {
            return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
        }
        let edit_mode = b
            .get("video_edit_mode")
            .and_then(Value::as_str)
            .unwrap_or("standard");
        let profile = ltx2_request_profile_for_mode(
            b["width"].as_i64().unwrap_or(0),
            b["height"].as_i64().unwrap_or(0),
            b["frames"].as_i64().unwrap_or(0),
            b["fps"].as_f64().unwrap_or(0.0),
            edit_mode,
        )
        .expect("validated LTX2 request must resolve to an admitted profile");
        if !ltx2_profile_runner_available(&profile) {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "LTX2 runtime-geometry Mojo runner is unavailable for {}x{}, {} frames at {} FPS: {}",
                    profile.width, profile.height, profile.frames, profile.fps,
                    LTX2_MOJO_REQUEST_RUNNER,
                ),
            );
        }
    }
    if model == "wan22" {
        b = match normalized_wan22_camera_motion_request(&b) {
            Ok(value) => value,
            Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
        };
    }
    // Fail closed before taking the GPU lease or evicting an idle image model.
    // A partially installed or non-accepted video arm is not a GPU operation.
    if model == "wan22" {
        let missing = wan22_missing();
        if !missing.is_empty() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "wan22 runtime prerequisites missing: {}",
                    missing.join(", ")
                ),
            );
        }
        if !wan22_product_gate_passed() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Wan2.2 is installed but its machine-local high-quality product gate has not passed",
            );
        }
    }
    if model == "wan22_a14b" {
        let missing = wan22_a14b_missing();
        if !missing.is_empty() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2 A14B preview prerequisites missing: {}",
                    missing.join(", ")
                ),
            );
        }
        if let Err(error) = validate_wan22_a14b_request(&b) {
            return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
        }
    }
    if model == "bernini" {
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
    }
    if model == "scail2" {
        let missing = scail2_missing();
        if !missing.is_empty() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!("SCAIL-2 prerequisites missing: {}", missing.join(", ")),
            );
        }
        if !scail2_product_gate_passed() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "SCAIL-2 is installed but its pinned full-animation product gate is not current",
            );
        }
        if b.get("prompt")
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
        {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "scail2: 'prompt' is required",
            );
        }
        for key in [
            "reference_image",
            "reference_mask",
            "driving_video",
            "driving_mask_video",
        ] {
            if b.get(key)
                .and_then(Value::as_str)
                .is_none_or(|value| value.trim().is_empty())
            {
                return err_detail(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &format!("scail2: '{key}' is required"),
                );
            }
        }
    }
    // Cross-path single-GPU lease (audit L3): a video render is minutes of GPU
    // work in a subprocess; it must not co-run with a generate/caption/magic
    // job on a 16GB card. Held (RAII) across the whole arm; 409 if busy.
    let gpu_tag = crate::gpu_lock::next_tag("video");
    let gpu = match crate::gpu_lock::try_acquire(&st.gpu_owner, "video", &gpu_tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                axum::Json(crate::gpu_lock::gpu_busy_conflict_report("video", &cur)),
            )
                .into_response();
        }
    };
    // The GPU lease excludes an active image job, but an IDLE Mojo worker can
    // still retain its resident model (measured ZImage floor: ~14.5 GiB). Kill
    // and reap that worker before launching LTX/Wan so switching from an image
    // model to video is as safe as switching between image-model families.
    let (evict_tx, evict_rx) = std::sync::mpsc::channel();
    if st.ctl.send(crate::DriverCtl::EvictIdle(evict_tx)).is_err() {
        return err_detail(
            StatusCode::SERVICE_UNAVAILABLE,
            "image worker driver unavailable before video launch",
        );
    }
    match evict_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(true) => {}
        Ok(false) => {
            return err_detail(
                StatusCode::CONFLICT,
                "image worker became active before video launch",
            );
        }
        Err(_) => {
            return err_detail(
                StatusCode::SERVICE_UNAVAILABLE,
                "timed out evicting idle image worker before video launch",
            );
        }
    }
    if is_ltx2_mojo_request {
        return start_ltx2_mojo_request(&st, &b, gpu);
    }
    match model.as_str() {
        "ltx2" => post_video_ltx2(&st, &b),
        "wan22" => post_video_wan22(&st, &b),
        "wan22_a14b" => post_video_wan22_a14b(&st, &b),
        "bernini" => post_video_bernini(&st, &b),
        "scail2" => post_video_scail2(&st, &b),
        _ => unreachable!("video model validated before GPU acquisition"),
    }
}

fn ltx2_feature_request(body: &Value) -> Result<Option<Value>, String> {
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

fn normalized_ltx2_feature_request(body: &Value) -> Result<Value, String> {
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

fn normalized_ltx2_video_edit_request(body: &Value) -> Result<Value, String> {
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

fn resolve_ltx2_request_checkpoint(
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
struct Ltx2DistillationAdapter {
    name: String,
    path: std::path::PathBuf,
    weight: f64,
    stage1_weight: Option<f64>,
    stage2_weight: Option<f64>,
    source: &'static str,
}

fn is_official_ltx2_dev_checkpoint_name(name: &str) -> bool {
    matches!(
        name.strip_suffix(".safetensors").unwrap_or(name),
        "ltx-2.3-22b-dev"
            | LTX2_REFHQ_CHECKPOINT
            | LTX2_REFHQ_BF16_CHECKPOINT
    )
}

fn is_official_ltx2_dev_checkpoint(checkpoint: &crate::models::ResolvedCheckpoint) -> bool {
    is_official_ltx2_dev_checkpoint_name(&checkpoint.name)
}

/// Resolve the sampling adapter that belongs to the selected checkpoint.
///
/// Ordinary authored LoRAs remain in `lora[]`. A row explicitly marked
/// `role=distillation` replaces the official creator adapter. The official
/// adapter is only implicit for the two official dev checkpoints; arbitrary
/// finetunes never inherit it by accident, and directly distilled full
/// checkpoints never receive a second distillation delta.
fn resolve_ltx2_distillation_adapter(
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
            let adapter = workflow
                .get("distillation_adapter")
                .ok_or_else(|| format!("creator workflow '{workflow_id}' has no distillation adapter"))?;
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

fn resolve_ltx2_retake_checkpoint(
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

fn ltx2_checkpoint_has_creator_edit_components(path: &std::path::Path) -> bool {
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

fn validate_ltx2_mojo_request(body: &Value) -> Result<(), String> {
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
    let checkpoint_path = model_path(&checkpoint_profile.path);
    if !nonempty_file(&checkpoint_path) {
        return Err(format!(
            "LTX2 checkpoint '{}' is registered but missing: {}",
            checkpoint_profile.id,
            checkpoint_path.display()
        ));
    }
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
    let checkpoint = resolve_ltx2_request_checkpoint(body)?;
    if quant == "int4" {
        let checkpoint_name = checkpoint
            .name
            .strip_suffix(".safetensors")
            .unwrap_or(&checkpoint.name);
        if checkpoint_name != LTX2_REFHQ_CHECKPOINT {
            return Err(format!(
                "LTX2 int4 uses a checkpoint-specific slab and is only available for '{LTX2_REFHQ_CHECKPOINT}'; select fp8 or bf16 for '{}'",
                checkpoint.name
            ));
        }
        let slab = model_path(LTX2_REFHQ_INT4_SLAB);
        if !slab.is_file() {
            return Err(format!(
                "LTX2 request selected int4 but the dev-model slab is missing: {}",
                slab.display()
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
    let selected_workflow = ltx2_checkpoint_workflow(&checkpoint.name)
        .filter(|profile| profile.get("id").and_then(Value::as_str) == Some(workflow_profile));
    if !workflow_profile.is_empty() && selected_workflow.is_none() {
        return Err(format!(
            "LTX2 workflow_profile '{workflow_profile}' is not registered for checkpoint '{}'",
            checkpoint.name
        ));
    }
    if let Some(workflow) = selected_workflow {
        let expected_guidance = workflow
            .get("guidance_mode")
            .and_then(Value::as_str)
            .unwrap_or("");
        let expected_sampler = workflow.get("sampler").and_then(Value::as_str).unwrap_or("");
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
        let enhancer = workflow
            .get("prompt_enhancer")
            .ok_or_else(|| {
                format!(
                    "LTX2 creator workflow '{workflow_profile}' has no prompt enhancer"
                )
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
        return Err(
            "LTX2 last-frame keyframe interpolation cannot use Retake/Extend".to_string(),
        );
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
    let _ = resolve_ltx2_distillation_adapter(body, &checkpoint)?;
    let _ = ltx2_feature_request(body)?;
    Ok(())
}

fn validate_wan22_a14b_request(body: &Value) -> Result<(), String> {
    if body
        .get("prompt")
        .and_then(Value::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err("Wan2.2 A14B preview requires a non-empty prompt".to_string());
    }
    for (key, required) in [
        ("width", WAN22_A14B_WIDTH),
        ("height", WAN22_A14B_HEIGHT),
        ("frames", WAN22_A14B_FRAMES),
        ("steps", WAN22_A14B_STEPS),
        ("fps", WAN22_A14B_FPS),
    ] {
        let actual = body.get(key).and_then(Value::as_i64).unwrap_or(required);
        if actual != required {
            return Err(format!(
                "Wan2.2 A14B preview requires {key}={required}; requested {actual}"
            ));
        }
    }
    let guidance = body
        .get("guidance")
        .and_then(Value::as_f64)
        .unwrap_or(WAN22_A14B_GUIDANCE);
    if !guidance.is_finite() || (guidance - WAN22_A14B_GUIDANCE).abs() > f64::EPSILON {
        return Err(format!(
            "Wan2.2 A14B preview requires CFG {WAN22_A14B_GUIDANCE}"
        ));
    }
    let _ = ltx2_seed(body)?;
    let _ = wan22_a14b_lora(body)?;
    Ok(())
}

fn ltx2_requested_post_upscale(body: &Value) -> Result<Option<(String, i64)>, String> {
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

fn run_realesrgan_video_post_upscale<F>(
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

fn remux_ltx2_source_audio(
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
fn stage_ltx2_creator_source_audio(
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
fn remux_ltx2_generated_audio(
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

fn start_ltx2_mojo_request(
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
            request.insert(
                "distillation_adapter".to_string(),
                adapter_document,
            );
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
fn post_video_ltx2(st: &AppState, b: &Value) -> Response {
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
fn post_video_ltx2_refhq(st: &AppState, b: &Value) -> Response {
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
fn wan22_command(bin_abs: &std::path::Path) -> std::process::Command {
    let root = repo_root();
    let pixi_lib = root.join(".pixi/envs/default/lib");
    let mut preload = [
        "libcudnn_graph.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_tensor_ir.so.9",
        "libcudnn_heuristic.so.9",
    ]
    .into_iter()
    .map(|name| pixi_lib.join(name))
    .collect::<Vec<_>>();
    preload.extend([
        std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"),
        std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvidia-nvvm70.so.4"),
    ]);
    if let Ok(entries) = std::fs::read_dir("/usr/lib/x86_64-linux-gnu") {
        if let Some(gpucomp) = entries.flatten().map(|entry| entry.path()).find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libnvidia-gpucomp.so."))
        }) {
            preload.push(gpucomp);
        }
    }
    if let Some(existing) = std::env::var_os("LD_PRELOAD") {
        preload.extend(std::env::split_paths(&existing));
    }
    let ram_cache_ready = std::fs::create_dir_all(WAN22_CUDA_CACHE).is_ok();
    let mut command = std::process::Command::new(bin_abs);
    command
        .current_dir(root)
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .env("LD_PRELOAD", std::env::join_paths(preload).unwrap_or_default())
        .env("CUDA_CACHE_PATH", WAN22_CUDA_CACHE);
    if !ram_cache_ready {
        command.env("CUDA_CACHE_DISABLE", "1");
    }
    command
}

/// First `*.mp4` produced under `dir` (newest by mtime). "" if none — the runner
/// owns the output name, so we glob rather than hardcode it.
fn find_mp4(dir: &std::path::Path) -> String {
    let mut best: Option<(std::time::SystemTime, String)> = None;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for ent in rd.flatten() {
            let p = ent.path();
            let is_mp4 = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("mp4"))
                .unwrap_or(false);
            if !is_mp4 {
                continue;
            }
            let mtime = ent
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            let path = p.to_string_lossy().into_owned();
            if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                best = Some((mtime, path));
            }
        }
    }
    best.map(|(_, p)| p).unwrap_or_default()
}

/// Count `frame_*.png` written by wan22_t2v under `dir`.
fn count_wan22_frames(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("frame_") && n.ends_with(".png"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

/// wan22_t2v writes `frame_%d.png` frames + prints a manual ffmpeg mux command
/// (it does NOT produce an mp4 itself). Do that mux here → `<dir>/wan22_t2v.mp4`.
/// Returns the mp4 path on success. Mirrors the runner's exact ffmpeg args (24fps,
/// libx264, yuv420p, +faststart).
fn mux_wan22_frames(dir: &std::path::Path, fps: i64) -> Result<String, String> {
    let pattern = dir.join("frame_%d.png");
    let mp4 = dir.join("wan22_t2v.mp4");
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            &fps.to_string(),
            "-start_number",
            "0",
            "-i",
            &pattern.to_string_lossy(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            &mp4.to_string_lossy(),
        ])
        .output()
        .map_err(|e| format!("ffmpeg spawn: {e}"))?;
    if out.status.success() && mp4.exists() {
        Ok(mp4.to_string_lossy().into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Creator `best_output_size` for TI2V-5B I2V. The CLI's 1280x704/704x1280
/// values are maximum-area buckets; the actual output follows the source
/// aspect ratio on a 32-pixel grid. Keep this byte-for-byte equivalent to
/// `wan/utils/utils.py::best_output_size` instead of stretching every source
/// into the two T2V shapes.
fn wan22_creator_i2v_size(source_width: u32, source_height: u32) -> (i64, i64) {
    const ALIGN: f64 = 32.0;
    const EXPECTED_AREA: f64 = (WAN22_WIDTH * WAN22_HEIGHT) as f64;
    let ratio = source_width as f64 / source_height as f64;
    let output_width = (EXPECTED_AREA * ratio).sqrt();
    let output_height = EXPECTED_AREA / output_width;

    let width_first = (output_width / ALIGN).floor() * ALIGN;
    let height_from_width = (EXPECTED_AREA / width_first / ALIGN).floor() * ALIGN;
    let ratio_width_first = width_first / height_from_width;

    let height_first = (output_height / ALIGN).floor() * ALIGN;
    let width_from_height = (EXPECTED_AREA / height_first / ALIGN).floor() * ALIGN;
    let ratio_height_first = width_from_height / height_first;

    let distortion_width =
        (ratio / ratio_width_first).max(ratio_width_first / ratio);
    let distortion_height =
        (ratio / ratio_height_first).max(ratio_height_first / ratio);
    if distortion_width < distortion_height {
        (width_first as i64, height_from_width as i64)
    } else {
        (width_from_height as i64, height_first as i64)
    }
}

/// Wan2.2 T2V arm — two-process orchestration (encode umt5 conds → t2v → mp4).
/// Synchronous, multi-minute blocking render (matches the LTX2 arm's convention). All
/// validation happens BEFORE the binary-presence check + spawn, so a bad request
/// never launches the GPU.
fn post_video_wan22(st: &AppState, b: &Value) -> Response {
    let s = |k: &str, d: &str| b.get(k).and_then(|v| v.as_str()).unwrap_or(d).to_string();
    let prompt = s("prompt", "").trim().to_string();
    if prompt.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "wan22: 'prompt' is required",
        );
    }
    let requested_negative = s("negative_prompt", "").trim().to_string();
    let neg_prompt = if requested_negative.is_empty() {
        WAN22_DEFAULT_NEGATIVE.to_string()
    } else {
        requested_negative
    };
    let image_path = s("image_path", "").trim().to_string();
    let is_i2v = !image_path.is_empty();
    if is_i2v && !std::path::Path::new(&image_path).is_file() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Wan TI2V first-frame image not found: {image_path}"),
        );
    }
    if b.get("last_image_path")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Wan2.2-TI2V-5B officially accepts one input image; last-frame conditioning requires a different FLF2V model and is not faked",
        );
    }
    for unsupported in ["video_path", "vace_path", "control_video_path", "motion_track_path"] {
        if b.get(unsupported)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
        {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2-TI2V-5B does not support '{unsupported}' with the installed weights; VACE/control/motion models are not installed"
                ),
            );
        }
    }
    let lora = match wan22_ti2v5b_lora(b) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    // Both official BF16 storage and the row-scaled E4M3 cache are explicit
    // runtime choices. Never silently alias a BF16 UI selection to FP8.
    let quant = s("quant", "bf16");
    if quant != "fp8" && quant != "bf16" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "wan22_t2v quant '{quant}' is unsupported; choose bf16 or fp8"
            ),
        );
    }
    let frames = b
        .get("frames")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_FRAMES);
    if !(1..=121).contains(&frames) {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'frames' out of range [1..121]",
        );
    }
    if frames != WAN22_FRAMES {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "wan22_t2v is comptime-compiled for {WAN22_FRAMES} frames ({WAN22_WIDTH}x{WAN22_HEIGHT}); requested {frames} requires a rebuild"
            ),
        );
    }
    let width = b
        .get("width")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_WIDTH);
    let height = b
        .get("height")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_HEIGHT);
    let fps = b.get("fps").and_then(|v| v.as_i64()).unwrap_or(24);
    let t2v_landscape = width == WAN22_WIDTH && height == WAN22_HEIGHT;
    let t2v_portrait =
        width == WAN22_PORTRAIT_WIDTH && height == WAN22_PORTRAIT_HEIGHT;
    let i2v_landscape =
        width == WAN22_I2V_LANDSCAPE_WIDTH && height == WAN22_I2V_LANDSCAPE_HEIGHT;
    let i2v_portrait =
        width == WAN22_I2V_PORTRAIT_WIDTH && height == WAN22_I2V_PORTRAIT_HEIGHT;
    if fps != WAN22_FPS {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Wan2.2 creator profile requires {WAN22_FPS} fps; requested {fps}"
            ),
        );
    }
    if is_i2v {
        let (source_width, source_height) =
            match image::image_dimensions(std::path::Path::new(&image_path)) {
                Ok(dimensions) => dimensions,
                Err(error) => {
                    return err_detail(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        &format!(
                            "cannot inspect Wan I2V source dimensions for creator sizing: {error}"
                        ),
                    )
                }
            };
        let creator_size = wan22_creator_i2v_size(source_width, source_height);
        if (width, height) != creator_size {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2 creator I2V sizing for source {source_width}x{source_height} is {}x{}, not {width}x{height}",
                    creator_size.0, creator_size.1
                ),
            );
        }
        if !(t2v_landscape || t2v_portrait || i2v_landscape || i2v_portrait) {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2 creator-derived I2V profile {width}x{height} is not precompiled on this installation"
                ),
            );
        }
    } else if !(t2v_landscape || t2v_portrait) {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Wan2.2 T2V native profiles are {WAN22_WIDTH}x{WAN22_HEIGHT} and {WAN22_PORTRAIT_WIDTH}x{WAN22_PORTRAIT_HEIGHT}; requested {width}x{height}"
            ),
        );
    }
    let steps = b
        .get("steps")
        .and_then(|v| v.as_i64())
        .unwrap_or(if is_i2v {
            WAN22_I2V_STEPS
        } else {
            WAN22_DEFAULT_STEPS
        });
    let required_steps = if is_i2v {
        WAN22_I2V_STEPS
    } else {
        WAN22_DEFAULT_STEPS
    };
    if steps != required_steps {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Wan2.2 {} creator profile requires exactly {required_steps} steps",
                if is_i2v { "I2V" } else { "T2V" }
            ),
        );
    }
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };
    let guidance = b
        .get("guidance")
        .and_then(|v| v.as_f64())
        .unwrap_or(WAN22_DEFAULT_GUIDANCE);
    if !guidance.is_finite() || (guidance - WAN22_DEFAULT_GUIDANCE).abs() > f64::EPSILON {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("wan22_t2v high-quality profile requires CFG {WAN22_DEFAULT_GUIDANCE}"),
        );
    }

    // Binary presence gate (cwd-relative, like the LTX2 runner). 422 naming absent.
    let missing = wan22_missing();
    if !missing.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "wan22 runtime prerequisites missing: {}",
                missing.join(", ")
            ),
        );
    }
    if !wan22_product_gate_passed() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Wan2.2 machine-local high-quality product gate is not current",
        );
    }

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    let _ = std::fs::create_dir_all(&out_dir);
    let conds = out_dir.join("wan22_conds.safetensors");
    let conds_s = conds.to_string_lossy().into_owned();
    let out_dir_s = out_dir.to_string_lossy().into_owned();
    let enc_log = out_dir.join("wan22_encode.log");
    let first_frame_log = out_dir.join("wan22_first_frame_encode.log");
    let t2v_log = out_dir.join("wan22_t2v.log");

    let abs_encode = repo_path(WAN22_ENCODE);
    let runner = match (width, height) {
        (WAN22_WIDTH, WAN22_HEIGHT) => WAN22_T2V,
        (WAN22_PORTRAIT_WIDTH, WAN22_PORTRAIT_HEIGHT) => WAN22_T2V_PORTRAIT,
        (WAN22_I2V_LANDSCAPE_WIDTH, WAN22_I2V_LANDSCAPE_HEIGHT) => {
            WAN22_I2V_LANDSCAPE
        }
        (WAN22_I2V_PORTRAIT_WIDTH, WAN22_I2V_PORTRAIT_HEIGHT) => {
            WAN22_I2V_PORTRAIT
        }
        _ => unreachable!("Wan profile validated before runner selection"),
    };
    let abs_t2v = repo_path(runner);
    let abs_first_frame = if is_i2v {
        let binary = match (width, height) {
            (WAN22_WIDTH, WAN22_HEIGHT) => WAN22_FIRST_FRAME_LANDSCAPE,
            (WAN22_PORTRAIT_WIDTH, WAN22_PORTRAIT_HEIGHT) => {
                WAN22_FIRST_FRAME_PORTRAIT
            }
            (WAN22_I2V_LANDSCAPE_WIDTH, WAN22_I2V_LANDSCAPE_HEIGHT) => {
                WAN22_FIRST_FRAME_I2V_LANDSCAPE
            }
            (WAN22_I2V_PORTRAIT_WIDTH, WAN22_I2V_PORTRAIT_HEIGHT) => {
                WAN22_FIRST_FRAME_I2V_PORTRAIT
            }
            _ => unreachable!("Wan I2V profile validated before encoder selection"),
        };
        Some(repo_path(binary))
    } else {
        None
    };
    let lora_path = lora
        .as_ref()
        .map(|(path, _, _, _)| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| "-".to_string());
    let lora_weight = lora.as_ref().map(|(_, weight, _, _)| *weight).unwrap_or(1.0);

    // ── Step A: encode umt5 conds (prompt strings passed directly as argv) ──
    let ta = std::time::Instant::now();
    let mut encode = wan22_command(&abs_encode);
    encode.args([&prompt, &neg_prompt, &conds_s]);
    let enc = run_logged_with_gpu_peak(&mut encode, &enc_log);
    let enc_secs = ta.elapsed().as_secs_f64();
    let (enc_rc, encode_peak_vram_mib) = match enc {
        Ok(measured) => measured,
        Err(e) => {
            let _ = std::fs::write(&enc_log, format!("spawn failed: {e}"));
            (-1, None)
        }
    };
    if enc_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "encode", "encode_exit_code": enc_rc,
                "encode_log": enc_log.to_string_lossy(), "out_dir": out_dir_s,
                "encode_seconds": enc_secs, "encode_peak_vram_mib": encode_peak_vram_mib,
                "error": "wan22_encode_prompt failed; inspect encode_log",
            }),
        );
    }

    // ── Step B: process-isolated creator first-frame VAE encode. ───────────
    // A same-process VAE->DiT handoff leaves allocator residue/fragmentation
    // and OOMs the exact BF16 stream on a 24 GB card. Persist only the small
    // first latent, then let process exit reclaim every VAE allocation.
    let first_frame_cache = out_dir.join("wan22_first_frame.safetensors");
    let first_frame_cache_s = first_frame_cache.to_string_lossy().into_owned();
    let (first_frame_arg, first_frame_rc, first_frame_secs, first_frame_peak_vram_mib) =
        if let Some(first_frame_binary) = abs_first_frame.as_ref() {
            let started = std::time::Instant::now();
            let mut encode_first = wan22_command(first_frame_binary);
            encode_first.args([&image_path, &first_frame_cache_s]);
            let measured = run_logged_with_gpu_peak(
                &mut encode_first,
                &first_frame_log,
            );
            let seconds = started.elapsed().as_secs_f64();
            let (code, peak) = match measured {
                Ok(result) => result,
                Err(error) => {
                    let _ = std::fs::write(
                        &first_frame_log,
                        format!("spawn failed: {error}"),
                    );
                    (-1, None)
                }
            };
            (first_frame_cache_s.clone(), code, seconds, peak)
        } else {
            (String::new(), 0, 0.0, None)
        };
    if first_frame_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "first_frame_encode",
                "encode_exit_code": enc_rc, "first_frame_exit_code": first_frame_rc,
                "encode_log": enc_log.to_string_lossy(),
                "first_frame_log": first_frame_log.to_string_lossy(),
                "out_dir": out_dir_s, "conds": conds_s,
                "encode_seconds": enc_secs,
                "first_frame_seconds": first_frame_secs,
                "encode_peak_vram_mib": encode_peak_vram_mib,
                "first_frame_peak_vram_mib": first_frame_peak_vram_mib,
                "error": "wan22_encode_first_frame failed; inspect first_frame_log",
            }),
        );
    }

    // ── Step C: creator-native 121-frame render in the selected precision ──
    let tb = std::time::Instant::now();
    let mut t2v = wan22_command(&abs_t2v);
    t2v.arg(&conds_s)
        .arg(&out_dir_s)
        .arg(frames.to_string())
        .arg(steps.to_string())
        .arg(seed.to_string())
        .arg(format!("{guidance}"))
        .arg("1")
        .arg(&first_frame_arg)
        .arg(&lora_path)
        .arg(format!("{lora_weight}"))
        .arg(&quant);
    let t2v = run_logged_with_gpu_peak(&mut t2v, &t2v_log);
    let t2v_secs = tb.elapsed().as_secs_f64();
    let (t2v_rc, t2v_peak_vram_mib) = match t2v {
        Ok(measured) => measured,
        Err(e) => {
            let _ = std::fs::write(&t2v_log, format!("spawn failed: {e}"));
            (-1, None)
        }
    };
    let total_wall = enc_secs + first_frame_secs + t2v_secs;
    if t2v_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "t2v",
                "encode_exit_code": enc_rc, "first_frame_exit_code": first_frame_rc,
                "t2v_exit_code": t2v_rc,
                "encode_log": enc_log.to_string_lossy(),
                "first_frame_log": first_frame_log.to_string_lossy(),
                "t2v_log": t2v_log.to_string_lossy(),
                "out_dir": out_dir_s, "conds": conds_s,
                "encode_seconds": enc_secs, "first_frame_seconds": first_frame_secs,
                "t2v_seconds": t2v_secs, "total_wall_seconds": total_wall,
                "encode_peak_vram_mib": encode_peak_vram_mib,
                "first_frame_peak_vram_mib": first_frame_peak_vram_mib,
                "t2v_peak_vram_mib": t2v_peak_vram_mib,
                "error": "wan22_t2v failed; inspect t2v_log",
            }),
        );
    }

    // ── Step D: mux the frame_*.png the runner wrote into an mp4 (24fps). ──
    let frames_written = count_wan22_frames(&out_dir);
    let (mp4, mux) = if frames_written > 0 {
        match mux_wan22_frames(&out_dir, 24) {
            Ok(p) => (p, "muxed".to_string()),
            Err(e) => (String::new(), format!("mux_failed: {e}")),
        }
    } else {
        // fallback in case a runner ever writes an mp4 directly
        (find_mp4(&out_dir), "no_frames_written".to_string())
    };
    let mux_ok = mux == "muxed";
    let probe = if mux_ok {
        probe_video_path(&mp4).ok()
    } else {
        None
    };
    let artifact_ok = probe.as_ref().is_some_and(|value| {
        probe_matches_video_profile(
            value,
            width,
            height,
            WAN22_FRAMES,
            WAN22_FPS,
            false,
        )
    }) && frames_written == WAN22_FRAMES as usize;
    let parity_ok = wan22_product_gate_passed();
    json_resp(
        if artifact_ok {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &json!({
            "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
            "backend": BACKEND_NAME, "control_plane": "serenity-server",
            "resident": if quant == "bf16" { "bf16_native_shards_pinned_host" } else { "fp8_e4m3_cached" },
            "mode": if is_i2v { "i2v_first_frame" } else { "t2v" },
            "readiness_label": if parity_ok { "quality_profile_ready" } else { "product_gate_required" },
            "accepted_video_artifact": artifact_ok, "accepted_video_parity": parity_ok,
            "target_width": width, "target_height": height, "frames": frames,
            "frames_written": frames_written, "mux": mux, "fps": WAN22_FPS,
            "steps": steps, "seed": seed, "guidance": guidance, "quant": quant,
            "flow_shift": 5.0,
            "image_path": if is_i2v { image_path.as_str() } else { "" },
            "camera_motion": b.get("camera_motion").and_then(Value::as_str).unwrap_or("none"),
            "creator_prompt": b.get("creator_prompt").and_then(Value::as_str).unwrap_or(&prompt),
            "lora": lora.as_ref().map(|(path, weight, name, pairs)| json!({
                "name": name,
                "weight": weight,
                "path": path,
                "matched_modules": pairs,
                "merge": if quant == "bf16" {
                    "exact_bf16_pinned_host_additive"
                } else {
                    "resident_fp8_requantized_once"
                },
            })),
            "negative_prompt_source": if b.get("negative_prompt").and_then(Value::as_str).is_some_and(|value| !value.trim().is_empty()) { "request" } else { "creator_default" },
            "encode_exit_code": enc_rc, "first_frame_exit_code": first_frame_rc,
            "t2v_exit_code": t2v_rc,
            "out_dir": out_dir_s, "conds": conds_s, "mp4": mp4,
            "first_frame_cache": if is_i2v { first_frame_cache_s.as_str() } else { "" },
            "mp4_url": if artifact_ok { format!("/out/{video_id}/wan22_t2v.mp4") } else { String::new() },
            "probe": probe,
            "encode_log": enc_log.to_string_lossy(),
            "first_frame_log": first_frame_log.to_string_lossy(),
            "t2v_log": t2v_log.to_string_lossy(),
            "encode_seconds": enc_secs, "first_frame_seconds": first_frame_secs,
            "t2v_seconds": t2v_secs, "total_wall_seconds": total_wall,
            "encode_peak_vram_mib": encode_peak_vram_mib,
            "first_frame_peak_vram_mib": first_frame_peak_vram_mib,
            "t2v_peak_vram_mib": t2v_peak_vram_mib,
            "note": if is_i2v {
                format!("Wan2.2-TI2V-5B creator-native I2V profile: process-isolated cover-resize/center-crop source VAE encode, clean frame-0 replacement before and after each step, per-token zero timestep for conditioned frame patches, {quant} DiT, Flow-UniPC 50-step shift-5 sampling, and 24 fps MP4 mux.")
            } else {
                format!("Wan2.2-TI2V-5B creator T2V profile: official UMT5 conditioning and default negative prompt, {quant} DiT, Flow-UniPC 50-step shift-5 sampling, tiled VAE decode, and 24 fps MP4 mux.")
            },
        }),
    )
}

/// Bounded Wan2.2 T2V-A14B LoRA preview. This is intentionally separate from
/// the accepted TI2V-5B profile: it runs the image-trained A14B adapter on the
/// matching dual-expert base and returns a short MP4 for checkpoint evaluation.
fn post_video_wan22_a14b(st: &AppState, b: &Value) -> Response {
    if let Err(error) = validate_wan22_a14b_request(b) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
    }
    let prompt = b["prompt"].as_str().unwrap_or("").trim().to_string();
    let requested_negative = b
        .get("negative_prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let negative = if requested_negative.is_empty() {
        WAN22_DEFAULT_NEGATIVE.to_string()
    } else {
        requested_negative
    };
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };
    let (lora_path, lora_weight, lora_name) = match wan22_a14b_lora(b) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    let _ = std::fs::create_dir_all(&out_dir);
    let conds = out_dir.join("wan22_a14b_conds.safetensors");
    let encode_log = out_dir.join("wan22_a14b_encode.log");
    let render_log = out_dir.join("wan22_a14b_t2v.log");
    let conds_s = conds.to_string_lossy().into_owned();
    let out_dir_s = out_dir.to_string_lossy().into_owned();

    let encode_started = std::time::Instant::now();
    let mut encode = wan22_command(&repo_path(WAN22_ENCODE));
    encode.args([&prompt, &negative, &conds_s]);
    let encode_result = run_logged_with_gpu_peak(&mut encode, &encode_log);
    let encode_seconds = encode_started.elapsed().as_secs_f64();
    let (encode_exit_code, encode_peak_vram_mib) = match encode_result {
        Ok(measured) => measured,
        Err(error) => {
            let _ = std::fs::write(&encode_log, format!("spawn failed: {error}"));
            (-1, None)
        }
    };
    if encode_exit_code != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1",
                "video_id": video_id,
                "model": "wan22_a14b",
                "state": "failed",
                "failed_step": "encode",
                "encode_exit_code": encode_exit_code,
                "encode_log": encode_log.to_string_lossy(),
                "out_dir": out_dir_s,
                "error": "wan22_encode_prompt failed; inspect encode_log",
            }),
        );
    }

    let render_started = std::time::Instant::now();
    let mut render = wan22_command(&repo_path(WAN22_A14B_LORA_T2V));
    render
        .arg(&conds_s)
        .arg(model_path(WAN22_A14B_HIGH))
        .arg(model_path(WAN22_A14B_LOW))
        .arg(&out_dir_s)
        .arg(&lora_path)
        .arg(format!("{lora_weight}"))
        .arg(WAN22_A14B_STEPS.to_string())
        .arg(seed.to_string())
        .arg("1")
        .arg(format!("{WAN22_A14B_GUIDANCE}"))
        .arg("12.0")
        .arg("0")
        .arg("4.0");
    let render_result = run_logged_with_gpu_peak(&mut render, &render_log);
    let render_seconds = render_started.elapsed().as_secs_f64();
    let (render_exit_code, render_peak_vram_mib) = match render_result {
        Ok(measured) => measured,
        Err(error) => {
            let _ = std::fs::write(&render_log, format!("spawn failed: {error}"));
            (-1, None)
        }
    };
    if render_exit_code != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1",
                "video_id": video_id,
                "model": "wan22_a14b",
                "state": "failed",
                "failed_step": "t2v",
                "render_exit_code": render_exit_code,
                "render_log": render_log.to_string_lossy(),
                "out_dir": out_dir_s,
                "error": "wan22_a14b_lora_t2v failed; inspect render_log",
            }),
        );
    }

    let mp4 = out_dir.join("wan22_a14b_lora_t2v.mp4");
    let mp4_s = mp4.to_string_lossy().into_owned();
    let probe = probe_video_path(&mp4_s).ok();
    let artifact_ok = probe.as_ref().is_some_and(|value| {
        probe_matches_video_profile(
            value,
            WAN22_A14B_WIDTH,
            WAN22_A14B_HEIGHT,
            WAN22_A14B_FRAMES,
            WAN22_A14B_FPS,
            false,
        )
    });
    json_resp(
        if artifact_ok {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &json!({
            "schema": "serenity.video_result.v1",
            "video_id": video_id,
            "model": "wan22_a14b",
            "backend": BACKEND_NAME,
            "control_plane": "serenity-server",
            "state": if artifact_ok { "complete" } else { "failed" },
            "readiness_label": "experimental_lora_preview",
            "accepted_video_artifact": artifact_ok,
            "accepted_video_parity": false,
            "width": WAN22_A14B_WIDTH,
            "height": WAN22_A14B_HEIGHT,
            "frames": WAN22_A14B_FRAMES,
            "fps": WAN22_A14B_FPS,
            "steps": WAN22_A14B_STEPS,
            "seed": seed,
            "guidance": WAN22_A14B_GUIDANCE,
            "shift": 12.0,
            "expert": "dual_expert_t2v_boundary_0.875",
            "lora": [{"name": lora_name, "weight": lora_weight, "path": lora_path}],
            "out_dir": out_dir_s,
            "conds": conds_s,
            "mp4": mp4_s,
            "mp4_url": if artifact_ok {
                format!("/out/{video_id}/wan22_a14b_lora_t2v.mp4")
            } else {
                String::new()
            },
            "probe": probe,
            "encode_log": encode_log.to_string_lossy(),
            "render_log": render_log.to_string_lossy(),
            "encode_seconds": encode_seconds,
            "render_seconds": render_seconds,
            "total_wall_seconds": encode_seconds + render_seconds,
            "encode_peak_vram_mib": encode_peak_vram_mib,
            "render_peak_vram_mib": render_peak_vram_mib,
            "note": "Bounded T2V-A14B LoRA checkpoint preview; not the accepted TI2V-5B product profile.",
        }),
    )
}

/// Bernini-R production arm. Rust only validates and sequences the existing
/// Mojo stages; all model, sampler, APG, and VAE math remains Mojo-owned.
fn post_video_bernini(st: &AppState, b: &Value) -> Response {
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

struct Scail2RunPaths {
    run_dir: std::path::PathBuf,
    decode_dir: std::path::PathBuf,
    public_dir: std::path::PathBuf,
    decoded_mp4: std::path::PathBuf,
    public_mp4: std::path::PathBuf,
}

fn scail2_run_paths(
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

fn publish_scail2_mp4(paths: &Scail2RunPaths) -> Result<(), String> {
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
fn post_video_scail2(st: &AppState, b: &Value) -> Response {
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

fn probe_matches_video_profile(
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

fn fps_from_rate(rate: &str) -> f64 {
    // "num/den"
    if let Some((n, d)) = rate.split_once('/') {
        let (n, d) = (
            n.parse::<f64>().unwrap_or(0.0),
            d.parse::<f64>().unwrap_or(0.0),
        );
        if d != 0.0 {
            return n / d;
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
            "-show_entries", "format=duration,format_name", "-of", "json", &mp4,
        ])
        .output()
        .map_err(|e| format!("cannot probe MP4: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cannot probe MP4: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let probe: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("cannot parse ffprobe response: {e}"))?;
    let fstr = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let ffloat = |v: &Value, k: &str| {
        v.get(k)
            .and_then(|x| {
                x.as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| x.as_f64())
            })
            .unwrap_or(0.0)
    };
    let fint = |v: &Value, k: &str| {
        v.get(k)
            .and_then(|x| {
                x.as_str()
                    .and_then(|s| s.parse::<i64>().ok())
                    .or_else(|| x.as_i64())
            })
            .unwrap_or(0)
    };
    let (mut fmt_dur, mut fmt_name) = (0.0, String::new());
    if let Some(format) = probe.get("format") {
        fmt_dur = ffloat(format, "duration");
        fmt_name = fstr(format, "format_name");
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
            match fstr(stream, "codec_type").as_str() {
                "video" if !has_video => {
                    has_video = true;
                    width = fint(stream, "width");
                    height = fint(stream, "height");
                    video_codec = fstr(stream, "codec_name");
                    duration = ffloat(stream, "duration");
                    fps = fps_from_rate(&fstr(stream, "avg_frame_rate"));
                    frames = fint(stream, "nb_read_frames");
                    if frames <= 0 {
                        frames = fint(stream, "nb_frames");
                    }
                    if frames <= 0 && duration > 0.0 && fps > 0.0 {
                        frames = (duration * fps + 0.5) as i64;
                    }
                }
                "audio" if !has_audio => {
                    has_audio = true;
                    audio_codec = fstr(stream, "codec_name");
                    audio_duration = ffloat(stream, "duration");
                    audio_sample_rate = fint(stream, "sample_rate");
                    audio_channels = fint(stream, "channels");
                }
                _ => {}
            }
        }
    }
    if duration <= 0.0 {
        duration = fmt_dur;
    }
    Ok(json!({
        "schema": "serenity.video_probe.v1", "mp4": mp4, "format_name": fmt_name,
        "stream_count": stream_count, "has_video": has_video, "has_audio": has_audio, "audio": has_audio,
        "width": width, "height": height, "frame_count": frames, "duration": duration, "fps": fps,
        "video_codec": video_codec, "audio_codec": audio_codec, "audio_duration": audio_duration,
        "audio_sample_rate": audio_sample_rate, "audio_channels": audio_channels,
        "muxing": if has_video && frames > 0 && duration > 0.0 { "probe_ok" } else { "incomplete_probe" },
        "audio_behavior": if has_audio { "audio_stream_present" } else { "video_only_no_audio_stream" },
    }))
}

/// GET /v1/video/probe?path=<mp4> — ffprobe wrapper reshaped to serenity.video_probe.v1.
pub async fn get_video_probe(Query(q): Query<HashMap<String, String>>) -> Response {
    let mp4 = q.get("path").cloned().unwrap_or_default();
    if mp4.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'path' query parameter is required",
        );
    }
    match probe_video_path(&mp4) {
        Ok(doc) => json_resp(StatusCode::OK, &doc),
        Err(error) => err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx2_prompt_normalization_removes_token_changing_edge_whitespace() {
        let normalized = normalize_ltx2_prompt_fields(&json!({
            "prompt": "  a woman turns toward camera \n",
            "negative": "\twatermark  ",
        }));
        assert_eq!(normalized["prompt"], "a woman turns toward camera");
        assert_eq!(normalized["negative"], "watermark");
    }

    #[test]
    fn ltx2_camera_motion_attaches_one_real_adapter_and_one_prompt_suffix() {
        let normalized = normalized_ltx2_camera_motion_request(&json!({
            "prompt": "A woman turns toward the camera",
            "camera_motion": "dolly_in",
        }))
        .unwrap();
        assert_eq!(
            normalized["prompt"],
            "A woman turns toward the camera, dolly in, camera pushing forward, smooth forward movement"
        );
        assert_eq!(
            normalized["creator_prompt"],
            "A woman turns toward the camera"
        );
        assert_eq!(normalized["creator_camera_motion_applied"], true);
        assert_eq!(normalized["lora"].as_array().unwrap().len(), 1);
        assert_eq!(
            normalized["lora"][0]["name"],
            "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
        );
        assert_eq!(normalized["lora"][0]["source"], "camera_control");
        let repeated = normalized_ltx2_camera_motion_request(&normalized).unwrap();
        assert_eq!(repeated["prompt"], normalized["prompt"]);
        assert_eq!(repeated["lora"].as_array().unwrap().len(), 1);
        assert!(normalized_ltx2_camera_motion_request(&json!({
            "prompt": "probe",
            "camera_motion": "orbit",
        }))
        .unwrap_err()
        .contains("unsupported LTX camera_motion"));
    }

    #[test]
    fn wan_camera_motion_is_explicit_and_idempotent() {
        let normalized = normalized_wan22_camera_motion_request(&json!({
            "prompt": "A blonde cyborg turns toward the camera",
            "camera_motion": "dolly_in",
        }))
        .unwrap();
        assert_eq!(
            normalized["prompt"],
            "A blonde cyborg turns toward the camera, dolly in, camera pushing forward, smooth forward movement"
        );
        assert_eq!(
            normalized["creator_prompt"],
            "A blonde cyborg turns toward the camera"
        );
        assert_eq!(normalized["creator_camera_motion_applied"], true);
        let repeated = normalized_wan22_camera_motion_request(&normalized).unwrap();
        assert_eq!(repeated["prompt"], normalized["prompt"]);
        assert!(normalized_wan22_camera_motion_request(&json!({
            "prompt": "probe",
            "camera_motion": "orbit",
        }))
        .unwrap_err()
        .contains("unsupported Wan camera_motion"));
    }

    #[test]
    fn wan_i2v_size_matches_creator_max_area_alignment() {
        assert_eq!(wan22_creator_i2v_size(544, 960), (704, 1248));
        assert_eq!(wan22_creator_i2v_size(960, 544), (1248, 704));
        assert_eq!(wan22_creator_i2v_size(704, 1280), (704, 1280));
        assert_eq!(wan22_creator_i2v_size(1280, 704), (1280, 704));
        assert_eq!(wan22_creator_i2v_size(1024, 1024), (960, 928));
    }

    #[test]
    fn ltx2_mojo_request_accepts_creator_first_and_last_keyframes() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "serenity-ltx2-keyframes-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let caps = root.join("caps.json");
        let first = root.join("first.png");
        let last = root.join("last.png");
        std::fs::write(&caps, b"{}").unwrap();
        std::fs::write(&first, b"first-frame fixture").unwrap();
        std::fs::write(&last, b"last-frame fixture").unwrap();
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "the subject turns and settles into the final pose",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
            "image_path": first,
            "image_strength": 1.0,
            "last_image_path": last,
            "last_image_strength": 1.0,
        });
        validate_ltx2_mojo_request(&request).unwrap();
        let mut with_video = request;
        with_video["video_path"] = json!(root.join("source.mp4"));
        std::fs::write(
            with_video["video_path"].as_str().unwrap(),
            b"video fixture",
        )
        .unwrap();
        assert!(validate_ltx2_mojo_request(&with_video)
            .unwrap_err()
            .contains("mutually exclusive"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sulphur_checkpoint_defaults_to_the_creator_workflow() {
        for checkpoint in [
            "sulphur_dev_bf16",
            "sulphur_dev_fp8mixed",
            "sulphur_dev_fp8_serenity",
        ] {
            let normalized = normalized_ltx2_checkpoint_workflow_request(&json!({
                "checkpoint": checkpoint,
                "prompt": "creator workflow probe",
                "negative": "",
                "guidance_mode": "dev",
                "sampler": "res2s",
                "scheduler": "ltx2",
                "steps": 20,
                "workflow_profile": "",
            }))
            .unwrap();
            assert_eq!(
                normalized["workflow_profile"],
                "sulphur-2-base-distilled-v1"
            );
            assert_eq!(normalized["guidance_mode"], "distilled");
            assert_eq!(normalized["sampler"], "euler_ancestral_cfg_pp");
            assert_eq!(normalized["scheduler"], "sulphur_creator_8_3");
            assert_eq!(normalized["steps"], 8);
            assert_eq!(
                normalized["negative"],
                "pc game, console game, video game, cartoon, childish, ugly"
            );
            assert_eq!(
                normalized["creator_workflow_source"],
                "https://huggingface.co/SulphurAI/Sulphur-2-base/blob/main/workflows/ltx23_t2v%20distilled.json"
            );
        }
    }

    #[test]
    fn official_ltx2_dev_aliases_share_the_creator_fast_identity() {
        for checkpoint in [
            "ltx-2.3-22b-dev",
            "ltx-2.3-22b-dev.safetensors",
            LTX2_REFHQ_CHECKPOINT,
            "ltx-2.3-22b-dev-fp8.safetensors",
            LTX2_REFHQ_BF16_CHECKPOINT,
            "ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors",
        ] {
            assert!(
                is_official_ltx2_dev_checkpoint_name(checkpoint),
                "official alias was not recognized: {checkpoint}"
            );
        }
        assert!(!is_official_ltx2_dev_checkpoint_name(
            "a-user-ltx23-full-finetune"
        ));
    }

    #[test]
    fn sulphur_creator_registry_uses_the_published_enhancer_artifacts() {
        let bf16_profile =
            ltx2_checkpoint_workflow("sulphur_dev_bf16.safetensors").unwrap();
        let profile =
            ltx2_checkpoint_workflow("sulphur_dev_fp8_serenity.safetensors").unwrap();
        assert_eq!(profile["id"], bf16_profile["id"]);
        assert_eq!(
            profile["prompt_enhancer"]["weights"],
            "prompt_enhancer/sulphur_prompt_enhancer_model-q8_0.gguf"
        );
        assert_eq!(
            profile["prompt_enhancer"]["mmproj"],
            "prompt_enhancer/mmproj-BF16.gguf"
        );
        assert_eq!(
            profile["distillation_adapter"]["stage1_weight"],
            0.7
        );
        assert_eq!(
            profile["distillation_adapter"]["stage2_weight"],
            0.5
        );
    }

    #[test]
    fn ltx2_temporal_edits_pin_creator_cudnn_before_general_mojo_runtime() {
        let standard = std::env::split_paths(&ltx2_request_ld_path("standard"))
            .next()
            .unwrap();
        let retake = std::env::split_paths(&ltx2_request_ld_path("retake"))
            .next()
            .unwrap();
        let extend = std::env::split_paths(&ltx2_request_ld_path("extend_end"))
            .next()
            .unwrap();
        assert_eq!(standard, repo_root().join(".pixi/envs/default/lib"));
        assert_eq!(retake, ltx2_decode_cudnn_lib());
        assert_eq!(extend, ltx2_decode_cudnn_lib());
    }

    #[test]
    fn ltx2_profile_runner_rejects_stale_build_inputs() {
        let base = std::time::UNIX_EPOCH;
        let runner = base + std::time::Duration::from_secs(20);
        assert!(
            !LTX2_REQUEST_RUNNER_BUILD_INPUTS
                .contains(&"serenitymojo/configs/ltx2_checkpoint_workflows.json"),
            "server-only workflow aliases must not stale every AOT geometry runner"
        );
        assert!(ltx2_runner_mtime_covers_inputs(
            runner,
            &[
                base + std::time::Duration::from_secs(10),
                base + std::time::Duration::from_secs(20),
            ],
        ));
        assert!(!ltx2_runner_mtime_covers_inputs(
            runner,
            &[base + std::time::Duration::from_secs(21)],
        ));
    }

    #[test]
    fn readiness_shape() {
        let d = readiness_doc();
        assert_eq!(d.get("schema").unwrap(), "serenity.video_status.v1");
        assert_eq!(d.get("endpoint").unwrap(), "/v1/video");
        // bin_x resolves against the active repo root, so runner presence is
        // machine-dependent (built on the dev boxes, absent on CI).
        let ltx2_request_ready = ltx2_resolved_profiles()
            .iter()
            .any(ltx2_profile_runner_available);
        let ltx2_ready =
            ltx2_request_ready || (runner_available() && ltx2_decode_runtime_available());
        let wan22_built = wan22_missing().is_empty();
        let bernini_built = bernini_missing().is_empty();
        let scail2_built = scail2_missing().is_empty();
        if !ltx2_ready && !wan22_built && !bernini_built && !scail2_built {
            assert_eq!(d.get("state").unwrap(), "runner_missing");
            assert_eq!(d.get("readiness_label").unwrap(), "build_required");
        }
        assert_eq!(d.get("accepted").unwrap(), false);
        assert_eq!(d.get("backend").unwrap(), "mojo");
        assert_eq!(d.get("control_plane").unwrap(), "serenity-server");
        let runners = d.get("candidate_runners").unwrap().as_array().unwrap();
        assert_eq!(runners.len(), 5);
        assert_eq!(runners[1].get("model").unwrap(), "ltx2_t2v_av");
        assert_eq!(runners[1].get("target_frame_count").unwrap(), 121);
        let refhq = &runners[1]["modes"]["ltx2_refhq"];
        assert_eq!(refhq.get("prompt_driven").unwrap(), true);
        assert_eq!(refhq.get("target_width").unwrap(), 1920);
        assert_eq!(refhq.get("target_height").unwrap(), 1088);
        assert_eq!(refhq.get("target_frame_count").unwrap(), 121);
        assert_eq!(refhq.get("checkpoint").unwrap(), LTX2_REFHQ_CHECKPOINT);
        assert_eq!(
            refhq.get("processes").unwrap(),
            &json!(["stage1", "upscaler", "stage2", "decode"])
        );
        assert!(refhq.get("accepted_audio_parity").unwrap().is_boolean());
        assert_eq!(
            refhq["conditioning_cache"].get("producer").unwrap(),
            LTX2_CONTEXT_SCRIPT
        );
        let request_runner = &runners[1]["modes"]["ltx2_mojo_request"];
        assert_eq!(
            request_runner["runner"],
            LTX2_MOJO_REQUEST_RUNNER
        );
        assert_eq!(
            request_runner["supported_profiles"]
                .as_array()
                .unwrap()
                .len(),
            31
        );
        let post_upscalers = request_runner["post_upscalers"].as_array().unwrap();
        assert_eq!(post_upscalers.len(), 3);
        let feature_adapters = request_runner["feature_adapters"].as_array().unwrap();
        assert_eq!(feature_adapters.len(), 19);
        let checkpoint_workflows = request_runner["checkpoint_workflows"].as_array().unwrap();
        let sulphur = checkpoint_workflows
            .iter()
            .find(|entry| entry["id"] == "sulphur-2-base-distilled-v1")
            .unwrap();
        assert_eq!(sulphur["sampler"], "euler_ancestral_cfg_pp");
        assert_eq!(sulphur["scheduler"], "sulphur_creator_8_3");
        assert!(sulphur["adapter_available"].is_boolean());
        assert_eq!(sulphur["prompt_enhancer_available"], false);
        assert!(sulphur["prompt_enhancer_files_available"].is_boolean());
        assert_eq!(sulphur["prompt_enhancer_runtime"], "not_implemented");
        assert_eq!(
            feature_adapters
                .iter()
                .find(|entry| entry["id"] == "cinemagraph")
                .unwrap()["status"],
            "overlay_admitted"
        );
        assert_eq!(
            feature_adapters
                .iter()
                .find(|entry| entry["id"] == "foley-v2a")
                .unwrap()["status"],
            "v2a_admitted"
        );
        let realesrgan = post_upscalers
            .iter()
            .find(|entry| entry["id"] == "realesrgan-x4plus")
            .unwrap();
        assert_eq!(
            realesrgan["available"],
            bin_x(REALESRGAN_X4_RUNNER) && nonempty_file(&model_path(REALESRGAN_X4_WEIGHTS))
        );
        if realesrgan["available"] == true {
            assert_eq!(realesrgan["status"], "experimental_slow");
        }
        let realesrgan_fast = post_upscalers
            .iter()
            .find(|entry| entry["id"] == "realesrgan-fast-x4v3")
            .unwrap();
        assert_eq!(
            realesrgan_fast["available"],
            bin_x(REALESRGAN_X4_RUNNER) && nonempty_file(&model_path(REALESRGAN_FAST_X4_WEIGHTS))
        );
        let seedvr2 = post_upscalers
            .iter()
            .find(|entry| entry["id"] == "seedvr2-3b")
            .unwrap();
        let seedvr2_available = bin_x(SEEDVR2_PRODUCT_RUNNER)
            && SEEDVR2_WEIGHTS
                .iter()
                .all(|weight| nonempty_file(&model_path(weight)));
        assert_eq!(seedvr2["available"], false);
        assert_eq!(seedvr2["status"], "source_only");
        assert!(
            seedvr2["missing"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry == "product user-video adapter is not implemented")
        );
        if !seedvr2_available {
            assert!(seedvr2["missing"].as_array().unwrap().len() > 1);
        }
        assert_eq!(request_runner["asynchronous"], true);
        assert_eq!(request_runner["ui_progress"], true);
        assert_eq!(request_runner["available"], ltx2_request_ready);
        assert_eq!(request_runner["requires_authored_conditioning"], false);
        assert_eq!(request_runner["automatic_conditioning"]["backend"], "mojo");
        assert_eq!(
            request_runner["automatic_conditioning"]["available"],
            ltx2_mojo_conditioning_missing().is_empty()
        );
        assert_eq!(runners[2].get("model").unwrap(), "wan22_t2v");
        if !wan22_built {
            assert_eq!(runners[2].get("status").unwrap(), "prerequisites_missing");
            assert!(
                !runners[2]
                    .get("missing")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        }
        assert_eq!(runners[2].get("target_frame_count").unwrap(), 121);
        assert_eq!(runners[2].get("target_width").unwrap(), WAN22_WIDTH);
        assert_eq!(runners[2].get("target_height").unwrap(), WAN22_HEIGHT);
        assert_eq!(
            runners[2].pointer("/native_profiles/1/width").unwrap(),
            WAN22_PORTRAIT_WIDTH
        );
        assert_eq!(
            runners[2].pointer("/native_profiles/1/height").unwrap(),
            WAN22_PORTRAIT_HEIGHT
        );
        assert_eq!(
            runners[2].pointer("/native_profiles/2/width").unwrap(),
            WAN22_I2V_LANDSCAPE_WIDTH
        );
        assert_eq!(
            runners[2].pointer("/native_profiles/3/height").unwrap(),
            WAN22_I2V_PORTRAIT_HEIGHT
        );
        assert_eq!(runners[2].get("i2v_steps").unwrap(), WAN22_I2V_STEPS);
        assert_eq!(runners[2].get("default_steps").unwrap(), 50);
        assert_eq!(runners[2].get("default_guidance").unwrap(), 5.0);
        assert_eq!(
            runners[2].get("quant_modes").unwrap(),
            &json!(["bf16", "fp8"])
        );
        assert_eq!(runners[2].pointer("/modes/lora/max_count").unwrap(), 1);
        assert_eq!(
            runners[2].pointer("/modes/lora/base_model").unwrap(),
            "Wan-AI/Wan2.2-TI2V-5B"
        );
        assert_eq!(
            runners[2].get("accepted_video_parity").unwrap(),
            &(wan22_built && wan22_product_gate_passed())
        );
        // both arms report readiness under arms_ready, matching disk state
        let arms = d.get("arms_ready").unwrap();
        assert_eq!(arms.get("ltx2_t2v_av").unwrap(), ltx2_ready);
        assert_eq!(arms.get("wan22_t2v").unwrap(), wan22_built);
        assert_eq!(runners[3].get("model").unwrap(), "bernini_r_t2v");
        assert_eq!(runners[3].get("target_width").unwrap(), BERNINI_WIDTH);
        assert_eq!(runners[3].get("target_height").unwrap(), BERNINI_HEIGHT);
        assert_eq!(
            runners[3].get("target_frame_count").unwrap(),
            BERNINI_FRAMES
        );
        assert_eq!(runners[3].get("target_fps").unwrap(), BERNINI_FPS);
        assert_eq!(
            runners[3].get("default_steps").unwrap(),
            BERNINI_DEFAULT_STEPS
        );
        assert_eq!(runners[3].get("quant_modes").unwrap(), &json!(["fp8"]));
        assert_eq!(
            arms.get("bernini_r_t2v").unwrap(),
            &(bernini_built && bernini_product_gate_passed())
        );
        assert_eq!(runners[4].get("model").unwrap(), "scail2_animation");
        assert_eq!(runners[4].get("target_width").unwrap(), SCAIL2_WIDTH);
        assert_eq!(runners[4].get("target_height").unwrap(), SCAIL2_HEIGHT);
        assert_eq!(runners[4].get("target_frame_count").unwrap(), SCAIL2_FRAMES);
        assert_eq!(runners[4].get("target_fps").unwrap(), SCAIL2_FPS);
        assert_eq!(runners[4].get("default_steps").unwrap(), SCAIL2_STEPS);
        assert_eq!(runners[4].get("default_guidance").unwrap(), SCAIL2_GUIDANCE);
        assert_eq!(runners[4].get("quant_modes").unwrap(), &json!(["fp8"]));
        assert_eq!(
            arms.get("scail2_animation").unwrap(),
            &(scail2_built && scail2_product_gate_passed())
        );
    }

    #[test]
    fn wan22_ti2v5b_lora_header_rejects_14b_dimensions() {
        let compatible = json!({
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": {
                "dtype": "BF16", "shape": [32, 3072], "data_offsets": [0, 1]
            },
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": {
                "dtype": "BF16", "shape": [3072, 32], "data_offsets": [1, 2]
            }
        });
        assert_eq!(wan22_ti2v5b_lora_header(&compatible).unwrap(), 1);

        let incompatible = json!({
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": {
                "dtype": "BF16", "shape": [32, 5120], "data_offsets": [0, 1]
            },
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": {
                "dtype": "BF16", "shape": [5120, 32], "data_offsets": [1, 2]
            }
        });
        assert!(
            wan22_ti2v5b_lora_header(&incompatible)
                .unwrap_err()
                .contains("probably a 14B adapter")
        );
    }

    #[test]
    fn realesrgan_video_post_upscale_product_smoke() {
        if !bin_x(REALESRGAN_X4_RUNNER) || !nonempty_file(&model_path(REALESRGAN_X4_WEIGHTS)) {
            return;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "serenity-realesrgan-video-smoke-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let native = root.join("native.mp4");
        let fixture = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x120:rate=24",
                "-frames:v",
                "2",
                "-pix_fmt",
                "yuv420p",
                &native.to_string_lossy(),
            ])
            .output()
            .unwrap();
        assert!(
            fixture.status.success(),
            "{}",
            String::from_utf8_lossy(&fixture.stderr)
        );
        let (artifact, probe) = run_realesrgan_video_post_upscale(
            "realesrgan-x4plus",
            &native,
            &root,
            160,
            120,
            2,
            24,
            2,
            |_, _, _| {},
        )
        .unwrap();
        assert!(artifact.is_file());
        assert_eq!(probe["width"], 320);
        assert_eq!(probe["height"], 240);
        assert_eq!(probe["frame_count"], 2);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn fps_parse() {
        assert_eq!(fps_from_rate("24/1"), 24.0);
        assert_eq!(fps_from_rate("30000/1001"), 30000.0 / 1001.0);
        assert_eq!(fps_from_rate("0/0"), 0.0);
    }

    #[test]
    fn wan22_profiles_accept_their_declared_fps() {
        let a14b = json!({
            "muxing": "probe_ok",
            "width": WAN22_A14B_WIDTH,
            "height": WAN22_A14B_HEIGHT,
            "frame_count": WAN22_A14B_FRAMES,
            "fps": WAN22_A14B_FPS as f64,
            "has_audio": false,
        });
        assert!(probe_matches_video_profile(
            &a14b,
            WAN22_A14B_WIDTH,
            WAN22_A14B_HEIGHT,
            WAN22_A14B_FRAMES,
            WAN22_A14B_FPS,
            false,
        ));
        assert!(!probe_matches_video_profile(
            &a14b,
            WAN22_A14B_WIDTH,
            WAN22_A14B_HEIGHT,
            WAN22_A14B_FRAMES,
            WAN22_FPS,
            false,
        ));

        let wan22 = json!({
            "muxing": "probe_ok",
            "width": WAN22_WIDTH,
            "height": WAN22_HEIGHT,
            "frame_count": WAN22_FRAMES,
            "fps": WAN22_FPS as f64,
            "has_audio": false,
        });
        assert!(probe_matches_video_profile(
            &wan22,
            WAN22_WIDTH,
            WAN22_HEIGHT,
            WAN22_FRAMES,
            WAN22_FPS,
            false,
        ));
    }

    #[test]
    fn wan22_command_keeps_cuda_runtime_io_before_sampling() {
        let command = wan22_command(std::path::Path::new("/tmp/wan22-test-runner"));
        let cache = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("CUDA_CACHE_PATH"))
            .and_then(|(_, value)| value)
            .and_then(std::ffi::OsStr::to_str)
            .unwrap();
        assert_eq!(cache, WAN22_CUDA_CACHE);

        let preload = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("LD_PRELOAD"))
            .and_then(|(_, value)| value)
            .and_then(std::ffi::OsStr::to_str)
            .unwrap();
        for required in [
            "libcudnn_graph.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_engines_tensor_ir.so.9",
            "libcudnn_heuristic.so.9",
            "libnvidia-ptxjitcompiler.so.1",
            "libnvidia-nvvm70.so.4",
            "libnvidia-gpucomp.so.",
        ] {
            assert!(preload.contains(required), "missing preload: {required}");
        }
    }

    #[test]
    fn scail2_publishes_only_the_final_mp4_to_output() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "serenity-scail2-output-policy-{}-{nonce}",
            std::process::id()
        ));
        let runtime_root = root.join("runtime");
        let output_root = root.join("output");
        let paths = scail2_run_paths(&runtime_root, &output_root, "video-0001");
        std::fs::create_dir_all(&paths.decode_dir).unwrap();
        std::fs::write(&paths.decoded_mp4, b"measured mp4 fixture").unwrap();
        std::fs::write(paths.decode_dir.join("frame_0.png"), b"internal frame").unwrap();
        std::fs::write(paths.run_dir.join("scail2_denoise.log"), b"internal log").unwrap();

        publish_scail2_mp4(&paths).unwrap();

        let public_files = std::fs::read_dir(&paths.public_dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .collect::<Vec<_>>();
        assert_eq!(public_files.len(), 1);
        assert_eq!(public_files[0].path(), paths.public_mp4);
        assert_eq!(
            std::fs::read(&paths.public_mp4).unwrap(),
            b"measured mp4 fixture"
        );
        assert!(!paths.decoded_mp4.exists());
        assert!(paths.decode_dir.join("frame_0.png").is_file());
        assert!(paths.run_dir.join("scail2_denoise.log").is_file());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn ltx2_mojo_request_preflight_preserves_ui_owned_profile() {
        let caps = std::env::temp_dir().join(format!(
            "serenity-ltx2-ui-caps-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&caps, b"{}").unwrap();
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "vrtlEri2 turns toward camera",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
        });
        validate_ltx2_mojo_request(&request).unwrap();
        let _ = std::fs::remove_file(caps);
    }

    #[test]
    fn ltx2_mojo_request_bf16_requires_and_accepts_dequantized_checkpoint() {
        let caps = std::env::temp_dir().join(format!(
            "serenity-ltx2-bf16-caps-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&caps, b"{}").unwrap();
        let request = json!({
            "checkpoint": LTX2_REFHQ_BF16_CHECKPOINT,
            "quant": "bf16",
            "prompt": "BF16 request contract probe",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
        });
        let result = validate_ltx2_mojo_request(&request);
        if nonempty_file(&model_path(LTX2_REFHQ_BF16)) {
            result.unwrap();
        } else {
            assert!(
                result
                    .unwrap_err()
                    .contains("dequantized dev checkpoint is missing")
            );
        }
        let _ = std::fs::remove_file(caps);
    }

    #[test]
    fn ltx2_checkpoint_registry_separates_dev_support_lora_from_baked_finetune() {
        let dev = ltx2_checkpoint_profile("ltx-2.3-22b-dev-fp8").unwrap();
        assert_eq!(dev.support_lora, "official");
        assert!(dev.guidance_modes.iter().any(|mode| mode == "dev"));
        assert!(dev.quant_modes.iter().any(|mode| mode == "int4"));

        let sulphur = ltx2_checkpoint_profile("sulphur_distill_fp8.safetensors").unwrap();
        assert_eq!(sulphur.id, "sulphur-distill-fp8");
        assert_eq!(sulphur.support_lora, "baked");
        assert_eq!(sulphur.guidance_modes, ["distilled"]);
        assert_eq!(sulphur.quant_modes, ["fp8"]);
        assert_eq!(sulphur.path, "checkpoints/sulphur_distill_fp8.safetensors");
    }

    #[test]
    fn ltx2_mojo_request_preflight_accepts_i2v_source_and_strength() {
        let suffix = format!(
            "{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        );
        let caps = std::env::temp_dir().join(format!("serenity-ltx2-i2v-caps-{suffix}.bin"));
        let image = std::env::temp_dir().join(format!("serenity-ltx2-i2v-source-{suffix}.png"));
        std::fs::write(&caps, b"conditioning fixture").unwrap();
        std::fs::write(&image, b"image fixture").unwrap();
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "the subject turns toward camera",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
            "image_path": image,
            "image_strength": 0.8,
        });
        validate_ltx2_mojo_request(&request).unwrap();
        let _ = std::fs::remove_file(caps);
        let _ = std::fs::remove_file(image);
    }

    #[test]
    fn ltx2_cinemagraph_feature_is_explicit_and_normalizes_to_one_overlay() {
        let adapter = model_path("loras/ltx-2.3-22b-lora-cinemagraph-0.9.safetensors");
        if !nonempty_file(&adapter) {
            return;
        }
        let suffix = format!(
            "{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        );
        let caps =
            std::env::temp_dir().join(format!("serenity-ltx2-cinemagraph-caps-{suffix}.bin"));
        let image =
            std::env::temp_dir().join(format!("serenity-ltx2-cinemagraph-source-{suffix}.png"));
        std::fs::write(&caps, b"conditioning fixture").unwrap();
        std::fs::write(&image, b"image fixture").unwrap();
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "CINEMAGRAPH_MOTION only the candle flame moves",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "audio_policy": "none",
            "lora": [],
            "image_path": image,
            "image_strength": 0.8,
            "feature_id": "cinemagraph",
            "feature_weight": 0.9,
        });
        validate_ltx2_mojo_request(&request).unwrap();
        let normalized = normalized_ltx2_feature_request(&request).unwrap();
        assert_eq!(normalized["feature_adapter"]["id"], "cinemagraph");
        assert_eq!(normalized["lora"].as_array().unwrap().len(), 1);
        assert_eq!(
            normalized["lora"][0]["name"],
            "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"
        );
        assert_eq!(normalized["lora"][0]["weight"], 0.9);
        let _ = std::fs::remove_file(caps);
        let _ = std::fs::remove_file(image);
    }

    #[test]
    fn ltx2_mojo_request_preflight_rejects_invalid_i2v_source_before_gpu() {
        let caps = std::env::temp_dir().join(format!(
            "serenity-ltx2-i2v-invalid-caps-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&caps, b"conditioning fixture").unwrap();
        let mut request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "the subject turns toward camera",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
            "image_path": "/definitely/missing/ltx2-source.png",
            "image_strength": 1.0,
        });
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("I2V source image not found")
        );
        request["image_path"] = json!("");
        request["image_strength"] = json!(1.5);
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("image_strength must be in [0, 1]")
        );
        let _ = std::fs::remove_file(caps);
    }

    #[test]
    fn ltx2_mojo_request_preflight_accepts_v2v_and_rejects_conflicting_sources() {
        let suffix = format!(
            "{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        );
        let caps = std::env::temp_dir().join(format!("serenity-ltx2-v2v-caps-{suffix}.bin"));
        let video = std::env::temp_dir().join(format!("serenity-ltx2-v2v-source-{suffix}.mp4"));
        let image = std::env::temp_dir().join(format!("serenity-ltx2-v2v-source-{suffix}.png"));
        let mask = std::env::temp_dir().join(format!("serenity-ltx2-v2v-mask-{suffix}.png"));
        std::fs::write(&caps, b"conditioning fixture").unwrap();
        std::fs::write(&video, b"video fixture").unwrap();
        std::fs::write(&image, b"image fixture").unwrap();
        std::fs::write(&mask, b"mask fixture").unwrap();
        let mut request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "restyle the source clip while preserving its motion",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
            "video_path": video,
            "video_strength": 0.7,
            "video_mask_path": mask,
        });
        validate_ltx2_mojo_request(&request).unwrap();
        request["image_path"] = json!(image);
        request["image_strength"] = json!(1.0);
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("mutually exclusive")
        );
        request.as_object_mut().unwrap().remove("image_path");
        request.as_object_mut().unwrap().remove("image_strength");
        request["video_strength"] = json!(1.5);
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("video_strength must be in [0, 1]")
        );
        request.as_object_mut().unwrap().remove("video_path");
        request.as_object_mut().unwrap().remove("video_strength");
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("video_mask_path requires video_path")
        );
        let _ = std::fs::remove_file(caps);
        let _ = std::fs::remove_file(video);
        let _ = std::fs::remove_file(image);
        let _ = std::fs::remove_file(mask);
    }

    #[test]
    fn ltx2_temporal_edit_normalizes_retake_and_extend_from_real_probe() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let video = std::env::temp_dir().join(format!(
            "serenity-ltx2-temporal-source-{}-{nonce}.mp4",
            std::process::id()
        ));
        let fixture = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:size=960x544:rate=24",
                "-frames:v",
                "121",
                "-pix_fmt",
                "yuv420p",
                &video.to_string_lossy(),
            ])
            .output()
            .unwrap();
        assert!(
            fixture.status.success(),
            "{}",
            String::from_utf8_lossy(&fixture.stderr)
        );

        let retake = normalized_ltx2_video_edit_request(&json!({
            "video_edit_mode": "retake",
            "video_edit_start": 1.0,
            "video_edit_end": 3.5,
            "video_path": video,
            "video_strength": 0.7,
            "width": 960,
            "height": 544,
            "frames": 121,
            "fps": 24.0,
        }))
        .unwrap();
        assert_eq!(retake["video_source_frames"], 121);
        assert_eq!(retake["video_strength"], 0.0);
        assert_eq!(retake["video_edit_start"], 1.0);
        assert_eq!(retake["video_edit_end"], 3.5);

        let extend_end = normalized_ltx2_video_edit_request(&json!({
            "video_edit_mode": "extend_end",
            "video_path": video,
            "width": 960,
            "height": 544,
            "frames": 193,
            "fps": 24.0,
        }))
        .unwrap();
        assert_eq!(extend_end["video_source_frames"], 121);
        assert_eq!(extend_end["video_extend_frames"], 72);
        assert_eq!(extend_end["video_extend_seconds"], 3.0);
        assert_eq!(extend_end["video_edit_start"], 4.5);
        assert_eq!(extend_end["video_edit_end"], 8.0);

        let extend_start = normalized_ltx2_video_edit_request(&json!({
            "video_edit_mode": "extend_start",
            "video_path": video,
            "width": 960,
            "height": 544,
            "frames": 193,
            "fps": 24.0,
        }))
        .unwrap();
        assert_eq!(extend_start["video_edit_start"], 0.0);
        assert_eq!(extend_start["video_edit_end"], 3.5);

        let too_short = normalized_ltx2_video_edit_request(&json!({
            "video_edit_mode": "retake",
            "video_edit_start": 1.0,
            "video_edit_end": 2.0,
            "video_path": video,
            "width": 960,
            "height": 544,
            "frames": 121,
            "fps": 24.0,
        }))
        .unwrap_err();
        assert!(too_short.contains("at least 2 seconds"));
        let _ = std::fs::remove_file(video);
    }

    #[test]
    fn ltx2_source_audio_policy_preflights_and_remuxes() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "serenity-ltx2-source-audio-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let caps = root.join("caps.bin");
        let source = root.join("source.mp4");
        let generated = root.join("generated.mp4");
        std::fs::write(&caps, b"conditioning fixture").unwrap();
        let source_result = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x120:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                &source.to_string_lossy(),
            ])
            .output()
            .unwrap();
        assert!(
            source_result.status.success(),
            "{}",
            String::from_utf8_lossy(&source_result.stderr)
        );
        let generated_result = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:size=160x120:rate=24",
                "-frames:v",
                "4",
                "-pix_fmt",
                "yuv420p",
                &generated.to_string_lossy(),
            ])
            .output()
            .unwrap();
        assert!(generated_result.status.success());

        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "preserve the source motion",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": caps,
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "audio_policy": "preserve",
            "lora": [],
            "video_path": source,
            "video_strength": 0.7,
        });
        validate_ltx2_mojo_request(&request).unwrap();

        let (artifact, probe) = remux_ltx2_source_audio(&generated, &source, &root, 4).unwrap();
        assert!(artifact.is_file());
        assert_eq!(probe["has_audio"], true);
        assert_eq!(probe["frame_count"], 4);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn ltx2_mojo_request_accepts_blank_conditioning_when_auto_encoder_is_available() {
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "a lighthouse on a rocky coast at sunset",
            "negative": "watermark",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": "",
            "caps_negative": "",
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
        });
        let missing = ltx2_mojo_conditioning_missing();
        let result = validate_ltx2_mojo_request(&request);
        if missing.is_empty() {
            result.unwrap();
        } else {
            assert!(
                result
                    .unwrap_err()
                    .contains("automatic prompt conditioning is unavailable")
            );
        }
    }

    #[test]
    fn ltx2_mojo_request_rejects_geometry_outside_published_profile_before_gpu() {
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "profile mismatch probe",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": "/not/reached",
            "width": 1024,
            "height": 1024,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 24,
            "include_audio": false,
            "lora": [],
        });
        let error = validate_ltx2_mojo_request(&request).unwrap_err();
        assert!(error.contains("unsupported LTX2 standard native profile 1024x1024"));
        assert!(!error.contains("conditioning artifact"));
    }

    #[test]
    fn ltx2_mojo_request_accepts_every_registry_profile_before_artifact_checks() {
        for profile in ltx2_resolved_profiles() {
            for mode in &profile.modes {
                assert!(
                    ltx2_request_profile_for_mode(
                        profile.width,
                        profile.height,
                        profile.frames,
                        profile.fps as f64,
                        mode,
                    )
                    .is_some(),
                    "profile {}x{} {}f@{} did not resolve for declared mode {mode}",
                    profile.width,
                    profile.height,
                    profile.frames,
                    profile.fps,
                );
            }
            if !profile.modes.iter().any(|mode| mode == "standard") {
                continue;
            }
            let request = json!({
                "checkpoint": LTX2_REFHQ_CHECKPOINT,
                "quant": "fp8",
                "prompt": "native profile contract probe",
                "sampler": "euler",
                "scheduler": "ltx2_distilled",
                "guidance_mode": "distilled",
                "caps_positive": "/not/reached",
                "width": profile.width,
                "height": profile.height,
                "frames": profile.frames,
                "steps": 8,
                "seed": 42,
                "fps": profile.fps,
                "include_audio": false,
                "lora": [],
            });
            let error = validate_ltx2_mojo_request(&request).unwrap_err();
            assert!(
                error.contains("conditioning artifact not found"),
                "profile {}x{} {}f@{} was rejected before artifact validation: {error}",
                profile.width,
                profile.height,
                profile.frames,
                profile.fps,
            );
        }
    }

    #[test]
    fn ltx2_mojo_request_preflight_rejects_missing_conditioning_before_gpu() {
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "vrtlEri2 turns toward camera",
            "sampler": "euler",
            "scheduler": "ltx2_distilled",
            "guidance_mode": "distilled",
            "caps_positive": "/definitely/missing/ltx2-conditioning.json",
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
        });
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("conditioning artifact not found")
        );
    }

    #[test]
    fn ltx2_mojo_request_preflight_rejects_mismatched_distilled_sampler_before_gpu() {
        let request = json!({
            "checkpoint": LTX2_REFHQ_CHECKPOINT,
            "quant": "fp8",
            "prompt": "vrtlEri2 turns toward camera",
            "sampler": "res2s",
            "scheduler": "ltx2",
            "guidance_mode": "distilled",
            "caps_positive": "/not/reached",
            "width": 512,
            "height": 768,
            "frames": 121,
            "steps": 8,
            "seed": 42,
            "fps": 25.0,
            "include_audio": false,
            "lora": [],
        });
        assert!(
            validate_ltx2_mojo_request(&request)
                .unwrap_err()
                .contains("distilled mode requires sampler=euler")
        );
    }

    #[test]
    fn ltx2_seed_is_bounded_and_defaults_only_when_absent() {
        assert_eq!(ltx2_seed(&json!({})), Ok(42));
        assert_eq!(ltx2_seed(&json!({ "seed": 0 })), Ok(0));
        assert_eq!(
            ltx2_seed(&json!({ "seed": 4_294_967_295_u64 })),
            Ok(4_294_967_295)
        );
        assert!(ltx2_seed(&json!({ "seed": -1 })).is_err());
        assert!(ltx2_seed(&json!({ "seed": 4_294_967_296_u64 })).is_err());
        assert!(ltx2_seed(&json!({ "seed": 1.5 })).is_err());
    }

    #[test]
    fn ltx2_context_cache_key_is_stable_and_prompt_specific() {
        let a = ltx2_context_key("a red balloon", "<creator-default-negative>");
        assert_eq!(
            a,
            ltx2_context_key("a red balloon", "<creator-default-negative>")
        );
        assert_ne!(
            a,
            ltx2_context_key("a blue balloon", "<creator-default-negative>")
        );
        assert_ne!(a, ltx2_context_key("a red balloon", "flicker"));
    }

    #[test]
    fn ltx2_parity_report_is_bound_to_exact_mojo_runner() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "serenity-ltx2-parity-runner-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let runner = root.join("ltx2_video_smoke_runner");
        let report = root.join("sampler.json");
        std::fs::write(&runner, b"measured runner A").unwrap();
        let digest = sha256sum(&runner).unwrap();
        std::fs::write(
            &report,
            serde_json::to_vec(&json!({
                "schema": "serenity.ltx2.sampler_parity.v1",
                "creator_revision": LTX2_CREATOR_REVISION,
                "mojo_runner_sha256": digest,
                "bar": 0.999,
                "passed": true,
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(ltx2_parity_report_matches(
            &report,
            &runner,
            "serenity.ltx2.sampler_parity.v1",
            false,
        ));

        std::fs::write(&runner, b"changed runner B").unwrap();
        assert!(!ltx2_parity_report_matches(
            &report,
            &runner,
            "serenity.ltx2.sampler_parity.v1",
            false,
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn ltx2_context_cache_reuses_existing_prompt_entry() {
        let root =
            std::env::temp_dir().join(format!("serenity-ltx2-cache-hit-{}", std::process::id()));
        let prompt = "a red balloon over a meadow";
        let key = ltx2_context_key(prompt, "<creator-default-negative>");
        let dir = root
            .join("conditioning_cache")
            .join("ltx2")
            .join("creator-refhq-v1");
        std::fs::create_dir_all(&dir).unwrap();
        let expected = dir.join(format!("{key}.safetensors"));
        let mut header = serde_json::to_vec(&json!({
            "__metadata__": { "creator_revision": LTX2_CREATOR_REVISION },
            "video_context": { "dtype": "BF16", "shape": [1, 1024, 4096], "data_offsets": [0, 0] },
            "audio_context": { "dtype": "BF16", "shape": [1, 1024, 2048], "data_offsets": [0, 0] },
            "neg_video_context": { "dtype": "BF16", "shape": [1, 1024, 4096], "data_offsets": [0, 0] },
            "neg_audio_context": { "dtype": "BF16", "shape": [1, 1024, 2048], "data_offsets": [0, 0] },
            "video_len": { "dtype": "F32", "shape": [1], "data_offsets": [0, 0] },
            "neg_video_len": { "dtype": "F32", "shape": [1], "data_offsets": [0, 0] },
        })).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut fixture = (header.len() as u64).to_le_bytes().to_vec();
        fixture.extend(header);
        std::fs::write(&expected, fixture).unwrap();

        let cache = prepare_ltx2_refhq_context(&root, prompt, "").unwrap();
        assert!(cache.hit);
        assert_eq!(cache.key, key);
        assert_eq!(cache.path, expected);
        assert_eq!(cache.encoder_seconds, 0.0);
        let _ = std::fs::remove_dir_all(root);
    }
}
