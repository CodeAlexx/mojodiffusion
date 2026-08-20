//! Shared `/v1/video` control plane.
//!
//! This module owns HTTP dispatch, aggregate readiness, common paths/hashing,
//! the single-GPU lease, and shared process helpers. Backend-specific admission
//! and Mojo runner orchestration live in the sibling modules under `video/`.
//! Learned model execution remains in those external Mojo runners; Rust owns
//! validation, lifecycle, and artifact delivery.

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use serenity_wire::WorkerEvent;

use crate::AppState;

pub(crate) mod bernini;
mod ltx2;
pub(crate) mod minimax_h3;
mod probe;
pub(crate) mod scail2;
#[cfg(test)]
mod tests;
mod wan22;
use bernini::*;
use ltx2::*;
use minimax_h3::*;
#[cfg(test)]
use probe::fps_from_rate;
use probe::probe_matches_video_profile;
pub(crate) use probe::{get_video_probe, probe_video_path};
use scail2::*;
use wan22::*;

/// Resolve the learned artifacts touched by a selected video profile. The page
/// warmer consumes these paths before a request starts; generation still
/// validates and opens the exact same files in the model-specific arm.
pub(crate) fn warm_artifacts(
    model: &str,
    resolved_checkpoint: Option<&std::path::Path>,
    quant: &str,
    task: &str,
) -> (String, Vec<crate::warm_load::WarmArtifact>) {
    use crate::warm_load::WarmArtifact;

    let identity = model.trim().to_ascii_lowercase();
    if identity.contains("minimax") || identity.contains("h3") {
        let is_ref = task == "ref2va";
        let root = model_path(if is_ref {
            MINIMAX_H3_REF2VA_MODEL_ROOT
        } else {
            MINIMAX_H3_MODEL_ROOT
        });
        let mut artifacts = vec![
            WarmArtifact::new(
                "MiniMax-H3 text encoder INT8 store",
                model_path(MINIMAX_H3_ENCODER_CACHE),
            ),
            WarmArtifact::new(
                "MiniMax-H3 conditioning cache",
                model_path(MINIMAX_H3_CONDITIONING_CACHE),
            ),
        ];
        let modulation = if is_ref {
            MINIMAX_H3_REF2VA_MODULATION_CACHE
        } else if matches!(task, "i2va" | "l2va" | "fl2va" | "continue") {
            MINIMAX_H3_CONDITIONED_MODULATION_CACHE
        } else {
            MINIMAX_H3_MODULATION_CACHE
        };
        artifacts.push(WarmArtifact::new(
            "MiniMax-H3 modulation runtime cache",
            model_path(modulation),
        ));
        if let Some(cache) = minimax_h3_resident_cache_path(quant, is_ref) {
            artifacts.push(WarmArtifact::new(
                "MiniMax-H3 resident denoiser runtime cache",
                cache,
            ));
        }
        // The quantized runtime stores own the blocks, while the source tree
        // still owns shared tensors. BF16 streams all transformer shards.
        artifacts.push(WarmArtifact::new(
            "MiniMax-H3 transformer source",
            root.join("transformer"),
        ));
        artifacts.extend([
            WarmArtifact::new(
                "MiniMax-H3 audio VAE",
                root.join("audio_vae/model.safetensors"),
            ),
            WarmArtifact::new(
                "MiniMax-H3 video VAE",
                root.join("video_vae/source/model.safetensors"),
            ),
            WarmArtifact::new("MiniMax-H3 processor", root.join("processor")),
        ]);
        return (format!("minimax_h3:{task}:{quant}"), artifacts);
    }

    if identity.contains("ltx") {
        let mut artifacts = vec![
            WarmArtifact::new("LTX Gemma text encoder", model_path(LTX2_GEMMA_FP8)),
            WarmArtifact::new("LTX Gemma tokenizer", model_path(LTX2_GEMMA_TOKENIZER)),
        ];
        if let Some(checkpoint) = resolved_checkpoint {
            artifacts.push(WarmArtifact::new("selected LTX checkpoint", checkpoint));
        } else {
            let checkpoint = if quant == "bf16" {
                LTX2_REFHQ_BF16
            } else if quant == "int4" {
                LTX2_REFHQ_INT4_SLAB
            } else {
                LTX2_CONDITIONING_CHECKPOINT
            };
            artifacts.push(WarmArtifact::new(
                "LTX denoiser checkpoint",
                model_path(checkpoint),
            ));
        }
        if quant == "int4" {
            artifacts.push(WarmArtifact::new(
                "LTX INT4 resident denoiser slab",
                model_path(LTX2_REFHQ_INT4_SLAB),
            ));
        }
        artifacts.push(WarmArtifact::new(
            "LTX distillation adapter",
            model_path(LTX2_REFHQ_DISTILLATION_ADAPTER),
        ));
        return (format!("ltx2:{quant}"), artifacts);
    }

    if identity.contains("wan") {
        let artifacts = if identity.contains("a14b") {
            vec![
                WarmArtifact::new("Wan A14B high denoiser", model_path(WAN22_A14B_HIGH)),
                WarmArtifact::new("Wan A14B low denoiser", model_path(WAN22_A14B_LOW)),
                WarmArtifact::new("Wan UMT5 encoder", model_path(WAN22_UMT5_FILE)),
                WarmArtifact::new("Wan tokenizer", model_path(WAN22_TOKENIZER)),
                WarmArtifact::new("Wan A14B VAE", model_path(WAN22_A14B_VAE)),
            ]
        } else {
            vec![
                WarmArtifact::new(
                    "Wan transformer shard",
                    model_path(WAN22_TRANSFORMER_SHARD_1),
                ),
                WarmArtifact::new(
                    "Wan transformer shard",
                    model_path(WAN22_TRANSFORMER_SHARD_2),
                ),
                WarmArtifact::new(
                    "Wan transformer shard",
                    model_path(WAN22_TRANSFORMER_SHARD_3),
                ),
                WarmArtifact::new("Wan UMT5 encoder", model_path(WAN22_UMT5_FILE)),
                WarmArtifact::new("Wan tokenizer", model_path(WAN22_TOKENIZER)),
                WarmArtifact::new("Wan sentencepiece tokenizer", model_path(WAN22_SPIECE)),
                WarmArtifact::new("Wan VAE", model_path(WAN22_VAE)),
            ]
        };
        return ("wan22".to_string(), artifacts);
    }

    if identity.contains("bernini") {
        return (
            "bernini".to_string(),
            vec![
                WarmArtifact::new("Bernini UMT5 encoder", model_path(WAN22_UMT5_FILE)),
                WarmArtifact::new(
                    "Bernini high denoiser cache",
                    model_path(BERNINI_HIGH_CACHE),
                ),
                WarmArtifact::new(
                    "Bernini low denoiser cache",
                    model_path(BERNINI_LOW_CACHE),
                ),
                WarmArtifact::new("Bernini VAE", model_path(BERNINI_VAE)),
            ],
        );
    }

    if identity.contains("scail") {
        return (
            "scail2".to_string(),
            vec![
                WarmArtifact::new("SCAIL UMT5 encoder", model_path(SCAIL2_UMT5)),
                WarmArtifact::new("SCAIL tokenizer", model_path(SCAIL2_TOKENIZER)),
                WarmArtifact::new("SCAIL CLIP vision encoder", model_path(SCAIL2_CLIP)),
                WarmArtifact::new(
                    "SCAIL denoiser FP8 cache",
                    model_path(SCAIL2_FP8_CACHE),
                ),
                WarmArtifact::new("SCAIL VAE", model_path(SCAIL2_VAE)),
            ],
        );
    }

    (String::new(), Vec::new())
}

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
    let minimax_h3_profiles = &minimax_h3_request_profile_registry().profiles;
    let minimax_h3_default = minimax_h3_default_profile();
    let minimax_h3_profile_documents = minimax_h3_profiles
        .iter()
        .map(minimax_h3_profile_document)
        .collect::<Vec<_>>();
    let minimax_h3_int8_missing = minimax_h3_missing(minimax_h3_default, "int8");
    let minimax_h3_int8_fast_missing = minimax_h3_missing(minimax_h3_default, "int8-fast");
    let minimax_h3_bf16_missing = minimax_h3_missing(minimax_h3_default, "bf16");
    let minimax_h3_int8_ready = minimax_h3_profiles
        .iter()
        .any(|profile| minimax_h3_profile_mode_ready(profile, "int8"));
    let minimax_h3_int8_fast_ready = minimax_h3_profiles
        .iter()
        .any(|profile| minimax_h3_profile_mode_ready(profile, "int8-fast"));
    let minimax_h3_bf16_ready = minimax_h3_profiles
        .iter()
        .any(|profile| minimax_h3_profile_mode_ready(profile, "bf16"));
    let minimax_h3_ready =
        minimax_h3_int8_fast_ready || minimax_h3_int8_ready || minimax_h3_bf16_ready;
    let minimax_h3_ck_attention_ready = minimax_h3_ready
        && nonempty_file(&repo_path("output/lib/libserenity_ck_attention.so"));
    let minimax_h3_conditioned_modes = vec![
        minimax_h3_conditioned_task_document("i2va", "First frame to video"),
        minimax_h3_conditioned_task_document("l2va", "Last frame to video"),
        minimax_h3_conditioned_task_document("fl2va", "First + last frame to video"),
        minimax_h3_conditioned_task_document("ref2va", "Omni-reference (image/video/audio)"),
        minimax_h3_continuation_task_document(),
    ];
    let minimax_h3_conditioned_ready = minimax_h3_conditioned_modes
        .iter()
        .any(|mode| mode.get("available").and_then(Value::as_bool) == Some(true));
    // top-level state reflects whether ANY arm is runnable.
    let any_ready = ltx2_ready
        || wan22_ready
        || bernini_ready
        || scail2_ready
        || minimax_h3_ready
        || minimax_h3_conditioned_ready;
    let state = if ltx2_request_ready || minimax_h3_ready || minimax_h3_conditioned_ready {
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
            "model": "minimax_h3_t2va",
            "status": if minimax_h3_ready { "runtime_geometry_ready" } else if minimax_h3_conditioned_ready { "conditioned_runtime_geometry_ready" } else { "prerequisites_missing" },
            "runner": MINIMAX_H3_REQUEST_RUNNER,
            "runner_topology": "one_request_runner_runtime_geometry_length_fps_and_quant",
            "mode": "phase-isolated: GPU text encode + denoise -> fresh GPU VAE decode -> NVENC mux",
            "request_schema": "serenity.genparams.v1",
            "status_schema": "serenity.minimax_h3.status.v1",
            "result_schema": "serenity.minimax_h3.result.v1",
            "default_steps": MINIMAX_H3_STEPS,
            "asynchronous": true,
            "process_separated_decode": true,
            "runtime_cache": {
                "conditioning": MINIMAX_H3_CONDITIONING_CACHE,
                "modulation_steps_20": MINIMAX_H3_MODULATION_CACHE,
                "int8_groupwise_cache_blocks_48": MINIMAX_H3_INT8_RESIDENT_CACHE,
                "int8_fast_w8a8_blocks_50": MINIMAX_H3_INT8_FAST_RESIDENT_CACHE,
                "generated_on_first_use": true,
                "byte_exact_latent_parity": true,
                "gpu_model_execution_only": true,
            },
            "supported_profiles": minimax_h3_profile_documents,
            "supported_profiles_role": "measured low-VRAM benchmark anchors; not user-facing H3 output resolutions",
            "geometry_constraints": {
                "shape_policy": "h3_base_adapt_shape_v1",
                "base_short_edge": MINIMAX_H3_BASE_SHORT_EDGE,
                "max_pixels": MINIMAX_H3_NATIVE_MAX_PIXELS,
                "width_min": MINIMAX_H3_MIN_DIMENSION,
                "width_max": MINIMAX_H3_MAX_DIMENSION,
                "height_min": MINIMAX_H3_MIN_DIMENSION,
                "height_max": MINIMAX_H3_MAX_DIMENSION,
                "dimension_step": MINIMAX_H3_DIMENSION_STEP,
                "resolutions": minimax_h3_native_resolution_documents(),
                "resolution_role": "tested_presets_not_an_exhaustive_allowlist",
                "seconds_min": MINIMAX_H3_MIN_SECONDS,
                "seconds_max": MINIMAX_H3_MAX_SECONDS,
                "trained_seconds_max": MINIMAX_H3_TRAINED_MAX_SECONDS,
                "long_context_max_sequence_tokens": MINIMAX_H3_LONG_CONTEXT_MAX_SEQUENCE_TOKENS,
                "long_context_tasks": ["t2va", "i2va", "l2va", "fl2va", "continue"],
                "long_context_policy": "experimental_single_pass_resolution_duration_tradeoff",
                "fps_min": 1,
                "fps_max": 120,
                "internal_frame_alignment": "17n+5; output is trimmed/resampled to authored seconds and FPS",
            },
            "quant_modes": [
                {
                    "id": "int8-fast",
                    "label": "INT8 Fast · profile-tuned W8A8 resident",
                    "available": minimax_h3_int8_fast_ready,
                    "missing": minimax_h3_int8_fast_missing,
                    "dtype_contract": "direct_w8a8_profile_resident_prefix_w8a8_cached_tail_int8_text_encoder_bf16_outputs_f32_reductions",
                    "accepted_base_full20_denoise_seconds": 174.577378981_f64,
                    "visual_inspection_passed": true,
                },
                {
                    "id": "int8",
                    "label": "INT8 Quality · profile-tuned groupwise + BF16 tail",
                    "available": minimax_h3_int8_ready,
                    "missing": minimax_h3_int8_missing,
                    "dtype_contract": "groupwise_int8_profile_resident_prefix_bf16_tail_int8_text_encoder_bf16_activations_f32_reductions",
                    "registered_resident_blocks": 41,
                    "hot_one_eval_seconds": 22.268046142_f64,
                    "historical_43_block_full20_seconds_not_registered": 377.228794159_f64,
                },
                {
                    "id": "bf16",
                    "label": "BF16 DiT quality · streamed",
                    "available": minimax_h3_bf16_ready,
                    "missing": minimax_h3_bf16_missing,
                    "dtype_contract": "bf16_dit_streamed_int8_text_encoder_bf16_activations_f32_reductions",
                }
            ],
            "attention_backends": [
                {
                    "id": "ck-int8",
                    "label": "CK INT8 · fastest H3",
                    "available": minimax_h3_ck_attention_ready,
                    "quant_modes": ["int8-fast", "int8"],
                    "accepted_quality_default": false,
                    "accepted_fast_default": true,
                    "kernel_cosine": 0.9998600836968164_f64,
                    "kernel_ms_s19029_h56": 69.63113825_f64,
                    "full20_denoise_seconds": 262.314619493_f64,
                    "second_prompt_full20_denoise_seconds": 274.286097767_f64,
                    "decoded_visual_prompt_gate_count": 2,
                    "decoded_visual_inspection_passed": true,
                },
                {
                    "id": "cudnn",
                    "label": "cU-DNN · quality default",
                    "available": minimax_h3_ready,
                    "accepted_quality_default": true,
                },
                {
                    "id": "sage-int8",
                    "label": "Sage INT8 · experimental",
                    "available": minimax_h3_ready,
                    "quant_modes": ["int8-fast", "int8"],
                    "accepted_quality_default": false,
                    "kernel_cosine": 0.9999076619386169_f64,
                    "one_step_video_cosine": 0.9989187544839178_f64,
                    "one_step_audio_cosine": 0.9962658226306607_f64,
                }
            ],
            "step_cache_modes": [
                {
                    "id": "exact",
                    "label": "Exact · every block, every step",
                    "available": minimax_h3_ready || minimax_h3_conditioned_ready,
                    "exact": true,
                },
                {
                    "id": "high",
                    "label": "Experimental cached · faster, quality loss",
                    "available": minimax_h3_ready || minimax_h3_conditioned_ready,
                    "exact": false,
                    "accepted_quality_default": false,
                    "policy": "fn8_bn8_warmup4_threshold0.12_max_continuous2_max_cached1_group32_int8_residual",
                }
            ],
            "test_prompt": MINIMAX_H3_TEST_PROMPT,
            "prompt_contract": "arbitrary_nonempty_prompt_runtime_conditioning",
            "conditioned_prompt": minimax_h3_conditioned_prompt(),
            "conditioned_prompt_contract": "arbitrary_nonempty_prompt_runtime_conditioning",
            "conditioned_modes": minimax_h3_conditioned_modes,
            "include_audio": true,
            "available": minimax_h3_ready || minimax_h3_conditioned_ready,
            "product_gate": MINIMAX_H3_PRODUCT_GATE,
            "product_gate_role": "informational_quality_evidence_only",
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
                    "format": "torchref/DiffusionModel block LoRA",
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
        "readiness_label": if ltx2_request_ready || minimax_h3_ready { "experimental_request_runner_ready" } else if any_ready { "bounded_daemon_smoke" } else { "build_required" },
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
            "minimax_h3_t2va": minimax_h3_ready,
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

/// POST /v1/video — dispatch to the selected model-specific orchestration
/// module after normalizing the small set of shared request fields.
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
        && model != "minimax_h3"
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "unsupported video model '{model}'; use ltx2, minimax_h3, wan22, wan22_a14b, bernini, or scail2"
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
        if let Err(error) = validate_ltx2_runtime_artifacts(&b) {
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
    if model == "minimax_h3" {
        if let Err(error) = validate_minimax_h3_request(&b) {
            return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
        }
        let quant = b.get("quant").and_then(Value::as_str).unwrap_or("int8");
        let task = minimax_h3_task(&b);
        let missing = if minimax_h3_continue_with_references(&b) {
            minimax_h3_conditioned_missing("ref2va", quant)
        } else if matches!(task, "t2va" | "continue") {
            minimax_h3_missing(minimax_h3_default_profile(), quant)
        } else {
            minimax_h3_conditioned_missing(task, quant)
        };
        if !missing.is_empty() {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "MiniMax-H3 runtime prerequisites missing: {}",
                    missing.join(", ")
                ),
            );
        }
        if task == "continue" {
            if let Err(error) = minimax_h3_continuation_source(&st.out_dir, &b) {
                return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
            }
        }
    }
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
    // Selection-time warming has done all useful work it can do. Stop its host
    // reader before the video subprocess starts opening encoder/model files.
    st.warm_load.cancel_for_generation();
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
    if model == "minimax_h3" {
        return start_minimax_h3_request(&st, &b, gpu);
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
