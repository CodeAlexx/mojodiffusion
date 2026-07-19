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
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::AppState;

const RUNNER: &str = "output/bin/ltx2_video_smoke_runner";
const LTX2_CSHIM: &str = "serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so";
const LTX2_CONTEXT_PYTHON: &str = ".local/share/LTXDesktop/python/bin/python3";
const LTX2_CONTEXT_SCRIPT: &str = "scripts/ltx2_refhq_contexts.py";
const LTX2_CONTEXT_SCHEMA: &str = "serenity.ltx2.refhq_context_cache.v1";
const LTX2_CREATOR_REVISION: &str = "780984275fd47128b02bef9b5c085404276866ee";
const LTX2_REFHQ_CHECKPOINT: &str = "ltx-2.3-22b-dev-fp8";
const LTX2_SAMPLER_PARITY_REPORT: &str = "output/checks/ltx2_sampler_parity.json";
const LTX2_VAE_PARITY_REPORT: &str = "output/checks/ltx2_vae_frame_parity.json";
const LTX2_AUDIO_PARITY_REPORT: &str = "output/checks/ltx2_audio_parity.json";
const LTX2_CREATOR_CUDNN_LIB: &str =
    ".local/share/LTXDesktop/python/lib/python3.13/site-packages/nvidia/cudnn/lib";
const BACKEND_NAME: &str = "mojo";

/// Resolve the checkout that built this server, with an explicit override for
/// packaged/two-machine deployments.  Never silently jump to a different local
/// clone: the Mojo runner, scripts, and product UI must come from one codebase.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("SERENITY_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
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
const WAN22_T2V: &str = "output/bin/wan22_t2v";
const WAN22_MODEL_ROOT: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo";
const WAN22_ARTIFACT_MANIFEST: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/serenity_wan22_manifest.json";
const WAN22_FP8_CACHE: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/wan22_dit_fp8_e4m3_b8fff7315c768468.safetensors";
const WAN22_UMT5_INDEX: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5/model.safetensors.index.json";
const WAN22_UMT5_SHARD_1: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5/model-00001-of-00003.safetensors";
const WAN22_UMT5_SHARD_2: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5/model-00002-of-00003.safetensors";
const WAN22_UMT5_SHARD_3: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5/model-00003-of-00003.safetensors";
const WAN22_TOKENIZER: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/tokenizer.json";
const WAN22_SPIECE: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/spiece.model";
const WAN22_VAE: &str = "vaes/wan2.2_vae.safetensors";
const WAN22_PRODUCT_GATE: &str = "output/checks/wan22_product_gate.json";
const WAN22_HF_REVISION: &str = "b8fff7315c768468a5333511427288870b2e9635";
const WAN22_CREATOR_REVISION: &str = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc";
const WAN22_FP8_CACHE_SHA256: &str =
    "84812d4fe806b7a414c47bd91d02498e8ac07ec5fa4db34ae58dc241524ccb49";
const WAN22_DEFAULT_NEGATIVE: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
const MOJO_CUDNN_RUNTIME: &str = "cudnn/lib/libcudnn.so.9";
/// Pixi runtime libs + cshim (cuDNN/int4) shims — required by the Mojo binaries.
fn mojo_ld_path() -> std::ffi::OsString {
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
    std::env::var_os("LTX2_CREATOR_CUDNN_LIB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| home_path(LTX2_CREATOR_CUDNN_LIB))
}

fn ltx2_decode_runtime_available() -> bool {
    nonempty_file(&ltx2_decode_cudnn_lib().join("libcudnn.so.9"))
}

fn ltx2_decode_ld_path() -> std::ffi::OsString {
    let root = repo_root();
    let mut parts = vec![
        root.join(".pixi/envs/default/lib"),
        root.join("serenitymojo/ops/cshim/lib"),
        ltx2_decode_cudnn_lib(),
    ];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        parts.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(parts).unwrap_or_default()
}
/// svdint4 slab matching the distilled-fp8 base the LTX2 runner streams
/// (`CKPT_FP8` in ltx2_t2v_av_hq.mojo). Selected via `LTX2_INT4_SLAB` for the
/// int4 W4A16-resident path. (Verified present on this box, 2026-07-11.)
const LTX2_INT4_SLAB: &str = "checkpoints/ltx-2.3-22b-distilled-svdint4-r32.safetensors";
/// wan22_t2v compiled (comptime) geometry — MUST match the binary. It raises on
/// any `frames != WAN22_FRAMES`, so the server rejects a mismatch up front rather
/// than burning a multi-minute render on a guaranteed failure.
const WAN22_FRAMES: i64 = 121;
const WAN22_WIDTH: i64 = 832;
const WAN22_HEIGHT: i64 = 480;
const WAN22_DEFAULT_STEPS: i64 = 50;
const WAN22_DEFAULT_GUIDANCE: f64 = 5.0;

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
    for path in [
        WAN22_ARTIFACT_MANIFEST,
        WAN22_FP8_CACHE,
        WAN22_UMT5_INDEX,
        WAN22_UMT5_SHARD_1,
        WAN22_UMT5_SHARD_2,
        WAN22_UMT5_SHARD_3,
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

/// Read acceptance only from the machine-local evidence gate. The report is
/// regenerated by scripts/check_wan22_product_gate.py after verifying the
/// pinned artifacts, exact cache digest, parity, representative frame bytes,
/// muxed MP4, visual inspection, wall time, and peak VRAM.
fn wan22_product_gate_passed() -> bool {
    let Ok(bytes) = std::fs::read(repo_path(WAN22_PRODUCT_GATE)) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    doc.get("schema").and_then(Value::as_str) == Some("serenity.wan22.product_gate.v1")
        && doc.get("passed").and_then(Value::as_bool) == Some(true)
        && doc.pointer("/pins/hf_revision").and_then(Value::as_str) == Some(WAN22_HF_REVISION)
        && doc
            .pointer("/pins/creator_revision")
            .and_then(Value::as_str)
            == Some(WAN22_CREATOR_REVISION)
        && doc
            .pointer("/pins/fp8_cache_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_FP8_CACHE_SHA256)
        && doc.pointer("/profile/width").and_then(Value::as_i64) == Some(WAN22_WIDTH)
        && doc.pointer("/profile/height").and_then(Value::as_i64) == Some(WAN22_HEIGHT)
        && doc.pointer("/profile/frames").and_then(Value::as_i64) == Some(WAN22_FRAMES)
        && doc.pointer("/profile/steps").and_then(Value::as_i64) == Some(WAN22_DEFAULT_STEPS)
        && doc.pointer("/profile/guidance").and_then(Value::as_f64) == Some(WAN22_DEFAULT_GUIDANCE)
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

fn readiness_doc() -> Value {
    let ltx2_runner_ready = runner_available();
    let ltx2_decode_ready = ltx2_decode_runtime_available();
    let ltx2_ready = ltx2_runner_ready && ltx2_decode_ready;
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
    // top-level state reflects whether ANY arm is runnable.
    let any_ready = ltx2_ready || wan22_ready || bernini_ready;
    let state = if any_ready {
        "bounded_smoke_ready"
    } else {
        "runner_missing"
    };
    let ltx2_status = if ltx2_ready {
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
            },
            "target_fps": 24,
            "quant_modes": ["fp8", "int4"],
            "quant_note": "staged smoke: distilled-fp8 resident or W4A16 int4-resident; refhq: official LTX-2.3 dev-fp8 base plus official 1.1 distilled LoRA. W4A4 int4-compute is NOT integrated.",
            "limit": "staged is bounded smoke; refhq parity claims are admitted only from current machine-local Creator reports at the 0.999 bar",
        },
        {
            "model": "wan22_t2v",
            "status": if wan22_product_accepted { "quality_profile_ready" } else if wan22_ready { "gate_required" } else { "prerequisites_missing" },
            "runner": WAN22_T2V,
            "encode_runner": WAN22_ENCODE,
            "missing": wan22_absent,
            "mode": "two-process: wan22_encode_prompt -> wan22_t2v",
            "artifact_root": model_path(WAN22_MODEL_ROOT).to_string_lossy(),
            "artifact_manifest": model_path(WAN22_ARTIFACT_MANIFEST).to_string_lossy(),
            "hf_revision": WAN22_HF_REVISION,
            "creator_revision": WAN22_CREATOR_REVISION,
            "product_gate": WAN22_PRODUCT_GATE,
            "accepted_video_parity": wan22_product_accepted,
            "target_width": WAN22_WIDTH,
            "target_height": WAN22_HEIGHT,
            "target_frame_count": WAN22_FRAMES,
            "default_steps": WAN22_DEFAULT_STEPS,
            "default_guidance": WAN22_DEFAULT_GUIDANCE,
            "sampler": "Flow-UniPC order 2, predict_x0, shift 5",
            "quant_modes": ["fp8"],
            "quant_note": "persistent row-scaled FP8 E4M3 DiT cache; BF16 exact tensors and BF16 on-use compute",
            "note": "Measured high-quality profile: 832x480, 121 frames, 24 fps, 50 Flow-UniPC steps, CFG 5. Geometry is comptime-compiled; frames must equal 121 without a rebuild.",
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
    ]);
    json!({
        "schema": "serenity.video_status.v1",
        "endpoint": "/v1/video",
        "state": state,
        "readiness_label": if any_ready { "bounded_daemon_smoke" } else { "build_required" },
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
        },
        "non_acceptance_reason": "bounded smoke wiring is not full SwarmUI video parity; artifact acceptance requires frame_count, duration, muxing, audio behavior, timings, and VRAM evidence",
        "probe_endpoint": "/v1/video/probe?path=<mp4>",
        "candidate_runners": runners,
    })
}

/// GET /v1/video — readiness contract.
pub async fn get_video() -> Response {
    json_resp(StatusCode::OK, &readiness_doc())
}

/// POST /v1/video — dispatch on `model`: `"ltx2"` (default) = the bounded LTX2
/// staged smoke; `"wan22"` = Wan2.2; `"bernini"` = the gated Bernini-R
/// three-process T2V orchestration.
pub async fn post_video(State(st): State<AppState>, body: String) -> Response {
    let b: Value = serde_json::from_str::<Value>(&body)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({}));
    let model = b
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("ltx2")
        .to_string();
    if model != "ltx2" && model != "wan22" && model != "bernini" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("unsupported video model '{model}'; use ltx2, wan22, or bernini"),
        );
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
    // Cross-path single-GPU lease (audit L3): a video render is minutes of GPU
    // work in a subprocess; it must not co-run with a generate/caption/magic
    // job on a 16GB card. Held (RAII) across the whole arm; 409 if busy.
    let gpu_tag = crate::gpu_lock::next_tag("video");
    let _gpu = match crate::gpu_lock::try_acquire(&st.gpu_owner, "video", &gpu_tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                axum::Json(crate::gpu_lock::gpu_busy_conflict_report("video", &cur)),
            )
                .into_response()
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
            )
        }
        Err(_) => {
            return err_detail(
                StatusCode::SERVICE_UNAVAILABLE,
                "timed out evicting idle image worker before video launch",
            )
        }
    }
    match model.as_str() {
        "ltx2" => post_video_ltx2(&st, &b),
        "wan22" => post_video_wan22(&st, &b),
        "bernini" => post_video_bernini(&st, &b),
        _ => unreachable!("video model validated before GPU acquisition"),
    }
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
    let mut command = std::process::Command::new(bin_abs);
    command
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", mojo_ld_path());
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
    // The admitted 16 GiB route is the persistent E4M3 cache. A BF16 request
    // would select the old OOM-prone profile and must fail before GPU launch.
    let quant = s("quant", "fp8");
    if quant != "fp8" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("wan22_t2v production profile is cached FP8 E4M3; quant '{quant}' is not admitted (use fp8)"),
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
    if width != WAN22_WIDTH || height != WAN22_HEIGHT || fps != 24 {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "wan22_t2v admitted profile is {WAN22_WIDTH}x{WAN22_HEIGHT} at 24 fps; requested {width}x{height} at {fps} fps"
            ),
        );
    }
    let steps = b
        .get("steps")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_DEFAULT_STEPS);
    if steps != WAN22_DEFAULT_STEPS {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("wan22_t2v high-quality profile requires exactly {WAN22_DEFAULT_STEPS} steps"),
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
    let t2v_log = out_dir.join("wan22_t2v.log");

    let abs_encode = repo_path(WAN22_ENCODE);
    let abs_t2v = repo_path(WAN22_T2V);

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

    // ── Step B: T2V render (comptime 832x480/121f, cached FP8 E4M3) ──
    let tb = std::time::Instant::now();
    let mut t2v = wan22_command(&abs_t2v);
    t2v.arg(&conds_s)
        .arg(&out_dir_s)
        .arg(frames.to_string())
        .arg(steps.to_string())
        .arg(seed.to_string())
        .arg(format!("{guidance}"));
    let t2v = run_logged_with_gpu_peak(&mut t2v, &t2v_log);
    let t2v_secs = tb.elapsed().as_secs_f64();
    let (t2v_rc, t2v_peak_vram_mib) = match t2v {
        Ok(measured) => measured,
        Err(e) => {
            let _ = std::fs::write(&t2v_log, format!("spawn failed: {e}"));
            (-1, None)
        }
    };
    let total_wall = enc_secs + t2v_secs;
    if t2v_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "t2v",
                "encode_exit_code": enc_rc, "t2v_exit_code": t2v_rc,
                "encode_log": enc_log.to_string_lossy(), "t2v_log": t2v_log.to_string_lossy(),
                "out_dir": out_dir_s, "conds": conds_s,
                "encode_seconds": enc_secs, "t2v_seconds": t2v_secs, "total_wall_seconds": total_wall,
                "encode_peak_vram_mib": encode_peak_vram_mib, "t2v_peak_vram_mib": t2v_peak_vram_mib,
                "error": "wan22_t2v failed; inspect t2v_log",
            }),
        );
    }

    // ── Step C: mux the frame_*.png the runner wrote into an mp4 (24fps). ──
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
        value.get("muxing").and_then(Value::as_str) == Some("probe_ok")
            && value.get("width").and_then(Value::as_i64) == Some(WAN22_WIDTH)
            && value.get("height").and_then(Value::as_i64) == Some(WAN22_HEIGHT)
            && value.get("frame_count").and_then(Value::as_i64) == Some(WAN22_FRAMES)
            && value.get("fps").and_then(Value::as_f64) == Some(24.0)
            && value.get("has_audio").and_then(Value::as_bool) == Some(false)
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
            "backend": BACKEND_NAME, "control_plane": "serenity-server", "resident": "fp8_e4m3_cached",
            "readiness_label": if parity_ok { "quality_profile_ready" } else { "product_gate_required" },
            "accepted_video_artifact": artifact_ok, "accepted_video_parity": parity_ok,
            "target_width": WAN22_WIDTH, "target_height": WAN22_HEIGHT, "frames": frames,
            "frames_written": frames_written, "mux": mux, "fps": 24,
            "steps": steps, "seed": seed, "guidance": guidance, "quant": quant,
            "negative_prompt_source": if b.get("negative_prompt").and_then(Value::as_str).is_some_and(|value| !value.trim().is_empty()) { "request" } else { "creator_default" },
            "encode_exit_code": enc_rc, "t2v_exit_code": t2v_rc,
            "out_dir": out_dir_s, "conds": conds_s, "mp4": mp4,
            "mp4_url": if artifact_ok { format!("/out/{video_id}/wan22_t2v.mp4") } else { String::new() },
            "probe": probe,
            "encode_log": enc_log.to_string_lossy(), "t2v_log": t2v_log.to_string_lossy(),
            "encode_seconds": enc_secs, "t2v_seconds": t2v_secs, "total_wall_seconds": total_wall,
            "encode_peak_vram_mib": encode_peak_vram_mib, "t2v_peak_vram_mib": t2v_peak_vram_mib,
            "note": "Wan2.2-TI2V-5B high-quality Mojo profile: official UMT5 conditioning and default negative prompt, cached FP8 E4M3 DiT, Flow-UniPC 50-step shift-5 sampling, tiled VAE decode, and verified 24 fps MP4 mux.",
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
            &format!("Bernini-R requires its bounded FP8 E4M3 expert caches; quant '{quant}' is unsupported"),
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

fn probe_video_path(mp4: &str) -> Result<Value, String> {
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
    fn readiness_shape() {
        let d = readiness_doc();
        assert_eq!(d.get("schema").unwrap(), "serenity.video_status.v1");
        assert_eq!(d.get("endpoint").unwrap(), "/v1/video");
        // bin_x resolves against the active repo root, so runner presence is
        // machine-dependent (built on the dev boxes, absent on CI).
        let ltx2_ready = runner_available() && ltx2_decode_runtime_available();
        let wan22_built = wan22_missing().is_empty();
        if !ltx2_ready && !wan22_built {
            assert_eq!(d.get("state").unwrap(), "runner_missing");
            assert_eq!(d.get("readiness_label").unwrap(), "build_required");
        }
        assert_eq!(d.get("accepted").unwrap(), false);
        assert_eq!(d.get("backend").unwrap(), "mojo");
        assert_eq!(d.get("control_plane").unwrap(), "serenity-server");
        let runners = d.get("candidate_runners").unwrap().as_array().unwrap();
        assert_eq!(runners.len(), 4);
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
        assert_eq!(runners[2].get("model").unwrap(), "wan22_t2v");
        if !wan22_built {
            assert_eq!(runners[2].get("status").unwrap(), "prerequisites_missing");
            assert!(!runners[2]
                .get("missing")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty());
        }
        assert_eq!(runners[2].get("target_frame_count").unwrap(), 121);
        assert_eq!(runners[2].get("target_width").unwrap(), 832);
        assert_eq!(runners[2].get("target_height").unwrap(), 480);
        assert_eq!(runners[2].get("default_steps").unwrap(), 50);
        assert_eq!(runners[2].get("default_guidance").unwrap(), 5.0);
        assert_eq!(runners[2].get("quant_modes").unwrap(), &json!(["fp8"]));
        assert_eq!(
            runners[2].get("accepted_video_parity").unwrap(),
            &(wan22_built && wan22_product_gate_passed())
        );
        // both arms report readiness under arms_ready, matching disk state
        let arms = d.get("arms_ready").unwrap();
        assert_eq!(arms.get("ltx2_t2v_av").unwrap(), ltx2_ready);
        assert_eq!(arms.get("wan22_t2v").unwrap(), wan22_built);
        let bernini_built = bernini_missing().is_empty();
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
    }

    #[test]
    fn fps_parse() {
        assert_eq!(fps_from_rate("24/1"), 24.0);
        assert_eq!(fps_from_rate("30000/1001"), 30000.0 / 1001.0);
        assert_eq!(fps_from_rate("0/0"), 0.0);
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
