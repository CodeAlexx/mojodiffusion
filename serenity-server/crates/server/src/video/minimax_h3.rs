//! MiniMax-H3 synchronized audio/video admission and orchestration.
//!
//! H3 owns its request profiles, runtime geometry, product gates, reference
//! staging, continuation state, progress parsing, validation, and Mojo process
//! lifecycle. Shared filesystem, hashing, probing, and response helpers remain
//! in the parent video module.

use super::*;

pub(super) const MINIMAX_H3_REQUEST_PROFILES_JSON: &str =
    include_str!("../../../../../serenitymojo/configs/minimax_h3_request_profiles.json");

pub(super) const MINIMAX_H3_REQUEST_RUNNER: &str = "output/bin/minimax_h3_serenity_runtime";
pub(super) const MINIMAX_H3_INT8_FAST_SHIM: &str = "output/lib/libserenity_minimax_h3_int8.so";
pub(super) const MINIMAX_H3_PRODUCT_GATE: &str = "output/checks/minimax_h3_product_gate.json";
pub(super) const MINIMAX_H3_MODEL_ROOT: &str = "checkpoints/MiniMax-H3/FL2VA";
pub(super) const MINIMAX_H3_ENCODER_CACHE: &str =
    "checkpoints/MiniMax-H3/FL2VA/text_encoder/serenity_int8_rowscale_v1";
pub(super) const MINIMAX_H3_CONDITIONING_CACHE: &str =
    "checkpoints/MiniMax-H3/FL2VA/serenity_runtime_cache_v1/conditioning_ff21f1ebd1c73098_int8_bf16_output.bin";
pub(super) const MINIMAX_H3_MODULATION_CACHE: &str =
    "checkpoints/MiniMax-H3/FL2VA/serenity_runtime_cache_v1/modcache_steps_20_blocks_50.safetensors";
pub(super) const MINIMAX_H3_INT8_RESIDENT_CACHE: &str =
    "checkpoints/MiniMax-H3/FL2VA/serenity_runtime_cache_v1/resident_groupwise_q16_o64_fc132_fc264_blocks_48.safetensors";
pub(super) const MINIMAX_H3_INT8_FAST_RESIDENT_CACHE: &str =
    "checkpoints/MiniMax-H3/FL2VA/serenity_runtime_cache_v1/resident_w8a8_row_blocks_50.safetensors";
pub(super) const MINIMAX_H3_WIDTH: i64 = 512;
pub(super) const MINIMAX_H3_HEIGHT: i64 = 320;
pub(super) const MINIMAX_H3_FRAMES: i64 = 175;
pub(super) const MINIMAX_H3_FPS: i64 = 24;

// ── Warm denoise worker (Phase A; OPT-IN via SERENITY_H3_WARM=1) ─────────────
// One resident `--serve` runtime holds process boot, preflight, JIT-cache
// checks, and the CUDA context across denoise jobs. Per-job progress stays
// log-driven: the worker dup2s its stdout onto each job's runner.log, so the
// existing polling contract is untouched. Success is still latents +
// motion_context on disk; failure is the "[serve] job FAILED" sentinel or a
// dead worker. Deferred video decode keeps its fresh process.
static H3_WARM_WORKER: std::sync::Mutex<Option<std::process::Child>> = std::sync::Mutex::new(None);

pub(super) fn minimax_h3_warm_enabled() -> bool {
    std::env::var("SERENITY_H3_WARM").ok().as_deref() == Some("1")
}

fn minimax_h3_warm_alive() -> bool {
    let mut guard = H3_WARM_WORKER.lock().unwrap();
    match guard.as_mut() {
        Some(child) => match child.try_wait() {
            Ok(None) => true,
            _ => {
                *guard = None;
                false
            }
        },
        None => false,
    }
}

pub(super) fn minimax_h3_warm_kill() {
    let mut guard = H3_WARM_WORKER.lock().unwrap();
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn minimax_h3_warm_submit(runner: &str, args: &[String]) -> std::io::Result<()> {
    use std::io::Write;
    let mut guard = H3_WARM_WORKER.lock().unwrap();
    if let Some(child) = guard.as_mut() {
        if child.try_wait()?.is_some() {
            *guard = None;
        }
    }
    if guard.is_none() {
        let serve_log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(repo_path("output/logs/minimax_h3_warm_worker.log"))?;
        let serve_err = serve_log.try_clone()?;
        let mut cmd = minimax_h3_capped_command(&repo_path(runner));
        cmd.current_dir(repo_root())
            // Flat memory band (MemoryHigh == MemoryMax): the default 10G
            // MemoryHigh reclaim band makes pinned host allocations fail
            // under pressure and surface as CUDA_ERROR_OUT_OF_MEMORY
            // (MJ-1140 signature; the identical job passes standalone with
            // the flat band and fails under the server's 10G-high).
            .env("MEM_MAX", "12G")
            .env("MEM_HIGH", "12G")
            .env("LD_LIBRARY_PATH", minimax_h3_ld_path())
            .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
            .env("CUDA_MODULE_LOADING", "EAGER")
            .env("LD_BIND_NOW", "1")
            .env("CUDA_FORCE_PRELOAD_LIBRARIES", "1")
            .arg("--serve")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::from(serve_log))
            .stderr(std::process::Stdio::from(serve_err));
        tracing::info!("spawning warm MiniMax-H3 denoise worker");
        *guard = Some(cmd.spawn()?);
    }
    let line = serde_json::to_string(args)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        + "\n";
    let child = guard.as_mut().expect("warm worker present");
    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "no stdin"))?;
    stdin.write_all(line.as_bytes())?;
    stdin.flush()?;
    Ok(())
}
pub(super) const MINIMAX_H3_STEPS: i64 = 20;
pub(super) const MINIMAX_H3_TEXT_TOKENS: i64 = 241;
// The unified Mojo executable can evaluate a wider aligned tensor envelope,
// but H3-Base's released product shape policy is a finite 768p video set.
// Keep the engine envelope private and expose/admit only these resolved H3
// canvases at the product boundary.
pub(super) const MINIMAX_H3_NATIVE_RESOLUTIONS: [(&str, i64, i64); 6] = [
    ("21:9", 1536, 672),
    ("16:9", 1344, 768),
    ("4:3", 1024, 768),
    ("1:1", 768, 768),
    ("3:4", 768, 1024),
    ("9:16", 768, 1344),
];
pub(super) const MINIMAX_H3_BASE_SHORT_EDGE: i64 = 768;
pub(super) const MINIMAX_H3_NATIVE_MAX_PIXELS: i64 = 768 * 1344;
pub(super) const MINIMAX_H3_MIN_DIMENSION: i64 = 32;
pub(super) const MINIMAX_H3_MAX_DIMENSION: i64 = 2048;
pub(super) const MINIMAX_H3_DIMENSION_STEP: i64 = 32;
pub(super) const MINIMAX_H3_MIN_SECONDS: f64 = 1.0;
// H3-Base was trained and released for at most 15 seconds, but its temporal
// VAE, packed RoPE, and flash-attention path are length-dynamic.  Admit a
// deliberately bounded single-pass extrapolation instead of pretending the
// trained limit is a structural one.  Requests past 15 seconds must also fit
// beneath the sequence-token envelope proven by the largest accepted 15 s
// request; the UI computes the same resolution-dependent limit.
pub(super) const MINIMAX_H3_TRAINED_MAX_SECONDS: f64 = 15.0;
// T2VA alone admits the strict three-minute research envelope. Conditioned
// modes retain their existing 60-second cap because their dedicated runners
// still enforce the 1,450-frame envelope.
pub(super) const MINIMAX_H3_MAX_SECONDS: f64 = 180.0;
pub(super) const MINIMAX_H3_MAX_INTERNAL_FRAMES: i64 = 4323;
pub(super) const MINIMAX_H3_CONDITIONED_MAX_SECONDS: f64 = 60.0;
pub(super) const MINIMAX_H3_CONDITIONED_MAX_INTERNAL_FRAMES: i64 = 1450;
pub(super) const MINIMAX_H3_LONG_CONTEXT_MAX_SEQUENCE_TOKENS: i64 = 107_000;
pub(super) const MINIMAX_H3_MOTION_CONTEXT_DEFAULT_FRAMES: i64 = 22;
pub(super) const MINIMAX_H3_MOTION_CONTEXT_WINDOWS: [i64; 3] = [5, 22, 39];
pub(super) const MINIMAX_H3_TEST_PROMPT: &str = r#"integrated_multimodal_description: [Shot 1] Cinematic live-action. A close-up shot of a fluffy ginger tabby cat sitting on a sun-drenched wooden windowsill. The cat has bright green eyes and white paws. The cat slowly blinks and then lets out a soft, high-pitched meow. The camera pushes in with small amplitude at slow speed toward the cat's face as it tilts its head curiously to the right. At 00:06.000, the shot cuts to a medium shot from a side angle. The cat suddenly pounces forward with a quick burst of energy, batting at a floating dust mote in the sunlight with its right paw. The cat's claws retract and extend rapidly as it misses the mote and lands softly on the wood with a slight slide.

overall_soundscape: Soft ambient wind whistling outside the window, the rhythmic purring of a cat, the light thud of paws hitting a wooden surface, and the sound of fabric rustling.

non_diegetic_music: A light, plucking pizzicato string melody with a medium tempo and a playful, bouncing rhythm.
 -"#;
pub(super) const MINIMAX_H3_CONDITIONED_PROMPT_FILE: &str =
    include_str!("../../../../../serenitymojo/configs/minimax_h3_conditioned_prompt.txt");
pub(super) const MINIMAX_H3_CONDITIONED_DECODE_RUNNER: &str =
    "output/bin/minimax_h3_decode_768x768x124";
pub(super) const MINIMAX_H3_CONDITIONED_MODULATION_CACHE: &str =
    "checkpoints/MiniMax-H3/FL2VA/serenity_runtime_cache_v1/modcache_keyframe_steps_20_blocks_50.safetensors";
pub(super) const MINIMAX_H3_REF2VA_MODEL_ROOT: &str = "checkpoints/MiniMax-H3/Ref2VA";
pub(super) const MINIMAX_H3_REF2VA_MODULATION_CACHE: &str =
    "checkpoints/MiniMax-H3/Ref2VA/serenity_runtime_cache_v1/modcache_ref_image_steps_20_blocks_50.safetensors";
pub(super) const MINIMAX_H3_REF2VA_INT8_RESIDENT_CACHE: &str =
    "checkpoints/MiniMax-H3/Ref2VA/serenity_runtime_cache_v1/resident_groupwise_q16_o64_fc132_fc264_blocks_48.safetensors";
pub(super) const MINIMAX_H3_REF2VA_INT8_FAST_RESIDENT_CACHE: &str =
    "checkpoints/MiniMax-H3/Ref2VA/serenity_runtime_cache_v1/resident_w8a8_row_blocks_50.safetensors";
pub(super) const MINIMAX_H3_REF2VA_RUNTIME_CACHE_RUNNER: &str =
    "output/bin/minimax_h3_ref2va_runtime_cache";
pub(super) const MINIMAX_H3_REF2VA_IMAGE_SIDE: u32 = 768;
pub(super) const MINIMAX_H3_REF2VA_RESIDENT_SEQUENCE_LIMIT: i64 = 9_200;
// A direct 1344x768x124, S=37_951 cU-DNN gate on the 24-GiB reference host
// proved eight W8A8 blocks resident with byte-identical output. Twelve blocks
// crossed the real attention-workspace boundary. The Mojo runner applies the
// final cap from its exact prompt-token count.
pub(super) const MINIMAX_H3_FAST_RESIDENT_LONG_SEQUENCE_LIMIT: i64 = 37_951;
pub(super) const MINIMAX_H3_FAST_RESIDENT_LONG_SEQUENCE_BLOCKS: i64 = 8;
pub(super) const MINIMAX_H3_REF2VA_MAX_IMAGES: usize = 9;
pub(super) const MINIMAX_H3_REF2VA_MAX_VIDEOS: usize = 3;
pub(super) const MINIMAX_H3_REF2VA_MAX_AUDIOS: usize = 3;
pub(super) const MINIMAX_H3_REF2VA_MAX_REFERENCES: usize = 12;
pub(super) const MINIMAX_H3_REF2VA_REFERENCE_MIN_SECONDS: f64 = 2.0;
pub(super) const MINIMAX_H3_REF2VA_REFERENCE_MAX_SECONDS: f64 = 15.0;

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct MiniMaxH3RequestProfileRegistry {
    pub(super) schema: String,
    pub(super) default_profile: String,
    pub(super) runner: String,
    pub(super) quant_modes: Vec<String>,
    pub(super) profiles: Vec<MiniMaxH3RequestProfile>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub(super) struct MiniMaxH3RequestProfile {
    pub(super) id: String,
    pub(super) label: String,
    pub(super) width: i64,
    pub(super) height: i64,
    pub(super) frames: i64,
    pub(super) fps: i64,
    pub(super) steps: i64,
    pub(super) text_tokens: i64,
    pub(super) sequence_tokens: i64,
    pub(super) duration: f64,
    pub(super) quant_modes: Vec<String>,
    pub(super) fast_resident_blocks: i64,
    pub(super) quality_resident_blocks: i64,
    pub(super) gate: String,
}

#[derive(Clone, Debug)]
pub(super) struct MiniMaxH3RuntimeGeometry {
    pub(super) width: i64,
    pub(super) height: i64,
    pub(super) output_frames: i64,
    pub(super) model_output_frames: i64,
    pub(super) internal_frames: i64,
    pub(super) fps: i64,
    pub(super) duration: f64,
    pub(super) sequence_tokens: i64,
    pub(super) motion_context_frames: i64,
    pub(super) trim_start_frames: i64,
}

pub(super) fn minimax_h3_request_profile_registry() -> &'static MiniMaxH3RequestProfileRegistry {
    static REGISTRY: std::sync::OnceLock<MiniMaxH3RequestProfileRegistry> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry: MiniMaxH3RequestProfileRegistry =
            serde_json::from_str(MINIMAX_H3_REQUEST_PROFILES_JSON)
                .expect("embedded MiniMax-H3 request profile registry must be valid JSON");
        assert_eq!(
            registry.schema, "serenity.minimax_h3.request_profiles.v3",
            "embedded MiniMax-H3 request profile registry schema mismatch"
        );
        assert_eq!(
            registry.runner, MINIMAX_H3_REQUEST_RUNNER,
            "embedded MiniMax-H3 registry must select the unified request runner"
        );
        assert!(
            ["int8-fast", "int8", "bf16"]
                .iter()
                .all(|mode| registry.quant_modes.iter().any(|value| value == mode)),
            "embedded MiniMax-H3 registry must declare all runtime quant modes"
        );
        assert!(
            registry
                .profiles
                .iter()
                .any(|profile| profile.id == registry.default_profile),
            "embedded MiniMax-H3 default profile must name a registered profile"
        );
        assert!(
            registry.profiles.iter().all(|profile| {
                profile.width > 0
                    && profile.height > 0
                    && profile.frames > 0
                    && profile.fps > 0
                    && profile.steps == MINIMAX_H3_STEPS
                    && profile.text_tokens == MINIMAX_H3_TEXT_TOKENS
                    && profile.sequence_tokens > 0
                    && profile.duration > 0.0
                    && !profile.quant_modes.is_empty()
                    && profile
                        .quant_modes
                        .iter()
                        .all(|mode| registry.quant_modes.iter().any(|known| known == mode))
                    && profile.fast_resident_blocks >= 0
                    && profile.quality_resident_blocks >= 0
                    && matches!(profile.gate.as_str(), "base" | "resolution" | "duration")
            }),
            "embedded MiniMax-H3 profiles must declare complete runtime geometry"
        );
        registry
    })
}

pub(super) fn minimax_h3_default_profile() -> &'static MiniMaxH3RequestProfile {
    let registry = minimax_h3_request_profile_registry();
    registry
        .profiles
        .iter()
        .find(|profile| profile.id == registry.default_profile)
        .expect("validated MiniMax-H3 registry must contain its default profile")
}

pub(super) fn minimax_h3_request_runner<'a>(
    profile: &'a MiniMaxH3RequestProfile,
    quant: &str,
) -> Option<&'a str> {
    let registry = minimax_h3_request_profile_registry();
    registry
        .quant_modes
        .iter()
        .any(|mode| mode == quant)
        .then(|| profile.quant_modes.iter().any(|mode| mode == quant))
        .unwrap_or(false)
        .then_some(registry.runner.as_str())
}

pub(super) fn minimax_h3_align_internal_frames(frames: i64) -> i64 {
    let mut aligned = frames.max(5);
    while aligned % 17 != 5 {
        aligned += 1;
    }
    aligned
}

pub(super) fn minimax_h3_native_resolution_documents() -> Vec<Value> {
    MINIMAX_H3_NATIVE_RESOLUTIONS
        .iter()
        .map(|(aspect_ratio, width, height)| {
            json!({
                "aspect_ratio": aspect_ratio,
                "width": width,
                "height": height,
                "label": format!("{} - {}x{}", aspect_ratio, width, height),
            })
        })
        .collect()
}

pub(super) fn minimax_h3_runtime_geometry(
    body: &Value,
) -> Result<MiniMaxH3RuntimeGeometry, String> {
    let width = body.get("width").and_then(Value::as_i64).unwrap_or(0);
    let height = body.get("height").and_then(Value::as_i64).unwrap_or(0);
    if !(MINIMAX_H3_MIN_DIMENSION..=MINIMAX_H3_MAX_DIMENSION).contains(&width)
        || !(MINIMAX_H3_MIN_DIMENSION..=MINIMAX_H3_MAX_DIMENSION).contains(&height)
        || width % MINIMAX_H3_DIMENSION_STEP != 0
        || height % MINIMAX_H3_DIMENSION_STEP != 0
    {
        return Err(format!(
            "MiniMax-H3 width and height must each be {} through {} in {}-pixel steps",
            MINIMAX_H3_MIN_DIMENSION, MINIMAX_H3_MAX_DIMENSION, MINIMAX_H3_DIMENSION_STEP,
        ));
    }
    if width * height > MINIMAX_H3_NATIVE_MAX_PIXELS {
        return Err(format!(
            "MiniMax-H3 resolution exceeds the {}-pixel 24-GB product envelope",
            MINIMAX_H3_NATIVE_MAX_PIXELS,
        ));
    }
    let fps = body
        .get("fps")
        .and_then(Value::as_i64)
        .unwrap_or(MINIMAX_H3_FPS);
    if !(1..=120).contains(&fps) {
        return Err("MiniMax-H3 FPS must be an integer from 1 through 120".to_string());
    }
    let requested_output_frames = body.get("frames").and_then(Value::as_i64).unwrap_or(0);
    if requested_output_frames < 1 {
        return Err("MiniMax-H3 output frame count must be positive".to_string());
    }
    let task = minimax_h3_task(body);
    let max_seconds = if task == "t2va" {
        MINIMAX_H3_MAX_SECONDS
    } else {
        MINIMAX_H3_CONDITIONED_MAX_SECONDS
    };
    let duration = body
        .get("duration_seconds")
        .and_then(Value::as_f64)
        .unwrap_or(requested_output_frames as f64 / fps as f64);
    if !(MINIMAX_H3_MIN_SECONDS..=max_seconds).contains(&duration) {
        return Err(format!(
            "MiniMax-H3 duration must be {} through {} seconds",
            MINIMAX_H3_MIN_SECONDS, max_seconds,
        ));
    }
    // Seconds is the authored control. Derive both delivery and model timeline
    // frames from it so a stale client-side frame value can never turn a 2 s
    // request back into a fixed profile duration.
    let output_frames = (duration * fps as f64).round().max(1.0) as i64;
    let model_output_frames = (duration * MINIMAX_H3_FPS as f64).round().max(1.0) as i64;
    let motion_context_frames = if task == "continue" {
        body.get("motion_context_frames")
            .and_then(Value::as_i64)
            .unwrap_or(MINIMAX_H3_MOTION_CONTEXT_DEFAULT_FRAMES)
    } else {
        0
    };
    let trim_start_frames = motion_context_frames;
    let internal_frames =
        minimax_h3_align_internal_frames(model_output_frames + motion_context_frames);
    let max_internal_frames = if task == "t2va" {
        MINIMAX_H3_MAX_INTERNAL_FRAMES
    } else {
        MINIMAX_H3_CONDITIONED_MAX_INTERNAL_FRAMES
    };
    if internal_frames > max_internal_frames {
        return Err(format!(
            "MiniMax-H3 single-pass internal frame envelope exceeds {} frames",
            max_internal_frames,
        ));
    }
    let latent_h = height / 16;
    let latent_w = width / 16;
    let latent_frames = (internal_frames - 5) / 17 * 5 + 2;
    let audio_latents = ((internal_frames as f64 / MINIMAX_H3_FPS as f64) * 40.0).round() as i64;
    let base_sequence_tokens = MINIMAX_H3_TEXT_TOKENS
        + audio_latents * 2
        + latent_frames * (latent_h / 2) * (latent_w / 2);
    let rows_per_frame = (latent_h / 2) * (latent_w / 2);
    let sequence_tokens = match task {
        "i2va" => base_sequence_tokens - MINIMAX_H3_TEXT_TOKENS + 970 + rows_per_frame,
        "l2va" => base_sequence_tokens - MINIMAX_H3_TEXT_TOKENS + 975 + rows_per_frame,
        "fl2va" => base_sequence_tokens - MINIMAX_H3_TEXT_TOKENS + 1_581 + rows_per_frame * 2,
        "continue" => {
            let context_steps = match motion_context_frames {
                5 => 2,
                22 => 7,
                39 => 12,
                _ => 0,
            };
            let context_audio_latents =
                ((motion_context_frames as f64 / MINIMAX_H3_FPS as f64) * 40.0).round() as i64;
            base_sequence_tokens + context_steps * rows_per_frame + context_audio_latents * 2
        }
        _ => base_sequence_tokens,
    };
    if duration > MINIMAX_H3_TRAINED_MAX_SECONDS {
        if task == "ref2va" {
            return Err(
                "MiniMax-H3 Ref2VA remains limited to the trained 15-second window; use T2VA, I2VA, L2VA, or FL2VA for experimental single-pass long context"
                    .to_string(),
            );
        }
        if minimax_h3_continue_with_references(body) {
            return Err(
                "MiniMax-H3 reference-conditioned Continue segments remain limited to the trained 15-second window; chain another Continue segment for longer video"
                    .to_string(),
            );
        }
        if sequence_tokens > MINIMAX_H3_LONG_CONTEXT_MAX_SEQUENCE_TOKENS {
            return Err(format!(
                "MiniMax-H3 experimental single-pass long context at {}x{} and {:.2} seconds requires about {} packed tokens, above the {}-token 24-GB envelope; lower resolution or duration",
                width,
                height,
                duration,
                sequence_tokens,
                MINIMAX_H3_LONG_CONTEXT_MAX_SEQUENCE_TOKENS,
            ));
        }
    }
    Ok(MiniMaxH3RuntimeGeometry {
        width,
        height,
        output_frames,
        model_output_frames,
        internal_frames,
        fps,
        duration,
        sequence_tokens,
        motion_context_frames,
        trim_start_frames,
    })
}

pub(super) fn minimax_h3_runtime_resident_blocks(
    geometry: &MiniMaxH3RuntimeGeometry,
    quant: &str,
) -> i64 {
    if geometry.sequence_tokens > MINIMAX_H3_FAST_RESIDENT_LONG_SEQUENCE_LIMIT {
        return 0;
    }
    if geometry.sequence_tokens > 9_200 {
        return match quant {
            "int8-fast" => MINIMAX_H3_FAST_RESIDENT_LONG_SEQUENCE_BLOCKS,
            _ => 0,
        };
    }
    match quant {
        "int8-fast" if geometry.width * geometry.height <= 512 * 320 => 48,
        "int8-fast" => 46,
        "int8" => 41,
        _ => 0,
    }
}

pub(super) fn minimax_h3_ref2va_resident_blocks(
    geometry: &MiniMaxH3RuntimeGeometry,
    quant: &str,
) -> i64 {
    if !matches!(quant, "int8-fast" | "int8") {
        return 0;
    }
    if geometry.sequence_tokens > MINIMAX_H3_REF2VA_RESIDENT_SEQUENCE_LIMIT {
        0
    } else {
        4
    }
}

pub(super) fn minimax_h3_encoder_cache_complete() -> bool {
    let root = model_path(MINIMAX_H3_ENCODER_CACHE);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return false;
    };
    let files = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .collect::<Vec<_>>();
    files.len() == 702
        && files.iter().all(|entry| {
            let name = entry.file_name();
            !name.to_string_lossy().ends_with(".tmp")
                && entry.metadata().is_ok_and(|metadata| metadata.len() > 0)
        })
}

pub(super) fn minimax_h3_conditioned_prompt() -> &'static str {
    MINIMAX_H3_CONDITIONED_PROMPT_FILE.trim_end_matches(['\r', '\n'])
}

pub(super) fn minimax_h3_task(body: &Value) -> &str {
    body.get("task").and_then(Value::as_str).unwrap_or("t2va")
}

pub(super) fn minimax_h3_continue_with_references(body: &Value) -> bool {
    minimax_h3_task(body) == "continue"
        && body
            .get("references")
            .and_then(Value::as_array)
            .is_some_and(|rows| !rows.is_empty())
}

pub(super) fn minimax_h3_video_job_id(value: &str) -> bool {
    value.strip_prefix("video-").is_some_and(|digits| {
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    })
}

#[derive(Clone, Debug)]
pub(super) struct MiniMaxH3ContinuationSource {
    pub(super) job_id: String,
    pub(super) latent_path: std::path::PathBuf,
}

pub(super) fn minimax_h3_continuation_source(
    output_root: &std::path::Path,
    body: &Value,
) -> Result<MiniMaxH3ContinuationSource, String> {
    let job_id = body
        .get("continue_from")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| minimax_h3_video_job_id(value))
        .ok_or_else(|| {
            "MiniMax-H3 continue_from must be a local video job id such as video-0100".to_string()
        })?;
    let source_dir = output_root.join(job_id);
    let result_path = source_dir.join("result.json");
    let result: Value = std::fs::read_to_string(&result_path)
        .map_err(|error| {
            format!("MiniMax-H3 continuation source {job_id} has no readable result: {error}")
        })
        .and_then(|text| {
            serde_json::from_str(&text).map_err(|error| {
                format!("MiniMax-H3 continuation source {job_id} result is invalid: {error}")
            })
        })?;
    if result.get("state").and_then(Value::as_str) != Some("done") {
        return Err(format!(
            "MiniMax-H3 continuation source {job_id} is not complete"
        ));
    }
    let width = body.get("width").and_then(Value::as_i64).unwrap_or(0);
    let height = body.get("height").and_then(Value::as_i64).unwrap_or(0);
    if result.get("width").and_then(Value::as_i64) != Some(width)
        || result.get("height").and_then(Value::as_i64) != Some(height)
    {
        return Err(format!(
            "MiniMax-H3 continuation must keep the source resolution; {job_id} is {}x{}, request is {width}x{height}",
            result.get("width").and_then(Value::as_i64).unwrap_or(0),
            result.get("height").and_then(Value::as_i64).unwrap_or(0),
        ));
    }
    let compact = source_dir.join("motion_context.safetensors");
    let legacy = source_dir.join("latents.safetensors");
    let latent_path = if nonempty_file(&compact) {
        compact
    } else if nonempty_file(&legacy) {
        legacy
    } else {
        return Err(format!(
            "MiniMax-H3 continuation source {job_id} has no retained native latent tail; use the last-frame I2VA fallback for this older job"
        ));
    };
    Ok(MiniMaxH3ContinuationSource {
        job_id: job_id.to_string(),
        latent_path,
    })
}

pub(super) fn minimax_h3_conditioned_runner(task: &str, quant: &str) -> Option<String> {
    if !matches!(task, "i2va" | "l2va" | "fl2va" | "ref2va")
        || !matches!(quant, "bf16" | "int8" | "int8-fast")
    {
        return None;
    }
    let suffix = if quant == "int8-fast" {
        "int8_fast"
    } else {
        quant
    };
    Some(if task == "ref2va" {
        format!("output/bin/minimax_h3_ref2va_768x768x124_{suffix}")
    } else {
        format!("output/bin/minimax_h3_{task}_768x768x124_{suffix}")
    })
}

pub(super) fn minimax_h3_request_media_path(
    body: &Value,
    key: &str,
) -> Result<std::path::PathBuf, String> {
    let raw = body
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("MiniMax-H3 {key} is required for this task"))?;
    if raw.contains(['\r', '\n']) {
        return Err(format!("MiniMax-H3 {key} must be one filesystem path"));
    }
    let path = if std::path::Path::new(raw).is_absolute() {
        std::path::PathBuf::from(raw)
    } else {
        repo_path(raw)
    };
    if !nonempty_file(&path) {
        return Err(format!(
            "MiniMax-H3 {key} is not a readable non-empty file: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[derive(Clone, Debug)]
pub(super) struct MiniMaxH3ReferenceInput {
    pub(super) kind: String,
    pub(super) path: std::path::PathBuf,
    pub(super) audio_use: String,
    pub(super) has_audio: bool,
    pub(super) duration: f64,
}

pub(super) fn minimax_h3_reference_media_path(
    raw: &str,
    label: &str,
) -> Result<std::path::PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() || raw.contains(['\r', '\n']) {
        return Err(format!("MiniMax-H3 {label} must be one filesystem path"));
    }
    let path = if std::path::Path::new(raw).is_absolute() {
        std::path::PathBuf::from(raw)
    } else {
        repo_path(raw)
    };
    if !nonempty_file(&path) {
        return Err(format!(
            "MiniMax-H3 {label} is not a readable non-empty file: {}",
            path.display()
        ));
    }
    Ok(path)
}

pub(super) fn minimax_h3_ref2va_references(
    body: &Value,
) -> Result<Vec<MiniMaxH3ReferenceInput>, String> {
    let mut references = Vec::new();
    if let Some(value) = body.get("references") {
        let rows = value
            .as_array()
            .ok_or_else(|| "MiniMax-H3 references must be an ordered array".to_string())?;
        for (index, row) in rows.iter().enumerate() {
            let kind = row.get("kind").and_then(Value::as_str).unwrap_or("").trim();
            if !matches!(kind, "image" | "video" | "audio") {
                return Err(format!(
                    "MiniMax-H3 reference {} kind must be 'image', 'video', or 'audio'",
                    index + 1,
                ));
            }
            let raw_path = row.get("path").and_then(Value::as_str).unwrap_or("");
            let path = minimax_h3_reference_media_path(
                raw_path,
                &format!("reference {} path", index + 1),
            )?;
            let audio_use = row
                .get("audio_use")
                .and_then(Value::as_str)
                .unwrap_or("reference")
                .trim();
            if !matches!(audio_use, "reference" | "reuse" | "voice_timbre") {
                return Err(format!(
                    "MiniMax-H3 reference {} audio_use must be 'reference', 'reuse', or 'voice_timbre'",
                    index + 1,
                ));
            }
            references.push(MiniMaxH3ReferenceInput {
                kind: kind.to_string(),
                path,
                audio_use: audio_use.to_string(),
                has_audio: false,
                duration: 0.0,
            });
        }
    }
    if references.is_empty()
        && body
            .get("source_image")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    {
        // Backward-compatible canvas/API contract: the legacy source image is
        // the single identity/style reference and retains its accepted square
        // preprocessing policy.
        references.push(MiniMaxH3ReferenceInput {
            kind: "image".to_string(),
            path: minimax_h3_request_media_path(body, "source_image")?,
            audio_use: "reference".to_string(),
            has_audio: false,
            duration: 0.0,
        });
    }

    if references.is_empty() {
        return Err("MiniMax-H3 Ref2VA needs at least one reference".to_string());
    }
    if references.len() > MINIMAX_H3_REF2VA_MAX_REFERENCES {
        return Err(format!(
            "MiniMax-H3 Ref2VA accepts at most {} combined references",
            MINIMAX_H3_REF2VA_MAX_REFERENCES,
        ));
    }
    let images = references.iter().filter(|row| row.kind == "image").count();
    let videos = references.iter().filter(|row| row.kind == "video").count();
    let audios = references.iter().filter(|row| row.kind == "audio").count();
    if images > MINIMAX_H3_REF2VA_MAX_IMAGES {
        return Err(format!(
            "MiniMax-H3 Ref2VA accepts at most {} image references",
            MINIMAX_H3_REF2VA_MAX_IMAGES,
        ));
    }
    if videos > MINIMAX_H3_REF2VA_MAX_VIDEOS {
        return Err(format!(
            "MiniMax-H3 Ref2VA accepts at most {} video references",
            MINIMAX_H3_REF2VA_MAX_VIDEOS,
        ));
    }
    if audios > MINIMAX_H3_REF2VA_MAX_AUDIOS {
        return Err(format!(
            "MiniMax-H3 Ref2VA accepts at most {} audio references",
            MINIMAX_H3_REF2VA_MAX_AUDIOS,
        ));
    }
    if images == 0 && videos == 0 {
        return Err(
            "MiniMax-H3 audio references must accompany at least one image or video reference"
                .to_string(),
        );
    }

    for (index, reference) in references.iter_mut().enumerate() {
        if reference.kind == "image" {
            continue;
        }
        let probe = probe_video_path(&reference.path.to_string_lossy())
            .map_err(|error| format!("cannot probe MiniMax-H3 reference {}: {error}", index + 1))?;
        let has_video = probe.get("has_video").and_then(Value::as_bool) == Some(true);
        let has_audio = probe.get("has_audio").and_then(Value::as_bool) == Some(true);
        reference.has_audio = has_audio;
        let duration = if reference.kind == "audio" {
            probe
                .get("audio_duration")
                .and_then(Value::as_f64)
                .filter(|duration| *duration > 0.0)
                .or_else(|| probe.get("duration").and_then(Value::as_f64))
                .unwrap_or(0.0)
        } else {
            probe.get("duration").and_then(Value::as_f64).unwrap_or(0.0)
        };
        reference.duration = duration;
        if reference.kind == "video" && !has_video {
            return Err(format!(
                "MiniMax-H3 reference {} has no video stream",
                index + 1
            ));
        }
        if reference.kind == "audio" && !has_audio {
            return Err(format!(
                "MiniMax-H3 reference {} has no audio stream",
                index + 1
            ));
        }
        if !(MINIMAX_H3_REF2VA_REFERENCE_MIN_SECONDS..=MINIMAX_H3_REF2VA_REFERENCE_MAX_SECONDS)
            .contains(&duration)
        {
            return Err(format!(
                "MiniMax-H3 {} reference {} must be 2 through 15 seconds; got {:.3}",
                reference.kind,
                index + 1,
                duration,
            ));
        }
    }
    Ok(references)
}

pub(super) fn minimax_h3_ref2va_prompt_with_audio_roles(
    prompt: &str,
    references: &[MiniMaxH3ReferenceInput],
) -> String {
    let mut directives = Vec::new();
    let mut audio_index = 0;
    for reference in references {
        if !reference.has_audio && reference.kind != "audio" {
            continue;
        }
        audio_index += 1;
        let directive = match reference.audio_use.as_str() {
            "reuse" => format!(
                "<Audio {audio_index}>: partially_copy - reuse this audio in the target soundtrack where instructed."
            ),
            "voice_timbre" => format!(
                "<Audio {audio_index}>: reference - use its voice timbre for generated speech, without copying its spoken content."
            ),
            _ => format!(
                "<Audio {audio_index}>: reference - use this as an audio-conditioning reference where relevant."
            ),
        };
        directives.push(directive);
    }
    if directives.is_empty() {
        return prompt.to_string();
    }
    format!(
        "reference_audio_intent:\n{}\n\n{}",
        directives.join("\n"),
        prompt,
    )
}

pub(super) fn stage_minimax_h3_ref2va_audio_reference(
    source_path: &std::path::Path,
    out_dir: &std::path::Path,
    index: usize,
) -> Result<std::path::PathBuf, String> {
    let prepared_path = out_dir.join(format!("ref_audio_input_{index}.wav"));
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &source_path.to_string_lossy(),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "32000",
            "-ac",
            "2",
            &prepared_path.to_string_lossy(),
        ])
        .output()
        .map_err(|error| format!("cannot launch ffmpeg for H3 audio reference: {error}"))?;
    if !output.status.success() || !nonempty_file(&prepared_path) {
        return Err(format!(
            "cannot prepare MiniMax-H3 audio reference '{}': {}",
            source_path.display(),
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    Ok(prepared_path)
}

pub(super) fn stage_minimax_h3_ref2va_image_reference(
    source_path: &std::path::Path,
    out_dir: &std::path::Path,
) -> Result<(std::path::PathBuf, Value), String> {
    // The quality-gated Ref2VA image product consumes one 48x48 vision grid.
    // Browser uploads may have any ordinary photo aspect ratio, so stage a
    // deterministic square identity crop before the Mojo process starts.
    // This is pixel preprocessing only; every model stage remains on GPU.
    let source = image::open(source_path)
        .map_err(|error| {
            format!(
                "cannot decode MiniMax-H3 Ref2VA source '{}': {error}",
                source_path.display(),
            )
        })?
        .to_rgb8();
    let source_width = source.width();
    let source_height = source.height();
    if source_width == 0 || source_height == 0 {
        return Err("MiniMax-H3 Ref2VA source dimensions must be positive".to_string());
    }
    let crop_side = source_width.min(source_height);
    let crop_left = (source_width - crop_side) / 2;
    let crop_top = (source_height - crop_side) / 2;
    let cropped =
        image::imageops::crop_imm(&source, crop_left, crop_top, crop_side, crop_side).to_image();
    let prepared = image::imageops::resize(
        &cropped,
        MINIMAX_H3_REF2VA_IMAGE_SIDE,
        MINIMAX_H3_REF2VA_IMAGE_SIDE,
        image::imageops::FilterType::Lanczos3,
    );
    let prepared_path = out_dir.join("ref_product_input_0.png");
    prepared
        .save_with_format(&prepared_path, image::ImageFormat::Png)
        .map_err(|error| {
            format!(
                "cannot save MiniMax-H3 Ref2VA prepared source '{}': {error}",
                prepared_path.display(),
            )
        })?;
    let metadata = json!({
        "policy": "cover_center_crop_square_768",
        "operation": "pixel_preprocess_only",
        "model_inference": "gpu",
        "original_path": source_path,
        "original_width": source_width,
        "original_height": source_height,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_width": crop_side,
        "crop_height": crop_side,
        "prepared_path": prepared_path,
        "prepared_width": MINIMAX_H3_REF2VA_IMAGE_SIDE,
        "prepared_height": MINIMAX_H3_REF2VA_IMAGE_SIDE,
    });
    Ok((prepared_path, metadata))
}

pub(super) fn minimax_h3_conditioned_missing(task: &str, quant: &str) -> Vec<String> {
    let mut missing = Vec::new();
    match minimax_h3_conditioned_runner(task, quant) {
        Some(runner) if bin_x(&runner) => {}
        Some(runner) => missing.push(runner),
        None => missing.push(format!("unsupported MiniMax-H3 task={task} quant={quant}")),
    }
    if !bin_x(MINIMAX_H3_CONDITIONED_DECODE_RUNNER) {
        missing.push(MINIMAX_H3_CONDITIONED_DECODE_RUNNER.to_string());
    }
    let root = model_path(if task == "ref2va" {
        MINIMAX_H3_REF2VA_MODEL_ROOT
    } else {
        MINIMAX_H3_MODEL_ROOT
    });
    for relative in [
        "transformer/model.safetensors.index.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/config.json",
        "processor/preprocessor_config.json",
        "audio_vae/model.safetensors",
        "video_vae/source/model.safetensors",
    ] {
        let path = root.join(relative);
        if !nonempty_file(&path) {
            missing.push(path.to_string_lossy().into_owned());
        }
    }
    if !minimax_h3_encoder_cache_complete() {
        missing.push(
            model_path(MINIMAX_H3_ENCODER_CACHE)
                .to_string_lossy()
                .into_owned(),
        );
    }
    if quant != "bf16" {
        let cache_builder = if task == "ref2va" {
            MINIMAX_H3_REF2VA_RUNTIME_CACHE_RUNNER
        } else {
            MINIMAX_H3_REQUEST_RUNNER
        };
        if minimax_h3_resident_cache_path(quant, task == "ref2va")
            .is_some_and(|path| !nonempty_file(&path))
            && !bin_x(cache_builder)
        {
            missing.push(cache_builder.to_string());
        }
    }
    // Modulation and resident stores are generated acceleration caches. They
    // are not model files and must never prevent the installed model or its
    // controls from loading.
    for runtime in [MINIMAX_H3_INT8_FAST_SHIM, LTX2_CSHIM] {
        let path = repo_path(runtime);
        if !nonempty_file(&path) {
            missing.push(path.to_string_lossy().into_owned());
        }
    }
    missing
}

pub(super) fn minimax_h3_missing(profile: &MiniMaxH3RequestProfile, quant: &str) -> Vec<String> {
    let mut missing = Vec::new();
    let runner = minimax_h3_request_runner(profile, quant);
    if let Some(runner) = runner {
        if !bin_x(runner) {
            missing.push(runner.to_string());
        }
    } else {
        missing.push(format!("{} runner for quant={quant}", profile.id));
    }
    let root = model_path(MINIMAX_H3_MODEL_ROOT);
    for relative in [
        "transformer/model.safetensors.index.json",
        "text_encoder/model.safetensors.index.json",
        "audio_vae/model.safetensors",
        "video_vae/source/model.safetensors",
    ] {
        let path = root.join(relative);
        if !nonempty_file(&path) {
            missing.push(path.to_string_lossy().into_owned());
        }
    }
    if !minimax_h3_encoder_cache_complete() {
        missing.push(
            model_path(MINIMAX_H3_ENCODER_CACHE)
                .to_string_lossy()
                .into_owned(),
        );
    }
    // The compiled runner links both runtime libraries in every precision
    // mode. These and the actual weights are hard prerequisites; generated
    // conditioning/modulation/resident caches and quality reports are not.
    for runtime in [MINIMAX_H3_INT8_FAST_SHIM, LTX2_CSHIM] {
        let path = repo_path(runtime);
        if !nonempty_file(&path) {
            missing.push(path.to_string_lossy().into_owned());
        }
    }
    missing
}

pub(super) fn minimax_h3_profile_mode_supported(
    profile: &MiniMaxH3RequestProfile,
    quant: &str,
) -> bool {
    profile.quant_modes.iter().any(|mode| mode == quant)
}

pub(super) fn minimax_h3_profile_mode_ready(
    profile: &MiniMaxH3RequestProfile,
    quant: &str,
) -> bool {
    minimax_h3_profile_mode_supported(profile, quant)
        && minimax_h3_missing(profile, quant).is_empty()
}

pub(super) fn minimax_h3_profile_document(profile: &MiniMaxH3RequestProfile) -> Value {
    let fast_ready = minimax_h3_profile_mode_ready(profile, "int8-fast");
    let quality_ready = minimax_h3_profile_mode_ready(profile, "int8");
    let bf16_ready = minimax_h3_profile_mode_ready(profile, "bf16");
    json!({
        "id": profile.id,
        "label": profile.label,
        "width": profile.width,
        "height": profile.height,
        "frames": profile.frames,
        "fps": profile.fps,
        "steps": profile.steps,
        "text_tokens": profile.text_tokens,
        "sequence_tokens": profile.sequence_tokens,
        "duration": profile.duration,
        "quant_modes": profile.quant_modes,
        "available": fast_ready || quality_ready || bf16_ready,
        "available_modes": {
            "int8-fast": fast_ready,
            "int8": quality_ready,
            "bf16": bf16_ready,
        },
        "resident_blocks": {
            "int8-fast": profile.fast_resident_blocks,
            "int8": profile.quality_resident_blocks,
        },
        "runner": minimax_h3_request_profile_registry().runner,
        "memory_policy": "independent_resolution_duration_matrix",
    })
}

pub(super) fn minimax_h3_conditioned_task_document(task: &str, label: &str) -> Value {
    let fast_missing = minimax_h3_conditioned_missing(task, "int8-fast");
    let quality_missing = minimax_h3_conditioned_missing(task, "int8");
    let bf16_missing = minimax_h3_conditioned_missing(task, "bf16");
    let fast_ready = fast_missing.is_empty();
    let quality_ready = quality_missing.is_empty();
    let bf16_ready = bf16_missing.is_empty();
    json!({
        "id": task,
        "label": label,
        "available": fast_ready || quality_ready || bf16_ready,
        "available_modes": {
            "int8-fast": fast_ready,
            "int8": quality_ready,
            "bf16": bf16_ready,
        },
        "missing": {
            "int8-fast": fast_missing,
            "int8": quality_missing,
            "bf16": bf16_missing,
        },
        "runners": {
            "int8-fast": minimax_h3_conditioned_runner(task, "int8-fast"),
            "int8": minimax_h3_conditioned_runner(task, "int8"),
            "bf16": minimax_h3_conditioned_runner(task, "bf16"),
        },
        "geometry": {
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
            "seconds_max": if task == "ref2va" {
                MINIMAX_H3_TRAINED_MAX_SECONDS
            } else {
                MINIMAX_H3_CONDITIONED_MAX_SECONDS
            },
            "trained_seconds_max": MINIMAX_H3_TRAINED_MAX_SECONDS,
            "long_context_max_sequence_tokens": MINIMAX_H3_LONG_CONTEXT_MAX_SEQUENCE_TOKENS,
            "long_context_policy": if task == "ref2va" {
                "unavailable_for_variable_reference_pack"
            } else {
                "experimental_single_pass_resolution_duration_tradeoff"
            },
            "fps_min": 1,
            "fps_max": 120,
        },
        "steps": 20,
        "include_audio": true,
        "gpu_vision_only": true,
        "reference_only": task == "ref2va",
        "reference_inputs": if task == "ref2va" {
            Some(json!({
                "ordered": true,
                "max_combined": MINIMAX_H3_REF2VA_MAX_REFERENCES,
                "image_max": MINIMAX_H3_REF2VA_MAX_IMAGES,
                "video_max": MINIMAX_H3_REF2VA_MAX_VIDEOS,
                "audio_max": MINIMAX_H3_REF2VA_MAX_AUDIOS,
                "video_seconds_each": [MINIMAX_H3_REF2VA_REFERENCE_MIN_SECONDS, MINIMAX_H3_REF2VA_REFERENCE_MAX_SECONDS],
                "audio_seconds_each": [MINIMAX_H3_REF2VA_REFERENCE_MIN_SECONDS, MINIMAX_H3_REF2VA_REFERENCE_MAX_SECONDS],
                "duration_policy": "each reference is independently truncated to the generated duration",
                "audio_requires_visual_reference": true,
                "audio_use": ["reference", "reuse", "voice_timbre"],
                "model_inference": "gpu",
            }))
        } else {
            None
        },
        "source_becomes_first_frame": matches!(task, "i2va" | "fl2va"),
        "reference_image_short_edge": if task == "ref2va" { Some(768) } else { None },
        "reference_image_preprocess": if task == "ref2va" {
            Some(json!({
                "policy": "cover_center_crop_square_768",
                "operation": "pixel_preprocess_only",
                "model_inference": "gpu",
            }))
        } else {
            None
        },
        "memory_policy": if task == "ref2va" {
            Some(json!({
                "policy": "sequence_adaptive_w8a8_streaming",
                "resident_blocks_short_sequence": 4,
                "resident_blocks_long_sequence": 0,
                "target_sequence_limit": MINIMAX_H3_REF2VA_RESIDENT_SEQUENCE_LIMIT,
                "model_inference": "gpu",
            }))
        } else {
            None
        },
        "cache_policy": if task == "ref2va" { "ref2va_dit_cache_and_shared_identical_encoder_cache" } else { "reuse_existing_fl2va_resident_cache" },
        "modulation_cache": if task == "ref2va" { MINIMAX_H3_REF2VA_MODULATION_CACHE } else { MINIMAX_H3_CONDITIONED_MODULATION_CACHE },
        "encoder_storage": "row_scaled_int8_weights_bf16_outputs",
    })
}

pub(super) fn minimax_h3_continuation_task_document() -> Value {
    let profile = minimax_h3_default_profile();
    let fast_missing = minimax_h3_missing(profile, "int8-fast");
    let quality_missing = minimax_h3_missing(profile, "int8");
    let bf16_missing = minimax_h3_missing(profile, "bf16");
    json!({
        "id": "continue",
        "label": "Continue previous H3 video + audio",
        "available": fast_missing.is_empty() || quality_missing.is_empty() || bf16_missing.is_empty(),
        "available_modes": {
            "int8-fast": fast_missing.is_empty(),
            "int8": quality_missing.is_empty(),
            "bf16": bf16_missing.is_empty(),
        },
        "missing": {
            "int8-fast": fast_missing,
            "int8": quality_missing,
            "bf16": bf16_missing,
        },
        "runner": MINIMAX_H3_REQUEST_RUNNER,
        "geometry": {
            "shape_policy": "same_resolution_as_source_latent",
            "base_short_edge": MINIMAX_H3_BASE_SHORT_EDGE,
            "max_pixels": MINIMAX_H3_NATIVE_MAX_PIXELS,
            "width_min": MINIMAX_H3_MIN_DIMENSION,
            "width_max": MINIMAX_H3_MAX_DIMENSION,
            "height_min": MINIMAX_H3_MIN_DIMENSION,
            "height_max": MINIMAX_H3_MAX_DIMENSION,
            "dimension_step": MINIMAX_H3_DIMENSION_STEP,
            "resolutions": minimax_h3_native_resolution_documents(),
            "resolution_role": "must_match_source_job",
            "seconds_min": MINIMAX_H3_MIN_SECONDS,
            "seconds_max": MINIMAX_H3_CONDITIONED_MAX_SECONDS,
            "trained_seconds_max": MINIMAX_H3_TRAINED_MAX_SECONDS,
            "long_context_max_sequence_tokens": MINIMAX_H3_LONG_CONTEXT_MAX_SEQUENCE_TOKENS,
            "long_context_policy": "experimental_single_pass_plus_pinned_overlap",
            "fps_min": 1,
            "fps_max": 120,
        },
        "motion_context": {
            "native_latent_tail": true,
            "preserve_video": true,
            "preserve_audio": true,
            "windows": MINIMAX_H3_MOTION_CONTEXT_WINDOWS,
            "default_frames": MINIMAX_H3_MOTION_CONTEXT_DEFAULT_FRAMES,
            "trim_overlap": true,
            "fallback": "decoded_last_frame_i2va_for_legacy_jobs",
        },
        "include_audio": true,
        "gpu_only": true,
    })
}

pub(super) fn minimax_h3_ld_path() -> std::ffi::OsString {
    let mut paths = vec![
        repo_path("output/lib"),
        repo_path("serenitymojo/ops/cshim/lib"),
    ];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).unwrap_or_default()
}

/// Keep safetensor page cache from placing the whole desktop in oomd's kill
/// path. Model kernels still execute on GPU; this cap governs host RSS and
/// file-backed cache for the runner process tree.
pub(super) fn minimax_h3_capped_command(runner: &std::path::Path) -> std::process::Command {
    let mut command = std::process::Command::new(repo_path("scripts/mem_safe.sh"));
    command
        .env("MEM_MAX", "12G")
        .env("MEM_HIGH", "10G")
        .env("SWAP_MAX", "2G")
        .arg(runner);
    command
}

pub(super) fn minimax_h3_resident_cache_path(
    quant: &str,
    ref2va: bool,
) -> Option<std::path::PathBuf> {
    let relative = match (ref2va, quant) {
        (false, "int8-fast") => MINIMAX_H3_INT8_FAST_RESIDENT_CACHE,
        (false, "int8") => MINIMAX_H3_INT8_RESIDENT_CACHE,
        (true, "int8-fast") => MINIMAX_H3_REF2VA_INT8_FAST_RESIDENT_CACHE,
        (true, "int8") => MINIMAX_H3_REF2VA_INT8_RESIDENT_CACHE,
        _ => return None,
    };
    Some(model_path(relative))
}

fn prepare_minimax_h3_resident_cache_if_needed(
    quant: &str,
    ref2va: bool,
    job_dir: &std::path::Path,
) -> Result<bool, String> {
    let Some(cache_path) = minimax_h3_resident_cache_path(quant, ref2va) else {
        return Ok(false);
    };
    if nonempty_file(&cache_path) {
        return Ok(false);
    }

    let prep_dir = job_dir.join("runtime-cache-build");
    std::fs::create_dir_all(&prep_dir)
        .map_err(|error| format!("cannot create H3 runtime-cache build directory: {error}"))?;
    let log_path = prep_dir.join(format!("{quant}.log"));
    let log = std::fs::File::create(&log_path)
        .map_err(|error| format!("cannot create H3 runtime-cache log: {error}"))?;
    let stderr = log
        .try_clone()
        .map_err(|error| format!("cannot clone H3 runtime-cache log: {error}"))?;

    let mut command = if ref2va {
        let runner = repo_path(MINIMAX_H3_REF2VA_RUNTIME_CACHE_RUNNER);
        if !bin_x(MINIMAX_H3_REF2VA_RUNTIME_CACHE_RUNNER) {
            return Err(format!(
                "MiniMax-H3 Ref2VA cache builder is missing: {}",
                runner.display()
            ));
        }
        let mut command = minimax_h3_capped_command(&runner);
        command.arg(if quant == "int8-fast" {
            "w8a8"
        } else {
            "groupwise"
        });
        command
    } else {
        let runner = repo_path(MINIMAX_H3_REQUEST_RUNNER);
        let resident_blocks = if quant == "int8-fast" { 50 } else { 48 };
        let mut command = minimax_h3_capped_command(&runner);
        command
            .arg(MINIMAX_H3_TEST_PROMPT)
            .arg(&prep_dir)
            .arg(MINIMAX_H3_STEPS.to_string())
            .arg("0")
            .arg("50")
            .arg(format!("--width={MINIMAX_H3_WIDTH}"))
            .arg(format!("--height={MINIMAX_H3_HEIGHT}"))
            .arg(format!("--frames={MINIMAX_H3_FRAMES}"))
            .arg(format!("--output-frames={MINIMAX_H3_FRAMES}"))
            .arg(format!("--fps={MINIMAX_H3_FPS}"))
            .arg(format!("--output-fps={MINIMAX_H3_FPS}"))
            .arg(format!("--quant={quant}"))
            .arg(format!("--resident-blocks={resident_blocks}"))
            .arg("--encoder-storage=int8")
            .arg("--attention-backend=cudnn")
            .arg("--step-cache=exact")
            .arg("--runtime-cache-exact-product-prompt")
            .arg("--prepare-runtime-cache");
        command
    };
    let status = command
        .current_dir(repo_root())
        .env("LD_LIBRARY_PATH", minimax_h3_ld_path())
        .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
        .env("CUDA_MODULE_LOADING", "EAGER")
        .env("LD_BIND_NOW", "1")
        .env("CUDA_FORCE_PRELOAD_LIBRARIES", "1")
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(stderr))
        .status()
        .map_err(|error| format!("cannot start H3 runtime-cache builder: {error}"))?;
    if !status.success() || !nonempty_file(&cache_path) {
        return Err(format!(
            "H3 {quant} runtime-cache build failed with {status}; inspect {}",
            log_path.display()
        ));
    }
    Ok(true)
}

pub(super) fn write_minimax_h3_job_status(
    out_dir: &std::path::Path,
    state: &str,
    phase: &str,
    step: i64,
    total: i64,
    message: &str,
) -> Result<(), String> {
    let path = out_dir.join("status.json");
    let tmp = out_dir.join("status.json.tmp");
    let document = json!({
        "schema": "serenity.minimax_h3.status.v1",
        "state": state,
        "phase": phase,
        "step": step,
        "total": total,
        "message": message,
    });
    let bytes = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("serialize MiniMax-H3 status: {error}"))?;
    std::fs::write(&tmp, bytes).map_err(|error| format!("write MiniMax-H3 status: {error}"))?;
    std::fs::rename(&tmp, &path).map_err(|error| format!("publish MiniMax-H3 status: {error}"))
}

pub(super) fn minimax_h3_log_number(line: &str, marker: &str) -> Option<i64> {
    let tail = line.split_once(marker)?.1.trim_start_matches([' ', '=']);
    let digits = tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

pub(super) fn minimax_h3_progress_from_log(
    log: &str,
    requested_steps: i64,
) -> Option<(String, i64, i64, String)> {
    let mut progress = None;
    for line in log.lines() {
        if line.contains("conditioning cache: HIT") {
            progress = Some((
                "conditioning_cache".to_string(),
                0,
                requested_steps,
                "Loaded cached GPU text conditioning".to_string(),
            ));
        } else if line.contains("conditioning [real]") || line.contains("REAL conditioning:") {
            progress = Some((
                "conditioning".to_string(),
                0,
                requested_steps,
                "GPU text conditioning complete".to_string(),
            ));
        } else if line.contains("modcache: HIT") {
            progress = Some((
                "modulation_cache".to_string(),
                0,
                requested_steps,
                "Loaded cached AdaLN modulation".to_string(),
            ));
        } else if line.contains("modcache: SAVED") || line.contains("modcache:") {
            progress = Some((
                "modulation_cache".to_string(),
                0,
                requested_steps,
                "Preparing AdaLN modulation cache".to_string(),
            ));
        } else if line.contains("resident cache: loaded block") {
            // Streamed INT8 tails may visit cache blocks between completed
            // evaluations. Cache messages must never regress a job that has
            // already entered denoising back to step zero.
            if matches!(progress.as_ref(), Some((phase, _, _, _)) if phase == "denoise") {
                continue;
            }
            let block = minimax_h3_log_number(line, "loaded block").unwrap_or(0);
            let total = line
                .split_once('/')
                .and_then(|(_, value)| value.trim().parse::<i64>().ok())
                .unwrap_or(40);
            progress = Some((
                "resident_cache".to_string(),
                0,
                requested_steps,
                format!("Loading cached INT8 DiT block {block} of {total}"),
            ));
        } else if line.contains("resident cache: HIT") {
            if matches!(progress.as_ref(), Some((phase, _, _, _)) if phase == "denoise") {
                continue;
            }
            progress = Some((
                "resident_cache".to_string(),
                0,
                requested_steps,
                "INT8 resident DiT cache ready".to_string(),
            ));
        } else if line.contains("fp8-resident: quantized block") {
            let block = minimax_h3_log_number(line, "quantized block").unwrap_or(0);
            let total = line
                .split_once('/')
                .and_then(|(_, value)| value.split_whitespace().next())
                .and_then(|value| value.parse::<i64>().ok())
                .unwrap_or(48);
            progress = Some((
                "resident_cache_build".to_string(),
                0,
                requested_steps,
                format!("One-time INT8 cache build: block {block} of {total}"),
            ));
        } else if line.contains("phase=denoise") {
            let step = minimax_h3_log_number(line, "step=").unwrap_or(0);
            let evaluations = minimax_h3_log_number(line, "total=").unwrap_or(0);
            progress = Some((
                "denoise".to_string(),
                step,
                requested_steps,
                format!("Denoising evaluation {step} of {evaluations}"),
            ));
        }
    }
    progress
}

pub(super) fn validate_minimax_h3_request(body: &Value) -> Result<(), String> {
    let task = minimax_h3_task(body);
    if !matches!(task, "t2va" | "continue") {
        return validate_minimax_h3_conditioned_request(body, task);
    }
    let prompt = body.get("prompt").and_then(Value::as_str).unwrap_or("");
    if prompt.trim().is_empty() {
        return Err("MiniMax-H3 prompt must not be empty".to_string());
    }
    let _ = minimax_h3_runtime_geometry(body)?;
    let steps = body
        .get("steps")
        .and_then(Value::as_i64)
        .unwrap_or(MINIMAX_H3_STEPS);
    if !(2..=50).contains(&steps) {
        return Err("MiniMax-H3 steps must be an integer from 2 through 50".to_string());
    }
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(0);
    if seed > u32::MAX as u64 {
        return Err("MiniMax-H3 seed must be an integer from 0 through 4294967295".to_string());
    }
    let quant = body.get("quant").and_then(Value::as_str).unwrap_or("int8");
    if !matches!(quant, "int8-fast" | "int8" | "bf16") {
        return Err("MiniMax-H3 quant must be 'int8-fast', 'int8', or 'bf16'".to_string());
    }
    let attention = body
        .get("attention_backend")
        .and_then(Value::as_str)
        .unwrap_or("cudnn");
    if !matches!(attention, "cudnn" | "sage-int8") {
        return Err("MiniMax-H3 attention_backend must be 'cudnn' or 'sage-int8'".to_string());
    }
    if quant == "bf16" && attention == "sage-int8" {
        return Err(
            "MiniMax-H3 Sage attention is available only with INT8 Fast or INT8 Quality; BF16 uses cU-DNN"
                .to_string(),
        );
    }
    let step_cache = body
        .get("step_cache")
        .and_then(Value::as_str)
        .unwrap_or("exact");
    if !matches!(step_cache, "exact" | "high") {
        return Err("MiniMax-H3 step_cache must be 'exact' or 'high'".to_string());
    }
    if body.get("include_audio").and_then(Value::as_bool) == Some(false) {
        return Err("MiniMax-H3 always generates synchronized audio".to_string());
    }
    if task == "continue" {
        let source = body
            .get("continue_from")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or("");
        if !minimax_h3_video_job_id(source) {
            return Err(
                "MiniMax-H3 continue_from must be a local video job id such as video-0100"
                    .to_string(),
            );
        }
        let context_frames = body
            .get("motion_context_frames")
            .and_then(Value::as_i64)
            .unwrap_or(MINIMAX_H3_MOTION_CONTEXT_DEFAULT_FRAMES);
        if !MINIMAX_H3_MOTION_CONTEXT_WINDOWS.contains(&context_frames) {
            return Err("MiniMax-H3 motion_context_frames must be 5, 22, or 39".to_string());
        }
        if minimax_h3_continue_with_references(body) {
            let _ = minimax_h3_ref2va_references(body)?;
        }
    }
    Ok(())
}

pub(super) fn validate_minimax_h3_conditioned_request(
    body: &Value,
    task: &str,
) -> Result<(), String> {
    if !matches!(task, "i2va" | "l2va" | "fl2va" | "ref2va") {
        return Err(
            "MiniMax-H3 task must be 't2va', 'continue', 'i2va', 'l2va', 'fl2va', or 'ref2va'"
                .to_string(),
        );
    }
    let prompt = body.get("prompt").and_then(Value::as_str).unwrap_or("");
    if prompt.trim().is_empty() {
        return Err(format!("MiniMax-H3 {task} prompt must not be empty"));
    }
    let _ = minimax_h3_runtime_geometry(body)?;
    let steps = body.get("steps").and_then(Value::as_i64).unwrap_or(20);
    if !(2..=50).contains(&steps) {
        return Err("MiniMax-H3 steps must be an integer from 2 through 50".to_string());
    }
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(0);
    if seed > u32::MAX as u64 {
        return Err("MiniMax-H3 seed must be an integer from 0 through 4294967295".to_string());
    }
    let quant = body.get("quant").and_then(Value::as_str).unwrap_or("int8");
    if !matches!(quant, "int8-fast" | "int8" | "bf16") {
        return Err("MiniMax-H3 quant must be 'int8-fast', 'int8', or 'bf16'".to_string());
    }
    let attention = body
        .get("attention_backend")
        .and_then(Value::as_str)
        .unwrap_or("cudnn");
    if !matches!(attention, "cudnn" | "sage-int8") {
        return Err("MiniMax-H3 attention_backend must be 'cudnn' or 'sage-int8'".to_string());
    }
    if quant == "bf16" && attention == "sage-int8" {
        return Err(
            "MiniMax-H3 Sage attention is available only with INT8 Fast or INT8 Quality; BF16 uses cU-DNN"
                .to_string(),
        );
    }
    let step_cache = body
        .get("step_cache")
        .and_then(Value::as_str)
        .unwrap_or("exact");
    if !matches!(step_cache, "exact" | "high") {
        return Err("MiniMax-H3 step_cache must be 'exact' or 'high'".to_string());
    }
    if body.get("include_audio").and_then(Value::as_bool) == Some(false) {
        return Err("MiniMax-H3 conditioned modes always generate synchronized audio".to_string());
    }
    if matches!(task, "i2va" | "fl2va") {
        minimax_h3_request_media_path(body, "source_image")?;
    }
    if task == "ref2va" {
        let _ = minimax_h3_ref2va_references(body)?;
    }
    if matches!(task, "l2va" | "fl2va") {
        minimax_h3_request_media_path(body, "last_frame")?;
    }
    Ok(())
}

pub(super) fn start_minimax_h3_request(
    st: &AppState,
    body: &Value,
    gpu: crate::gpu_lock::GpuGuard,
) -> Response {
    let task = minimax_h3_task(body);
    if minimax_h3_continue_with_references(body) {
        return start_minimax_h3_conditioned_request(st, body, gpu);
    }
    if !matches!(task, "t2va" | "continue") {
        return start_minimax_h3_conditioned_request(st, body, gpu);
    }
    let continuation_source = if task == "continue" {
        Some(
            minimax_h3_continuation_source(&st.out_dir, body)
                .expect("validated MiniMax-H3 continuation source must resolve"),
        )
    } else {
        None
    };
    let geometry = minimax_h3_runtime_geometry(body)
        .expect("validated MiniMax-H3 request must resolve runtime geometry");
    let profile_id = format!(
        "runtime-{}x{}-{:.3}s-{}fps",
        geometry.width, geometry.height, geometry.duration, geometry.fps,
    );
    let profile_width = geometry.width;
    let profile_height = geometry.height;
    let internal_frames = geometry.internal_frames;
    let model_output_frames = geometry.model_output_frames;
    let output_frames = geometry.output_frames;
    let output_fps = geometry.fps;
    let motion_context_frames = geometry.motion_context_frames;
    let trim_start_frames = geometry.trim_start_frames;
    let prompt = body
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let quant = body
        .get("quant")
        .and_then(Value::as_str)
        .unwrap_or("int8")
        .to_string();
    let attention = body
        .get("attention_backend")
        .and_then(Value::as_str)
        .unwrap_or("cudnn")
        .to_string();
    let step_cache = body
        .get("step_cache")
        .and_then(Value::as_str)
        .unwrap_or("exact")
        .to_string();
    let steps = body
        .get("steps")
        .and_then(Value::as_i64)
        .unwrap_or(MINIMAX_H3_STEPS);
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(0);
    let runner = MINIMAX_H3_REQUEST_RUNNER.to_string();
    let resident_blocks = minimax_h3_runtime_resident_blocks(&geometry, &quant);
    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    if let Err(error) = std::fs::create_dir_all(&out_dir) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot create MiniMax-H3 output directory: {error}"),
        );
    }
    let request_path = out_dir.join("request.json");
    let mut request = body.clone();
    if let Some(object) = request.as_object_mut() {
        object.insert("runner".to_string(), json!("minimax_h3_mojo_request"));
        object.insert("profile".to_string(), json!(profile_id.clone()));
        object.insert("defer_decode".to_string(), json!(true));
        object.insert("encoder_storage".to_string(), json!("int8"));
        object.insert(
            "experimental_long_context".to_string(),
            json!(geometry.duration > MINIMAX_H3_TRAINED_MAX_SECONDS),
        );
        object.insert(
            "trained_seconds_max".to_string(),
            json!(MINIMAX_H3_TRAINED_MAX_SECONDS),
        );
        object.insert(
            "sequence_tokens".to_string(),
            json!(geometry.sequence_tokens),
        );
    }
    let request_bytes = match serde_json::to_vec_pretty(&request) {
        Ok(bytes) => bytes,
        Err(error) => {
            return err_detail(
                StatusCode::BAD_REQUEST,
                &format!("cannot serialize MiniMax-H3 request: {error}"),
            );
        }
    };
    if let Err(error) = std::fs::write(&request_path, request_bytes) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot write MiniMax-H3 request: {error}"),
        );
    }
    let _ = write_minimax_h3_job_status(
        &out_dir,
        "queued",
        "queued",
        0,
        steps,
        "MiniMax-H3 request queued",
    );

    let bus = st.comfy_ws.clone();
    let thread_video_id = video_id.clone();
    let thread_out_dir = out_dir.clone();
    let thread_quant = quant.clone();
    let thread_attention = attention.clone();
    let thread_step_cache = step_cache.clone();
    let thread_runner = runner.clone();
    let thread_profile_id = profile_id.clone();
    let thread_prompt = prompt.clone();
    let thread_task = task.to_string();
    let thread_continuation_source = continuation_source.clone();
    std::thread::spawn(move || {
        let _gpu = gpu;
        let publish = |event: WorkerEvent| {
            let _ = bus.send((thread_video_id.clone(), event));
        };
        let fail = |phase: &str, error: String| {
            // Decode-side failures leave a complete denoise behind; keep the
            // latents so the decode can be retried instead of losing the
            // render (see cleanup_minimax_h3_intermediates).
            let keep_latents = phase == "decode" || phase == "decode_start" || phase == "result";
            cleanup_minimax_h3_intermediates(&thread_out_dir, keep_latents);
            let _ = write_minimax_h3_job_status(&thread_out_dir, "failed", phase, 0, steps, &error);
            publish(WorkerEvent::Failed { error });
        };
        let needs_runtime_cache = minimax_h3_resident_cache_path(&thread_quant, false)
            .is_some_and(|path| !nonempty_file(&path));
        if needs_runtime_cache {
            let cache_message = format!(
                "Preparing MiniMax-H3 {} acceleration cache once on GPU",
                thread_quant.to_uppercase()
            );
            let _ = write_minimax_h3_job_status(
                &thread_out_dir,
                "running",
                "runtime_cache",
                0,
                steps,
                &cache_message,
            );
            publish(WorkerEvent::Progress {
                step: 0,
                total: steps,
                phase: cache_message,
                preview: String::new(),
            });
            if let Err(error) =
                prepare_minimax_h3_resident_cache_if_needed(&thread_quant, false, &thread_out_dir)
            {
                fail("runtime_cache", error);
                return;
            }
        }
        let message = format!(
            "Starting MiniMax-H3 {} {} DiT with {} attention and {} step cache",
            if thread_task == "continue" {
                "native continuation"
            } else {
                "T2VA"
            },
            thread_quant.to_uppercase(),
            thread_attention,
            thread_step_cache,
        );
        let _ = write_minimax_h3_job_status(
            &thread_out_dir,
            "running",
            "conditioning_and_denoising",
            0,
            steps,
            &message,
        );
        publish(WorkerEvent::Progress {
            step: 0,
            total: steps,
            phase: message,
            preview: String::new(),
        });
        let log_path = thread_out_dir.join("runner.log");
        let log = match std::fs::File::create(&log_path) {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "runner_start",
                    format!("cannot create MiniMax-H3 runner log: {error}"),
                );
                return;
            }
        };
        let stderr = match log.try_clone() {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "runner_start",
                    format!("cannot clone MiniMax-H3 log handle: {error}"),
                );
                return;
            }
        };
        let mut runner_args: Vec<String> = vec![
            thread_prompt.clone(),
            thread_out_dir.to_string_lossy().to_string(),
            steps.to_string(),
            seed.to_string(),
            "50".to_string(),
            format!("--width={profile_width}"),
            format!("--height={profile_height}"),
            format!("--frames={internal_frames}"),
            format!("--output-frames={model_output_frames}"),
            format!("--fps={MINIMAX_H3_FPS}"),
            format!("--output-fps={output_fps}"),
            format!("--quant={}", thread_quant),
            format!("--resident-blocks={resident_blocks}"),
            "--encoder-storage=int8".to_string(),
            format!("--attention-backend={}", thread_attention),
            format!("--step-cache={}", thread_step_cache),
            "--defer-video-decode".to_string(),
        ];
        if let Some(source) = thread_continuation_source.as_ref() {
            runner_args.push(format!(
                "--motion-context={}",
                source.latent_path.to_string_lossy()
            ));
            runner_args.push(format!("--motion-context-frames={motion_context_frames}"));
            runner_args.push(format!("--trim-start-frames={trim_start_frames}"));
        }
        if thread_prompt == MINIMAX_H3_TEST_PROMPT {
            runner_args.push("--runtime-cache-exact-product-prompt".to_string());
        }
        // Warm path: hand the job line to the resident --serve worker; the
        // worker re-points its stdout at this job's runner.log itself.
        let warm = minimax_h3_warm_enabled();
        let mut child_opt: Option<std::process::Child> = None;
        if warm {
            drop(log);
            drop(stderr);
            if let Err(error) = minimax_h3_warm_submit(&thread_runner, &runner_args) {
                fail(
                    "runner_start",
                    format!("cannot submit to warm MiniMax-H3 worker: {error}"),
                );
                return;
            }
        } else {
            let mut command = minimax_h3_capped_command(&repo_path(&thread_runner));
            command
                .current_dir(repo_root())
                .env("LD_LIBRARY_PATH", minimax_h3_ld_path())
                .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
                .env("CUDA_MODULE_LOADING", "EAGER")
                .env("LD_BIND_NOW", "1")
                .env("CUDA_FORCE_PRELOAD_LIBRARIES", "1")
                .args(&runner_args)
                .stdout(std::process::Stdio::from(log))
                .stderr(std::process::Stdio::from(stderr));
            child_opt = Some(match command.spawn() {
                Ok(child) => child,
                Err(error) => {
                    fail(
                        "runner_start",
                        format!("cannot start MiniMax-H3 runner: {error}"),
                    );
                    return;
                }
            });
        }
        let mut last_progress: Option<(String, i64, i64, String)> = None;
        let status = loop {
            if let Ok(text) = std::fs::read_to_string(&log_path) {
                if let Some(progress) = minimax_h3_progress_from_log(&text, steps) {
                    if last_progress.as_ref() != Some(&progress) {
                        let (phase, step, total, message) = &progress;
                        let _ = write_minimax_h3_job_status(
                            &thread_out_dir,
                            "running",
                            phase,
                            *step,
                            *total,
                            message,
                        );
                        publish(WorkerEvent::Progress {
                            step: *step,
                            total: *total,
                            phase: message.clone(),
                            preview: String::new(),
                        });
                        last_progress = Some(progress);
                    }
                }
            }
            if let Some(child) = child_opt.as_mut() {
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status.success()),
                    Ok(None) => std::thread::sleep(std::time::Duration::from_millis(500)),
                    Err(error) => break Err(error),
                }
            } else {
                // Warm worker: completion is marker-driven — the process
                // outlives the job by design.
                let text = std::fs::read_to_string(&log_path).unwrap_or_default();
                if text.contains("[serve] job FAILED") {
                    break Ok(false);
                }
                if text.contains("[serve] job complete") {
                    break Ok(true);
                }
                if !minimax_h3_warm_alive() {
                    break Ok(false);
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        };
        match status {
            Ok(true)
                if thread_out_dir.join("latents.safetensors").is_file()
                    && thread_out_dir.join("motion_context.safetensors").is_file() => {}
            Ok(_) => {
                cleanup_minimax_h3_conditioned_intermediates(&thread_out_dir);
                fail(
                    "denoise",
                    format!(
                        "MiniMax-H3 denoiser failed; inspect {}",
                        log_path.to_string_lossy()
                    ),
                );
                return;
            }
            Err(error) => {
                cleanup_minimax_h3_conditioned_intermediates(&thread_out_dir);
                fail(
                    "runner",
                    format!("cannot monitor MiniMax-H3 runner: {error}"),
                );
                return;
            }
        }

        let decode_message = "Denoiser released; starting fresh GPU video decode and NVENC mux";
        let _ = write_minimax_h3_job_status(
            &thread_out_dir,
            "running",
            "decode",
            steps,
            steps,
            decode_message,
        );
        publish(WorkerEvent::Progress {
            step: steps,
            total: steps,
            phase: decode_message.to_string(),
            preview: String::new(),
        });
        let decode_log_path = thread_out_dir.join("decode.log");
        let decode_log = match std::fs::File::create(&decode_log_path) {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "decode_start",
                    format!("cannot create MiniMax-H3 decode log: {error}"),
                );
                return;
            }
        };
        let decode_stderr = match decode_log.try_clone() {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "decode_start",
                    format!("cannot clone MiniMax-H3 decode log: {error}"),
                );
                return;
            }
        };
        let mut decode_command = minimax_h3_capped_command(&repo_path(&thread_runner));
        decode_command
            .current_dir(repo_root())
            .env("LD_LIBRARY_PATH", minimax_h3_ld_path())
            .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
            .env("CUDA_MODULE_LOADING", "EAGER")
            .env("LD_BIND_NOW", "1")
            .env("CUDA_FORCE_PRELOAD_LIBRARIES", "1")
            .arg("decode")
            .arg(&thread_out_dir)
            .arg(steps.to_string())
            .arg(seed.to_string())
            .arg("50")
            .arg("decode_only")
            .arg(format!("--width={profile_width}"))
            .arg(format!("--height={profile_height}"))
            .arg(format!("--frames={internal_frames}"))
            .arg(format!("--output-frames={output_frames}"))
            .arg(format!("--fps={MINIMAX_H3_FPS}"))
            .arg(format!("--output-fps={output_fps}"))
            .arg(format!("--quant={}", thread_quant))
            .arg(format!("--resident-blocks={resident_blocks}"));
        if let Some(source) = thread_continuation_source.as_ref() {
            decode_command
                .arg(format!(
                    "--motion-context={}",
                    source.latent_path.to_string_lossy()
                ))
                .arg(format!("--motion-context-frames={motion_context_frames}"))
                .arg(format!("--trim-start-frames={trim_start_frames}"));
        }
        let decode_status = decode_command
            .stdout(std::process::Stdio::from(decode_log))
            .stderr(std::process::Stdio::from(decode_stderr))
            .status();
        if !decode_status
            .as_ref()
            .is_ok_and(std::process::ExitStatus::success)
        {
            fail(
                "decode",
                format!(
                    "MiniMax-H3 fresh decode failed with {:?}; inspect {}",
                    decode_status,
                    decode_log_path.to_string_lossy()
                ),
            );
            return;
        }
        let result_path = thread_out_dir.join("result.json");
        let mut result = std::fs::read_to_string(&result_path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        let authored = result
            .as_ref()
            .filter(|doc| doc.get("state").and_then(Value::as_str) == Some("done"))
            .and_then(|doc| doc.get("artifact_path"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let artifact = if std::path::Path::new(authored).is_absolute() {
            std::path::PathBuf::from(authored)
        } else {
            repo_root().join(authored)
        };
        if !artifact.is_file() {
            fail(
                "result",
                format!(
                    "MiniMax-H3 decode did not publish a valid artifact; inspect {}",
                    result_path.to_string_lossy()
                ),
            );
            return;
        }
        if let Some(document) = result.as_mut().and_then(Value::as_object_mut) {
            document.insert("model".to_string(), json!("minimax_h3"));
            document.insert("runner".to_string(), json!("minimax_h3_mojo_request"));
            document.insert("quant".to_string(), json!(thread_quant));
            document.insert("attention_backend".to_string(), json!(thread_attention));
            document.insert("step_cache".to_string(), json!(thread_step_cache));
            document.insert("profile".to_string(), json!(thread_profile_id));
            document.insert("task".to_string(), json!(thread_task));
            document.insert(
                "motion_context_available".to_string(),
                json!(thread_out_dir.join("motion_context.safetensors").is_file()),
            );
            document.insert(
                "motion_context_url".to_string(),
                json!(format!(
                    "/out/{}/motion_context.safetensors",
                    thread_video_id
                )),
            );
            document.insert(
                "motion_context_windows".to_string(),
                json!(MINIMAX_H3_MOTION_CONTEXT_WINDOWS),
            );
            if let Some(source) = thread_continuation_source.as_ref() {
                document.insert("continue_from".to_string(), json!(source.job_id));
                document.insert(
                    "motion_context_frames".to_string(),
                    json!(motion_context_frames),
                );
                document.insert("trim_start_frames".to_string(), json!(trim_start_frames));
            }
            document.insert(
                "experimental_long_context".to_string(),
                json!(internal_frames > 362),
            );
            document.insert(
                "trained_seconds_max".to_string(),
                json!(MINIMAX_H3_TRAINED_MAX_SECONDS),
            );
            document.insert(
                "mp4_url".to_string(),
                json!(format!("/out/{}/video.mp4", thread_video_id)),
            );
        }
        if let Some(document) = result.as_ref() {
            if let Ok(bytes) = serde_json::to_vec_pretty(document) {
                let _ = std::fs::write(&result_path, bytes);
            }
        }
        cleanup_minimax_h3_conditioned_intermediates(&thread_out_dir);
        let _ = write_minimax_h3_job_status(
            &thread_out_dir,
            "done",
            "done",
            steps,
            steps,
            "MiniMax-H3 synchronized video and audio ready",
        );
        publish(WorkerEvent::Done {
            output_path: artifact.to_string_lossy().into_owned(),
        });
    });

    json_resp(
        StatusCode::ACCEPTED,
        &json!({
            "schema": "serenity.video_job.v1",
            "video_id": video_id,
            "prompt_id": video_id,
            "model": "minimax_h3",
            "runner": "minimax_h3_mojo_request",
            "task": task,
            "request_runner": runner,
            "profile": profile_id,
            "backend": "mojo",
            "quant": quant,
            "attention_backend": attention,
            "step_cache": step_cache,
            "state": "queued",
            "status_url": format!("/out/{video_id}/status.json"),
            "result_url": format!("/out/{video_id}/result.json"),
            "request_url": format!("/out/{video_id}/request.json"),
            "continue_from": continuation_source.as_ref().map(|source| source.job_id.as_str()),
            "motion_context_frames": geometry.motion_context_frames,
        }),
    )
}

pub(super) fn cleanup_minimax_h3_conditioned_intermediates(out_dir: &std::path::Path) {
    cleanup_minimax_h3_intermediates(out_dir, false);
}

/// `keep_latents` retains `latents.safetensors`/`latents_ckpt.safetensors` so a
/// failed decode can be retried without repeating the denoise. A decode failure
/// is usually transient VRAM co-tenancy, while the latents represent the whole
/// GPU cost of the job: video-0190 lost a 57-minute 1344x768 render to this
/// cleanup, and the isolated decode retry at the same geometry succeeded with
/// ~10 GiB free (decode peak ~12.9 GiB, measured 2026-08-12).
pub(super) fn cleanup_minimax_h3_intermediates(out_dir: &std::path::Path, keep_latents: bool) {
    let Ok(entries) = std::fs::read_dir(out_dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let generated_frame =
            (name.starts_with("frame_") && name.ends_with(".png")) || name == "frames.rgb";
        let latent_artifact = name == "latents.safetensors" || name == "latents_ckpt.safetensors";
        let transient = generated_frame
            || name == "audio.wav"
            || (latent_artifact && !keep_latents)
            || name == ".gpu_guard"
            || name == "ref_prompt.txt"
            || name.starts_with("keyframe_first.")
            || name.starts_with("keyframe_last.")
            || name.starts_with("ref_image_")
            || name.starts_with("ref_audio_input_")
            || name.starts_with("ref_product_input_")
            || name.starts_with("ref_probe_")
            || name.starts_with("ref_soundtrack_");
        if transient && entry.file_type().is_ok_and(|kind| kind.is_file()) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

pub(super) fn start_minimax_h3_conditioned_request(
    st: &AppState,
    body: &Value,
    gpu: crate::gpu_lock::GpuGuard,
) -> Response {
    let task = minimax_h3_task(body).to_string();
    let combined_continuation = minimax_h3_continue_with_references(body);
    let continuation_source = combined_continuation.then(|| {
        minimax_h3_continuation_source(&st.out_dir, body)
            .expect("validated MiniMax-H3 continuation source must resolve")
    });
    let geometry = minimax_h3_runtime_geometry(body)
        .expect("validated conditioned MiniMax-H3 request must resolve geometry");
    let profile_id = format!(
        "runtime-{}x{}-{:.3}s-{}fps",
        geometry.width, geometry.height, geometry.duration, geometry.fps,
    );
    let prompt = body
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let quant = body
        .get("quant")
        .and_then(Value::as_str)
        .unwrap_or("int8")
        .to_string();
    let attention = body
        .get("attention_backend")
        .and_then(Value::as_str)
        .unwrap_or("cudnn")
        .to_string();
    let step_cache = body
        .get("step_cache")
        .and_then(Value::as_str)
        .unwrap_or("exact")
        .to_string();
    let steps = body.get("steps").and_then(Value::as_i64).unwrap_or(20);
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(0);
    let runner_task = if combined_continuation {
        "ref2va"
    } else {
        task.as_str()
    };
    let runner = minimax_h3_conditioned_runner(runner_task, &quant)
        .expect("validated conditioned MiniMax-H3 request must resolve a runner");
    let resident_blocks = if task == "ref2va" || combined_continuation {
        minimax_h3_ref2va_resident_blocks(&geometry, &quant)
    } else {
        0
    };
    let legacy_ref2va = task == "ref2va"
        && body
            .get("references")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
        && body
            .get("source_image")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());
    let references = if task == "ref2va" || combined_continuation {
        minimax_h3_ref2va_references(body)
            .expect("validated MiniMax-H3 Ref2VA references must resolve")
    } else {
        Vec::new()
    };
    let mut media = Vec::new();
    if matches!(task.as_str(), "i2va" | "fl2va") {
        media.push(
            minimax_h3_request_media_path(body, "source_image")
                .expect("validated MiniMax-H3 source_image must resolve"),
        );
    }
    if matches!(task.as_str(), "l2va" | "fl2va") {
        media.push(
            minimax_h3_request_media_path(body, "last_frame")
                .expect("validated MiniMax-H3 last_frame must resolve"),
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
            &format!("cannot create MiniMax-H3 output directory: {error}"),
        );
    }
    let effective_media = media.clone();
    let mut effective_references = references.clone();
    let mut reference_preprocesses = Vec::new();
    let legacy_reference_preprocess = if legacy_ref2va {
        match stage_minimax_h3_ref2va_image_reference(&references[0].path, &out_dir) {
            Ok((prepared_path, metadata)) => {
                effective_references[0].path = prepared_path;
                reference_preprocesses.push(metadata.clone());
                Some(metadata)
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(&out_dir);
                return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
            }
        }
    } else {
        None
    };
    if task == "ref2va" || combined_continuation {
        for (index, reference) in effective_references.iter_mut().enumerate() {
            if reference.kind != "audio" {
                continue;
            }
            match stage_minimax_h3_ref2va_audio_reference(&reference.path, &out_dir, index) {
                Ok(prepared_path) => {
                    reference_preprocesses.push(json!({
                        "index": index,
                        "kind": "audio",
                        "policy": "ffmpeg_pcm_s16le_32000hz_stereo",
                        "operation": "media_preprocess_only",
                        "model_inference": "gpu",
                        "original_path": reference.path,
                        "prepared_path": prepared_path,
                    }));
                    reference.path = prepared_path;
                }
                Err(error) => {
                    let _ = std::fs::remove_dir_all(&out_dir);
                    return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
                }
            }
        }
    }
    let request_path = out_dir.join("request.json");
    let mut request = body.clone();
    if let Some(object) = request.as_object_mut() {
        object.insert("runner".to_string(), json!("minimax_h3_mojo_request"));
        object.insert("profile".to_string(), json!(profile_id.clone()));
        object.insert("defer_decode".to_string(), json!(true));
        object.insert("gpu_model_execution_only".to_string(), json!(true));
        object.insert(
            "cache_policy".to_string(),
            json!(if task == "ref2va" || combined_continuation {
                "ref2va_dit_cache_and_shared_identical_encoder_cache"
            } else {
                "reuse_existing_fl2va_resident_cache"
            }),
        );
        object.insert(
            "reference_only".to_string(),
            json!(task == "ref2va" || combined_continuation),
        );
        object.insert("encoder_storage".to_string(), json!("int8"));
        object.insert(
            "experimental_long_context".to_string(),
            json!(geometry.duration > MINIMAX_H3_TRAINED_MAX_SECONDS),
        );
        object.insert(
            "trained_seconds_max".to_string(),
            json!(MINIMAX_H3_TRAINED_MAX_SECONDS),
        );
        object.insert(
            "sequence_tokens".to_string(),
            json!(geometry.sequence_tokens),
        );
        if task == "ref2va" || combined_continuation {
            object.insert("resident_blocks".to_string(), json!(resident_blocks));
            object.insert(
                "memory_policy".to_string(),
                json!("sequence_adaptive_w8a8_streaming"),
            );
            object.insert(
                "effective_references".to_string(),
                Value::Array(
                    effective_references
                        .iter()
                        .enumerate()
                        .map(|(index, reference)| {
                            json!({
                                "index": index,
                                "kind": reference.kind,
                                "path": reference.path,
                                "audio_use": reference.audio_use,
                                "has_audio": reference.has_audio,
                                "duration": reference.duration,
                            })
                        })
                        .collect(),
                ),
            );
            object.insert(
                "reference_preprocesses".to_string(),
                Value::Array(reference_preprocesses.clone()),
            );
        }
        if let Some(source) = continuation_source.as_ref() {
            object.insert("continue_from".to_string(), json!(source.job_id));
            object.insert("motion_context_path".to_string(), json!(source.latent_path));
            object.insert(
                "motion_context_frames".to_string(),
                json!(geometry.motion_context_frames),
            );
        }
        if let Some(metadata) = legacy_reference_preprocess.as_ref() {
            object.insert("reference_preprocess".to_string(), metadata.clone());
            object.insert(
                "effective_source_image".to_string(),
                json!(effective_references[0].path),
            );
        }
    }
    let request_bytes = match serde_json::to_vec_pretty(&request) {
        Ok(bytes) => bytes,
        Err(error) => {
            return err_detail(
                StatusCode::BAD_REQUEST,
                &format!("cannot serialize MiniMax-H3 request: {error}"),
            );
        }
    };
    if let Err(error) = std::fs::write(&request_path, request_bytes) {
        return err_detail(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("cannot write MiniMax-H3 request: {error}"),
        );
    }
    let _ = write_minimax_h3_job_status(
        &out_dir,
        "queued",
        "queued",
        0,
        steps,
        &format!("MiniMax-H3 {task} request queued"),
    );

    let bus = st.comfy_ws.clone();
    let thread_video_id = video_id.clone();
    let thread_out_dir = out_dir.clone();
    let thread_task = task.clone();
    let thread_quant = quant.clone();
    let thread_attention = attention.clone();
    let thread_step_cache = step_cache.clone();
    let thread_runner = runner.clone();
    let thread_media = effective_media;
    let thread_references = effective_references;
    let thread_prompt = prompt.clone();
    let thread_geometry = geometry.clone();
    let thread_resident_blocks = resident_blocks;
    let thread_combined_continuation = combined_continuation;
    let thread_continuation_source = continuation_source.clone();
    std::thread::spawn(move || {
        let _gpu = gpu;
        let publish = |event: WorkerEvent| {
            let _ = bus.send((thread_video_id.clone(), event));
        };
        let fail = |phase: &str, error: String| {
            // Decode-side failures leave a complete denoise behind; keep the
            // latents so the decode can be retried instead of losing the
            // render (see cleanup_minimax_h3_intermediates).
            let keep_latents = phase == "decode" || phase == "decode_start" || phase == "result";
            cleanup_minimax_h3_intermediates(&thread_out_dir, keep_latents);
            let _ = write_minimax_h3_job_status(&thread_out_dir, "failed", phase, 0, steps, &error);
            publish(WorkerEvent::Failed { error });
        };
        let ref2va_cache = thread_task == "ref2va" || thread_combined_continuation;
        let needs_runtime_cache = minimax_h3_resident_cache_path(&thread_quant, ref2va_cache)
            .is_some_and(|path| !nonempty_file(&path));
        if needs_runtime_cache {
            let cache_message = format!(
                "Preparing MiniMax-H3 {} {} acceleration cache once on GPU",
                thread_task.to_uppercase(),
                thread_quant.to_uppercase()
            );
            let _ = write_minimax_h3_job_status(
                &thread_out_dir,
                "running",
                "runtime_cache",
                0,
                steps,
                &cache_message,
            );
            publish(WorkerEvent::Progress {
                step: 0,
                total: steps,
                phase: cache_message,
                preview: String::new(),
            });
            if let Err(error) = prepare_minimax_h3_resident_cache_if_needed(
                &thread_quant,
                ref2va_cache,
                &thread_out_dir,
            ) {
                fail("runtime_cache", error);
                return;
            }
        }
        let message = format!(
            "Starting MiniMax-H3 {} {} with GPU vision, {} attention, and {} step cache",
            thread_task.to_uppercase(),
            thread_quant.to_uppercase(),
            thread_attention,
            thread_step_cache,
        );
        let _ = write_minimax_h3_job_status(
            &thread_out_dir,
            "running",
            "keyframe_conditioning_and_denoising",
            0,
            steps,
            &message,
        );
        publish(WorkerEvent::Progress {
            step: 0,
            total: steps,
            phase: message,
            preview: String::new(),
        });
        let log_path = thread_out_dir.join("runner.log");
        let log = match std::fs::File::create(&log_path) {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "runner_start",
                    format!("cannot create MiniMax-H3 runner log: {error}"),
                );
                return;
            }
        };
        let stderr = match log.try_clone() {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "runner_start",
                    format!("cannot clone MiniMax-H3 log handle: {error}"),
                );
                return;
            }
        };
        let mut command = minimax_h3_capped_command(&repo_path(&thread_runner));
        command
            .current_dir(repo_root())
            .env("LD_LIBRARY_PATH", minimax_h3_ld_path())
            .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
            .env("CUDA_MODULE_LOADING", "EAGER")
            .env("LD_BIND_NOW", "1")
            .env("CUDA_FORCE_PRELOAD_LIBRARIES", "1");
        if thread_task == "ref2va" || thread_combined_continuation {
            let prompt_path = thread_out_dir.join("ref_prompt.txt");
            let conditioned_prompt =
                minimax_h3_ref2va_prompt_with_audio_roles(&thread_prompt, &thread_references);
            if let Err(error) = std::fs::write(&prompt_path, format!("{}\n", conditioned_prompt)) {
                fail(
                    "runner_start",
                    format!("cannot write MiniMax-H3 Ref2VA prompt: {error}"),
                );
                return;
            }
            command
                .arg(prompt_path)
                .arg(&thread_out_dir)
                .arg(steps.to_string())
                .arg(seed.to_string());
            for reference in &thread_references {
                command.arg(format!(
                    "{}:{}",
                    reference.kind,
                    reference.path.to_string_lossy(),
                ));
            }
            command
                .arg(format!("--attention-backend={}", thread_attention))
                .arg(format!("--step-cache={}", thread_step_cache))
                .arg(format!("--resident-blocks={thread_resident_blocks}"));
            if let Some(source) = thread_continuation_source.as_ref() {
                command
                    .arg(format!(
                        "--motion-context={}",
                        source.latent_path.to_string_lossy(),
                    ))
                    .arg(format!(
                        "--motion-context-frames={}",
                        thread_geometry.motion_context_frames,
                    ))
                    .arg(format!(
                        "--continuation-end-frames={}",
                        thread_geometry.motion_context_frames + thread_geometry.model_output_frames,
                    ));
            }
        } else {
            command.arg(&thread_task).arg(&thread_prompt);
            for path in &thread_media {
                command.arg(path);
            }
            command
                .arg(&thread_out_dir)
                .arg(steps.to_string())
                .arg(seed.to_string())
                .arg("50")
                .arg(format!("--attention-backend={}", thread_attention))
                .arg(format!("--step-cache={}", thread_step_cache))
                .arg("--defer-video-decode");
        }
        command
            .arg(format!("--width={}", thread_geometry.width))
            .arg(format!("--height={}", thread_geometry.height))
            .arg(format!("--frames={}", thread_geometry.internal_frames))
            .arg(format!("--fps={MINIMAX_H3_FPS}"));
        if thread_quant == "int8-fast" {
            command.arg("--resident-backend=w8a8");
        } else if thread_quant == "int8" {
            command.arg("--resident-backend=groupwise");
        }
        command
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(stderr));
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                fail(
                    "runner_start",
                    format!("cannot start MiniMax-H3 runner: {error}"),
                );
                return;
            }
        };
        let mut last_progress: Option<(String, i64, i64, String)> = None;
        let status = loop {
            if let Ok(text) = std::fs::read_to_string(&log_path) {
                if let Some(progress) = minimax_h3_progress_from_log(&text, steps) {
                    if last_progress.as_ref() != Some(&progress) {
                        let (phase, step, total, progress_message) = &progress;
                        let _ = write_minimax_h3_job_status(
                            &thread_out_dir,
                            "running",
                            phase,
                            *step,
                            *total,
                            progress_message,
                        );
                        publish(WorkerEvent::Progress {
                            step: *step,
                            total: *total,
                            phase: progress_message.clone(),
                            preview: String::new(),
                        });
                        last_progress = Some(progress);
                    }
                }
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(500)),
                Err(error) => break Err(error),
            }
        };
        match status {
            Ok(status)
                if status.success()
                    && thread_out_dir.join("latents.safetensors").is_file()
                    && thread_out_dir.join("motion_context.safetensors").is_file() => {}
            Ok(status) => {
                fail(
                    "denoise",
                    format!(
                        "MiniMax-H3 {} denoiser failed with {status}; inspect {}",
                        thread_task,
                        log_path.to_string_lossy()
                    ),
                );
                return;
            }
            Err(error) => {
                fail(
                    "runner",
                    format!("cannot monitor MiniMax-H3 runner: {error}"),
                );
                return;
            }
        }

        let decode_message = "Denoiser released; starting fresh GPU video decode and NVENC mux";
        let _ = write_minimax_h3_job_status(
            &thread_out_dir,
            "running",
            "decode",
            steps,
            steps,
            decode_message,
        );
        publish(WorkerEvent::Progress {
            step: steps,
            total: steps,
            phase: decode_message.to_string(),
            preview: String::new(),
        });
        let decode_log_path = thread_out_dir.join("decode.log");
        let decode_log = match std::fs::File::create(&decode_log_path) {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "decode_start",
                    format!("cannot create MiniMax-H3 decode log: {error}"),
                );
                return;
            }
        };
        let decode_stderr = match decode_log.try_clone() {
            Ok(file) => file,
            Err(error) => {
                fail(
                    "decode_start",
                    format!("cannot clone MiniMax-H3 decode log: {error}"),
                );
                return;
            }
        };
        let mut decode_command =
            minimax_h3_capped_command(&repo_path(MINIMAX_H3_CONDITIONED_DECODE_RUNNER));
        decode_command
            .current_dir(repo_root())
            .env("LD_LIBRARY_PATH", minimax_h3_ld_path())
            .env("CUDA_CACHE_PATH", LTX2_CUDA_CACHE)
            .env("CUDA_MODULE_LOADING", "EAGER")
            .env("LD_BIND_NOW", "1")
            .env("CUDA_FORCE_PRELOAD_LIBRARIES", "1")
            .arg("decode")
            .arg(&thread_out_dir)
            .arg(steps.to_string())
            .arg(seed.to_string())
            .arg("50")
            .arg("decode_only")
            .arg(format!("--width={}", thread_geometry.width))
            .arg(format!("--height={}", thread_geometry.height))
            .arg(format!("--frames={}", thread_geometry.internal_frames))
            .arg(format!("--output-frames={}", thread_geometry.output_frames))
            .arg(format!("--fps={MINIMAX_H3_FPS}"))
            .arg(format!("--output-fps={}", thread_geometry.fps));
        if thread_combined_continuation {
            decode_command.arg(format!(
                "--trim-start-frames={}",
                thread_geometry.trim_start_frames,
            ));
        }
        let decode_status = decode_command
            .stdout(std::process::Stdio::from(decode_log))
            .stderr(std::process::Stdio::from(decode_stderr))
            .status();
        if !decode_status
            .as_ref()
            .is_ok_and(std::process::ExitStatus::success)
        {
            fail(
                "decode",
                format!(
                    "MiniMax-H3 fresh decode failed with {:?}; inspect {}",
                    decode_status,
                    decode_log_path.to_string_lossy()
                ),
            );
            return;
        }
        let result_path = thread_out_dir.join("result.json");
        let mut result = std::fs::read_to_string(&result_path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok());
        let authored = result
            .as_ref()
            .filter(|doc| doc.get("state").and_then(Value::as_str) == Some("done"))
            .and_then(|doc| doc.get("artifact_path"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let artifact = if std::path::Path::new(authored).is_absolute() {
            std::path::PathBuf::from(authored)
        } else {
            repo_root().join(authored)
        };
        if !artifact.is_file() {
            fail(
                "result",
                format!(
                    "MiniMax-H3 decode did not publish a valid artifact; inspect {}",
                    result_path.to_string_lossy()
                ),
            );
            return;
        }
        if let Some(document) = result.as_mut().and_then(Value::as_object_mut) {
            document.insert("model".to_string(), json!("minimax_h3"));
            document.insert("runner".to_string(), json!("minimax_h3_mojo_request"));
            document.insert("task".to_string(), json!(thread_task.clone()));
            document.insert("quant".to_string(), json!(thread_quant.clone()));
            document.insert(
                "attention_backend".to_string(),
                json!(thread_attention.clone()),
            );
            document.insert("step_cache".to_string(), json!(thread_step_cache.clone()));
            document.insert("encoder_storage".to_string(), json!("int8"));
            document.insert(
                "ref2va_motion_context".to_string(),
                json!(thread_combined_continuation),
            );
            if let Some(source) = thread_continuation_source.as_ref() {
                document.insert("continue_from".to_string(), json!(source.job_id.clone()));
                document.insert(
                    "motion_context_frames".to_string(),
                    json!(thread_geometry.motion_context_frames),
                );
            }
            if !thread_references.is_empty() {
                document.insert(
                    "references".to_string(),
                    Value::Array(
                        thread_references
                            .iter()
                            .map(|reference| {
                                json!({
                                    "kind": reference.kind,
                                    "path": reference.path,
                                    "audio_use": reference.audio_use,
                                })
                            })
                            .collect(),
                    ),
                );
            }
            document.insert(
                "experimental_long_context".to_string(),
                json!(thread_geometry.duration > MINIMAX_H3_TRAINED_MAX_SECONDS),
            );
            document.insert(
                "trained_seconds_max".to_string(),
                json!(MINIMAX_H3_TRAINED_MAX_SECONDS),
            );
            document.insert(
                "profile".to_string(),
                json!(format!(
                    "runtime-{}x{}-{:.3}s-{}fps",
                    thread_geometry.width,
                    thread_geometry.height,
                    thread_geometry.duration,
                    thread_geometry.fps,
                )),
            );
            document.insert(
                "mp4_url".to_string(),
                json!(format!("/out/{}/video.mp4", thread_video_id)),
            );
            document.insert("intermediates_cleaned".to_string(), json!(true));
            document.insert("motion_context_available".to_string(), json!(true));
            document.insert(
                "motion_context_windows".to_string(),
                json!(MINIMAX_H3_MOTION_CONTEXT_WINDOWS),
            );
        }
        if let Some(document) = result.as_ref() {
            if let Ok(bytes) = serde_json::to_vec_pretty(document) {
                let _ = std::fs::write(&result_path, bytes);
            }
        }
        cleanup_minimax_h3_conditioned_intermediates(&thread_out_dir);
        let _ = write_minimax_h3_job_status(
            &thread_out_dir,
            "done",
            "done",
            steps,
            steps,
            &format!(
                "MiniMax-H3 {} synchronized video and audio ready",
                thread_task
            ),
        );
        publish(WorkerEvent::Done {
            output_path: artifact.to_string_lossy().into_owned(),
        });
    });

    json_resp(
        StatusCode::ACCEPTED,
        &json!({
            "schema": "serenity.video_job.v1",
            "video_id": video_id,
            "prompt_id": video_id,
            "model": "minimax_h3",
            "runner": "minimax_h3_mojo_request",
            "profile_runner": runner,
            "profile": profile_id,
            "task": task,
            "ref2va_motion_context": combined_continuation,
            "continue_from": continuation_source.as_ref().map(|source| source.job_id.as_str()),
            "motion_context_frames": geometry.motion_context_frames,
            "backend": "mojo",
            "quant": quant,
            "attention_backend": attention,
            "step_cache": step_cache,
            "state": "queued",
            "status_url": format!("/out/{video_id}/status.json"),
            "result_url": format!("/out/{video_id}/result.json"),
            "request_url": format!("/out/{video_id}/request.json"),
        }),
    )
}
