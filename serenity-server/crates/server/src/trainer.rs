//! /v1/train — launch + observe + poke a Serenity live-trainer run (Phase D1/D5).
//!
//! DESIGN (audit doc "D1"): the trainer stack already has everything except a
//! server route. The route admits product inputs and launches a family lifecycle
//! runner; it never asks the caller to create a config or cache. Krea is the
//! first admitted vertical. Other families fail explicitly until their cache
//! producers and lifecycle runners satisfy the same contract:
//!   POST /v1/train          {model,dataset,output_name,steps,recipe:{...}}
//!   GET  /v1/train/status   -> child alive? + progress-log tail
//!   POST /v1/train/sample   {prompt} -> append {"action":"sample",...} to the
//!        trainer command jsonl (the D5 trigger's server half; the trainer polls
//!        the file at step boundaries and runs the inline live-LoRA sampler).
//!
//! GPU: a training run owns the card. The cross-path lease (gpu_lock) is
//! acquired at launch with kind="train" and held by a watcher thread until the
//! child exits — generate jobs queue behind it, subprocess paths 409.
//! One training run at a time (a second POST /v1/train 409s on the lease).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::{AppState, gpu_lock};

fn repository_root() -> PathBuf {
    crate::repository_root_path()
}

fn mojo_library_path(root: &Path) -> std::ffi::OsString {
    let mut paths = vec![
        root.join(".pixi/envs/default/lib"),
        root.join("serenitymojo/ops/cshim/lib"),
    ];
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).expect("join Mojo runtime library path")
}

/// Live state of the (single) training run this server launched.
#[derive(Default)]
pub(crate) struct TrainState {
    pub train_id: String,
    pub model: String,
    pub pid: u32,
    pub log_path: String,
    pub command_path: String,
    pub running: bool,
}

pub(crate) type TrainShared = Arc<Mutex<TrainState>>;

pub(crate) fn new_shared() -> TrainShared {
    Arc::new(Mutex::new(TrainState::default()))
}

/// model string -> live-trainer stem. Sources exist for every family below
/// (src/serenity_trainer/trainer/train_<stem>_real.mojo); a family whose
/// binary is not yet BUILT gets a fail-loud "build it" error at POST time.
fn trainer_stem_for_model(model: &str) -> Option<&'static str> {
    let m = model.to_ascii_lowercase();
    Some(if m.contains("klein") || m.contains("flux2") {
        "klein"
    } else if m.contains("zimage") || m.contains("z-image") {
        "zimage"
    } else if m.contains("ideogram") {
        "ideogram4"
    } else if m.contains("chroma") {
        "chroma"
    } else if m.contains("sd3") {
        "sd35"
    } else if m.contains("sdxl") || m.contains("sd_xl") {
        "sdxl"
    } else if m.contains("qwen") {
        "qwenimage"
    } else if m.contains("krea") {
        "krea2"
    } else if m.contains("anima") {
        "anima"
    } else if m.contains("ernie") {
        "ernie"
    } else if m.contains("hidream") {
        "hidream"
    } else if m.contains("flux") {
        "flux"
    } else {
        return None;
    })
}

fn err(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({ "error": msg }))).into_response()
}

/// POST /v1/train — validate, write the runner config, take the GPU lease,
/// spawn the live trainer, hold the lease until it exits.
pub async fn post_train(State(st): State<AppState>, body: String) -> Response {
    let repo_root = repository_root();
    let v: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, &format!("bad json: {e}")),
    };
    let model = v["model"].as_str().unwrap_or("").trim().to_string();
    if model.is_empty() {
        return err(StatusCode::BAD_REQUEST, "model required");
    }
    let steps = v["steps"].as_i64().unwrap_or(0);
    if !(1..=1_000_000).contains(&steps) {
        return err(StatusCode::BAD_REQUEST, "steps must be in [1, 1000000]");
    }
    let stem = match trainer_stem_for_model(&model) {
        Some(stem) => stem,
        None => {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                &format!("no trainer family is known for '{model}'"),
            );
        }
    };
    if stem != "krea2" {
        return err(
            StatusCode::NOT_IMPLEMENTED,
            &format!(
                "trainer family '{stem}' is not product-admitted: its automatic cache lifecycle runner is not implemented"
            ),
        );
    }
    let dataset = match v["dataset"]
        .as_str()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        Some(path) => match Path::new(path).canonicalize() {
            Ok(path) if path.is_dir() => path,
            Ok(_) => return err(StatusCode::BAD_REQUEST, "dataset must be a directory"),
            Err(e) => {
                return err(
                    StatusCode::BAD_REQUEST,
                    &format!("dataset is not readable: {e}"),
                );
            }
        },
        None => return err(StatusCode::BAD_REQUEST, "dataset required"),
    };
    let output_name = v["output_name"].as_str().unwrap_or("train").trim();
    if output_name.is_empty()
        || !output_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(
            StatusCode::BAD_REQUEST,
            "output_name may contain only letters, numbers, '-' and '_'",
        );
    }
    let recipe = &v["recipe"];
    if !recipe.is_null() && !recipe.is_object() {
        return err(StatusCode::BAD_REQUEST, "recipe must be an object");
    }
    let bin = repo_root.join("trainer/webui/target/release/serenity-trainer-runner");
    if !bin.is_file() {
        return err(
            StatusCode::NOT_IMPLEMENTED,
            &format!("Krea lifecycle runner is not built: {}", bin.display()),
        );
    }

    // One run at a time + cross-path GPU exclusivity, via the shared lease.
    let tag = gpu_lock::next_tag("train");
    let guard = match gpu_lock::try_acquire(&st.gpu_owner, "train", &tag) {
        Ok(g) => g,
        Err(cur) => {
            return (
                StatusCode::CONFLICT,
                Json(gpu_lock::gpu_busy_conflict_report("train", &cur)),
            )
                .into_response();
        }
    };

    // Rotate the command channel at run start (audit #5: three appenders,
    // zero truncators -> unbounded growth, which is the precondition for the
    // stale-replay class). The lease is held, so no live trainer is reading.
    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let train_id = format!("{output_name}-{n:04}");
    let out_dir = repo_root.join("output").join(&train_id);
    for directory in ["logs", "cache", "checkpoints", "samples"] {
        if let Err(e) = std::fs::create_dir_all(out_dir.join(directory)) {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("create workspace: {e}"),
            );
        }
    }
    let command_path = out_dir.join("commands.jsonl");
    let _ = std::fs::write(&command_path, b"");
    let defaults_path = repo_root.join("serenitymojo/configs/krea2.json");
    let mut config: Value = match std::fs::read(&defaults_path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|e| e.to_string()))
    {
        Ok(config) => config,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("load Krea defaults: {e}"),
            );
        }
    };
    let Some(object) = config.as_object_mut() else {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Krea defaults are not an object",
        );
    };
    if let Some(overrides) = recipe.as_object() {
        const PRODUCT_OWNED_FIELDS: &[&str] = &[
            "schema",
            "run_id",
            "model_type",
            "dataset_path",
            "workspace_dir",
            "cache_dir",
            "dataset_cache_dir",
            "output_model_destination",
            "save_filename_prefix",
            "only_cache",
            "resume_state",
            "start_step",
            "resume_parent_cache_identity",
            "effective_config_hash",
            "resume_state_identity",
            "resume_manifest",
            "resume_parent_checkpoint_identity",
            "checkpoint",
            "vae",
            "vae_encoder",
            "qwen_encoder",
            "qwen_tokenizer",
            "t5_encoder",
            "t5_tokenizer",
            "validation_prompts_file",
        ];
        for (key, value) in overrides {
            if PRODUCT_OWNED_FIELDS.contains(&key.as_str()) {
                return err(
                    StatusCode::BAD_REQUEST,
                    &format!("recipe may not override product-owned field '{key}'"),
                );
            }
            object.insert(key.clone(), value.clone());
        }
    }
    object.insert("model_type".into(), json!(stem));
    object.insert("schema".into(), json!("serenity.run.v1"));
    object.insert("run_id".into(), json!(train_id));
    object.insert("dataset_path".into(), json!(dataset));
    object.insert("workspace_dir".into(), json!(out_dir));
    object.insert("cache_dir".into(), json!(out_dir.join("cache/data")));
    object.insert(
        "dataset_cache_dir".into(),
        json!(out_dir.join("cache/data")),
    );
    object.insert("max_steps".into(), json!(steps));
    object.insert(
        "output_model_destination".into(),
        json!(out_dir.join(format!("checkpoints/{train_id}.safetensors"))),
    );
    object.insert("save_filename_prefix".into(), json!(train_id));
    object.insert("only_cache".into(), v["only_cache"].clone());
    if let Some(resume) = v["resume"]
        .as_str()
        .filter(|value| !value.trim().is_empty())
    {
        object.insert("resume_state".into(), json!(resume));
    }
    let config_path = out_dir.join("run.json");
    if let Err(e) = std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("write config: {e}"),
        );
    }
    let log_path = out_dir.join("logs/train.log");
    let log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, &format!("log: {e}")),
    };
    let args = [
        "--repo".to_string(),
        repo_root.to_string_lossy().into_owned(),
        "--run".to_string(),
        config_path.to_string_lossy().into_owned(),
    ];
    let child = std::process::Command::new(&bin)
        .args(&args)
        .current_dir(&repo_root)
        .env("LD_LIBRARY_PATH", mojo_library_path(&repo_root))
        .stdout(log_file.try_clone().expect("dup log"))
        .stderr(log_file)
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("spawn {}: {e}", bin.display()),
            );
        }
    };
    let pid = child.id();
    {
        let mut ts = st.train.lock().expect("train state");
        *ts = TrainState {
            train_id: train_id.clone(),
            model: model.clone(),
            pid,
            log_path: log_path.to_string_lossy().into_owned(),
            command_path: command_path.to_string_lossy().into_owned(),
            running: true,
        };
    }
    // Hold the GPU lease until the trainer exits; then mark not-running.
    let train_shared = st.train.clone();
    std::thread::Builder::new()
        .name("trainer-wait".into())
        .spawn(move || {
            let _hold = guard; // RAII: released when the child exits (or this thread dies)
            let status = child.wait();
            tracing::info!(?status, "trainer exited");
            if let Ok(mut ts) = train_shared.lock() {
                ts.running = false;
            }
        })
        .ok();
    (
        StatusCode::ACCEPTED,
        Json(json!({
            "train_id": train_id,
            "model": model,
            "pid": pid,
            "binary": bin,
            "steps": steps,
            "config_path": config_path.to_string_lossy(),
            "log_path": log_path.to_string_lossy(),
        })),
    )
        .into_response()
}

/// GET /v1/train/status — child liveness + the trainer log tail (the
/// print_trainer_progress standard lines ride the log verbatim).
pub async fn get_train_status(State(st): State<AppState>) -> Response {
    let ts = st.train.lock().expect("train state");
    if ts.train_id.is_empty() {
        return err(
            StatusCode::NOT_FOUND,
            "no training run launched this session",
        );
    }
    let tail = std::fs::read_to_string(&ts.log_path)
        .map(|s| {
            let lines: Vec<&str> = s.lines().rev().take(40).collect();
            lines.into_iter().rev().collect::<Vec<_>>().join("\n")
        })
        .unwrap_or_default();
    Json(json!({
        "train_id": ts.train_id,
        "model": ts.model,
        "pid": ts.pid,
        "running": ts.running,
        "log_path": ts.log_path,
        "log_tail": tail,
    }))
    .into_response()
}

/// POST /v1/train/sample {prompt} — the D5 on-demand trigger's SERVER half:
/// append the command line the trainer polls at step boundaries. Determinism
/// contract (trainer side): the inline sampler runs with the LIVE adapters and
/// its own seed derivation, never drawing from the training RNG stream.
pub async fn post_train_sample(State(st): State<AppState>, body: String) -> Response {
    let v: Value = serde_json::from_str(&body).unwrap_or(json!({}));
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return err(StatusCode::BAD_REQUEST, "prompt required");
    }
    let command_path = {
        let ts = st.train.lock().expect("train state");
        if !ts.running {
            return err(
                StatusCode::CONFLICT,
                "no training run is active; the sample trigger targets the live trainer",
            );
        }
        // Audit #3: only trainers that POLL the command channel may be
        // triggered — a 200 for a non-polling trainer is a contract lie (the
        // command would sit unread and replay into a later run).
        let stem = trainer_stem_for_model(&ts.model).unwrap_or("");
        if stem != "zimage" {
            return err(
                StatusCode::NOT_IMPLEMENTED,
                &format!(
                    "on-demand sampling is wired for the zimage trainer only (running: '{}'); \
                     the other trainers do not poll the command channel yet",
                    ts.model
                ),
            );
        }
        PathBuf::from(&ts.command_path)
    };
    let line = json!({ "action": "sample", "prompt": prompt }).to_string();
    let res = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&command_path)
        .and_then(|mut f| writeln!(f, "{line}"));
    match res {
        Ok(()) => Json(json!({ "queued": true, "command_file": command_path })).into_response(),
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("append {}: {e}", command_path.display()),
        ),
    }
}
