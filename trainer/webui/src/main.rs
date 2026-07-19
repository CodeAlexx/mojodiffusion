use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Write},
    path::{Path as FsPath, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
use tokio::{process::Command, sync::Mutex};

#[derive(Clone)]
struct AppState {
    repo_root: PathBuf,
    output_root: PathBuf,
    children: Arc<Mutex<HashMap<String, u32>>>,
}

#[derive(Debug, Deserialize)]
struct RunRequest {
    model: String,
    dataset: String,
    output_name: String,
    #[serde(default)]
    recipe: Value,
    #[serde(default)]
    resume: Option<String>,
    #[serde(default)]
    only_cache: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct StatusRecord {
    schema: String,
    run_id: String,
    state: String,
    stage: String,
    updated_at: u64,
    pid: Option<u32>,
    error_code: Option<String>,
    message: String,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn hash_bytes(mut state: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

fn file_content_hash(path: &FsPath) -> Result<String, String> {
    let file = File::open(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut state = FNV_OFFSET;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|e| format!("hash {}: {e}", path.display()))?;
        if count == 0 {
            return Ok(format!("fnv1a64:{state:016x}"));
        }
        state = hash_bytes(state, &buffer[..count]);
    }
}

fn value_content_hash(value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    Ok(format!("fnv1a64:{:016x}", hash_bytes(FNV_OFFSET, &bytes)))
}

fn resume_manifest_path(state_path: &FsPath) -> PathBuf {
    PathBuf::from(format!("{}.resume.json", state_path.display()))
}

fn discover_repo_root() -> Result<PathBuf, String> {
    if let Ok(raw) = std::env::var("SERENITY_ROOT") {
        let path = PathBuf::from(raw);
        if path.join("pixi.toml").is_file() {
            return path.canonicalize().map_err(|e| e.to_string());
        }
        return Err("SERENITY_ROOT does not point to a Serenity repository".into());
    }
    let mut path = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        if path.join("pixi.toml").is_file() && path.join("serenitymojo").is_dir() {
            return path.canonicalize().map_err(|e| e.to_string());
        }
        if !path.pop() {
            return Err("run the trainer from the repository or set SERENITY_ROOT".into());
        }
    }
}

fn safe_name(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty() || value.len() > 80 {
        return None;
    }
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        Some(value.to_string())
    } else {
        None
    }
}

fn safe_run_id(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty() || value.len() > 128 {
        return None;
    }
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        Some(value.to_string())
    } else {
        None
    }
}

fn new_run_id(name: &str) -> String {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{name}-{stamp}")
}

fn write_json_atomic(path: &FsPath, value: &impl Serialize) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    let body = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    fs::write(&tmp, body).map_err(|e| e.to_string())?;
    fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn initial_status(run_id: &str) -> StatusRecord {
    StatusRecord {
        schema: "serenity.run-status.v1".into(),
        run_id: run_id.into(),
        state: "created".into(),
        stage: "admission".into(),
        updated_at: now(),
        pid: None,
        error_code: None,
        message: "workspace and log created".into(),
    }
}

fn mark_admission_failure(workspace: &FsPath, mut status: StatusRecord, code: &str, message: &str) {
    status.state = "failed".into();
    status.stage = "validating".into();
    status.updated_at = now();
    status.error_code = Some(code.into());
    status.message = message.into();
    let _ = write_json_atomic(&workspace.join("status.json"), &status);
    if let Ok(mut log) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(workspace.join("logs/train.log"))
    {
        let _ = writeln!(log, "[failed] {code}: {message}");
    }
}

fn serenity_models_root() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("SERENITY_MODEL_ROOT") {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .ok_or("HOME is unavailable; set SERENITY_MODEL_ROOT to the installed model registry")?;
    Ok(PathBuf::from(home).join(".serenity/models"))
}

fn materialize_effective_paths(repo: &FsPath, config: &mut Value) -> Result<(), String> {
    const PATH_KEYS: &[&str] = &[
        "checkpoint",
        "vae",
        "vae_encoder",
        "qwen_encoder",
        "qwen_tokenizer",
        "t5_encoder",
        "t5_tokenizer",
        "validation_prompts_file",
    ];
    for key in PATH_KEYS {
        let Some(raw) = config[*key].as_str() else {
            continue;
        };
        let path = PathBuf::from(raw);
        let resolved = if let Ok(relative) = path.strip_prefix("serenity-models") {
            serenity_models_root()?.join(relative)
        } else if path.is_absolute() {
            path
        } else {
            repo.join(path)
        };
        config[*key] = json!(resolved);
    }
    Ok(())
}

fn resume_compatibility_view(mut config: Value) -> Value {
    if let Some(object) = config.as_object_mut() {
        for key in [
            "schema",
            "run_id",
            "workspace_dir",
            "cache_dir",
            "dataset_cache_dir",
            "output_model_destination",
            "save_filename_prefix",
            "only_cache",
            "max_steps",
            "save_every",
            "sample_every",
            "resume_state",
            "start_step",
            "resume_parent_cache_identity",
            "effective_config_hash",
            "resume_state_identity",
            "resume_manifest",
            "resume_parent_checkpoint_identity",
        ] {
            object.remove(key);
        }
    }
    config
}

fn merge_effective_config(
    state: &AppState,
    request: &RunRequest,
    run_id: &str,
    workspace: &FsPath,
) -> Result<Value, String> {
    let model = request.model.to_ascii_lowercase();
    let defaults_path = match model.as_str() {
        "krea2" => state.repo_root.join("serenitymojo/configs/krea2.json"),
        "anima" => state.repo_root.join("serenitymojo/configs/anima.json"),
        "chroma" => state.repo_root.join("serenitymojo/configs/chroma.json"),
        _ => return Err(format!("model '{}' is not admitted yet", request.model)),
    };
    let mut config: Value = serde_json::from_slice(
        &fs::read(&defaults_path).map_err(|e| format!("read {}: {e}", defaults_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", defaults_path.display()))?;
    let object = config
        .as_object_mut()
        .ok_or_else(|| "model defaults must be a JSON object".to_string())?;
    if let Some(overrides) = request.recipe.as_object() {
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
                return Err(format!(
                    "recipe may not override product-owned field '{key}'"
                ));
            }
            object.insert(key.clone(), value.clone());
        }
    } else if !request.recipe.is_null() {
        return Err("recipe must be a JSON object".into());
    }
    let dataset = FsPath::new(request.dataset.trim())
        .canonicalize()
        .map_err(|e| format!("dataset is not readable: {e}"))?;
    if !dataset.is_dir() {
        return Err("dataset must be a directory of image and caption pairs".into());
    }
    let cache_prefix = workspace.join("cache/data");
    object.insert("schema".into(), json!("serenity.run.v1"));
    object.insert("run_id".into(), json!(run_id));
    object.insert("dataset_path".into(), json!(dataset));
    object.insert("workspace_dir".into(), json!(workspace));
    object.insert("cache_dir".into(), json!(cache_prefix));
    object.insert("dataset_cache_dir".into(), json!(cache_prefix));
    object.insert(
        "output_model_destination".into(),
        json!(workspace.join(format!("checkpoints/{run_id}.safetensors"))),
    );
    object.insert("save_filename_prefix".into(), json!(run_id));
    object.insert("only_cache".into(), json!(request.only_cache));
    let _ = object;
    materialize_effective_paths(&state.repo_root, &mut config)?;
    let effective_config_hash = value_content_hash(&resume_compatibility_view(config.clone()))?;
    config
        .as_object_mut()
        .expect("model defaults were validated as an object")
        .insert("effective_config_hash".into(), json!(effective_config_hash));
    if let Some(resume) = request.resume.as_ref().filter(|s| !s.trim().is_empty()) {
        let resume_path = FsPath::new(resume.trim())
            .canonicalize()
            .map_err(|e| format!("resume state is not readable: {e}"))?;
        if !resume_path.is_file() {
            return Err("resume state must be a file".into());
        }
        let parent_workspace = resume_path
            .parent()
            .and_then(FsPath::parent)
            .ok_or("resume state must belong to a run checkpoint directory")?;
        if resume_path.parent() != Some(&parent_workspace.join("checkpoints")) {
            return Err("resume state must belong to a run checkpoint directory".into());
        }
        let parent_config: Value = serde_json::from_slice(
            &fs::read(parent_workspace.join("run.json"))
                .map_err(|e| format!("read resume parent run config: {e}"))?,
        )
        .map_err(|e| format!("parse resume parent run config: {e}"))?;
        if resume_compatibility_view(parent_config.clone())
            != resume_compatibility_view(config.clone())
        {
            return Err("resume state is incompatible with the requested model, dataset, assets, or training recipe".into());
        }
        let parent_manifest: Value = serde_json::from_slice(
            &fs::read(parent_workspace.join("cache/manifest.json"))
                .map_err(|e| format!("read resume parent cache manifest: {e}"))?,
        )
        .map_err(|e| format!("parse resume parent cache manifest: {e}"))?;
        if parent_manifest["complete"].as_bool() != Some(true) {
            return Err("resume parent cache manifest is incomplete".into());
        }
        let parent_cache_identity = parent_manifest["identity_hash"]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or("resume parent cache manifest has no identity hash")?;
        let resume_manifest_path = resume_manifest_path(&resume_path);
        let resume_manifest: Value = serde_json::from_slice(
            &fs::read(&resume_manifest_path)
                .map_err(|e| format!("read resume state manifest: {e}"))?,
        )
        .map_err(|e| format!("parse resume state manifest: {e}"))?;
        if resume_manifest["schema"] != json!("serenity.trainer-resume.v1") {
            return Err("resume state manifest schema is unsupported".into());
        }
        if resume_manifest["model"] != config["model_type"] {
            return Err("resume state manifest belongs to a different model".into());
        }
        if resume_manifest["state_filename"].as_str()
            != resume_path.file_name().and_then(|value| value.to_str())
        {
            return Err("resume state manifest does not name the selected state file".into());
        }
        let state_identity = file_content_hash(&resume_path)?;
        if resume_manifest["state_content_hash"].as_str() != Some(&state_identity) {
            return Err("resume state content hash does not match its manifest".into());
        }
        if resume_manifest["cache_identity"].as_str() != Some(parent_cache_identity) {
            return Err("resume state cache identity does not match its parent run".into());
        }
        if resume_manifest["effective_config_hash"].as_str()
            != config["effective_config_hash"].as_str()
        {
            return Err("resume state effective configuration identity is incompatible".into());
        }
        let start_step = resume_manifest["completed_step"]
            .as_u64()
            .ok_or("resume state manifest has no completed step")?;
        let max_steps = config["max_steps"]
            .as_u64()
            .ok_or("requested max_steps must be a positive integer")?;
        if start_step == 0 || start_step >= max_steps {
            return Err("resume state completed step must be below requested max_steps".into());
        }
        let object = config
            .as_object_mut()
            .expect("model defaults were validated as an object");
        object.insert("resume_state".into(), json!(resume_path));
        object.insert("start_step".into(), json!(start_step));
        object.insert(
            "resume_state_identity".into(),
            json!(state_identity.clone()),
        );
        object.insert("resume_manifest".into(), json!(resume_manifest_path));
        object.insert(
            "resume_parent_checkpoint_identity".into(),
            json!(state_identity),
        );
        object.insert(
            "resume_parent_cache_identity".into(),
            json!(parent_cache_identity),
        );
    }
    Ok(config)
}

async fn spawn_runner(
    state: &Arc<AppState>,
    run_id: &str,
    run_path: &FsPath,
    log_path: &FsPath,
) -> Result<u32, String> {
    let runner = state
        .repo_root
        .join("trainer/webui/target/release/serenity-trainer-runner");
    if !runner.is_file() {
        return Err(format!(
            "trainer lifecycle runner is missing: {} (run `pixi run build-trainer`)",
            runner.display()
        ));
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|error| format!("open run log: {error}"))?;
    let mut command = Command::new(&runner);
    command
        .arg("--repo")
        .arg(&state.repo_root)
        .arg("--run")
        .arg(run_path)
        .current_dir(&state.repo_root)
        .process_group(0)
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log));
    let mut child = command
        .spawn()
        .map_err(|error| format!("start trainer runner {}: {error}", runner.display()))?;
    let pid = child.id().ok_or("trainer runner did not report a pid")?;
    state.children.lock().await.insert(run_id.to_string(), pid);
    let children = state.children.clone();
    let finished_id = run_id.to_string();
    tokio::spawn(async move {
        let _ = child.wait().await;
        children.lock().await.remove(&finished_id);
    });
    Ok(pid)
}

async fn create_run(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RunRequest>,
) -> impl IntoResponse {
    let requested_name = safe_name(&request.output_name).unwrap_or_else(|| "invalid-name".into());
    let run_id = new_run_id(&requested_name);
    let workspace = state.output_root.join(&run_id);
    for directory in ["logs", "cache", "checkpoints", "samples"] {
        if let Err(error) = fs::create_dir_all(workspace.join(directory)) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("create workspace: {error}")})),
            );
        }
    }
    let log_path = workspace.join("logs/train.log");
    if let Err(error) = File::create(&log_path) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!("create log: {error}")})),
        );
    }
    let status = initial_status(&run_id);
    if let Err(error) = write_json_atomic(&workspace.join("status.json"), &status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": error})),
        );
    }

    if safe_name(&request.output_name).is_none() {
        let message = "output name may contain only letters, numbers, '-' and '_'";
        mark_admission_failure(&workspace, status, "INVALID_OUTPUT_NAME", message);
        return (
            StatusCode::ACCEPTED,
            Json(json!({"run_id": run_id, "state": "failed", "message": message})),
        );
    }

    let config = match merge_effective_config(&state, &request, &run_id, &workspace) {
        Ok(config) => config,
        Err(message) => {
            mark_admission_failure(&workspace, status, "INVALID_RUN", &message);
            return (
                StatusCode::ACCEPTED,
                Json(json!({"run_id": run_id, "state": "failed", "message": message})),
            );
        }
    };
    let run_path = workspace.join("run.json");
    if let Err(message) = write_json_atomic(&run_path, &config) {
        mark_admission_failure(&workspace, status, "WRITE_RUN_CONFIG", &message);
        return (
            StatusCode::ACCEPTED,
            Json(json!({"run_id": run_id, "state": "failed", "message": message})),
        );
    }

    if let Err(message) = spawn_runner(&state, &run_id, &run_path, &log_path).await {
        mark_admission_failure(&workspace, status, "START_RUNNER", &message);
        return (
            StatusCode::ACCEPTED,
            Json(json!({"run_id": run_id, "state": "failed", "message": message})),
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(
            json!({"run_id": run_id, "state": "created", "log": format!("/api/runs/{run_id}/log")}),
        ),
    )
}

fn read_runs(output_root: &FsPath) -> Vec<Value> {
    let mut rows = Vec::new();
    let Ok(entries) = fs::read_dir(output_root) else {
        return rows;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(status) = fs::read(path.join("status.json")) else {
            continue;
        };
        if let Ok(mut value) = serde_json::from_slice::<Value>(&status) {
            value["workspace"] = json!(path);
            if let Ok(run) = fs::read(path.join("run.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .ok_or(())
            {
                value["model"] = run["model_type"].clone();
                value["dataset"] = run["dataset_path"].clone();
            }
            let mut states: Vec<PathBuf> = fs::read_dir(path.join("checkpoints"))
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|entry| {
                    let name = entry.file_name().and_then(|v| v.to_str()).unwrap_or("");
                    entry.is_file()
                        && (name.ends_with(".state")
                            || name.contains("state") && name.ends_with(".safetensors"))
                        && resume_manifest_path(entry).is_file()
                })
                .collect();
            states.sort();
            value["resume_states"] = json!(states);
            rows.push(value);
        }
    }
    rows.sort_by(|a, b| b["updated_at"].as_u64().cmp(&a["updated_at"].as_u64()));
    rows
}

fn is_terminal_state(state: &str) -> bool {
    matches!(state, "completed" | "failed" | "cancelled" | "interrupted")
}

fn model_catalog() -> Value {
    json!({"models": [
        {"id":"krea2", "label":"Krea 2 Raw", "admitted":true, "readiness":"ready", "reason":"automatic dataset stage, conditioning/VAE cache, training, checkpoint, and resume"},
        {"id":"anima", "label":"Anima", "admitted":true, "readiness":"ready", "reason":"automatic dataset stage, Qwen/VAE cache, training, checkpoint, and resume"},
        {"id":"chroma", "label":"Chroma1 HD", "admitted":true, "readiness":"ready", "reason":"automatic dataset stage, T5/Flux-VAE cache, FP8-resident training, checkpoint, and exact resume"},
        {"id":"ideogram4", "label":"Ideogram 4", "admitted":false, "readiness":"cache-producer-required", "reason":"optimizer path passed two steps; trainer-owned raw cache producer is not wired"},
        {"id":"ernie", "label":"ERNIE Image", "admitted":false, "readiness":"assets-and-cache-required", "reason":"configured transformer weights and trainer-owned cache producer are absent"},
        {"id":"flux", "label":"FLUX.1", "admitted":false, "readiness":"cache-producer-required", "reason":"optimizer path passed two steps with an existing cache; raw cache producer is not wired"},
        {"id":"hidream", "label":"HiDream O1", "admitted":false, "readiness":"cache-producer-required", "reason":"optimizer path passed two steps with an existing cache; raw cache producer is not wired"},
        {"id":"klein", "label":"Klein", "admitted":false, "readiness":"raw-cache-contract-required", "reason":"optimizer path passed two steps; current Flux2 VAE path emits prepared rather than required raw cache latents"},
        {"id":"l2p", "label":"L2P", "admitted":false, "readiness":"assets-and-cache-required", "reason":"configured checkpoint and native trainer cache are absent"},
        {"id":"ltx2", "label":"LTX-2", "admitted":false, "readiness":"implementation-required", "reason":"production audio/video trainer entrypoint is explicitly not implemented"},
        {"id":"qwenimage", "label":"Qwen Image", "admitted":false, "readiness":"lifecycle-wiring-required", "reason":"optimizer path and cache components exist; the automatic product lifecycle is not wired"},
        {"id":"sd35", "label":"Stable Diffusion 3.5", "admitted":false, "readiness":"lifecycle-wiring-required", "reason":"optimizer path and conditioning components exist; the automatic product lifecycle is not wired"},
        {"id":"sdxl", "label":"SDXL", "admitted":false, "readiness":"cache-producer-required", "reason":"installed UNet passed loading; trainer-owned cache producer is absent"},
        {"id":"wan22", "label":"Wan 2.2", "admitted":false, "readiness":"assets-and-cache-required", "reason":"configured 14B training checkpoint and native trainer cache are absent"},
        {"id":"zimage", "label":"Z-Image", "admitted":false, "readiness":"cache-producer-required", "reason":"optimizer path passed two steps with an existing cache; automatic conditioning/VAE cache is not wired"}
    ]})
}

async fn list_models() -> Json<Value> {
    Json(model_catalog())
}

fn reconcile_runs(output_root: &FsPath) -> HashMap<String, u32> {
    let mut active = HashMap::new();
    let Ok(entries) = fs::read_dir(output_root) else {
        return active;
    };
    for entry in entries.flatten() {
        let workspace = entry.path();
        let status_path = workspace.join("status.json");
        let Ok(bytes) = fs::read(&status_path) else {
            continue;
        };
        let Ok(mut status) = serde_json::from_slice::<StatusRecord>(&bytes) else {
            continue;
        };
        if is_terminal_state(&status.state) {
            continue;
        }
        let alive = status
            .pid
            .map(|pid| FsPath::new("/proc").join(pid.to_string()).exists())
            .unwrap_or(false);
        if alive {
            active.insert(status.run_id.clone(), status.pid.expect("checked pid"));
            continue;
        }
        status.state = "interrupted".into();
        status.stage = "interrupted".into();
        status.updated_at = now();
        status.pid = None;
        status.error_code = Some("RUNNER_NOT_ALIVE".into());
        status.message = "runner was not alive when the trainer UI recovered durable state".into();
        let _ = write_json_atomic(&status_path, &status);
    }
    active
}

async fn list_runs(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"runs": read_runs(&state.output_root)}))
}

async fn run_log(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    let Some(run_id) = safe_run_id(&run_id) else {
        return (StatusCode::BAD_REQUEST, "invalid run id".to_string());
    };
    let path = state.output_root.join(run_id).join("logs/train.log");
    match fs::read_to_string(path) {
        Ok(body) => (StatusCode::OK, body),
        Err(_) => (StatusCode::NOT_FOUND, "run log not found".into()),
    }
}

async fn cancel_run(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    let Some(run_id) = safe_run_id(&run_id) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid run id"})),
        );
    };
    let pid = state.children.lock().await.get(&run_id).copied();
    let Some(pid) = pid else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "run is not active"})),
        );
    };
    let result = std::process::Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .status();
    match result {
        Ok(status) if status.success() => {
            let workspace = state.output_root.join(&run_id);
            let cancelled = StatusRecord {
                schema: "serenity.run-status.v1".into(),
                run_id: run_id.clone(),
                state: "cancelled".into(),
                stage: "cancelled".into(),
                updated_at: now(),
                pid: None,
                error_code: None,
                message: "cancelled by user".into(),
            };
            let _ = write_json_atomic(&workspace.join("status.json"), &cancelled);
            if let Ok(mut log) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(workspace.join("logs/train.log"))
            {
                let _ = writeln!(log, "[cancelled] cancelled by user");
            }
            (StatusCode::ACCEPTED, Json(json!({"cancelled": run_id})))
        }
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "cancel signal failed"})),
        ),
    }
}

async fn restart_run(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    let Some(run_id) = safe_run_id(&run_id) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid run id"})),
        );
    };
    if state.children.lock().await.contains_key(&run_id) {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "run is already active"})),
        );
    }
    let workspace = state.output_root.join(&run_id);
    let run_path = workspace.join("run.json");
    let log_path = workspace.join("logs/train.log");
    if !run_path.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "durable run config not found"})),
        );
    }
    let status = StatusRecord {
        schema: "serenity.run-status.v1".into(),
        run_id: run_id.clone(),
        state: "created".into(),
        stage: "restart".into(),
        updated_at: now(),
        pid: None,
        error_code: None,
        message: "restarting from durable run config".into(),
    };
    if let Err(error) = write_json_atomic(&workspace.join("status.json"), &status) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": error})),
        );
    }
    if let Ok(mut log) = OpenOptions::new().create(true).append(true).open(&log_path) {
        let _ = writeln!(log, "[restart] restarting from durable run config");
    }
    match spawn_runner(&state, &run_id, &run_path, &log_path).await {
        Ok(pid) => (
            StatusCode::ACCEPTED,
            Json(json!({"restarted": run_id, "pid": pid})),
        ),
        Err(message) => {
            mark_admission_failure(&workspace, status, "START_RUNNER", &message);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": message})),
            )
        }
    }
}

#[tokio::main]
async fn main() {
    let repo_root = discover_repo_root().unwrap_or_else(|message| panic!("{message}"));
    let output_root = repo_root.join("output");
    fs::create_dir_all(&output_root).expect("create output/runs");
    let recovered_children = reconcile_runs(&output_root);
    let state = Arc::new(AppState {
        repo_root: repo_root.clone(),
        output_root,
        children: Arc::new(Mutex::new(recovered_children)),
    });
    let static_dir = repo_root.join("trainer/webui/static");
    let app = Router::new()
        .route("/api/models", get(list_models))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/:id/log", get(run_log))
        .route("/api/runs/:id/cancel", post(cancel_run))
        .route("/api/runs/:id/restart", post(restart_run))
        .nest_service(
            "/",
            tower_http::services::ServeDir::new(static_dir).append_index_html_on_directories(true),
        )
        .with_state(state);
    let bind = std::env::var("SERENITY_TRAINER_BIND").unwrap_or_else(|_| "127.0.0.1:8188".into());
    println!("Serenity trainer UI: http://{bind}");
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("bind trainer UI");
    axum::serve(listener, app).await.expect("serve trainer UI");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "serenity-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&path).expect("create temporary test directory");
        path
    }

    #[test]
    fn run_names_are_path_safe() {
        assert_eq!(safe_name("eri_krea"), Some("eri_krea".into()));
        assert_eq!(safe_name("../escape"), None);
        assert_eq!(safe_name("with space"), None);
        let long_name = "a".repeat(80);
        let run_id = new_run_id(&long_name);
        assert!(safe_run_id(&run_id).is_some());
        assert_eq!(safe_run_id("../escape"), None);
    }

    #[test]
    fn trainer_generates_config_and_cache_paths_inside_workspace() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(FsPath::parent)
            .expect("repository root")
            .to_path_buf();
        let scratch = temporary_directory("generated-paths");
        let dataset = scratch.join("dataset");
        let workspace = scratch.join("output/eri-krea-test");
        fs::create_dir_all(&dataset).expect("create dataset");
        fs::create_dir_all(&workspace).expect("create workspace");
        let state = AppState {
            repo_root,
            output_root: scratch.join("output"),
            children: Arc::new(Mutex::new(HashMap::new())),
        };
        let request = RunRequest {
            model: "krea2".into(),
            dataset: dataset.display().to_string(),
            output_name: "eri-krea".into(),
            recipe: json!({"max_steps": 1}),
            resume: None,
            only_cache: false,
        };
        let config = merge_effective_config(&state, &request, "eri-krea-test", &workspace)
            .expect("generate effective config");
        assert_eq!(config["workspace_dir"], json!(workspace));
        assert_eq!(config["cache_dir"], json!(workspace.join("cache/data")));
        assert_eq!(
            config["output_model_destination"],
            json!(workspace.join("checkpoints/eri-krea-test.safetensors"))
        );
        let protected = RunRequest {
            model: "krea2".into(),
            dataset: dataset.display().to_string(),
            output_name: "eri-krea".into(),
            recipe: json!({"checkpoint": "/tmp/operator-selected.safetensors"}),
            resume: None,
            only_cache: false,
        };
        let error =
            merge_effective_config(&state, &protected, "eri-krea-protected-test", &workspace)
                .expect_err("recipe must not replace product-owned asset paths");
        assert!(error.contains("product-owned field 'checkpoint'"));
        fs::remove_dir_all(scratch).expect("remove temporary test directory");
    }

    #[test]
    fn catalog_reports_every_trainer_family_without_false_admission() {
        let catalog = model_catalog();
        let models = catalog["models"].as_array().expect("model catalog array");
        assert_eq!(models.len(), 15);
        let admitted: Vec<&str> = models
            .iter()
            .filter(|model| model["admitted"] == json!(true))
            .filter_map(|model| model["id"].as_str())
            .collect();
        assert_eq!(admitted, vec!["krea2", "anima", "chroma"]);
        assert!(models
            .iter()
            .all(|model| model["reason"].as_str().is_some()));
    }

    #[test]
    fn resume_compatibility_ignores_only_run_scoped_fields() {
        let parent = json!({
            "schema": "serenity.run.v1",
            "run_id": "parent",
            "model_type": "anima",
            "dataset_path": "/data/eri",
            "checkpoint": "/models/anima.safetensors",
            "learning_rate": 0.0001,
            "max_steps": 2,
            "save_every": 1,
            "workspace_dir": "/output/parent",
            "cache_dir": "/output/parent/cache/data",
            "output_model_destination": "/output/parent/checkpoints/parent.safetensors"
        });
        let continuation = json!({
            "schema": "serenity.run.v1",
            "run_id": "continuation",
            "model_type": "anima",
            "dataset_path": "/data/eri",
            "checkpoint": "/models/anima.safetensors",
            "learning_rate": 0.0001,
            "max_steps": 4,
            "save_every": 0,
            "workspace_dir": "/output/continuation",
            "cache_dir": "/output/continuation/cache/data",
            "output_model_destination": "/output/continuation/checkpoints/continuation.safetensors",
            "resume_state": "/output/parent/checkpoints/parent.state.safetensors",
            "start_step": 2,
            "resume_parent_cache_identity": "fnv1a64:abc"
        });
        assert_eq!(
            resume_compatibility_view(parent.clone()),
            resume_compatibility_view(continuation.clone())
        );

        let mut changed_recipe = continuation.clone();
        changed_recipe["learning_rate"] = json!(0.0002);
        assert_ne!(
            resume_compatibility_view(parent.clone()),
            resume_compatibility_view(changed_recipe)
        );

        let mut changed_dataset = continuation;
        changed_dataset["dataset_path"] = json!("/data/someone-else");
        assert_ne!(
            resume_compatibility_view(parent),
            resume_compatibility_view(changed_dataset)
        );
    }
}
