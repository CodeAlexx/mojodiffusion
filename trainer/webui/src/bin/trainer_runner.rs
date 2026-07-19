use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::{SystemTime, UNIX_EPOCH},
};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

struct CacheLock(PathBuf);

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn hash_bytes(mut state: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

fn hash_file_into(mut state: u64, path: &Path) -> Result<u64, String> {
    let file = fs::File::open(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|e| format!("hash {}: {e}", path.display()))?;
        if count == 0 {
            return Ok(state);
        }
        state = hash_bytes(state, &buffer[..count]);
    }
}

fn hash_tree_into(mut state: u64, root: &Path, path: &Path) -> Result<u64, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    let relative = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
    state = hash_bytes(state, relative.as_bytes());
    state = hash_bytes(state, &[0]);
    if metadata.is_dir() {
        let mut children: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("read directory {}: {e}", path.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect();
        children.sort();
        for child in children {
            state = hash_tree_into(state, root, &child)?;
        }
        Ok(state)
    } else if metadata.is_file() {
        hash_file_into(state, path)
    } else {
        Err(format!(
            "unsupported cache identity input: {}",
            path.display()
        ))
    }
}

fn content_hash(path: &Path) -> Result<String, String> {
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    Ok(format!(
        "fnv1a64:{:016x}",
        hash_tree_into(FNV_OFFSET, root, path)?
    ))
}

fn file_content_hash(path: &Path) -> Result<String, String> {
    Ok(format!(
        "fnv1a64:{:016x}",
        hash_file_into(FNV_OFFSET, path)?
    ))
}

fn resume_config_view(mut config: Value) -> Value {
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

fn value_content_hash(value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    Ok(format!("fnv1a64:{:016x}", hash_bytes(FNV_OFFSET, &bytes)))
}

fn resume_manifest_path(state_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.resume.json", state_path.display()))
}

fn infer_completed_step(path: &Path, config: &Value) -> Result<u64, String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("trainer state filename is not UTF-8")?;
    if let Some(at) = name.rfind("_step") {
        let digits: String = name[at + 5..]
            .chars()
            .take_while(|value| value.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            return digits
                .parse()
                .map_err(|e| format!("parse trainer state step: {e}"));
        }
    }
    let before_tensor = name.split(".safetensors").next().unwrap_or(name);
    if let Some(step) = before_tensor
        .rsplit('_')
        .next()
        .and_then(|value| value.parse::<u64>().ok())
    {
        return Ok(step);
    }
    config["max_steps"]
        .as_u64()
        .ok_or("final trainer state has no max_steps identity".into())
}

fn checkpoint_state_files(workspace: &Path) -> Result<Vec<PathBuf>, String> {
    let mut states: Vec<PathBuf> = fs::read_dir(workspace.join("checkpoints"))
        .map_err(|e| format!("read checkpoint directory: {e}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("");
            path.is_file()
                && (name.ends_with(".state")
                    || name.contains("state") && name.ends_with(".safetensors"))
        })
        .collect();
    states.sort();
    Ok(states)
}

fn write_resume_manifests(
    workspace: &Path,
    config: &Value,
    model: &str,
    sample_count: usize,
    cache_identity: &str,
    trainer: &Path,
) -> Result<(), String> {
    let effective_config_hash = config["effective_config_hash"]
        .as_str()
        .ok_or("run.json effective_config_hash missing")?;
    let trainer_identity = file_content_hash(trainer)?;
    let parent_checkpoint_identity = config["resume_state_identity"].clone();
    let batch_size = config["batch_size"].as_u64().unwrap_or(1);
    let grad_accum_steps = config["grad_accum_steps"]
        .as_u64()
        .or_else(|| config["gradient_accumulation_steps"].as_u64())
        .unwrap_or(1);
    let configured_seed = config["seed"].as_u64().unwrap_or(42);
    let scheduler_kind = config["learning_rate_scheduler"]
        .as_str()
        .unwrap_or("constant");
    let scheduler_warmup = config["learning_rate_warmup_steps"].as_u64().unwrap_or(0);
    let data_scheme = match model {
        "anima" => "fixed-primary-and-optional-partner",
        "chroma" => "step-modulo-sorted-cache-files",
        "krea2" => "trainer-defined-step-derived",
        _ => "unsupported",
    };
    for state_path in checkpoint_state_files(workspace)? {
        let completed_step = infer_completed_step(&state_path, config)?;
        let state_content_hash = file_content_hash(&state_path)?;
        let manifest = json!({
            "schema": "serenity.trainer-resume.v1",
            "model": model,
            "state_filename": state_path.file_name().and_then(|value| value.to_str()),
            "state_content_hash": state_content_hash,
            "completed_step": completed_step,
            "epoch_state": {
                "completed": completed_step,
                "next": completed_step + 1,
            },
            "optimizer_state": {
                "configuration": config["optimizer"],
                "moments": "embedded-in-state-safetensors",
                "completed_step": completed_step,
                "next_step": completed_step + 1,
            },
            "scheduler_state": {
                "kind": scheduler_kind,
                "warmup_steps": scheduler_warmup,
                "completed_step": completed_step,
                "next_step": completed_step + 1,
            },
            "data_ordering_state": {
                "scheme": data_scheme,
                "sample_count": sample_count,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "completed_step": completed_step,
                "next_step": completed_step + 1,
            },
            "random_state": {
                "scheme": "stateless-step-derived-by-trainer-binary",
                "configured_seed": configured_seed,
                "completed_step": completed_step,
                "next_step": completed_step + 1,
            },
            "cache_identity": cache_identity,
            "effective_config_hash": effective_config_hash,
            "trainer_binary_identity": trainer_identity,
            "parent_checkpoint_identity": parent_checkpoint_identity,
            "created_at": now(),
        });
        write_json_atomic(&resume_manifest_path(&state_path), &manifest)?;
    }
    Ok(())
}

fn acquire_cache_lock(cache_dir: &Path) -> Result<CacheLock, String> {
    let path = cache_dir.join(".lock");
    if path.exists() {
        let owner = fs::read_to_string(&path)
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok());
        let owner_alive = owner
            .map(|pid| Path::new("/proc").join(pid.to_string()).exists())
            .unwrap_or(false);
        if owner_alive {
            return Err(format!(
                "cache workspace is locked by live runner pid {} ({})",
                owner.expect("checked owner"),
                path.display()
            ));
        }
        fs::remove_file(&path).map_err(|e| format!("remove stale cache lock: {e}"))?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("cache workspace is locked ({}): {e}", path.display()))?;
    writeln!(file, "{}", std::process::id())
        .map_err(|e| format!("record cache lock owner: {e}"))?;
    Ok(CacheLock(path))
}

fn tensor_inventory(path: &Path) -> Result<Value, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("read cache header: {e}"))?;
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)
        .map_err(|e| format!("read cache header length: {e}"))?;
    let length = u64::from_le_bytes(length);
    if length > 64 * 1024 * 1024 {
        return Err(format!("cache header is unreasonably large: {length}"));
    }
    let mut header = vec![0_u8; length as usize];
    file.read_exact(&mut header)
        .map_err(|e| format!("read cache header: {e}"))?;
    let value: Value =
        serde_json::from_slice(&header).map_err(|e| format!("parse cache tensor header: {e}"))?;
    let tensors = value
        .as_object()
        .ok_or("cache tensor header is not an object")?
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
        .map(|(name, metadata)| {
            json!({
                "name": name,
                "dtype": metadata["dtype"],
                "shape": metadata["shape"],
            })
        })
        .collect::<Vec<_>>();
    Ok(json!(tensors))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    fs::write(
        &tmp,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())
}

fn update_status(workspace: &Path, state: &str, stage: &str, code: Option<&str>, message: &str) {
    let run_id = workspace
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("unknown");
    let status = json!({
        "schema": "serenity.run-status.v1",
        "run_id": run_id,
        "state": state,
        "stage": stage,
        "updated_at": now(),
        "pid": std::process::id(),
        "error_code": code,
        "message": message,
    });
    let _ = write_json_atomic(&workspace.join("status.json"), &status);
}

fn record_command(workspace: &Path, stage: &str, program: &Path, args: &[String]) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(workspace.join("commands.jsonl"))
    {
        let row = json!({"at": now(), "stage": stage, "program": program, "args": args});
        let _ = writeln!(file, "{row}");
    }
}

fn run_stage(
    repo: &Path,
    workspace: &Path,
    stage: &str,
    program: &Path,
    args: &[String],
) -> Result<(), String> {
    record_command(workspace, stage, program, args);
    println!("[stage] {stage}");
    let status = Command::new(program)
        .args(args)
        .current_dir(repo)
        .status()
        .map_err(|e| format!("start {}: {e}", program.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{stage} exited with {status}"))
    }
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let args: Vec<String> = std::env::args().collect();
    let repo_at = args
        .iter()
        .position(|v| v == "--repo")
        .ok_or("--repo required")?;
    let run_at = args
        .iter()
        .position(|v| v == "--run")
        .ok_or("--run required")?;
    let repo = args.get(repo_at + 1).ok_or("--repo value required")?;
    let run = args.get(run_at + 1).ok_or("--run value required")?;
    Ok((PathBuf::from(repo), PathBuf::from(run)))
}

fn sample_count(stage_dir: &Path) -> Result<usize, String> {
    let count = fs::read_dir(stage_dir)
        .map_err(|e| format!("read stage {}: {e}", stage_dir.display()))?
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("sample_") && name.ends_with(".safetensors")
        })
        .count();
    if count == 0 {
        Err("dataset stage produced no image-caption samples".into())
    } else {
        Ok(count)
    }
}

struct FamilyPlan {
    model: String,
    required: Vec<(String, PathBuf)>,
    producers: Vec<PathBuf>,
    cache_data: PathBuf,
    cache_schema: &'static str,
    conditioning: &'static str,
}

fn serenity_models_root() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("SERENITY_MODEL_ROOT") {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .ok_or("HOME is unavailable; set SERENITY_MODEL_ROOT to the installed model registry")?;
    Ok(PathBuf::from(home).join(".serenity/models"))
}

fn resolve_config_path(repo: &Path, config: &Value, key: &str) -> Result<PathBuf, String> {
    let raw = config[key]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("run.json {key} missing"))?;
    let path = PathBuf::from(raw);
    Ok(if let Ok(relative) = path.strip_prefix("serenity-models") {
        serenity_models_root()?.join(relative)
    } else if path.is_absolute() {
        path
    } else {
        repo.join(path)
    })
}

fn family_plan(repo: &Path, config: &Value, cache_prefix: &Path) -> Result<FamilyPlan, String> {
    let model = config["model_type"]
        .as_str()
        .unwrap_or("")
        .to_ascii_lowercase();
    match model.as_str() {
        "krea2" => Ok(FamilyPlan {
            model,
            required: vec![
                (
                    "checkpoint".into(),
                    resolve_config_path(repo, config, "checkpoint")?,
                ),
                ("qwen3-vl-4b".into(), repo.join("models/qwen3-vl-4b")),
                (
                    "qwen-image-vae-encoder".into(),
                    repo.join("models/qwen-image/vae_encoder.safetensors"),
                ),
            ],
            producers: vec![
                repo.join("output/bin/serenity_image_dataset_stage"),
                repo.join("output/bin/serenity_krea2_prepare_cache"),
            ],
            cache_data: cache_prefix.with_extension("safetensors"),
            cache_schema: "serenity.krea-cache.v1",
            conditioning: "qwen3-vl-context-and-qwen-image-vae-latents",
        }),
        "anima" => Ok(FamilyPlan {
            model,
            required: vec![
                (
                    "checkpoint".into(),
                    resolve_config_path(repo, config, "checkpoint")?,
                ),
                (
                    "qwen3-encoder".into(),
                    resolve_config_path(repo, config, "qwen_encoder")?,
                ),
                (
                    "qwen3-tokenizer".into(),
                    resolve_config_path(repo, config, "qwen_tokenizer")?,
                ),
                (
                    "t5-tokenizer".into(),
                    resolve_config_path(repo, config, "t5_tokenizer")?,
                ),
                (
                    "qwen-image-vae-encoder".into(),
                    resolve_config_path(repo, config, "vae_encoder")?,
                ),
            ],
            producers: vec![
                repo.join("output/bin/serenity_image_dataset_stage"),
                repo.join("output/bin/serenity_anima_prepare_cache"),
            ],
            cache_data: cache_prefix.to_path_buf(),
            cache_schema: "serenity.anima-cache.v1",
            conditioning: "anima-qwen3-adapter-context-and-qwen-image-vae-latents",
        }),
        "chroma" => Ok(FamilyPlan {
            model,
            required: vec![
                (
                    "checkpoint".into(),
                    resolve_config_path(repo, config, "checkpoint")?,
                ),
                ("flux-vae".into(), resolve_config_path(repo, config, "vae")?),
                (
                    "t5-xxl".into(),
                    resolve_config_path(repo, config, "t5_encoder")?,
                ),
                (
                    "t5-tokenizer".into(),
                    resolve_config_path(repo, config, "t5_tokenizer")?,
                ),
            ],
            producers: vec![
                repo.join("output/bin/serenity_image_dataset_stage"),
                repo.join("output/bin/serenity_chroma_prepare_cache"),
            ],
            cache_data: cache_prefix.to_path_buf(),
            cache_schema: "serenity.chroma-cache.v1",
            conditioning: "t5-xxl-context-and-flux-vae-latents",
        }),
        _ => Err(format!(
            "model '{model}' is not admitted by the lifecycle runner"
        )),
    }
}

fn cache_inventory(path: &Path) -> Result<Value, String> {
    if path.is_file() {
        return tensor_inventory(path);
    }
    let mut files: Vec<PathBuf> = fs::read_dir(path)
        .map_err(|e| format!("read cache directory {}: {e}", path.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|entry| entry.extension().and_then(|v| v.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    let mut rows = Vec::new();
    for file in files {
        rows.push(json!({
            "file": file.file_name().and_then(|value| value.to_str()),
            "tensors": tensor_inventory(&file)?,
        }));
    }
    if rows.is_empty() {
        return Err(format!(
            "cache directory contains no safetensors: {}",
            path.display()
        ));
    }
    Ok(json!(rows))
}

fn remove_pending(path: &Path) -> Result<(), String> {
    if path.is_dir() {
        fs::remove_dir_all(path).map_err(|e| format!("remove stale pending cache: {e}"))
    } else if path.exists() {
        fs::remove_file(path).map_err(|e| format!("remove stale pending cache: {e}"))
    } else {
        Ok(())
    }
}

fn publish_cache_atomic(pending: &Path, destination: &Path) -> Result<(), String> {
    if !destination.exists() {
        return fs::rename(pending, destination).map_err(|e| format!("publish encoded cache: {e}"));
    }

    let backup = destination.with_extension(format!("previous-{}", std::process::id()));
    remove_pending(&backup)?;
    fs::rename(destination, &backup)
        .map_err(|e| format!("preserve previous encoded cache: {e}"))?;
    match fs::rename(pending, destination) {
        Ok(()) => remove_pending(&backup)
            .map_err(|e| format!("published cache but could not retire previous cache: {e}")),
        Err(publish_error) => {
            let restore = fs::rename(&backup, destination);
            match restore {
                Ok(()) => Err(format!(
                    "publish encoded cache: {publish_error}; previous cache restored"
                )),
                Err(restore_error) => Err(format!(
                    "publish encoded cache: {publish_error}; restore previous cache: {restore_error}; preserved at {}",
                    backup.display()
                )),
            }
        }
    }
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    if source.is_file() {
        fs::copy(source, destination).map_err(|e| format!("copy warm cache: {e}"))?;
        return Ok(());
    }
    fs::create_dir_all(destination).map_err(|e| format!("create warm cache directory: {e}"))?;
    let mut entries: Vec<PathBuf> = fs::read_dir(source)
        .map_err(|e| format!("read warm cache {}: {e}", source.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect();
    entries.sort();
    for entry in entries {
        let target = destination.join(entry.file_name().ok_or("cache entry has no filename")?);
        copy_tree(&entry, &target)?;
    }
    Ok(())
}

fn cache_identity(repo: &Path, config: &Value, plan: &FamilyPlan) -> Result<Value, String> {
    let dataset = config["dataset_path"]
        .as_str()
        .map(PathBuf::from)
        .ok_or("run.json dataset_path missing")?;
    let native_libraries = [repo.join("serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so")];

    let mut models = Vec::new();
    for (identity, path) in &plan.required {
        models.push(json!({
            "identity": identity,
            "revision": "installed-content",
            "path": path,
            "weight_hash": content_hash(&path)?,
        }));
    }
    let mut producer_hashes = Vec::new();
    for path in &plan.producers {
        producer_hashes.push(json!({"path": path, "hash": content_hash(&path)?}));
    }
    let mut native_hashes = Vec::new();
    for path in native_libraries {
        native_hashes.push(json!({"path": path, "hash": content_hash(&path)?}));
    }

    let policy = json!({
        "resolution": config["resolution"],
        "aspect_ratio_bucketing": config["aspect_ratio_bucketing"],
        "caption_dropout_probability": config["caption_dropout_probability"],
        "caption_dropout_prob": config["caption_dropout_prob"],
        "text_encoder_layer_skip": config["text_encoder_layer_skip"],
        "conditioning": plan.conditioning,
        "crop": config["crop"],
        "augmentation": config["augmentation"],
    });
    Ok(json!({
        "schema": "serenity.cache-identity.v1",
        "model": plan.model,
        "dataset": {
            "manifest_hash": content_hash(&dataset)?,
            "path": dataset,
        },
        "models_and_encoders": models,
        "policy": policy,
        "producers": producer_hashes,
        "native_libraries": native_hashes,
    }))
}

fn identity_hash(identity: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(identity).map_err(|e| e.to_string())?;
    Ok(format!("fnv1a64:{:016x}", hash_bytes(FNV_OFFSET, &bytes)))
}

fn reuse_compatible_cache(
    workspace: &Path,
    cache_data: &Path,
    identity_hash: &str,
) -> Result<Option<Value>, String> {
    let Some(output_root) = workspace.parent() else {
        return Ok(None);
    };
    let mut candidates: Vec<PathBuf> = fs::read_dir(output_root)
        .map_err(|e| format!("scan cache workspaces: {e}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != workspace)
        .collect();
    candidates.sort();
    candidates.reverse();
    for candidate in candidates {
        let manifest_path = candidate.join("cache/manifest.json");
        let Ok(bytes) = fs::read(&manifest_path) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if manifest["complete"].as_bool() != Some(true)
            || manifest["identity_hash"].as_str() != Some(identity_hash)
        {
            continue;
        }
        let Some(source) = manifest["data"].as_str().map(PathBuf::from) else {
            continue;
        };
        if !source.exists() {
            continue;
        }
        let expected_hash = manifest["content_hash"].as_str().unwrap_or("");
        if expected_hash.is_empty() || content_hash(&source)? != expected_hash {
            continue;
        }
        let pending = cache_data.with_extension("pending");
        remove_pending(&pending)?;
        copy_tree(&source, &pending)?;
        publish_cache_atomic(&pending, cache_data)
            .map_err(|e| format!("publish warm cache: {e}"))?;
        return Ok(Some(json!({
            "run_id": candidate.file_name().and_then(|v| v.to_str()),
            "manifest": manifest_path,
            "content_hash": expected_hash,
            "sample_count": manifest["sample_count"],
        })));
    }
    Ok(None)
}

fn reuse_current_cache(
    workspace: &Path,
    cache_data: &Path,
    identity_hash: &str,
) -> Result<Option<Value>, String> {
    let manifest_path = workspace.join("cache/manifest.json");
    let Ok(bytes) = fs::read(&manifest_path) else {
        return Ok(None);
    };
    let Ok(manifest) = serde_json::from_slice::<Value>(&bytes) else {
        return Ok(None);
    };
    if manifest["complete"].as_bool() != Some(true)
        || manifest["identity_hash"].as_str() != Some(identity_hash)
        || !cache_data.exists()
    {
        return Ok(None);
    }
    let expected_hash = manifest["content_hash"].as_str().unwrap_or("");
    if expected_hash.is_empty() || content_hash(cache_data)? != expected_hash {
        return Ok(None);
    }
    Ok(Some(json!({
        "run_id": workspace.file_name().and_then(|value| value.to_str()),
        "manifest": manifest_path,
        "content_hash": expected_hash,
        "sample_count": manifest["sample_count"],
        "same_workspace": true,
    })))
}

fn execute(repo: &Path, run_path: &Path) -> Result<(), String> {
    let mut config: Value = serde_json::from_slice(&fs::read(run_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let actual_config_hash = value_content_hash(&resume_config_view(config.clone()))?;
    if let Some(expected_config_hash) = config["effective_config_hash"].as_str() {
        if expected_config_hash != actual_config_hash {
            return Err(
                "run.json effective configuration identity does not match its contents".into(),
            );
        }
    }
    config["effective_config_hash"] = json!(actual_config_hash.clone());
    let resume_manifest = if let Some(resume) = config["resume_state"].as_str() {
        let resume_path = PathBuf::from(resume);
        let actual_state_identity = file_content_hash(&resume_path)?;
        if config["resume_state_identity"]
            .as_str()
            .is_some_and(|expected| expected != actual_state_identity)
        {
            return Err("resume state content changed after admission".into());
        }
        let manifest_path = config["resume_manifest"]
            .as_str()
            .map(PathBuf::from)
            .unwrap_or_else(|| resume_manifest_path(&resume_path));
        let manifest: Value = serde_json::from_slice(
            &fs::read(&manifest_path).map_err(|e| format!("read resume manifest: {e}"))?,
        )
        .map_err(|e| format!("parse resume manifest: {e}"))?;
        if manifest["schema"] != json!("serenity.trainer-resume.v1")
            || manifest["state_content_hash"].as_str() != Some(actual_state_identity.as_str())
            || manifest["effective_config_hash"].as_str() != Some(actual_config_hash.as_str())
        {
            return Err("resume manifest identity changed after admission".into());
        }
        let completed_step = manifest["completed_step"]
            .as_u64()
            .ok_or("resume manifest completed_step missing")?;
        let max_steps = config["max_steps"]
            .as_u64()
            .ok_or("run.json max_steps missing")?;
        if completed_step == 0 || completed_step >= max_steps {
            return Err("resume state completed step must be below max_steps".into());
        }
        config["resume_state_identity"] = json!(actual_state_identity.clone());
        config["resume_manifest"] = json!(manifest_path);
        config["start_step"] = json!(completed_step);
        config["resume_parent_cache_identity"] = manifest["cache_identity"].clone();
        config["resume_parent_checkpoint_identity"] = json!(actual_state_identity);
        Some(manifest)
    } else {
        None
    };
    let workspace = run_path.parent().ok_or("run.json has no workspace")?;
    write_json_atomic(
        &workspace.join("admission.json"),
        &json!({
            "schema": "serenity.runner-admission.v1",
            "effective_config_hash": config["effective_config_hash"],
            "resume_state": config["resume_state"],
            "resume_state_identity": config["resume_state_identity"],
            "resume_manifest": config["resume_manifest"],
            "start_step": config["start_step"],
            "resume_parent_cache_identity": config["resume_parent_cache_identity"],
            "resume_parent_checkpoint_identity": config["resume_parent_checkpoint_identity"],
            "admitted_at": now(),
        }),
    )?;
    let cache_prefix = config["cache_dir"]
        .as_str()
        .map(PathBuf::from)
        .ok_or("run.json cache_dir missing")?;
    let stage_dir = PathBuf::from(format!("{}_stage", cache_prefix.display()));
    let plan = family_plan(repo, &config, &cache_prefix)?;
    let cache_data = plan.cache_data.clone();
    let trainer = match plan.model.as_str() {
        "krea2" => repo.join("output/bin/serenity_krea2_live_trainer"),
        "anima" => repo.join("output/bin/serenity_anima_live_trainer"),
        "chroma" => repo.join("output/bin/serenity_chroma_live_trainer"),
        _ => unreachable!(),
    };
    if let Some(manifest) = resume_manifest.as_ref() {
        if manifest["trainer_binary_identity"].as_str()
            != Some(file_content_hash(&trainer)?.as_str())
        {
            return Err("resume state was produced by a different trainer binary".into());
        }
    }
    let steps = config["max_steps"].as_u64().unwrap_or(2000).to_string();
    let resolution = config["resolution"]
        .as_str()
        .and_then(|v| v.split(',').next())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(512)
        .to_string();

    update_status(
        workspace,
        "validating",
        "validating",
        None,
        &format!("validating {} run inputs", plan.model),
    );
    for (identity, required) in &plan.required {
        if !required.exists() {
            return Err(format!(
                "required {identity} model asset is missing: {}",
                required.display()
            ));
        }
    }
    for producer in &plan.producers {
        if !producer.is_file() {
            return Err(format!(
                "required cache producer is missing: {} (run the repository setup/build task)",
                producer.display()
            ));
        }
    }

    let _cache_lock = acquire_cache_lock(&workspace.join("cache"))?;
    let identity = cache_identity(repo, &config, &plan)?;
    let identity_hash = identity_hash(&identity)?;
    if let Some(parent_identity) = config["resume_parent_cache_identity"].as_str() {
        if parent_identity != identity_hash {
            return Err("resume parent cache identity does not match the current dataset, assets, and cache policy".into());
        }
    }
    if let Some(manifest) = resume_manifest.as_ref() {
        if manifest["cache_identity"].as_str() != Some(identity_hash.as_str()) {
            return Err("resume manifest cache identity does not match the current run".into());
        }
    }
    let reused_from = match reuse_current_cache(workspace, &cache_data, &identity_hash)? {
        Some(current) => Some(current),
        None => reuse_compatible_cache(workspace, &cache_data, &identity_hash)?,
    };

    let count;
    if let Some(reuse) = &reused_from {
        update_status(
            workspace,
            "cache-ready",
            "warm-cache",
            None,
            "reused a compatible encoded cache by content identity",
        );
        count = reuse["sample_count"]
            .as_u64()
            .ok_or("compatible cache manifest is missing sample_count")? as usize;
    } else {
        update_status(
            workspace,
            "caching",
            "dataset-stage",
            None,
            "decoding and bucketing the dataset",
        );
        run_stage(
            repo,
            workspace,
            "dataset-stage",
            &repo.join("output/bin/serenity_image_dataset_stage"),
            &[run_path.display().to_string()],
        )?;
        count = sample_count(&stage_dir)?;

        update_status(
            workspace,
            "caching",
            "encoder-cache",
            None,
            "encoding the trainer cache",
        );
        let pending = cache_data.with_extension("pending");
        remove_pending(&pending)?;
        match plan.model.as_str() {
            "krea2" => run_stage(
                repo,
                workspace,
                "encoder-cache",
                &repo.join("output/bin/serenity_krea2_prepare_cache"),
                &[
                    stage_dir.display().to_string(),
                    pending.display().to_string(),
                    count.to_string(),
                    resolution.clone(),
                ],
            )?,
            "anima" => run_stage(
                repo,
                workspace,
                "encoder-cache",
                &repo.join("output/bin/serenity_anima_prepare_cache"),
                &[
                    stage_dir.display().to_string(),
                    pending.display().to_string(),
                    count.to_string(),
                    resolve_config_path(repo, &config, "checkpoint")?
                        .display()
                        .to_string(),
                    resolve_config_path(repo, &config, "qwen_encoder")?
                        .display()
                        .to_string(),
                    resolve_config_path(repo, &config, "qwen_tokenizer")?
                        .display()
                        .to_string(),
                    resolve_config_path(repo, &config, "t5_tokenizer")?
                        .display()
                        .to_string(),
                    resolve_config_path(repo, &config, "vae_encoder")?
                        .display()
                        .to_string(),
                ],
            )?,
            "chroma" => run_stage(
                repo,
                workspace,
                "encoder-cache",
                &repo.join("output/bin/serenity_chroma_prepare_cache"),
                &[
                    stage_dir.display().to_string(),
                    pending.display().to_string(),
                    count.to_string(),
                    resolve_config_path(repo, &config, "vae")?
                        .display()
                        .to_string(),
                    resolve_config_path(repo, &config, "t5_encoder")?
                        .display()
                        .to_string(),
                    resolve_config_path(repo, &config, "t5_tokenizer")?
                        .display()
                        .to_string(),
                ],
            )?,
            _ => unreachable!(),
        }
        publish_cache_atomic(&pending, &cache_data)
            .map_err(|e| format!("atomically publish encoded cache: {e}"))?;
    }

    let tensors = cache_inventory(&cache_data)?;
    let cache_content_hash = content_hash(&cache_data)?;
    write_json_atomic(
        &workspace.join("cache/manifest.json"),
        &json!({
            "schema": plan.cache_schema,
            "model": plan.model,
            "complete": true,
            "identity_hash": identity_hash,
            "identity": identity,
            "sample_count": count,
            "data": cache_data,
            "content_hash": cache_content_hash,
            "tensors": tensors,
            "reused_from": reused_from,
            "created_at": now(),
        }),
    )?;

    if config["only_cache"].as_bool().unwrap_or(false) {
        update_status(
            workspace,
            "completed",
            "cache-ready",
            None,
            "cache-only run completed",
        );
        return Ok(());
    }

    update_status(
        workspace,
        "training",
        "training",
        None,
        &format!("{} training started", plan.model),
    );
    let mut args = match plan.model.as_str() {
        "krea2" => {
            vec![
                cache_data.display().to_string(),
                steps,
                run_path.display().to_string(),
            ]
        }
        "anima" => {
            let mut anima_args = vec![run_path.display().to_string(), steps];
            if let Some(resume) = config["resume_state"].as_str() {
                anima_args.push(resume.to_string());
                anima_args.push(config["start_step"].as_u64().unwrap_or(0).to_string());
            }
            anima_args
        }
        "chroma" => {
            let mut chroma_args = vec![run_path.display().to_string(), steps];
            if let Some(resume) = config["resume_state"].as_str() {
                chroma_args.push(resume.to_string());
                chroma_args.push(config["start_step"].as_u64().unwrap_or(0).to_string());
            }
            chroma_args
        }
        _ => unreachable!(),
    };
    if plan.model == "krea2" {
        if let Some(resume) = config["resume_state"].as_str() {
            args.push(resume.to_string());
            args.push(config["start_step"].as_u64().unwrap_or(0).to_string());
        }
    }
    run_stage(repo, workspace, "training", &trainer, &args)?;
    write_resume_manifests(
        workspace,
        &config,
        &plan.model,
        count,
        &identity_hash,
        &trainer,
    )?;
    update_status(
        workspace,
        "completed",
        "completed",
        None,
        "training completed",
    );
    Ok(())
}

fn main() -> ExitCode {
    let (repo, run_path) = match parse_args() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("[failed] {error}");
            return ExitCode::FAILURE;
        }
    };
    let workspace = run_path.parent().unwrap_or(Path::new("."));
    match execute(&repo, &run_path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("[failed] {error}");
            update_status(
                workspace,
                "failed",
                "failed",
                Some("RUN_STAGE_FAILED"),
                &error,
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_cache(path: &Path) {
        let header = serde_json::to_vec(&json!({
            "context.0": {
                "dtype": "BF16",
                "shape": [1, 2, 3],
                "data_offsets": [0, 12]
            }
        }))
        .expect("serialize test cache header");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0_u8; 12]);
        fs::write(path, bytes).expect("write test cache");
    }

    #[test]
    fn missing_model_fails_before_cache_is_required() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let scratch = std::env::temp_dir().join(format!(
            "serenity-runner-test-{}-{stamp}",
            std::process::id()
        ));
        let repo = scratch.join("repo");
        let workspace = scratch.join("output/runs/test");
        fs::create_dir_all(&repo).expect("create fake repo");
        fs::create_dir_all(workspace.join("cache")).expect("create workspace");
        let run_path = workspace.join("run.json");
        let config = json!({
            "model_type": "krea2",
            "checkpoint": "models/krea2/raw.safetensors",
            "cache_dir": workspace.join("cache/data"),
            "max_steps": 1,
            "resolution": "512",
            "only_cache": false
        });
        fs::write(
            &run_path,
            serde_json::to_vec(&config).expect("serialize run"),
        )
        .expect("write run");
        let error = execute(&repo, &run_path).expect_err("missing model must fail");
        assert!(error.contains("model asset is missing"));
        assert!(!error.contains("cache_dir missing"));
        let admission: Value = serde_json::from_slice(
            &fs::read(workspace.join("admission.json")).expect("read runner admission"),
        )
        .expect("parse runner admission");
        assert_eq!(admission["schema"], json!("serenity.runner-admission.v1"));
        assert!(admission["effective_config_hash"]
            .as_str()
            .is_some_and(|value| value.starts_with("fnv1a64:")));
        fs::remove_dir_all(scratch).expect("remove temporary test directory");
    }

    #[test]
    fn compatible_cache_is_verified_and_copied_into_new_workspace() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let output = std::env::temp_dir().join(format!(
            "serenity-cache-reuse-test-{}-{stamp}",
            std::process::id()
        ));
        let old_workspace = output.join("old-run");
        let new_workspace = output.join("new-run");
        fs::create_dir_all(old_workspace.join("cache")).expect("create old cache directory");
        fs::create_dir_all(new_workspace.join("cache")).expect("create new cache directory");
        let source = old_workspace.join("cache/data.safetensors");
        let destination = new_workspace.join("cache/data.safetensors");
        write_test_cache(&source);
        let source_hash = content_hash(&source).expect("hash source cache");
        write_json_atomic(
            &old_workspace.join("cache/manifest.json"),
            &json!({
                "schema": "serenity.krea-cache.v1",
                "complete": true,
                "identity_hash": "fnv1a64:test-identity",
                "content_hash": source_hash,
                "data": source,
            }),
        )
        .expect("write old manifest");
        let current = reuse_current_cache(&old_workspace, &source, "fnv1a64:test-identity")
            .expect("reuse current cache");
        assert_eq!(current.unwrap()["same_workspace"], json!(true));

        let _lock = acquire_cache_lock(&new_workspace.join("cache")).expect("lock new cache");
        let reused = reuse_compatible_cache(&new_workspace, &destination, "fnv1a64:test-identity")
            .expect("scan compatible cache");
        assert!(reused.is_some());
        assert_eq!(
            content_hash(&destination).expect("hash copied cache"),
            content_hash(&old_workspace.join("cache/data.safetensors"))
                .expect("hash source cache again")
        );
        let tensors = tensor_inventory(&destination).expect("read tensor inventory");
        assert_eq!(tensors[0]["name"], json!("context.0"));
        assert_eq!(tensors[0]["dtype"], json!("BF16"));
        fs::remove_dir_all(output).expect("remove temporary test directory");
    }

    #[test]
    fn atomic_cache_publish_replaces_a_nonempty_directory() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let scratch = std::env::temp_dir().join(format!(
            "serenity-cache-publish-test-{}-{stamp}",
            std::process::id()
        ));
        let destination = scratch.join("data");
        let pending = scratch.join("data.pending");
        fs::create_dir_all(&destination).expect("create current cache");
        fs::create_dir_all(&pending).expect("create pending cache");
        fs::write(destination.join("old.safetensors"), b"old").expect("write old cache");
        fs::write(pending.join("new.safetensors"), b"new").expect("write pending cache");

        publish_cache_atomic(&pending, &destination).expect("publish replacement cache");

        assert!(!pending.exists());
        assert!(!destination.join("old.safetensors").exists());
        assert_eq!(
            fs::read(destination.join("new.safetensors")).expect("read new cache"),
            b"new"
        );
        fs::remove_dir_all(scratch).expect("remove temporary test directory");
    }

    #[test]
    fn stale_cache_lock_is_recovered_after_interrupted_runner() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let cache = std::env::temp_dir().join(format!(
            "serenity-stale-lock-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&cache).expect("create cache directory");
        fs::write(cache.join(".lock"), "4294967295\n").expect("write stale lock");
        let lock = acquire_cache_lock(&cache).expect("recover stale lock");
        assert!(cache.join(".lock").is_file());
        drop(lock);
        assert!(!cache.join(".lock").exists());
        fs::remove_dir_all(cache).expect("remove cache directory");
    }

    #[test]
    fn resume_bundle_binds_state_and_all_continuation_identities() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let scratch = std::env::temp_dir().join(format!(
            "serenity-resume-bundle-test-{}-{stamp}",
            std::process::id()
        ));
        let workspace = scratch.join("output/test-run");
        let checkpoints = workspace.join("checkpoints");
        fs::create_dir_all(&checkpoints).expect("create checkpoints");
        let state = checkpoints.join("test-run.safetensors.state.safetensors");
        let trainer = scratch.join("trainer-bin");
        fs::write(&state, b"state-with-weights-and-moments").expect("write state");
        fs::write(&trainer, b"trainer-identity").expect("write trainer");
        let mut config = json!({
            "model_type": "anima",
            "max_steps": 2,
            "batch_size": 1,
            "learning_rate_scheduler": "constant",
            "learning_rate_warmup_steps": 0,
            "optimizer": {"beta1": 0.9, "beta2": 0.999},
        });
        let config_hash =
            value_content_hash(&resume_config_view(config.clone())).expect("hash effective config");
        config["effective_config_hash"] = json!(config_hash);
        write_resume_manifests(&workspace, &config, "anima", 2, "fnv1a64:cache", &trainer)
            .expect("write resume manifest");
        let manifest: Value = serde_json::from_slice(
            &fs::read(resume_manifest_path(&state)).expect("read resume manifest"),
        )
        .expect("parse resume manifest");
        assert_eq!(manifest["schema"], json!("serenity.trainer-resume.v1"));
        assert_eq!(manifest["completed_step"], json!(2));
        assert_eq!(manifest["cache_identity"], json!("fnv1a64:cache"));
        assert_eq!(manifest["effective_config_hash"], json!(config_hash));
        assert_eq!(
            manifest["state_content_hash"],
            json!(file_content_hash(&state).expect("hash state"))
        );
        assert!(manifest["optimizer_state"].is_object());
        assert!(manifest["scheduler_state"].is_object());
        assert!(manifest["data_ordering_state"].is_object());
        assert!(manifest["random_state"].is_object());
        assert_eq!(
            manifest["data_ordering_state"]["gradient_accumulation_steps"],
            json!(1)
        );
        assert_eq!(manifest["random_state"]["configured_seed"], json!(42));
        fs::write(&state, b"tampered").expect("tamper state");
        assert_ne!(
            manifest["state_content_hash"],
            json!(file_content_hash(&state).expect("hash tampered state"))
        );
        fs::remove_dir_all(scratch).expect("remove temporary test directory");
    }
}
