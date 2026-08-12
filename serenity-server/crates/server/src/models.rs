//! GET /v1/models — faithful port of the daemon's model/LoRA browser endpoint
//! (serenity_daemon.mojo @2923 + serve/model_scan.mojo + the card builders @2646-2904).
//!
//! Scans the configured Serenity model registry (header reads
//! only, never weights) for arch tags, plus a few known diffusers-tree dirs, and emits
//! `serenity.models.v1` card JSON. Verified byte-identical vs `serenity_daemon stub`.
//!
//! Fidelity notes:
//! - File sizes = st_size (find -printf %s == fs::metadata().len(), symlinks followed).
//! - Known-DIR sizes = `du -sb` (shelled out, identical to the daemon) — re-implementing
//!   du's apparent-size semantics natively is not worth the byte-exact risk.
//! - Output order is fully re-sorted by `scan_entry_cmp`, so scan order is irrelevant.
//! - JSON key order matches the daemon's insertion order via json!{} literal order;
//!   serde_json `preserve_order` (workspace-unified) keeps it on serialize.
//! - resident="" here (matches the stub oracle; the Rust server tracks no resident model
//!   name yet) → `loaded` is always false and selected_model defaults to "".

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use axum::extract::{Json, Path as AxPath, Query};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const LTX2_FEATURE_ADAPTERS_JSON: &str =
    include_str!("../../../../serenitymojo/configs/ltx2_feature_adapters.json");

fn model_root() -> PathBuf {
    std::env::var_os("SERENITY_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".serenity/models")
        })
}

type ExtraModelRoots = std::sync::RwLock<Vec<PathBuf>>;

fn extra_model_roots_store() -> &'static ExtraModelRoots {
    static ROOTS: std::sync::OnceLock<ExtraModelRoots> = std::sync::OnceLock::new();
    ROOTS.get_or_init(|| {
        let roots = std::env::var_os("SERENITY_EXTRA_MODEL_ROOTS")
            .map(|value| std::env::split_paths(&value).collect())
            .unwrap_or_default();
        std::sync::RwLock::new(normalize_extra_model_roots(roots))
    })
}

fn normalize_extra_model_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    let primary = std::fs::canonicalize(model_root()).ok();
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for root in roots {
        let Ok(root) = std::fs::canonicalize(root) else {
            continue;
        };
        if !root.is_dir()
            || primary.as_ref().is_some_and(|primary| primary == &root)
            || !seen.insert(root.clone())
        {
            continue;
        }
        normalized.push(root);
    }
    normalized
}

fn extra_model_roots() -> Vec<PathBuf> {
    extra_model_roots_store()
        .read()
        .map(|roots| roots.clone())
        .unwrap_or_default()
}

/// Replace the live external registry roots after Settings changes. The model
/// cache is invalidated immediately so "Add Directory" is a real scanner
/// contract, not a cosmetic list that takes up to a minute to become visible.
pub(crate) fn set_extra_model_roots(roots: Vec<PathBuf>) {
    if let Ok(mut current) = extra_model_roots_store().write() {
        *current = normalize_extra_model_roots(roots);
    }
    invalidate_checkpoint_scan_cache();
}
const HEADER_PROBE_CAP: u64 = 16 * 1024 * 1024;
/// Cap on an inlined sidecar preview image (encoded as a `data:` URI). Sidecar
/// previews are meant to be small thumbnails; oversize files are skipped rather
/// than bloating the /v1/models JSON. 2 MiB raw → ~2.7 MiB base64.
const PREVIEW_INLINE_CAP: u64 = 2 * 1024 * 1024;
const MODEL_TYPE_OVERRIDES_FILENAME: &str = "model_type_overrides.json";
const MODEL_TYPE_OVERRIDES_SCHEMA: &str = "serenity.model_type_overrides.v1";
const MODEL_TYPE_OPTIONS: &[(&str, &str)] = &[
    ("sdxl", "SDXL / Pony / Illustrious"),
    ("krea2", "Krea 2"),
    ("zimage", "Z-Image"),
    ("qwen-image", "Qwen Image"),
    ("sd3", "Stable Diffusion 3 / 3.5"),
    ("flux", "FLUX.1"),
    ("flux-2/klein", "FLUX.2 / Klein"),
    ("chroma", "Chroma"),
    ("anima", "Anima"),
    ("ideogram4", "Ideogram 4"),
    ("sensenova", "SenseNova"),
    ("lens", "Microsoft Lens"),
    ("ltx2", "LTX 2 / 2.3 video"),
    ("minimax-h3", "MiniMax-H3 audio/video"),
    ("wan2.2", "Wan 2.2 video"),
    ("nava", "NAVA audio/video"),
    ("bernini", "Bernini video"),
    ("scail2", "SCAIL-2 video"),
];
const REGISTRY_ARTIFACT_ROOTS: &[(&str, &str)] = &[
    ("vae", "vaes"),
    ("controlnet", "controlnets"),
    ("controlnet", "model_patches"),
    ("embedding", "Embeddings"),
    ("embedding", "embeddings"),
    ("clip", "text_encoders"),
    ("clip", "clip"),
    ("clip_vision", "clip_vision"),
    ("ipadapter", "ipadapters"),
    ("upscaler", "upscale_models"),
    ("upscaler", "latent_upscale_models"),
    ("upscaler", "upscalers"),
    ("upscaler", "ltx2_upscalers"),
    ("upscaler", "pid/checkpoints"),
    ("runtime_component", "style_models"),
    ("runtime_component", "sam3"),
    ("vae", "lance"),
];

/// Sidecar metadata distilled from a `<model>.json` / `.civitai.info` next to a
/// checkpoint or LoRA. All fields are "" when absent. ADD-only on the wire.
#[derive(Default, Clone)]
struct Sidecar {
    /// data: URI for an adjacent preview image ("" if none / too large).
    preview: String,
    /// human description (Civitai `description` / generic `description`/`notes`).
    description: String,
    /// trigger / activation words (Civitai `trainedWords` joined, or `trigger`).
    trigger: String,
    /// base-model / arch hint from the sidecar (only used to fill `unknown`).
    arch_hint: String,
}

#[derive(Clone)]
struct ScanEntry {
    name: String,
    path: String,
    arch: String,
    /// Architecture found without a user override.
    detected_arch: String,
    /// `user_override`, `metadata`, `tensor_signature`, `sidecar`,
    /// `filename`, `bundled_identity`, or `unknown`.
    arch_source: String,
    /// Stable architecture id saved by the user, or "" when automatic.
    arch_override: String,
    format: String,
    size: i64,
    /// Subdir of the entry RELATIVE to its scan root ("" = top level). Lets the
    /// browser show a folder tree without guessing the root from the abs path.
    folder: String,
    /// Distilled sidecar preview/metadata (empty `Sidecar::default()` if none).
    sidecar: Sidecar,
}

#[derive(Clone)]
struct RegistryArtifact {
    name: String,
    path: String,
    folder: String,
    artifact_type: String,
    size: i64,
}

#[derive(Default, Deserialize, Serialize)]
struct ModelTypeOverrides {
    schema: String,
    #[serde(default)]
    models: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelTypeOverrideRequest {
    model: String,
    #[serde(default)]
    arch: Option<String>,
    #[serde(default)]
    kind: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedCheckpoint {
    pub name: String,
    pub path: PathBuf,
    pub arch: String,
    pub arch_source: String,
    pub arch_override: String,
    pub format: String,
}

fn ltx2_feature_adapter_registry() -> &'static Value {
    static REGISTRY: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry: Value = serde_json::from_str(LTX2_FEATURE_ADAPTERS_JSON)
            .expect("embedded LTX2 feature adapter registry must be valid JSON");
        assert_eq!(
            registry.get("schema").and_then(Value::as_str),
            Some("serenity.ltx2.feature_adapters.v1"),
            "embedded LTX2 feature adapter registry schema mismatch"
        );
        assert!(
            registry.get("adapters").and_then(Value::as_array).is_some(),
            "embedded LTX2 feature adapter registry requires adapters"
        );
        registry
    })
}

fn normalized_lora_filename(name: &str) -> String {
    let basename = Path::new(name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(name);
    if basename.ends_with(".safetensors") {
        basename.to_string()
    } else {
        format!("{basename}.safetensors")
    }
}

fn ltx2_feature_adapter(name: &str) -> Option<&'static Value> {
    let filename = normalized_lora_filename(name);
    ltx2_feature_adapter_registry()
        .get("adapters")
        .and_then(Value::as_array)
        .and_then(|adapters| {
            adapters.iter().find(|adapter| {
                adapter.get("filename").and_then(Value::as_str) == Some(filename.as_str())
            })
        })
}

/// Resolve one feature-adapter document by its stable product id. The embedded
/// registry remains the single source of truth for both readiness and request
/// normalization.
pub fn ltx2_feature_document(id: &str) -> Option<Value> {
    ltx2_feature_adapter_registry()
        .get("adapters")
        .and_then(Value::as_array)
        .and_then(|adapters| {
            adapters
                .iter()
                .find(|adapter| adapter.get("id").and_then(Value::as_str) == Some(id))
        })
        .cloned()
}

/// Classify an artifact stored under the LoRA root. IC-LoRA feature adapters
/// and companion embeddings require dedicated conditioning paths and must
/// never be submitted through the ordinary trained-LoRA overlay route.
pub fn lora_usage(name: &str) -> String {
    if let Some(usage) = ltx2_feature_adapter(name)
        .and_then(|adapter| adapter.get("usage"))
        .and_then(Value::as_str)
    {
        return usage.to_string();
    }
    let filename = normalized_lora_filename(name).to_lowercase();
    if filename.contains("ic-lora") || filename.contains("ic_lora") {
        "ic_lora_feature".to_string()
    } else {
        "overlay".to_string()
    }
}

fn lora_selectable_as_overlay(name: &str) -> bool {
    lora_usage(name) == "overlay"
}

pub fn ltx2_feature_documents() -> Value {
    let lora_root = model_root().join("loras");
    let rows = ltx2_feature_adapter_registry()
        .get("adapters")
        .and_then(Value::as_array)
        .map(|adapters| {
            adapters
                .iter()
                .map(|adapter| {
                    let mut doc = adapter.clone();
                    let filename = adapter
                        .get("filename")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let path = lora_root.join(filename);
                    if let Some(object) = doc.as_object_mut() {
                        object.insert("installed".to_string(), json!(path.is_file()));
                        object.insert(
                            "path".to_string(),
                            json!(path.to_string_lossy().into_owned()),
                        );
                        object.insert(
                            "selectable_as_lora".to_string(),
                            json!(lora_selectable_as_overlay(filename)),
                        );
                    }
                    doc
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Value::Array(rows)
}

// ── architecture + artifact-format detection ──────────────────────────────────

/// Collapse ModelSpec/Civitai/Comfy spellings to the stable IDs used by the
/// Serenity capability registry. This is deliberately architecture-based: a
/// creator filename is presentation metadata, never runtime routing authority.
fn normalize_architecture_id(raw: &str) -> String {
    let value = raw.trim().to_ascii_lowercase();
    if value.is_empty() {
        return "unknown".to_string();
    }
    if value.contains("stable-diffusion-xl")
        || value.contains("stable diffusion xl")
        || value == "sdxl"
        || value.starts_with("pony")
        || value.starts_with("illustrious")
    {
        "sdxl".to_string()
    } else if value.contains("krea 2") || value.contains("krea-2") || value.contains("krea2") {
        "krea2".to_string()
    } else if value.contains("flux.2") || value.contains("flux-2") || value.contains("klein") {
        "flux-2/klein".to_string()
    } else if value.contains("flux") {
        "flux".to_string()
    } else if value.contains("z-image") || value.contains("zimage") || value.contains("z_image") {
        "zimage".to_string()
    } else if value.contains("qwen") && value.contains("image") {
        "qwen-image".to_string()
    } else if value.contains("stable-diffusion-3")
        || value.contains("stable-diffusion-v3")
        || value.contains("stable diffusion 3")
        || value == "sd3"
        || value.starts_with("sd3.")
    {
        "sd3".to_string()
    } else if value.contains("lens") {
        "lens".to_string()
    } else if value.contains("minimax") && value.contains("h3") {
        "minimax-h3".to_string()
    } else if value.contains("nava") {
        "nava".to_string()
    } else if value.contains("ltx") {
        "ltx2".to_string()
    } else if value.contains("wan") {
        "wan2.2".to_string()
    } else if value.contains("ideogram") {
        "ideogram4".to_string()
    } else if value.contains("chroma") {
        "chroma".to_string()
    } else if value.contains("anima") {
        "anima".to_string()
    } else if value.contains("sensenova") || value.contains("sense-nova") {
        "sensenova".to_string()
    } else {
        value
    }
}

fn normalize_user_model_type(raw: &str) -> Option<String> {
    let normalized = normalize_architecture_id(raw);
    MODEL_TYPE_OPTIONS
        .iter()
        .any(|(id, _)| *id == normalized)
        .then_some(normalized)
}

fn architecture_has_selected_checkpoint_loader(arch: &str) -> bool {
    matches!(
        arch,
        "sdxl" | "krea2" | "zimage" | "qwen-image" | "sd3" | "flux" | "chroma" | "anima" | "ltx2"
    )
}

fn selected_checkpoint_scope(
    arch: &str,
    format: &str,
    name: &str,
    arch_source: &str,
    arch_override: &str,
) -> Option<&'static str> {
    match (arch, format) {
        ("sdxl", "diffusion_model" | "full_checkpoint") => Some("denoiser"),
        ("krea2", "diffusion_model") => Some("denoiser"),
        ("zimage", "diffusion_model") => Some("denoiser"),
        ("qwen-image", "diffusion_model") if !name.to_ascii_lowercase().contains("edit") => {
            Some("denoiser")
        }
        ("sd3", "sd3_large_full_checkpoint") => Some("full_checkpoint"),
        ("flux", "diffusion_model") => Some("denoiser"),
        ("chroma", "diffusion_model") => Some("denoiser"),
        ("anima", "diffusion_model") => Some("denoiser"),
        ("ltx2", "diffusion_model" | "full_checkpoint") => {
            let stem = name.strip_suffix(".safetensors").unwrap_or(name);
            let known_ltx23_single_file = matches!(
                stem,
                "ltx-2.3-22b-dev-fp8"
                    | "ltx-2.3-22b-dev-fp8-dequant-bf16"
                    | "ltx-2.3-22b-distilled-fp8"
                    | "ltx-2.3-22b-distilled-fp8-dequant-bf16"
            );
            (arch_source != "filename" || !arch_override.is_empty() || known_ltx23_single_file)
                .then_some("video_denoiser")
        }
        _ => None,
    }
}

fn entry_selected_checkpoint_scope(e: &ScanEntry) -> Option<&'static str> {
    selected_checkpoint_scope(
        &entry_arch(e),
        &e.format,
        &e.name,
        &e.arch_source,
        &e.arch_override,
    )
}

pub(crate) fn selected_checkpoint_scope_for_resolved(
    checkpoint: &ResolvedCheckpoint,
) -> Option<&'static str> {
    selected_checkpoint_scope(
        &checkpoint.arch,
        &checkpoint.format,
        &checkpoint.name,
        &checkpoint.arch_source,
        &checkpoint.arch_override,
    )
}

fn model_type_options_json() -> Value {
    Value::Array(
        MODEL_TYPE_OPTIONS
            .iter()
            .map(|(id, label)| {
                json!({
                    "id": id,
                    "label": label,
                    "route": if matches!(*id, "ltx2" | "wan2.2" | "bernini" | "scail2") {
                        "video"
                    } else if crate::capabilities::model_family_for_arch(id).is_some() {
                        "image"
                    } else {
                        "unsupported"
                    },
                    "arbitrary_checkpoint_supported": architecture_has_selected_checkpoint_loader(id),
                })
            })
            .collect(),
    )
}

fn model_type_overrides_path(root: &Path) -> PathBuf {
    root.join(MODEL_TYPE_OVERRIDES_FILENAME)
}

fn lora_type_override_key(name: &str) -> String {
    format!("lora:{name}")
}

fn load_model_type_overrides_from(root: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = model_type_overrides_path(root);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    let document: ModelTypeOverrides = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    if document.schema != MODEL_TYPE_OVERRIDES_SCHEMA {
        return Err(format!(
            "{} has unsupported schema {:?}",
            path.display(),
            document.schema
        ));
    }
    let mut normalized = BTreeMap::new();
    for (model, arch) in document.models {
        if model.trim().is_empty() {
            return Err(format!(
                "{} contains an empty model identity",
                path.display()
            ));
        }
        let Some(arch) = normalize_user_model_type(&arch) else {
            return Err(format!(
                "{} contains unsupported model type {:?} for {:?}",
                path.display(),
                arch,
                model
            ));
        };
        normalized.insert(model, arch);
    }
    Ok(normalized)
}

fn persist_model_type_overrides_to(
    root: &Path,
    models: &BTreeMap<String, String>,
) -> Result<(), String> {
    let path = model_type_overrides_path(root);
    if models.is_empty() {
        return match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!("remove {}: {error}", path.display())),
        };
    }
    std::fs::create_dir_all(root)
        .map_err(|error| format!("create model registry {}: {error}", root.display()))?;
    let document = ModelTypeOverrides {
        schema: MODEL_TYPE_OVERRIDES_SCHEMA.to_string(),
        models: models.clone(),
    };
    let mut bytes = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("serialize model type overrides: {error}"))?;
    bytes.push(b'\n');
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = root.join(format!(
        ".{MODEL_TYPE_OVERRIDES_FILENAME}.{}.{}.tmp",
        std::process::id(),
        nonce
    ));
    std::fs::write(&temp_path, bytes)
        .map_err(|error| format!("write {}: {error}", temp_path.display()))?;
    if let Err(error) = std::fs::rename(&temp_path, &path) {
        let _ = std::fs::remove_file(&temp_path);
        return Err(format!(
            "replace {} with {}: {error}",
            path.display(),
            temp_path.display()
        ));
    }
    Ok(())
}

fn model_type_override_write_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn metadata_architecture(header: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(header) else {
        return String::new();
    };
    let metadata = value.get("__metadata__").unwrap_or(&Value::Null);
    for key in [
        "modelspec.architecture",
        "architecture",
        "general.architecture",
        "model_type",
    ] {
        if let Some(raw) = metadata
            .get(key)
            .and_then(Value::as_str)
            .or_else(|| value.get(key).and_then(Value::as_str))
        {
            let normalized = normalize_architecture_id(raw);
            if normalized != "unknown" {
                return normalized;
            }
        }
    }
    String::new()
}

fn detect_arch(header: &str) -> String {
    let metadata = metadata_architecture(header);
    if !metadata.is_empty() {
        return metadata;
    }
    if header.contains("\"backbone.double_blocks.")
        && header.contains("_audio")
        && header.contains("\"backbone.single_blocks.")
    {
        "nava".to_string()
    } else if header.contains("\"noise_refiner.") {
        "zimage".to_string()
    } else if header.contains("\"double_stream_modulation_img") {
        "flux-2/klein".to_string()
    } else if header.contains("\"distilled_guidance_layer") {
        "chroma".to_string()
    } else if header.contains("\"double_blocks.") {
        "flux".to_string()
    } else if header.contains("\"audio_vae.") {
        "ltx2".to_string()
    } else if header.contains("\"txtfusion.") {
        "krea2".to_string()
    } else if header.contains("\"embed_image_indicator.weight\"")
        || header.contains("\"llm_cond_proj.weight\"")
    {
        "ideogram4".to_string()
    } else if header.contains("\"model.diffusion_model.joint_blocks") {
        "sd3".to_string()
    } else if header.contains("\"model.diffusion_model.input_blocks") {
        "sdxl".to_string()
    } else if header.contains("\"input_blocks.0.") {
        "sdxl".to_string()
    } else if header.contains("\"txt_norm.") {
        "qwen-image".to_string()
    } else if header.contains("\"time_projection.") {
        "wan2.2".to_string()
    } else {
        "unknown".to_string()
    }
}

fn detect_checkpoint_arch(header: &str, sidecar_hint: &str, name: &str) -> (String, String) {
    let metadata = metadata_architecture(header);
    if !metadata.is_empty() {
        return (metadata, "metadata".to_string());
    }
    let signature = detect_arch(header);
    if signature != "unknown" {
        return (signature, "tensor_signature".to_string());
    }
    let sidecar = normalize_architecture_id(sidecar_hint);
    if sidecar != "unknown" {
        return (sidecar, "sidecar".to_string());
    }
    let filename = detect_arch_from_name(name);
    if filename != "unknown" {
        return (filename.to_string(), "filename".to_string());
    }
    ("unknown".to_string(), "unknown".to_string())
}

fn detect_checkpoint_format(header: &str, arch: &str) -> &'static str {
    if arch == "sdxl"
        && header.contains("\"model.diffusion_model.")
        && header.contains("\"conditioner.embedders.")
    {
        "full_checkpoint"
    } else if arch == "sd3" {
        let pos_embed_shape = serde_json::from_str::<Value>(header)
            .ok()
            .and_then(|document| {
                document
                    .get("model.diffusion_model.pos_embed")
                    .and_then(|tensor| tensor.get("shape"))
                    .and_then(Value::as_array)
                    .map(|shape| shape.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
            });
        match pos_embed_shape.as_deref() {
            Some([1, 36_864, 2_432]) => "sd3_large_full_checkpoint",
            Some([1, 147_456, 1_536]) => "sd3_medium_full_checkpoint",
            _ => "sd3_unknown_checkpoint",
        }
    } else {
        "diffusion_model"
    }
}

fn detect_arch_from_name(name: &str) -> &'static str {
    let lo = name.to_lowercase();
    let c = |s: &str| lo.contains(s);
    if c("minimax") && c("h3") {
        "minimax-h3"
    } else if c("nava") {
        "nava"
    } else if c("scail") {
        "scail2"
    } else if c("bernini") {
        "bernini"
    } else if c("wan2.2") || c("wan 2.2") || c("wan-2.2") || c("wan_2_2") || c("wan22") {
        "wan2.2"
    } else if c("zimage_l2p") || c("z-image-l2p") || c("z_image_l2p") || c("l2p") {
        "zimage-l2p"
    } else if c("hidream") || c("hi-dream") || c("hi_dream") {
        "hidream"
    } else if c("sensenova") || c("sense_nova") || c("sense-nova") {
        "sensenova"
    } else if c("qwen") {
        "qwen-image"
    } else if c("microsoft_lens") || c("microsoft-lens") || c("lenspipeline") {
        "lens"
    } else if c("sd3") || c("sd35") {
        "sd3"
    } else if c("chroma") || c("crhroma") {
        // Keep the common historical EriCrhroma spelling selectable as the
        // Chroma adapter it is; tensor admission still validates every target.
        "chroma"
    } else if c("sdxl")
        || c("sd_xl")
        || c("sd-xl")
        || c("sd xl")
        || c("stable-diffusion-xl")
        || c("animagine")
    {
        "sdxl"
    } else if c("flux2") || c("flux-2") || c("flux_2") {
        "flux-2"
    } else if c("flux1") || c("flux-1") || c("flux_1") || c("flux-dev") {
        "flux"
    } else if c("ltx") {
        "ltx2"
    } else if c("krea") {
        "krea2"
    } else {
        "unknown"
    }
}

fn detect_lora_target_arch(header: &str) -> &'static str {
    if header.contains("\"noise_refiner.") || header.contains("\"context_refiner.") {
        "zimage"
    } else if header.contains("\"layers.") && header.contains(".lora_") {
        "zimage"
    } else if header.contains("zimage") || header.contains("z_image") {
        "zimage"
    } else if header.contains("\"lora_unet_")
        || header.contains("\"lora_te1")
        || header.contains("\"lora_te2")
    {
        "sdxl"
    } else if header.contains("\"lycoris_transformer_blocks_")
        && header.contains("_attn_add_q_proj.")
        && header.contains(".lokr_w1\"")
        && header.contains(".lokr_w2\"")
    {
        // LyCORIS MMDiT exports use flattened Diffusers joint-attention names.
        // The added-context q projection distinguishes SD3/3.5 from ordinary
        // FLUX/Chroma LoRA surfaces.
        "sd3"
    } else if header.contains("\"lora_transformer_distilled_guidance_layer") {
        "chroma"
    } else if header.contains("\"diffusion_model.transformer_blocks.")
        && (header.contains(".lora_A.weight\"") || header.contains(".lora_down.weight\""))
    {
        // Native/PEFT LTX exports use diffusion_model.transformer_blocks
        // directly. Do this before the generic transformer checks below so an
        // arbitrary user-trained LTX adapter does not require "ltx" in its
        // filename to become selectable.
        "ltx2"
    } else if header.contains("\"transformer.transformer_blocks.")
        && header.contains(".attn.to_q.")
        && header.contains(".attn.add_q_proj.")
        && header.contains(".img_mlp.net.0.proj.")
        && (header.contains(".lora_A.weight\"") || header.contains(".lora_down.weight\""))
    {
        // Canonical Qwen-Image PEFT/Serenity export: separate image and added
        // context attention projections plus image/text MLP targets.
        "qwen-image"
    } else if header.contains("\"transformer.layers.")
        && header.contains(".attention.qkv.")
        && header.contains(".feed_forward.w1.")
        && (header.contains(".lora_A.weight\"") || header.contains(".lora_down.weight\""))
    {
        // Canonical Ideogram-4 trainer export: fused qkv plus SwiGLU feed-forward
        // projections on the 34-layer AuraFlow-style trunk.
        "ideogram4"
    } else if header.contains("\"transformer.transformer_blocks.")
        && header.contains(".attn1.to_q.")
        && header.contains(".attn2.to_k.")
        && header.contains(".ff.net.0.proj.")
        && (header.contains(".lora_A.weight\"") || header.contains(".lora_down.weight\""))
    {
        // Canonical Anima SerenityTrainer/PEFT export: 28 blocks with distinct
        // self-attention (attn1), cross-attention (attn2), and GELU FF targets.
        "anima"
    } else if header.contains("\"transformer.single_transformer_blocks.")
        && header.contains("\"transformer.transformer_blocks.")
        && (header.contains(".lora_A.weight\"") || header.contains(".lora_down.weight\""))
    {
        // Diffusers PEFT FLUX.1 adapters do not use the older
        // lora_transformer_* Kohya key prefix.
        "flux"
    } else if header.contains("\"lora_transformer_single_transformer_blocks")
        || header.contains("\"lora_transformer_transformer_blocks")
    {
        "flux"
    } else if header.contains("\"lora_transformer.") {
        "flux"
    } else if header.contains("qwen") {
        "qwen-image"
    } else if header.contains("ideogram") {
        "ideogram4"
    } else if header.contains("ltx") {
        "ltx2"
    } else {
        "unknown"
    }
}

/// First min(header_len, 16 MiB) bytes of a safetensors JSON header ("" on any error).
fn header_text(path: &str) -> String {
    (|| -> std::io::Result<String> {
        let mut f = std::fs::File::open(path)?;
        let mut lenbuf = [0u8; 8];
        f.read_exact(&mut lenbuf)?;
        let header_len = u64::from_le_bytes(lenbuf);
        if header_len == 0 {
            return Ok(String::new());
        }
        let want = header_len.min(HEADER_PROBE_CAP) as usize;
        let mut buf = vec![0u8; want];
        f.seek(SeekFrom::Start(8))?;
        // best-effort fill (short reads tolerated, like the Mojo pread loop break)
        let mut done = 0;
        while done < want {
            match f.read(&mut buf[done..]) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(_) => break,
            }
        }
        buf.truncate(done);
        Ok(String::from_utf8_lossy(&buf).into_owned())
    })()
    .unwrap_or_default()
}

// ── sidecar preview + metadata (ADD-only browser fields) ─────────────────────────

/// Minimal standard-alphabet base64 (RFC 4648, padded). Self-contained so this
/// crate gains no new dependency — matches the file's other shell-free helpers.
fn base64_encode(bytes: &[u8]) -> String {
    const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = *chunk.get(1).unwrap_or(&0) as usize;
        let b2 = *chunk.get(2).unwrap_or(&0) as usize;
        out.push(TBL[b0 >> 2] as char);
        out.push(TBL[((b0 & 0x03) << 4) | (b1 >> 4)] as char);
        out.push(if chunk.len() > 1 {
            TBL[((b1 & 0x0f) << 2) | (b2 >> 6)] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TBL[b2 & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

/// MIME type for a preview image by extension (lowercased), "" if unsupported.
fn image_mime(ext_lower: &str) -> &'static str {
    match ext_lower {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "",
    }
}

/// Read a small image file and return a `data:` URI, or "" (missing / too big /
/// unsupported ext). The browser's `thumbUrl` consumes `data:` URIs directly, so
/// no extra server route is needed to surface the preview.
fn preview_data_uri(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    let mime = image_mime(&ext);
    if mime.is_empty() {
        return String::new();
    }
    let meta = match std::fs::metadata(path) {
        Ok(m) if m.is_file() => m,
        _ => return String::new(),
    };
    if meta.len() == 0 || meta.len() > PREVIEW_INLINE_CAP {
        return String::new();
    }
    match std::fs::read(path) {
        Ok(bytes) if !bytes.is_empty() => format!("data:{};base64,{}", mime, base64_encode(&bytes)),
        _ => String::new(),
    }
}

/// First adjacent preview image for a model whose file path is `model_path`
/// (full path incl. `.safetensors`). Probes the common reference UI/Civitai sidecar
/// names in priority order: `<model>.preview.<ext>`, `<model>.<ext>`. Returns a
/// `data:` URI or "".
fn find_sidecar_preview(model_path: &str) -> String {
    let p = Path::new(model_path);
    let parent = p.parent().unwrap_or_else(|| Path::new("."));
    let stem = p
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.strip_suffix(".safetensors").unwrap_or(n).to_string())
        .unwrap_or_default();
    if stem.is_empty() {
        return String::new();
    }
    const EXTS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "gif"];
    // `<model>.preview.<ext>` first (explicit), then bare `<model>.<ext>`.
    for suffix in [".preview", ""] {
        for ext in EXTS {
            let cand = parent.join(format!("{stem}{suffix}.{ext}"));
            let uri = preview_data_uri(&cand);
            if !uri.is_empty() {
                return uri;
            }
        }
    }
    String::new()
}

/// First in-dir preview for a diffusers-tree model directory `dir`. Probes the
/// conventional cover names. Returns a `data:` URI or "".
fn find_dir_preview(dir: &str) -> String {
    let base = Path::new(dir);
    const NAMES: [&str; 8] = [
        "preview.png",
        "preview.jpg",
        "cover.png",
        "cover.jpg",
        "teaser.jpg",
        "teaser.png",
        "thumbnail.png",
        "thumbnail.jpg",
    ];
    for n in NAMES {
        let uri = preview_data_uri(&base.join(n));
        if !uri.is_empty() {
            return uri;
        }
    }
    String::new()
}

/// Pull a string field by any of several keys from a JSON object (first hit).
fn json_str_any(obj: &Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(s) = obj.get(*k).and_then(|v| v.as_str()) {
            let t = s.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    String::new()
}

/// Parse a sidecar JSON value into description/trigger/arch hints. Handles both
/// the flat generic `{description, trigger|activation text, baseModel|arch}` and
/// the Civitai `.civitai.info` shape (`trainedWords:[…]`, `model.description`,
/// `baseModel`).
fn parse_sidecar_json(v: &Value) -> Sidecar {
    let mut description = json_str_any(v, &["description", "notes", "about"]);
    if description.is_empty() {
        // Civitai nests the human description under `model.description`.
        if let Some(m) = v.get("model") {
            description = json_str_any(m, &["description", "notes"]);
        }
    }
    let mut trigger = json_str_any(
        v,
        &[
            "trigger",
            "trigger_words",
            "triggerWords",
            "activation text",
            "activation_text",
        ],
    );
    if trigger.is_empty() {
        if let Some(arr) = v.get("trainedWords").and_then(|x| x.as_array()) {
            let words: Vec<String> = arr
                .iter()
                .filter_map(|w| w.as_str())
                .map(|w| w.trim().to_string())
                .filter(|w| !w.is_empty())
                .collect();
            trigger = words.join(", ");
        }
    }
    let arch_hint = json_str_any(v, &["arch", "architecture", "baseModel", "base_model"]);
    Sidecar {
        preview: String::new(),
        description,
        trigger,
        arch_hint,
    }
}

/// Read the first existing sidecar metadata JSON for a model file path, returning
/// the distilled fields (preview is filled separately). Probes `<model>.json`,
/// `<model>.civitai.info`, `<model>.cm-info.json` in that order.
fn read_sidecar_metadata(model_path: &str) -> Sidecar {
    let p = Path::new(model_path);
    let parent = p.parent().unwrap_or_else(|| Path::new("."));
    let stem = p
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.strip_suffix(".safetensors").unwrap_or(n).to_string())
        .unwrap_or_default();
    if stem.is_empty() {
        return Sidecar::default();
    }
    for fname in [
        format!("{stem}.json"),
        format!("{stem}.civitai.info"),
        format!("{stem}.cm-info.json"),
    ] {
        let cand = parent.join(&fname);
        if let Ok(text) = std::fs::read_to_string(&cand) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                return parse_sidecar_json(&v);
            }
        }
    }
    Sidecar::default()
}

/// Build the full sidecar bundle (preview + metadata) for a `.safetensors` file.
fn sidecar_for_file(model_path: &str) -> Sidecar {
    let mut s = read_sidecar_metadata(model_path);
    s.preview = find_sidecar_preview(model_path);
    s
}

/// Build the sidecar bundle for a diffusers-tree model DIR (preview only; an
/// adjacent `<dir>.json` is also honored for description/trigger if present).
fn sidecar_for_dir(dir: &str) -> Sidecar {
    let mut s = Sidecar::default();
    // optional `<dir>.json` next to the directory (same convention as files).
    let djson = format!("{dir}.json");
    if let Ok(text) = std::fs::read_to_string(&djson) {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            s = parse_sidecar_json(&v);
        }
    }
    s.preview = find_dir_preview(dir);
    s
}

/// Subdir of `path` relative to scan `root` ("" when `path` is a direct child of
/// `root` or `root` is not a prefix). E.g. root=…/checkpoints,
/// path=…/checkpoints/ltx-video/x.safetensors → "ltx-video".
fn folder_relative_to(path: &str, root: &str) -> String {
    let rest = match path.strip_prefix(root).and_then(|r| r.strip_prefix('/')) {
        Some(r) => r,
        None => return String::new(),
    };
    match rest.rfind('/') {
        Some(i) => rest[..i].to_string(),
        None => String::new(),
    }
}

// ── scanners ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum SafetensorsScanKind {
    Checkpoint,
    Lora,
}

fn should_skip_safetensors_dir(name: &str, kind: SafetensorsScanKind) -> bool {
    let lower = name.to_ascii_lowercase();
    name.starts_with('.')
        || lower == "temp"
        || (kind == SafetensorsScanKind::Checkpoint
            && matches!(
                lower.as_str(),
                "text_encoder"
                    | "text_encoders"
                    | "audio_vae"
                    | "connectors"
                    | "latent_upsampler"
                    | "vocoder"
                    | "tokenizer"
                    | "tokenizers"
                    | "vae"
                    | "scheduler"
                    | "feature_extractor"
                    | "vision_encoder"
            ))
}

fn should_skip_safetensors_file(name: &str, kind: SafetensorsScanKind) -> bool {
    let lower = name.to_ascii_lowercase();
    let sharded_part = lower
        .strip_suffix(".safetensors")
        .and_then(|stem| stem.rsplit_once("-of-"))
        .is_some_and(|(left, right)| {
            right.chars().all(|c| c.is_ascii_digit())
                && left
                    .rsplit_once('-')
                    .is_some_and(|(_, index)| index.chars().all(|c| c.is_ascii_digit()))
        });
    lower.contains(".fp8cache.")
        || sharded_part
        || (kind == SafetensorsScanKind::Checkpoint && lower.contains("lora"))
}

fn list_safetensors_recursive(
    root: &Path,
    dir: &Path,
    kind: SafetensorsScanKind,
    seen_dirs: &mut HashSet<PathBuf>,
    out: &mut Vec<ScanEntry>,
) {
    let canonical = match std::fs::canonicalize(dir) {
        Ok(path) => path,
        Err(_) => return,
    };
    if !seen_dirs.insert(canonical) {
        return;
    }
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };
    for ent in rd.flatten() {
        let path = ent.path();
        let fname = ent.file_name().to_string_lossy().into_owned();
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if metadata.is_dir() {
            if !should_skip_safetensors_dir(&fname, kind) {
                list_safetensors_recursive(root, &path, kind, seen_dirs, out);
            }
            continue;
        }
        if !metadata.is_file()
            || !fname.to_ascii_lowercase().ends_with(".safetensors")
            || should_skip_safetensors_file(&fname, kind)
        {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let name = relative
            .strip_suffix(".safetensors")
            .unwrap_or(&relative)
            .to_string();
        out.push(ScanEntry {
            name,
            path: path.to_string_lossy().into_owned(),
            arch: String::new(),
            detected_arch: String::new(),
            arch_source: String::new(),
            arch_override: String::new(),
            format: String::new(),
            size: metadata.len() as i64,
            folder: String::new(),
            sidecar: Sidecar::default(),
        });
    }
}

/// Recursively discover safetensors beneath a registry root. This is the same
/// user contract as SwarmUI's model folders: category subdirectories are valid
/// and do not require a code change. Symlinked directories are followed once.
fn list_safetensors(dir: &str) -> Vec<ScanEntry> {
    let mut out = Vec::new();
    let root = Path::new(dir);
    let mut seen_dirs = HashSet::new();
    list_safetensors_recursive(
        root,
        root,
        SafetensorsScanKind::Checkpoint,
        &mut seen_dirs,
        &mut out,
    );
    out
}

fn list_lora_safetensors(dir: &str) -> Vec<ScanEntry> {
    let mut out = Vec::new();
    let root = Path::new(dir);
    let mut seen_dirs = HashSet::new();
    list_safetensors_recursive(
        root,
        root,
        SafetensorsScanKind::Lora,
        &mut seen_dirs,
        &mut out,
    );
    out
}

fn external_checkpoint_scan_roots(root: &Path) -> Vec<(String, PathBuf)> {
    let mut roots = Vec::new();
    for (prefix, subdir) in [
        ("", "checkpoints"),
        ("diffusion_models", "diffusion_models"),
        ("unet", "unet"),
        ("dits", "dits"),
    ] {
        let candidate = root.join(subdir);
        if candidate.is_dir() {
            roots.push((prefix.to_string(), candidate));
        }
    }
    // A user may point Settings directly at a folder of checkpoint files, as
    // SwarmUI allows. Only fall back to the root itself when it is not already
    // a conventional multi-category model tree.
    if roots.is_empty() {
        roots.push((String::new(), root.to_path_buf()));
    }
    roots
}

fn external_identity_prefix(root: &Path) -> String {
    root.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("external")
        .replace(['/', '\\'], "_")
}

fn append_checkpoint_files(
    out: &mut Vec<ScanEntry>,
    scan_root: &Path,
    identity_prefix: &str,
    identity_collision_prefix: &str,
    overrides: &BTreeMap<String, String>,
    used_paths: &mut HashSet<PathBuf>,
    used_names: &mut HashSet<String>,
) {
    let scan_root_text = scan_root.to_string_lossy().into_owned();
    for mut entry in list_safetensors(&scan_root_text) {
        let Ok(canonical_path) = std::fs::canonicalize(&entry.path) else {
            continue;
        };
        if !used_paths.insert(canonical_path) {
            continue;
        }
        if !identity_prefix.is_empty() {
            entry.name = format!("{identity_prefix}/{}", entry.name);
        }
        if checkpoint_component_type(&entry.name).is_some() {
            continue;
        }
        if used_names.contains(&entry.name) {
            entry.name = format!("external/{identity_collision_prefix}/{}", entry.name);
        }
        used_names.insert(entry.name.clone());
        let header = header_text(&entry.path);
        entry.sidecar = sidecar_for_file(&entry.path);
        let (detected_arch, detected_source) =
            detect_checkpoint_arch(&header, &entry.sidecar.arch_hint, &entry.name);
        entry.detected_arch = detected_arch.clone();
        if let Some(arch_override) = overrides.get(&entry.name) {
            entry.arch = arch_override.clone();
            entry.arch_override = arch_override.clone();
            entry.arch_source = "user_override".to_string();
        } else {
            entry.arch = detected_arch;
            entry.arch_source = detected_source;
        }
        entry.format = detect_checkpoint_format(&header, &entry.arch).to_string();
        entry.folder = folder_relative_to(&entry.path, &scan_root_text);
        out.push(entry);
    }
}

/// Some installations keep product support weights in `checkpoints/` because
/// that is where the upstream download instructions put them. They remain
/// visible in the registry, but are not independent generator checkpoints.
fn checkpoint_component_type(name: &str) -> Option<&'static str> {
    let normalized = name.replace('\\', "/").to_ascii_lowercase();
    let basename = Path::new(&normalized)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(&normalized)
        .strip_suffix(".safetensors")
        .unwrap_or_else(|| {
            Path::new(&normalized)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(&normalized)
        });
    if normalized.contains("audio_vae") || basename == "wan2.2_vae" || basename == "wan2_2_vae" {
        Some("vae")
    } else if normalized.contains("spatial-upscaler") || normalized.contains("spatial_upscaler") {
        Some("upscaler")
    } else if normalized.contains("umt5_xxl_enc")
        || normalized.contains("qwen_2.5_vl_7b_fp8_scaled")
    {
        Some("clip")
    } else if basename.contains("ltx")
        && (basename.contains("lora") || basename.contains("svdint4") || basename.contains("svdw"))
    {
        Some("runtime_component")
    } else {
        None
    }
}

fn discover_model_index_dirs_recursive(
    dir: &Path,
    seen_dirs: &mut HashSet<PathBuf>,
    out: &mut Vec<PathBuf>,
) {
    let canonical = match std::fs::canonicalize(dir) {
        Ok(path) => path,
        Err(_) => return,
    };
    if !seen_dirs.insert(canonical) {
        return;
    }
    if dir.join("model_index.json").is_file() {
        out.push(dir.to_path_buf());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if !metadata.is_dir() {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        if name.starts_with('.') || matches!(lower.as_str(), "temp" | "output") {
            continue;
        }
        discover_model_index_dirs_recursive(&path, seen_dirs, out);
    }
}

/// Discover complete Diffusers-style bundles by their manifest rather than by
/// a hardcoded folder-name allowlist. This includes arbitrary user subfolders.
fn discover_model_index_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    discover_model_index_dirs_recursive(root, &mut HashSet::new(), &mut out);
    out.sort();
    out
}

fn diffusers_directory_identity(root: &Path, dir: &Path) -> String {
    let relative = dir
        .strip_prefix(root)
        .unwrap_or(dir)
        .to_string_lossy()
        .replace('\\', "/");
    relative
        .strip_prefix("checkpoints/")
        .unwrap_or(&relative)
        .to_string()
}

fn diffusers_directory_arch(dir: &Path, identity: &str) -> String {
    let name_arch = detect_arch_from_name(identity);
    if name_arch != "unknown" {
        return name_arch.to_string();
    }
    let class_name = std::fs::read_to_string(dir.join("model_index.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|document| {
            document
                .get("_class_name")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default();
    normalize_user_model_type(&class_name).unwrap_or_else(|| "unknown".to_string())
}

fn is_registry_artifact_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("safetensors" | "ckpt" | "pt" | "pth" | "bin")
    )
}

fn scan_registry_artifact_dir(
    category: &str,
    root: &Path,
    dir: &Path,
    seen_dirs: &mut HashSet<PathBuf>,
    seen_files: &mut HashSet<PathBuf>,
    out: &mut Vec<RegistryArtifact>,
) {
    let canonical_dir = match std::fs::canonicalize(dir) {
        Ok(path) => path,
        Err(_) => return,
    };
    if !seen_dirs.insert(canonical_dir) {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if metadata.is_dir() {
            let lower = name.to_ascii_lowercase();
            if !name.starts_with('.') && lower != "temp" {
                scan_registry_artifact_dir(category, root, &path, seen_dirs, seen_files, out);
            }
            continue;
        }
        if !metadata.is_file() || !is_registry_artifact_file(&path) {
            continue;
        }
        let canonical_file = match std::fs::canonicalize(&path) {
            Ok(path) => path,
            Err(_) => continue,
        };
        if !seen_files.insert(canonical_file) {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let folder = Path::new(&relative)
            .parent()
            .map(|parent| parent.to_string_lossy().replace('\\', "/"))
            .filter(|parent| parent != ".")
            .unwrap_or_default();
        out.push(RegistryArtifact {
            name: relative,
            path: path.to_string_lossy().into_owned(),
            folder,
            artifact_type: category.to_string(),
            size: metadata.len() as i64,
        });
    }
}

fn scan_registry_artifacts() -> Vec<RegistryArtifact> {
    let models = model_root();
    let mut out = Vec::new();
    let mut seen_dirs = HashSet::new();
    let mut seen_files = HashSet::new();
    for (category, relative_root) in REGISTRY_ARTIFACT_ROOTS {
        let root = models.join(relative_root);
        scan_registry_artifact_dir(
            category,
            &root,
            &root,
            &mut seen_dirs,
            &mut seen_files,
            &mut out,
        );
    }
    // Reclassify support weights placed in checkpoints/ by upstream download
    // recipes. The base-model scan applies the same predicate and skips them.
    let checkpoint_root = models.join("checkpoints");
    let mut checkpoint_candidates = Vec::new();
    scan_registry_artifact_dir(
        "runtime_component",
        &checkpoint_root,
        &checkpoint_root,
        &mut seen_dirs,
        &mut seen_files,
        &mut checkpoint_candidates,
    );
    for mut artifact in checkpoint_candidates {
        if let Some(category) = checkpoint_component_type(&artifact.name) {
            artifact.artifact_type = category.to_string();
            out.push(artifact);
        }
    }
    out.sort_by(|left, right| {
        left.artifact_type
            .cmp(&right.artifact_type)
            .then_with(|| {
                left.name
                    .to_ascii_lowercase()
                    .cmp(&right.name.to_ascii_lowercase())
            })
            .then_with(|| left.path.cmp(&right.path))
    });
    out
}

fn dir_exists(dir: &str) -> bool {
    Path::new(dir).is_dir()
}

/// `du -sb <dir>` (apparent size in bytes) — shelled out for byte-identity.
fn du_sb(dir: &str) -> i64 {
    std::process::Command::new("du")
        .args(["-sb", dir])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<i64>().ok())
        })
        .unwrap_or(0)
}

fn scan_checkpoints_uncached() -> Vec<ScanEntry> {
    let mut out = Vec::new();
    let mut directory_identities = HashSet::new();
    let models = model_root();
    let overrides = load_model_type_overrides_from(&models).unwrap_or_else(|error| {
        eprintln!("warning: ignoring invalid model type overrides: {error}");
        BTreeMap::new()
    });
    let checkpoints = models.join("checkpoints");
    let checkpoints = checkpoints.to_string_lossy().into_owned();
    for (identity_prefix, scan_root) in [
        ("", checkpoints.clone()),
        (
            "diffusion_models",
            models
                .join("diffusion_models")
                .to_string_lossy()
                .into_owned(),
        ),
        ("unet", models.join("unet").to_string_lossy().into_owned()),
        ("dits", models.join("dits").to_string_lossy().into_owned()),
    ] {
        for mut e in list_safetensors(&scan_root) {
            if !identity_prefix.is_empty() {
                e.name = format!("{identity_prefix}/{}", e.name);
            }
            if checkpoint_component_type(&e.name).is_some() {
                continue;
            }
            let header = header_text(&e.path);
            e.sidecar = sidecar_for_file(&e.path);
            let (detected_arch, detected_source) =
                detect_checkpoint_arch(&header, &e.sidecar.arch_hint, &e.name);
            e.detected_arch = detected_arch.clone();
            if let Some(arch_override) = overrides.get(&e.name) {
                e.arch = arch_override.clone();
                e.arch_override = arch_override.clone();
                e.arch_source = "user_override".to_string();
            } else {
                e.arch = detected_arch;
                e.arch_source = detected_source;
            }
            e.format = detect_checkpoint_format(&header, &e.arch).to_string();
            e.folder = if identity_prefix.is_empty() {
                folder_relative_to(&e.path, &scan_root)
            } else {
                folder_relative_to(&e.path, &models.to_string_lossy())
            };
            out.push(e);
        }
    }
    let mut used_checkpoint_paths = out
        .iter()
        .filter_map(|entry| std::fs::canonicalize(&entry.path).ok())
        .collect::<HashSet<_>>();
    let mut used_checkpoint_names = out
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<HashSet<_>>();
    let external_roots = extra_model_roots();
    for external_root in &external_roots {
        let collision_prefix = external_identity_prefix(external_root);
        for (identity_prefix, scan_root) in external_checkpoint_scan_roots(external_root) {
            append_checkpoint_files(
                &mut out,
                &scan_root,
                &identity_prefix,
                &collision_prefix,
                &overrides,
                &mut used_checkpoint_paths,
                &mut used_checkpoint_names,
            );
        }
    }
    // Complete Diffusers bundles are discovered recursively by model_index.json.
    // Users may add them in any subdirectory without requiring a code change.
    for discovery_root in std::iter::once(models.clone()).chain(external_roots.into_iter()) {
        let collision_prefix = external_identity_prefix(&discovery_root);
        for dir_path in discover_model_index_dirs(&discovery_root) {
            let mut name = diffusers_directory_identity(&discovery_root, &dir_path);
            let Ok(canonical_path) = std::fs::canonicalize(&dir_path) else {
                continue;
            };
            if name.is_empty() || !used_checkpoint_paths.insert(canonical_path) {
                continue;
            }
            if used_checkpoint_names.contains(&name) {
                name = format!("external/{collision_prefix}/{name}");
            }
            if !directory_identities.insert(name.clone()) {
                continue;
            }
            used_checkpoint_names.insert(name.clone());
            let detected_arch = diffusers_directory_arch(&dir_path, &name);
            let (arch, arch_override, arch_source) =
                if let Some(arch_override) = overrides.get(&name) {
                    (
                        arch_override.clone(),
                        arch_override.clone(),
                        "user_override".to_string(),
                    )
                } else {
                    (
                        detected_arch.clone(),
                        String::new(),
                        "model_index".to_string(),
                    )
                };
            let dir = dir_path.to_string_lossy().into_owned();
            out.push(ScanEntry {
                name,
                path: dir.clone(),
                arch,
                detected_arch,
                arch_source,
                arch_override,
                format: "diffusers_directory".to_string(),
                size: du_sb(&dir),
                folder: folder_relative_to(&dir, &discovery_root.to_string_lossy()),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    // Bundled directory models without a Diffusers model_index manifest.
    for (name, arch) in [("anima", "anima"), ("sensenova_u1", "sensenova")] {
        let dir = models.join(name).to_string_lossy().into_owned();
        if dir_exists(&dir) && directory_identities.insert(name.to_string()) {
            let size = du_sb(&dir);
            out.push(ScanEntry {
                name: name.to_string(),
                path: dir.clone(),
                arch: arch.to_string(),
                detected_arch: arch.to_string(),
                arch_source: "bundled_identity".to_string(),
                arch_override: String::new(),
                format: "diffusers_directory".to_string(),
                size,
                folder: folder_relative_to(&dir, &models.to_string_lossy()),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    // Bernini-R is deliberately not discoverable/selectable until its local
    // pinned creator-parity, cache-provenance, full-render, visual, mux, and
    // VRAM product gate passes. The directory may be partially downloaded for
    // hours; directory presence alone is not model readiness.
    if crate::video::bernini::bernini_product_gate_passed() {
        let name = "Bernini-R-Diffusers";
        let dir = format!("{checkpoints}/{name}");
        if dir_exists(&dir) && directory_identities.insert(name.to_string()) {
            out.push(ScanEntry {
                name: name.to_string(),
                path: dir.clone(),
                arch: "bernini".to_string(),
                detected_arch: "bernini".to_string(),
                arch_source: "bundled_identity".to_string(),
                arch_override: String::new(),
                format: "diffusers_directory".to_string(),
                size: du_sb(&dir),
                folder: folder_relative_to(&dir, &checkpoints),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    // SCAIL-2 is a directory-backed multi-artifact model. Expose one product
    // identity only after the local full-animation gate binds the current
    // seven repo-built runners and installed FP8 cache.
    if crate::video::scail2::scail2_product_gate_passed() {
        let name = "SCAIL-2-Mojo";
        let dir = format!("{checkpoints}/{name}");
        if dir_exists(&dir) && directory_identities.insert(name.to_string()) {
            out.push(ScanEntry {
                name: name.to_string(),
                path: dir.clone(),
                arch: "scail2".to_string(),
                detected_arch: "scail2".to_string(),
                arch_source: "bundled_identity".to_string(),
                arch_override: String::new(),
                format: "diffusers_directory".to_string(),
                size: du_sb(&dir),
                folder: folder_relative_to(&dir, &checkpoints),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    // MiniMax-H3 is a multi-directory audio/video model. Installed model
    // identity is a filesystem fact, not a benchmark verdict: validation
    // reports may describe confidence, but never hide a user's model.
    let name = "MiniMax-H3-Mojo";
    let dir = format!("{checkpoints}/MiniMax-H3");
    if dir_exists(&dir) && directory_identities.insert(name.to_string()) {
        out.push(ScanEntry {
            name: name.to_string(),
            path: dir.clone(),
            arch: "minimax-h3".to_string(),
            detected_arch: "minimax-h3".to_string(),
            arch_source: "bundled_identity".to_string(),
            arch_override: String::new(),
            format: "diffusers_directory".to_string(),
            size: du_sb(&dir),
            folder: folder_relative_to(&dir, &checkpoints),
            sidecar: sidecar_for_dir(&dir),
        });
    }
    // known multi-shard checkpoint subdirs under checkpoints/.
    for (name, arch) in [
        ("qwen-image-2512", "qwen-image"),
        ("ideogram-4-fp8", "ideogram4"),
        ("ltx2-diffusers", "ltx2"),
        ("Wan2.2-TI2V-5B", "wan2.2"),
        ("Wan2.2-TI2V-5B-bf16", "wan2.2"),
        ("Wan2.2-TI2V-5B-Mojo", "wan2.2"),
        ("wan2.2_t2v_a14b_fp8_e4m3", "wan2.2"),
    ] {
        let dir = format!("{checkpoints}/{name}");
        if dir_exists(&dir) && directory_identities.insert(name.to_string()) {
            let size = du_sb(&dir);
            out.push(ScanEntry {
                name: name.to_string(),
                path: dir.clone(),
                arch: arch.to_string(),
                detected_arch: arch.to_string(),
                arch_source: "bundled_identity".to_string(),
                arch_override: String::new(),
                format: "diffusers_directory".to_string(),
                size,
                folder: folder_relative_to(&dir, &checkpoints),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    out
}

/// Model headers are immutable for the common request path, while capability
/// validation may ask for the selected architecture several times. Keep a
/// short snapshot instead of rereading every safetensors header per request.
/// The one-minute window remains responsive through explicit UI refreshes
/// without requiring a long-lived file watcher.
type CheckpointScanCache = std::sync::Mutex<Option<(std::time::Instant, Vec<ScanEntry>)>>;

fn checkpoint_scan_cache() -> &'static CheckpointScanCache {
    static CACHE: std::sync::OnceLock<CheckpointScanCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

fn invalidate_checkpoint_scan_cache() {
    if let Ok(mut guard) = checkpoint_scan_cache().lock() {
        *guard = None;
    }
}

fn scan_checkpoints_with_refresh(force: bool) -> Vec<ScanEntry> {
    let cache = checkpoint_scan_cache();
    if !force {
        if let Ok(guard) = cache.lock() {
            if let Some((created, entries)) = guard.as_ref() {
                if created.elapsed() < std::time::Duration::from_secs(60) {
                    return entries.clone();
                }
            }
        }
    }
    let entries = scan_checkpoints_uncached();
    if let Ok(mut guard) = cache.lock() {
        *guard = Some((std::time::Instant::now(), entries.clone()));
    }
    entries
}

fn scan_checkpoints() -> Vec<ScanEntry> {
    scan_checkpoints_with_refresh(false)
}

fn scan_loras() -> Vec<ScanEntry> {
    let mut out = Vec::new();
    let overrides = load_model_type_overrides_from(&model_root()).unwrap_or_else(|error| {
        eprintln!("warning: ignoring invalid LoRA type overrides: {error}");
        BTreeMap::new()
    });
    let primary = model_root().join("loras");
    let mut roots = vec![primary];
    for external in extra_model_roots() {
        let candidate = if external
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("loras"))
        {
            external
        } else {
            external.join("loras")
        };
        if candidate.is_dir() {
            roots.push(candidate);
        }
    }
    let mut used_paths = HashSet::new();
    let mut used_names = HashSet::new();
    for root in roots {
        let root_text = root.to_string_lossy().into_owned();
        let collision_prefix = external_identity_prefix(&root);
        for mut entry in list_lora_safetensors(&root_text) {
            let Ok(canonical_path) = std::fs::canonicalize(&entry.path) else {
                continue;
            };
            if !used_paths.insert(canonical_path) {
                continue;
            }
            if used_names.contains(&entry.name) {
                entry.name = format!("external/{collision_prefix}/{}", entry.name);
            }
            used_names.insert(entry.name.clone());
            let mut target = detect_arch_from_name(&entry.name).to_string();
            if target == "unknown" {
                target = detect_lora_target_arch(&header_text(&entry.path)).to_string();
            }
            entry.arch = target.clone();
            entry.detected_arch = target;
            entry.arch_source = if entry.detected_arch == "unknown" {
                "unknown".to_string()
            } else {
                "tensor_or_filename".to_string()
            };
            entry.format = "lora".to_string();
            entry.folder = folder_relative_to(&entry.path, &root_text);
            entry.sidecar = sidecar_for_file(&entry.path);
            if entry.arch == "unknown" && !entry.sidecar.arch_hint.is_empty() {
                entry.arch = entry.sidecar.arch_hint.clone();
            }
            if let Some(arch_override) = overrides.get(&lora_type_override_key(&entry.name)) {
                entry.arch = arch_override.clone();
                entry.arch_override = arch_override.clone();
                entry.arch_source = "user_override".to_string();
            }
            out.push(entry);
        }
    }
    out.sort_by(|left, right| scan_entry_cmp(left, right, "name"));
    out
}

// ── browser filter / sort / compatibility ───────────────────────────────────────

fn entry_arch(e: &ScanEntry) -> String {
    if e.arch.is_empty() {
        "unknown".to_string()
    } else {
        e.arch.clone()
    }
}

fn contains_ci(text: &str, q: &str) -> bool {
    q.is_empty() || text.to_lowercase().contains(&q.to_lowercase())
}

fn filter_matches(value: &str, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    let f = filter.to_lowercase();
    if f == "all" || f == "any" {
        return true;
    }
    value.to_lowercase().contains(&f)
}

fn matches_browser(e: &ScanEntry, search: &str, filter: &str) -> bool {
    if !search.is_empty()
        && !contains_ci(&e.name, search)
        && !contains_ci(&e.path, search)
        && !contains_ci(&entry_arch(e), search)
    {
        return false;
    }
    filter_matches(&entry_arch(e), filter)
}

/// `_scan_entry_before` as a total ordering (paths are unique → total).
fn scan_entry_cmp(a: &ScanEntry, b: &ScanEntry, sort: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let s = sort.to_lowercase();
    let (an, bn) = (a.name.to_lowercase(), b.name.to_lowercase());
    let primary = match s.as_str() {
        "size" | "size_desc" => b.size.cmp(&a.size),
        "size_asc" => a.size.cmp(&b.size),
        "name_desc" => bn.cmp(&an),
        "arch" | "family" => entry_arch(a)
            .to_lowercase()
            .cmp(&entry_arch(b).to_lowercase()),
        _ => Ordering::Equal,
    };
    if primary != Ordering::Equal {
        return primary;
    }
    match an.cmp(&bn) {
        Ordering::Equal => a.path.cmp(&b.path),
        o => o,
    }
}

fn model_lora_compatible(model_arch: &str, target_arch: &str) -> bool {
    let m = model_arch.to_lowercase();
    let t = target_arch.to_lowercase();
    if t.is_empty() || t == "unknown" || m.is_empty() || m == "unknown" {
        return false;
    }
    m == t
}

fn model_arch_for(models: &[ScanEntry], model: &str) -> String {
    if model.is_empty() {
        return String::new();
    }
    let needle = model.to_lowercase();
    for e in models {
        if e.name.to_lowercase() == needle || e.path.to_lowercase() == needle {
            return entry_arch(e);
        }
    }
    model.to_string()
}

fn compatible_models_json(lora: &ScanEntry, models: &[ScanEntry]) -> Value {
    let target = entry_arch(lora);
    let names: Vec<Value> = models
        .iter()
        .filter(|m| model_lora_compatible(&entry_arch(m), &target))
        .map(|m| Value::String(m.name.clone()))
        .collect();
    Value::Array(names)
}

fn lora_incompatible_reason(
    selected_model: &str,
    selected_arch: &str,
    target_arch: &str,
    compatible: bool,
) -> String {
    if selected_model.is_empty() {
        "no model selected".to_string()
    } else if target_arch == "unknown" {
        "unknown LoRA target_arch".to_string()
    } else if selected_arch.is_empty() || selected_arch == "unknown" {
        "unknown selected model arch".to_string()
    } else if !compatible {
        format!("target_arch {target_arch} is not compatible with model arch {selected_arch}")
    } else {
        String::new()
    }
}

// ── card builders (exact key order) ─────────────────────────────────────────────

fn model_entry_json(e: &ScanEntry, resident: &str) -> Value {
    let arch = entry_arch(e);
    let generation_defaults =
        crate::capabilities::generation_defaults_for_model_arch(&e.name, &arch);
    let loaded = !resident.is_empty() && e.name == resident;
    let preview = e.sidecar.preview.clone();
    let selected_checkpoint_scope = entry_selected_checkpoint_scope(e);
    let selected_checkpoint_supported = selected_checkpoint_scope.is_some();
    // These are the finite, pre-existing product profiles whose workers choose
    // a bundled artifact by profile identity. Keep them selectable, but never
    // pretend that an arbitrary same-architecture filename reaches the worker.
    // New user checkpoints must use a selected-checkpoint loader instead of
    // being added to this compatibility list.
    let bundled_profile = matches!(
        (arch.as_str(), e.name.as_str()),
        ("anima", "anima")
            | ("chroma", "chroma1_hd_bf16")
            | ("flux", "flux1-dev")
            | ("flux-2/klein", "flux-2-klein-base-4b")
            | ("flux-2/klein", "flux-2-klein-base-9b")
            | ("flux-2/klein", "flux-2-klein-base-9b_fp8_e4m3fn")
            | ("ideogram4", "ideogram-4-fp8")
            | ("lens", "microsoft_lens")
            | ("qwen-image", "qwen-image-2512")
            | ("sd3", "sd3.5_large")
            | ("zimage", "zimage_base")
    );
    let route = if matches!(
        arch.as_str(),
        "ltx2" | "minimax-h3" | "wan2.2" | "nava" | "bernini" | "scail2"
    ) {
        "video"
    } else if crate::capabilities::model_family_for_arch(&arch).is_some() {
        "image"
    } else {
        "unsupported"
    };
    // Non-LTX video workers still consume finite compiled product identities.
    // LTX2 is selected-checkpoint capable: geometry remains bounded by an AOT
    // runner, while the denoiser artifact is the exact scanned file selected
    // by the user.
    let video_profile = e.arch_override.is_empty()
        && matches!(
            (arch.as_str(), e.name.as_str()),
            ("wan2.2", "Wan2.2-TI2V-5B-Mojo")
                | ("wan2.2", "wan2.2_t2v_a14b_fp8_e4m3")
                | ("bernini", "Bernini-R-Diffusers")
                | ("scail2", "SCAIL-2-Mojo")
                | ("minimax-h3", "MiniMax-H3-Mojo")
        );
    let blocked_reason = crate::capabilities::blocked_image_model_reason(&e.name);
    let runtime_supported = blocked_reason.is_none()
        && (video_profile || selected_checkpoint_supported || bundled_profile);
    let selected_checkpoint_scope = if let Some(scope) = selected_checkpoint_scope {
        scope
    } else if bundled_profile {
        "bundled_profile"
    } else if video_profile {
        "video_route"
    } else {
        ""
    };
    let runtime_reason = if let Some(reason) = blocked_reason {
        reason.to_string()
    } else if runtime_supported {
        String::new()
    } else if route == "video" {
        format!(
            "This {arch} artifact is visible, but it is not one of the compiled video product profiles and its selected path is not consumed"
        )
    } else if route == "image" {
        format!(
            "The {arch} backend is installed, but it does not yet consume arbitrary selected checkpoints; this file is disabled to prevent silently loading different weights"
        )
    } else {
        format!("No production runtime is registered for architecture {arch}")
    };
    let metadata = json!({
        "schema": "serenity.model.metadata.v1",
        "source": "disk_scan",
        "family": arch,
        "detected_arch": e.detected_arch,
        "arch_source": e.arch_source,
        "arch_override": e.arch_override,
        "format": e.format,
        "notes": e.sidecar.description,        // ADD: from <model>.json/.civitai.info
        "description": e.sidecar.description,  // ADD: alias for clarity
        "trigger": e.sidecar.trigger,          // ADD: sidecar trigger words (if any)
    });
    let card = json!({
        "schema": "serenity.model.card.v1",
        "title": e.name,
        "subtitle": arch,
        "path": e.path,
        "arch": arch,
        "detected_arch": e.detected_arch,
        "arch_source": e.arch_source,
        "arch_override": e.arch_override,
        "format": e.format,
        "generation_route": route,
        "runtime_supported": runtime_supported,
        "runtime_reason": runtime_reason,
        "uses_selected_checkpoint": selected_checkpoint_supported,
        "selected_checkpoint_scope": selected_checkpoint_scope,
        "folder": e.folder,                    // ADD: subdir under scan root ("" = top)
        "size": e.size,
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "favorite": false,
        "loaded": loaded,
        "generation_defaults": generation_defaults.clone(),
        "metadata": metadata,
    });
    json!({
        "name": e.name,
        "path": e.path,
        "folder": e.folder,                    // ADD: subdir under scan root ("" = top)
        "arch": arch,
        "detected_arch": e.detected_arch,
        "arch_source": e.arch_source,
        "arch_override": e.arch_override,
        "format": e.format,
        "generation_route": route,
        "runtime_supported": runtime_supported,
        "runtime_reason": runtime_reason,
        "uses_selected_checkpoint": selected_checkpoint_supported,
        "selected_checkpoint_scope": selected_checkpoint_scope,
        "size": e.size,
        "loaded": loaded,
        "type": "checkpoint",
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "trigger": e.sidecar.trigger,          // ADD: sidecar trigger words (if any)
        "favorite": false,
        "generation_defaults": generation_defaults,
        "metadata": metadata,
        "card": card,
    })
}

fn lora_entry_json(
    e: &ScanEntry,
    models: &[ScanEntry],
    selected_model: &str,
    selected_arch: &str,
) -> Value {
    let target = entry_arch(e);
    let usage = lora_usage(&e.name);
    let selectable_as_lora = usage == "overlay";
    let compatible = model_lora_compatible(selected_arch, &target);
    let reason = lora_incompatible_reason(selected_model, selected_arch, &target, compatible);
    let compatibility = json!({
        "compatible": compatible,
        "model": selected_model,
        "model_arch": selected_arch,
        "target_arch": target,
        "incompatible_reason": reason,
    });
    let preview = e.sidecar.preview.clone();
    let trigger = e.sidecar.trigger.clone();
    let metadata = json!({
        "schema": "serenity.lora.metadata.v1",
        "source": "safetensors_header_probe",
        "target_arch": target,
        "trigger": trigger,                    // ADD: from <lora>.json/.civitai.info
        "notes": e.sidecar.description,        // ADD: sidecar description (if any)
        "description": e.sidecar.description,  // ADD: alias for clarity
        "usage": usage,
        "selectable_as_lora": selectable_as_lora,
    });
    let card = json!({
        "schema": "serenity.lora.card.v1",
        "title": e.name,
        "subtitle": target,
        "path": e.path,
        "folder": e.folder,                    // ADD: subdir under scan root ("" = top)
        "size": e.size,
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "favorite": false,
        "metadata": metadata,
        "compatibility": compatibility,
    });
    json!({
        "name": e.name,
        "path": e.path,
        "folder": e.folder,                    // ADD: subdir under scan root ("" = top)
        "size": e.size,
        "arch": target,
        "target_arch": target,
        "detected_arch": e.detected_arch,
        "arch_source": e.arch_source,
        "arch_override": e.arch_override,
        "trigger": trigger,                    // populated from sidecar (was always "")
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "favorite": false,
        "compatible_models": compatible_models_json(e, models),
        "compatible": compatible,
        "compatibility": compatibility,
        "incompatible_reason": reason,
        "usage": usage,
        "selectable_as_lora": selectable_as_lora,
        "metadata": metadata,
        "card": card,
    })
}

fn registry_artifact_json(artifact: &RegistryArtifact) -> Value {
    json!({
        "schema": "serenity.model.artifact.v1",
        "name": artifact.name,
        "path": artifact.path,
        "folder": artifact.folder,
        "type": artifact.artifact_type,
        "arch": "any",
        "size": artifact.size,
    })
}

fn sorted_filtered<F>(
    entries: &[ScanEntry],
    search: &str,
    filter: &str,
    sort: &str,
    build: F,
) -> Value
where
    F: Fn(&ScanEntry) -> Value,
{
    let mut idx: Vec<usize> = (0..entries.len())
        .filter(|&i| matches_browser(&entries[i], search, filter))
        .collect();
    idx.sort_by(|&a, &b| scan_entry_cmp(&entries[a], &entries[b], sort));
    Value::Array(idx.into_iter().map(|i| build(&entries[i])).collect())
}

// ── handler ─────────────────────────────────────────────────────────────────────

/// GET /v1/models — disk-scanned model + LoRA browser cards (`serenity.models.v1`).
pub async fn get_models(Query(q): Query<HashMap<String, String>>) -> Response {
    let g = |k: &str| q.get(k).cloned().unwrap_or_default();
    let resident = ""; // stub-parity: the Rust server tracks no resident model name yet

    let mut search = g("search");
    let qparam = g("q");
    if search.is_empty() && !qparam.is_empty() {
        search = qparam;
    }
    let filter = g("filter");
    let sort = g("sort");
    let mut lora_search = g("lora_search");
    if lora_search.is_empty() {
        lora_search = search.clone();
    }
    let lora_filter = g("lora_filter");
    let mut lora_sort = g("lora_sort");
    if lora_sort.is_empty() {
        lora_sort = sort.clone();
    }
    let mut selected_model = g("model");
    if selected_model.is_empty() {
        selected_model = resident.to_string();
    }

    let force_refresh = matches!(
        g("refresh").to_ascii_lowercase().as_str(),
        "1" | "true" | "yes"
    );
    let models = scan_checkpoints_with_refresh(force_refresh);
    let scanned_loras = scan_loras();
    let loras = scanned_loras
        .iter()
        .filter(|entry| lora_selectable_as_overlay(&entry.name))
        .cloned()
        .collect::<Vec<_>>();
    let mut artifacts = scan_registry_artifacts();
    artifacts.extend(
        scanned_loras
            .iter()
            .filter(|entry| !lora_selectable_as_overlay(&entry.name))
            .map(|entry| RegistryArtifact {
                name: entry.name.clone(),
                path: entry.path.clone(),
                folder: entry.folder.clone(),
                artifact_type: "feature_adapter".to_string(),
                size: entry.size,
            }),
    );
    artifacts.sort_by(|left, right| {
        left.artifact_type
            .cmp(&right.artifact_type)
            .then_with(|| {
                left.name
                    .to_ascii_lowercase()
                    .cmp(&right.name.to_ascii_lowercase())
            })
            .then_with(|| left.path.cmp(&right.path))
    });
    let selected_arch = model_arch_for(&models, &selected_model);

    let query = json!({
        "search": search,
        "filter": filter,
        "sort": sort,
        "q": search,
        "lora_search": lora_search,
        "lora_filter": lora_filter,
        "lora_sort": lora_sort,
        "model": selected_model,
    });
    let models_json = sorted_filtered(&models, &search, &filter, &sort, |e| {
        model_entry_json(e, resident)
    });
    let loras_json = sorted_filtered(&loras, &lora_search, &lora_filter, &lora_sort, |e| {
        lora_entry_json(e, &models, &selected_model, &selected_arch)
    });
    let artifacts_json = Value::Array(artifacts.iter().map(registry_artifact_json).collect());

    let doc = json!({
        "schema": "serenity.models.v1",
        "query": query,
        "models_total": models.len(),
        "loras_total": loras.len(),
        "artifacts_total": artifacts.len(),
        "model_selected": selected_model,
        "model_selected_arch": selected_arch,
        "model_type_options": model_type_options_json(),
        "models": models_json,
        "loras": loras_json,
        "artifacts": artifacts_json,
        "ltx2_features": ltx2_feature_documents(),
    });

    (
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}

fn model_type_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&json!({ "error": message.into() }))
            .unwrap_or_else(|_| String::from(r#"{"error":"model type update failed"}"#)),
    )
        .into_response()
}

/// POST /v1/models/type — persist or reset the architecture assigned to one
/// exact checkpoint or LoRA registry identity. The server still controls which
/// architecture loaders exist; this endpoint changes classification/routing,
/// never runtime capability claims.
pub async fn post_model_type_override(Json(payload): Json<ModelTypeOverrideRequest>) -> Response {
    let model = payload.model.trim();
    if model.is_empty() {
        return model_type_error(StatusCode::BAD_REQUEST, "model is required");
    }

    let kind = payload
        .kind
        .as_deref()
        .unwrap_or("checkpoint")
        .trim()
        .to_ascii_lowercase();
    let is_lora = kind == "lora";
    if !is_lora && !matches!(kind.as_str(), "checkpoint" | "unet" | "model") {
        return model_type_error(
            StatusCode::BAD_REQUEST,
            format!("unsupported registry kind {kind:?}; expected checkpoint or lora"),
        );
    }
    let entries = if is_lora {
        scan_loras()
    } else {
        scan_checkpoints_uncached()
    };
    let Some(target) = entries
        .iter()
        .find(|entry| entry.name == model && Path::new(&entry.path).is_file())
    else {
        return model_type_error(
            StatusCode::NOT_FOUND,
            format!(
                "{} registry model not found: {model}",
                if is_lora { "LoRA" } else { "checkpoint" }
            ),
        );
    };
    let model_name = target.name.clone();
    let override_key = if is_lora {
        lora_type_override_key(&model_name)
    } else {
        model_name.clone()
    };
    let requested = payload.arch.as_deref().unwrap_or("").trim();
    let reset = requested.is_empty()
        || matches!(
            requested.to_ascii_lowercase().as_str(),
            "auto" | "automatic" | "detected"
        );
    let arch_override = if reset {
        None
    } else {
        match normalize_user_model_type(requested) {
            Some(arch) => Some(arch),
            None => {
                let supported = MODEL_TYPE_OPTIONS
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<Vec<_>>()
                    .join(", ");
                return model_type_error(
                    StatusCode::BAD_REQUEST,
                    format!("unsupported model type {requested:?}; choose one of: {supported}"),
                );
            }
        }
    };

    {
        let _guard = match model_type_override_write_lock().lock() {
            Ok(guard) => guard,
            Err(_) => {
                return model_type_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "model type override lock is poisoned",
                );
            }
        };
        let root = model_root();
        let mut overrides = match load_model_type_overrides_from(&root) {
            Ok(overrides) => overrides,
            Err(error) => {
                return model_type_error(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
        };
        if let Some(arch) = arch_override.as_ref() {
            overrides.insert(override_key.clone(), arch.clone());
        } else {
            overrides.remove(&override_key);
        }
        if let Err(error) = persist_model_type_overrides_to(&root, &overrides) {
            return model_type_error(StatusCode::INTERNAL_SERVER_ERROR, error);
        }
    }

    invalidate_checkpoint_scan_cache();
    let updated_entries = if is_lora {
        scan_loras()
    } else {
        scan_checkpoints_with_refresh(true)
    };
    let Some(updated) = updated_entries
        .into_iter()
        .find(|entry| entry.name == model_name)
    else {
        return model_type_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "model disappeared while saving its type",
        );
    };
    let updated_json = if is_lora {
        let models = scan_checkpoints();
        lora_entry_json(&updated, &models, "", "")
    } else {
        model_entry_json(&updated, "")
    };
    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&json!({
            "schema": "serenity.model_type_override.result.v1",
            "reset": reset,
            "kind": if is_lora { "lora" } else { "checkpoint" },
            "model": updated_json,
            "model_type_options": model_type_options_json(),
        }))
        .unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}

// ── public name accessors (for the ComfyUI /object_info + /models/loras adapters) ──

/// Checkpoint + known-diffusers-dir model NAMES from the same disk scan `/v1/models`
/// uses. Fed into the ComfyUI `/object_info` combo lists (`ckpt_name`/`unet_name`) so
/// the canvas model dropdown is populated. Names match what `model_family` classifies.
pub fn checkpoint_names() -> Vec<String> {
    scan_checkpoints().into_iter().map(|e| e.name).collect()
}

/// Resolve a UI/API model identity to the exact artifact discovered under the
/// configured registry root. Absolute paths are accepted only when they equal a
/// scanned entry, preventing the generate API from becoming an arbitrary-file
/// reader.
pub(crate) fn resolve_checkpoint(selection: &str) -> Option<ResolvedCheckpoint> {
    let wanted = selection.trim();
    let wanted_without_ext = wanted.strip_suffix(".safetensors").unwrap_or(wanted);
    scan_checkpoints().into_iter().find_map(|entry| {
        let entry_without_ext = entry
            .name
            .strip_suffix(".safetensors")
            .unwrap_or(&entry.name);
        if entry.name == wanted || entry_without_ext == wanted_without_ext || entry.path == wanted {
            Some(ResolvedCheckpoint {
                name: entry.name,
                path: PathBuf::from(entry.path),
                arch: entry.arch,
                arch_source: entry.arch_source,
                arch_override: entry.arch_override,
                format: entry.format,
            })
        } else {
            None
        }
    })
}

pub(crate) fn architecture_for_model(selection: &str) -> Option<String> {
    resolve_checkpoint(selection).map(|entry| entry.arch)
}

/// LoRA NAMES from the disk scan. ComfyUI `/models/loras` is a bare string array.
pub fn lora_names() -> Vec<String> {
    scan_loras()
        .into_iter()
        .filter(|entry| lora_selectable_as_overlay(&entry.name))
        .map(|entry| entry.name)
        .collect()
}

/// Auxiliary model artifacts from the same recursive registry inventory used
/// by the Models tab. Names remain relative to their category root so nested
/// user organization is preserved.
pub fn artifact_name_inventory() -> BTreeMap<String, Vec<String>> {
    let mut inventory = BTreeMap::<String, Vec<String>>::new();
    for artifact in scan_registry_artifacts() {
        inventory
            .entry(artifact.artifact_type)
            .or_default()
            .push(artifact.name);
    }
    inventory
}

/// Resolve a registry LoRA by its API name (with or without `.safetensors`) or
/// exact scanned path. Video request preflight uses this before taking the GPU
/// lease so missing and cross-architecture adapters never reach CUDA.
pub fn lora_path_and_arch(name: &str) -> Option<(PathBuf, String)> {
    if !lora_selectable_as_overlay(name) {
        return None;
    }
    let wanted = name.strip_suffix(".safetensors").unwrap_or(name);
    scan_loras().into_iter().find_map(|entry| {
        if entry.name == name || entry.name == wanted || entry.path == name {
            Some((PathBuf::from(entry.path), entry.arch))
        } else {
            None
        }
    })
}

// ── DELETE /models/{type}/{name} — guarded single-file removal ────────────────────

/// Map a URL `{type}` segment to its scan-root dir. Accepts the singular + plural
/// forms the frontend might send. `None` = unknown type → 400.
fn models_root_for_type(mtype: &str) -> Option<PathBuf> {
    match mtype.to_lowercase().as_str() {
        "checkpoint" | "checkpoints" | "model" | "models" => Some(model_root().join("checkpoints")),
        "lora" | "loras" => Some(model_root().join("loras")),
        _ => None,
    }
}

/// DELETE /models/{type}/{name} — delete ONE `<name>.safetensors` directly under the
/// checkpoints/ or loras/ root. Strictly guarded: known type, a bare filename (no
/// `/`, no `..`), a `.safetensors` extension, and an existing regular file inside the
/// root. Returns `{deleted}`; 400 on a bad type/name, 404 when the file is absent.
/// Multi-shard model DIRS are intentionally NOT deletable here (files only).
pub async fn delete_model(AxPath((mtype, name)): AxPath<(String, String)>) -> Response {
    let root = match models_root_for_type(&mtype) {
        Some(r) => r,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                [(CONTENT_TYPE, "application/json")],
                serde_json::to_string(&json!({ "error": format!("unknown model type: {mtype}") }))
                    .unwrap(),
            )
                .into_response();
        }
    };
    // filename guard: single path component, no traversal, .safetensors only.
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || !name.ends_with(".safetensors")
    {
        return (
            StatusCode::BAD_REQUEST,
            [(CONTENT_TYPE, "application/json")],
            serde_json::to_string(&json!({ "error": "invalid model name" })).unwrap(),
        )
            .into_response();
    }
    let p = root.join(&name);
    match std::fs::symlink_metadata(&p) {
        // symlink_metadata: do NOT follow a symlink OUT of the root before checking;
        // we only ever unlink the entry named directly in the root.
        Ok(m) if m.file_type().is_file() || m.file_type().is_symlink() => {}
        _ => {
            return (
                StatusCode::NOT_FOUND,
                [(CONTENT_TYPE, "application/json")],
                serde_json::to_string(&json!({ "error": format!("model not found: {name}") }))
                    .unwrap(),
            )
                .into_response();
        }
    }
    match std::fs::remove_file(&p) {
        Ok(_) => (
            StatusCode::OK,
            [(CONTENT_TYPE, "application/json")],
            serde_json::to_string(&json!({ "deleted": name })).unwrap(),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(CONTENT_TYPE, "application/json")],
            serde_json::to_string(&json!({ "error": format!("delete failed: {e}") })).unwrap(),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Filename probes remain a compatibility fallback for legacy files without
    // useful metadata or recognizable tensor keys.
    #[test]
    fn arch_name_precedence_known_families() {
        assert_eq!(detect_arch_from_name("flux-2-klein-base-9b"), "flux-2");
        assert_eq!(detect_arch_from_name("sd_xl_base_1.0"), "sdxl");
        assert_eq!(
            detect_arch_from_name("wan2.2_t2v_low_noise_14b_fp16"),
            "wan2.2"
        );
        assert_eq!(
            detect_arch_from_name("qwen_2.5_vl_7b_fp8_scaled"),
            "qwen-image"
        );
        assert_eq!(detect_arch_from_name("ltx-2.3-22b-dev"), "ltx2");
        assert_eq!(detect_arch_from_name("EriCrhroma_000004800"), "chroma");
        assert_eq!(detect_arch_from_name("NAVA/NAVA_fp8"), "nava");
        assert_eq!(detect_arch_from_name("Bernini-R-Diffusers"), "bernini");
        assert_eq!(detect_arch_from_name("SCAIL-2-Mojo"), "scail2");
        assert_eq!(detect_arch_from_name("MiniMax-H3-Mojo"), "minimax-h3");
        assert_eq!(detect_arch_from_name("some-random-checkpoint"), "unknown");
    }

    #[test]
    fn arch_header_probes() {
        assert_eq!(detect_arch("...\"noise_refiner.x\": ..."), "zimage");
        assert_eq!(
            detect_arch("...\"double_stream_modulation_img\": ..."),
            "flux-2/klein"
        );
        assert_eq!(
            detect_arch(
                "...\"backbone.double_blocks.0.self_attn.q_audio.bias\":\
                 ...\"backbone.single_blocks.0.self_attn.q.bias\":..."
            ),
            "nava"
        );
        assert_eq!(detect_arch("...\"time_projection.\": ..."), "wan2.2");
        assert_eq!(detect_arch("{}"), "unknown");
        assert_eq!(detect_lora_target_arch("...\"lora_unet_x\": ..."), "sdxl");
        assert_eq!(
            detect_lora_target_arch(
                r#"{"transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight":{},
                    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight":{}}"#
            ),
            "flux"
        );
        assert_eq!(
            detect_lora_target_arch(
                r#"{"diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight":{}}"#
            ),
            "ltx2"
        );
        assert_eq!(
            detect_lora_target_arch(
                r#"{"lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w1":{},
                    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w2":{}}"#
            ),
            "sd3"
        );
        assert_eq!(
            detect_lora_target_arch(
                r#"{"transformer.transformer_blocks.0.attn1.to_q.lora_A.weight":{},
                    "transformer.transformer_blocks.0.attn2.to_k.lora_B.weight":{},
                    "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight":{}}"#
            ),
            "anima"
        );
        assert_eq!(
            detect_lora_target_arch(
                r#"{"transformer.transformer_blocks.0.attn.to_q.lora_A.weight":{},
                    "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight":{},
                    "transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight":{}}"#
            ),
            "qwen-image"
        );
        assert_eq!(
            detect_lora_target_arch(
                r#"{"transformer.layers.0.attention.qkv.lora_A.weight":{},
                    "transformer.layers.0.feed_forward.w1.lora_B.weight":{}}"#
            ),
            "ideogram4"
        );
        assert_eq!(
            detect_checkpoint_format(
                r#"{"model.diffusion_model.pos_embed":{"shape":[1,36864,2432]}}"#,
                "sd3"
            ),
            "sd3_large_full_checkpoint"
        );
        assert_eq!(
            detect_checkpoint_format(
                r#"{"model.diffusion_model.pos_embed":{"shape":[1,147456,1536]}}"#,
                "sd3"
            ),
            "sd3_medium_full_checkpoint"
        );
    }

    #[test]
    fn modelspec_architecture_beats_random_creator_filename() {
        assert_eq!(
            detect_arch(
                r#"{"__metadata__":{"modelspec.architecture":"stable-diffusion-xl-v1-base"}}"#
            ),
            "sdxl"
        );
        assert_eq!(
            detect_arch(r#"{"__metadata__":{"modelspec.architecture":"Krea 2"}}"#),
            "krea2"
        );
        assert_eq!(normalize_architecture_id("Pony Diffusion V6 XL"), "sdxl");
        assert_eq!(
            normalize_architecture_id("stable-diffusion-v3.5-large"),
            "sd3"
        );
    }

    #[test]
    fn model_type_override_values_are_normalized_and_bounded() {
        assert_eq!(
            normalize_user_model_type("Pony Diffusion V6 XL"),
            Some("sdxl".to_string())
        );
        assert_eq!(
            normalize_user_model_type("FLUX-2 Klein"),
            Some("flux-2/klein".to_string())
        );
        assert_eq!(
            normalize_user_model_type("LensPipeline"),
            Some("lens".to_string())
        );
        assert_eq!(
            normalize_user_model_type("Baidu NAVA"),
            Some("nava".to_string())
        );
        assert_eq!(normalize_user_model_type("made-up-runtime"), None);
    }

    #[test]
    fn model_type_override_file_round_trips_and_reset_removes_it() {
        let unique = format!(
            "serenity_model_type_override_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let mut overrides = BTreeMap::new();
        overrides.insert("portraits/cyber-pony".to_string(), "sdxl".to_string());
        overrides.insert(
            lora_type_override_key("styles/custom-adapter"),
            "chroma".to_string(),
        );
        persist_model_type_overrides_to(&root, &overrides).unwrap();
        assert_eq!(load_model_type_overrides_from(&root).unwrap(), overrides);

        persist_model_type_overrides_to(&root, &BTreeMap::new()).unwrap();
        assert!(!model_type_overrides_path(&root).exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_detection_reports_automatic_source() {
        assert_eq!(
            detect_checkpoint_arch(
                r#"{"__metadata__":{"modelspec.architecture":"stable-diffusion-xl-v1-base"}}"#,
                "Krea 2",
                "krea-name"
            ),
            ("sdxl".to_string(), "metadata".to_string())
        );
        assert_eq!(
            detect_checkpoint_arch("{}", "Pony Diffusion V6 XL", "random-name"),
            ("sdxl".to_string(), "sidecar".to_string())
        );
    }

    #[test]
    fn arbitrary_checkpoints_are_enabled_only_when_the_worker_uses_the_file() {
        let entry = |name: &str, arch: &str, format: &str| ScanEntry {
            name: name.to_string(),
            path: format!("/models/{name}.safetensors"),
            arch: arch.to_string(),
            detected_arch: arch.to_string(),
            arch_source: "tensor_signature".to_string(),
            arch_override: String::new(),
            format: format.to_string(),
            size: 1,
            folder: String::new(),
            sidecar: Sidecar::default(),
        };

        let arbitrary_sdxl =
            model_entry_json(&entry("creator-name", "sdxl", "full_checkpoint"), "");
        assert_eq!(arbitrary_sdxl["runtime_supported"], true);
        assert_eq!(arbitrary_sdxl["uses_selected_checkpoint"], true);
        assert_eq!(arbitrary_sdxl["selected_checkpoint_scope"], "denoiser");
        assert_eq!(arbitrary_sdxl["generation_defaults"]["steps"], 50);

        let arbitrary_qwen =
            model_entry_json(&entry("creator-name", "qwen-image", "diffusion_model"), "");
        assert_eq!(arbitrary_qwen["runtime_supported"], true);
        assert_eq!(arbitrary_qwen["uses_selected_checkpoint"], true);
        assert_eq!(arbitrary_qwen["selected_checkpoint_scope"], "denoiser");

        let qwen_edit = model_entry_json(
            &entry("qwen-image-edit-2511", "qwen-image", "diffusion_model"),
            "",
        );
        assert_eq!(qwen_edit["runtime_supported"], false);
        assert_eq!(qwen_edit["uses_selected_checkpoint"], false);

        let bundled_flux = model_entry_json(&entry("flux1-dev", "flux", "diffusion_model"), "");
        assert_eq!(bundled_flux["runtime_supported"], true);
        assert_eq!(bundled_flux["uses_selected_checkpoint"], true);
        assert_eq!(bundled_flux["selected_checkpoint_scope"], "denoiser");

        for arch in ["chroma", "anima"] {
            let arbitrary = model_entry_json(&entry("creator-name", arch, "diffusion_model"), "");
            assert_eq!(arbitrary["runtime_supported"], true, "{arch}");
            assert_eq!(arbitrary["uses_selected_checkpoint"], true, "{arch}");
            assert_eq!(arbitrary["selected_checkpoint_scope"], "denoiser", "{arch}");
        }

        let arbitrary_sd3 = model_entry_json(
            &entry("creator-name", "sd3", "sd3_large_full_checkpoint"),
            "",
        );
        assert_eq!(arbitrary_sd3["runtime_supported"], true);
        assert_eq!(arbitrary_sd3["uses_selected_checkpoint"], true);
        assert_eq!(
            arbitrary_sd3["selected_checkpoint_scope"],
            "full_checkpoint"
        );

        let bundled_lens =
            model_entry_json(&entry("microsoft_lens", "lens", "diffusers_directory"), "");
        assert_eq!(bundled_lens["runtime_supported"], true);
        assert_eq!(bundled_lens["selected_checkpoint_scope"], "bundled_profile");
        assert_eq!(bundled_lens["generation_defaults"]["steps"], 20);
        assert_eq!(bundled_lens["card"]["generation_defaults"]["cfg"], 5.0);

        let arbitrary_zimage_turbo = model_entry_json(
            &entry("z_image_turbo_bf16", "zimage", "diffusion_model"),
            "",
        );
        assert_eq!(arbitrary_zimage_turbo["runtime_supported"], true);
        assert_eq!(arbitrary_zimage_turbo["uses_selected_checkpoint"], true);
        assert_eq!(
            arbitrary_zimage_turbo["selected_checkpoint_scope"],
            "denoiser"
        );
        assert_eq!(arbitrary_zimage_turbo["generation_defaults"]["steps"], 8);

        let arbitrary_zimage = model_entry_json(
            &entry("creator-custom-name", "zimage", "diffusion_model"),
            "",
        );
        assert_eq!(arbitrary_zimage["runtime_supported"], true);
        assert_eq!(arbitrary_zimage["uses_selected_checkpoint"], true);

        let blocked_l2p = model_entry_json(
            &entry("L2P/model-1k-merge", "zimage", "diffusion_model"),
            "",
        );
        assert_eq!(blocked_l2p["runtime_supported"], false);
        assert!(blocked_l2p["runtime_reason"]
            .as_str()
            .unwrap_or("")
            .contains("no production serenity-server worker route"));

        let ltx_product =
            model_entry_json(&entry("ltx-2.3-22b-dev-fp8", "ltx2", "diffusion_model"), "");
        assert_eq!(ltx_product["runtime_supported"], true);
        assert_eq!(ltx_product["uses_selected_checkpoint"], true);
        assert_eq!(ltx_product["selected_checkpoint_scope"], "video_denoiser");
        let ltx_bf16_product = model_entry_json(
            &entry(
                "ltx-2.3-22b-dev-fp8-dequant-bf16",
                "ltx2",
                "diffusion_model",
            ),
            "",
        );
        assert_eq!(ltx_bf16_product["runtime_supported"], true);
        assert_eq!(
            ltx_bf16_product["selected_checkpoint_scope"],
            "video_denoiser"
        );

        let arbitrary_ltx = model_entry_json(
            &entry("creator-full-finetune", "ltx2", "diffusion_model"),
            "",
        );
        assert_eq!(arbitrary_ltx["runtime_supported"], true);
        assert_eq!(arbitrary_ltx["uses_selected_checkpoint"], true);
        assert_eq!(arbitrary_ltx["selected_checkpoint_scope"], "video_denoiser");

        let mut user_classified_ltx = entry("sulphur_dev_bf16", "ltx2", "diffusion_model");
        user_classified_ltx.detected_arch = "unknown".to_string();
        user_classified_ltx.arch_source = "user_override".to_string();
        user_classified_ltx.arch_override = "ltx2".to_string();
        let user_classified_ltx = model_entry_json(&user_classified_ltx, "");
        assert_eq!(user_classified_ltx["runtime_supported"], true);
        assert_eq!(
            user_classified_ltx["selected_checkpoint_scope"],
            "video_denoiser"
        );

        let source_wan = model_entry_json(
            &entry("Wan2.2-TI2V-5B", "wan2.2", "diffusers_directory"),
            "",
        );
        assert_eq!(source_wan["runtime_supported"], false);
        let product_wan = model_entry_json(
            &entry("Wan2.2-TI2V-5B-Mojo", "wan2.2", "diffusers_directory"),
            "",
        );
        assert_eq!(product_wan["runtime_supported"], true);

        let nava = model_entry_json(&entry("NAVA/NAVA_fp8", "nava", "diffusion_model"), "");
        assert_eq!(nava["generation_route"], "video");
        assert_eq!(nava["runtime_supported"], false);
        assert!(nava["runtime_reason"]
            .as_str()
            .unwrap()
            .contains("not one of the compiled video product profiles"));

        let mut classified_video = entry("creator-video", "ltx2", "diffusion_model");
        classified_video.detected_arch = "unknown".to_string();
        classified_video.arch_source = "user_override".to_string();
        classified_video.arch_override = "ltx2".to_string();
        let classified_video = model_entry_json(&classified_video, "");
        assert_eq!(classified_video["generation_route"], "video");
        assert_eq!(classified_video["runtime_supported"], true);
        assert_eq!(classified_video["uses_selected_checkpoint"], true);
        assert_eq!(
            classified_video["selected_checkpoint_scope"],
            "video_denoiser"
        );
    }

    #[test]
    fn checkpoint_support_weights_are_classified_as_components() {
        assert_eq!(
            checkpoint_component_type("ltx-2.3-22b-dev-fp8.safetensors"),
            None
        );
        assert_eq!(
            checkpoint_component_type("ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors"),
            None
        );
        assert_eq!(
            checkpoint_component_type("ltx-2.3-22b-distilled-fp8.safetensors"),
            None
        );
        assert_eq!(
            checkpoint_component_type("ltx-2.3-22b-distilled-lora-384.safetensors"),
            Some("runtime_component")
        );
        assert_eq!(
            checkpoint_component_type("ltx-2.3-22b-svdint4-r32.safetensors"),
            Some("runtime_component")
        );
        assert_eq!(
            checkpoint_component_type("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            Some("upscaler")
        );
        assert_eq!(
            checkpoint_component_type("NAVA/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"),
            Some("vae")
        );
        assert_eq!(
            checkpoint_component_type("NAVA/umt5_xxl_enc.safetensors"),
            Some("clip")
        );
        assert_eq!(
            checkpoint_component_type("creator/cyberrealisticPony.safetensors"),
            None
        );
    }

    #[test]
    fn diffusers_bundles_are_discovered_recursively_and_lens_is_detected() {
        let unique = format!(
            "serenity_diffusers_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let lens = root.join("creators/microsoft_lens");
        std::fs::create_dir_all(lens.join("transformer")).unwrap();
        std::fs::write(
            lens.join("model_index.json"),
            br#"{"_class_name":"LensPipeline"}"#,
        )
        .unwrap();
        std::fs::write(lens.join("transformer/config.json"), b"{}").unwrap();

        let dirs = discover_model_index_dirs(&root);
        assert_eq!(dirs, vec![lens.clone()]);
        assert_eq!(
            diffusers_directory_identity(&root, &lens),
            "creators/microsoft_lens"
        );
        assert_eq!(
            diffusers_directory_arch(&lens, "creators/microsoft_lens"),
            "lens"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_scan_recurses_category_folders_and_skips_components() {
        let unique = format!(
            "serenity_model_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        let portraits = root.join("portraits");
        let text_encoder = root.join("bundle/text_encoder");
        std::fs::create_dir_all(&portraits).unwrap();
        std::fs::create_dir_all(&text_encoder).unwrap();
        std::fs::write(portraits.join("random-name.safetensors"), b"fixture").unwrap();
        std::fs::write(text_encoder.join("model.safetensors"), b"component").unwrap();
        std::fs::write(root.join("model-00001-of-00002.safetensors"), b"shard").unwrap();

        let entries = list_safetensors(root.to_str().unwrap());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "portraits/random-name");
        assert_eq!(entries[0].folder, "");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn external_checkpoint_root_accepts_direct_user_model_folder() {
        let unique = format!(
            "serenity_external_model_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&root).unwrap();
        let checkpoint = root.join("creator-custom-model.safetensors");
        std::fs::write(&checkpoint, b"fixture").unwrap();

        assert_eq!(
            external_checkpoint_scan_roots(&root),
            vec![(String::new(), root.clone())]
        );
        let mut entries = Vec::new();
        append_checkpoint_files(
            &mut entries,
            &root,
            "",
            "external-test",
            &BTreeMap::new(),
            &mut HashSet::new(),
            &mut HashSet::new(),
        );
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "creator-custom-model");
        assert_eq!(Path::new(&entries[0].path), checkpoint);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lora_scan_keeps_lora_named_files_in_subdirectories() {
        let unique = format!(
            "serenity_lora_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(root.join("LTX/features")).unwrap();
        std::fs::write(
            root.join("LTX/features/reference-lora.safetensors"),
            b"fixture",
        )
        .unwrap();

        assert!(list_safetensors(root.to_str().unwrap()).is_empty());
        let entries = list_lora_safetensors(root.to_str().unwrap());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "LTX/features/reference-lora");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn auxiliary_artifact_scan_recurses_and_skips_cache_noise() {
        let unique = format!(
            "serenity_artifact_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(root.join("family")).unwrap();
        std::fs::create_dir_all(root.join(".cache")).unwrap();
        std::fs::write(root.join("family/decoder.safetensors"), b"weights").unwrap();
        std::fs::write(root.join("legacy.pth"), b"weights").unwrap();
        std::fs::write(root.join("notes.json"), b"metadata").unwrap();
        std::fs::write(root.join(".cache/duplicate.safetensors"), b"cache").unwrap();

        let mut artifacts = Vec::new();
        scan_registry_artifact_dir(
            "vae",
            &root,
            &root,
            &mut HashSet::new(),
            &mut HashSet::new(),
            &mut artifacts,
        );
        artifacts.sort_by(|left, right| left.name.cmp(&right.name));
        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0].name, "family/decoder.safetensors");
        assert_eq!(artifacts[0].folder, "family");
        assert_eq!(artifacts[0].artifact_type, "vae");
        assert_eq!(artifacts[1].name, "legacy.pth");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ltx2_feature_artifacts_are_not_plain_loras() {
        assert_eq!(
            lora_usage("ltx-2.3-22b-ic-lora-ingredients-0.9"),
            "ic_lora_feature"
        );
        assert_eq!(
            lora_usage("ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"),
            "companion_embedding"
        );
        assert_eq!(
            lora_usage("ltx-2.3-22b-lora-foley-v2a-1.0.safetensors"),
            "v2a_feature"
        );
        assert_eq!(
            lora_usage("ltx-2.3-22b-lora-cinemagraph-0.9.safetensors"),
            "overlay"
        );
        assert_eq!(lora_usage("eri2_krea2_v2_2000.safetensors"), "overlay");
    }

    #[test]
    fn ltx2_feature_registry_is_complete_and_unique() {
        let adapters = ltx2_feature_adapter_registry()["adapters"]
            .as_array()
            .expect("adapters");
        assert_eq!(adapters.len(), 19);
        let mut filenames = std::collections::HashSet::new();
        let mut ids = std::collections::HashSet::new();
        for adapter in adapters {
            assert!(ids.insert(adapter["id"].as_str().expect("id")));
            assert!(filenames.insert(adapter["filename"].as_str().expect("filename")));
        }
    }

    #[test]
    fn compat_and_reason() {
        assert!(model_lora_compatible("sdxl", "sdxl"));
        assert!(!model_lora_compatible("sdxl", "flux"));
        assert!(!model_lora_compatible("unknown", "sdxl"));
        assert_eq!(
            lora_incompatible_reason("", "", "sdxl", false),
            "no model selected"
        );
        assert_eq!(
            lora_incompatible_reason("m", "flux", "sdxl", false),
            "target_arch sdxl is not compatible with model arch flux"
        );
    }

    #[test]
    fn lora_inventory_order_is_case_insensitive_and_stable() {
        let make = |name: &str, path: &str| ScanEntry {
            name: name.to_string(),
            path: path.to_string(),
            arch: "flux".to_string(),
            detected_arch: "flux".to_string(),
            arch_source: "tensor_signature".to_string(),
            arch_override: String::new(),
            format: "lora".to_string(),
            size: 1,
            folder: String::new(),
            sidecar: Sidecar::default(),
        };
        let mut entries = vec![
            make("zeta", "/loras/zeta.safetensors"),
            make("Alpha", "/loras/Alpha.safetensors"),
            make("alpha", "/loras/alpha.safetensors"),
            make("beta", "/loras/beta.safetensors"),
        ];
        entries.sort_by(|left, right| scan_entry_cmp(left, right, "name"));
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Alpha", "alpha", "beta", "zeta"]
        );
    }

    // ── sidecar / preview / folder additions ─────────────────────────────────

    #[test]
    fn base64_matches_rfc4648() {
        // standard vectors (incl. all three padding cases)
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        // a byte with the high bits set, exercising the + / chars region
        assert_eq!(base64_encode(&[0xff, 0xff, 0xff]), "////");
        assert_eq!(base64_encode(&[0xfb]), "+w==");
    }

    #[test]
    fn folder_relative_strips_root_and_basename() {
        let root = "/portable/models/checkpoints";
        // direct child → no folder
        assert_eq!(
            folder_relative_to(&format!("{root}/x.safetensors"), root),
            ""
        );
        // one subdir
        assert_eq!(
            folder_relative_to(&format!("{root}/ltx-video/x.safetensors"), root),
            "ltx-video"
        );
        // nested subdir
        assert_eq!(
            folder_relative_to(&format!("{root}/a/b/x.safetensors"), root),
            "a/b"
        );
        // path not under root → ""
        assert_eq!(folder_relative_to("/other/x.safetensors", root), "");
    }

    #[test]
    fn image_mime_known_exts() {
        assert_eq!(image_mime("png"), "image/png");
        assert_eq!(image_mime("jpg"), "image/jpeg");
        assert_eq!(image_mime("jpeg"), "image/jpeg");
        assert_eq!(image_mime("webp"), "image/webp");
        assert_eq!(image_mime("txt"), "");
    }

    #[test]
    fn sidecar_json_flat_and_civitai() {
        // flat generic shape
        let flat = serde_json::json!({
            "description": "a portrait lora",
            "trigger": "ohwx person",
            "baseModel": "sdxl"
        });
        let s = parse_sidecar_json(&flat);
        assert_eq!(s.description, "a portrait lora");
        assert_eq!(s.trigger, "ohwx person");
        assert_eq!(s.arch_hint, "sdxl");

        // Civitai .civitai.info shape (trainedWords[], nested model.description)
        let civ = serde_json::json!({
            "trainedWords": ["ohwx", " person "],
            "baseModel": "SDXL 1.0",
            "model": { "description": "civitai desc" }
        });
        let c = parse_sidecar_json(&civ);
        assert_eq!(c.description, "civitai desc");
        assert_eq!(c.trigger, "ohwx, person");
        assert_eq!(c.arch_hint, "SDXL 1.0");

        // empty object → all empty (no panic)
        let e = parse_sidecar_json(&serde_json::json!({}));
        assert!(e.description.is_empty() && e.trigger.is_empty() && e.arch_hint.is_empty());
    }

    #[test]
    fn preview_data_uri_roundtrip_tmpfile() {
        // a tiny valid-enough PNG payload (bytes are arbitrary; we only encode)
        let dir = std::env::temp_dir().join(format!("serenity_mb_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let png = dir.join("m.preview.png");
        std::fs::write(&png, [0x89u8, b'P', b'N', b'G', 1, 2, 3]).unwrap();
        let uri = preview_data_uri(&png);
        assert!(uri.starts_with("data:image/png;base64,"), "got: {uri}");
        assert_eq!(
            &uri["data:image/png;base64,".len()..],
            base64_encode(&[0x89, b'P', b'N', b'G', 1, 2, 3])
        );

        // unsupported ext → ""
        let txt = dir.join("m.txt");
        std::fs::write(&txt, b"hi").unwrap();
        assert_eq!(preview_data_uri(&txt), "");

        // missing file → ""
        assert_eq!(preview_data_uri(&dir.join("nope.png")), "");

        // find_sidecar_preview prefers <model>.preview.png for a model file path
        let model = dir.join("m.safetensors");
        std::fs::write(&model, b"x").unwrap();
        let found = find_sidecar_preview(model.to_str().unwrap());
        assert_eq!(found, uri);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
