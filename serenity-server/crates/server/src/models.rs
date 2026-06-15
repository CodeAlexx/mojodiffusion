//! GET /v1/models — faithful port of the daemon's model/LoRA browser endpoint
//! (serenity_daemon.mojo @2923 + serve/model_scan.mojo + the card builders @2646-2904).
//!
//! Scans /home/alex/.serenity/models/{checkpoints,loras}/*.safetensors (header reads
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

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use axum::extract::Query;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

const CHECKPOINTS_DIR: &str = "/home/alex/.serenity/models/checkpoints";
const LORAS_DIR: &str = "/home/alex/.serenity/models/loras";
const HEADER_PROBE_CAP: u64 = 16 * 1024 * 1024;
/// Cap on an inlined sidecar preview image (encoded as a `data:` URI). Sidecar
/// previews are meant to be small thumbnails; oversize files are skipped rather
/// than bloating the /v1/models JSON. 2 MiB raw → ~2.7 MiB base64.
const PREVIEW_INLINE_CAP: u64 = 2 * 1024 * 1024;

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

struct ScanEntry {
    name: String,
    path: String,
    arch: String,
    size: i64,
    /// Subdir of the entry RELATIVE to its scan root ("" = top level). Lets the
    /// browser show a folder tree without guessing the root from the abs path.
    folder: String,
    /// Distilled sidecar preview/metadata (empty `Sidecar::default()` if none).
    sidecar: Sidecar,
}

// ── arch detection (exact substring probes from model_scan.mojo) ────────────────

fn detect_arch(header: &str) -> &'static str {
    if header.contains("\"noise_refiner.") {
        "zimage"
    } else if header.contains("\"double_stream_modulation_img") {
        "flux-2/klein"
    } else if header.contains("\"distilled_guidance_layer") {
        "chroma"
    } else if header.contains("\"double_blocks.") {
        "flux"
    } else if header.contains("\"audio_vae.") {
        "ltx2"
    } else if header.contains("\"embed_image_indicator.weight\"") || header.contains("\"llm_cond_proj.weight\"") {
        "ideogram4"
    } else if header.contains("\"model.diffusion_model.joint_blocks") {
        "sd3"
    } else if header.contains("\"model.diffusion_model.input_blocks") {
        "sdxl"
    } else if header.contains("\"input_blocks.0.") {
        "sdxl"
    } else if header.contains("\"txt_norm.") {
        "qwen-image"
    } else if header.contains("\"time_projection.") {
        "wan"
    } else {
        "unknown"
    }
}

fn detect_arch_from_name(name: &str) -> &'static str {
    let lo = name.to_lowercase();
    let c = |s: &str| lo.contains(s);
    if c("wan2.2") || c("wan 2.2") || c("wan-2.2") || c("wan_2_2") || c("wan22") {
        "wan2.2"
    } else if c("zimage_l2p") || c("z-image-l2p") || c("z_image_l2p") || c("l2p") {
        "zimage-l2p"
    } else if c("hidream") || c("hi-dream") || c("hi_dream") {
        "hidream"
    } else if c("sensenova") || c("sense_nova") || c("sense-nova") {
        "sensenova"
    } else if c("qwen") {
        "qwen-image"
    } else if c("sd3") || c("sd35") {
        "sd3"
    } else if c("sdxl") || c("stable-diffusion-xl") || c("animagine") {
        "sdxl"
    } else if c("flux2") || c("flux-2") || c("flux_2") {
        "flux-2"
    } else if c("flux1") || c("flux-1") || c("flux_1") || c("flux-dev") {
        "flux"
    } else if c("ltx") {
        "ltx2"
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
    } else if header.contains("\"lora_unet_") || header.contains("\"lora_te1") || header.contains("\"lora_te2") {
        "sdxl"
    } else if header.contains("\"lora_transformer_distilled_guidance_layer") {
        "chroma"
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
        out.push(if chunk.len() > 1 { TBL[((b1 & 0x0f) << 2) | (b2 >> 6)] as char } else { '=' });
        out.push(if chunk.len() > 2 { TBL[b2 & 0x3f] as char } else { '=' });
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
/// (full path incl. `.safetensors`). Probes the common SwarmUI/Civitai sidecar
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
        "preview.png", "preview.jpg", "cover.png", "cover.jpg", "teaser.jpg", "teaser.png",
        "thumbnail.png", "thumbnail.jpg",
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
        &["trigger", "trigger_words", "triggerWords", "activation text", "activation_text"],
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
    Sidecar { preview: String::new(), description, trigger, arch_hint }
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
    for fname in [format!("{stem}.json"), format!("{stem}.civitai.info"), format!("{stem}.cm-info.json")] {
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

/// `find -L <dir> -maxdepth 1 -type f -name '*.safetensors'` — symlinks followed,
/// arch left "". Order is irrelevant (the response re-sorts).
fn list_safetensors(dir: &str) -> Vec<ScanEntry> {
    let mut out = Vec::new();
    let rd = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return out,
    };
    for ent in rd.flatten() {
        let fname = ent.file_name().to_string_lossy().into_owned();
        if !fname.ends_with(".safetensors") {
            continue;
        }
        // -type f with -L: stat the target (follows symlinks).
        let meta = match std::fs::metadata(ent.path()) {
            Ok(m) if m.is_file() => m,
            _ => continue,
        };
        let stem = fname[..fname.len() - ".safetensors".len()].to_string();
        out.push(ScanEntry {
            name: stem,
            path: format!("{dir}/{fname}"),
            arch: String::new(),
            size: meta.len() as i64,
            folder: String::new(), // top-level scan: filled by caller if nested
            sidecar: Sidecar::default(),
        });
    }
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

fn scan_checkpoints() -> Vec<ScanEntry> {
    let mut out = Vec::new();
    for mut e in list_safetensors(CHECKPOINTS_DIR) {
        let mut arch = detect_arch_from_name(&e.name).to_string();
        if arch == "unknown" {
            arch = detect_arch(&header_text(&e.path)).to_string();
        }
        e.arch = arch;
        e.folder = folder_relative_to(&e.path, CHECKPOINTS_DIR);
        e.sidecar = sidecar_for_file(&e.path);
        // sidecar may rescue an unknown arch (only when it offers a hint).
        if e.arch == "unknown" && !e.sidecar.arch_hint.is_empty() {
            e.arch = e.sidecar.arch_hint.clone();
        }
        out.push(e);
    }
    // known diffusers-tree DIRS (arch by identity), size via du -sb.
    for (name, arch) in [
        ("zimage_base", "zimage"),
        ("anima", "anima"),
        ("ideogram-4-fp8", "ideogram4"),
    ] {
        let dir = format!("/home/alex/.serenity/models/{name}");
        if dir_exists(&dir) {
            let size = du_sb(&dir);
            out.push(ScanEntry {
                name: name.to_string(),
                path: dir.clone(),
                arch: arch.to_string(),
                size,
                folder: folder_relative_to(&dir, "/home/alex/.serenity/models"),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    // known multi-shard checkpoint subdirs under checkpoints/.
    for (name, arch) in [("qwen-image-2512", "qwen-image"), ("ideogram-4-fp8", "ideogram4")] {
        let dir = format!("{CHECKPOINTS_DIR}/{name}");
        if dir_exists(&dir) {
            let size = du_sb(&dir);
            out.push(ScanEntry {
                name: name.to_string(),
                path: dir.clone(),
                arch: arch.to_string(),
                size,
                folder: folder_relative_to(&dir, CHECKPOINTS_DIR),
                sidecar: sidecar_for_dir(&dir),
            });
        }
    }
    out
}

fn scan_loras() -> Vec<ScanEntry> {
    let mut out = Vec::new();
    for mut e in list_safetensors(LORAS_DIR) {
        let mut target = detect_arch_from_name(&e.name).to_string();
        if target == "unknown" {
            target = detect_lora_target_arch(&header_text(&e.path)).to_string();
        }
        e.arch = target;
        e.folder = folder_relative_to(&e.path, LORAS_DIR);
        e.sidecar = sidecar_for_file(&e.path);
        if e.arch == "unknown" && !e.sidecar.arch_hint.is_empty() {
            e.arch = e.sidecar.arch_hint.clone();
        }
        out.push(e);
    }
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
        "arch" | "family" => entry_arch(a).to_lowercase().cmp(&entry_arch(b).to_lowercase()),
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

fn lora_incompatible_reason(selected_model: &str, selected_arch: &str, target_arch: &str, compatible: bool) -> String {
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
    let loaded = !resident.is_empty() && e.name == resident;
    let preview = e.sidecar.preview.clone();
    let metadata = json!({
        "schema": "serenity.model.metadata.v1",
        "source": "disk_scan",
        "family": arch,
        "notes": e.sidecar.description,        // ADD: from <model>.json/.civitai.info
        "description": e.sidecar.description,  // ADD: alias for clarity
        "trigger": e.sidecar.trigger,          // ADD: sidecar trigger words (if any)
    });
    let card = json!({
        "schema": "serenity.model.card.v1",
        "title": e.name,
        "subtitle": arch,
        "path": e.path,
        "folder": e.folder,                    // ADD: subdir under scan root ("" = top)
        "size": e.size,
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "favorite": false,
        "loaded": loaded,
        "metadata": metadata,
    });
    json!({
        "name": e.name,
        "path": e.path,
        "folder": e.folder,                    // ADD: subdir under scan root ("" = top)
        "arch": arch,
        "size": e.size,
        "loaded": loaded,
        "type": "checkpoint",
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "trigger": e.sidecar.trigger,          // ADD: sidecar trigger words (if any)
        "favorite": false,
        "metadata": metadata,
        "card": card,
    })
}

fn lora_entry_json(e: &ScanEntry, models: &[ScanEntry], selected_model: &str, selected_arch: &str) -> Value {
    let target = entry_arch(e);
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
        "trigger": trigger,                    // populated from sidecar (was always "")
        "thumbnail": "",
        "preview": preview,                    // ADD: data: URI if a sidecar exists
        "favorite": false,
        "compatible_models": compatible_models_json(e, models),
        "compatible": compatible,
        "compatibility": compatibility,
        "incompatible_reason": reason,
        "metadata": metadata,
        "card": card,
    })
}

fn sorted_filtered<F>(entries: &[ScanEntry], search: &str, filter: &str, sort: &str, build: F) -> Value
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

    let models = scan_checkpoints();
    let loras = scan_loras();
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
    let models_json = sorted_filtered(&models, &search, &filter, &sort, |e| model_entry_json(e, resident));
    let loras_json = sorted_filtered(&loras, &lora_search, &lora_filter, &lora_sort, |e| {
        lora_entry_json(e, &models, &selected_model, &selected_arch)
    });

    let doc = json!({
        "schema": "serenity.models.v1",
        "query": query,
        "models_total": models.len(),
        "loras_total": loras.len(),
        "model_selected": selected_model,
        "model_selected_arch": selected_arch,
        "models": models_json,
        "loras": loras_json,
    });

    (
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Locks the arch probes + the NAME-PRECEDENCE scan order (HEAD model_scan.mojo:
    // arch = detect_arch_from_name(name); header only fills "unknown"). This is the
    // documented "disabled-family name tag" feature — and the one place the Rust
    // diverges from the STALE Jun-13 daemon binary, which predates this order and
    // tags header-first (flux-2-klein→"flux-2/klein", wan2.2→"wan"). A daemon rebuilt
    // from HEAD agrees with the Rust.
    #[test]
    fn arch_name_precedence_known_families() {
        assert_eq!(detect_arch_from_name("flux-2-klein-base-9b"), "flux-2");
        assert_eq!(detect_arch_from_name("wan2.2_t2v_low_noise_14b_fp16"), "wan2.2");
        assert_eq!(detect_arch_from_name("qwen_2.5_vl_7b_fp8_scaled"), "qwen-image");
        assert_eq!(detect_arch_from_name("ltx-2.3-22b-dev"), "ltx2");
        assert_eq!(detect_arch_from_name("some-random-checkpoint"), "unknown");
    }

    #[test]
    fn arch_header_probes() {
        assert_eq!(detect_arch("...\"noise_refiner.x\": ..."), "zimage");
        assert_eq!(detect_arch("...\"double_stream_modulation_img\": ..."), "flux-2/klein");
        assert_eq!(detect_arch("...\"time_projection.\": ..."), "wan");
        assert_eq!(detect_arch("{}"), "unknown");
        assert_eq!(detect_lora_target_arch("...\"lora_unet_x\": ..."), "sdxl");
    }

    #[test]
    fn compat_and_reason() {
        assert!(model_lora_compatible("sdxl", "sdxl"));
        assert!(!model_lora_compatible("sdxl", "flux"));
        assert!(!model_lora_compatible("unknown", "sdxl"));
        assert_eq!(lora_incompatible_reason("", "", "sdxl", false), "no model selected");
        assert_eq!(
            lora_incompatible_reason("m", "flux", "sdxl", false),
            "target_arch sdxl is not compatible with model arch flux"
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
        let root = "/home/alex/.serenity/models/checkpoints";
        // direct child → no folder
        assert_eq!(folder_relative_to(&format!("{root}/x.safetensors"), root), "");
        // one subdir
        assert_eq!(folder_relative_to(&format!("{root}/ltx-video/x.safetensors"), root), "ltx-video");
        // nested subdir
        assert_eq!(folder_relative_to(&format!("{root}/a/b/x.safetensors"), root), "a/b");
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
        assert_eq!(&uri["data:image/png;base64,".len()..], base64_encode(&[0x89, b'P', b'N', b'G', 1, 2, 3]));

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
