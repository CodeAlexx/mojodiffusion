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

struct ScanEntry {
    name: String,
    path: String,
    arch: String,
    size: i64,
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
            out.push(ScanEntry { name: name.to_string(), path: dir, arch: arch.to_string(), size });
        }
    }
    // known multi-shard checkpoint subdirs under checkpoints/.
    for (name, arch) in [("qwen-image-2512", "qwen-image"), ("ideogram-4-fp8", "ideogram4")] {
        let dir = format!("{CHECKPOINTS_DIR}/{name}");
        if dir_exists(&dir) {
            let size = du_sb(&dir);
            out.push(ScanEntry { name: name.to_string(), path: dir, arch: arch.to_string(), size });
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
    let metadata = json!({
        "schema": "serenity.model.metadata.v1",
        "source": "disk_scan",
        "family": arch,
        "notes": "",
    });
    let card = json!({
        "schema": "serenity.model.card.v1",
        "title": e.name,
        "subtitle": arch,
        "path": e.path,
        "size": e.size,
        "thumbnail": "",
        "preview": "",
        "favorite": false,
        "loaded": loaded,
        "metadata": metadata,
    });
    json!({
        "name": e.name,
        "path": e.path,
        "arch": arch,
        "size": e.size,
        "loaded": loaded,
        "type": "checkpoint",
        "thumbnail": "",
        "preview": "",
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
    let metadata = json!({
        "schema": "serenity.lora.metadata.v1",
        "source": "safetensors_header_probe",
        "target_arch": target,
        "trigger": "",
    });
    let card = json!({
        "schema": "serenity.lora.card.v1",
        "title": e.name,
        "subtitle": target,
        "path": e.path,
        "size": e.size,
        "thumbnail": "",
        "preview": "",
        "favorite": false,
        "metadata": metadata,
        "compatibility": compatibility,
    });
    json!({
        "name": e.name,
        "path": e.path,
        "size": e.size,
        "arch": target,
        "target_arch": target,
        "trigger": "",
        "thumbnail": "",
        "preview": "",
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
}
