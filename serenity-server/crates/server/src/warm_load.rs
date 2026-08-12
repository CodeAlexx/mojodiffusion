//! Selection-driven model artifact warming.
//!
//! The workers are process-isolated so a model-family swap necessarily drops
//! their private host stores.  Linux' page cache is the one safe cache shared by
//! every Mojo encoder/denoiser/VAE process.  This module fills that cache while a
//! user is choosing settings or writing a prompt, then stops the instant a real
//! generation is submitted so it never competes with inference I/O.

use serde::Serialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const READ_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const MAX_WARM_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const MIN_AVAILABLE_RESERVE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const WARM_READER_THREADS: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct WarmArtifact {
    pub(crate) label: String,
    pub(crate) path: PathBuf,
}

impl WarmArtifact {
    pub(crate) fn new(label: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            label: label.into(),
            path: path.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct WarmFile {
    label: String,
    path: PathBuf,
    size: u64,
    priority: u8,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WarmLoadStatus {
    pub(crate) schema: &'static str,
    pub(crate) state: String,
    pub(crate) generation: u64,
    pub(crate) model: String,
    pub(crate) profile: String,
    pub(crate) planned_files: usize,
    pub(crate) planned_bytes: u64,
    pub(crate) budget_bytes: u64,
    pub(crate) warmed_files: usize,
    pub(crate) warmed_bytes: u64,
    pub(crate) current_label: String,
    pub(crate) current_path: String,
    pub(crate) error: String,
    pub(crate) started_unix_ms: u128,
    pub(crate) elapsed_ms: u128,
    pub(crate) average_mib_per_s: f64,
}

impl Default for WarmLoadStatus {
    fn default() -> Self {
        Self {
            schema: "serenity.model_warm.v1",
            state: "idle".to_string(),
            generation: 0,
            model: String::new(),
            profile: String::new(),
            planned_files: 0,
            planned_bytes: 0,
            budget_bytes: 0,
            warmed_files: 0,
            warmed_bytes: 0,
            current_label: String::new(),
            current_path: String::new(),
            error: String::new(),
            started_unix_ms: 0,
            elapsed_ms: 0,
            average_mib_per_s: 0.0,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct WarmLoadManager {
    generation: Arc<AtomicU64>,
    status: Arc<Mutex<WarmLoadStatus>>,
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn mem_available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kib.saturating_mul(1024))
    })
}

fn warm_budget_for_available(planned: u64, available: u64) -> u64 {
    // MemAvailable includes reclaimable page cache, so a percentage alone can
    // still churn nearly the whole machine. Keep both a 25% proportional
    // reserve and a hard 16-GiB reserve, then cap speculative work at 32 GiB.
    let available_budget = available
        .saturating_mul(3)
        .checked_div(4)
        .unwrap_or(0)
        .min(available.saturating_sub(MIN_AVAILABLE_RESERVE_BYTES));
    planned.min(available_budget.min(MAX_WARM_BYTES))
}

fn warm_budget_bytes(planned: u64) -> u64 {
    warm_budget_for_available(
        planned,
        mem_available_bytes().unwrap_or(32 * 1024 * 1024 * 1024),
    )
}

fn mib_per_second(bytes: u64, elapsed_ms: u128) -> f64 {
    if elapsed_ms == 0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0)) / (elapsed_ms as f64 / 1000.0)
}

fn artifact_priority(label: &str, path: &Path) -> u8 {
    let identity = format!("{} {}", label, path.to_string_lossy()).to_ascii_lowercase();
    if [
        "encoder",
        "tokenizer",
        "clip",
        "t5",
        "qwen",
        "gemma",
        "umt5",
        "language_model",
        "processor",
    ]
    .iter()
    .any(|needle| identity.contains(needle))
    {
        0
    } else if [
        "checkpoint",
        "transformer",
        "denoiser",
        "resident",
        "modulation",
        "runtime cache",
        "model",
    ]
    .iter()
    .any(|needle| identity.contains(needle))
    {
        1
    } else if identity.contains("vae") || identity.contains("decoder") {
        2
    } else {
        3
    }
}

fn collect_path(
    label: &str,
    path: &Path,
    files: &mut Vec<WarmFile>,
    seen_files: &mut HashSet<PathBuf>,
    seen_dirs: &mut HashSet<PathBuf>,
) {
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };
    if metadata.is_file() {
        let identity = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if seen_files.insert(identity) && metadata.len() > 0 {
            files.push(WarmFile {
                label: label.to_string(),
                path: path.to_path_buf(),
                size: metadata.len(),
                priority: artifact_priority(label, path),
            });
        }
        return;
    }
    if !metadata.is_dir() {
        return;
    }
    let identity = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if !seen_dirs.insert(identity) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    let mut children = entries
        .flatten()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    children.sort();
    for child in children {
        collect_path(label, &child, files, seen_files, seen_dirs);
    }
}

fn expand_artifacts(artifacts: &[WarmArtifact]) -> Vec<WarmFile> {
    let mut files = Vec::new();
    let mut seen_files = HashSet::new();
    let mut seen_dirs = HashSet::new();
    for artifact in artifacts {
        collect_path(
            &artifact.label,
            &artifact.path,
            &mut files,
            &mut seen_files,
            &mut seen_dirs,
        );
    }
    files.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then_with(|| a.path.cmp(&b.path))
    });
    files
}

impl WarmLoadManager {
    pub(crate) fn status(&self) -> WarmLoadStatus {
        self.status
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default()
    }

    pub(crate) fn cancel_for_generation(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut status) = self.status.lock() {
            if matches!(status.state.as_str(), "queued" | "warming") {
                status.state = "cancelled_for_generation".to_string();
                status.current_label.clear();
                status.current_path.clear();
            }
        }
    }

    pub(crate) fn start(
        &self,
        model: String,
        profile: String,
        artifacts: Vec<WarmArtifact>,
    ) -> WarmLoadStatus {
        let files = expand_artifacts(&artifacts);
        let planned_bytes = files
            .iter()
            .fold(0_u64, |total, file| total.saturating_add(file.size));
        let budget_bytes = warm_budget_bytes(planned_bytes);
        let token = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let initial = WarmLoadStatus {
            state: if files.is_empty() { "empty" } else { "queued" }.to_string(),
            generation: token,
            model,
            profile,
            planned_files: files.len(),
            planned_bytes,
            budget_bytes,
            started_unix_ms: unix_ms(),
            ..WarmLoadStatus::default()
        };
        if let Ok(mut status) = self.status.lock() {
            *status = initial.clone();
        }
        if files.is_empty() || budget_bytes == 0 {
            return initial;
        }

        // Bound the work list before starting readers. The final file may be a
        // deliberate partial read, but the group can never exceed the RAM cap.
        let mut remaining_budget = budget_bytes;
        let mut work = Vec::new();
        for file in files {
            if remaining_budget == 0 {
                break;
            }
            let limit = file.size.min(remaining_budget);
            remaining_budget = remaining_budget.saturating_sub(limit);
            work.push((file, limit));
        }

        let generation = self.generation.clone();
        let shared_status = self.status.clone();
        let spawn_status = shared_status.clone();
        let spawn_result = std::thread::Builder::new()
            .name("model-page-warmer".to_string())
            .spawn(move || {
                let started = Instant::now();
                if let Ok(mut status) = shared_status.lock() {
                    if status.generation == token {
                        status.state = "warming".to_string();
                    }
                }
                let work = Arc::new(work);
                let next_file = Arc::new(AtomicUsize::new(0));
                let warmed_bytes = Arc::new(AtomicU64::new(0));
                let warmed_files = Arc::new(AtomicUsize::new(0));
                let last_error = Arc::new(Mutex::new(String::new()));

                // Large model stores are sharded. Four independent sequential
                // readers give NVMe enough queue depth without turning warming
                // into an unbounded I/O storm.
                std::thread::scope(|scope| {
                    for _ in 0..WARM_READER_THREADS.min(work.len()) {
                        let work = work.clone();
                        let next_file = next_file.clone();
                        let warmed_bytes = warmed_bytes.clone();
                        let warmed_files = warmed_files.clone();
                        let last_error = last_error.clone();
                        let generation = generation.clone();
                        let shared_status = shared_status.clone();
                        scope.spawn(move || {
                            let mut buffer = vec![0_u8; READ_CHUNK_BYTES];
                            loop {
                                if generation.load(Ordering::Acquire) != token {
                                    break;
                                }
                                let index = next_file.fetch_add(1, Ordering::AcqRel);
                                let Some((file, limit)) = work.get(index) else {
                                    break;
                                };
                                if let Ok(mut status) = shared_status.lock() {
                                    if status.generation != token {
                                        break;
                                    }
                                    status.current_label = file.label.clone();
                                    status.current_path = file.path.to_string_lossy().into_owned();
                                }
                                let mut input = match File::open(&file.path) {
                                    Ok(file) => file,
                                    Err(error) => {
                                        if let Ok(mut message) = last_error.lock() {
                                            if message.is_empty() {
                                                *message =
                                                    format!("{}: {error}", file.path.display());
                                            }
                                        }
                                        continue;
                                    }
                                };
                                let mut file_bytes = 0_u64;
                                let mut read_failed = false;
                                while file_bytes < *limit {
                                    if generation.load(Ordering::Acquire) != token {
                                        return;
                                    }
                                    let want =
                                        buffer.len().min(limit.saturating_sub(file_bytes) as usize);
                                    match input.read(&mut buffer[..want]) {
                                        Ok(0) => {
                                            if file_bytes < *limit {
                                                if let Ok(mut message) = last_error.lock() {
                                                    if message.is_empty() {
                                                        *message = format!(
                                                            "{}: unexpected EOF after {} of {} bytes",
                                                            file.path.display(),
                                                            file_bytes,
                                                            limit
                                                        );
                                                    }
                                                }
                                                read_failed = true;
                                            }
                                            break;
                                        }
                                        Ok(count) => {
                                            file_bytes = file_bytes.saturating_add(count as u64);
                                            let total = warmed_bytes
                                                .fetch_add(count as u64, Ordering::AcqRel)
                                                .saturating_add(count as u64);
                                            if let Ok(mut status) = shared_status.lock() {
                                                if status.generation == token {
                                                    status.warmed_bytes = total;
                                                    let elapsed_ms = started.elapsed().as_millis();
                                                    status.elapsed_ms = elapsed_ms;
                                                    status.average_mib_per_s =
                                                        mib_per_second(total, elapsed_ms);
                                                }
                                            }
                                        }
                                        Err(error) => {
                                            if let Ok(mut message) = last_error.lock() {
                                                if message.is_empty() {
                                                    *message =
                                                        format!("{}: {error}", file.path.display());
                                                }
                                            }
                                            read_failed = true;
                                            break;
                                        }
                                    }
                                }
                                if !read_failed && file_bytes == file.size {
                                    let count = warmed_files
                                        .fetch_add(1, Ordering::AcqRel)
                                        .saturating_add(1);
                                    if let Ok(mut status) = shared_status.lock() {
                                        if status.generation == token {
                                            status.warmed_files = count;
                                        }
                                    }
                                }
                            }
                        });
                    }
                });

                if generation.load(Ordering::Acquire) == token {
                    if let Ok(mut status) = shared_status.lock() {
                        if status.generation == token {
                            status.warmed_bytes = warmed_bytes.load(Ordering::Acquire);
                            status.warmed_files = warmed_files.load(Ordering::Acquire);
                            status.current_label.clear();
                            status.current_path.clear();
                            status.elapsed_ms = started.elapsed().as_millis();
                            status.average_mib_per_s =
                                mib_per_second(status.warmed_bytes, status.elapsed_ms);
                            status.error = last_error
                                .lock()
                                .map(|value| value.clone())
                                .unwrap_or_default();
                            status.state = if status.warmed_bytes >= budget_bytes
                                && budget_bytes < status.planned_bytes
                            {
                                "budget_complete"
                            } else if status.error.is_empty() {
                                "complete"
                            } else {
                                "complete_with_errors"
                            }
                            .to_string();
                        }
                    }
                }
            });
        if let Err(error) = spawn_result {
            if let Ok(mut status) = spawn_status.lock() {
                if status.generation == token {
                    status.state = "failed".to_string();
                    status.error = format!("failed to start model warmer: {error}");
                }
            }
        }
        initial
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_files_are_warmed_before_denoiser_and_vae() {
        let root = std::env::temp_dir().join(format!(
            "serenity-warm-load-{}-{}",
            std::process::id(),
            unix_ms()
        ));
        let encoder = root.join("text_encoder");
        std::fs::create_dir_all(&encoder).unwrap();
        std::fs::write(encoder.join("model.safetensors"), [1_u8; 8]).unwrap();
        std::fs::write(root.join("transformer.safetensors"), [2_u8; 8]).unwrap();
        std::fs::write(root.join("vae.safetensors"), [3_u8; 8]).unwrap();

        let files = expand_artifacts(&[
            WarmArtifact::new("denoiser checkpoint", root.join("transformer.safetensors")),
            WarmArtifact::new("text encoder", &encoder),
            WarmArtifact::new("VAE", root.join("vae.safetensors")),
        ]);
        assert_eq!(files.len(), 3);
        assert!(files[0].path.ends_with("text_encoder/model.safetensors"));
        assert!(files[1].path.ends_with("transformer.safetensors"));
        assert!(files[2].path.ends_with("vae.safetensors"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn overlapping_directory_and_file_specs_are_deduplicated() {
        let root = std::env::temp_dir().join(format!(
            "serenity-warm-dedup-{}-{}",
            std::process::id(),
            unix_ms()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("model.safetensors");
        std::fs::write(&file, [1_u8; 4]).unwrap();
        let files = expand_artifacts(&[
            WarmArtifact::new("encoder directory", &root),
            WarmArtifact::new("encoder shard", &file),
        ]);
        assert_eq!(files.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn warm_budget_preserves_absolute_and_proportional_ram_headroom() {
        let gib = 1024_u64 * 1024 * 1024;
        assert_eq!(warm_budget_for_available(100 * gib, 64 * gib), 32 * gib);
        assert_eq!(warm_budget_for_available(100 * gib, 24 * gib), 8 * gib);
        assert_eq!(warm_budget_for_available(100 * gib, 12 * gib), 0);
        assert_eq!(warm_budget_for_available(4 * gib, 64 * gib), 4 * gib);
    }

    #[test]
    fn generation_cancels_an_active_warmer_immediately() {
        let manager = WarmLoadManager::default();
        {
            let mut status = manager.status.lock().unwrap();
            status.state = "warming".to_string();
            status.generation = 1;
        }
        manager.cancel_for_generation();
        assert_eq!(manager.status().state, "cancelled_for_generation");
        assert_eq!(manager.generation.load(Ordering::Acquire), 1);
    }

    #[test]
    fn video_profiles_include_their_encoder_and_denoiser_artifacts() {
        let (_, h3) = crate::video::warm_artifacts("MiniMax-H3-Mojo", None, "int8-fast", "t2va");
        assert!(
            h3.iter()
                .any(|artifact| artifact.label.contains("text encoder"))
        );
        assert!(
            h3.iter()
                .any(|artifact| artifact.label.contains("resident denoiser"))
        );
        assert!(
            h3.iter()
                .any(|artifact| artifact.label.contains("audio VAE"))
        );

        let (_, ltx) = crate::video::warm_artifacts("ltx-2.3-22b-dev-fp8", None, "fp8", "t2v");
        assert!(
            ltx.iter()
                .any(|artifact| artifact.label.contains("Gemma text encoder"))
        );
        assert!(
            ltx.iter()
                .any(|artifact| artifact.label.contains("denoiser checkpoint"))
        );

        let (_, wan) = crate::video::warm_artifacts("wan22", None, "fp8", "t2v");
        assert!(
            wan.iter()
                .any(|artifact| artifact.label.contains("UMT5 encoder"))
        );
        assert!(
            wan.iter()
                .any(|artifact| artifact.label.contains("transformer shard"))
        );

        let (_, scail) = crate::video::warm_artifacts("scail2", None, "fp8", "animation");
        assert!(
            scail
                .iter()
                .any(|artifact| artifact.label.contains("UMT5 encoder"))
        );
        assert!(
            scail
                .iter()
                .any(|artifact| artifact.label.contains("CLIP vision encoder"))
        );
    }
}
