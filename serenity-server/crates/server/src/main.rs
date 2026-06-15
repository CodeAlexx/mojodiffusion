//! serenity-server — Phase-A Rust control plane.
//!
//! GOAL (Phase A exit): a Rust HTTP/WS server that accepts a generate request,
//! drives the UNCHANGED Mojo `output/bin/serenity_worker_stub` over serenity-ipc,
//! streams progress over WebSocket, and reports the produced PNG path — proving the
//! Rust->Mojo seam end-to-end with zero GPU.
//!
//! Endpoints (reproduce the daemon's shapes — see serenitymojo/serve/serenity_daemon.mojo):
//!   POST /v1/generate    -> accept JobParams (or a minimal subset), enqueue, return {"job_id":...}
//!   GET  /v1/progress     -> WebSocket; replay buffered events THEN stream live until terminal
//!   POST /v1/cancel       -> {"job":<id>}; cancel the in-flight job (200 accepted / 404 not in flight)
//!   GET  /v1/health       -> {"backend":"isolated-rust","ok":true}
//!
//! WS SUBSCRIBE-RACE (HIGH): a WS client that connects AFTER the worker has already
//! emitted some events (including the terminal Done — fast jobs finish in
//! milliseconds) must NOT lose them. Each job therefore keeps a per-job event
//! HISTORY (Vec<WorkerEvent>) alongside its live broadcast sender. On WS connect we
//! (under one lock) snapshot the history AND subscribe, then replay the buffered
//! frames before streaming live ones — so no frame can slip through the gap between
//! "snapshot" and "subscribe". The job entry is retained until terminal + a short
//! grace window, then evicted, so a slightly-late client still replays the full run.
//!
//! WORKER PEER-CLOSE (lazy respawn): if the worker exits mid-job (crash / OOM-kill /
//! its own exit), the driver maps the typed EventPoll::PeerClosed to: synthesize a
//! Failed event for the in-flight job, fan it out, then respawn the worker LAZILY
//! and keep serving. It does NOT tear the server down. Mirrors
//! process_isolated_backend.step()'s `still_open==False` branch.
//! ARCHITECTURE NOTE: the serenity-ipc driver is blocking/poll-based (non-async),
//! and the single-GPU contract is one in-flight job. So run ONE worker-driver loop
//! on a dedicated std::thread that owns the WorkerHandle, pulls jobs from an mpsc
//! queue, steps the worker, and fans each WorkerEvent to subscribers (tokio
//! broadcast keyed by job_id). The async axum handlers only enqueue + subscribe.
//!
//! Build: `cargo build` (safe). Run: `serenity-server --worker output/bin/serenity_worker_stub`.
//! NEVER run `mojo build` / `pixi run build-*` (OOM-kills the desktop).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use serenity_graph::lower_request;
use serenity_ipc::{spawn_worker, EventPoll, WorkerHandle};
use serenity_wire::{JobParams, LoraSpec, WorkerEvent};

mod gallery;
mod grid;
mod jobs;
mod models;
mod video;

/// How many buffered events a slow WS subscriber may lag before it's dropped.
const BROADCAST_CAP: usize = 256;
/// Worker-driver poll cadence while a job is in flight (~15-20ms requested).
const POLL_INTERVAL: Duration = Duration::from_millis(16);
/// How long we wait for the worker's initial `Ready` handshake.
const READY_TIMEOUT: Duration = Duration::from_secs(15);
/// After a job reaches a terminal event, keep its history around this long so a
/// late WS subscriber can still replay the whole run (incl. the terminal frame).
const TERMINAL_GRACE: Duration = Duration::from_secs(10);

/// The backend identity reported by /v1/health (this is the isolated-process
/// Rust control plane driving a Mojo child).
const BACKEND_NAME: &str = "isolated-rust";

// ── shared state ──────────────────────────────────────────────────────────────

/// Per-job event channel: a live `broadcast` sender AND a replayable HISTORY of
/// every event fanned out so far. The driver appends to `history` (and broadcasts)
/// under the SAME lock used by a connecting WS client to snapshot+subscribe, so no
/// frame can leak through the snapshot→subscribe gap (the WS subscribe-race fix).
pub(crate) struct JobChannel {
    /// Live fan-out for events that arrive AFTER a subscriber attaches.
    tx: tokio::sync::broadcast::Sender<WorkerEvent>,
    /// Append-only history + terminal marker, guarded so subscribe is atomic w.r.t.
    /// publish. `terminal_at` is set when the job reaches a terminal event.
    inner: Mutex<JobChannelInner>,
}

struct JobChannelInner {
    /// Every event published for this job, in order (incl. the terminal one).
    history: Vec<WorkerEvent>,
    /// When the terminal event was published (drives grace-window eviction). None
    /// while the job is still in flight.
    terminal_at: Option<Instant>,
}

impl JobChannel {
    pub(crate) fn new() -> Arc<Self> {
        let (tx, _rx) = tokio::sync::broadcast::channel::<WorkerEvent>(BROADCAST_CAP);
        Arc::new(JobChannel {
            tx,
            inner: Mutex::new(JobChannelInner {
                history: Vec::new(),
                terminal_at: None,
            }),
        })
    }

    /// Driver side: record an event into history (marking terminal if so) and fan it
    /// out live — atomically, under the lock, so a concurrent subscriber sees it in
    /// exactly one of {history snapshot, live stream}, never neither/both-missed.
    fn publish(&self, ev: WorkerEvent) {
        let terminal = ev.is_terminal();
        let mut g = self.inner.lock().expect("JobChannel poisoned");
        g.history.push(ev.clone());
        if terminal && g.terminal_at.is_none() {
            g.terminal_at = Some(Instant::now());
        }
        // Broadcast while holding the lock so subscribe()+snapshot in ws_progress
        // cannot interleave between the history push and the live send.
        let _ = self.tx.send(ev);
    }

    /// WS side: atomically snapshot the history-so-far AND subscribe to live events.
    /// Returns (replay_frames, live_receiver, already_terminal).
    fn snapshot_and_subscribe(
        &self,
    ) -> (
        Vec<WorkerEvent>,
        tokio::sync::broadcast::Receiver<WorkerEvent>,
        bool,
    ) {
        let g = self.inner.lock().expect("JobChannel poisoned");
        let rx = self.tx.subscribe();
        let history = g.history.clone();
        let already_terminal = g.terminal_at.is_some();
        (history, rx, already_terminal)
    }
}

/// Per-job registry. The driver publishes WorkerEvents into the matching
/// `JobChannel` (history + live broadcast); WS subscribers snapshot+subscribe. An
/// entry is retained until terminal + a grace window so a late subscriber can still
/// replay the entire run (incl. the terminal Done). See `evict_expired`.
type Registry = Arc<Mutex<HashMap<String, Arc<JobChannel>>>>;

/// Control messages the HTTP layer sends to the worker-driver thread (which owns
/// the `WorkerHandle`). Cancel must run on the driver thread, not a handler.
/// `Job` is boxed so the two variants stay close in size (JobParams is large).
pub(crate) enum DriverCtl {
    /// A new job was pushed to the JobBook — wake the driver to promote it.
    Wake,
    /// Cancel the job with this id IF it is the one currently in flight.
    Cancel(String),
}

#[derive(Clone)]
pub(crate) struct AppState {
    /// Control channel to the worker-driver thread (jobs + cancels).
    pub(crate) ctl: std::sync::mpsc::Sender<DriverCtl>,
    /// job_id -> JobChannel (history + live broadcast), for WS subscription/replay.
    pub(crate) registry: Registry,
    /// The job currently in flight (driver writes; cancel handler reads to decide
    /// 200-accepted vs 404-not-in-flight without blocking the driver thread).
    pub(crate) in_flight: Arc<Mutex<Option<String>>>,
    /// Monotonic counter feeding job_id generation (no rand).
    pub(crate) next_id: Arc<AtomicU64>,
    /// Server-configured output directory written into every JobParams.out_dir.
    pub(crate) out_dir: PathBuf,
    /// Prior jobs.db rows loaded ONCE at startup (history half of /v1/jobs).
    pub(crate) prior: Arc<Vec<[String; 6]>>,
    /// Current-session JobBook (live half of /v1/jobs + the serial queue), ordered by
    /// enqueue; the driver promotes the first active-queued entry. /v1/reorder + /v1/remove
    /// mutate it.
    pub(crate) jobs: Arc<Mutex<Vec<jobs::JobEntry>>>,
    /// Backend identity for /v1/health (the worker kind: "zimage"/"stub"/… — derived from
    /// --kind or the worker binary name). Cosmetic for the bridge (it gates on status only).
    pub(crate) backend_name: String,
}

// ── request body ────────────────────────────────────────────────────────────--

/// POST /v1/generate body. Only the core fields are accepted here; everything
/// else rides JobParams::default(). All optional so a minimal `{model,prompt}`
/// body works.
///
/// This struct is ALSO the shape the server reads after `lower_request` has
/// flattened a `workflow` graph: `lower_request` writes flat keys
/// (`model`/`prompt`/`width`/`steps`/`cfg`/`sampler`/`scheduler`/`negative` plus
/// `sigma_shift`/`images`/`creativity`/`workflow_save_prefix`/`lora`) onto the
/// request object, and the extra fields below let those reach `JobParams`. The
/// no-workflow path simply doesn't carry them.
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    negative: Option<String>,
    #[serde(default)]
    width: Option<i64>,
    #[serde(default)]
    height: Option<i64>,
    #[serde(default)]
    steps: Option<i64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    sampler: Option<String>,
    #[serde(default)]
    scheduler: Option<String>,
    #[serde(default)]
    cfg: Option<f64>,
    // ── workflow-lowered extras (present only after lower_request) ──
    #[serde(default)]
    sigma_shift: Option<f64>,
    #[serde(default)]
    cfg_override: Option<f64>,
    #[serde(default)]
    cfg_override_start_percent: Option<f64>,
    #[serde(default)]
    cfg_override_end_percent: Option<f64>,
    #[serde(default)]
    images: Option<i64>,
    #[serde(default)]
    creativity: Option<f64>,
    #[serde(default)]
    workflow_save_prefix: Option<String>,
    /// Lowered LoRA overlays — `lower_request` writes the `lora` array.
    #[serde(default, rename = "lora")]
    loras: Option<Vec<LoraSpec>>,
    // ── img2img / inpaint passthrough (JobParams already carries these) ──
    /// img2img init latent source (a decodable image path). Empty = txt2img.
    #[serde(default)]
    init_image: Option<String>,
    /// SetLatentNoiseMask source (requires `init_image`); empty = no mask.
    #[serde(default)]
    mask_image: Option<String>,
    /// Which channel of `mask_image` is the mask (alpha|red|green|blue|luminance).
    #[serde(default)]
    lanpaint_mask_channel: Option<String>,
    // ── hires-fix (control-plane only; NOT forwarded to the worker wire) ──
    /// >1.0 enables a second img2img refine pass at `scale*resolution`.
    #[serde(default)]
    hires_scale: Option<f64>,
    /// Creativity (denoise) of the hires refine pass; lower = closer to base.
    #[serde(default)]
    hires_denoise: Option<f64>,
    // ── advanced-sampling knobs (UI _section_advanced). Forwarded to the worker
    //    wire so it can HONOR what it supports and warn-loud on what it can't. ──
    #[serde(default)]
    clip_skip: Option<i64>,
    #[serde(default)]
    eta: Option<f64>,
    #[serde(default)]
    sigma_min: Option<f64>,
    #[serde(default)]
    sigma_max: Option<f64>,
    #[serde(default)]
    restart_sampling: Option<bool>,
    #[serde(default)]
    vae: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProgressQuery {
    job: String,
}

/// POST /v1/cancel body — `{"job":<id>}`.
#[derive(Debug, Deserialize)]
struct CancelRequest {
    job: String,
}

// ── CLI ─────────────────────────────────────────────────────────────────────--

/// CLI: --worker <path> [--kind <k>] [--port 7801] [--out-dir DIR]
///
/// `--kind <k>` makes the worker spawn as `<worker> worker <k> <fd>` (the daemon's
/// internal worker entry, e.g. `serenity_daemon worker zimage <fd>`). Without it,
/// the worker is spawned as `<worker> <fd>` (the standalone stub takes only the fd).
fn parse_args() -> (PathBuf, u16, PathBuf, Vec<String>) {
    let mut worker =
        PathBuf::from("/home/alex/mojodiffusion/output/bin/serenity_worker_stub");
    let mut port: u16 = 7801;
    let mut out_dir = PathBuf::from("./serenity_out");
    let mut kind: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--worker" => {
                if let Some(v) = args.next() {
                    worker = PathBuf::from(v);
                }
            }
            "--kind" => {
                if let Some(v) = args.next() {
                    kind = Some(v);
                }
            }
            "--port" => {
                if let Some(v) = args.next() {
                    match v.parse::<u16>() {
                        Ok(p) => port = p,
                        Err(_) => eprintln!("warning: invalid --port {v:?}, using {port}"),
                    }
                }
            }
            "--out-dir" => {
                if let Some(v) = args.next() {
                    out_dir = PathBuf::from(v);
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "serenity-server --worker <path> [--kind <k>] [--port {port}] [--out-dir {}]",
                    out_dir.display()
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("warning: ignoring unknown arg {other:?}");
            }
        }
    }
    // The daemon worker entry takes `worker <kind> <fd>`; the standalone stub takes
    // just `<fd>`. --kind <k> => prepend ["worker", k]; absent => no pre-args.
    let worker_args = match kind {
        Some(k) => vec!["worker".to_string(), k],
        None => Vec::new(),
    };
    (worker, port, out_dir, worker_args)
}

// ── worker-driver thread ──────────────────────────────────────────────────────

/// The single std::thread that owns the WorkerHandle. It pulls control messages
/// (jobs + cancels) off `ctl`, and for each job: send_start, poll next_event at
/// ~16ms cadence WHILE ALSO draining `ctl` each tick so a cancel for the in-flight
/// job is forwarded to the worker IMMEDIATELY (not after the job ends), publish
/// every WorkerEvent to the job's channel (history + broadcast), and stop at the
/// first terminal event. One job in flight at a time (single-GPU contract); jobs
/// that arrive while one is running are buffered in `pending` and run next. If the
/// worker peer-closes mid-job it fails that job and respawns the worker LAZILY
/// rather than tearing the server down.
/// Update the current-session JobRecord with `id` in place (no-op if absent).
fn update_record(
    jobs: &Arc<Mutex<Vec<jobs::JobEntry>>>,
    id: &str,
    f: impl FnOnce(&mut jobs::JobRecord),
) {
    if let Ok(mut v) = jobs.lock() {
        if let Some(e) = v.iter_mut().find(|e| e.record.id == id) {
            f(&mut e.record);
        }
    }
}

/// Apply a WorkerEvent to the matching JobRecord (state/step/progress/output/error).
fn apply_event_to_record(jobs: &Arc<Mutex<Vec<jobs::JobEntry>>>, id: &str, ev: &WorkerEvent) {
    match ev {
        WorkerEvent::Progress { step, total, .. } => update_record(jobs, id, |r| {
            r.state = "running".to_string();
            r.step = *step;
            if *total > 0 {
                r.total = *total;
                r.progress = step * 100 / total;
            }
        }),
        WorkerEvent::Done { output_path } => {
            // the worker has written the genparams tEXt into the PNG; capture it for
            // the jobs.db row (so the persisted history carries params, like the daemon).
            let pj = gallery::png_genparams_or_empty(output_path);
            update_record(jobs, id, |r| {
                r.state = "done".to_string();
                r.progress = 100;
                r.output_path = output_path.clone();
                r.params_json = pj;
            })
        }
        WorkerEvent::Failed { error } => update_record(jobs, id, |r| {
            r.state = "failed".to_string();
            r.error = error.clone();
        }),
        WorkerEvent::Cancelled => update_record(jobs, id, |r| r.state = "cancelled".to_string()),
        WorkerEvent::Ready => {}
    }
}

fn kind_from_bin(bin: &std::path::Path) -> String {
    let name = bin.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name.contains("ideogram") { "ideogram4".to_string() }
    else if name.contains("stub") { "stub".to_string() }
    else { "zimage".to_string() }
}

/// Map a request's model -> (kind, worker binary in the same dir). One GPU = one
/// resident model, so the driver swaps the worker binary when the kind changes.
fn worker_for_model(cur_bin: &std::path::Path, model: &str) -> (String, PathBuf) {
    let dir = cur_bin
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("output/bin"));
    let m = model.to_lowercase();
    if m.contains("ideogram") {
        ("ideogram4".to_string(), dir.join("serenity_worker_ideogram4"))
    } else {
        ("zimage".to_string(), dir.join("serenity_worker_zimage"))
    }
}

fn run_worker_driver(
    worker_bin: PathBuf,
    worker_args: Vec<String>,
    ctl: std::sync::mpsc::Receiver<DriverCtl>,
    registry: Registry,
    in_flight: Arc<Mutex<Option<String>>>,
    jobs: Arc<Mutex<Vec<jobs::JobEntry>>>,
    prior: Arc<Vec<[String; 6]>>,
    db_path: PathBuf,
) {
    // worker_bin/current_kind are mutable so the driver can SWAP the worker binary
    // when a job needs a different model (one GPU = one resident model at a time).
    let mut worker_bin = worker_bin;
    let mut current_kind = kind_from_bin(&worker_bin);
    // `handle` is Option so we can drop a dead worker and respawn lazily on the next
    // job (mirrors process_isolated_backend: a peer-close fails the job, the next
    // start() respawns). It starts None and is brought up on first need / handshake.
    let mut handle: Option<WorkerHandle> = match spawn_and_handshake(&worker_bin, &worker_args) {
        Ok(h) => Some(h),
        Err(e) => {
            // Non-fatal: keep the thread alive so the server stays up; we respawn
            // lazily when the first job arrives.
            eprintln!("WARN: initial worker spawn/handshake failed: {e:#}; will retry lazily");
            None
        }
    };

    loop {
        // Promote the FIRST active-queued job from the shared JobBook (the daemon's
        // tick_worker model — so /v1/reorder + /v1/remove, which mutate the book, take
        // effect on the very next promotion). Marking it `running` under the lock makes
        // promotion atomic. When nothing is queued, block on `ctl` for a Wake (a new job
        // was enqueued) or a Cancel (ignored while idle), then re-scan.
        let promoted = match jobs.lock() {
            Ok(mut book) => match book.iter().position(|e| e.record.state == "queued") {
                Some(i) => {
                    book[i].record.state = "running".to_string();
                    Some((
                        book[i].record.id.clone(),
                        book[i].params.clone(),
                        book[i].hires_scale,
                        book[i].hires_denoise,
                    ))
                }
                None => None,
            },
            Err(_) => None,
        };
        let (job_id, params, hires_scale, hires_denoise) = match promoted {
            Some(x) => x,
            None => match ctl.recv() {
                Ok(DriverCtl::Wake) => continue,
                Ok(DriverCtl::Cancel(target)) => {
                    tracing::info!(job = %target, "cancel ignored: not in flight (idle)");
                    continue;
                }
                Err(_) => break, // all senders dropped — server shutting down.
            },
        };

        // The per-job channel was registered in the registry by post_generate BEFORE the
        // JobEntry became promotable, so it is present here.
        let channel = match registry.lock().ok().and_then(|m| m.get(&job_id).cloned()) {
            Some(c) => c,
            None => {
                tracing::warn!(%job_id, "no channel for promoted job (evicted?); skipping");
                update_record(&jobs, &job_id, |r| r.state = "failed".to_string());
                continue;
            }
        };

        // Per-job model -> worker SWAP (single GPU: one resident model at a time).
        // The CPU 'stub' worker serves everything, so never swap it out.
        if current_kind != "stub" {
            let (want_kind, want_bin) = worker_for_model(&worker_bin, &params.model);
            if want_kind != current_kind && want_bin.exists() {
                tracing::info!(%job_id, from = %current_kind, to = %want_kind, "swapping worker for model");
                if let Some(mut h) = handle.take() {
                    h.kill(); // SIGKILL + wait -> process dies -> GPU driver reclaims its VRAM
                }
                // brief settle so the GPU fully releases before the next model loads
                std::thread::sleep(std::time::Duration::from_millis(800));
                worker_bin = want_bin;
                current_kind = want_kind;
            }
        }

        // Lazily (re)spawn if we have no live worker (first job, or after a prior
        // peer-close).
        if handle.is_none() {
            match spawn_and_handshake(&worker_bin, &worker_args) {
                Ok(h) => handle = Some(h),
                Err(e) => {
                    tracing::error!(%job_id, "worker respawn failed: {e:#}");
                    let ev = WorkerEvent::Failed {
                        error: format!("worker unavailable (respawn failed): {e}"),
                    };
                    apply_event_to_record(&jobs, &job_id, &ev);
                    channel.publish(ev);
                    schedule_evict(&registry, &job_id);
                    continue;
                }
            }
        }
        let h = handle.as_mut().expect("handle present after spawn");

        tracing::info!(%job_id, "starting job");
        set_in_flight(&in_flight, Some(job_id.clone()));

        if let Err(e) = h.send_start(&params) {
            tracing::error!(%job_id, "send_start failed: {e:#}");
            let ev = WorkerEvent::Failed { error: format!("send_start failed: {e}") };
            apply_event_to_record(&jobs, &job_id, &ev);
            channel.publish(ev);
            set_in_flight(&in_flight, None);
            schedule_evict(&registry, &job_id);
            // send_start failing usually means the worker is dead; drop it so the
            // next job respawns lazily.
            if let Some(mut dead) = handle.take() {
                dead.kill();
            }
            continue;
        }

        // Poll until terminal OR the worker peer-closes, servicing `ctl` each tick so
        // a cancel for THIS job reaches the worker promptly (and stray jobs buffer).
        // Hires-fix runs the job as TWO worker passes under one job_id (base pass +
        // an upscaled img2img refine); a single-pass job takes the plain driver.
        let outcome = if hires_scale > 1.0 {
            drive_hires_two_pass(
                h, &channel, &job_id, &ctl, &jobs, &params, hires_scale, hires_denoise,
            )
        } else {
            drive_one_job(h, &channel, &job_id, &ctl, &jobs)
        };
        set_in_flight(&in_flight, None);
        schedule_evict(&registry, &job_id);
        // Persist the jobs.db after this job's terminal so session history survives a
        // restart (rewrite = prior rows + started session jobs; the daemon's F7/F10 model).
        if let Ok(j) = jobs.lock() {
            jobs::save_jobs_db(&prior, &j, &db_path);
        }

        match outcome {
            JobOutcome::Terminal => {
                tracing::info!(%job_id, "job finished");
            }
            JobOutcome::PeerClosed => {
                // Worker exited mid-job. We already published a synthetic Failed
                // inside drive_one_job. Reap+drop the dead handle so the NEXT job
                // respawns lazily — server stays up.
                tracing::error!(%job_id, "worker peer-closed mid-job; respawning lazily");
                if let Some(mut dead) = handle.take() {
                    dead.kill();
                }
            }
            JobOutcome::IpcError => {
                // A genuine IPC fault (not a clean peer-close): also publish Failed
                // (done in drive_one_job) and drop the worker to be safe; respawn
                // lazily.
                tracing::error!(%job_id, "worker IPC error; respawning lazily");
                if let Some(mut dead) = handle.take() {
                    dead.kill();
                }
            }
            JobOutcome::ChannelClosed => {
                // All control senders dropped while driving: finish reaping & exit.
                tracing::info!(%job_id, "control channel closed mid-job; shutting down");
                break;
            }
        }
    }

    if let Some(mut h) = handle.take() {
        h.kill();
    }
}

/// Spawn the worker and block (driver thread only) until its `Ready` handshake.
fn spawn_and_handshake(
    worker_bin: &std::path::Path,
    pre_args: &[String],
) -> anyhow::Result<WorkerHandle> {
    let refs: Vec<&str> = pre_args.iter().map(String::as_str).collect();
    let mut handle = spawn_worker(worker_bin, &refs)?;
    let ready_deadline = Instant::now() + READY_TIMEOUT;
    loop {
        match handle.next_event_poll() {
            Ok(EventPoll::Event(WorkerEvent::Ready)) => {
                tracing::info!("worker ready");
                return Ok(handle);
            }
            Ok(EventPoll::Event(other)) => {
                tracing::warn!("ignoring pre-ready event from worker: {other:?}");
            }
            Ok(EventPoll::Idle) => {
                if Instant::now() >= ready_deadline {
                    handle.kill();
                    return Err(anyhow::anyhow!(
                        "worker did not become Ready within {READY_TIMEOUT:?}"
                    ));
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Ok(EventPoll::PeerClosed) => {
                handle.kill();
                return Err(anyhow::anyhow!("worker exited during handshake"));
            }
            Err(e) => {
                handle.kill();
                return Err(anyhow::anyhow!("worker IPC error during handshake: {e}"));
            }
        }
    }
}

/// How a single job's polling loop ended.
enum JobOutcome {
    /// A terminal WorkerEvent (Done/Failed/Cancelled) was published.
    Terminal,
    /// The worker process exited (clean peer-close); a synthetic Failed was published.
    PeerClosed,
    /// A genuine IPC fault occurred; a synthetic Failed was published.
    IpcError,
    /// All control senders dropped mid-job (server shutdown).
    ChannelClosed,
}

/// Poll one job to its terminal event, publishing every WorkerEvent to `channel`
/// (history + live broadcast). On peer-close or IPC fault a synthetic Failed is
/// published and the corresponding outcome returned so the caller can respawn.
///
/// CRUCIAL: while polling the worker, this ALSO drains `ctl` every tick (non-block).
/// A `Cancel(job_id)` for the CURRENTLY in-flight job is forwarded to the worker via
/// `send_cancel` right away — otherwise the cancel would sit in the channel until the
/// (uncancelled) job finished, defeating the purpose. A `Job(..)` that arrives mid-
/// flight is buffered into `pending` to run after this one (single-GPU contract); a
/// `Cancel` for any OTHER id is dropped (only the in-flight job is cancelable).
fn drive_one_job(
    handle: &mut WorkerHandle,
    channel: &Arc<JobChannel>,
    job_id: &str,
    ctl: &std::sync::mpsc::Receiver<DriverCtl>,
    jobs: &Arc<Mutex<Vec<jobs::JobEntry>>>,
) -> JobOutcome {
    loop {
        // 1. Service any pending control messages WITHOUT blocking.
        loop {
            match ctl.try_recv() {
                Ok(DriverCtl::Cancel(target)) => {
                    if target == job_id {
                        match handle.send_cancel() {
                            Ok(()) => tracing::info!(job = %job_id, "cancel forwarded to worker"),
                            Err(e) => tracing::warn!(job = %job_id, "send_cancel failed: {e:#}"),
                        }
                    } else {
                        tracing::info!(job = %target, "cancel ignored: not the in-flight job");
                    }
                }
                // A Wake (new job enqueued) while one is in flight: nothing to do now —
                // it'll be promoted from the JobBook after this job finishes.
                Ok(DriverCtl::Wake) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return JobOutcome::ChannelClosed;
                }
            }
        }

        // 2. Pump the worker.
        match handle.next_event_poll() {
            Ok(EventPoll::Event(ev)) => {
                let terminal = ev.is_terminal();
                apply_event_to_record(jobs, job_id, &ev);
                channel.publish(ev);
                if terminal {
                    return JobOutcome::Terminal;
                }
            }
            Ok(EventPoll::Idle) => {
                std::thread::sleep(POLL_INTERVAL);
            }
            Ok(EventPoll::PeerClosed) => {
                let ev = WorkerEvent::Failed {
                    error: "worker process exited unexpectedly".to_string(),
                };
                apply_event_to_record(jobs, job_id, &ev);
                channel.publish(ev);
                return JobOutcome::PeerClosed;
            }
            Err(e) => {
                let ev = WorkerEvent::Failed { error: format!("worker IPC error: {e}") };
                apply_event_to_record(jobs, job_id, &ev);
                channel.publish(ev);
                return JobOutcome::IpcError;
            }
        }
    }
}

/// Outcome of one worker pass (used by the hires 2-pass orchestrator).
enum PassResult {
    /// Worker emitted Done; carries the output_path. The terminal Done was
    /// published to the channel iff `publish_done` was true.
    Done(String),
    /// Failed or Cancelled — already applied to the record and published.
    Terminal,
    PeerClosed,
    IpcError,
    ChannelClosed,
}

/// Drive the worker for ONE pass. Identical to `drive_one_job` except the terminal
/// `Done` is intercepted: its `output_path` is returned, and it is published as the
/// job's terminal only when `publish_done` is true. For the base pass of a hires
/// job we pass `false` so the record stays `running` until the refine pass finishes.
fn drive_one_pass(
    handle: &mut WorkerHandle,
    channel: &Arc<JobChannel>,
    job_id: &str,
    ctl: &std::sync::mpsc::Receiver<DriverCtl>,
    jobs: &Arc<Mutex<Vec<jobs::JobEntry>>>,
    publish_done: bool,
) -> PassResult {
    loop {
        loop {
            match ctl.try_recv() {
                Ok(DriverCtl::Cancel(target)) => {
                    if target == job_id {
                        match handle.send_cancel() {
                            Ok(()) => tracing::info!(job = %job_id, "cancel forwarded to worker"),
                            Err(e) => tracing::warn!(job = %job_id, "send_cancel failed: {e:#}"),
                        }
                    }
                }
                Ok(DriverCtl::Wake) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return PassResult::ChannelClosed;
                }
            }
        }
        match handle.next_event_poll() {
            Ok(EventPoll::Event(ev)) => {
                if let WorkerEvent::Done { output_path } = &ev {
                    let path = output_path.clone();
                    if publish_done {
                        apply_event_to_record(jobs, job_id, &ev);
                        channel.publish(ev);
                    }
                    return PassResult::Done(path);
                }
                let terminal = ev.is_terminal();
                apply_event_to_record(jobs, job_id, &ev);
                channel.publish(ev);
                if terminal {
                    return PassResult::Terminal;
                }
            }
            Ok(EventPoll::Idle) => std::thread::sleep(POLL_INTERVAL),
            Ok(EventPoll::PeerClosed) => {
                let ev = WorkerEvent::Failed {
                    error: "worker process exited unexpectedly".to_string(),
                };
                apply_event_to_record(jobs, job_id, &ev);
                channel.publish(ev);
                return PassResult::PeerClosed;
            }
            Err(e) => {
                let ev = WorkerEvent::Failed { error: format!("worker IPC error: {e}") };
                apply_event_to_record(jobs, job_id, &ev);
                channel.publish(ev);
                return PassResult::IpcError;
            }
        }
    }
}

/// Hires-fix: base pass → upscale the base PNG to `scale*resolution` → img2img
/// refine pass at that resolution with `denoise` creativity. Both passes run on the
/// same resident worker under one `job_id`; only the refine pass publishes the
/// terminal Done, so the client sees a single job whose final image is the refine.
fn drive_hires_two_pass(
    handle: &mut WorkerHandle,
    channel: &Arc<JobChannel>,
    job_id: &str,
    ctl: &std::sync::mpsc::Receiver<DriverCtl>,
    jobs: &Arc<Mutex<Vec<jobs::JobEntry>>>,
    base_params: &JobParams,
    scale: f64,
    denoise: f64,
) -> JobOutcome {
    let base_path = match drive_one_pass(handle, channel, job_id, ctl, jobs, false) {
        PassResult::Done(p) => p,
        PassResult::Terminal => return JobOutcome::Terminal,
        PassResult::PeerClosed => return JobOutcome::PeerClosed,
        PassResult::IpcError => return JobOutcome::IpcError,
        PassResult::ChannelClosed => return JobOutcome::ChannelClosed,
    };

    // The worker's img2img VAE encoder only supports the 512 and 1024 latent
    // grids, and zimage start() admits only 512x512 / 1024x1024 (square). So snap
    // the refine target to the nearest supported square size: scale that lands at
    // >=768 px refines at 1024, otherwise at 512. This keeps every hires job on a
    // valid encoder grid (no >1024 alloc, no unsupported-grid fail) regardless of
    // the requested scale.
    let raw = ((base_params.width.max(base_params.height) as f64) * scale).round() as i64;
    let target = if raw >= 768 { 1024 } else { 512 };
    let sw = target;
    let sh = target;
    let hires_src = match upscale_png(&base_path, sw as u32, sh as u32) {
        Ok(p) => p,
        Err(e) => {
            let ev = WorkerEvent::Failed { error: format!("hires upscale failed: {e}") };
            apply_event_to_record(jobs, job_id, &ev);
            channel.publish(ev);
            return JobOutcome::Terminal;
        }
    };

    // Refine pass = img2img from the upscaled base. Cloning base_params (which has no
    // hires state — that lives on the JobEntry) means no recursion.
    let mut p2 = base_params.clone();
    p2.init_image = hires_src.clone();
    p2.creativity = denoise;
    p2.width = sw;
    p2.height = sh;
    tracing::info!(%job_id, base_w = base_params.width, hires_w = sw, "hires refine pass");
    if let Err(e) = handle.send_start(&p2) {
        let _ = std::fs::remove_file(&hires_src);
        let ev = WorkerEvent::Failed { error: format!("hires pass-2 send_start failed: {e}") };
        apply_event_to_record(jobs, job_id, &ev);
        channel.publish(ev);
        return JobOutcome::PeerClosed;
    }
    let outcome = match drive_one_pass(handle, channel, job_id, ctl, jobs, true) {
        PassResult::Done(_) | PassResult::Terminal => JobOutcome::Terminal,
        PassResult::PeerClosed => JobOutcome::PeerClosed,
        PassResult::IpcError => JobOutcome::IpcError,
        PassResult::ChannelClosed => JobOutcome::ChannelClosed,
    };
    // The upscaled init was only needed to seed the refine pass; drop it so it
    // neither accumulates on disk nor lingers as a stray file in out_dir.
    let _ = std::fs::remove_file(&hires_src);
    outcome
}

/// Upscale a PNG to `w x h` (Lanczos3) and write it beside the source with a
/// `hires_src_` prefix. The prefix matters: the gallery scanner only picks up
/// files matching `job-*.png`, so `hires_src_job-XXXX.png` is invisible to it
/// (no phantom gallery item). The caller deletes it after the refine pass. Uses
/// the `image` crate (already a server dep for gallery thumbnails).
fn upscale_png(src: &str, w: u32, h: u32) -> Result<String, String> {
    let img = image::open(src).map_err(|e| e.to_string())?;
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
    let p = std::path::Path::new(src);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("hires");
    let dst = match p.parent() {
        Some(dir) => dir.join(format!("hires_src_{stem}.png")),
        None => std::path::PathBuf::from(format!("hires_src_{stem}.png")),
    };
    let dst = dst.to_string_lossy().into_owned();
    resized
        .save_with_format(&dst, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(dst)
}

/// Record the current in-flight job id (None when idle).
fn set_in_flight(in_flight: &Arc<Mutex<Option<String>>>, v: Option<String>) {
    if let Ok(mut g) = in_flight.lock() {
        *g = v;
    }
}

/// Schedule eviction of a terminal job's channel after the grace window so a late
/// WS subscriber can still replay the full run. Runs on a detached timer thread.
fn schedule_evict(registry: &Registry, job_id: &str) {
    let registry = registry.clone();
    let job_id = job_id.to_string();
    std::thread::spawn(move || {
        std::thread::sleep(TERMINAL_GRACE);
        if let Ok(mut map) = registry.lock() {
            map.remove(&job_id);
        }
    });
}

// ── handlers ──────────────────────────────────────────────────────────────────

/// POST /v1/generate — accept the RAW request JSON so a Comfy/Swarm `workflow`
/// graph survives. If the body carries a `workflow` key, lower it through the
/// serenity-graph executor FIRST (flattening the graph into the flat keys the rest
/// of this handler reads); a lowering error is surfaced with its own HTTP status
/// (501 unsupported / 422 bad request) and the message as the body — NOT a 500 and
/// NOT a silent fallback to txt2img. On success (or for a body without `workflow`),
/// build JobParams (default + overrides + job_id + out_dir), register a broadcast
/// channel, enqueue, return {"job_id":...}.
async fn post_generate(
    State(st): State<AppState>,
    Json(mut req_value): Json<serde_json::Value>,
) -> Response {
    // ── workflow lowering (control-plane handles the full graph) ──
    // If the incoming request contains a `workflow` key, run it through the typed
    // executor BEFORE building JobParams. lower_request mutates `req_value` in place,
    // replacing `workflow` with the flat params it derived. On Err(GraphError) we
    // fail LOUD with the structured status (501/422) so an unsupported/active-but-
    // unknown node never silently degrades into a plain txt2img.
    let has_workflow = req_value
        .get("workflow")
        .map(|w| !w.is_null())
        .unwrap_or(false);
    if has_workflow {
        if let Err(err) = lower_request(&mut req_value) {
            let status = StatusCode::from_u16(err.http_status())
                .unwrap_or(StatusCode::NOT_IMPLEMENTED);
            tracing::warn!(
                status = err.http_status(),
                "workflow lowering rejected: {err}"
            );
            // Body is the GraphError message (mirrors the Mojo daemon's text body);
            // do NOT 500.
            return (status, err.to_string()).into_response();
        }
    }

    // Keep the (lowered) request as the genparams body the worker embeds as the PNG's
    // serenity.genparams.v1 tEXt chunk — that's what /v1/gallery + SerenityUI's
    // "reuse params" read back. We inject schema + the server job_id below.
    let genparams_value = req_value.clone();

    // The request is now flat (either it always was, or lower_request flattened it).
    // Deserialize the flat shape; a malformed core (missing model/prompt) is a 400.
    let req: GenerateRequest = match serde_json::from_value(req_value) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("invalid request body: {e}")})),
            )
                .into_response();
        }
    };

    // job-XXXX id (the daemon's scheme, so gallery `job-*.png` + jobs.db stay
    // coherent). next_id was initialized to max_prior_id, so the first new job is
    // max_prior_id+1 and never collides with a prior row.
    let n = st.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let job_id = format!("job-{n:04}");

    // default() then override the provided fields.
    let mut params = JobParams::default();
    params.job_id = job_id.clone();
    params.model = req.model;
    params.prompt = req.prompt;
    if let Some(v) = req.negative {
        params.negative = v;
    }
    if let Some(v) = req.width {
        params.width = v;
    }
    if let Some(v) = req.height {
        params.height = v;
    }
    if let Some(v) = req.steps {
        params.steps = v;
    }
    if let Some(v) = req.seed {
        params.seed = v;
    }
    if let Some(v) = req.sampler {
        params.sampler = v;
    }
    if let Some(v) = req.scheduler {
        params.scheduler = v;
    }
    if let Some(v) = req.cfg {
        params.cfg = v;
    }
    // Workflow-lowered extras (only present after lower_request).
    if let Some(v) = req.sigma_shift {
        params.sigma_shift = v;
    }
    if let Some(v) = req.cfg_override {
        params.cfg_override = v;
    }
    if let Some(v) = req.cfg_override_start_percent {
        params.cfg_override_start_percent = v;
    }
    if let Some(v) = req.cfg_override_end_percent {
        params.cfg_override_end_percent = v;
    }
    if let Some(v) = req.images {
        params.images = v;
    }
    if let Some(v) = req.creativity {
        params.creativity = v;
    }
    if let Some(v) = req.workflow_save_prefix {
        params.workflow_save_prefix = v;
    }
    if let Some(v) = req.loras {
        params.loras = v;
    }
    // img2img / inpaint passthrough — without these the worker silently runs
    // txt2img even when the request carried an init/mask (measured 2026-06-14).
    if let Some(v) = req.init_image {
        params.init_image = v;
    }
    if let Some(v) = req.mask_image {
        params.mask_image = v;
    }
    if let Some(v) = req.lanpaint_mask_channel {
        params.lanpaint_mask_channel = v;
    }
    // Advanced-sampling knobs — forwarded to the worker wire verbatim. The worker
    // decides per-knob whether it can honor it; unsupported ones produce a one-line
    // warning on the worker (never a silent drop). See zimage_backend warn helper.
    if let Some(v) = req.clip_skip {
        params.clip_skip = v;
    }
    if let Some(v) = req.eta {
        params.eta = v;
    }
    if let Some(v) = req.sigma_min {
        params.sigma_min = v;
    }
    if let Some(v) = req.sigma_max {
        params.sigma_max = v;
    }
    if let Some(v) = req.restart_sampling {
        params.restart_sampling = v;
    }
    if let Some(v) = req.vae {
        params.vae = v;
    }
    // Hires-fix lives on the JobEntry (control plane), never in the worker wire.
    // Clamp hard: an unbounded scale would drive a multi-TB image-crate alloc on
    // the driver thread (OOM kills the desktop, per CLAUDE.md); NaN -> disabled.
    let hires_scale = match req.hires_scale {
        Some(v) if v.is_finite() => v.clamp(1.0, 4.0),
        _ => 1.0,
    };
    let hires_denoise = match req.hires_denoise {
        Some(v) if v.is_finite() => v.clamp(0.0, 1.0),
        _ => 0.4,
    };
    params.out_dir = st.out_dir.to_string_lossy().into_owned();

    // Build the genparams the worker embeds in the PNG tEXt: the lowered request +
    // schema + the server-assigned job_id (the UI's "reuse params" round-trips this,
    // and /v1/jobs/gallery read it back). Without it the worker embeds an empty chunk.
    {
        let mut gp = genparams_value;
        if let Some(obj) = gp.as_object_mut() {
            obj.entry("schema").or_insert(json!("serenity.genparams.v1"));
            obj.insert("job_id".into(), json!(job_id));
            params.params_json = serde_json::to_string(&gp).unwrap_or_default();
        }
    }

    // Register the per-job channel (history + broadcast) BEFORE the JobEntry becomes
    // promotable, so the driver (which looks the channel up by id) finds it and a fast
    // WS subscriber can attach before any event is produced — the HISTORY replays even
    // if the whole job finishes before the client connects.
    let channel = JobChannel::new();
    {
        let mut map = match st.registry.lock() {
            Ok(m) => m,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "registry poisoned"})),
                )
                    .into_response();
            }
        };
        map.insert(job_id.clone(), channel);
    }

    // Push the JobEntry (queued) to the shared JobBook. The driver promotes the FIRST
    // active-queued entry; /v1/jobs shows it live; /v1/reorder + /v1/remove mutate it.
    {
        let record = jobs::JobRecord {
            id: job_id.clone(),
            created: httpdate::fmt_http_date(SystemTime::now()),
            model: params.model.clone(),
            state: "queued".to_string(),
            progress: 0,
            step: 0,
            total: params.steps,
            image_index: 0,
            image_count: params.images.max(1),
            output_path: String::new(),
            error: String::new(),
            params_json: String::new(),
        };
        match st.jobs.lock() {
            Ok(mut book) => book.push(jobs::JobEntry {
                record,
                params,
                cancel_requested: false,
                hires_scale,
                hires_denoise,
            }),
            Err(_) => {
                if let Ok(mut map) = st.registry.lock() {
                    map.remove(&job_id);
                }
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "job book poisoned"})),
                )
                    .into_response();
            }
        }
    }

    // Wake the driver to promote the new job.
    if st.ctl.send(DriverCtl::Wake).is_err() {
        if let Ok(mut map) = st.registry.lock() {
            map.remove(&job_id);
        }
        if let Ok(mut book) = st.jobs.lock() {
            book.retain(|e| e.record.id != job_id);
        }
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "worker driver unavailable"})),
        )
            .into_response();
    }

    (StatusCode::OK, Json(json!({ "job_id": job_id }))).into_response()
}

/// POST /v1/cancel — body `{"job":<id>}`. Sends a cancel to the worker IF this job
/// is the one currently in flight: 200 `{accepted:true}` when accepted, 404 when the
/// job is unknown or not in flight (single-GPU: only the in-flight job is cancelable).
async fn post_cancel(
    State(st): State<AppState>,
    Json(req): Json<CancelRequest>,
) -> Response {
    let cur = match st.in_flight.lock() {
        Ok(g) => g.clone(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "in_flight poisoned"})),
            )
                .into_response();
        }
    };

    if cur.as_deref() != Some(req.job.as_str()) {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "accepted": false, "job": req.job, "reason": "not in flight" })),
        )
            .into_response();
    }

    // Hand the cancel to the driver thread (it owns the WorkerHandle). If the driver
    // is gone, report unavailable.
    if st.ctl.send(DriverCtl::Cancel(req.job.clone())).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "worker driver unavailable"})),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        Json(json!({ "accepted": true, "job": req.job })),
    )
        .into_response()
}

/// POST /v1/cancel/:id — the PATH form SerenityUI's per-job cancel uses (daemon_cancel).
/// Matches the daemon: 404 unknown, 409 already-terminal, else 200
/// {job_id, state, cancel_requested:true} where `state` is the pre-cancel state. A QUEUED
/// job is finalized to cancelled right here (so the driver's `state=="queued"` promote
/// skips it); a RUNNING job's cancel is forwarded to the worker, which emits Cancelled.
async fn post_cancel_path(State(st): State<AppState>, Path(id): Path<String>) -> Response {
    let djson = |status: StatusCode, v: serde_json::Value| -> Response {
        (
            status,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&v).unwrap_or_else(|_| String::from("{}")),
        )
            .into_response()
    };
    // Classify under the JobBook lock; finalize a queued cancel, mark a running cancel.
    let prev_state = {
        let mut book = match st.jobs.lock() {
            Ok(b) => b,
            Err(_) => return djson(StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": "job book poisoned"})),
        };
        match book.iter_mut().find(|e| e.record.id == id) {
            None => return djson(StatusCode::NOT_FOUND, json!({"detail": format!("no such job: {id}")})),
            Some(e) => {
                let s = e.record.state.clone();
                if s == "done" || s == "failed" || s == "cancelled" {
                    return djson(StatusCode::CONFLICT, json!({"detail": format!("job already {s}")}));
                }
                e.cancel_requested = true;
                if s == "queued" {
                    e.record.state = "cancelled".to_string(); // the driver won't promote it now
                }
                s
            }
        }
    };
    if prev_state == "queued" {
        // finalize: notify any WS subscriber, evict the channel after grace, persist.
        if let Some(ch) = st.registry.lock().ok().and_then(|m| m.get(&id).cloned()) {
            ch.publish(WorkerEvent::Cancelled);
        }
        schedule_evict(&st.registry, &id);
        if let Ok(b) = st.jobs.lock() {
            jobs::save_jobs_db(&st.prior, &b, &st.out_dir.join("jobs.db"));
        }
    } else {
        // running: hand the cancel to the driver thread (owns the WorkerHandle).
        let _ = st.ctl.send(DriverCtl::Cancel(id.clone()));
    }
    djson(StatusCode::OK, json!({ "job_id": id, "state": prev_state, "cancel_requested": true }))
}

/// GET /v1/progress?job=<id> — upgrade to WebSocket, REPLAY all buffered events for
/// this job, THEN stream live ones until a terminal event, then close.
///
/// The snapshot+subscribe is done atomically under the JobChannel lock so a frame
/// published concurrently can't slip through the gap (WS subscribe-race fix). A
/// client that connects AFTER a fast job already finished still gets the full
/// history (incl. the terminal Done) replayed, because the entry is retained for a
/// grace window past terminal.
// ---- Serenity Studio static frontend + browser-fetchable result images ----
const CANVAS_DIR: &str = "/home/alex/mojodiffusion/serenity-server/canvas";

fn static_content_type(p: &std::path::Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}

fn serve_static_file(base: &std::path::Path, rel: &str) -> Response {
    if rel.contains("..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let mut p = base.to_path_buf();
    for seg in rel.split('/').filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    match std::fs::read(&p) {
        Ok(bytes) => (
            [
                (axum::http::header::CONTENT_TYPE, static_content_type(&p)),
                (axum::http::header::CACHE_CONTROL, "no-store, must-revalidate"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

/// Fallback: serve the Konva frontend (index.html + js/css/lib) from CANVAS_DIR.
async fn serve_canvas(uri: axum::http::Uri) -> Response {
    let path = uri.path();
    let rel = if path == "/" || path.is_empty() {
        "index.html"
    } else {
        path.trim_start_matches('/')
    };
    serve_static_file(std::path::Path::new(CANVAS_DIR), rel)
}

/// Serve a result image (out_dir/<path>) so the browser can display it.
async fn serve_out(
    State(st): State<AppState>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> Response {
    serve_static_file(st.out_dir.as_path(), &path)
}

async fn ws_progress(
    State(st): State<AppState>,
    Query(q): Query<ProgressQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    let job_id = q.job;

    // Atomically snapshot history + subscribe to live, while the channel still
    // exists. If the job is unknown/evicted, we have nothing to replay.
    let prepared = {
        let map = match st.registry.lock() {
            Ok(m) => m,
            Err(_) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, "registry poisoned").into_response();
            }
        };
        map.get(&job_id).map(|ch| ch.snapshot_and_subscribe())
    };

    ws.on_upgrade(move |socket| handle_progress_socket(socket, job_id, prepared))
}

/// Drive one WS connection: replay the buffered HISTORY first, then forward live
/// broadcast events as text frames until a terminal event (or lag/closed), close.
async fn handle_progress_socket(
    mut socket: WebSocket,
    job_id: String,
    prepared: Option<(
        Vec<WorkerEvent>,
        tokio::sync::broadcast::Receiver<WorkerEvent>,
        bool,
    )>,
) {
    let (history, mut rx, already_terminal) = match prepared {
        Some(p) => p,
        None => {
            // Unknown or already-evicted job: tell the client and close.
            let body = json!({
                "ev": "failed",
                "error": format!("unknown or finished job {job_id}")
            })
            .to_string();
            let _ = socket.send(Message::Text(body)).await;
            let _ = socket.close().await;
            return;
        }
    };

    // 1. Replay buffered history. If the run already finished (terminal present in
    //    history), replaying it is sufficient — every frame incl. Done is delivered.
    for ev in &history {
        let text = match serde_json::to_string(ev) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(%job_id, "serialize replay event failed: {e}");
                let _ = socket.close().await;
                return;
            }
        };
        if socket.send(Message::Text(text)).await.is_err() {
            // Client hung up during replay.
            let _ = socket.close().await;
            return;
        }
    }
    if already_terminal {
        // Whole run was in history (incl. terminal); nothing live remains.
        let _ = socket.close().await;
        return;
    }

    // 2. Stream live events. The receiver was subscribed UNDER the same lock that
    //    snapshotted `history`, so events already in `history` may also surface here;
    //    they're harmless duplicates only if the timing overlapped exactly — but the
    //    snapshot is taken with `tx.subscribe()` first, so `rx` only yields events
    //    published AFTER subscribe, and `history` holds those published BEFORE. There
    //    is no overlap: publish() pushes-then-sends under the lock, and subscribe()
    //    happens under the same lock, so a given event is in exactly one of the two.
    loop {
        match rx.recv().await {
            Ok(ev) => {
                let terminal = ev.is_terminal();
                let text = match serde_json::to_string(&ev) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(%job_id, "serialize event failed: {e}");
                        break;
                    }
                };
                if socket.send(Message::Text(text)).await.is_err() {
                    break; // client hung up
                }
                if terminal {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                // Sender dropped without our seeing terminal. Nothing more is coming.
                break;
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                tracing::warn!(%job_id, skipped, "ws subscriber lagged; some events dropped");
                // Keep going; rx is now positioned past the dropped events.
            }
        }
    }

    let _ = socket.close().await;
}

/// GET /v1/health — the daemon's shape `{status,backend,model,resident}` so SerenityUI's
/// daemon bridge (`ok = status=="ok"`) treats the Rust server as the live daemon on :7801
/// and routes generate/jobs/cancel here instead of falling back to per-job CLI spawn.
async fn get_health(State(st): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "backend": st.backend_name,
        "model": "-",
        "resident": "",
    }))
}

/// `/v1/samplers` body — the daemon's static SwarmUI/Comfy sampler catalog
/// (`swarmui_sampler_registry_json()`), captured byte-for-byte as a versioned
/// asset so the Rust surface is parity-identical. It is pinned data, not logic,
/// so embedding the exact bytes IS the faithful port (verified by oracle diff vs
/// `serenity_daemon stub`).
const SAMPLERS_V1: &str = include_str!("assets/samplers_v1.json");

/// GET /v1/samplers — the pinned sampler catalog, byte-identical to the daemon.
async fn get_samplers() -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        SAMPLERS_V1,
    )
        .into_response()
}

// ── /v1/state + /v1/presets — file-backed UI state & named presets ──────────────
// Faithful ports of serenity_daemon.mojo's handlers (_load_state_doc /
// _load_presets_doc / _state_doc_from_body / _upsert_preset_doc / _delete_preset_doc).
// JSON is emitted via serde_json::to_string which, with the workspace-unified
// `preserve_order` feature, produces compact insertion-ordered bytes == the daemon's
// `dumps()`. Error bodies match the daemon's `{"detail": <msg>}` shape. State files
// live under <out_dir>/state/ (== the daemon's output/serenity_daemon/state/ when
// run with that out-dir), so UI state carries across the daemon→Rust cutover.

/// Serialize `doc` as compact JSON with content-type application/json.
fn json_compact(status: StatusCode, doc: &serde_json::Value) -> Response {
    let body = serde_json::to_string(doc).unwrap_or_else(|_| String::from("{}"));
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

/// Daemon-shape error: `{"detail": <msg>}`.
fn error_detail(status: StatusCode, detail: &str) -> Response {
    json_compact(status, &json!({ "detail": detail }))
}

fn state_path(out_dir: &std::path::Path) -> PathBuf {
    out_dir.join("state").join("last_state.json")
}
fn presets_path(out_dir: &std::path::Path) -> PathBuf {
    out_dir.join("state").join("presets.json")
}

/// `_load_state_doc`: a valid persisted state, else the default empty doc.
fn load_state_doc(out_dir: &std::path::Path) -> serde_json::Value {
    std::fs::read_to_string(state_path(out_dir))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .filter(|d| d.is_object() && d.get("state").map_or(false, |s| s.is_object()))
        .unwrap_or_else(|| json!({ "schema": "serenity.ui_state.v1", "state": {} }))
}

/// `_load_presets_doc`: a valid persisted presets doc, else the default empty doc.
fn load_presets_doc(out_dir: &std::path::Path) -> serde_json::Value {
    std::fs::read_to_string(presets_path(out_dir))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .filter(|d| d.is_object() && d.get("presets").map_or(false, |p| p.is_array()))
        .unwrap_or_else(|| json!({ "schema": "serenity.presets.v1", "presets": [] }))
}

/// GET /v1/state.
async fn get_state(State(st): State<AppState>) -> Response {
    json_compact(StatusCode::OK, &load_state_doc(&st.out_dir))
}

/// POST /v1/state — `_state_doc_from_body` then persist; returns the stored doc.
async fn post_state(State(st): State<AppState>, body: String) -> Response {
    let obj: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "state body must be a JSON object"),
    };
    if !obj.is_object() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "state body must be a JSON object");
    }
    let state = match obj.get("state") {
        Some(s) if s.is_object() => s.clone(),
        Some(_) => return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "'state' must be an object"),
        None => obj.clone(),
    };
    let doc = json!({ "schema": "serenity.ui_state.v1", "state": state });
    let dir = st.out_dir.join("state");
    if std::fs::create_dir_all(&dir).is_err() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot create state dir");
    }
    if std::fs::write(state_path(&st.out_dir), serde_json::to_string(&doc).unwrap()).is_err() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot persist state");
    }
    json_compact(StatusCode::OK, &doc)
}

/// `params` = body["params"] (must be object) else the whole body.
fn preset_params_from_body(body: &serde_json::Value) -> Result<serde_json::Value, &'static str> {
    match body.get("params") {
        Some(p) if p.is_object() => Ok(p.clone()),
        Some(_) => Err("'params' must be an object"),
        None => Ok(body.clone()),
    }
}

fn preset_entry(name: &str, params: serde_json::Value) -> serde_json::Value {
    json!({ "name": name, "params": params })
}

fn preset_by_name(doc: &serde_json::Value, name: &str) -> Option<serde_json::Value> {
    doc.get("presets")?
        .as_array()?
        .iter()
        .find(|e| e.get("name").and_then(|n| n.as_str()) == Some(name))
        .cloned()
}

/// `_upsert_preset_doc`: replace-or-append the named preset, preserving order.
fn upsert_preset_doc(doc: &serde_json::Value, name: &str, params: serde_json::Value) -> serde_json::Value {
    let mut out = Vec::new();
    let mut replaced = false;
    if let Some(arr) = doc.get("presets").and_then(|p| p.as_array()) {
        for ent in arr {
            if ent.get("name").and_then(|n| n.as_str()) == Some(name) {
                out.push(preset_entry(name, params.clone()));
                replaced = true;
            } else {
                out.push(ent.clone());
            }
        }
    }
    if !replaced {
        out.push(preset_entry(name, params));
    }
    json!({ "schema": "serenity.presets.v1", "presets": out })
}

/// GET /v1/presets — the full presets doc.
async fn get_presets(State(st): State<AppState>) -> Response {
    json_compact(StatusCode::OK, &load_presets_doc(&st.out_dir))
}

/// GET /v1/presets/:name — one preset, 404 if absent.
async fn get_preset_one(State(st): State<AppState>, Path(name): Path<String>) -> Response {
    match preset_by_name(&load_presets_doc(&st.out_dir), &name) {
        Some(p) => json_compact(StatusCode::OK, &p),
        None => error_detail(StatusCode::NOT_FOUND, &format!("no such preset: {name}")),
    }
}

fn upsert_and_save(out_dir: &std::path::Path, name: &str, params: serde_json::Value) -> Response {
    let doc = load_presets_doc(out_dir);
    let updated = upsert_preset_doc(&doc, name, params);
    let dir = out_dir.join("state");
    if std::fs::create_dir_all(&dir).is_err() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot create state dir");
    }
    if std::fs::write(presets_path(out_dir), serde_json::to_string(&updated).unwrap()).is_err() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot persist presets");
    }
    // daemon returns the upserted preset entry
    match preset_by_name(&updated, name) {
        Some(p) => json_compact(StatusCode::OK, &p),
        None => error_detail(StatusCode::UNPROCESSABLE_ENTITY, "preset upsert failed"),
    }
}

/// POST /v1/presets — body carries `name` (required) + `params`.
async fn post_presets_root(State(st): State<AppState>, body: String) -> Response {
    let obj: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "preset body must be a JSON object"),
    };
    let name = match obj.get("name").and_then(|n| n.as_str()) {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "'name' (string) is required"),
    };
    if obj.get("params").is_none() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "'params' (object) is required");
    }
    let params = match preset_params_from_body(&obj) {
        Ok(p) => p,
        Err(e) => return error_detail(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    upsert_and_save(&st.out_dir, &name, params)
}

/// POST /v1/presets/:name — name from path, params from body.
async fn post_preset_named(State(st): State<AppState>, Path(name): Path<String>, body: String) -> Response {
    if name.is_empty() || name.contains('/') {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "invalid preset name in path");
    }
    let obj: serde_json::Value =
        serde_json::from_str(&body).unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
    let params = match preset_params_from_body(&obj) {
        Ok(p) => p,
        Err(e) => return error_detail(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    upsert_and_save(&st.out_dir, &name, params)
}

/// DELETE /v1/presets/:name — `_delete_preset_doc`; 404 if absent.
async fn delete_preset(State(st): State<AppState>, Path(name): Path<String>) -> Response {
    let doc = load_presets_doc(&st.out_dir);
    if preset_by_name(&doc, &name).is_none() {
        return error_detail(StatusCode::NOT_FOUND, &format!("no such preset: {name}"));
    }
    let kept: Vec<serde_json::Value> = doc
        .get("presets")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|e| e.get("name").and_then(|n| n.as_str()) != Some(name.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let updated = json!({ "schema": "serenity.presets.v1", "presets": kept });
    let dir = st.out_dir.join("state");
    let _ = std::fs::create_dir_all(&dir);
    if std::fs::write(presets_path(&st.out_dir), serde_json::to_string(&updated).unwrap()).is_err() {
        return error_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot persist presets");
    }
    json_compact(
        StatusCode::OK,
        &json!({ "name": name, "deleted": true, "presets": updated.get("presets").cloned().unwrap_or(json!([])) }),
    )
}

// ── main ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod endpoint_tests {
    use super::*;

    // Locks the daemon-parity shapes (verified byte-identical vs `serenity_daemon
    // stub` via the oracle-diff harness): preset entry = {name,params}, upsert
    // replaces in place + preserves order, presets doc = {schema,presets}.
    #[test]
    fn preset_upsert_and_replace_shape() {
        let empty = json!({ "schema": "serenity.presets.v1", "presets": [] });
        let one = upsert_preset_doc(&empty, "a", json!({ "cfg": 4.5 }));
        assert_eq!(
            serde_json::to_string(&one).unwrap(),
            r#"{"schema":"serenity.presets.v1","presets":[{"name":"a","params":{"cfg":4.5}}]}"#
        );
        // replace-in-place keeps the single entry, updates params
        let replaced = upsert_preset_doc(&one, "a", json!({ "cfg": 9.9 }));
        assert_eq!(
            serde_json::to_string(&replaced).unwrap(),
            r#"{"schema":"serenity.presets.v1","presets":[{"name":"a","params":{"cfg":9.9}}]}"#
        );
        // append second preserves order
        let two = upsert_preset_doc(&replaced, "b", json!({ "x": 1 }));
        assert!(serde_json::to_string(&two)
            .unwrap()
            .contains(r#""presets":[{"name":"a","params":{"cfg":9.9}},{"name":"b","params":{"x":1}}]"#));
        assert!(preset_by_name(&two, "a").is_some());
        assert!(preset_by_name(&two, "zzz").is_none());
    }

    #[test]
    fn preset_params_default_to_whole_body() {
        // no "params" key -> the whole body is the params (daemon _preset_params_from_body)
        let body = json!({ "cfg": 3.0, "steps": 8 });
        assert_eq!(preset_params_from_body(&body).unwrap(), body);
        // "params" present but not object -> error
        assert!(preset_params_from_body(&json!({ "params": 5 })).is_err());
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // 1. CLI.
    let (worker_bin, port, out_dir, worker_args) = parse_args();
    // Backend identity for /v1/health: the --kind, else the worker binary name with the
    // "serenity_worker_" prefix stripped (e.g. .../serenity_worker_zimage -> "zimage").
    let backend_name = if worker_args.len() >= 2 && worker_args[0] == "worker" {
        worker_args[1].clone()
    } else {
        worker_bin
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.strip_prefix("serenity_worker_").unwrap_or(s).to_string())
            .unwrap_or_else(|| BACKEND_NAME.to_string())
    };
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| anyhow::anyhow!("create out_dir {}: {e}", out_dir.display()))?;
    let out_dir = std::fs::canonicalize(&out_dir).unwrap_or(out_dir);
    tracing::info!(
        worker = %worker_bin.display(),
        port,
        out_dir = %out_dir.display(),
        "serenity-server starting"
    );

    // Load prior jobs.db rows ONCE (history half of /v1/jobs) + base the new-id
    // counter at max_prior_id so generated ids never collide with a prior row (F7).
    let prior = Arc::new(jobs::load_prior_rows(&out_dir.join("jobs.db")));
    let next_id = Arc::new(AtomicU64::new(jobs::max_prior_id(&prior) as u64));
    let job_book: Arc<Mutex<Vec<jobs::JobEntry>>> = Arc::new(Mutex::new(Vec::new()));
    tracing::info!(prior_rows = prior.len(), next_id_base = next_id.load(Ordering::Relaxed), "jobs.db loaded");

    // 2. Worker-driver thread (owns WorkerHandle; handshake waits for Ready). It
    //    receives both jobs and cancels over one control channel so cancel runs on
    //    the thread that owns the handle.
    let (ctl_tx, ctl_rx) = std::sync::mpsc::channel::<DriverCtl>();
    let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
    let in_flight: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    {
        let registry = registry.clone();
        let in_flight = in_flight.clone();
        let worker_bin = worker_bin.clone();
        let job_book = job_book.clone();
        let prior_c = prior.clone();
        let db_path = out_dir.join("jobs.db");
        std::thread::Builder::new()
            .name("worker-driver".into())
            .spawn(move || {
                run_worker_driver(worker_bin, worker_args, ctl_rx, registry, in_flight, job_book, prior_c, db_path)
            })
            .map_err(|e| anyhow::anyhow!("spawn worker-driver thread: {e}"))?;
    }

    let state = AppState {
        ctl: ctl_tx,
        registry,
        in_flight,
        next_id,
        out_dir,
        prior,
        jobs: job_book,
        backend_name,
    };

    // 3. Router.
    let app = Router::new()
        .route("/v1/generate", post(post_generate))
        .route("/v1/grid", post(grid::post_grid))
        .route("/v1/cancel", post(post_cancel))
        .route("/v1/cancel/:id", post(post_cancel_path))
        .route("/v1/progress", get(ws_progress))
        .route("/v1/health", get(get_health))
        .route("/v1/samplers", get(get_samplers))
        .route("/v1/models", get(models::get_models))
        .route("/v1/jobs", get(jobs::get_jobs))
        .route("/v1/job/:id", get(jobs::get_job_one))
        .route("/v1/reorder", post(jobs::post_reorder))
        .route("/v1/remove", post(jobs::post_remove))
        .route("/v1/video", get(video::get_video).post(video::post_video))
        .route("/v1/video/probe", get(video::get_video_probe))
        .route("/v1/gallery", get(gallery::get_gallery))
        .route("/v1/gallery/read", get(gallery::get_gallery_read))
        .route("/v1/gallery/import", post(gallery::post_import))
        .route("/v1/gallery/order", post(gallery::post_order))
        .route("/v1/gallery/:id", get(gallery::get_gallery_one).delete(gallery::delete_item))
        .route("/v1/gallery/:id/rename", post(gallery::post_rename))
        .route("/v1/gallery/:id/favorite", post(gallery::post_favorite))
        .route("/v1/state", get(get_state).post(post_state))
        .route("/v1/presets", get(get_presets).post(post_presets_root))
        .route(
            "/v1/presets/:name",
            get(get_preset_one).post(post_preset_named).delete(delete_preset),
        )
        // --- Serenity Studio Konva frontend (static) + browser-fetchable result images ---
        .route("/out/*path", get(serve_out))
        .fallback(serve_canvas)
        .with_state(state);

    // 4. Serve on 127.0.0.1:<port>.
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("serve: {e}"))?;
    Ok(())
}
