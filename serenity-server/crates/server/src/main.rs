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

/// A job handed to the worker-driver thread: the full JobParams plus the shared
/// channel (history + broadcast) every event for this job is fanned out on.
struct QueuedJob {
    params: JobParams,
    channel: Arc<JobChannel>,
}

/// Per-job event channel: a live `broadcast` sender AND a replayable HISTORY of
/// every event fanned out so far. The driver appends to `history` (and broadcasts)
/// under the SAME lock used by a connecting WS client to snapshot+subscribe, so no
/// frame can leak through the snapshot→subscribe gap (the WS subscribe-race fix).
struct JobChannel {
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
    fn new() -> Arc<Self> {
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
enum DriverCtl {
    /// A new job to run (single-GPU contract: one in flight at a time).
    Job(Box<QueuedJob>),
    /// Cancel the job with this id IF it is the one currently in flight.
    Cancel(String),
}

#[derive(Clone)]
struct AppState {
    /// Control channel to the worker-driver thread (jobs + cancels).
    ctl: std::sync::mpsc::Sender<DriverCtl>,
    /// job_id -> JobChannel (history + live broadcast), for WS subscription/replay.
    registry: Registry,
    /// The job currently in flight (driver writes; cancel handler reads to decide
    /// 200-accepted vs 404-not-in-flight without blocking the driver thread).
    in_flight: Arc<Mutex<Option<String>>>,
    /// Monotonic counter feeding job_id generation (no rand).
    next_id: Arc<AtomicU64>,
    /// Server-configured output directory written into every JobParams.out_dir.
    out_dir: PathBuf,
    /// Prior jobs.db rows loaded ONCE at startup (history half of /v1/jobs).
    prior: Arc<Vec<[String; 6]>>,
    /// Current-session JobRecords (live half of /v1/jobs), ordered by enqueue.
    jobs: Arc<Mutex<Vec<jobs::JobRecord>>>,
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
    images: Option<i64>,
    #[serde(default)]
    creativity: Option<f64>,
    #[serde(default)]
    workflow_save_prefix: Option<String>,
    /// Lowered LoRA overlays — `lower_request` writes the `lora` array.
    #[serde(default, rename = "lora")]
    loras: Option<Vec<LoraSpec>>,
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
    jobs: &Arc<Mutex<Vec<jobs::JobRecord>>>,
    id: &str,
    f: impl FnOnce(&mut jobs::JobRecord),
) {
    if let Ok(mut v) = jobs.lock() {
        if let Some(r) = v.iter_mut().find(|r| r.id == id) {
            f(r);
        }
    }
}

/// Apply a WorkerEvent to the matching JobRecord (state/step/progress/output/error).
fn apply_event_to_record(jobs: &Arc<Mutex<Vec<jobs::JobRecord>>>, id: &str, ev: &WorkerEvent) {
    match ev {
        WorkerEvent::Progress { step, total, .. } => update_record(jobs, id, |r| {
            r.state = "running".to_string();
            r.step = *step;
            if *total > 0 {
                r.total = *total;
                r.progress = step * 100 / total;
            }
        }),
        WorkerEvent::Done { output_path } => update_record(jobs, id, |r| {
            r.state = "done".to_string();
            r.progress = 100;
            r.output_path = output_path.clone();
        }),
        WorkerEvent::Failed { error } => update_record(jobs, id, |r| {
            r.state = "failed".to_string();
            r.error = error.clone();
        }),
        WorkerEvent::Cancelled => update_record(jobs, id, |r| r.state = "cancelled".to_string()),
        WorkerEvent::Ready => {}
    }
}

fn run_worker_driver(
    worker_bin: PathBuf,
    worker_args: Vec<String>,
    ctl: std::sync::mpsc::Receiver<DriverCtl>,
    registry: Registry,
    in_flight: Arc<Mutex<Option<String>>>,
    jobs: Arc<Mutex<Vec<jobs::JobRecord>>>,
) {
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

    // Jobs that arrived while another was in flight (single-GPU: run them in order).
    let mut pending: std::collections::VecDeque<Box<QueuedJob>> =
        std::collections::VecDeque::new();

    loop {
        // Pick the next job to run: a buffered one first, else block on the channel.
        let job = match pending.pop_front() {
            Some(j) => j,
            None => match ctl.recv() {
                Ok(DriverCtl::Job(j)) => j,
                Ok(DriverCtl::Cancel(target)) => {
                    // Cancel arriving while idle: nothing is in flight to cancel.
                    tracing::info!(job = %target, "cancel ignored: not in flight (idle)");
                    continue;
                }
                Err(_) => break, // all senders dropped — server shutting down.
            },
        };

        let job = *job; // unbox once; QueuedJob owns its params + channel now.
        let job_id = job.params.job_id.clone();
        let channel = job.channel;

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
        update_record(&jobs, &job_id, |r| r.state = "running".to_string());

        if let Err(e) = h.send_start(&job.params) {
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
        let outcome = drive_one_job(h, &channel, &job_id, &ctl, &mut pending, &jobs);
        set_in_flight(&in_flight, None);
        schedule_evict(&registry, &job_id);

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
    pending: &mut std::collections::VecDeque<Box<QueuedJob>>,
    jobs: &Arc<Mutex<Vec<jobs::JobRecord>>>,
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
                Ok(DriverCtl::Job(j)) => {
                    // Another job arrived while one is in flight — queue it.
                    pending.push_back(j);
                }
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
    params.out_dir = st.out_dir.to_string_lossy().into_owned();

    // Track a current-session JobRecord (queued) so /v1/jobs shows it live. The
    // driver flips it running -> done/failed/cancelled as worker events flow.
    if let Ok(mut jobs) = st.jobs.lock() {
        jobs.push(jobs::JobRecord {
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
        });
    }

    // Register the per-job channel (history + broadcast) BEFORE enqueueing so a fast
    // WS subscriber can attach before any event is produced — and so the HISTORY
    // exists to replay even if the whole job finishes before the client connects.
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
        map.insert(job_id.clone(), channel.clone());
    }

    // Enqueue to the worker-driver thread.
    if st
        .ctl
        .send(DriverCtl::Job(Box::new(QueuedJob { params, channel })))
        .is_err()
    {
        // Driver thread is gone — pull the registry entry back out.
        if let Ok(mut map) = st.registry.lock() {
            map.remove(&job_id);
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

/// GET /v1/progress?job=<id> — upgrade to WebSocket, REPLAY all buffered events for
/// this job, THEN stream live ones until a terminal event, then close.
///
/// The snapshot+subscribe is done atomically under the JobChannel lock so a frame
/// published concurrently can't slip through the gap (WS subscribe-race fix). A
/// client that connects AFTER a fast job already finished still gets the full
/// history (incl. the terminal Done) replayed, because the entry is retained for a
/// grace window past terminal.
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

/// GET /v1/health — {"backend":"isolated-rust","ok":true}.
async fn get_health() -> Json<serde_json::Value> {
    Json(json!({ "backend": BACKEND_NAME, "ok": true }))
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
    let job_book: Arc<Mutex<Vec<jobs::JobRecord>>> = Arc::new(Mutex::new(Vec::new()));
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
        std::thread::Builder::new()
            .name("worker-driver".into())
            .spawn(move || run_worker_driver(worker_bin, worker_args, ctl_rx, registry, in_flight, job_book))
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
    };

    // 3. Router.
    let app = Router::new()
        .route("/v1/generate", post(post_generate))
        .route("/v1/cancel", post(post_cancel))
        .route("/v1/progress", get(ws_progress))
        .route("/v1/health", get(get_health))
        .route("/v1/samplers", get(get_samplers))
        .route("/v1/models", get(models::get_models))
        .route("/v1/jobs", get(jobs::get_jobs))
        .route("/v1/job/:id", get(jobs::get_job_one))
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
