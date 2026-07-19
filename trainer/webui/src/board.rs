// board.rs — SerenityBoard, built INTO the serenity web trainer supervisor.
// A single Rust binary = training UI + live metrics board, no Python in the path.
//
// Ported from the external Python SerenityBoard development reference. The original
// keeps one board.db PER run directory and a server that scans a logdir; here we
// consolidate into ONE db (webui/board.db) with a run-discriminated scalars table
// so the ported frontend's REST calls map 1:1 (see the schema-mapping table in the
// handoff). The frontend is served verbatim at /board with a fetch/WebSocket shim
// (board/board_boot.js) that namespaces /api/* -> /api/board/* and /ws/live.
//
// Two ingestion sources feed the same tables, deduped by primary key:
//   (a) LIVE hook  — the supervisor's stdout pump hands every parsed progress line
//                    to board::ingest (web-launched runs).
//   (b) TAILER     — a tokio task scanning output/*/train_{cli,web}.log for active
//                    files (mtime < 60s), re-parsing new lines through the SAME
//                    crate::parse_progress (covers CLI runs launched outside the UI).
// Run identity  = basename(workspace_dir); scalar identity = (run, tag, step).
// INSERT OR IGNORE makes both sources idempotent; a terminal status set by
// run_ended is never downgraded by the tailer.

use std::{
    collections::HashMap,
    path::{Path as FsPath, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use rusqlite::{params, Connection};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;

fn output_dir() -> PathBuf {
    std::env::var_os("SERENITY_TRAINER_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(FsPath::parent)
                .expect("trainer repository root")
                .join("output")
        })
}
// scalar tags emitted from a parsed progress line (grouped by "/" in the UI)
const TAG_LOSS: &str = "loss/train_step";
const TAG_GRAD: &str = "grad_norm/train_step";
const TAG_SPS: &str = "perf/s_per_step";
const TAG_EPOCH: &str = "progress/epoch";
// a "running" run whose newest point is older than this reads as "stopped"
const STALE_S: f64 = 120.0;
// a log file is a live training run if touched within this many seconds
const TAILER_ACTIVE_S: u64 = 60;
// skip the backlog of a large pre-existing log the first time we see it
const TAILER_BACKLOG_CAP: u64 = 2 * 1024 * 1024;

#[derive(Clone)]
struct LiveEvent {
    run: String,
    tag: String,
    json: String, // pre-serialized frontend message, sent verbatim over the socket
}

struct Board {
    db: Mutex<Connection>,
    board_dir: PathBuf,
    tx: broadcast::Sender<LiveEvent>,
}

static BOARD: OnceLock<Board> = OnceLock::new();

fn board() -> &'static Board {
    BOARD.get().expect("board::init() not called")
}

fn now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn basename(path: &str) -> String {
    FsPath::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

// ── initialization ─────────────────────────────────────────────────────────

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runs (
    name              TEXT PRIMARY KEY,
    workspace_dir     TEXT NOT NULL,
    source            TEXT NOT NULL,
    preset_id         TEXT,
    status            TEXT NOT NULL,
    start_time        REAL NOT NULL,
    last_wall_time    REAL,
    last_step         INTEGER,
    max_steps         INTEGER,
    active_session_id TEXT,
    hparams_json      TEXT
);
CREATE TABLE IF NOT EXISTS lora_cache (
    path         TEXT PRIMARY KEY,
    mtime        REAL NOT NULL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scalars (
    run       TEXT    NOT NULL,
    tag       TEXT    NOT NULL,
    step      INTEGER NOT NULL,
    wall_time REAL    NOT NULL,
    value     REAL    NOT NULL,
    PRIMARY KEY (run, tag, step)
);
CREATE TABLE IF NOT EXISTS run_notes (
    run        TEXT PRIMARY KEY,
    note       TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scalars_run_tag_step ON scalars(run, tag, step);
";

/// Open webui/board.db, create tables, and spawn the workspace-log tailer.
/// Called once from main() inside the tokio runtime.
pub fn init() {
    let repo = crate::REPO_ROOT;
    let db_path = format!("{repo}/webui/board.db");
    let board_dir = PathBuf::from(format!("{repo}/webui/board"));

    let conn = Connection::open(&db_path).expect("open board.db");
    conn.pragma_update(None, "journal_mode", "WAL").ok();
    conn.pragma_update(None, "synchronous", "NORMAL").ok();
    conn.pragma_update(None, "busy_timeout", 5000).ok();
    conn.execute_batch(SCHEMA).expect("board schema");
    // Older board.db files predate the hparams column; add it if missing (the
    // CREATE above is a no-op for an existing table). Duplicate-column error is
    // the expected already-migrated case and is ignored.
    conn.execute("ALTER TABLE runs ADD COLUMN hparams_json TEXT", []).ok();

    let (tx, _rx) = broadcast::channel(1024);
    let _ = BOARD.set(Board {
        db: Mutex::new(conn),
        board_dir,
        tx,
    });

    tokio::spawn(tailer_loop());
    println!("serenity board on /board (db {db_path})");
}

// ── ingestion (shared by live hook + tailer) ────────────────────────────────

/// INSERT OR IGNORE one scalar point; broadcast a live event iff a new row landed.
fn insert_point(conn: &Connection, run: &str, tag: &str, step: u64, wt: f64, value: f64, sid: &str) {
    if !value.is_finite() {
        return;
    }
    let n = conn
        .execute(
            "INSERT OR IGNORE INTO scalars (run, tag, step, wall_time, value) VALUES (?, ?, ?, ?, ?)",
            params![run, tag, step as i64, wt, value],
        )
        .unwrap_or(0);
    if n > 0 {
        let msg = json!({
            "type": "scalar",
            "run": run,
            "tag": tag,
            "session_id": sid,
            "points": [{"step": step, "wall_time": wt, "value": value}],
        })
        .to_string();
        let _ = board().tx.send(LiveEvent {
            run: run.to_string(),
            tag: tag.to_string(),
            json: msg,
        });
    }
}

/// Core ingest: write the four scalar tags for one parsed step and refresh the run row.
/// `source` marks who observed it first; the run row is upserted, never resurrected
/// out of a terminal status.
fn ingest_step(
    conn: &Connection,
    run: &str,
    workspace: &str,
    source: &str,
    step: u64,
    epoch: u64,
    loss: f64,
    grad_norm: f64,
    s_per_step: f64,
    max_steps: u64,
) {
    let wt = now_s();
    let sid = session_id(conn, run, workspace, source, max_steps);
    if step > 0 {
        insert_point(conn, run, TAG_LOSS, step, wt, loss, &sid);
        insert_point(conn, run, TAG_GRAD, step, wt, grad_norm, &sid);
        if s_per_step > 0.0 {
            insert_point(conn, run, TAG_SPS, step, wt, s_per_step, &sid);
        }
        insert_point(conn, run, TAG_EPOCH, step, wt, epoch as f64, &sid);
    }
    // refresh liveness; bump status to running only if it was already running
    conn.execute(
        "UPDATE runs SET last_wall_time = ?, last_step = ?, max_steps = CASE WHEN ? > 0 THEN ? ELSE max_steps END WHERE name = ?",
        params![wt, step as i64, max_steps as i64, max_steps as i64, run],
    )
    .ok();
    conn.execute(
        "UPDATE runs SET status = 'running' WHERE name = ? AND status = 'running'",
        params![run],
    )
    .ok();
}

/// Ensure a run row exists and return its session id. Creates the row (status
/// running) on first sight; never overwrites an existing row here.
fn session_id(conn: &Connection, run: &str, workspace: &str, source: &str, max_steps: u64) -> String {
    let existing: Option<String> = conn
        .query_row(
            "SELECT active_session_id FROM runs WHERE name = ?",
            params![run],
            |r| r.get(0),
        )
        .ok()
        .flatten();
    if let Some(sid) = existing {
        return sid;
    }
    let start = now_s();
    let sid = format!("s{}", (start * 1000.0) as i64);
    conn.execute(
        "INSERT OR IGNORE INTO runs (name, workspace_dir, source, preset_id, status, start_time, max_steps, active_session_id) \
         VALUES (?, ?, ?, NULL, 'running', ?, ?, ?)",
        params![run, workspace, source, start, max_steps as i64, sid],
    )
    .ok();
    sid
}

/// LIVE hook — called from the stdout pump for every parsed progress line of a
/// web-launched run. `workspace` is the run's absolute workspace dir.
pub fn ingest(workspace: &str, step: u64, epoch: u64, loss: f64, grad_norm: f64, s_per_step: f64, max_steps: u64) {
    if BOARD.get().is_none() {
        return;
    }
    let run = basename(workspace);
    let conn = board().db.lock().unwrap();
    ingest_step(&conn, &run, workspace, "web", step, epoch, loss, grad_norm, s_per_step, max_steps);
}

/// Pull the recipe fields the HParams tab compares from the merged runner config.
/// Only keys that are actually present are stored — nothing is fabricated. The
/// optimizer field is an object (`{"optimizer":"ADAMW",...}`) in every runner
/// config, so its inner name is unwrapped to a plain string.
fn extract_hparams(cfg: &Value, preset_id: &str) -> Value {
    let mut m = serde_json::Map::new();
    for k in [
        "learning_rate", "lora_rank", "lora_alpha", "max_steps",
        "batch_size", "timestep_shift", "quantized_resident",
    ] {
        if let Some(v) = cfg.get(k) {
            m.insert(k.to_string(), v.clone());
        }
    }
    let opt_name = match cfg.get("optimizer") {
        Some(Value::Object(o)) => o.get("optimizer").and_then(|v| v.as_str()).map(String::from),
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };
    if let Some(name) = opt_name {
        m.insert("optimizer".into(), json!(name));
    }
    if !preset_id.is_empty() {
        m.insert("preset_id".into(), json!(preset_id));
    }
    Value::Object(m)
}

/// Called at run launch — upsert the run row as running (web source) and record
/// the merged recipe (learning_rate, lora_rank, ... ) for the HParams tab.
pub fn run_started(workspace: &str, preset_id: &str, max_steps: u64, cfg: &Value) {
    if BOARD.get().is_none() {
        return;
    }
    let run = basename(workspace);
    let start = now_s();
    let sid = format!("s{}", (start * 1000.0) as i64);
    let hparams_json = extract_hparams(cfg, preset_id).to_string();
    let conn = board().db.lock().unwrap();
    conn.execute(
        "INSERT INTO runs (name, workspace_dir, source, preset_id, status, start_time, max_steps, active_session_id, hparams_json) \
         VALUES (?, ?, 'web', ?, 'running', ?, ?, ?, ?) \
         ON CONFLICT(name) DO UPDATE SET \
           workspace_dir = excluded.workspace_dir, source = 'web', preset_id = excluded.preset_id, \
           status = 'running', start_time = excluded.start_time, max_steps = excluded.max_steps, \
           active_session_id = excluded.active_session_id, last_wall_time = NULL, last_step = NULL, \
           hparams_json = excluded.hparams_json",
        params![run, workspace, preset_id, start, max_steps as i64, sid, hparams_json],
    )
    .ok();
}

/// Called at run end (reap) — set the terminal status. "exited" reads as "completed".
pub fn run_ended(workspace: &str, status: &str) {
    if BOARD.get().is_none() {
        return;
    }
    let run = basename(workspace);
    let mapped = if status == "exited" { "completed" } else { status };
    let conn = board().db.lock().unwrap();
    conn.execute(
        "UPDATE runs SET status = ? WHERE name = ?",
        params![mapped, run],
    )
    .ok();
}

// ── workspace-log tailer ────────────────────────────────────────────────────

struct TailState {
    offset: u64,
    scratch: crate::RunInfo,
    workspace: String,
    source: &'static str,
}

async fn tailer_loop() {
    let mut tracked: HashMap<String, TailState> = HashMap::new();
    loop {
        scan_once(&mut tracked);
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

fn scan_once(tracked: &mut HashMap<String, TailState>) {
    let Ok(rd) = std::fs::read_dir(output_dir()) else {
        return;
    };
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let workspace = dir.to_string_lossy().to_string();
        for (fname, source) in [("train_cli.log", "cli"), ("train_web.log", "web")] {
            let log = dir.join(fname);
            let Ok(meta) = std::fs::metadata(&log) else {
                continue;
            };
            // active = touched recently
            let fresh = meta
                .modified()
                .ok()
                .and_then(|m| SystemTime::now().duration_since(m).ok())
                .map(|d| d.as_secs() < TAILER_ACTIVE_S)
                .unwrap_or(false);
            if !fresh {
                continue;
            }
            let len = meta.len();
            let key = log.to_string_lossy().to_string();
            let st = tracked.entry(key.clone()).or_insert_with(|| TailState {
                // skip a large pre-existing backlog; otherwise read from the top
                offset: if len > TAILER_BACKLOG_CAP { len } else { 0 },
                scratch: fresh_runinfo(&workspace),
                workspace: workspace.clone(),
                source,
            });
            if len < st.offset {
                st.offset = 0; // file truncated / rotated
            }
            tail_file(&log, st);
        }
    }
}

fn tail_file(log: &FsPath, st: &mut TailState) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(log) else {
        return;
    };
    if f.seek(SeekFrom::Start(st.offset)).is_err() {
        return;
    }
    let mut buf = String::new();
    let Ok(n) = f.read_to_string(&mut buf) else {
        return;
    };
    if n == 0 {
        return;
    }
    // only consume up to the last newline; keep the partial tail for next scan
    let consumed = match buf.rfind('\n') {
        Some(i) => i + 1,
        None => return,
    };
    let run = basename(&st.workspace);
    let conn = board().db.lock().unwrap();
    for line in buf[..consumed].lines() {
        if crate::parse_progress(line, &mut st.scratch) {
            let s = &st.scratch;
            ingest_step(
                &conn, &run, &st.workspace, st.source,
                s.step, s.epoch, s.loss, s.grad_norm, s.s_per_step, s.total_steps,
            );
        }
    }
    st.offset += consumed as u64;
}

/// A blank RunInfo scratch to feed crate::parse_progress; only the progress
/// fields it touches matter.
fn fresh_runinfo(workspace: &str) -> crate::RunInfo {
    crate::RunInfo {
        id: 0,
        preset_id: String::new(),
        backend: String::new(),
        workspace_dir: workspace.to_string(),
        log_path: String::new(),
        status: "running".into(),
        message: String::new(),
        pid: None,
        step: 0,
        total_steps: 0,
        epoch: 0,
        total_epochs: 0,
        loss: 0.0,
        grad_norm: 0.0,
        s_per_step: 0.0,
        eta: String::new(),
        resume_kind: String::new(),
        last_save: String::new(),
        stage: String::new(),
    }
}

// ── REST endpoints (ported from serenityboard data_provider/app/routes) ──────

fn scalar_rows(conn: &Connection, run: &str, tag: &str) -> Vec<(i64, f64, f64)> {
    let mut out = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT step, wall_time, value FROM scalars WHERE run = ? AND tag = ? ORDER BY step",
    ) {
        if let Ok(rows) = stmt.query_map(params![run, tag], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?))
        }) {
            for row in rows.flatten() {
                out.push(row);
            }
        }
    }
    out
}

/// Port of read_scalars_downsampled: n<=0 => full; else evenly sample n keeping
/// first + last.
fn downsample(rows: Vec<(i64, f64, f64)>, n: i64) -> Vec<(i64, f64, f64)> {
    if n <= 0 || (rows.len() as i64) <= n {
        return rows;
    }
    let n = n.max(3) as usize;
    let total = rows.len();
    let mut idx: Vec<usize> = Vec::with_capacity(n);
    idx.push(0);
    for i in 1..(n - 1) {
        let j = (i as f64) * (total as f64 - 1.0) / (n as f64 - 1.0);
        idx.push(j.round() as usize);
    }
    idx.push(total - 1);
    idx.dedup();
    idx.into_iter().map(|i| rows[i]).collect()
}

fn project_xaxis(rows: &[(i64, f64, f64)], x_axis: &str) -> Value {
    let t0 = rows.first().map(|r| r.1).unwrap_or(0.0);
    let arr: Vec<Value> = rows
        .iter()
        .map(|&(step, wt, v)| match x_axis {
            "wall_time" => json!([wt, wt, v]),
            "relative" => json!([wt - t0, wt, v]),
            _ => json!([step, wt, v]),
        })
        .collect();
    Value::Array(arr)
}

async fn list_runs() -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let now = now_s();
    let mut out: Vec<Value> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT name, source, status, start_time, last_wall_time, last_step, max_steps, active_session_id \
         FROM runs ORDER BY start_time DESC",
    ) {
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f64>(3)?,
                r.get::<_, Option<f64>>(4)?,
                r.get::<_, Option<i64>>(5)?,
                r.get::<_, Option<i64>>(6)?,
                r.get::<_, Option<String>>(7)?,
            ))
        });
        if let Ok(rows) = rows {
            for row in rows.flatten() {
                let (name, source, mut status, start, last_wt, last_step, max_steps, sid) = row;
                if status == "running" {
                    if let Some(wt) = last_wt {
                        if now - wt > STALE_S {
                            status = "stopped".into();
                        }
                    }
                }
                let hparams = match max_steps {
                    Some(ms) => json!({ "max_steps": ms }),
                    None => json!({}),
                };
                out.push(json!({
                    "name": name,
                    "source": source,
                    "status": status,
                    "start_time": start,
                    "last_activity": last_wt,
                    "last_step": last_step,
                    "max_steps": max_steps,
                    "active_session_id": sid,
                    "hparams": hparams,
                }));
            }
        }
    }
    Json(Value::Array(out))
}

async fn tags(Path(run): Path<String>) -> Json<Value> {
    let ws;
    let mut scalars: Vec<String> = Vec::new();
    {
        let conn = board().db.lock().unwrap();
        if let Ok(mut stmt) =
            conn.prepare("SELECT DISTINCT tag FROM scalars WHERE run = ? ORDER BY tag")
        {
            if let Ok(rows) = stmt.query_map(params![run], |r| r.get::<_, String>(0)) {
                scalars = rows.flatten().collect();
            }
        }
        ws = run_workspace(&conn, &run);
    }
    // artifact "tags" = per-image-slot groups discovered on disk (e.g.
    // "turbo_samples/step_0_0"), one series the Artifacts filmstrip slides over.
    let mut artifacts: Vec<String> = collect_artifacts(&ws).into_iter().map(|a| a.tag).collect();
    artifacts.sort();
    artifacts.dedup();
    Json(json!({
        "scalars": scalars,
        "tensors": [], "artifacts": artifacts, "text_events": [], "audio": [],
        "trace_events": [], "eval_suites": [], "pr_curves": [],
        "graphs": [], "meshes": [], "embeddings": [],
    }))
}

async fn metrics(Path(run): Path<String>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let mut scalars: Vec<Value> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT tag, COUNT(*), MAX(step), MAX(wall_time) FROM scalars WHERE run = ? GROUP BY tag ORDER BY tag",
    ) {
        if let Ok(rows) = stmt.query_map(params![run], |r| {
            Ok(json!({
                "tag": r.get::<_, String>(0)?,
                "count": r.get::<_, i64>(1)?,
                "last_step": r.get::<_, Option<i64>>(2)?,
                "last_wall_time": r.get::<_, Option<f64>>(3)?,
            }))
        }) {
            scalars = rows.flatten().collect();
        }
    }
    Json(json!({
        "scalars": scalars,
        "tensors": [], "artifacts": [], "text_events": [], "audio": [],
        "pr_curves": [], "graphs": [], "meshes": [], "embeddings": [],
    }))
}

#[derive(Deserialize)]
struct ScalarQ {
    tag: String,
    #[serde(default)]
    downsample: Option<i64>,
    #[serde(default)]
    x_axis: Option<String>,
}

async fn scalars(Path(run): Path<String>, Query(q): Query<ScalarQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let rows = scalar_rows(&conn, &run, &q.tag);
    let rows = downsample(rows, q.downsample.unwrap_or(5000));
    Json(project_xaxis(&rows, q.x_axis.as_deref().unwrap_or("step")))
}

#[derive(Deserialize)]
struct TagsQ {
    tags: String,
}

async fn scalars_last(Path(run): Path<String>, Query(q): Query<TagsQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let mut out = serde_json::Map::new();
    for tag in q.tags.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let row = conn
            .query_row(
                "SELECT step, wall_time, value FROM scalars WHERE run = ? AND tag = ? ORDER BY step DESC LIMIT 1",
                params![run, tag],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?)),
            )
            .ok();
        if let Some((step, wt, v)) = row {
            out.insert(tag.to_string(), json!({"step": step, "wall_time": wt, "value": v}));
        }
    }
    Json(Value::Object(out))
}

#[derive(Deserialize)]
struct CompareQ {
    tag: String,
    runs: String,
    #[serde(default)]
    downsample: Option<i64>,
    #[serde(default)]
    x_axis: Option<String>,
}

async fn compare_scalars(Query(q): Query<CompareQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let x_axis = q.x_axis.as_deref().unwrap_or("step");
    let mut out = serde_json::Map::new();
    for run in q.runs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let rows = scalar_rows(&conn, run, &q.tag);
        let rows = downsample(rows, q.downsample.unwrap_or(5000));
        out.insert(run.to_string(), project_xaxis(&rows, x_axis));
    }
    Json(Value::Object(out))
}

async fn get_notes(Path(run): Path<String>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let row = conn
        .query_row(
            "SELECT note, updated_at FROM run_notes WHERE run = ?",
            params![run],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)),
        )
        .ok();
    match row {
        Some((note, updated)) => Json(json!({"note": note, "updated_at": updated})),
        None => Json(json!({"note": "", "updated_at": Value::Null})),
    }
}

#[derive(Deserialize)]
struct NoteBody {
    note: String,
}

async fn put_notes(Path(run): Path<String>, Json(b): Json<NoteBody>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    conn.execute(
        "INSERT INTO run_notes (run, note, updated_at) VALUES (?, ?, ?) \
         ON CONFLICT(run) DO UPDATE SET note = excluded.note, updated_at = excluded.updated_at",
        params![run, b.note, now_s()],
    )
    .ok();
    Json(json!({"ok": true}))
}

async fn delete_run(Path(run): Path<String>) -> Json<Value> {
    // Board-only delete: drop the metrics rows. We do NOT touch the training
    // workspace on disk (that's the run's real output, out of board scope).
    let conn = board().db.lock().unwrap();
    conn.execute("DELETE FROM scalars WHERE run = ?", params![run]).ok();
    conn.execute("DELETE FROM run_notes WHERE run = ?", params![run]).ok();
    let n = conn.execute("DELETE FROM runs WHERE name = ?", params![run]).unwrap_or(0);
    if n > 0 {
        Json(json!({"deleted": run}))
    } else {
        Json(json!({"error": "run not found"}))
    }
}

async fn empty_arr() -> Json<Value> {
    Json(json!([]))
}
async fn empty_obj() -> Json<Value> {
    Json(json!({}))
}
async fn null_json() -> Json<Value> {
    Json(Value::Null)
}

// ── artifacts (workspace sample images, served via the /files scope) ──────────

/// The run's workspace dir from the DB, falling back to the repository output
/// directory so the
/// artifacts view works for CLI runs the tailer has not registered yet.
fn run_workspace(conn: &Connection, run: &str) -> String {
    conn.query_row(
        "SELECT workspace_dir FROM runs WHERE name = ?",
        params![run],
        |r| r.get::<_, String>(0),
    )
    .ok()
    .unwrap_or_else(|| output_dir().join(run).to_string_lossy().into_owned())
}

struct ArtifactRow {
    tag: String,
    step: i64,
    path: String, // absolute
    width: i64,
    height: i64,
    mtime: f64,
}

/// Read a PNG's IHDR to get (width, height); returns (0,0) if not a PNG.
fn png_dims(path: &FsPath) -> (i64, i64) {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return (0, 0); };
    let mut buf = [0u8; 24];
    if f.read_exact(&mut buf).is_err() {
        return (0, 0);
    }
    if &buf[0..8] != b"\x89PNG\r\n\x1a\n" {
        return (0, 0);
    }
    let w = u32::from_be_bytes([buf[16], buf[17], buf[18], buf[19]]) as i64;
    let h = u32::from_be_bytes([buf[20], buf[21], buf[22], buf[23]]) as i64;
    (w, h)
}

fn trailing_digits(s: &str) -> Option<i64> {
    let d: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if d.is_empty() { None } else { d.parse().ok() }
}

fn mtime_secs(path: &FsPath) -> f64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Discover sample PNGs under samples/ and turbo_samples/. Two on-disk shapes:
///  - step-subdir: `turbo_samples/step_1500/step_0_0.png` → tag
///    "turbo_samples/step_0_0" (the image slot), step 1500 (from the subdir).
///    The filmstrip then slides one slot across checkpoints.
///  - flat file:   `samples/sample_500.png` → tag "samples", step from filename.
fn collect_artifacts(workspace: &str) -> Vec<ArtifactRow> {
    let mut out = Vec::new();
    for sub in ["samples", "turbo_samples"] {
        let root = FsPath::new(workspace).join(sub);
        let Ok(rd) = std::fs::read_dir(&root) else { continue; };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let step = p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(trailing_digits)
                    .unwrap_or(0);
                if let Ok(rd2) = std::fs::read_dir(&p) {
                    for e2 in rd2.flatten() {
                        let f = e2.path();
                        if f.extension().map(|x| x == "png").unwrap_or(false) {
                            let stem = f.file_stem().and_then(|n| n.to_str()).unwrap_or("img");
                            let (w, h) = png_dims(&f);
                            out.push(ArtifactRow {
                                tag: format!("{sub}/{stem}"),
                                step,
                                path: f.to_string_lossy().to_string(),
                                width: w,
                                height: h,
                                mtime: mtime_secs(&f),
                            });
                        }
                    }
                }
            } else if p.extension().map(|x| x == "png").unwrap_or(false) {
                let stem = p.file_stem().and_then(|n| n.to_str()).unwrap_or("img");
                let (w, h) = png_dims(&p);
                out.push(ArtifactRow {
                    tag: sub.to_string(),
                    step: trailing_digits(stem).unwrap_or(0),
                    path: p.to_string_lossy().to_string(),
                    width: w,
                    height: h,
                    mtime: mtime_secs(&p),
                });
            }
        }
    }
    out.sort_by(|a, b| a.tag.cmp(&b.tag).then(a.step.cmp(&b.step)));
    out
}

#[derive(Deserialize)]
struct ArtifactQ {
    tag: String,
}

/// Port of read_artifacts: metadata list for one tag. `blob_key` carries the
/// absolute path and `img_url` the /files URL the browser actually loads.
async fn artifacts(Path(run): Path<String>, Query(q): Query<ArtifactQ>) -> Json<Value> {
    let ws = {
        let conn = board().db.lock().unwrap();
        run_workspace(&conn, &run)
    };
    let out: Vec<Value> = collect_artifacts(&ws)
        .into_iter()
        .filter(|a| a.tag == q.tag)
        .map(|a| {
            json!({
                "step": a.step,
                "wall_time": a.mtime,
                "blob_key": a.path,
                "img_url": format!("/files{}", a.path),
                "mime_type": "image/png",
                "width": a.width,
                "height": a.height,
                "kind": "image",
                "meta": {},
            })
        })
        .collect();
    Json(Value::Array(out))
}

// ── hparams (real recipe stored at run launch) ────────────────────────────────

fn run_hparams(conn: &Connection, run: &str) -> Value {
    let hp: Option<String> = conn
        .query_row("SELECT hparams_json FROM runs WHERE name = ?", params![run], |r| r.get(0))
        .ok()
        .flatten();
    hp.and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}))
}

/// Final observed scalars, exposed as hparams "metrics" (the parcoords color dim).
fn run_metrics(conn: &Connection, run: &str) -> Value {
    let mut m = serde_json::Map::new();
    let last = |tag: &str| {
        conn.query_row(
            "SELECT value FROM scalars WHERE run = ? AND tag = ? ORDER BY step DESC LIMIT 1",
            params![run, tag],
            |r| r.get::<_, f64>(0),
        )
        .ok()
    };
    if let Some(v) = last(TAG_LOSS) {
        if v.is_finite() {
            m.insert("final_loss".into(), json!(v));
        }
    }
    if let Some(v) = last(TAG_GRAD) {
        if v.is_finite() {
            m.insert("final_grad_norm".into(), json!(v));
        }
    }
    Value::Object(m)
}

async fn hparams(Path(run): Path<String>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    Json(json!({
        "hparams": run_hparams(&conn, &run),
        "metrics": run_metrics(&conn, &run),
    }))
}

#[derive(Deserialize)]
struct RunsQ {
    runs: String,
}

async fn compare_hparams(Query(q): Query<RunsQ>) -> Json<Value> {
    let conn = board().db.lock().unwrap();
    let mut out: Vec<Value> = Vec::new();
    for run in q.runs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let exists = conn
            .query_row("SELECT 1 FROM runs WHERE name = ?", params![run], |_| Ok(()))
            .is_ok();
        if !exists {
            continue;
        }
        out.push(json!({
            "run": run,
            "hparams": run_hparams(&conn, run),
            "metrics": run_metrics(&conn, run),
        }));
    }
    Json(Value::Array(out))
}

// ── LoRA weight analytics (pure-Rust safetensors reader) ──────────────────────
//
// Port of serenityboard/lora_analytics.py. We read A/B weight pairs straight
// from the checkpoint header, convert bf16/f16→f32, and compute per-layer norms.
// L1/L2 (Frobenius) are exact; spectral norms are the top singular value via
// power iteration on the small (rank×rank) Gram matrix — no linear-algebra
// dependency. Metrics that need the full singular spectrum (effective_rank,
// condition_number) are intentionally omitted; the frontend renders them as "-".

/// finite→Number, non-finite→Null (JSON has no Inf/NaN; matches the frontend's
/// null→"-" rendering).
fn numj(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}

/// (file, data_section_start, tensor-header-map) for a safetensors file.
fn read_st_header(path: &str) -> Option<(std::fs::File, u64, serde_json::Map<String, Value>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb).ok()?;
    let hlen = u64::from_le_bytes(lenb);
    if hlen > 100 * 1024 * 1024 {
        return None; // sanity guard against a bogus header length
    }
    let mut hbuf = vec![0u8; hlen as usize];
    f.read_exact(&mut hbuf).ok()?;
    let v: Value = serde_json::from_slice(&hbuf).ok()?;
    Some((f, 8 + hlen, v.as_object()?.clone()))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

/// Load one tensor as f32 with its (rows, cols) shape (cols = product of dims>0).
fn load_tensor(
    f: &mut std::fs::File,
    data_start: u64,
    ent: &Value,
) -> Option<(Vec<f32>, usize, usize)> {
    use std::io::{Read, Seek, SeekFrom};
    let dtype = ent.get("dtype")?.as_str()?;
    let shape: Vec<usize> = ent
        .get("shape")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    let offs = ent.get("data_offsets")?.as_array()?;
    let a = offs.first()?.as_u64()?;
    let b = offs.get(1)?.as_u64()?;
    f.seek(SeekFrom::Start(data_start + a)).ok()?;
    let mut raw = vec![0u8; (b - a) as usize];
    f.read_exact(&mut raw).ok()?;
    let vals: Vec<f32> = match dtype {
        "F32" | "float32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "BF16" | "bfloat16" => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        "F16" | "float16" => raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => return None,
    };
    let (rows, cols) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product())
    } else if shape.len() == 1 {
        (shape[0], 1)
    } else {
        (vals.len(), 1)
    };
    Some((vals, rows, cols))
}

fn l1_l2(v: &[f32]) -> (f64, f64) {
    let mut l1 = 0f64;
    let mut sq = 0f64;
    for &x in v {
        let x = x as f64;
        l1 += x.abs();
        sq += x * x;
    }
    (l1, sq.sqrt())
}

/// Largest singular value of a row-major (rows×cols) matrix, via power iteration
/// on the smaller of the two Gram matrices (dimension = min(rows,cols) = rank).
fn spectral_norm(m: &[f32], rows: usize, cols: usize) -> f64 {
    if rows == 0 || cols == 0 || m.len() < rows * cols {
        return 0.0;
    }
    let k = rows.min(cols);
    let mut g = vec![0f64; k * k];
    if rows <= cols {
        // G = M Mᵀ  (k = rows)
        for i in 0..rows {
            let ri = &m[i * cols..(i + 1) * cols];
            for j in i..rows {
                let rj = &m[j * cols..(j + 1) * cols];
                let mut s = 0f64;
                for c in 0..cols {
                    s += ri[c] as f64 * rj[c] as f64;
                }
                g[i * k + j] = s;
                g[j * k + i] = s;
            }
        }
    } else {
        // G = Mᵀ M  (k = cols)
        for r in 0..rows {
            let row = &m[r * cols..(r + 1) * cols];
            for i in 0..cols {
                let a = row[i] as f64;
                for j in i..cols {
                    g[i * k + j] += a * row[j] as f64;
                }
            }
        }
        for i in 0..cols {
            for j in (i + 1)..cols {
                g[j * k + i] = g[i * k + j];
            }
        }
    }
    // power iteration for λ_max(G); σ_max = sqrt(λ_max)
    let mut v = vec![1f64 / (k as f64).sqrt(); k];
    let mut lambda = 0f64;
    for _ in 0..128 {
        let mut w = vec![0f64; k];
        for i in 0..k {
            let gi = &g[i * k..(i + 1) * k];
            let mut s = 0f64;
            for j in 0..k {
                s += gi[j] * v[j];
            }
            w[i] = s;
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-20 {
            lambda = 0.0;
            break;
        }
        for i in 0..k {
            v[i] = w[i] / norm;
        }
        lambda = norm; // ||G v|| with unit v ≈ λ_max as it converges
    }
    lambda.max(0.0).sqrt()
}

/// Per-layer metrics for one LoRA safetensors file (name → metric map).
fn analyze_file(path: &str) -> Option<serde_json::Map<String, Value>> {
    let (mut f, data_start, hdr) = read_st_header(path)?;
    // Suffixes are matched against the lowercased key, so ".lora_A.weight" and
    // ".lora_a.weight" collapse to one entry.
    let a_suf = [".lora_down.weight", ".lora_a.weight", ".lora.down.weight"];
    let b_suf = [".lora_up.weight", ".lora_b.weight", ".lora.up.weight"];
    let mut a_keys: HashMap<String, String> = HashMap::new();
    let mut b_keys: HashMap<String, String> = HashMap::new();
    for (k, ent) in hdr.iter() {
        if k == "__metadata__" {
            continue;
        }
        let ndim = ent.get("shape").and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(0);
        if ndim < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        let mut matched = false;
        for suf in a_suf {
            if lower.ends_with(suf) {
                a_keys.insert(k[..k.len() - suf.len()].to_string(), k.clone());
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        for suf in b_suf {
            if lower.ends_with(suf) {
                b_keys.insert(k[..k.len() - suf.len()].to_string(), k.clone());
                break;
            }
        }
    }
    let mut bases: Vec<&String> = a_keys.keys().filter(|b| b_keys.contains_key(*b)).collect();
    bases.sort();
    let mut layers = serde_json::Map::new();
    for base in bases {
        let (av, ar, ac) = load_tensor(&mut f, data_start, &hdr[&a_keys[base]])?;
        let (bv, br, bc) = load_tensor(&mut f, data_start, &hdr[&b_keys[base]])?;
        let (a_l1, a_l2) = l1_l2(&av);
        let (b_l1, b_l2) = l1_l2(&bv);
        let a_spec = spectral_norm(&av, ar, ac);
        let b_spec = spectral_norm(&bv, br, bc);
        let ab_ratio = if a_spec > 1e-12 { b_spec / a_spec } else { f64::INFINITY };
        layers.insert(
            base.clone(),
            json!({
                "a_l1_norm": numj(a_l1),
                "a_l2_norm": numj(a_l2),
                "a_spectral_norm": numj(a_spec),
                "b_l1_norm": numj(b_l1),
                "b_l2_norm": numj(b_l2),
                "b_spectral_norm": numj(b_spec),
                "ab_ratio": numj(ab_ratio),
            }),
        );
    }
    Some(layers)
}

/// analyze_file with a DB cache keyed by (path, mtime) so repeat views are instant.
fn analyze_cached(path: &str) -> Option<serde_json::Map<String, Value>> {
    let mtime = mtime_secs(FsPath::new(path));
    {
        let conn = board().db.lock().unwrap();
        let hit: Option<(f64, String)> = conn
            .query_row(
                "SELECT mtime, metrics_json FROM lora_cache WHERE path = ?",
                params![path],
                |r| Ok((r.get::<_, f64>(0)?, r.get::<_, String>(1)?)),
            )
            .ok();
        if let Some((mt, js)) = hit {
            if (mt - mtime).abs() < 1e-6 {
                if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&js) {
                    return Some(m);
                }
            }
        }
    }
    let layers = analyze_file(path)?;
    let js = Value::Object(layers.clone()).to_string();
    {
        let conn = board().db.lock().unwrap();
        conn.execute(
            "INSERT INTO lora_cache (path, mtime, metrics_json) VALUES (?, ?, ?) \
             ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime, metrics_json = excluded.metrics_json",
            params![path, mtime, js],
        )
        .ok();
    }
    Some(layers)
}

fn summary_stats(layers: &serde_json::Map<String, Value>) -> Value {
    let n = layers.len().max(1) as f64;
    let mut sum_ab = 0f64;
    let mut cnt_ab = 0f64;
    let mut sum_b = 0f64;
    let mut max_b = 0f64;
    for m in layers.values() {
        if let Some(r) = m.get("ab_ratio").and_then(|v| v.as_f64()) {
            sum_ab += r;
            cnt_ab += 1.0;
        }
        if let Some(b) = m.get("b_spectral_norm").and_then(|v| v.as_f64()) {
            sum_b += b;
            if b > max_b {
                max_b = b;
            }
        }
    }
    json!({
        "mean_ab_ratio": if cnt_ab > 0.0 { sum_ab / cnt_ab } else { 0.0 },
        "mean_b_spectral": sum_b / n,
        "max_b_spectral": max_b,
        "num_layers": layers.len(),
    })
}

/// The ab_ratio-dominance check from lora_analytics.diagnose (the eff-rank /
/// condition checks need the full spectrum, which we do not compute).
fn diagnose(layers: &serde_json::Map<String, Value>) -> Vec<String> {
    let n = layers.len();
    if n == 0 {
        return Vec::new();
    }
    let dominant = layers
        .values()
        .filter(|m| m.get("ab_ratio").and_then(|v| v.as_f64()).map(|r| r > 2.0).unwrap_or(false))
        .count();
    let mut out = Vec::new();
    if dominant as f64 > n as f64 * 0.5 {
        let pct = dominant as f64 / n as f64 * 100.0;
        out.push(format!(
            "B matrices dominating A across {pct:.0}% of layers (ab_ratio > 2.0) — possible overtraining or LR too high"
        ));
    }
    out
}

#[derive(Deserialize)]
struct LoraAnalyzeReq {
    path: String,
}

async fn lora_analyze(Json(req): Json<LoraAnalyzeReq>) -> (StatusCode, Json<Value>) {
    let path = req.path.clone();
    if !path.ends_with(".safetensors") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": "Only .safetensors files are supported"})));
    }
    if !FsPath::new(&path).is_file() {
        return (StatusCode::NOT_FOUND, Json(json!({"detail": format!("File not found: {path}")})));
    }
    let layers = match tokio::task::spawn_blocking(move || analyze_cached(&path)).await {
        Ok(Some(l)) => l,
        _ => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"detail": "failed to read LoRA weights"}))),
    };
    let n = layers.len();
    // NOTE: no "summary" key here — the frontend uses `!!data.summary` to switch
    // into two-file compare layout, so a single-file analyze must omit it.
    (
        StatusCode::OK,
        Json(json!({
            "layers": layers,
            "diagnostics": diagnose(&layers),
            "num_layers": n,
        })),
    )
}

#[derive(Deserialize)]
struct LoraCompareReq {
    path_a: String,
    path_b: String,
}

async fn lora_compare(Json(req): Json<LoraCompareReq>) -> (StatusCode, Json<Value>) {
    for p in [&req.path_a, &req.path_b] {
        if !p.ends_with(".safetensors") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": "Only .safetensors files are supported"})));
        }
        if !FsPath::new(p).is_file() {
            return (StatusCode::NOT_FOUND, Json(json!({"detail": format!("File not found: {p}")})));
        }
    }
    let (pa, pb) = (req.path_a.clone(), req.path_b.clone());
    let (ma, mb) = match tokio::task::spawn_blocking(move || (analyze_cached(&pa), analyze_cached(&pb))).await {
        Ok((Some(a), Some(b))) => (a, b),
        _ => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"detail": "failed to read LoRA weights"}))),
    };
    // union of layer names, sorted
    let mut names: Vec<String> = ma.keys().chain(mb.keys()).cloned().collect();
    names.sort();
    names.dedup();
    let diff_keys = [
        "a_l1_norm", "a_l2_norm", "a_spectral_norm",
        "b_l1_norm", "b_l2_norm", "b_spectral_norm", "ab_ratio",
    ];
    let mut layers = serde_json::Map::new();
    let (mut sum_da, mut sum_db, mut cnt) = (0f64, 0f64, 0f64);
    for name in &names {
        let m1 = ma.get(name);
        let m2 = mb.get(name);
        let mut e = serde_json::Map::new();
        e.insert("lora1".into(), m1.cloned().unwrap_or(Value::Null));
        e.insert("lora2".into(), m2.cloned().unwrap_or(Value::Null));
        if let (Some(m1), Some(m2)) = (m1, m2) {
            for key in diff_keys {
                let va = m1.get(key).and_then(|v| v.as_f64());
                let vb = m2.get(key).and_then(|v| v.as_f64());
                let pct = match (va, vb) {
                    (Some(a), Some(b)) if a.abs() > 1e-12 => (b - a) / a.abs() * 100.0,
                    _ => 0.0,
                };
                e.insert(format!("diff_{key}_pct"), numj(pct));
            }
            if let Some(v) = e.get("diff_a_spectral_norm_pct").and_then(|v| v.as_f64()) {
                sum_da += v;
                sum_db += e.get("diff_b_spectral_norm_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
                cnt += 1.0;
            }
        }
        layers.insert(name.clone(), Value::Object(e));
    }
    let n = layers.len();
    let mut diags = diagnose(&ma);
    diags.extend(diagnose(&mb));
    (
        StatusCode::OK,
        Json(json!({
            "layers": layers,
            "file_b": true,
            "summary": {
                "mean_diff_a_spectral_pct": if cnt > 0.0 { sum_da / cnt } else { 0.0 },
                "mean_diff_b_spectral_pct": if cnt > 0.0 { sum_db / cnt } else { 0.0 },
            },
            "summary_a": summary_stats(&ma),
            "summary_b": summary_stats(&mb),
            "diagnostics": diags,
            "num_layers": n,
        })),
    )
}

/// Upload variants need multipart (an extra dependency + feature); the path
/// endpoints cover server-side checkpoint reads. Return a clear hint instead of 404.
async fn lora_upload_unsupported() -> (StatusCode, Json<Value>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({"detail": "upload analysis is not supported here — use the path input (server reads the checkpoint directly)"})),
    )
}

// ── live WebSocket ───────────────────────────────────────────────────────────

async fn ws_live(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_ws)
}

async fn handle_ws(mut socket: WebSocket) {
    let mut rx = board().tx.subscribe();
    let mut sub_runs: Vec<String> = Vec::new();
    let mut sub_tags: Vec<String> = vec!["*".into()];
    loop {
        tokio::select! {
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Text(t))) => {
                        if let Ok(v) = serde_json::from_str::<Value>(&t) {
                            if let Some(sub) = v.get("subscribe") {
                                sub_runs = str_vec(sub.get("runs"));
                                let tags = str_vec(sub.get("tags"));
                                sub_tags = if tags.is_empty() { vec!["*".into()] } else { tags };
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
            ev = rx.recv() => {
                match ev {
                    Ok(ev) => {
                        let run_ok = sub_runs.iter().any(|r| r == &ev.run);
                        let tag_ok = sub_tags.iter().any(|p| glob_match(p, &ev.tag));
                        if run_ok && tag_ok {
                            if socket.send(Message::Text(ev.json)).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
}

fn str_vec(v: Option<&Value>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn glob_match(pat: &str, s: &str) -> bool {
    if pat == "*" {
        return true;
    }
    if !pat.contains('*') {
        return pat == s;
    }
    let parts: Vec<&str> = pat.split('*').collect();
    let mut pos = 0usize;
    let last = parts.len() - 1;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !s[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if i == last {
            if !s[pos..].ends_with(part) {
                return false;
            }
        } else {
            match s[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

// ── static frontend at /board ────────────────────────────────────────────────

async fn board_index() -> Response {
    serve_file(&board().board_dir.join("index.html"))
}

async fn board_asset(Path(file): Path<String>) -> Response {
    if file.contains("..") {
        return StatusCode::FORBIDDEN.into_response();
    }
    serve_file(&board().board_dir.join(&file))
}

fn serve_file(path: &FsPath) -> Response {
    match std::fs::read(path) {
        Ok(bytes) => {
            let ct = content_type(path);
            ([(header::CONTENT_TYPE, ct)], bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

fn content_type(path: &FsPath) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

// ── router (merged into the supervisor's app) ────────────────────────────────

pub fn router() -> Router<std::sync::Arc<crate::AppState>> {
    Router::new()
        .route("/board", get(board_index))
        .route("/board/", get(board_index))
        .route("/board/*file", get(board_asset))
        .route("/api/board/runs", get(list_runs))
        .route("/api/board/runs/:run", axum::routing::delete(delete_run))
        .route("/api/board/runs/:run/tags", get(tags))
        .route("/api/board/runs/:run/metrics", get(metrics))
        .route("/api/board/runs/:run/scalars", get(scalars))
        .route("/api/board/runs/:run/scalars/last", get(scalars_last))
        .route("/api/board/runs/:run/notes", get(get_notes).put(put_notes))
        .route("/api/board/runs/:run/traces", get(empty_arr))
        .route("/api/board/runs/:run/eval", get(empty_arr))
        .route("/api/board/runs/:run/artifacts", get(artifacts))
        .route("/api/board/runs/:run/hparams", get(hparams))
        .route("/api/board/runs/:run/pr-curves", get(empty_arr))
        .route("/api/board/runs/:run/audio", get(empty_arr))
        .route("/api/board/runs/:run/histograms", get(empty_arr))
        .route("/api/board/runs/:run/distributions", get(empty_arr))
        .route("/api/board/runs/:run/embeddings", get(empty_arr))
        .route("/api/board/runs/:run/custom-scalars/layout", get(null_json))
        .route("/api/board/runs/:run/custom-scalars/data", get(empty_obj))
        .route("/api/board/compare/scalars", get(compare_scalars))
        .route("/api/board/compare/hparams", get(compare_hparams))
        .route("/api/board/lora/analyze", post(lora_analyze))
        .route("/api/board/lora/compare", post(lora_compare))
        .route("/api/board/lora/analyze-upload", post(lora_upload_unsupported))
        .route("/api/board/lora/compare-upload", post(lora_upload_unsupported))
        .route("/api/board/ws/live", get(ws_live))
}
