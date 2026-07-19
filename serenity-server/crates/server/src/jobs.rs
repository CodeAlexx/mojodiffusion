//! GET /v1/jobs — faithful port of the daemon's job-history endpoint
//! (serenity_daemon.mojo @3295 + load_prior_rows @2181 + prior_job_json_value @1517).
//!
//! Reads the daemon's `jobs.db` (SQLite table `jobs(id,created,model,params_json,
//! state,output_path)`) — the on-disk schema is the contract — repairs any
//! non-terminal "running" row to "interrupted" (F10), and emits the same JSON the
//! daemon produces for each PRIOR row. Verified byte-identical vs `serenity_daemon
//! stub` reading the same db.
//!
//! SCOPE: this is the PRIOR-ROWS (history) path only. The daemon's /v1/jobs also
//! appends CURRENT-session JobRecords (`job_json_value`); the Rust server's job
//! registry evicts terminal jobs and does not yet track the full JobRecord
//! (no per-job `created`/`state`/`step`/`total`). Current-session records +
//! persistence (R-DB-1/2), and the queue mutations /v1/reorder + /v1/remove +
//! /v1/gallery/import (all current-job/counter coupled), are the write-side follow-up.

use std::path::Path;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Map, Value};
use serenity_wire::JobParams;

use crate::{result_manifest, AppState};

/// A current-session job in the shared JobBook (the daemon's `jobs` list): the
/// emitted `record`, the `params` the driver needs to send_start, and the queued
/// cancel flag. The driver promotes the FIRST active-queued entry (daemon model).
pub struct JobEntry {
    pub record: JobRecord,
    pub params: JobParams,
    pub cancel_requested: bool,
    /// Hires-fix control-plane state (NOT part of the frozen worker wire). When
    /// `hires_scale > 1.0`, the driver runs the job as two worker passes: a base
    /// pass, then an img2img refine pass at `scale*res` with `hires_denoise`
    /// creativity. Default `1.0 / 0.0` = single pass (no hires).
    pub hires_scale: f64,
    pub hires_denoise: f64,
}

impl JobEntry {
    /// `_is_active_queued`: a job eligible to run / be reordered / be removed.
    pub fn is_active_queued(&self) -> bool {
        self.record.state == "queued" && !self.cancel_requested
    }
}

// ── queue helpers (faithful ports; queue-position = index among active-queued) ──

pub fn find_job(jobs: &[JobEntry], id: &str) -> Option<usize> {
    jobs.iter().position(|e| e.record.id == id)
}

fn active_queued_count(jobs: &[JobEntry]) -> usize {
    jobs.iter().filter(|e| e.is_active_queued()).count()
}

fn queued_position_of_index(jobs: &[JobEntry], idx: usize) -> i64 {
    let mut pos = 0i64;
    for (i, e) in jobs.iter().enumerate() {
        if !e.is_active_queued() {
            continue;
        }
        if i == idx {
            return pos;
        }
        pos += 1;
    }
    -1
}

fn queued_index_at_position(jobs: &[JobEntry], target: i64) -> i64 {
    let mut pos = 0i64;
    for (i, e) in jobs.iter().enumerate() {
        if !e.is_active_queued() {
            continue;
        }
        if pos == target {
            return i as i64;
        }
        pos += 1;
    }
    -1
}

/// `_queued_jobs_json`: [{id, position}, …] over the active-queued jobs.
fn queued_jobs_json(jobs: &[JobEntry]) -> Value {
    let mut arr = Vec::new();
    let mut pos = 0i64;
    for e in jobs.iter() {
        if !e.is_active_queued() {
            continue;
        }
        arr.push(json!({ "id": e.record.id, "position": pos }));
        pos += 1;
    }
    Value::Array(arr)
}

/// `_move_queued_job_to_position`: adjacent swaps to slot src into target_pos.
fn move_queued_job_to_position(jobs: &mut [JobEntry], src_idx: usize, target_pos: i64) -> i64 {
    let mut pos = queued_position_of_index(jobs, src_idx);
    while pos > target_pos {
        let cur = queued_index_at_position(jobs, pos);
        let prev = queued_index_at_position(jobs, pos - 1);
        if cur < 0 || prev < 0 {
            return pos;
        }
        jobs.swap(cur as usize, prev as usize);
        pos -= 1;
    }
    while pos < target_pos {
        let cur = queued_index_at_position(jobs, pos);
        let nxt = queued_index_at_position(jobs, pos + 1);
        if cur < 0 || nxt < 0 {
            return pos;
        }
        jobs.swap(cur as usize, nxt as usize);
        pos += 1;
    }
    pos
}

/// `_reorder_target_position`: resolve `before_id` or `position` to a queue index.
fn reorder_target_position(jobs: &[JobEntry], src_idx: usize, body: &Value) -> Result<i64, String> {
    let count = active_queued_count(jobs) as i64;
    let src_pos = queued_position_of_index(jobs, src_idx);
    if count <= 0 || src_pos < 0 {
        return Err("job is not in the active queue".into());
    }
    if let Some(b) = body.get("before_id").filter(|v| !v.is_null()) {
        let before_id = b.as_str().ok_or("'before_id' must be a string")?;
        if before_id == jobs[src_idx].record.id {
            return Ok(src_pos);
        }
        let before_idx =
            find_job(jobs, before_id).ok_or_else(|| format!("no such before_id: {before_id}"))?;
        if !jobs[before_idx].is_active_queued() {
            return Err(format!(
                "'before_id' is not an active queued job: {before_id}"
            ));
        }
        let mut before_pos = queued_position_of_index(jobs, before_idx);
        if before_pos > src_pos {
            before_pos -= 1;
        }
        return Ok(before_pos);
    }
    match body.get("position").and_then(|v| v.as_i64()) {
        Some(t) if t >= 0 && t < count => Ok(t),
        Some(_) => Err(format!("'position' out of range [0..{}]", count - 1)),
        None => Err("'position' (integer) or 'before_id' (string) is required".into()),
    }
}

fn jobs_json_resp(status: StatusCode, doc: &Value) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}
fn jobs_err(status: StatusCode, detail: &str) -> Response {
    jobs_json_resp(status, &json!({ "detail": detail }))
}

fn body_object(body: &str) -> Value {
    serde_json::from_str::<Value>(body)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or_else(|| json!({}))
}

/// POST /v1/reorder — move an active-queued job to a new position (by `position`
/// or `before_id`). Returns {job_id, position, queue}.
pub async fn post_reorder(State(st): State<AppState>, body: String) -> Response {
    let b = body_object(&body);
    let id = match b.get("id").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            return jobs_err(
                StatusCode::UNPROCESSABLE_ENTITY,
                "'id' (string) is required",
            )
        }
    };
    let mut jobs = match st.jobs.lock() {
        Ok(j) => j,
        Err(_) => return jobs_err(StatusCode::INTERNAL_SERVER_ERROR, "job book poisoned"),
    };
    let i = match find_job(&jobs, &id) {
        Some(i) => i,
        None => return jobs_err(StatusCode::NOT_FOUND, &format!("no such job: {id}")),
    };
    if !jobs[i].is_active_queued() {
        return jobs_err(
            StatusCode::CONFLICT,
            &format!("only active queued jobs can be reordered: {id}"),
        );
    }
    let target = match reorder_target_position(&jobs, i, &b) {
        Ok(t) => t,
        Err(e) => return jobs_err(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };
    let new_pos = move_queued_job_to_position(&mut jobs, i, target);
    jobs_json_resp(
        StatusCode::OK,
        &json!({ "job_id": id, "position": new_pos, "queue": queued_jobs_json(&jobs) }),
    )
}

/// POST /v1/remove — remove an active-queued job before it starts.
/// Returns {job_id, removed, queue}.
pub async fn post_remove(State(st): State<AppState>, body: String) -> Response {
    let b = body_object(&body);
    let id = match b.get("id").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            return jobs_err(
                StatusCode::UNPROCESSABLE_ENTITY,
                "'id' (string) is required",
            )
        }
    };
    let mut jobs = match st.jobs.lock() {
        Ok(j) => j,
        Err(_) => return jobs_err(StatusCode::INTERNAL_SERVER_ERROR, "job book poisoned"),
    };
    let i = match find_job(&jobs, &id) {
        Some(i) => i,
        None => return jobs_err(StatusCode::NOT_FOUND, &format!("no such job: {id}")),
    };
    if !jobs[i].is_active_queued() {
        return jobs_err(
            StatusCode::CONFLICT,
            &format!("only active queued jobs can be removed before execution; use /v1/cancel/<id> for running jobs: {id}"),
        );
    }
    jobs.remove(i);
    jobs_json_resp(
        StatusCode::OK,
        &json!({ "job_id": id, "removed": true, "queue": queued_jobs_json(&jobs) }),
    )
}

fn terminal_state(s: &str) -> bool {
    s == "done" || s == "failed" || s == "cancelled" || s == "interrupted"
}

/// A current-session job record (the daemon's `JobRecord`, emitted-fields view).
#[derive(Clone)]
pub struct JobRecord {
    pub id: String,
    pub created: String,
    pub model: String,
    pub state: String, // queued | running | done | failed | cancelled
    pub progress: i64,
    pub step: i64,
    pub total: i64,
    pub image_index: i64,
    pub image_count: i64,
    pub output_path: String,
    pub error: String,
    /// genparams JSON for the jobs.db row (read from the output PNG at done-time);
    /// empty until terminal. NOT emitted by job_json_value (current-row shape).
    pub params_json: String,
}

const DB_PARAMS_MAX: usize = 2048;

/// `_db_safe_params`: cap params_json for a db row at a UTF-8 boundary (F1).
pub fn db_safe_params(s: &str) -> String {
    if s.len() <= DB_PARAMS_MAX {
        return s.to_string();
    }
    let mut cut = DB_PARAMS_MAX;
    let b = s.as_bytes();
    while cut > 0 && (b[cut] & 0xC0) == 0x80 {
        cut -= 1;
    }
    format!("{}...truncated", &s[..cut])
}

/// `derive_metadata_json`: a compact, queryable `serenity.gallery_meta.v1` summary
/// derived from a row's params_json (model/seed/dims/sampler/steps/cfg flattened +
/// the parsed params). Empty in → empty out. Stored in the additive `metadata_json`
/// db column so the gallery lightbox + external tooling can query it without
/// re-parsing the PNG. NOTE: derived, never authoritative — params_json is the
/// source of truth (kept for back-compat).
pub fn derive_metadata_json(
    id: &str,
    model: &str,
    state: &str,
    output_path: &str,
    params_json: &str,
) -> String {
    let params = if params_json.is_empty() {
        Value::Null
    } else {
        serde_json::from_str::<Value>(params_json)
            .ok()
            .filter(|v| v.is_object())
            .unwrap_or(Value::Null)
    };
    let pick = |k: &str| -> Value { params.get(k).cloned().unwrap_or(Value::Null) };
    let blob = json!({
        "schema": "serenity.gallery_meta.v1",
        "id": id,
        "model": model,
        "state": state,
        "output_path": output_path,
        "result_manifests": result_manifest::manifest_refs_for_output(output_path),
        "has_params": !params_json.is_empty(),
        "params": params,
        "seed": pick("seed"),
        "prompt": pick("prompt"),
        "negative": pick("negative"),
        "width": pick("width"),
        "height": pick("height"),
        "steps": pick("steps"),
        "cfg": pick("cfg"),
        "sampler": pick("sampler"),
        "scheduler": pick("scheduler"),
    });
    serde_json::to_string(&blob).unwrap_or_default()
}

/// `save_jobs_db`: rewrite jobs.db = prior rows + session jobs that have STARTED
/// (state != queued), in the daemon's `jobs` schema PLUS an additive `metadata_json`
/// column (a derived `serenity.gallery_meta.v1` blob; ignored by older readers that
/// SELECT only the 6 base columns). Crash-safe full rewrite.
pub fn save_jobs_db(prior: &[[String; 6]], jobs: &[JobEntry], db_path: &Path) {
    let _ = (|| -> rusqlite::Result<()> {
        let conn = rusqlite::Connection::open(db_path)?;
        conn.execute_batch(
            "DROP TABLE IF EXISTS jobs;\
             CREATE TABLE jobs (id TEXT, created TEXT, model TEXT, params_json TEXT, state TEXT, output_path TEXT, metadata_json TEXT);",
        )?;
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare("INSERT INTO jobs VALUES (?1,?2,?3,?4,?5,?6,?7)")?;
            for r in prior {
                // r is the 6-col in-memory contract; derive the 7th from its params.
                let meta = derive_metadata_json(&r[0], &r[2], &r[4], &r[5], &r[3]);
                stmt.execute(rusqlite::params![r[0], r[1], r[2], r[3], r[4], r[5], meta])?;
            }
            for e in jobs {
                let j = &e.record;
                if j.state == "queued" {
                    continue; // never started — nothing truthful to index yet
                }
                let pj = db_safe_params(&j.params_json);
                let meta = derive_metadata_json(&j.id, &j.model, &j.state, &j.output_path, &pj);
                stmt.execute(rusqlite::params![
                    j.id,
                    j.created,
                    j.model,
                    pj,
                    j.state,
                    j.output_path,
                    meta
                ])?;
            }
        }
        tx.commit()
    })();
}

/// `job_json_value`: a current-session JobRecord → JSON (the daemon's shape; note
/// current jobs carry NO params_json/params/history — those are prior-row only).
/// A `done` record additionally carries the derived `metadata` blob (its
/// params_json is populated at Done-time from the output PNG), so a live lightbox
/// gets the same queryable metadata as history without waiting for persistence.
pub fn job_json_value(r: &JobRecord) -> Value {
    let mut o = Map::new();
    o.insert("id".into(), json!(r.id));
    o.insert("created".into(), json!(r.created));
    o.insert("model".into(), json!(r.model));
    o.insert("state".into(), json!(r.state));
    o.insert("progress".into(), json!(r.progress));
    o.insert("step".into(), json!(r.step));
    o.insert("total".into(), json!(r.total));
    o.insert("image_index".into(), json!(r.image_index));
    o.insert("image_count".into(), json!(r.image_count));
    o.insert("output_path".into(), json!(r.output_path));
    o.insert("error".into(), json!(r.error));
    o.insert(
        "result_manifests".into(),
        result_manifest::manifest_refs_for_output(&r.output_path),
    );
    // Additive: a metadata blob once the record has params (done-time). Older
    // consumers ignore the extra key; the daemon's base shape is unchanged above.
    if !r.params_json.is_empty() {
        let meta = derive_metadata_json(&r.id, &r.model, &r.state, &r.output_path, &r.params_json);
        if let Ok(v) = serde_json::from_str::<Value>(&meta) {
            o.insert("metadata".into(), v);
        }
    }
    Value::Object(o)
}

fn attach_visual_health(mut v: Value, output_path: &str, params_json: &str) -> Value {
    if output_path.is_empty() {
        return v;
    }
    let visual_health = result_manifest::visual_health_for_params_json(output_path, params_json);
    if let Some(o) = v.as_object_mut() {
        o.insert("visual_health".into(), visual_health.clone());
        if let Some(meta) = o.get_mut("metadata").and_then(|m| m.as_object_mut()) {
            meta.insert("visual_health".into(), visual_health);
        }
    }
    v
}

fn attach_output_location(mut v: Value, output_path: &str, output_root: &Path) -> Value {
    if output_path.is_empty() {
        return v;
    }
    let output_location = result_manifest::output_location_for_root(output_path, Some(output_root));
    if let Some(o) = v.as_object_mut() {
        o.insert("output_location".into(), output_location.clone());
        if let Some(meta) = o.get_mut("metadata").and_then(|m| m.as_object_mut()) {
            meta.insert("output_location".into(), output_location);
        }
    }
    v
}

fn job_id_num(id: &str) -> i64 {
    id.strip_prefix("job-")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0)
}

/// `max_prior_id`: highest job number among prior rows (the new-id counter base, F7).
pub fn max_prior_id(prior: &[[String; 6]]) -> i64 {
    prior.iter().map(|r| job_id_num(&r[0])).max().unwrap_or(0)
}

/// `load_prior_rows`: jobs.db `jobs` rows in insertion order, non-terminal repaired
/// to "interrupted". Missing/unreadable db → empty (fresh start), like the daemon.
pub fn load_prior_rows(db_path: &Path) -> Vec<[String; 6]> {
    let mut out = Vec::new();
    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    ) {
        Ok(c) => c,
        Err(_) => return out,
    };
    let mut stmt = match conn
        .prepare("SELECT id, created, model, params_json, state, output_path FROM jobs")
    {
        Ok(s) => s,
        Err(_) => return out,
    };
    let rows = stmt.query_map([], |r| {
        Ok([
            r.get::<_, String>(0).unwrap_or_default(),
            r.get::<_, String>(1).unwrap_or_default(),
            r.get::<_, String>(2).unwrap_or_default(),
            r.get::<_, String>(3).unwrap_or_default(),
            r.get::<_, String>(4).unwrap_or_default(),
            r.get::<_, String>(5).unwrap_or_default(),
        ])
    });
    if let Ok(rows) = rows {
        for row in rows.flatten() {
            let mut cols = row;
            if !terminal_state(&cols[4]) {
                cols[4] = "interrupted".to_string();
            }
            out.push(cols);
        }
    }
    out
}

/// `prior_job_json_value`: a jobs.db row → the daemon's history-job JSON.
fn prior_job_json_value(row: &[String; 6]) -> Value {
    let mut o = Map::new();
    o.insert("id".into(), json!(row[0]));
    o.insert("created".into(), json!(row[1]));
    o.insert("model".into(), json!(row[2]));
    o.insert("state".into(), json!(row[4]));
    o.insert(
        "progress".into(),
        json!(if terminal_state(&row[4]) { 100 } else { 0 }),
    );
    o.insert("step".into(), json!(0));
    o.insert("total".into(), json!(0));
    o.insert("image_index".into(), json!(0));
    o.insert("image_count".into(), json!(1));
    o.insert("output_path".into(), json!(row[5]));
    o.insert("error".into(), json!(""));
    o.insert("params_json".into(), json!(row[3]));
    o.insert(
        "result_manifests".into(),
        result_manifest::manifest_refs_for_output(&row[5]),
    );
    if !row[3].is_empty() {
        if let Ok(params) = serde_json::from_str::<Value>(&row[3]) {
            if params.is_object() {
                // overwrite-in-place (keeps position) for the two defaults, append `steps`
                for k in ["image_index", "image_count", "steps"] {
                    if let Some(v) = params.get(k) {
                        o.insert(k.into(), v.clone());
                    }
                }
                o.insert("params".into(), params);
            }
        }
    }
    // Additive: the derived metadata blob for the lightbox (queryable, single obj).
    // Derived from the row's params_json (cols: id=0, model=2, state=4, output=5).
    let meta = derive_metadata_json(&row[0], &row[2], &row[4], &row[5], &row[3]);
    if let Ok(v) = serde_json::from_str::<Value>(&meta) {
        o.insert("metadata".into(), v);
    }
    o.insert("history".into(), json!(true));
    Value::Object(o)
}

/// GET /v1/jobs — prior (jobs.db history) rows THEN current-session JobRecords,
/// matching the daemon (prior loaded once at startup + the live `jobs` list).
pub async fn get_jobs(State(st): State<AppState>) -> Response {
    let out = st.out_dir.as_path();
    let mut arr: Vec<Value> = st
        .prior
        .iter()
        .map(|row| attach_output_location(prior_job_json_value(row), &row[5], out))
        .collect();
    if let Ok(jobs) = st.jobs.lock() {
        for e in jobs.iter() {
            arr.push(attach_output_location(
                job_json_value(&e.record),
                &e.record.output_path,
                out,
            ));
        }
    }
    (
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&Value::Array(arr)).unwrap_or_else(|_| String::from("[]")),
    )
        .into_response()
}

/// GET /v1/job/:id — current-session job FIRST (the daemon checks `jobs` then prior).
pub async fn get_job_one(
    State(st): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let ok = |v: Value| -> Response {
        (
            [(CONTENT_TYPE, "application/json")],
            serde_json::to_string(&v).unwrap_or_else(|_| String::from("{}")),
        )
            .into_response()
    };
    if let Ok(jobs) = st.jobs.lock() {
        if let Some(e) = jobs.iter().find(|e| e.record.id == id) {
            return ok(attach_output_location(
                attach_visual_health(
                    job_json_value(&e.record),
                    &e.record.output_path,
                    &e.record.params_json,
                ),
                &e.record.output_path,
                st.out_dir.as_path(),
            ));
        }
    }
    if let Some(row) = st.prior.iter().find(|r| r[0] == id) {
        return ok(attach_output_location(
            attach_visual_health(prior_job_json_value(row), &row[5], &row[3]),
            &row[5],
            st.out_dir.as_path(),
        ));
    }
    (
        StatusCode::NOT_FOUND,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&json!({ "detail": format!("no such job: {id}") })).unwrap(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_row_shape_and_repair() {
        // a terminal row, with parseable params carrying steps/image_count
        let row = [
            "job-0042".into(),
            "Fri, 12 Jun 2026 08:43:17 GMT".into(),
            "stub".into(),
            r#"{"steps":8,"image_count":2,"image_index":1}"#.into(),
            "done".into(),
            "out/job-0042.png".into(),
        ];
        let v = prior_job_json_value(&row);
        let s = serde_json::to_string(&v).unwrap();
        // key order + repaired/derived fields
        assert!(s.starts_with(r#"{"id":"job-0042","created":"Fri, 12 Jun 2026 08:43:17 GMT","model":"stub","state":"done","progress":100,"step":0,"total":0,"image_index":1,"image_count":2,"output_path":"out/job-0042.png","error":"","params_json":"#));
        assert_eq!(
            v["result_manifests"]["server_result_manifest"]["present"],
            false
        );
        assert!(s.contains(r#""steps":8,"params":{"#));
        assert!(s.ends_with(r#""history":true}"#));
    }

    #[test]
    fn non_terminal_repaired_to_interrupted() {
        let v = prior_job_json_value(&[
            "job-1".into(),
            "c".into(),
            "m".into(),
            "".into(),
            "interrupted".into(),
            "".into(),
        ]);
        assert_eq!(v.get("state").unwrap(), "interrupted");
        assert_eq!(v.get("progress").unwrap(), 100);
        assert_eq!(v.get("history").unwrap(), true);
        assert!(v.get("params").is_none());
    }

    #[test]
    fn terminal_state_set() {
        assert!(terminal_state("done"));
        assert!(terminal_state("interrupted"));
        assert!(!terminal_state("running"));
        assert!(!terminal_state("queued"));
    }

    fn entry(id: &str, state: &str) -> JobEntry {
        JobEntry {
            record: JobRecord {
                id: id.into(),
                created: "c".into(),
                model: "m".into(),
                state: state.into(),
                progress: 0,
                step: 0,
                total: 0,
                image_index: 0,
                image_count: 1,
                output_path: String::new(),
                error: String::new(),
                params_json: String::new(),
            },
            params: JobParams::default(),
            cancel_requested: false,
            hires_scale: 1.0,
            hires_denoise: 0.0,
        }
    }

    #[test]
    fn queue_math_and_reorder() {
        // a running job + 3 queued; queue positions count only the active-queued.
        let mut v = vec![
            entry("r", "running"),
            entry("a", "queued"),
            entry("b", "queued"),
            entry("c", "queued"),
        ];
        assert_eq!(active_queued_count(&v), 3);
        assert_eq!(
            serde_json::to_string(&queued_jobs_json(&v)).unwrap(),
            r#"[{"id":"a","position":0},{"id":"b","position":1},{"id":"c","position":2}]"#
        );
        // move c (queue-pos 2) to the front (pos 0): order becomes c,a,b
        let src = find_job(&v, "c").unwrap();
        let pos = move_queued_job_to_position(&mut v, src, 0);
        assert_eq!(pos, 0);
        assert_eq!(
            serde_json::to_string(&queued_jobs_json(&v)).unwrap(),
            r#"[{"id":"c","position":0},{"id":"a","position":1},{"id":"b","position":2}]"#
        );
        // reorder target by before_id: move b before c
        let src = find_job(&v, "b").unwrap();
        assert_eq!(
            reorder_target_position(&v, src, &json!({"before_id":"c"})).unwrap(),
            0
        );
        // by position, out of range
        assert!(reorder_target_position(&v, src, &json!({"position": 9})).is_err());
        assert!(reorder_target_position(&v, src, &json!({})).is_err());
    }

    #[test]
    fn derive_metadata_shape() {
        let m = derive_metadata_json(
            "job-0042",
            "zimage",
            "done",
            "out/job-0042.png",
            r#"{"seed":7,"width":768,"height":1024,"sampler":"euler"}"#,
        );
        let v: Value = serde_json::from_str(&m).unwrap();
        assert_eq!(v.get("schema").unwrap(), "serenity.gallery_meta.v1");
        assert_eq!(v.get("id").unwrap(), "job-0042");
        assert_eq!(v.get("model").unwrap(), "zimage");
        assert_eq!(v.get("has_params").unwrap(), true);
        assert_eq!(v.get("seed").unwrap(), 7);
        assert_eq!(v.get("width").unwrap(), 768);
        assert_eq!(v.get("sampler").unwrap(), "euler");
        assert_eq!(
            v["result_manifests"]["server_result_manifest"]["present"],
            false
        );
        assert_eq!(
            v["result_manifests"]["worker_result_manifest"]["present"],
            false
        );
        // params is the full parsed object
        assert_eq!(v.get("params").and_then(|p| p.get("height")).unwrap(), 1024);
        // absent fields are null, not missing
        assert!(v.get("cfg").unwrap().is_null());
    }

    #[test]
    fn derive_metadata_empty_params() {
        let m = derive_metadata_json("job-1", "m", "interrupted", "", "");
        let v: Value = serde_json::from_str(&m).unwrap();
        assert_eq!(v.get("has_params").unwrap(), false);
        assert!(v.get("params").unwrap().is_null());
        assert!(v.get("seed").unwrap().is_null());
        assert_eq!(
            v["result_manifests"]["server_result_manifest"]["present"],
            false
        );
    }

    #[test]
    fn jobs_db_additive_column_roundtrip() {
        // write the 7-col db, then read it back with the LEGACY 6-col SELECT
        // (load_prior_rows) to prove the extra column is backward-compatible.
        let mut db = std::env::temp_dir();
        db.push(format!("serenity_jobs_test_{}.db", std::process::id()));
        let _ = std::fs::remove_file(&db);
        let prior = vec![[
            "job-0001".to_string(),
            "Fri, 12 Jun 2026 08:43:17 GMT".to_string(),
            "zimage".to_string(),
            r#"{"seed":9,"width":512}"#.to_string(),
            "done".to_string(),
            "out/job-0001.png".to_string(),
        ]];
        save_jobs_db(&prior, &[], &db);
        // legacy 6-col read still works (ignores metadata_json)
        let rows = load_prior_rows(&db);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][0], "job-0001");
        assert_eq!(rows[0][4], "done");
        // the additive column is actually present + carries the derived blob
        let conn = rusqlite::Connection::open(&db).unwrap();
        let meta: String = conn
            .query_row(
                "SELECT metadata_json FROM jobs WHERE id='job-0001'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let v: Value = serde_json::from_str(&meta).unwrap();
        assert_eq!(v.get("schema").unwrap(), "serenity.gallery_meta.v1");
        assert_eq!(v.get("seed").unwrap(), 9);
        drop(conn);
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn job_json_value_metadata_when_done() {
        // a record with params_json (done-time) carries a metadata blob
        let mut r = entry("job-0005", "done").record;
        r.params_json = r#"{"seed":3,"model":"zimage"}"#.to_string();
        let v = job_json_value(&r);
        assert_eq!(v.get("state").unwrap(), "done");
        assert!(v.get("metadata").is_some());
        assert_eq!(v.get("metadata").unwrap().get("seed").unwrap(), 3);
        assert_eq!(
            v["result_manifests"]["server_result_manifest"]["present"],
            false
        );
        // a queued record (no params) has no metadata key (base daemon shape)
        let q = job_json_value(&entry("job-0006", "queued").record);
        assert!(q.get("metadata").is_none());
        assert_eq!(
            q["result_manifests"]["worker_result_manifest"]["present"],
            false
        );
    }

    #[test]
    fn single_job_visual_health_is_attached_without_polluting_lists() {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "serenity_jobs_visual_health_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("job-0090.png");
        let mut img = image::RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        ((x * 4) % 256) as u8,
                        ((y * 4) % 256) as u8,
                        (((x + y) * 3) % 256) as u8,
                    ]),
                );
            }
        }
        img.save(&png).unwrap();
        let mut r = entry("job-0090", "done").record;
        r.output_path = png.to_string_lossy().into_owned();
        r.params_json = r#"{"width":64,"height":64}"#.to_string();

        let listed = job_json_value(&r);
        assert!(listed.get("visual_health").is_none());

        let single = attach_visual_health(listed, &r.output_path, &r.params_json);
        assert_eq!(single["visual_health"]["status"], "pass");
        assert_eq!(single["metadata"]["visual_health"]["status"], "pass");
        let located = attach_output_location(single.clone(), &r.output_path, &dir);
        assert_eq!(
            located["output_location"]["root_kind"],
            "ui_workflow_gallery"
        );
        assert_eq!(located["output_location"]["inside_root"], true);
        assert_eq!(located["output_location"]["relative_path"], "job-0090.png");
        assert_eq!(located["metadata"]["output_location"]["inside_root"], true);

        let wrong_dims =
            attach_visual_health(single, &r.output_path, r#"{"width":512,"height":512}"#);
        assert_eq!(wrong_dims["visual_health"]["status"], "fail");
        assert!(wrong_dims["visual_health"]["failures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .unwrap_or_default()
                .starts_with("wrong_dimensions")));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
