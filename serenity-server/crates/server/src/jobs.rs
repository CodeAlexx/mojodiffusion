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

use crate::AppState;

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

/// `save_jobs_db`: rewrite jobs.db = prior rows + session jobs that have STARTED
/// (state != queued), in the daemon's `jobs` schema. Crash-safe full rewrite.
pub fn save_jobs_db(prior: &[[String; 6]], jobs: &[JobRecord], db_path: &Path) {
    let _ = (|| -> rusqlite::Result<()> {
        let conn = rusqlite::Connection::open(db_path)?;
        conn.execute_batch(
            "DROP TABLE IF EXISTS jobs;\
             CREATE TABLE jobs (id TEXT, created TEXT, model TEXT, params_json TEXT, state TEXT, output_path TEXT);",
        )?;
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare("INSERT INTO jobs VALUES (?1,?2,?3,?4,?5,?6)")?;
            for r in prior {
                stmt.execute(rusqlite::params![r[0], r[1], r[2], r[3], r[4], r[5]])?;
            }
            for j in jobs {
                if j.state == "queued" {
                    continue; // never started — nothing truthful to index yet
                }
                stmt.execute(rusqlite::params![
                    j.id, j.created, j.model, db_safe_params(&j.params_json), j.state, j.output_path
                ])?;
            }
        }
        tx.commit()
    })();
}

/// `job_json_value`: a current-session JobRecord → JSON (the daemon's shape; note
/// current jobs carry NO params_json/params/history — those are prior-row only).
pub fn job_json_value(r: &JobRecord) -> Value {
    json!({
        "id": r.id, "created": r.created, "model": r.model, "state": r.state,
        "progress": r.progress, "step": r.step, "total": r.total,
        "image_index": r.image_index, "image_count": r.image_count,
        "output_path": r.output_path, "error": r.error,
    })
}

fn job_id_num(id: &str) -> i64 {
    id.strip_prefix("job-").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0)
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
    let mut stmt = match conn.prepare("SELECT id, created, model, params_json, state, output_path FROM jobs") {
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
    o.insert("progress".into(), json!(if terminal_state(&row[4]) { 100 } else { 0 }));
    o.insert("step".into(), json!(0));
    o.insert("total".into(), json!(0));
    o.insert("image_index".into(), json!(0));
    o.insert("image_count".into(), json!(1));
    o.insert("output_path".into(), json!(row[5]));
    o.insert("error".into(), json!(""));
    o.insert("params_json".into(), json!(row[3]));
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
    o.insert("history".into(), json!(true));
    Value::Object(o)
}

/// GET /v1/jobs — prior (jobs.db history) rows THEN current-session JobRecords,
/// matching the daemon (prior loaded once at startup + the live `jobs` list).
pub async fn get_jobs(State(st): State<AppState>) -> Response {
    let mut arr: Vec<Value> = st.prior.iter().map(prior_job_json_value).collect();
    if let Ok(jobs) = st.jobs.lock() {
        for r in jobs.iter() {
            arr.push(job_json_value(r));
        }
    }
    (
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&Value::Array(arr)).unwrap_or_else(|_| String::from("[]")),
    )
        .into_response()
}

/// GET /v1/job/:id — current-session job FIRST (the daemon checks `jobs` then prior).
pub async fn get_job_one(State(st): State<AppState>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    let ok = |v: Value| -> Response {
        (
            [(CONTENT_TYPE, "application/json")],
            serde_json::to_string(&v).unwrap_or_else(|_| String::from("{}")),
        )
            .into_response()
    };
    if let Ok(jobs) = st.jobs.lock() {
        if let Some(r) = jobs.iter().find(|r| r.id == id) {
            return ok(job_json_value(r));
        }
    }
    if let Some(row) = st.prior.iter().find(|r| r[0] == id) {
        return ok(prior_job_json_value(row));
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
        assert!(s.contains(r#""steps":8,"params":{"#));
        assert!(s.ends_with(r#""history":true}"#));
    }

    #[test]
    fn non_terminal_repaired_to_interrupted() {
        let v = prior_job_json_value(&[
            "job-1".into(), "c".into(), "m".into(), "".into(), "interrupted".into(), "".into(),
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
}
