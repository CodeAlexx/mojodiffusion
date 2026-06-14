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

/// `load_prior_rows`: jobs.db `jobs` rows in insertion order, non-terminal repaired
/// to "interrupted". Missing/unreadable db → empty (fresh start), like the daemon.
fn load_prior_rows(db_path: &Path) -> Vec<[String; 6]> {
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

/// GET /v1/jobs — prior (jobs.db history) rows. Current-session records: see SCOPE.
pub async fn get_jobs(State(st): State<AppState>) -> Response {
    let db = st.out_dir.join("jobs.db");
    let arr: Vec<Value> = load_prior_rows(&db).iter().map(prior_job_json_value).collect();
    (
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(&Value::Array(arr)).unwrap_or_else(|_| String::from("[]")),
    )
        .into_response()
}

/// GET /v1/job/:id — one job by id (prior rows; current-session: see SCOPE).
pub async fn get_job_one(State(st): State<AppState>, axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    let db = st.out_dir.join("jobs.db");
    for row in load_prior_rows(&db) {
        if row[0] == id {
            return (
                [(CONTENT_TYPE, "application/json")],
                serde_json::to_string(&prior_job_json_value(&row)).unwrap_or_else(|_| String::from("{}")),
            )
                .into_response();
        }
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
