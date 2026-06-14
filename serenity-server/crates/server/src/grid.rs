//! POST /v1/grid — SwarmUI-style XYZ-plot / parameter-sweep grid generator.
//!
//! Enqueues one generate job per axis value (reusing the EXACT `/v1/generate`
//! enqueue mechanics — build a `JobParams`, register a `JobChannel`, push a
//! `jobs::JobEntry`, wake the driver), waits for every cell job to reach a terminal
//! state, then composites the produced PNGs into a single labelled grid image.
//!
//! CONTRACT
//!   request  (flat JSON, like /v1/generate, plus two grid keys):
//!     { "axis":  "seed"|"cfg"|"steps"|"sampler"|"scheduler",
//!       "values": [ ... ],         // numbers for seed/cfg/steps, strings for sampler/scheduler
//!       "model": "...", "prompt": "...",   // + the usual flat generate fields
//!       "negative"?, "width"?, "height"?, "steps"?, "seed"?,
//!       "sampler"?, "scheduler"?, "cfg"?, "out_dir"? }
//!   response (200):
//!     { "grid_id": "grid-NNNN",
//!       "path": "<abs path to the composite png>",
//!       "axis": "cfg",
//!       "cells": [ { "value": "3.0", "job_id": "job-0007", "output_path": "..." }, ... ] }
//!
//! WHAT IS REPLICATED FROM post_generate (rather than calling it): the JobParams
//! build (default + flat overrides + job_id + out_dir) and the four-step enqueue
//! (register JobChannel → push JobEntry → DriverCtl::Wake). post_generate's body is
//! a single private async fn that also lowers `workflow` graphs and embeds the PNG
//! genparams tEXt; we replicate only the plain-params subset we need. The image
//! compositing reuses the `image` crate already pulled in for gallery thumbnails.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use image::{ImageFormat, Rgb, RgbImage};
use serde_json::{json, Value};

use crate::jobs;
use crate::{AppState, DriverCtl, JobChannel};
use serenity_wire::JobParams;

// ── tunables ────────────────────────────────────────────────────────────────────

/// Max number of cells (axis values) accepted in one grid request.
const MAX_VALUES: usize = 16;
/// Fixed cell image size (square), px.
const CELL: u32 = 320;
/// Height of the per-cell label band drawn beneath each cell, px.
const LABEL_BAND: u32 = 18;
/// Poll cadence while waiting for the enqueued cell jobs to finish.
const POLL: Duration = Duration::from_millis(250);
/// Hard ceiling on the whole grid wait (~10 min); past this we 504 with partial info.
/// Per-cell wait budget. Cells run SERIALLY on the single GPU, so the total
/// budget scales with cell count (a flat cap falsely 504s a legitimately slow
/// large grid). Capped at GRID_TIMEOUT_MAX overall.
const GRID_PER_CELL_TIMEOUT: Duration = Duration::from_secs(180);
const GRID_TIMEOUT_MAX: Duration = Duration::from_secs(3600);

// ── small JSON response helpers (mirror gallery.rs json_compact/err_detail) ──────

fn json_compact(status: StatusCode, doc: &Value) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}

fn err_detail(status: StatusCode, detail: &str) -> Response {
    json_compact(status, &json!({ "detail": detail }))
}

// ── minimal 5x7 bitmap font ──────────────────────────────────────────────────────
//
// A self-contained 5-wide × 7-tall glyph table for the printable ASCII subset the
// grid labels actually use: 0-9, A-Z, a-z, space, '=', '.', '-', '+', ':', '_'. Each
// glyph is 7 rows; each row is the low 5 bits of a u8 (bit4 = leftmost column). An
// unknown char renders as a hollow box so a label is never silently dropped.

/// One glyph = 7 rows of 5-bit columns (MSB-of-5 = leftmost pixel).
type Glyph = [u8; 7];

const GLYPH_W: u32 = 5;
const GLYPH_H: u32 = 7;
const GLYPH_GAP: u32 = 1; // 1px between glyphs

const UNKNOWN: Glyph = [
    0b11111,
    0b10001,
    0b10001,
    0b10001,
    0b10001,
    0b10001,
    0b11111,
];

/// Map a printable ASCII char to its 5x7 glyph (UNKNOWN box otherwise).
fn glyph_for(c: char) -> Glyph {
    match c {
        ' ' => [0; 7],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        'b' => [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b11110],
        'c' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10000, 0b10001, 0b01110],
        'd' => [0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10001, 0b01111],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        'f' => [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
        'g' => [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'h' => [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'j' => [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
        'k' => [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
        'l' => [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'm' => [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101],
        'n' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'o' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'p' => [0b00000, 0b10110, 0b11001, 0b10001, 0b11110, 0b10000, 0b10000],
        'q' => [0b00000, 0b01101, 0b10011, 0b10001, 0b01111, 0b00001, 0b00001],
        'r' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        's' => [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
        't' => [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
        'u' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        'v' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'w' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
        'x' => [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
        'y' => [0b00000, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'z' => [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        _ => UNKNOWN,
    }
}

/// Blit `s` into `img` with top-left at (x, y) in `color`. Glyphs are 5x7 with a 1px
/// gap; characters that run past the image right/bottom edge are clipped (not wrapped).
fn draw_text(img: &mut RgbImage, x: u32, y: u32, s: &str, color: Rgb<u8>) {
    let (iw, ih) = (img.width(), img.height());
    let mut cx = x;
    for ch in s.chars() {
        let g = glyph_for(ch);
        for (row, bits) in g.iter().enumerate() {
            let py = y + row as u32;
            if py >= ih {
                continue;
            }
            for col in 0..GLYPH_W {
                // bit (GLYPH_W-1-col) is the leftmost column.
                if (bits >> (GLYPH_W - 1 - col)) & 1 == 1 {
                    let px = cx + col;
                    if px < iw {
                        img.put_pixel(px, py, color);
                    }
                }
            }
        }
        cx += GLYPH_W + GLYPH_GAP;
        if cx >= iw {
            break;
        }
    }
}

/// Pixel width a label string would occupy at the 5x7 font (for centering).
fn text_px_width(s: &str) -> u32 {
    let n = s.chars().count() as u32;
    if n == 0 {
        0
    } else {
        n * GLYPH_W + (n - 1) * GLYPH_GAP
    }
}

// ── request parsing ───────────────────────────────────────────────────────────────

/// The five swept axes. Numeric axes consume JSON numbers; string axes consume JSON
/// strings.
#[derive(Clone, Copy, PartialEq)]
enum Axis {
    Seed,
    Cfg,
    Steps,
    Sampler,
    Scheduler,
}

impl Axis {
    fn parse(s: &str) -> Option<Axis> {
        match s {
            "seed" => Some(Axis::Seed),
            "cfg" => Some(Axis::Cfg),
            "steps" => Some(Axis::Steps),
            "sampler" => Some(Axis::Sampler),
            "scheduler" => Some(Axis::Scheduler),
            _ => None,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Axis::Seed => "seed",
            Axis::Cfg => "cfg",
            Axis::Steps => "steps",
            Axis::Sampler => "sampler",
            Axis::Scheduler => "scheduler",
        }
    }
    fn is_numeric(self) -> bool {
        matches!(self, Axis::Seed | Axis::Cfg | Axis::Steps)
    }
}

/// Read an optional i64 from a flat request field (accepts a JSON number).
fn opt_i64(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(|x| x.as_i64())
}
fn opt_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}
fn opt_str(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(|s| s.to_string())
}

/// Build the base `JobParams` from the flat request fields EXACTLY like
/// post_generate's plain-params path (default() then override model/prompt/negative/
/// width/height/steps/seed/sampler/scheduler/cfg; images forced to 1 per cell; out_dir
/// = request out_dir or the server out_dir). The per-job `job_id`/`params_json` are
/// set later, once per cell, by the enqueue.
fn base_params(st: &AppState, req: &Value) -> Result<JobParams, String> {
    let model = opt_str(req, "model").unwrap_or_default();
    if model.is_empty() {
        return Err("'model' (string) is required".into());
    }
    let prompt = opt_str(req, "prompt").unwrap_or_default();
    if prompt.is_empty() {
        return Err("'prompt' (string) is required".into());
    }
    let mut p = JobParams::default();
    p.model = model;
    p.prompt = prompt;
    if let Some(v) = opt_str(req, "negative") {
        p.negative = v;
    }
    if let Some(v) = opt_i64(req, "width") {
        p.width = v;
    }
    if let Some(v) = opt_i64(req, "height") {
        p.height = v;
    }
    if let Some(v) = opt_i64(req, "steps") {
        p.steps = v;
    }
    if let Some(v) = opt_i64(req, "seed") {
        p.seed = v;
    }
    if let Some(v) = opt_str(req, "sampler") {
        p.sampler = v;
    }
    if let Some(v) = opt_str(req, "scheduler") {
        p.scheduler = v;
    }
    if let Some(v) = opt_f64(req, "cfg") {
        p.cfg = v;
    }
    // One image per cell — the grid IS the multi-image layout.
    p.images = 1;
    // out_dir: explicit request override else the server-configured out_dir.
    p.out_dir = opt_str(req, "out_dir").unwrap_or_else(|| st.out_dir.to_string_lossy().into_owned());
    Ok(p)
}

/// Apply one axis value to a (cloned base) JobParams, returning the canonical label
/// string for that value (e.g. "3.0", "euler"). Numeric axes read `value.as_*`; string
/// axes read `value.as_str`.
fn apply_axis(p: &mut JobParams, axis: Axis, value: &Value) -> Result<String, String> {
    match axis {
        Axis::Seed => {
            let n = value.as_i64().ok_or("seed values must be integers")?;
            p.seed = n;
            Ok(n.to_string())
        }
        Axis::Cfg => {
            let n = value.as_f64().ok_or("cfg values must be numbers")?;
            p.cfg = n;
            // Stable, readable label: trim to a short decimal.
            Ok(fmt_num(n))
        }
        Axis::Steps => {
            let n = value.as_i64().ok_or("steps values must be integers")?;
            p.steps = n;
            Ok(n.to_string())
        }
        Axis::Sampler => {
            let s = value.as_str().ok_or("sampler values must be strings")?;
            p.sampler = s.to_string();
            Ok(s.to_string())
        }
        Axis::Scheduler => {
            let s = value.as_str().ok_or("scheduler values must be strings")?;
            p.scheduler = s.to_string();
            Ok(s.to_string())
        }
    }
}

/// Format an f64 for a cfg label: integral values as "N.0", else a trimmed decimal.
fn fmt_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{:.1}", n)
    } else {
        // up to 3 decimals, trailing zeros trimmed
        let s = format!("{:.3}", n);
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    }
}

// ── enqueue (replicates post_generate's register→push→wake) ─────────────────────-

/// One enqueued cell: its label and the assigned job id.
struct Cell {
    value: String,
    job_id: String,
}

/// Enqueue one cell job. Mirrors post_generate: take a new job-XXXX id off next_id,
/// stamp it on the params, register a JobChannel BEFORE the entry is promotable, push
/// the queued JobEntry, and wake the driver. Returns the job_id (caller already holds
/// the label). On a poisoned lock / dead driver returns Err so the grid can fail loud.
fn enqueue_cell(st: &AppState, mut params: JobParams) -> Result<String, String> {
    let n = st.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let job_id = format!("job-{n:04}");
    params.job_id = job_id.clone();

    // Minimal genparams tEXt so the produced PNG round-trips in /v1/gallery just like a
    // /v1/generate image (schema + job_id + the core flat fields).
    let gp = json!({
        "schema": "serenity.genparams.v1",
        "job_id": job_id,
        "model": params.model,
        "prompt": params.prompt,
        "negative": params.negative,
        "width": params.width,
        "height": params.height,
        "steps": params.steps,
        "seed": params.seed,
        "sampler": params.sampler,
        "scheduler": params.scheduler,
        "cfg": params.cfg,
    });
    params.params_json = serde_json::to_string(&gp).unwrap_or_default();

    // 1. Register the per-job channel BEFORE the entry becomes promotable (so the
    //    driver — which looks the channel up by id — finds it; same ordering as
    //    post_generate).
    let channel = JobChannel::new();
    {
        let mut map = st.registry.lock().map_err(|_| "registry poisoned".to_string())?;
        map.insert(job_id.clone(), channel);
    }

    // 2. Push the queued JobEntry onto the shared JobBook.
    let record = jobs::JobRecord {
        id: job_id.clone(),
        created: httpdate::fmt_http_date(std::time::SystemTime::now()),
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
    {
        match st.jobs.lock() {
            Ok(mut book) => book.push(jobs::JobEntry {
                record,
                params,
                cancel_requested: false,
                hires_scale: 1.0,
                hires_denoise: 0.0,
            }),
            Err(_) => {
                if let Ok(mut map) = st.registry.lock() {
                    map.remove(&job_id);
                }
                return Err("job book poisoned".into());
            }
        }
    }

    // 3. Wake the driver to promote the new job.
    if st.ctl.send(DriverCtl::Wake).is_err() {
        if let Ok(mut map) = st.registry.lock() {
            map.remove(&job_id);
        }
        if let Ok(mut book) = st.jobs.lock() {
            book.retain(|e| e.record.id != job_id);
        }
        return Err("worker driver unavailable".into());
    }

    Ok(job_id)
}

// ── terminal-state polling (NEVER holds st.jobs across a sleep) ──────────────────-

fn is_terminal(state: &str) -> bool {
    state == "done" || state == "failed" || state == "cancelled" || state == "interrupted"
}

/// Snapshot the (state, output_path) for each target id under one short lock, then
/// RELEASE the lock and return. The caller sleeps OUTSIDE the lock. Returns, per id:
/// (state, output_path). Missing ids (never happens — we just pushed them) report
/// ("missing", "").
fn snapshot_states(st: &AppState, ids: &[String]) -> Vec<(String, String)> {
    let book = match st.jobs.lock() {
        Ok(b) => b,
        Err(_) => return ids.iter().map(|_| ("failed".to_string(), String::new())).collect(),
    };
    ids.iter()
        .map(|id| {
            match book.iter().find(|e| e.record.id == *id) {
                Some(e) => (e.record.state.clone(), e.record.output_path.clone()),
                None => ("missing".to_string(), String::new()),
            }
        })
        .collect()
}

// ── compositing ────────────────────────────────────────────────────────────────--

const BG: Rgb<u8> = Rgb([32, 32, 36]); // dark gray background
const CELL_BG: Rgb<u8> = Rgb([20, 20, 24]); // slightly darker per-cell fill
const PLACEHOLDER_BG: Rgb<u8> = Rgb([60, 24, 24]); // dark red for a missing/failed cell
const BAND_BG: Rgb<u8> = Rgb([16, 16, 18]);
const TEXT: Rgb<u8> = Rgb([230, 230, 235]);
const PAD: u32 = 6; // padding around / between cells

/// One composited cell: its label and the (optional) source PNG path. A None path (or
/// an unreadable PNG) renders a placeholder tile rather than aborting the whole grid.
struct CompCell {
    label: String,
    src: Option<String>,
}

/// Composite the cells into a grid PNG and save it to `<out_dir>/grid-<id>.png`.
/// cols = ceil(sqrt(n)), rows = ceil(n/cols). Each cell is CELL px square with a
/// LABEL_BAND-px label strip beneath. Returns the absolute output path.
fn render_grid(out_dir: &std::path::Path, grid_id: &str, cells: &[CompCell]) -> Result<String, String> {
    let n = cells.len().max(1);
    let cols = (n as f64).sqrt().ceil() as u32;
    let cols = cols.max(1);
    let rows = ((n as u32) + cols - 1) / cols;

    let cell_total_h = CELL + LABEL_BAND;
    let img_w = PAD + cols * (CELL + PAD);
    let img_h = PAD + rows * (cell_total_h + PAD);

    let mut canvas = RgbImage::from_pixel(img_w, img_h, BG);

    for (i, cell) in cells.iter().enumerate() {
        let r = (i as u32) / cols;
        let c = (i as u32) % cols;
        let x0 = PAD + c * (CELL + PAD);
        let y0 = PAD + r * (cell_total_h + PAD);

        // Decode the source PNG (if any) and resize to a CELL square; on any failure
        // draw a placeholder fill so one bad cell never aborts the grid.
        let mut drew_image = false;
        if let Some(src) = &cell.src {
            match image::open(src) {
                Ok(img) => {
                    let resized = img.resize_exact(CELL, CELL, image::imageops::FilterType::Lanczos3);
                    let rgb = resized.to_rgb8();
                    for yy in 0..CELL.min(rgb.height()) {
                        for xx in 0..CELL.min(rgb.width()) {
                            canvas.put_pixel(x0 + xx, y0 + yy, *rgb.get_pixel(xx, yy));
                        }
                    }
                    drew_image = true;
                }
                Err(_) => {}
            }
        }
        if !drew_image {
            let fill = if cell.src.is_some() { PLACEHOLDER_BG } else { CELL_BG };
            fill_rect(&mut canvas, x0, y0, CELL, CELL, fill);
            // a small "no image" marker centered in the cell
            let marker = "no image";
            let tw = text_px_width(marker);
            let mx = x0 + (CELL.saturating_sub(tw)) / 2;
            let my = y0 + CELL / 2 - GLYPH_H / 2;
            draw_text(&mut canvas, mx, my, marker, TEXT);
        }

        // Label band beneath the cell.
        let by = y0 + CELL;
        fill_rect(&mut canvas, x0, by, CELL, LABEL_BAND, BAND_BG);
        let tw = text_px_width(&cell.label);
        let lx = x0 + (CELL.saturating_sub(tw)) / 2;
        let ly = by + (LABEL_BAND.saturating_sub(GLYPH_H)) / 2;
        draw_text(&mut canvas, lx, ly, &cell.label, TEXT);
    }

    let path = out_dir.join(format!("{grid_id}.png"));
    canvas
        .save_with_format(&path, ImageFormat::Png)
        .map_err(|e| format!("grid composite save failed: {e}"))?;
    Ok(path.to_string_lossy().into_owned())
}

/// Fill an axis-aligned rectangle (clipped to the canvas).
fn fill_rect(img: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, color: Rgb<u8>) {
    let (iw, ih) = (img.width(), img.height());
    for yy in y..(y + h).min(ih) {
        for xx in x..(x + w).min(iw) {
            img.put_pixel(xx, yy, color);
        }
    }
}

// ── handler ────────────────────────────────────────────────────────────────────--

/// POST /v1/grid — sweep one axis, enqueue a cell job per value, wait for all to
/// finish, composite into a labelled grid PNG. See the module contract.
pub async fn post_grid(State(st): State<AppState>, Json(req): Json<Value>) -> Response {
    // 1. axis + values validation.
    let axis_str = match opt_str(&req, "axis") {
        Some(s) if !s.is_empty() => s,
        _ => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'axis' (string) is required"),
    };
    let axis = match Axis::parse(&axis_str) {
        Some(a) => a,
        None => {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                "'axis' must be one of: seed, cfg, steps, sampler, scheduler",
            )
        }
    };
    let values = match req.get("values").and_then(|v| v.as_array()) {
        Some(a) => a.clone(),
        None => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'values' (array) is required"),
    };
    if values.is_empty() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'values' must not be empty");
    }
    if values.len() > MAX_VALUES {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("'values' has too many entries ({}, max {MAX_VALUES})", values.len()),
        );
    }
    // Type-check every value up-front against the axis (numeric vs string) so a bad
    // entry fails the whole request before any job is enqueued.
    for (i, v) in values.iter().enumerate() {
        let ok = if axis.is_numeric() { v.is_number() } else { v.is_string() };
        if !ok {
            let want = if axis.is_numeric() { "number" } else { "string" };
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!("'values[{i}]' must be a {want} for axis '{}'", axis.as_str()),
            );
        }
    }

    // 2. base params from the remaining flat fields (like post_generate).
    let base = match base_params(&st, &req) {
        Ok(p) => p,
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };

    // 3. enqueue one cell per value.
    let mut cells: Vec<Cell> = Vec::with_capacity(values.len());
    for v in &values {
        let mut p = base.clone();
        let label = match apply_axis(&mut p, axis, v) {
            Ok(l) => l,
            Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
        };
        let job_id = match enqueue_cell(&st, p) {
            Ok(id) => id,
            // A driver/lock failure mid-enqueue: surface it (partial jobs already queued
            // will still run; we report the failure rather than hang).
            Err(e) => return err_detail(StatusCode::SERVICE_UNAVAILABLE, &e),
        };
        cells.push(Cell { value: label, job_id });
    }

    let ids: Vec<String> = cells.iter().map(|c| c.job_id.clone()).collect();
    let labels: Vec<String> = cells.iter().map(|c| c.value.clone()).collect();

    // 4. wait for ALL cell jobs to reach a terminal state. CRITICAL: snapshot under a
    //    short lock, RELEASE it, then sleep — never hold st.jobs across the await.
    let start = Instant::now();
    let budget = (GRID_PER_CELL_TIMEOUT * ids.len() as u32).min(GRID_TIMEOUT_MAX);
    let mut last_snapshot: Vec<(String, String)>;
    let timed_out;
    loop {
        last_snapshot = snapshot_states(&st, &ids);
        let all_terminal = last_snapshot.iter().all(|(s, _)| is_terminal(s) || s == "missing");
        if all_terminal {
            timed_out = false;
            break;
        }
        if start.elapsed() >= budget {
            timed_out = true;
            break;
        }
        tokio::time::sleep(POLL).await;
    }

    // 5. composite: a done cell with a readable output_path renders its image; a
    //    failed/missing/empty-path cell renders a placeholder.
    let comp_cells: Vec<CompCell> = labels
        .iter()
        .zip(last_snapshot.iter())
        .map(|(label, (state, out))| {
            let src = if state == "done" && !out.is_empty() {
                Some(out.clone())
            } else {
                None
            };
            CompCell { label: format!("{}={}", axis.as_str(), label), src }
        })
        .collect();

    // grid id off the same counter as jobs (so grid-NNNN never collides with a future job).
    let gn = st.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let grid_id = format!("grid-{gn:04}");

    let path = match render_grid(st.out_dir.as_path(), &grid_id, &comp_cells) {
        Ok(p) => p,
        Err(e) => return err_detail(StatusCode::INTERNAL_SERVER_ERROR, &e),
    };

    // per-cell report (value, job_id, output_path) — empty output_path for non-done cells.
    let cells_json: Vec<Value> = cells
        .iter()
        .zip(last_snapshot.iter())
        .map(|(c, (_, out))| {
            json!({
                "value": c.value,
                "job_id": c.job_id,
                "output_path": out,
            })
        })
        .collect();

    let doc = json!({
        "grid_id": grid_id,
        "path": path,
        "axis": axis.as_str(),
        "cells": cells_json,
    });

    if timed_out {
        // Partial grid: some cells never reached terminal within the timeout. Report a
        // 504 but still include the (partial) grid path + the per-cell states.
        let pending: Vec<&str> = last_snapshot
            .iter()
            .zip(ids.iter())
            .filter(|((s, _), _)| !(is_terminal(s) || *s == "missing"))
            .map(|(_, id)| id.as_str())
            .collect();
        let mut doc2 = doc;
        if let Some(obj) = doc2.as_object_mut() {
            obj.insert("detail".into(), json!("grid timed out waiting for cells"));
            obj.insert("timed_out".into(), json!(true));
            obj.insert("pending".into(), json!(pending));
        }
        return json_compact(StatusCode::GATEWAY_TIMEOUT, &doc2);
    }

    json_compact(StatusCode::OK, &doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_parse_and_type() {
        assert!(Axis::parse("cfg").unwrap().is_numeric());
        assert!(Axis::parse("seed").unwrap().is_numeric());
        assert!(Axis::parse("steps").unwrap().is_numeric());
        assert!(!Axis::parse("sampler").unwrap().is_numeric());
        assert!(!Axis::parse("scheduler").unwrap().is_numeric());
        assert!(Axis::parse("bogus").is_none());
    }

    #[test]
    fn fmt_num_labels() {
        assert_eq!(fmt_num(3.0), "3.0");
        assert_eq!(fmt_num(7.5), "7.5");
        assert_eq!(fmt_num(1.250), "1.25");
        assert_eq!(fmt_num(10.0), "10.0");
    }

    #[test]
    fn text_width_matches_glyph_layout() {
        // 3 glyphs => 3*5 + 2*1 = 17
        assert_eq!(text_px_width("abc"), 17);
        assert_eq!(text_px_width(""), 0);
        assert_eq!(text_px_width("x"), 5);
    }

    #[test]
    fn apply_axis_sets_field_and_label() {
        let mut p = JobParams::default();
        assert_eq!(apply_axis(&mut p, Axis::Cfg, &json!(3.0)).unwrap(), "3.0");
        assert_eq!(p.cfg, 3.0);
        assert_eq!(apply_axis(&mut p, Axis::Seed, &json!(42)).unwrap(), "42");
        assert_eq!(p.seed, 42);
        assert_eq!(apply_axis(&mut p, Axis::Steps, &json!(8)).unwrap(), "8");
        assert_eq!(p.steps, 8);
        assert_eq!(apply_axis(&mut p, Axis::Sampler, &json!("euler")).unwrap(), "euler");
        assert_eq!(p.sampler, "euler");
        // wrong JSON type for axis -> Err
        assert!(apply_axis(&mut p, Axis::Cfg, &json!("nope")).is_err());
        assert!(apply_axis(&mut p, Axis::Sampler, &json!(1)).is_err());
    }

    #[test]
    fn grid_renders_with_placeholder_cells() {
        let dir = std::env::temp_dir().join(format!("grid_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let cells = vec![
            CompCell { label: "cfg=3.0".into(), src: None },
            CompCell { label: "cfg=5.0".into(), src: Some("/no/such/file.png".into()) },
            CompCell { label: "cfg=7.0".into(), src: None },
        ];
        let p = render_grid(&dir, "grid-0001", &cells).unwrap();
        assert!(std::path::Path::new(&p).exists());
        // 3 cells -> cols = ceil(sqrt(3)) = 2, rows = 2
        let img = image::open(&p).unwrap();
        let cols = 2u32;
        let rows = 2u32;
        let expect_w = PAD + cols * (CELL + PAD);
        let expect_h = PAD + rows * (CELL + LABEL_BAND + PAD);
        assert_eq!(img.width(), expect_w);
        assert_eq!(img.height(), expect_h);
        let _ = std::fs::remove_file(&p);
    }
}
