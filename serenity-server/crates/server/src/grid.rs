//! POST /v1/grid — SwarmUI-style XYZ-plot / parameter-sweep grid generator.
//!
//! Multi-axis (X / Y / Z): the cartesian product of up to three axes (each
//! seed|cfg|steps|sampler|scheduler) becomes one cell job apiece. Enqueues one
//! generate job per product cell (reusing the EXACT `/v1/generate` enqueue
//! mechanics — build a `JobParams`, register a `JobChannel`, push a
//! `jobs::JobEntry`, wake the driver), waits for every cell job to reach a
//! terminal state, then composites the produced PNGs into one labelled 2-D grid
//! image PER Z-PAGE (X across, Y down). Single Z (or no Z) is one page.
//!
//! CONTRACT
//!   request  (flat JSON, like /v1/generate, plus grid keys):
//!     LEGACY single-axis (still accepted):
//!       { "axis": "seed"|"cfg"|"steps"|"sampler"|"scheduler",
//!         "values": [ ... ], ...generate fields... }
//!     MULTI-axis (X required; Y, Z optional):
//!       { "x_axis": "<axis>", "x_values": [ ... ],
//!         "y_axis"?: "<axis>", "y_values"?: [ ... ],
//!         "z_axis"?: "<axis>", "z_values"?: [ ... ],
//!         "model": "...", "prompt": "...",   // + the usual flat generate fields
//!         "negative"?, "width"?, "height"?, "steps"?, "seed"?,
//!         "sampler"?, "scheduler"?, "cfg"?, "out_dir"? }
//!     Numeric axes (seed/cfg/steps) consume JSON numbers; string axes
//!     (sampler/scheduler) consume JSON strings. The SAME axis may not be used
//!     on two dimensions. Total cells = |X|*|Y|*|Z| is capped at MAX_CELLS.
//!   response (200):
//!     { "grid_id": "grid-NNNN",
//!       "path": "<abs path to the first/only composite png>",
//!       "paths": [ "<page-0 png>", "<page-1 png>", ... ],   // one per Z value
//!       "axis":  "cfg",            // X axis (back-compat alias)
//!       "x_axis": "cfg", "y_axis": "steps", "z_axis": "seed",  // (omitted if unused)
//!       "cells": [ { "value": "3.0",            // X label (back-compat)
//!                    "x": "3.0", "y": "20", "z": "42",  // per-dim labels
//!                    "job_id": "job-0007", "output_path": "..." }, ... ] }
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

/// Max number of values on ANY single axis (X, Y, or Z).
const MAX_VALUES: usize = 16;
/// Max number of TOTAL cells (|X|*|Y|*|Z|) across the whole cartesian product.
/// Each cell is a serial GPU job + a CELL-square decode; cap keeps the wait and
/// the composite alloc bounded.
const MAX_CELLS: usize = 64;
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
#[derive(Clone, Copy, PartialEq, Debug)]
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
    // Advanced-sampling knobs (clip_skip/eta/sigma_min/sigma_max/restart_sampling/
    // vae): a swept grid must honor the SAME advanced config the user set for a
    // plain Generate, so lift them onto every cell's base params (the per-cell axis
    // value then overrides only its own field). The worker HONORS what it supports
    // and warns-loud on the rest (honor-or-warn); missing keys keep JobParams'
    // "unset" sentinels (clip_skip 0 / eta+sigma -1.0 / restart_sampling false /
    // vae "").
    if let Some(v) = opt_i64(req, "clip_skip") {
        p.clip_skip = v;
    }
    if let Some(v) = opt_f64(req, "eta") {
        p.eta = v;
    }
    if let Some(v) = opt_f64(req, "sigma_min") {
        p.sigma_min = v;
    }
    if let Some(v) = opt_f64(req, "sigma_max") {
        p.sigma_max = v;
    }
    if let Some(v) = req.get("restart_sampling").and_then(|x| x.as_bool()) {
        p.restart_sampling = v;
    }
    if let Some(v) = opt_str(req, "vae") {
        p.vae = v;
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

// ── axis spec + cartesian product ───────────────────────────────────────────────

/// One parsed grid dimension: its axis and the (validated, JSON) value list. The
/// per-value canonical LABELS are computed once (`apply_axis` against a throwaway
/// JobParams) so labelling never re-parses.
#[derive(Debug)]
struct AxisSpec {
    axis: Axis,
    values: Vec<Value>,
    labels: Vec<String>,
}

/// Parse one optional grid dimension from `<prefix>_axis` + `<prefix>_values`.
/// Returns:
///   Ok(None)      — the dimension is absent (no `<prefix>_axis` key),
///   Ok(Some(_))   — a present, fully-validated dimension,
///   Err(msg)      — present but malformed (bad axis / missing/empty/oversized
///                   values / a value of the wrong JSON type for the axis).
/// Computing labels here means a type error fails the request before any enqueue.
fn parse_axis_spec(req: &Value, prefix: &str) -> Result<Option<AxisSpec>, String> {
    let axis_key = format!("{prefix}_axis");
    let axis_str = match opt_str(req, &axis_key) {
        Some(s) if !s.is_empty() => s,
        Some(_) => return Err(format!("'{axis_key}' must not be empty")),
        None => return Ok(None),
    };
    let axis = Axis::parse(&axis_str).ok_or_else(|| {
        format!("'{axis_key}' must be one of: seed, cfg, steps, sampler, scheduler")
    })?;
    let values_key = format!("{prefix}_values");
    let values = req
        .get(&values_key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("'{values_key}' (array) is required when '{axis_key}' is set"))?
        .clone();
    if values.is_empty() {
        return Err(format!("'{values_key}' must not be empty"));
    }
    if values.len() > MAX_VALUES {
        return Err(format!(
            "'{values_key}' has too many entries ({}, max {MAX_VALUES})",
            values.len()
        ));
    }
    // Type-check + compute labels up front (against a throwaway JobParams), so a
    // bad entry fails the whole request before any job is enqueued.
    let mut labels = Vec::with_capacity(values.len());
    let mut throwaway = JobParams::default();
    for (i, v) in values.iter().enumerate() {
        let ok = if axis.is_numeric() { v.is_number() } else { v.is_string() };
        if !ok {
            let want = if axis.is_numeric() { "number" } else { "string" };
            return Err(format!(
                "'{values_key}[{i}]' must be a {want} for axis '{}'",
                axis.as_str()
            ));
        }
        labels.push(apply_axis(&mut throwaway, axis, v)?);
    }
    Ok(Some(AxisSpec { axis, values, labels }))
}

// ── enqueue (replicates post_generate's register→push→wake) ─────────────────────-

/// One enqueued cell. The labels are the canonical per-dimension strings; `value` is
/// the X label (kept for single-axis back-compat in the response). Cells are stored
/// z-outer / y-middle / x-inner, so a cell's (xi, yi, zi) coords are implied by its
/// linear position — see the compositing loop.
struct Cell {
    value: String,    // X-dimension label (back-compat alias of x_label)
    x_label: String,
    y_label: String,  // "" when there is no Y axis
    z_label: String,  // "" when there is no Z axis
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
        // Advanced-sampling knobs in the PNG tEXt so a grid cell round-trips in
        // /v1/gallery reuse-params exactly like a /v1/generate image.
        "clip_skip": params.clip_skip,
        "eta": params.eta,
        "sigma_min": params.sigma_min,
        "sigma_max": params.sigma_max,
        "restart_sampling": params.restart_sampling,
        "vae": params.vae,
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

/// Height of the column-header / row-header / page-title bands, px.
const HEADER_BAND: u32 = 18;
/// Width of the left row-header gutter (Y labels), px.
const ROW_HEADER_W: u32 = 96;

/// One composited cell: its full label (drawn in the per-cell band beneath) and the
/// (optional) source PNG path. A None path (or an unreadable PNG) renders a
/// placeholder tile rather than aborting the whole page.
struct CompCell {
    label: String,
    src: Option<String>,
}

/// Draw one CELL-square tile + its label band at (x0, y0). Decodes `cell.src` (if
/// any) into a CELL square; on any failure draws a placeholder fill + "no image" so
/// a single bad cell never aborts the page (per-cell placeholder-on-failure).
fn draw_cell(canvas: &mut RgbImage, x0: u32, y0: u32, cell: &CompCell) {
    let mut drew_image = false;
    if let Some(src) = &cell.src {
        if let Ok(img) = image::open(src) {
            let resized = img.resize_exact(CELL, CELL, image::imageops::FilterType::Lanczos3);
            let rgb = resized.to_rgb8();
            for yy in 0..CELL.min(rgb.height()) {
                for xx in 0..CELL.min(rgb.width()) {
                    canvas.put_pixel(x0 + xx, y0 + yy, *rgb.get_pixel(xx, yy));
                }
            }
            drew_image = true;
        }
    }
    if !drew_image {
        let fill = if cell.src.is_some() { PLACEHOLDER_BG } else { CELL_BG };
        fill_rect(canvas, x0, y0, CELL, CELL, fill);
        let marker = "no image";
        let tw = text_px_width(marker);
        let mx = x0 + (CELL.saturating_sub(tw)) / 2;
        let my = y0 + CELL / 2 - GLYPH_H / 2;
        draw_text(canvas, mx, my, marker, TEXT);
    }
    // Per-cell label band beneath the tile.
    let by = y0 + CELL;
    fill_rect(canvas, x0, by, CELL, LABEL_BAND, BAND_BG);
    let tw = text_px_width(&cell.label);
    let lx = x0 + (CELL.saturating_sub(tw)) / 2;
    let ly = by + (LABEL_BAND.saturating_sub(GLYPH_H)) / 2;
    draw_text(canvas, lx, ly, &cell.label, TEXT);
}

/// Draw a centered label inside a band rect (used for X column headers / page title).
fn draw_band_centered(canvas: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, s: &str) {
    fill_rect(canvas, x, y, w, h, BAND_BG);
    let tw = text_px_width(s);
    let lx = x + (w.saturating_sub(tw)) / 2;
    let ly = y + (h.saturating_sub(GLYPH_H)) / 2;
    draw_text(canvas, lx, ly, s, TEXT);
}

/// Composite ONE Z-page (X across the columns, Y down the rows) into a labelled PNG
/// and save it to `<out_dir>/<page_id>.png`. `cells` is laid out row-major:
/// `cells[y * cols + x]` (cols = X count, rows = Y count). `x_labels`/`y_labels` are
/// the column/row headers; `page_title` (e.g. the Z label, or the X-axis name for
/// single-axis) labels the whole page. A column for the Y row-header gutter is added
/// only when `y_labels` has >1 entry (multi-row). Returns the absolute output path.
fn render_page(
    out_dir: &std::path::Path,
    page_id: &str,
    cells: &[CompCell],
    cols: usize,
    rows: usize,
    x_labels: &[String],
    y_labels: &[String],
    page_title: &str,
) -> Result<String, String> {
    let cols = cols.max(1) as u32;
    let rows = rows.max(1) as u32;

    // A Y-header gutter + a per-column X-header row appear only when there is more
    // than one row / column respectively; a single-axis page (1 row) draws neither.
    let has_row_header = rows > 1 && y_labels.len() > 1;
    let has_col_header = cols > 1 && x_labels.len() > 1;
    let row_gutter = if has_row_header { ROW_HEADER_W + PAD } else { 0 };
    let title_h = if page_title.is_empty() { 0 } else { HEADER_BAND + PAD };
    let col_header_h = if has_col_header { HEADER_BAND + PAD } else { 0 };

    let cell_total_h = CELL + LABEL_BAND;
    let grid_w = cols * (CELL + PAD);
    let grid_h = rows * (cell_total_h + PAD);
    let img_w = PAD + row_gutter + grid_w;
    let img_h = PAD + title_h + col_header_h + grid_h;

    let mut canvas = RgbImage::from_pixel(img_w, img_h, BG);

    // Page title (Z label / axis name) spanning the full width.
    if title_h > 0 {
        draw_band_centered(&mut canvas, PAD, PAD, img_w - 2 * PAD, HEADER_BAND, page_title);
    }
    let grid_x0 = PAD + row_gutter;
    let grid_y0 = PAD + title_h + col_header_h;

    // X column headers.
    if has_col_header {
        let cy = PAD + title_h;
        for (c, lbl) in x_labels.iter().enumerate().take(cols as usize) {
            let cx = grid_x0 + (c as u32) * (CELL + PAD);
            draw_band_centered(&mut canvas, cx, cy, CELL, HEADER_BAND, lbl);
        }
    }

    // Cells (+ Y row headers down the left gutter).
    for r in 0..rows {
        let y0 = grid_y0 + r * (cell_total_h + PAD);
        if has_row_header {
            if let Some(lbl) = y_labels.get(r as usize) {
                // vertically center the row header against the cell square.
                let hy = y0 + CELL / 2 - GLYPH_H / 2;
                draw_band_centered(&mut canvas, PAD, y0, ROW_HEADER_W, CELL, "");
                let tw = text_px_width(lbl);
                let hx = PAD + (ROW_HEADER_W.saturating_sub(tw)) / 2;
                draw_text(&mut canvas, hx, hy, lbl, TEXT);
            }
        }
        for c in 0..cols {
            let idx = (r * cols + c) as usize;
            if idx >= cells.len() {
                continue;
            }
            let x0 = grid_x0 + c * (CELL + PAD);
            draw_cell(&mut canvas, x0, y0, &cells[idx]);
        }
    }

    let path = out_dir.join(format!("{page_id}.png"));
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

// ── axis-spec collection (legacy single-axis + X/Y/Z) ─────────────────────────────

/// Resolve the request's grid dimensions into an ORDERED `Vec<AxisSpec>` (X, then Y,
/// then Z). Accepts BOTH the legacy single-axis form (`axis`/`values`, mapped onto
/// X) and the new `x_/y_/z_` form. Rejects: no axis at all, a Y/Z without an X, the
/// same axis on two dimensions, and a total cartesian product exceeding MAX_CELLS.
/// (All per-axis validation already happened in `parse_axis_spec`.)
fn collect_specs(req: &Value) -> Result<Vec<AxisSpec>, String> {
    // Legacy `axis`/`values` is treated as the X dimension (translate then reuse the
    // x_ parser path is awkward; parse it directly here).
    let mut specs: Vec<AxisSpec> = Vec::new();
    if req.get("axis").is_some() {
        let axis_str = opt_str(req, "axis")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "'axis' (string) is required".to_string())?;
        let axis = Axis::parse(&axis_str)
            .ok_or_else(|| "'axis' must be one of: seed, cfg, steps, sampler, scheduler".to_string())?;
        let values = req
            .get("values")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "'values' (array) is required".to_string())?
            .clone();
        if values.is_empty() {
            return Err("'values' must not be empty".into());
        }
        if values.len() > MAX_VALUES {
            return Err(format!(
                "'values' has too many entries ({}, max {MAX_VALUES})",
                values.len()
            ));
        }
        let mut labels = Vec::with_capacity(values.len());
        let mut throwaway = JobParams::default();
        for (i, v) in values.iter().enumerate() {
            let ok = if axis.is_numeric() { v.is_number() } else { v.is_string() };
            if !ok {
                let want = if axis.is_numeric() { "number" } else { "string" };
                return Err(format!("'values[{i}]' must be a {want} for axis '{}'", axis.as_str()));
            }
            labels.push(apply_axis(&mut throwaway, axis, v)?);
        }
        specs.push(AxisSpec { axis, values, labels });
    }

    // New x_/y_/z_ form. X may also have been supplied via the legacy keys above; if
    // BOTH legacy `axis` and `x_axis` are present, that's a conflict.
    if let Some(xs) = parse_axis_spec(req, "x")? {
        if !specs.is_empty() {
            return Err("provide either legacy 'axis'/'values' OR 'x_axis'/'x_values', not both".into());
        }
        specs.push(xs);
    }
    if let Some(ys) = parse_axis_spec(req, "y")? {
        specs.push(ys);
    }
    if let Some(zs) = parse_axis_spec(req, "z")? {
        specs.push(zs);
    }

    if specs.is_empty() {
        return Err("at least one axis is required: 'axis'/'values' or 'x_axis'/'x_values'".into());
    }
    // A Y/Z without an X is impossible by construction (Y/Z append after X), but a
    // y_/z_ supplied with NO x_ and NO legacy axis would have failed the empty check.
    // Reject reuse of the same axis on two dimensions.
    for i in 0..specs.len() {
        for j in (i + 1)..specs.len() {
            if specs[i].axis == specs[j].axis {
                return Err(format!(
                    "axis '{}' used on more than one dimension",
                    specs[i].axis.as_str()
                ));
            }
        }
    }
    // Cap the total cartesian product.
    let total: usize = specs.iter().map(|s| s.values.len()).product();
    if total > MAX_CELLS {
        return Err(format!(
            "grid has too many cells ({total}, max {MAX_CELLS}); reduce the value lists"
        ));
    }
    Ok(specs)
}

// ── handler ────────────────────────────────────────────────────────────────────--

/// POST /v1/grid — sweep up to 3 axes (X/Y/Z), enqueue one cell job per cartesian
/// product entry, wait for all to finish, composite one labelled 2-D grid PNG per
/// Z-page. See the module contract.
pub async fn post_grid(State(st): State<AppState>, Json(req): Json<Value>) -> Response {
    // 1. resolve the grid dimensions (legacy single-axis OR x_/y_/z_).
    let specs = match collect_specs(&req) {
        Ok(s) => s,
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };
    let x = &specs[0];
    let y = specs.get(1);
    let z = specs.get(2);

    let nx = x.values.len();
    let ny = y.map(|s| s.values.len()).unwrap_or(1);
    let nz = z.map(|s| s.values.len()).unwrap_or(1);

    // 2. base params from the remaining flat fields (like post_generate).
    let base = match base_params(&st, &req) {
        Ok(p) => p,
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };

    // 3. enqueue one cell per cartesian product entry. Iteration order is
    //    z-outer / y-middle / x-inner so each Z-page's cells are contiguous and
    //    laid out row-major (y down, x across).
    let mut cells: Vec<Cell> = Vec::with_capacity(nx * ny * nz);
    for zi in 0..nz {
        for yi in 0..ny {
            for xi in 0..nx {
                let mut p = base.clone();
                // apply X (always), then Y, then Z — each writes its own field; the
                // labels were precomputed in the AxisSpec, but re-applying yields the
                // SAME canonical label and sets the field on this cell's params.
                let x_label = match apply_axis(&mut p, x.axis, &x.values[xi]) {
                    Ok(l) => l,
                    Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
                };
                let y_label = if let Some(ys) = y {
                    match apply_axis(&mut p, ys.axis, &ys.values[yi]) {
                        Ok(l) => l,
                        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
                    }
                } else {
                    String::new()
                };
                let z_label = if let Some(zs) = z {
                    match apply_axis(&mut p, zs.axis, &zs.values[zi]) {
                        Ok(l) => l,
                        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
                    }
                } else {
                    String::new()
                };
                let job_id = match enqueue_cell(&st, p) {
                    Ok(id) => id,
                    // A driver/lock failure mid-enqueue: surface it (partial jobs already
                    // queued will still run; we report the failure rather than hang).
                    Err(e) => return err_detail(StatusCode::SERVICE_UNAVAILABLE, &e),
                };
                cells.push(Cell {
                    value: x_label.clone(),
                    x_label,
                    y_label,
                    z_label,
                    job_id,
                });
            }
        }
    }

    let ids: Vec<String> = cells.iter().map(|c| c.job_id.clone()).collect();

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

    // 5. composite ONE PNG PER Z-PAGE. Per page, lay the page's cells out row-major
    //    (y down, x across). A done cell with a readable output_path renders its
    //    image; a failed/missing/empty-path cell renders a placeholder.
    let x_labels: Vec<String> = x.labels.clone();
    let y_labels: Vec<String> = y.map(|s| s.labels.clone()).unwrap_or_else(|| vec![String::new()]);

    // grid id off the same counter as jobs (so grid-NNNN never collides with a future job).
    let gn = st.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let grid_id = format!("grid-{gn:04}");

    let mut page_paths: Vec<String> = Vec::with_capacity(nz);
    for zi in 0..nz {
        // gather this page's cells in row-major (yi*nx + xi) order.
        let mut comp: Vec<CompCell> = Vec::with_capacity(nx * ny);
        for yi in 0..ny {
            for xi in 0..nx {
                // find the cell at (xi, yi, zi). cells are stored z-outer/y/x so the
                // linear index is deterministic: zi*ny*nx + yi*nx + xi.
                let lin = zi * ny * nx + yi * nx + xi;
                let c = &cells[lin];
                let (state, out) = &last_snapshot[lin];
                let src = if state == "done" && !out.is_empty() {
                    Some(out.clone())
                } else {
                    None
                };
                // per-cell label: just the X value when single-axis, else x|y coords.
                let label = if y.is_some() {
                    format!("{}|{}", c.x_label, c.y_label)
                } else {
                    format!("{}={}", x.axis.as_str(), c.x_label)
                };
                comp.push(CompCell { label, src });
            }
        }
        // page title: Z label when there is a Z axis, else the X-axis name banner.
        let page_title = if let Some(zs) = z {
            format!("{}={}", zs.axis.as_str(), zs.labels[zi])
        } else if y.is_some() {
            // 2-D page: title shows both axis names.
            format!("X:{}  Y:{}", x.axis.as_str(), y.unwrap().axis.as_str())
        } else {
            String::new()
        };
        let page_id = if nz > 1 { format!("{grid_id}-z{zi:02}") } else { grid_id.clone() };
        let path = match render_page(
            st.out_dir.as_path(),
            &page_id,
            &comp,
            nx,
            ny,
            &x_labels,
            &y_labels,
            &page_title,
        ) {
            Ok(p) => p,
            Err(e) => return err_detail(StatusCode::INTERNAL_SERVER_ERROR, &e),
        };
        page_paths.push(path);
    }

    // per-cell report (value/x/y/z, job_id, output_path) — empty output_path for non-done cells.
    let cells_json: Vec<Value> = cells
        .iter()
        .zip(last_snapshot.iter())
        .map(|(c, (_, out))| {
            json!({
                "value": c.value,
                "x": c.x_label,
                "y": c.y_label,
                "z": c.z_label,
                "job_id": c.job_id,
                "output_path": out,
            })
        })
        .collect();

    let mut doc = json!({
        "grid_id": grid_id,
        "path": page_paths.first().cloned().unwrap_or_default(),
        "paths": page_paths,
        "axis": x.axis.as_str(),
        "x_axis": x.axis.as_str(),
        "cells": cells_json,
    });
    if let Some(obj) = doc.as_object_mut() {
        if let Some(ys) = y {
            obj.insert("y_axis".into(), json!(ys.axis.as_str()));
        }
        if let Some(zs) = z {
            obj.insert("z_axis".into(), json!(zs.axis.as_str()));
        }
    }

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
    fn page_renders_single_axis_one_row() {
        // single-axis (no Y): 1 row, N cols, no row gutter, no col header (cols>1 but
        // x_labels.len()>1 so col header IS drawn), no page title -> title empty.
        let dir = std::env::temp_dir().join(format!("grid_test_a_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let cells = vec![
            CompCell { label: "cfg=3.0".into(), src: None },
            CompCell { label: "cfg=5.0".into(), src: Some("/no/such/file.png".into()) },
            CompCell { label: "cfg=7.0".into(), src: None },
        ];
        let x_labels = vec!["3.0".to_string(), "5.0".to_string(), "7.0".to_string()];
        let y_labels = vec![String::new()];
        let p = render_page(&dir, "grid-page-a", &cells, 3, 1, &x_labels, &y_labels, "").unwrap();
        assert!(std::path::Path::new(&p).exists());
        let img = image::open(&p).unwrap();
        // 3 cols, 1 row. col header drawn (cols>1 && labels>1); no row gutter (1 row);
        // no title.
        let col_header_h = HEADER_BAND + PAD;
        let expect_w = PAD + 3 * (CELL + PAD);
        let expect_h = PAD + col_header_h + 1 * (CELL + LABEL_BAND + PAD);
        assert_eq!(img.width(), expect_w);
        assert_eq!(img.height(), expect_h);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn page_renders_two_axis_with_headers_and_title() {
        // 2-D page: 2 cols x 2 rows, a title, row gutter + col header.
        let dir = std::env::temp_dir().join(format!("grid_test_b_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let cells = vec![
            CompCell { label: "3.0|10".into(), src: None },
            CompCell { label: "5.0|10".into(), src: None },
            CompCell { label: "3.0|20".into(), src: None },
            CompCell { label: "5.0|20".into(), src: None },
        ];
        let x_labels = vec!["3.0".to_string(), "5.0".to_string()];
        let y_labels = vec!["10".to_string(), "20".to_string()];
        let p = render_page(&dir, "grid-page-b", &cells, 2, 2, &x_labels, &y_labels, "X:cfg  Y:steps").unwrap();
        let img = image::open(&p).unwrap();
        let title_h = HEADER_BAND + PAD;
        let col_header_h = HEADER_BAND + PAD;
        let row_gutter = ROW_HEADER_W + PAD;
        let expect_w = PAD + row_gutter + 2 * (CELL + PAD);
        let expect_h = PAD + title_h + col_header_h + 2 * (CELL + LABEL_BAND + PAD);
        assert_eq!(img.width(), expect_w);
        assert_eq!(img.height(), expect_h);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn collect_specs_legacy_single_axis() {
        let req = json!({ "axis": "cfg", "values": [3.0, 5.0, 7.0], "model": "m", "prompt": "p" });
        let specs = collect_specs(&req).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].axis, Axis::Cfg);
        assert_eq!(specs[0].values.len(), 3);
        assert_eq!(specs[0].labels, vec!["3.0", "5.0", "7.0"]);
    }

    #[test]
    fn collect_specs_xyz_product_and_order() {
        let req = json!({
            "x_axis": "cfg", "x_values": [3.0, 5.0],
            "y_axis": "steps", "y_values": [10, 20, 30],
            "z_axis": "seed", "z_values": [1, 2],
        });
        let specs = collect_specs(&req).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].axis, Axis::Cfg); // X
        assert_eq!(specs[1].axis, Axis::Steps); // Y
        assert_eq!(specs[2].axis, Axis::Seed); // Z
        // total product 2*3*2 = 12 (under MAX_CELLS).
        let total: usize = specs.iter().map(|s| s.values.len()).product();
        assert_eq!(total, 12);
    }

    #[test]
    fn collect_specs_rejects_duplicate_axis() {
        let req = json!({
            "x_axis": "cfg", "x_values": [3.0, 5.0],
            "y_axis": "cfg", "y_values": [7.0, 9.0],
        });
        let e = collect_specs(&req).unwrap_err();
        assert!(e.contains("more than one dimension"), "got: {e}");
    }

    #[test]
    fn collect_specs_rejects_legacy_and_x_conflict() {
        let req = json!({
            "axis": "cfg", "values": [3.0],
            "x_axis": "steps", "x_values": [10, 20],
        });
        let e = collect_specs(&req).unwrap_err();
        assert!(e.contains("not both"), "got: {e}");
    }

    #[test]
    fn collect_specs_caps_total_cells() {
        // 16 * 16 = 256 > MAX_CELLS (64).
        let many: Vec<Value> = (0..16).map(|i| json!(i)).collect();
        let req = json!({
            "x_axis": "seed", "x_values": many,
            "y_axis": "steps", "y_values": (0..16).map(|i| json!(i + 1)).collect::<Vec<_>>(),
        });
        let e = collect_specs(&req).unwrap_err();
        assert!(e.contains("too many cells"), "got: {e}");
    }

    #[test]
    fn collect_specs_caps_single_axis_values() {
        let many: Vec<Value> = (0..(MAX_VALUES + 1)).map(|i| json!(i)).collect();
        let req = json!({ "x_axis": "seed", "x_values": many });
        let e = collect_specs(&req).unwrap_err();
        assert!(e.contains("too many entries"), "got: {e}");
    }

    #[test]
    fn collect_specs_rejects_wrong_value_type() {
        let req = json!({ "x_axis": "cfg", "x_values": ["nope"] });
        let e = collect_specs(&req).unwrap_err();
        assert!(e.contains("must be a number"), "got: {e}");
    }

    #[test]
    fn collect_specs_requires_an_axis() {
        let req = json!({ "model": "m", "prompt": "p" });
        let e = collect_specs(&req).unwrap_err();
        assert!(e.contains("at least one axis"), "got: {e}");
    }

    /// The five advanced-sampling knobs are read from the flat grid request by the
    /// SAME accessors `base_params` uses, so a cell's JobParams carries them. (We
    /// test the accessor reads directly — `base_params` needs an AppState.)
    #[test]
    fn advanced_knobs_read_from_request() {
        let req = json!({
            "model": "m", "prompt": "p",
            "clip_skip": 2,
            "eta": 0.3,
            "sigma_min": 0.03,
            "sigma_max": 14.6,
            "restart_sampling": true,
            "vae": "sdxl_vae.safetensors",
        });
        assert_eq!(opt_i64(&req, "clip_skip"), Some(2));
        assert_eq!(opt_f64(&req, "eta"), Some(0.3));
        assert_eq!(opt_f64(&req, "sigma_min"), Some(0.03));
        assert_eq!(opt_f64(&req, "sigma_max"), Some(14.6));
        assert_eq!(req.get("restart_sampling").and_then(|x| x.as_bool()), Some(true));
        assert_eq!(opt_str(&req, "vae").as_deref(), Some("sdxl_vae.safetensors"));
        // absent knobs => None (base_params then keeps the JobParams sentinels)
        let bare = json!({ "model": "m", "prompt": "p" });
        assert_eq!(opt_i64(&bare, "clip_skip"), None);
        assert_eq!(opt_f64(&bare, "eta"), None);
        assert_eq!(bare.get("restart_sampling").and_then(|x| x.as_bool()), None);
        assert_eq!(opt_str(&bare, "vae"), None);
    }
}
