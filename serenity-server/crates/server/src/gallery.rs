//! /v1/gallery — faithful port of the daemon's generated-image gallery
//! (serenity_daemon.mojo @3001 + the gallery helpers @1582-2110). READ PATH:
//! `GET /v1/gallery`, `GET /v1/gallery/:id`, `GET /v1/gallery/read?path=`.
//!
//! Scans `<out_dir>/job-*.png` for the embedded `serenity.genparams.v1` tEXt chunk
//! (read_png_text_bytes: uncompressed tEXt only, CRC-verified) + favorites/names/
//! order/imports state at `<out_dir>/state/gallery.json`, generates 256px thumbnails
//! under `<out_dir>/thumbnails/`, and emits `serenity.gallery.v1` items.
//!
//! PATH REPRESENTATION: the Rust server uses its (absolute, canonicalized) out_dir for
//! the path strings (`path`, `thumbnail_path*`, `thumbnail_path_root`); the daemon used
//! the relative literal `output/serenity_daemon`. The LOGIC (which items, fields, order,
//! params extraction) is byte-faithful; only the path prefix differs (absolute is more
//! robust). Verified logic-identical vs the oracle with the path prefix normalized.

use std::collections::HashMap;
use std::path::Path;

use axum::extract::{Path as AxPath, Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::AppState;

const GENPARAMS_TEXT_KEY: &str = "serenity.genparams.v1";

// ── PNG tEXt reader (read_png_text_bytes: tEXt only, CRC-verified) ──────────────

fn crc32(buf: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in buf {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Every (keyword, value) tEXt pair, in file order. Err on bad signature / CRC /
/// malformed tEXt / chunk overrun (mirrors the daemon's raising parser).
fn png_text_pairs(path: &str) -> Result<Vec<(String, String)>, String> {
    // Mirror Mojo `open()`'s error text: "Failed to open file '<path>': <reason>"
    // (std's Display appends " (os error N)" which Mojo's does not — strip it).
    let data = std::fs::read(path).map_err(|e| {
        let msg = e.to_string();
        let reason = msg.split(" (os error").next().unwrap_or(&msg);
        format!("Failed to open file '{path}': {reason}")
    })?;
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if data.len() < 8 {
        return Err("png: too short".into());
    }
    if data[..8] != SIG {
        return Err("png: bad signature".into());
    }
    let mut out = Vec::new();
    let mut pos = 8usize;
    while pos + 8 <= data.len() {
        let clen = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let ctype_off = pos + 4;
        let ctype = &data[ctype_off..ctype_off + 4];
        let dstart = pos + 8;
        if dstart + clen + 4 > data.len() {
            return Err(format!("png: chunk overruns file: {}", String::from_utf8_lossy(ctype)));
        }
        if ctype == b"tEXt" {
            let stored = u32::from_be_bytes([
                data[dstart + clen], data[dstart + clen + 1], data[dstart + clen + 2], data[dstart + clen + 3],
            ]);
            let calc = crc32(&data[ctype_off..ctype_off + 4 + clen]);
            if stored != calc {
                return Err("png: CRC mismatch in tEXt chunk".into());
            }
            let chunk = &data[dstart..dstart + clen];
            let sep = chunk.iter().position(|&b| b == 0).unwrap_or(0);
            if sep < 1 {
                return Err("png: malformed tEXt (no keyword separator)".into());
            }
            let kw = String::from_utf8_lossy(&chunk[..sep]).into_owned();
            let val = String::from_utf8_lossy(&chunk[sep + 1..]).into_owned();
            out.push((kw, val));
        }
        if ctype == b"IEND" {
            break;
        }
        pos = dstart + clen + 4;
    }
    Ok(out)
}

fn png_genparams(path: &str) -> Result<String, String> {
    Ok(png_text_pairs(path)?
        .into_iter()
        .find(|(k, _)| k == GENPARAMS_TEXT_KEY)
        .map(|(_, v)| v)
        .unwrap_or_default())
}

pub fn png_genparams_or_empty(path: &str) -> String {
    png_genparams(path).unwrap_or_default()
}

// ── paths ───────────────────────────────────────────────────────────────────────

fn state_file(out: &Path) -> std::path::PathBuf {
    out.join("state").join("gallery.json")
}
fn thumb_dir(out: &Path) -> std::path::PathBuf {
    out.join("thumbnails")
}
fn thumb_path(out: &Path, id: &str) -> String {
    thumb_dir(out).join(format!("{id}.png")).to_string_lossy().into_owned()
}
fn png_path(out: &Path, id: &str) -> String {
    out.join(format!("{id}.png")).to_string_lossy().into_owned()
}

// ── gallery state ───────────────────────────────────────────────────────────────

fn gallery_state_default() -> Value {
    json!({
        "schema": "serenity.gallery_state.v1",
        "favorites": [], "names": [], "order": [], "imports": [],
    })
}

fn load_gallery_state(out: &Path) -> Value {
    let _ = std::fs::create_dir_all(out.join("state"));
    let _ = std::fs::create_dir_all(thumb_dir(out));
    std::fs::read_to_string(state_file(out))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .filter(|d| d.is_object() && d.get("favorites").map_or(false, |f| f.is_array()))
        .unwrap_or_else(gallery_state_default)
}

fn gallery_favorite(state: &Value, id: &str) -> bool {
    state.get("favorites").and_then(|f| f.as_array()).map_or(false, |a| {
        a.iter().any(|v| v.as_str() == Some(id))
    })
}

fn gallery_name(state: &Value, id: &str) -> String {
    if let Some(arr) = state.get("names").and_then(|n| n.as_array()) {
        for ent in arr {
            if ent.get("id").and_then(|v| v.as_str()) == Some(id) {
                if let Some(n) = ent.get("name").and_then(|v| v.as_str()) {
                    return n.to_string();
                }
            }
        }
    }
    id.to_string()
}

fn gallery_import_source(state: &Value, id: &str) -> String {
    if let Some(arr) = state.get("imports").and_then(|n| n.as_array()) {
        for ent in arr {
            if ent.get("id").and_then(|v| v.as_str()) == Some(id) {
                if let Some(s) = ent.get("source_path").and_then(|v| v.as_str()) {
                    return s.to_string();
                }
            }
        }
    }
    String::new()
}

fn gallery_order_index(state: &Value, id: &str) -> i64 {
    if let Some(arr) = state.get("order").and_then(|n| n.as_array()) {
        for (i, v) in arr.iter().enumerate() {
            if v.as_str() == Some(id) {
                return i as i64;
            }
        }
    }
    -1
}

// ── thumbnails (decode → 256px lanczos → encode; skip if cached) ─────────────────

fn ensure_thumbnail(out: &Path, id: &str, png: &str) -> Result<String, String> {
    let _ = std::fs::create_dir_all(thumb_dir(out));
    let tp = thumb_dir(out).join(format!("{id}.png"));
    let tps = tp.to_string_lossy().into_owned();
    if tp.exists() {
        return Ok(tps);
    }
    let img = image::open(png).map_err(|e| e.to_string())?;
    let (w, h) = (img.width().max(1), img.height().max(1));
    let (mut tw, mut th) = (256u32, 256u32);
    if w >= h {
        th = ((h as u64 * 256) / w as u64) as u32;
        th = th.max(1);
    } else {
        tw = ((w as u64 * 256) / h as u64) as u32;
        tw = tw.max(1);
    }
    let thumb = img.resize_exact(tw, th, image::imageops::FilterType::Lanczos3);
    thumb.save_with_format(&tp, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(tps)
}

// ── scan / search / filter / sort ────────────────────────────────────────────────

fn scan_gallery_ids(out: &Path) -> Vec<String> {
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(out) {
        for ent in rd.flatten() {
            let f = ent.file_name().to_string_lossy().into_owned();
            if f.starts_with("job-") && f.ends_with(".png") {
                if std::fs::metadata(ent.path()).map_or(false, |m| m.is_file()) {
                    ids.push(f[..f.len() - 4].to_string());
                }
            }
        }
    }
    ids.sort(); // find | sort
    ids
}

fn contains_ci(text: &str, q: &str) -> bool {
    q.is_empty() || text.to_lowercase().contains(&q.to_lowercase())
}

fn search_matches(id: &str, path: &str, params_json: &str, search: &str) -> bool {
    search.is_empty() || contains_ci(id, search) || contains_ci(path, search) || contains_ci(params_json, search)
}

fn filter_matches(params_json: &str, favorite: bool, filter: &str, favorite_query: &str) -> bool {
    let favq = favorite_query.to_lowercase();
    if (favq == "1" || favq == "true" || favq == "yes") && !favorite {
        return false;
    }
    if (favq == "0" || favq == "false" || favq == "no") && favorite {
        return false;
    }
    let f = filter.to_lowercase();
    if f.is_empty() || f == "all" || f == "any" {
        return true;
    }
    if f == "favorite" || f == "favorites" || f == "star" || f == "starred" {
        return favorite;
    }
    if f == "has_params" {
        return !params_json.is_empty();
    }
    if f == "missing_params" {
        return params_json.is_empty();
    }
    contains_ci(params_json, filter)
}

fn id_num(id: &str) -> i64 {
    id.strip_prefix("job-").and_then(|s| s.parse::<i64>().ok()).unwrap_or(0)
}

/// Precomputed per-candidate facts for sort/filter (params, favorite, size, order).
struct Cand {
    id: String,
    favorite: bool,
    size: i64,
    order_index: i64,
}

/// `_gallery_id_before` as a total ordering (id_num is the unique tiebreak).
fn id_before_cmp(a: &Cand, b: &Cand, sort: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let s = sort.to_lowercase();
    let primary = match s.as_str() {
        "manual" | "manual_asc" | "order" => {
            let (ai, bi) = (a.order_index, b.order_index);
            if ai >= 0 && bi >= 0 && ai != bi {
                ai.cmp(&bi)
            } else if ai >= 0 && bi < 0 {
                Ordering::Less
            } else if ai < 0 && bi >= 0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        "name_asc" => a.id.cmp(&b.id),
        "name_desc" => b.id.cmp(&a.id),
        "created_asc" => id_num(&a.id).cmp(&id_num(&b.id)),
        "size_asc" => a.size.cmp(&b.size),
        "size_desc" => b.size.cmp(&a.size),
        "favorite_desc" => b.favorite.cmp(&a.favorite),
        _ => Ordering::Equal,
    };
    if primary != Ordering::Equal {
        return primary;
    }
    // default tiebreak: id_num DESC (newest first)
    id_num(&b.id).cmp(&id_num(&a.id))
}

// ── item builder ──────────────────────────────────────────────────────────────────

fn item_from_png_state(out: &Path, state: &Value, id: &str, path: &str, favorite: bool, ensure_thumb: bool) -> Value {
    let params_json = png_genparams_or_empty(path);
    let mut thumb = String::new();
    let mut thumb_state = "not_requested".to_string();
    if ensure_thumb && !id.is_empty() {
        match ensure_thumbnail(out, id, path) {
            Ok(p) => {
                thumb = p;
                thumb_state = "cached".to_string();
            }
            Err(e) => thumb_state = format!("error: {e}"),
        }
    }
    let size = std::fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
    let name = gallery_name(state, id);
    let imported_from = gallery_import_source(state, id);
    let mut o = serde_json::Map::new();
    o.insert("id".into(), json!(id));
    o.insert("name".into(), json!(name));
    o.insert("display_name".into(), json!(name));
    o.insert("path".into(), json!(path));
    o.insert("size".into(), json!(size));
    o.insert("favorite".into(), json!(favorite));
    o.insert("manual_order_index".into(), json!(gallery_order_index(state, id)));
    o.insert("imported".into(), json!(!imported_from.is_empty()));
    o.insert("imported_from".into(), json!(imported_from));
    o.insert("thumbnail_path".into(), json!(thumb));
    o.insert("thumb_path".into(), json!(thumb));
    o.insert("thumbnail".into(), json!(thumb));
    o.insert("thumbnail_state".into(), json!(thumb_state));
    o.insert("metadata_key".into(), json!(GENPARAMS_TEXT_KEY));
    o.insert("has_params".into(), json!(!params_json.is_empty()));
    o.insert("params_json".into(), json!(params_json));
    if !params_json.is_empty() {
        match serde_json::from_str::<Value>(&params_json) {
            Ok(mut params) => {
                for k in ["model", "prompt", "seed", "width", "height"] {
                    if let Some(v) = params.get(k) {
                        o.insert(k.into(), v.clone());
                    }
                }
                if !id.is_empty() {
                    if let Some(m) = params.as_object_mut() {
                        m.entry("params_source").or_insert(json!("gallery"));
                        m.entry("reused_from_gallery_id").or_insert(json!(id));
                        m.entry("reused_from_job_id").or_insert(json!(id));
                        m.entry("reused_from_path").or_insert(json!(path));
                    }
                }
                o.insert("params".into(), params);
            }
            Err(e) => {
                o.insert("params_error".into(), json!(e.to_string()));
            }
        }
    }
    Value::Object(o)
}

fn error_item(path: &str, id: &str, err: &str) -> Value {
    let size = std::fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
    json!({
        "id": id, "name": id, "display_name": id, "path": path, "size": size,
        "favorite": false, "thumbnail_path": "", "thumb_path": "", "thumbnail": "",
        "thumbnail_state": "error", "metadata_key": GENPARAMS_TEXT_KEY,
        "has_params": false, "params_json": "", "error": err,
    })
}

fn safe_gallery_id(id: &str) -> bool {
    !id.is_empty() && id.starts_with("job-") && !id.contains('/')
}

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

// ── handlers ──────────────────────────────────────────────────────────────────────

/// GET /v1/gallery — list job PNGs + embedded genparams (`serenity.gallery.v1`).
pub async fn get_gallery(State(st): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let out = st.out_dir.as_path();
    let g = |k: &str| q.get(k).cloned().unwrap_or_default();
    let mut search = g("search");
    let qp = g("q");
    if search.is_empty() && !qp.is_empty() {
        search = qp;
    }
    let filter = g("filter");
    let sort = g("sort");
    let favorite_query = g("favorite");
    let state = load_gallery_state(out);
    let ids = scan_gallery_ids(out);

    // filter, then sort by the gallery_id_before total order (== the daemon's selection sort)
    let mut cands: Vec<Cand> = Vec::new();
    for id in &ids {
        let p = png_path(out, id);
        let params = png_genparams_or_empty(&p);
        let fav = gallery_favorite(&state, id);
        if !search_matches(id, &p, &params, &search) {
            continue;
        }
        if !filter_matches(&params, fav, &filter, &favorite_query) {
            continue;
        }
        let size = std::fs::metadata(&p).map(|m| m.len() as i64).unwrap_or(0);
        cands.push(Cand { id: id.clone(), favorite: fav, size, order_index: gallery_order_index(&state, id) });
    }
    cands.sort_by(|a, b| id_before_cmp(a, b, &sort));

    let items: Vec<Value> = cands
        .iter()
        .map(|c| {
            let p = png_path(out, &c.id);
            // mirrors the daemon try/except -> error item on a malformed PNG
            match png_genparams(&p) {
                Ok(_) => item_from_png_state(out, &state, &c.id, &p, c.favorite, true),
                Err(e) => error_item(&p, &c.id, &e),
            }
        })
        .collect();

    let doc = json!({
        "schema": "serenity.gallery.v1",
        "search": search,
        "filter": filter,
        "sort": sort,
        "favorite": favorite_query,
        "thumbnail_path_root": thumb_dir(out).to_string_lossy(),
        "count": items.len(),
        "total": ids.len(),
        "items": items,
    });
    json_compact(StatusCode::OK, &doc)
}

/// GET /v1/gallery/read?path=<png> — read any local PNG's genparams as an item.
pub async fn get_gallery_read(State(st): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let png = q.get("path").cloned().unwrap_or_default();
    if png.is_empty() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'path' query parameter is required");
    }
    match png_genparams(&png) {
        Ok(_) => json_compact(StatusCode::OK, &item_from_png_state(st.out_dir.as_path(), &load_gallery_state(st.out_dir.as_path()), "", &png, false, true)),
        Err(e) => err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("cannot read PNG genparams: {e}")),
    }
}

/// GET /v1/gallery/:id — one gallery item by job id.
pub async fn get_gallery_one(State(st): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    if !safe_gallery_id(&id) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("invalid gallery id: {id}"));
    }
    let out = st.out_dir.as_path();
    let p = png_path(out, &id);
    let state = load_gallery_state(out);
    match png_genparams(&p) {
        Ok(_) => json_compact(StatusCode::OK, &item_from_png_state(out, &state, &id, &p, gallery_favorite(&state, &id), true)),
        Err(e) => err_detail(StatusCode::NOT_FOUND, &format!("cannot read gallery item: {e}")),
    }
}

// ── state mutators (sub-routes) ─────────────────────────────────────────────────

fn state_arr(doc: &Value, key: &str) -> Value {
    doc.get(key).filter(|v| v.is_array()).cloned().unwrap_or_else(|| json!([]))
}

fn save_gallery_state(out: &Path, doc: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(out.join("state"))?;
    let _ = std::fs::create_dir_all(thumb_dir(out));
    std::fs::write(state_file(out), serde_json::to_string(doc).unwrap())
}

/// `_set_gallery_favorite_doc` — rebuild favorites, preserve names/order/imports.
fn set_favorite_doc(doc: &Value, id: &str, favorite: bool) -> Value {
    let mut favs = Vec::new();
    let mut found = false;
    if let Some(arr) = doc.get("favorites").and_then(|f| f.as_array()) {
        for v in arr {
            if let Some(s) = v.as_str() {
                if s == id {
                    found = true;
                    if favorite {
                        favs.push(json!(id));
                    }
                } else {
                    favs.push(json!(s));
                }
            }
        }
    }
    if favorite && !found {
        favs.push(json!(id));
    }
    json!({ "schema": "serenity.gallery_state.v1", "favorites": favs,
        "names": state_arr(doc, "names"), "order": state_arr(doc, "order"), "imports": state_arr(doc, "imports") })
}

/// `_set_gallery_name_doc` — upsert {id,name} in names, preserve the rest.
fn set_name_doc(doc: &Value, id: &str, name: &str) -> Value {
    let mut names = Vec::new();
    let mut replaced = false;
    if let Some(arr) = doc.get("names").and_then(|n| n.as_array()) {
        for ent in arr {
            if ent.get("id").and_then(|v| v.as_str()) == Some(id) {
                names.push(json!({ "id": id, "name": name }));
                replaced = true;
            } else {
                names.push(ent.clone());
            }
        }
    }
    if !replaced {
        names.push(json!({ "id": id, "name": name }));
    }
    json!({ "schema": "serenity.gallery_state.v1", "favorites": state_arr(doc, "favorites"),
        "names": names, "order": state_arr(doc, "order"), "imports": state_arr(doc, "imports") })
}

/// `_set_gallery_order_doc` — validate each id (safe + png exists), set order.
fn set_order_doc(out: &Path, doc: &Value, ids: &Value) -> Result<Value, String> {
    let arr = ids.as_array().ok_or("'ids' must be an array of gallery ids")?;
    let mut order = Vec::new();
    for (i, v) in arr.iter().enumerate() {
        let id = v.as_str().ok_or_else(|| format!("'ids[{i}]' must be a string"))?;
        if !safe_gallery_id(id) {
            return Err(format!("invalid gallery id: {id}"));
        }
        if !Path::new(&png_path(out, id)).exists() {
            return Err(format!("gallery item not found: {id}"));
        }
        order.push(json!(id));
    }
    Ok(json!({ "schema": "serenity.gallery_state.v1", "favorites": state_arr(doc, "favorites"),
        "names": state_arr(doc, "names"), "order": order, "imports": state_arr(doc, "imports") }))
}

/// `_remove_gallery_item_doc` — drop `id` from every array.
fn remove_item_doc(doc: &Value, id: &str) -> Value {
    let strip_strs = |key: &str| -> Vec<Value> {
        doc.get(key).and_then(|a| a.as_array())
            .map(|a| a.iter().filter(|v| v.as_str() != Some(id)).cloned().collect())
            .unwrap_or_default()
    };
    let strip_objs = |key: &str| -> Vec<Value> {
        doc.get(key).and_then(|a| a.as_array())
            .map(|a| a.iter().filter(|e| e.get("id").and_then(|v| v.as_str()) != Some(id)).cloned().collect())
            .unwrap_or_default()
    };
    json!({ "schema": "serenity.gallery_state.v1", "favorites": strip_strs("favorites"),
        "names": strip_objs("names"), "order": strip_strs("order"), "imports": strip_objs("imports") })
}

/// `_set_gallery_import_doc` — upsert {id, source_path} in imports, preserve the rest.
fn set_import_doc(doc: &Value, id: &str, source_path: &str) -> Value {
    let mut imports = Vec::new();
    let mut replaced = false;
    if let Some(arr) = doc.get("imports").and_then(|n| n.as_array()) {
        for ent in arr {
            if ent.get("id").and_then(|v| v.as_str()) == Some(id) {
                imports.push(json!({ "id": id, "source_path": source_path }));
                replaced = true;
            } else {
                imports.push(ent.clone());
            }
        }
    }
    if !replaced {
        imports.push(json!({ "id": id, "source_path": source_path }));
    }
    json!({ "schema": "serenity.gallery_state.v1", "favorites": state_arr(doc, "favorites"),
        "names": state_arr(doc, "names"), "order": state_arr(doc, "order"), "imports": imports })
}

fn required_string(body: &Value, key: &str) -> Result<String, String> {
    match body.get(key).and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => Ok(s.to_string()),
        Some(_) => Err(format!("'{key}' must be non-empty")),
        None => Err(format!("'{key}' (string) is required")),
    }
}

fn body_object(body: &str) -> Value {
    serde_json::from_str::<Value>(body).ok().filter(|v| v.is_object()).unwrap_or_else(|| json!({}))
}

/// POST /v1/gallery/:id/favorite — set/clear favorite, return the item.
pub async fn post_favorite(State(st): State<AppState>, AxPath(id): AxPath<String>, body: String) -> Response {
    let out = st.out_dir.as_path();
    if !safe_gallery_id(&id) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("invalid gallery id: {id}"));
    }
    let p = png_path(out, &id);
    if !Path::new(&p).exists() {
        return err_detail(StatusCode::NOT_FOUND, &format!("gallery item not found: {id}"));
    }
    let b = body_object(&body);
    let favorite = match b.get("favorite") {
        None => true,
        Some(v) if v.is_boolean() => v.as_bool().unwrap(),
        Some(_) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'favorite' must be a boolean"),
    };
    let next = set_favorite_doc(&load_gallery_state(out), &id, favorite);
    if save_gallery_state(out, &next).is_err() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot persist gallery state");
    }
    json_compact(StatusCode::OK, &item_from_png_state(out, &next, &id, &p, favorite, true))
}

/// POST /v1/gallery/:id/rename — set display name, return the item.
pub async fn post_rename(State(st): State<AppState>, AxPath(id): AxPath<String>, body: String) -> Response {
    let out = st.out_dir.as_path();
    if !safe_gallery_id(&id) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("invalid gallery id: {id}"));
    }
    let p = png_path(out, &id);
    if !Path::new(&p).exists() {
        return err_detail(StatusCode::NOT_FOUND, &format!("gallery item not found: {id}"));
    }
    let name = match required_string(&body_object(&body), "name") {
        Ok(n) => n,
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };
    let next = set_name_doc(&load_gallery_state(out), &id, &name);
    if save_gallery_state(out, &next).is_err() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot persist gallery state");
    }
    json_compact(StatusCode::OK, &item_from_png_state(out, &next, &id, &p, gallery_favorite(&next, &id), true))
}

/// POST /v1/gallery/order — set the manual order; returns {schema, order}.
pub async fn post_order(State(st): State<AppState>, body: String) -> Response {
    let out = st.out_dir.as_path();
    let b = body_object(&body);
    let ids = match b.get("ids") {
        Some(v) => v.clone(),
        None => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "'ids' array is required"),
    };
    let next = match set_order_doc(out, &load_gallery_state(out), &ids) {
        Ok(d) => d,
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };
    if save_gallery_state(out, &next).is_err() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "cannot persist gallery state");
    }
    json_compact(StatusCode::OK, &json!({ "schema": "serenity.gallery_order.v1", "order": next.get("order").cloned().unwrap_or(json!([])) }))
}

/// POST /v1/gallery/import — copy a local PNG (with genparams) in as a new job-XXXX
/// gallery item, recording the import source. Uses the shared job counter.
pub async fn post_import(State(st): State<AppState>, body: String) -> Response {
    let out = st.out_dir.as_path();
    let b = body_object(&body);
    let source = match required_string(&b, "path") {
        Ok(s) => s,
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    };
    if source.contains('\n') || source.contains('\r') {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "invalid gallery import path");
    }
    if !Path::new(&source).is_file() {
        return err_detail(StatusCode::NOT_FOUND, &format!("gallery import source not found: {source}"));
    }
    // must carry the genparams tEXt (else nothing to import)
    match png_genparams(&source) {
        Ok(p) if !p.is_empty() => {}
        Ok(_) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("gallery import source has no {GENPARAMS_TEXT_KEY}")),
        Err(e) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &e),
    }
    let n = st.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let id = format!("job-{n:04}");
    let dest = png_path(out, &id);
    if std::fs::copy(&source, &dest).is_err() {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, "gallery import copy failed");
    }
    let next = set_import_doc(&load_gallery_state(out), &id, &source);
    let _ = save_gallery_state(out, &next);
    json_compact(StatusCode::OK, &item_from_png_state(out, &next, &id, &dest, gallery_favorite(&next, &id), true))
}

/// DELETE /v1/gallery/:id — unlink the PNG (+ thumb) and drop it from state.
pub async fn delete_item(State(st): State<AppState>, AxPath(id): AxPath<String>) -> Response {
    let out = st.out_dir.as_path();
    if !safe_gallery_id(&id) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &format!("invalid gallery id: {id}"));
    }
    let p = png_path(out, &id);
    if !Path::new(&p).exists() {
        return err_detail(StatusCode::NOT_FOUND, &format!("gallery item not found: {id}"));
    }
    let deleted = std::fs::remove_file(&p).is_ok();
    let tp = thumb_path(out, &id);
    let mut thumb_deleted = false;
    if Path::new(&tp).exists() {
        thumb_deleted = std::fs::remove_file(&tp).is_ok();
    }
    let next = remove_item_doc(&load_gallery_state(out), &id);
    let _ = save_gallery_state(out, &next);
    json_compact(StatusCode::OK, &json!({
        "id": id, "deleted": deleted, "path": p, "thumbnail_path": tp, "thumbnail_deleted": thumb_deleted,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_answer() {
        // standard CRC-32/ISO-HDLC check value for "123456789"
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn id_num_and_safe_id() {
        assert_eq!(id_num("job-0042"), 42);
        assert_eq!(id_num("job-0001"), 1);
        assert_eq!(id_num("notjob"), 0);
        assert!(safe_gallery_id("job-0001"));
        assert!(!safe_gallery_id("notjob"));
        assert!(!safe_gallery_id("job-../x"));
        assert!(!safe_gallery_id(""));
    }

    #[test]
    fn filter_semantics() {
        assert!(filter_matches("p", true, "favorite", ""));
        assert!(!filter_matches("p", false, "favorite", ""));
        assert!(filter_matches("", false, "missing_params", ""));
        assert!(!filter_matches("x", false, "missing_params", ""));
        assert!(filter_matches("x", false, "has_params", ""));
        // favorite query gate
        assert!(!filter_matches("x", false, "", "true"));
        assert!(filter_matches("x", true, "", "true"));
    }
}
