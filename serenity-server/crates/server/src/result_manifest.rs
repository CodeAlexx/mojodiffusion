use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::SystemTime;

use axum::extract::{Path as AxPath, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use serenity_wire::JobParams;

use crate::{jobs, AppState};

const SERVER_RESULT_SUFFIX: &str = ".serenity_server_result.json";
const VISUAL_MIN_AVG_STDDEV: f64 = 18.0;
const VISUAL_MIN_LUMINANCE_RANGE: f64 = 55.0;
const VISUAL_MIN_EDGE_ENERGY: f64 = 1.0;
const VISUAL_MIN_COLOR_BINS: usize = 48;
const VISUAL_MIN_REGION_GRAY_STDDEV: f64 = 8.0;
const VISUAL_MIN_REGION_COLOR_BINS: usize = 16;
const VISUAL_MIN_REGION_CHANNEL_STDDEV: f64 = 0.5;
const VISUAL_REGION_CHANNEL_FLAT_MAX_COLOR_BINS: usize = 32;

fn parse_manifest_json(s: &str) -> Value {
    if s.is_empty() {
        return Value::Null;
    }
    serde_json::from_str::<Value>(s)
        .ok()
        .filter(|v| v.is_object())
        .unwrap_or(Value::Null)
}

fn manifest_pick(obj: &Value, key: &str) -> Value {
    obj.get(key).cloned().unwrap_or(Value::Null)
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

struct RegionStats {
    name: &'static str,
    gray_stddev: f64,
    edge_energy: f64,
    color_bins: usize,
    rgb_stddev: [f64; 3],
}

fn region_stats(
    img: &image::RgbImage,
    name: &'static str,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> RegionStats {
    let mut sums = [0.0_f64; 3];
    let mut sums2 = [0.0_f64; 3];
    let mut gray_sum = 0.0_f64;
    let mut gray_sum2 = 0.0_f64;
    let mut edge_sum = 0.0_f64;
    let mut edge_count = 0.0_f64;
    let mut color_bins = BTreeSet::<(u8, u8, u8)>::new();
    for y in y0..y1 {
        for x in x0..x1 {
            let p = img.get_pixel(x, y).0;
            let r = p[0] as f64;
            let g = p[1] as f64;
            let b = p[2] as f64;
            sums[0] += r;
            sums[1] += g;
            sums[2] += b;
            sums2[0] += r * r;
            sums2[1] += g * g;
            sums2[2] += b * b;
            let gray = (r + g + b) / 3.0;
            gray_sum += gray;
            gray_sum2 += gray * gray;
            color_bins.insert((p[0] >> 4, p[1] >> 4, p[2] >> 4));

            if x > x0 {
                let q = img.get_pixel(x - 1, y).0;
                edge_sum += (i16::from(p[0]) - i16::from(q[0])).abs() as f64;
                edge_sum += (i16::from(p[1]) - i16::from(q[1])).abs() as f64;
                edge_sum += (i16::from(p[2]) - i16::from(q[2])).abs() as f64;
                edge_count += 3.0;
            }
            if y > y0 {
                let q = img.get_pixel(x, y - 1).0;
                edge_sum += (i16::from(p[0]) - i16::from(q[0])).abs() as f64;
                edge_sum += (i16::from(p[1]) - i16::from(q[1])).abs() as f64;
                edge_sum += (i16::from(p[2]) - i16::from(q[2])).abs() as f64;
                edge_count += 3.0;
            }
        }
    }
    let n = ((x1 - x0) as f64) * ((y1 - y0) as f64);
    let mean = if n > 0.0 {
        [sums[0] / n, sums[1] / n, sums[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    };
    let rgb_stddev = if n > 0.0 {
        [
            (sums2[0] / n - mean[0] * mean[0]).max(0.0).sqrt(),
            (sums2[1] / n - mean[1] * mean[1]).max(0.0).sqrt(),
            (sums2[2] / n - mean[2] * mean[2]).max(0.0).sqrt(),
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    let gray_mean = if n > 0.0 { gray_sum / n } else { 0.0 };
    let gray_stddev = if n > 0.0 {
        (gray_sum2 / n - gray_mean * gray_mean).max(0.0).sqrt()
    } else {
        0.0
    };
    let edge_energy = if edge_count > 0.0 {
        edge_sum / edge_count
    } else {
        0.0
    };
    RegionStats {
        name,
        gray_stddev,
        edge_energy,
        color_bins: color_bins.len(),
        rgb_stddev,
    }
}

pub(crate) fn visual_health_for_output(output_path: &str, expected: Option<(u32, u32)>) -> Value {
    if output_path.is_empty() {
        return json!({
            "schema": "serenity.visual_health.v1",
            "status": "error",
            "error": "empty output_path",
            "failures": ["missing_output_path"],
        });
    }
    let img = match image::open(output_path) {
        Ok(img) => img.to_rgb8(),
        Err(error) => {
            return json!({
                "schema": "serenity.visual_health.v1",
                "status": "error",
                "path": output_path,
                "error": error.to_string(),
                "failures": ["image_decode_failed"],
            });
        }
    };
    let (width, height) = img.dimensions();
    let mut sums = [0.0_f64; 3];
    let mut sums2 = [0.0_f64; 3];
    let mut min_lum = 255.0_f64;
    let mut max_lum = 0.0_f64;
    let mut edge_sum = 0.0_f64;
    let mut edge_count = 0.0_f64;
    let mut color_bins = BTreeSet::<(u8, u8, u8)>::new();

    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x, y).0;
            let r = p[0] as f64;
            let g = p[1] as f64;
            let b = p[2] as f64;
            sums[0] += r;
            sums[1] += g;
            sums[2] += b;
            sums2[0] += r * r;
            sums2[1] += g * g;
            sums2[2] += b * b;
            let lum = (r + g + b) / 3.0;
            min_lum = min_lum.min(lum);
            max_lum = max_lum.max(lum);
            color_bins.insert((p[0] >> 4, p[1] >> 4, p[2] >> 4));

            if x > 0 {
                let q = img.get_pixel(x - 1, y).0;
                edge_sum += (i16::from(p[0]) - i16::from(q[0])).abs() as f64;
                edge_sum += (i16::from(p[1]) - i16::from(q[1])).abs() as f64;
                edge_sum += (i16::from(p[2]) - i16::from(q[2])).abs() as f64;
                edge_count += 3.0;
            }
            if y > 0 {
                let q = img.get_pixel(x, y - 1).0;
                edge_sum += (i16::from(p[0]) - i16::from(q[0])).abs() as f64;
                edge_sum += (i16::from(p[1]) - i16::from(q[1])).abs() as f64;
                edge_sum += (i16::from(p[2]) - i16::from(q[2])).abs() as f64;
                edge_count += 3.0;
            }
        }
    }

    let n = (width as f64) * (height as f64);
    let mean = if n > 0.0 {
        [sums[0] / n, sums[1] / n, sums[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    };
    let stddev = if n > 0.0 {
        [
            (sums2[0] / n - mean[0] * mean[0]).max(0.0).sqrt(),
            (sums2[1] / n - mean[1] * mean[1]).max(0.0).sqrt(),
            (sums2[2] / n - mean[2] * mean[2]).max(0.0).sqrt(),
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    let avg_stddev = (stddev[0] + stddev[1] + stddev[2]) / 3.0;
    let luminance_range = max_lum - min_lum;
    let edge_energy = if edge_count > 0.0 {
        edge_sum / edge_count
    } else {
        0.0
    };
    let region_defs = [
        ("top", 0, 0, width, height / 2),
        ("bottom", 0, height / 2, width, height),
        ("left", 0, 0, width / 2, height),
        ("right", width / 2, 0, width, height),
        ("top_left", 0, 0, width / 2, height / 2),
        ("top_right", width / 2, 0, width, height / 2),
        ("bottom_left", 0, height / 2, width / 2, height),
        ("bottom_right", width / 2, height / 2, width, height),
    ];
    let regions = region_defs
        .into_iter()
        .map(|(name, x0, y0, x1, y1)| region_stats(&img, name, x0, y0, x1, y1))
        .collect::<Vec<_>>();

    let mut failures = Vec::<String>::new();
    if let Some((expected_width, expected_height)) = expected {
        if width != expected_width || height != expected_height {
            failures.push(format!(
                "wrong_dimensions:{width}x{height}:expected:{expected_width}x{expected_height}"
            ));
        }
    }
    if avg_stddev < VISUAL_MIN_AVG_STDDEV {
        failures.push(format!("low_rgb_stddev:{:.3}", avg_stddev));
    }
    if luminance_range < VISUAL_MIN_LUMINANCE_RANGE {
        failures.push(format!("low_luminance_range:{:.3}", luminance_range));
    }
    if edge_energy < VISUAL_MIN_EDGE_ENERGY {
        failures.push(format!("low_edge_energy:{:.3}", edge_energy));
    }
    if color_bins.len() < VISUAL_MIN_COLOR_BINS {
        failures.push(format!("low_color_bins:{}", color_bins.len()));
    }
    for region in &regions {
        let min_channel = region
            .rgb_stddev
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if region.gray_stddev < VISUAL_MIN_REGION_GRAY_STDDEV {
            failures.push(format!(
                "low_region_gray_stddev:{}:{:.3}",
                region.name, region.gray_stddev
            ));
        }
        if region.color_bins < VISUAL_MIN_REGION_COLOR_BINS {
            failures.push(format!(
                "low_region_color_bins:{}:{}",
                region.name, region.color_bins
            ));
        }
        if min_channel < VISUAL_MIN_REGION_CHANNEL_STDDEV
            && region.color_bins <= VISUAL_REGION_CHANNEL_FLAT_MAX_COLOR_BINS
        {
            failures.push(format!(
                "flat_region_channel:{}:{:.3}:color_bins:{}",
                region.name, min_channel, region.color_bins
            ));
        }
    }
    let regions_json = regions
        .iter()
        .map(|region| {
            json!({
                "name": region.name,
                "gray_stddev": round3(region.gray_stddev),
                "edge_energy": round3(region.edge_energy),
                "color_bins": region.color_bins,
                "rgb_stddev": [
                    round3(region.rgb_stddev[0]),
                    round3(region.rgb_stddev[1]),
                    round3(region.rgb_stddev[2]),
                ],
            })
        })
        .collect::<Vec<_>>();

    json!({
        "schema": "serenity.visual_health.v1",
        "status": if failures.is_empty() { "pass" } else { "fail" },
        "path": output_path,
        "width": width,
        "height": height,
        "expected_width": expected.map(|(w, _)| w),
        "expected_height": expected.map(|(_, h)| h),
        "avg_stddev": round3(avg_stddev),
        "rgb_stddev": [round3(stddev[0]), round3(stddev[1]), round3(stddev[2])],
        "luminance_range": round3(luminance_range),
        "edge_energy": round3(edge_energy),
        "color_bins": color_bins.len(),
        "regions": regions_json,
        "thresholds": {
            "min_avg_stddev": VISUAL_MIN_AVG_STDDEV,
            "min_luminance_range": VISUAL_MIN_LUMINANCE_RANGE,
            "min_edge_energy": VISUAL_MIN_EDGE_ENERGY,
            "min_color_bins": VISUAL_MIN_COLOR_BINS,
            "min_region_gray_stddev": VISUAL_MIN_REGION_GRAY_STDDEV,
            "min_region_color_bins": VISUAL_MIN_REGION_COLOR_BINS,
            "min_region_channel_stddev": VISUAL_MIN_REGION_CHANNEL_STDDEV,
            "region_channel_flat_max_color_bins": VISUAL_REGION_CHANNEL_FLAT_MAX_COLOR_BINS,
        },
        "failures": failures,
        "note": "Heuristic guard against blank, flat, posterized, half-frame/channel-flat, or placeholder-like PNG outputs; not a sampler-parity or aesthetic-quality score.",
    })
}

pub(crate) fn server_result_manifest_path(output_path: &str) -> String {
    format!("{output_path}{SERVER_RESULT_SUFFIX}")
}

pub(crate) fn find_worker_result_manifest(output_path: &str) -> Option<String> {
    let output = Path::new(output_path);
    let file_name = output.file_name()?.to_str()?;
    let dir = output.parent().unwrap_or_else(|| Path::new("."));
    let prefix = format!("{file_name}.");
    let mut matches = Vec::new();
    for entry in fs::read_dir(dir).ok()? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with(&prefix) && name.ends_with("_daemon_result.json") {
            matches.push(path.to_string_lossy().into_owned());
        }
    }
    matches.sort();
    matches.into_iter().next()
}

pub(crate) fn manifest_refs_for_output(output_path: &str) -> Value {
    if output_path.is_empty() {
        return json!({
            "server_result_manifest": { "present": false, "path": null },
            "worker_result_manifest": { "present": false, "path": null },
        });
    }
    let server_path = server_result_manifest_path(output_path);
    let server_present = Path::new(&server_path).is_file();
    let server_result_manifest = if server_present {
        json!({ "present": true, "path": server_path })
    } else {
        json!({ "present": false, "path": null })
    };
    let worker_result_manifest = match find_worker_result_manifest(output_path) {
        Some(path) => json!({ "present": true, "path": path }),
        None => json!({ "present": false, "path": null }),
    };
    json!({
        "server_result_manifest": server_result_manifest,
        "worker_result_manifest": worker_result_manifest,
    })
}

fn read_json_file(path: &str) -> Result<Value, String> {
    let body = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str::<Value>(&body).map_err(|e| e.to_string())
}

fn expected_dims_from_value(v: &Value) -> Option<(u32, u32)> {
    let request = v.get("request")?;
    let width = request.get("width")?.as_u64()?;
    let height = request.get("height")?.as_u64()?;
    Some((u32::try_from(width).ok()?, u32::try_from(height).ok()?))
}

fn expected_dims_from_params(params: &JobParams) -> Option<(u32, u32)> {
    if params.width <= 0 || params.height <= 0 {
        return None;
    }
    Some((
        u32::try_from(params.width).ok()?,
        u32::try_from(params.height).ok()?,
    ))
}

pub(crate) fn visual_health_for_params_json(output_path: &str, params_json: &str) -> Value {
    let params = parse_manifest_json(params_json);
    let expected = params
        .get("width")
        .and_then(|v| v.as_u64())
        .zip(params.get("height").and_then(|v| v.as_u64()))
        .and_then(|(width, height)| {
            Some((u32::try_from(width).ok()?, u32::try_from(height).ok()?))
        });
    visual_health_for_output(output_path, expected)
}

fn manifest_path_from_refs<'a>(refs: &'a Value, key: &str) -> Option<&'a str> {
    refs.get(key)
        .and_then(|v| v.get("path"))
        .and_then(|v| v.as_str())
}

pub(crate) fn output_location_for_root(output_path: &str, output_root: Option<&Path>) -> Value {
    let Some(root) = output_root else {
        return json!(null);
    };
    let output = Path::new(output_path);
    let relative_path = output
        .strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().into_owned());
    json!({
        "root_kind": "ui_workflow_gallery",
        "root": root.to_string_lossy(),
        "inside_root": relative_path.is_some(),
        "relative_path": relative_path,
    })
}

pub(crate) fn result_document_for_output(job_id: &str, output_path: &str) -> Value {
    result_document_for_output_with_root(job_id, output_path, None)
}

pub(crate) fn result_document_for_output_with_root(
    job_id: &str,
    output_path: &str,
    output_root: Option<&Path>,
) -> Value {
    let refs = manifest_refs_for_output(output_path);
    let output_location = output_location_for_root(output_path, output_root);
    let mut doc = json!({
        "schema": "serenity.job_result.v1",
        "job_id": job_id,
        "output_path": output_path,
        "output_location": output_location,
        "result_manifests": refs.clone(),
        "server_result": null,
        "worker_result": null,
        "visual_health": null,
    });
    if let Some(path) = manifest_path_from_refs(&refs, "server_result_manifest") {
        match read_json_file(path) {
            Ok(mut v) => {
                let visual_health =
                    visual_health_for_output(output_path, expected_dims_from_value(&v));
                if v.get("visual_health").is_none() {
                    v["visual_health"] = visual_health.clone();
                }
                doc["visual_health"] = visual_health;
                doc["server_result"] = v;
            }
            Err(error) => doc["server_result_error"] = json!(error),
        }
    }
    if let Some(path) = manifest_path_from_refs(&refs, "worker_result_manifest") {
        match read_json_file(path) {
            Ok(v) => doc["worker_result"] = v,
            Err(error) => doc["worker_result_error"] = json!(error),
        }
    }
    if doc["visual_health"].is_null() {
        doc["visual_health"] = visual_health_for_output(output_path, None);
    }
    doc
}

fn json_response(status: StatusCode, doc: &Value) -> Response {
    (
        status,
        [(CONTENT_TYPE, "application/json")],
        serde_json::to_string(doc).unwrap_or_else(|_| String::from("{}")),
    )
        .into_response()
}

fn safe_job_id(id: &str) -> bool {
    !id.is_empty() && id.starts_with("job-") && !id.contains('/') && !id.contains('\\')
}

fn output_path_for_job(st: &AppState, id: &str) -> Option<String> {
    if let Ok(jobs) = st.jobs.lock() {
        if let Some(e) = jobs.iter().find(|e| e.record.id == id) {
            if !e.record.output_path.is_empty() {
                return Some(e.record.output_path.clone());
            }
        }
    }
    if let Some(row) = st.prior.iter().find(|r| r[0] == id) {
        if !row[5].is_empty() {
            return Some(row[5].clone());
        }
    }
    let fallback = st.out_dir.join(format!("{id}.png"));
    if fallback.is_file() {
        return Some(fallback.to_string_lossy().into_owned());
    }
    None
}

pub(crate) async fn get_job_result(
    State(st): State<AppState>,
    AxPath(id): AxPath<String>,
) -> Response {
    if !safe_job_id(&id) {
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({ "detail": "invalid job id" }),
        );
    }
    let Some(output_path) = output_path_for_job(&st, &id) else {
        return json_response(
            StatusCode::NOT_FOUND,
            &json!({ "detail": format!("no result for job: {id}") }),
        );
    };
    json_response(
        StatusCode::OK,
        &result_document_for_output_with_root(&id, &output_path, Some(st.out_dir.as_path())),
    )
}

pub(crate) fn write_server_result_manifest(
    record: &jobs::JobRecord,
    params: &JobParams,
) -> Result<String, String> {
    if record.output_path.is_empty() {
        return Err("done record has no output_path".to_string());
    }

    let embedded = parse_manifest_json(&record.params_json);
    let submitted = parse_manifest_json(&params.params_json);
    let source = if embedded.is_object() {
        &embedded
    } else {
        &submitted
    };
    let dimensions = image::image_dimensions(&record.output_path).ok();
    let file_size_bytes = fs::metadata(&record.output_path).map(|m| m.len()).ok();
    let output_root_path = if params.out_dir.trim().is_empty() {
        None
    } else {
        Some(Path::new(params.out_dir.as_str()))
    };
    let output_location = output_location_for_root(&record.output_path, output_root_path);
    let visual_health =
        visual_health_for_output(&record.output_path, expected_dims_from_params(params));
    let worker_manifest = find_worker_result_manifest(&record.output_path);
    let mut limitations = Vec::<String>::new();
    if worker_manifest.is_none() {
        limitations.push("worker_specific_result_manifest_missing".to_string());
    }
    if visual_health.get("status").and_then(|v| v.as_str()) != Some("pass") {
        limitations.push("visual_health_not_passed".to_string());
    }
    let worker_result_manifest = match worker_manifest {
        Some(path) => json!({ "present": true, "path": path }),
        None => json!({ "present": false, "path": null }),
    };
    let output = match dimensions {
        Some((w, h)) => json!({
            "path": &record.output_path,
            "location": output_location,
            "exists": true,
            "file_size_bytes": file_size_bytes,
            "width": w,
            "height": h,
        }),
        None => json!({
            "path": &record.output_path,
            "location": output_location,
            "exists": Path::new(&record.output_path).exists(),
            "file_size_bytes": file_size_bytes,
            "width": null,
            "height": null,
        }),
    };
    let manifest = json!({
        "schema": "serenity.server_result.v1",
        "job_id": &record.id,
        "created": &record.created,
        "completed_at": httpdate::fmt_http_date(SystemTime::now()),
        "state": &record.state,
        "model": &record.model,
        "output": output,
        "request": {
            "model": &params.model,
            "prompt": &params.prompt,
            "negative": &params.negative,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "seed": params.seed,
            "cfg": params.cfg,
            "sampler": &params.sampler,
            "scheduler": &params.scheduler,
            "images": params.images,
            "image_index": params.image_index,
            "image_count": params.image_count,
            "init_image": &params.init_image,
            "mask_image": &params.mask_image,
            "vae": &params.vae,
            "workflow_save_prefix": &params.workflow_save_prefix,
        },
        "workflow": {
            "client": manifest_pick(source, "workflow_client"),
            "schema": manifest_pick(source, "workflow_schema"),
            "executor": manifest_pick(source, "workflow_executor"),
            "source": manifest_pick(source, "workflow_source"),
            "route_kind": manifest_pick(source, "workflow_route_kind"),
            "node_count": manifest_pick(source, "workflow_node_count"),
            "edge_count": manifest_pick(source, "workflow_edge_count"),
            "plan": manifest_pick(source, "workflow_plan"),
        },
        "evidence": {
            "embedded_genparams_present": !record.params_json.is_empty(),
            "submitted_params_json_present": !params.params_json.is_empty(),
            "worker_result_manifest": worker_result_manifest,
        },
        "visual_health": visual_health,
        "readiness": {
            "label": "server_completion_evidence",
            "limits": limitations,
            "note": "Confirms workflow dispatch, artifact completion, and a heuristic visual-health check; sampler parity and aesthetic quality require separate gates."
        },
    });
    let manifest_path = server_result_manifest_path(&record.output_path);
    let body = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(&manifest_path, format!("{body}\n")).map_err(|e| e.to_string())?;
    Ok(manifest_path)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::{json, Value};

    use super::*;

    fn temp_manifest_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "serenity_server_manifest_{label}_{}_{}",
            std::process::id(),
            nonce
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn manifest_test_record(output_path: &Path, params_json: &str) -> (jobs::JobRecord, JobParams) {
        let mut params = JobParams::default();
        params.job_id = "job-0099".to_string();
        params.model = "flux-2-klein-base-9b_fp8_e4m3fn".to_string();
        params.prompt = "workflow manifest test".to_string();
        params.width = 512;
        params.height = 512;
        params.steps = 4;
        params.seed = 20260616;
        params.cfg = 1.0;
        params.sampler = "euler".to_string();
        params.scheduler = "simple".to_string();
        params.out_dir = output_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_string_lossy()
            .into_owned();
        params.params_json = params_json.to_string();
        let record = jobs::JobRecord {
            id: "job-0099".to_string(),
            created: "Tue, 16 Jun 2026 19:00:00 GMT".to_string(),
            model: params.model.clone(),
            state: "done".to_string(),
            progress: 100,
            step: 4,
            total: 4,
            image_index: 0,
            image_count: 1,
            output_path: output_path.to_string_lossy().into_owned(),
            error: String::new(),
            params_json: params_json.to_string(),
        };
        (record, params)
    }

    #[test]
    fn server_result_manifest_records_workflow_done_without_worker_sidecar() {
        let dir = temp_manifest_dir("no_worker_sidecar");
        let output_path = dir.join("job-0099.png");
        image::RgbImage::from_pixel(2, 3, image::Rgb([7, 8, 9]))
            .save(&output_path)
            .unwrap();
        let params_json = serde_json::to_string(&json!({
            "schema": "serenity.genparams.v1",
            "job_id": "job-0099",
            "workflow_client": "serenity.canvas.generate_ws",
            "workflow_schema": "serenity.workflow_graph.v1",
            "workflow_executor": "serenity.workflow_graph.executor.v1",
            "workflow_source": "comfy_api_prompt_graph",
            "workflow_node_count": 7,
            "workflow_edge_count": 9,
            "workflow_route_kind": "image",
            "workflow_plan": {
                "schema": "serenity.workflow_plan.v1",
                "route_kind": "image",
                "terminal_nodes": [{"node_id": 7, "type": "SaveImage", "kind": "image"}]
            },
            "model": "flux-2-klein-base-9b_fp8_e4m3fn",
            "width": 512,
            "height": 512,
            "steps": 4
        }))
        .unwrap();
        let (record, params) = manifest_test_record(&output_path, &params_json);

        let manifest_path = write_server_result_manifest(&record, &params).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();

        assert_eq!(v["schema"], "serenity.server_result.v1");
        assert_eq!(v["job_id"], "job-0099");
        assert_eq!(v["output"]["width"].as_u64(), Some(2));
        assert_eq!(v["output"]["height"].as_u64(), Some(3));
        assert_eq!(v["output"]["location"]["root_kind"], "ui_workflow_gallery");
        assert_eq!(v["output"]["location"]["inside_root"], true);
        assert_eq!(v["output"]["location"]["relative_path"], "job-0099.png");
        assert_eq!(v["workflow"]["client"], "serenity.canvas.generate_ws");
        assert_eq!(v["workflow"]["schema"], "serenity.workflow_graph.v1");
        assert_eq!(v["workflow"]["route_kind"], "image");
        assert_eq!(v["visual_health"]["status"], "fail");
        assert_eq!(
            v["evidence"]["worker_result_manifest"]["present"].as_bool(),
            Some(false)
        );
        assert!(v["readiness"]["limits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item.as_str() == Some("worker_specific_result_manifest_missing")));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn visual_health_flags_flat_placeholder_and_accepts_varied_png() {
        let dir = temp_manifest_dir("visual_health");
        let flat_path = dir.join("flat.png");
        image::RgbImage::from_pixel(64, 64, image::Rgb([8, 8, 8]))
            .save(&flat_path)
            .unwrap();
        let flat = visual_health_for_output(&flat_path.to_string_lossy(), Some((64, 64)));
        assert_eq!(flat["status"], "fail");
        assert!(flat["failures"].as_array().unwrap().iter().any(|item| item
            .as_str()
            .unwrap_or_default()
            .starts_with("low_rgb_stddev")));

        let varied_path = dir.join("varied.png");
        let mut varied = image::RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                varied.put_pixel(
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
        varied.save(&varied_path).unwrap();
        let ok = visual_health_for_output(&varied_path.to_string_lossy(), Some((64, 64)));
        assert_eq!(ok["status"], "pass", "{ok:#}");
        let wrong_size = visual_health_for_output(&varied_path.to_string_lossy(), Some((512, 512)));
        assert_eq!(wrong_size["status"], "fail");
        assert!(wrong_size["failures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .unwrap_or_default()
                .starts_with("wrong_dimensions")));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn visual_health_flags_half_frame_channel_flat_corruption() {
        let dir = temp_manifest_dir("visual_health_half_flat");
        let path = dir.join("half_flat.png");
        let mut img = image::RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let pixel = if y < 32 {
                    image::Rgb([
                        ((x * 5 + y * 3) % 256) as u8,
                        ((x * 7 + y * 11) % 256) as u8,
                        ((x * 13 + y * 17) % 256) as u8,
                    ])
                } else {
                    image::Rgb([(96 + (x % 4) * 8) as u8, (80 + (y % 4) * 8) as u8, 12])
                };
                img.put_pixel(x, y, pixel);
            }
        }
        img.save(&path).unwrap();
        let health = visual_health_for_output(&path.to_string_lossy(), Some((64, 64)));
        assert_eq!(health["status"], "fail", "{health:#}");
        let failures = health["failures"].as_array().unwrap();
        assert!(failures.iter().any(|item| item
            .as_str()
            .unwrap_or_default()
            .starts_with("flat_region_channel:bottom")));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn worker_result_manifest_detection_uses_backend_sidecar_only() {
        let dir = temp_manifest_dir("worker_sidecar");
        let output_path = dir.join("job-0100.png");
        fs::write(&output_path, b"placeholder").unwrap();
        let server_path = dir.join("job-0100.png.serenity_server_result.json");
        let worker_path = dir.join("job-0100.png.klein_daemon_result.json");
        fs::write(&server_path, b"{}").unwrap();
        fs::write(&worker_path, b"{}").unwrap();

        assert_eq!(
            find_worker_result_manifest(&output_path.to_string_lossy()),
            Some(worker_path.to_string_lossy().into_owned())
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn manifest_refs_report_server_and_worker_presence() {
        let dir = temp_manifest_dir("refs");
        let output_path = dir.join("job-0101.png");
        fs::write(&output_path, b"placeholder").unwrap();
        let output = output_path.to_string_lossy().into_owned();

        let refs = manifest_refs_for_output(&output);
        assert_eq!(refs["server_result_manifest"]["present"], false);
        assert_eq!(refs["worker_result_manifest"]["present"], false);

        let server_path = server_result_manifest_path(&output);
        fs::write(&server_path, b"{}").unwrap();
        let worker_path = dir.join("job-0101.png.zimage_daemon_result.json");
        fs::write(&worker_path, b"{}").unwrap();

        let refs = manifest_refs_for_output(&output);
        assert_eq!(refs["server_result_manifest"]["present"], true);
        assert_eq!(refs["server_result_manifest"]["path"], server_path);
        assert_eq!(refs["worker_result_manifest"]["present"], true);
        assert_eq!(
            refs["worker_result_manifest"]["path"],
            worker_path.to_string_lossy().into_owned()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn result_document_embeds_server_and_worker_json() {
        let dir = temp_manifest_dir("doc");
        let output_path = dir.join("job-0102.png");
        fs::write(&output_path, b"placeholder").unwrap();
        let output = output_path.to_string_lossy().into_owned();
        let server_path = server_result_manifest_path(&output);
        let worker_path = dir.join("job-0102.png.zimage_daemon_result.json");
        fs::write(
            &server_path,
            r#"{"schema":"serenity.server_result.v1","ok":true}"#,
        )
        .unwrap();
        fs::write(
            &worker_path,
            r#"{"schema":"zimage.daemon.result.v1","ok":true}"#,
        )
        .unwrap();

        let doc = result_document_for_output_with_root("job-0102", &output, Some(dir.as_path()));
        assert_eq!(doc["schema"], "serenity.job_result.v1");
        assert_eq!(doc["job_id"], "job-0102");
        assert_eq!(doc["output_location"]["inside_root"], true);
        assert_eq!(doc["output_location"]["relative_path"], "job-0102.png");
        assert_eq!(doc["server_result"]["schema"], "serenity.server_result.v1");
        assert_eq!(doc["worker_result"]["schema"], "zimage.daemon.result.v1");
        assert_eq!(
            doc["result_manifests"]["server_result_manifest"]["present"],
            true
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn result_document_reports_parse_error_without_hiding_refs() {
        let dir = temp_manifest_dir("doc_parse_error");
        let output_path = dir.join("job-0103.png");
        fs::write(&output_path, b"placeholder").unwrap();
        let output = output_path.to_string_lossy().into_owned();
        let server_path = server_result_manifest_path(&output);
        fs::write(&server_path, b"not json").unwrap();

        let doc = result_document_for_output("job-0103", &output);
        assert_eq!(doc["server_result"].is_null(), true);
        assert!(doc["server_result_error"]
            .as_str()
            .unwrap()
            .contains("expected"));
        assert_eq!(
            doc["result_manifests"]["server_result_manifest"]["present"],
            true
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
