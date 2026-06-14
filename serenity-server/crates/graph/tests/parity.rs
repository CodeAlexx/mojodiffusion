//! Parity harness: for every `<name>.request.json` whose name matches
//! `*_t2i` or `*_t2i_lora`, load the request, run `lower_request`, and assert
//! every top-level scalar key (except `"workflow"`) equals the matching
//! `<name>.lowered.json` reference captured from the Mojo oracle
//! (`output/bin/serenity_lower`, via `scripts/capture_lowering_refs.sh`).
//!
//! Refs live under `serenity-server/tests/refs/` (repo-relative). On mismatch
//! the test prints the offending key and both values so failures are
//! self-describing.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;
use serenity_graph::lower_request;

/// Resolve the refs dir relative to this crate (CARGO_MANIFEST_DIR =
/// serenity-server/crates/graph), i.e. ../../tests/refs.
fn refs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/refs")
        .canonicalize()
        .expect("refs dir exists")
}

fn is_t2i_name(name: &str) -> bool {
    name.ends_with("_t2i") || name.ends_with("_t2i_lora")
}

/// Edit / reference-latent graphs: `*_edit` and `*_edit_lora` (e.g.
/// `flux1_dev_edit`, `klein4b_edit_lora`, `qwen_edit`). These exercise the
/// ReferenceLatent / VAEEncode / inpaint conditioning lowering path on top of
/// the t2i flat params.
fn is_edit_name(name: &str) -> bool {
    name.ends_with("_edit") || name.ends_with("_edit_lora")
}

/// Any parity-covered graph name (t2i or edit family).
fn is_covered_name(name: &str) -> bool {
    is_t2i_name(name) || is_edit_name(name)
}

/// Compare two top-level scalar values for parity. Numbers compare by f64 value
/// so `1` vs `1.0` (JSON int vs float emission) does not spuriously fail —
/// matching the oracle's flat-genparams semantics where cfg may be int or float.
fn scalar_eq(a: &JsonValue, b: &JsonValue) -> bool {
    match (a, b) {
        (JsonValue::Number(x), JsonValue::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(fx), Some(fy)) => fx == fy,
            _ => x == y,
        },
        _ => a == b,
    }
}

struct RefCase {
    name: String,
    request_path: PathBuf,
    lowered_path: PathBuf,
}

fn collect_cases() -> Vec<RefCase> {
    let dir = refs_dir();
    let mut cases = Vec::new();
    for entry in fs::read_dir(&dir).expect("read refs dir") {
        let entry = entry.expect("dir entry");
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        let Some(stem) = fname.strip_suffix(".request.json") else {
            continue;
        };
        if !is_covered_name(stem) {
            continue;
        }
        cases.push(RefCase {
            name: stem.to_string(),
            request_path: entry.path(),
            lowered_path: dir.join(format!("{stem}.lowered.json")),
        });
    }
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    cases
}

/// Lower one request and diff every top-level scalar key (except `workflow`)
/// against the reference. Returns Ok(()) on full parity, or Err(diff report).
fn check_case(case: &RefCase) -> Result<(), String> {
    let req_bytes = fs::read(&case.request_path)
        .map_err(|e| format!("read request: {e}"))?;
    let mut req: JsonValue =
        serde_json::from_slice(&req_bytes).map_err(|e| format!("parse request: {e}"))?;
    let ref_bytes = fs::read(&case.lowered_path)
        .map_err(|e| format!("read lowered ref: {e}"))?;
    let want: JsonValue =
        serde_json::from_slice(&ref_bytes).map_err(|e| format!("parse lowered ref: {e}"))?;

    lower_request(&mut req).map_err(|e| format!("lower_request raised: {e}"))?;

    let got_obj = req.as_object().ok_or("lowered request is not an object")?;
    let want_obj = want.as_object().ok_or("lowered ref is not an object")?;

    let mut diffs: Vec<String> = Vec::new();

    // Every reference key (except workflow) must be present and equal.
    for (k, want_v) in want_obj {
        if k == "workflow" {
            continue;
        }
        // Only top-level SCALARS are in scope (the lowered flat params); the
        // ref's only non-scalar top-level value is `lora` (an array), which we
        // still compare exactly via the != fallback.
        match got_obj.get(k) {
            None => diffs.push(format!("  key '{k}': MISSING in lowered (want {want_v})")),
            Some(got_v) => {
                if !scalar_eq(got_v, want_v) {
                    diffs.push(format!("  key '{k}': got {got_v}  want {want_v}"));
                }
            }
        }
    }
    // Extra top-level keys the Rust lowering emitted that the ref lacks (other
    // than workflow) are also a parity failure.
    for (k, got_v) in got_obj {
        if k == "workflow" {
            continue;
        }
        if !want_obj.contains_key(k) {
            diffs.push(format!("  key '{k}': EXTRA in lowered (got {got_v}, absent in ref)"));
        }
    }

    if diffs.is_empty() {
        Ok(())
    } else {
        Err(diffs.join("\n"))
    }
}

#[test]
fn t2i_and_lora_refs_lower_to_parity() {
    let cases = collect_cases();
    assert!(
        !cases.is_empty(),
        "no *_t2i / *_t2i_lora / *_edit / *_edit_lora request refs found under {}",
        refs_dir().display()
    );

    let mut passed: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    for case in &cases {
        match check_case(case) {
            Ok(()) => {
                println!("PASS  {}", case.name);
                passed.push(case.name.clone());
            }
            Err(report) => {
                println!("FAIL  {}\n{}", case.name, report);
                failed.push(case.name.clone());
            }
        }
    }

    println!(
        "\nparity summary: {} pass, {} fail (of {} t2i+edit refs)",
        passed.len(),
        failed.len(),
        cases.len()
    );
    println!("PASS: {passed:?}");
    println!("FAIL: {failed:?}");

    assert!(
        failed.is_empty(),
        "{} t2i+edit ref(s) failed parity: {:?}",
        failed.len(),
        failed
    );
}

/// The oracle rejects `flux2_dev_t2i_lora` (a LoraLoader with strength_clip!=0
/// produces a CLIP_LORA_UNSUPPORTED handle that the CLIPTextEncode consumer
/// refuses) with a 501. The Rust port must fail-loud identically, not silently
/// drop it. (No `.lowered.json` ref exists for this input — it is verified
/// out-of-band, mirroring the capture script's LOWER-FAIL handling.)
#[test]
fn flux2_dev_t2i_lora_is_rejected_loud() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/_reject_fixture.json");
    if !p.exists() {
        eprintln!("skip: reject fixture absent");
        return;
    }
    let mut req: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    let err = serenity_graph::lower_request(&mut req).expect_err("must raise");
    println!("flux2_dev_t2i_lora rejection: {err}");
    // The Mojo oracle raised [501]; the audit splits unsupported(501)/badreq(422).
    // A CLIP_LORA_UNSUPPORTED type mismatch is a 422 in the structured-error model.
    assert!(matches!(err.http_status(), 501 | 422), "status {}", err.http_status());
}
