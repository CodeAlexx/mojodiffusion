//! Parity + behavior tests for the Ideogram4-export and Comfy-UI-canvas
//! importers (`import::apply_ideogram4_comfy_ui_export`,
//! `import::comfy_ui_canvas_to_typed_body`, wired into `lower_request`).
//!
//! The positive parity refs were captured from the Mojo oracle
//! (`output/bin/serenity_lower`) on byte-identical input requests and live next
//! to the t2i corpus under `serenity-server/tests/refs/`:
//!   - `ideogram4__basic_txt2img_v3` / `_v4`: real Ideogram4 ComfyUI canvas
//!     exports (Downloads `ideogram4_basic_txt2img_workflow_by_AI_Characters_v{3,4}`),
//!     wrapped with a top-level seed (the export uses a randomized Seed (rgthree)
//!     widget; the oracle 501s without a top-level seed).
//!   - `comfy_canvas__min_t2i`: a minimal allowlisted Comfy-UI canvas t2i graph
//!     (the real-world canvas exports in Downloads all reference custom node
//!     types the executor allowlist rejects, so the oracle 501s on them; this
//!     hand-built fixture is the smallest graph that lowers cleanly through the
//!     oracle and so can be a positive parity ref).
//!
//! The mute (mode==2) / ComfySwitchNode-lazy behaviors are audit-parity FIXES
//! the Mojo source has not yet applied, so they are verified as Rust-side
//! behavior tests (the oracle would 501 on the muted graph; that divergence is
//! exactly what the fix corrects).

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value as JsonValue};
use serenity_graph::{comfy_ui_canvas_to_typed_body, lower_request};

fn refs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/refs")
        .canonicalize()
        .expect("refs dir exists")
}

/// Numbers compare by f64 (1 vs 1.0 must not spuriously fail), everything else
/// by exact equality — same rule as the t2i parity harness.
fn scalar_eq(a: &JsonValue, b: &JsonValue) -> bool {
    match (a, b) {
        (JsonValue::Number(x), JsonValue::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(fx), Some(fy)) => fx == fy,
            _ => x == y,
        },
        _ => a == b,
    }
}

/// Lower `<name>.request.json` and diff every non-`workflow` top-level key
/// against `<name>.lowered.json`.
fn check_ref(name: &str) {
    let dir = refs_dir();
    let req_path = dir.join(format!("{name}.request.json"));
    let low_path = dir.join(format!("{name}.lowered.json"));

    let mut req: JsonValue =
        serde_json::from_slice(&fs::read(&req_path).expect("read request")).expect("parse request");
    let want: JsonValue =
        serde_json::from_slice(&fs::read(&low_path).expect("read lowered")).expect("parse lowered");

    lower_request(&mut req).unwrap_or_else(|e| panic!("{name}: lower_request raised: {e}"));

    let got = req.as_object().expect("lowered is object");
    let want = want.as_object().expect("ref is object");

    let mut diffs = Vec::new();
    for (k, want_v) in want {
        if k == "workflow" {
            continue;
        }
        match got.get(k) {
            None => diffs.push(format!("  key '{k}': MISSING (want {want_v})")),
            Some(got_v) if !scalar_eq(got_v, want_v) => {
                diffs.push(format!("  key '{k}': got {got_v}  want {want_v}"))
            }
            _ => {}
        }
    }
    for (k, got_v) in got {
        if k == "workflow" {
            continue;
        }
        if !want.contains_key(k) {
            diffs.push(format!("  key '{k}': EXTRA (got {got_v}, absent in ref)"));
        }
    }
    assert!(diffs.is_empty(), "{name} parity failed:\n{}", diffs.join("\n"));
}

#[test]
fn ideogram4_export_v3_parity() {
    check_ref("ideogram4__basic_txt2img_v3");
}

#[test]
fn ideogram4_export_v4_parity() {
    check_ref("ideogram4__basic_txt2img_v4");
}

#[test]
fn comfy_canvas_min_t2i_parity() {
    check_ref("comfy_canvas__min_t2i");
}

/// The Ideogram4 export with a randomized Seed (rgthree) widget and NO top-level
/// seed must 501-class fail-loud (Mojo `_workflow_set_seed_from_widget_if_missing`).
#[test]
fn ideogram4_export_randomized_seed_is_rejected_loud() {
    let dir = refs_dir();
    let mut req: JsonValue = serde_json::from_slice(
        &fs::read(dir.join("ideogram4__basic_txt2img_v3.request.json")).expect("read request"),
    )
    .expect("parse");
    // Drop the top-level seed -> the importer must demand it.
    req.as_object_mut().unwrap().remove("seed");
    let err = lower_request(&mut req).expect_err("must raise without a top-level seed");
    assert_eq!(err.http_status(), 501, "got {}", err);
    assert!(
        format!("{err}").contains("seed"),
        "unexpected message: {err}"
    );
}

/// The Ideogram4 export with NO top-level prompt override must 501 (the
/// prompt-builder subgraph needs external Gemma/KJ execution).
#[test]
fn ideogram4_export_missing_prompt_is_rejected_loud() {
    let dir = refs_dir();
    let mut req: JsonValue = serde_json::from_slice(
        &fs::read(dir.join("ideogram4__basic_txt2img_v3.request.json")).expect("read request"),
    )
    .expect("parse");
    let o = req.as_object_mut().unwrap();
    o.remove("prompt");
    o.remove("prompt_raw");
    let err = lower_request(&mut req).expect_err("must raise without a prompt override");
    assert_eq!(err.http_status(), 501, "got {}", err);
}

// ---------------------------------------------------------------------------
// Audit-parity behavior: mode==2 (mute) is inactive, not just mode==4 (bypass).
// These verify Rust-side behavior the Mojo oracle does NOT yet implement (the
// oracle would 501 on the muted-node graph because it emits the dead node live).
// ---------------------------------------------------------------------------

/// Build the minimal canvas graph; `extra_nodes` / `extra_links` are appended.
fn canvas_t2i(extra_nodes: Vec<JsonValue>, extra_links: Vec<JsonValue>) -> JsonValue {
    let mut nodes = vec![
        json!({"id":1,"type":"CheckpointLoaderSimple","mode":0,"inputs":[],
            "outputs":[{"name":"MODEL","type":"MODEL","links":[10]},
                       {"name":"CLIP","type":"CLIP","links":[11,12]},
                       {"name":"VAE","type":"VAE","links":[13]}],
            "widgets_values":["model.safetensors"]}),
        json!({"id":2,"type":"CLIPTextEncode","mode":0,
            "inputs":[{"name":"clip","type":"CLIP","link":11}],
            "outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[14]}],
            "widgets_values":["a scenic mountain"]}),
        json!({"id":3,"type":"CLIPTextEncode","mode":0,
            "inputs":[{"name":"clip","type":"CLIP","link":12}],
            "outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[15]}],
            "widgets_values":["blurry"]}),
        json!({"id":4,"type":"EmptyLatentImage","mode":0,"inputs":[],
            "outputs":[{"name":"LATENT","type":"LATENT","links":[16]}],
            "widgets_values":[768,512,2]}),
        json!({"id":5,"type":"KSampler","mode":0,
            "inputs":[{"name":"model","type":"MODEL","link":10},
                      {"name":"positive","type":"CONDITIONING","link":14},
                      {"name":"negative","type":"CONDITIONING","link":15},
                      {"name":"latent_image","type":"LATENT","link":16}],
            "outputs":[{"name":"LATENT","type":"LATENT","links":[17]}],
            "widgets_values":[123,"randomize",25,6.5,"dpmpp_2m","karras",1.0]}),
        json!({"id":6,"type":"VAEDecode","mode":0,
            "inputs":[{"name":"samples","type":"LATENT","link":17},
                      {"name":"vae","type":"VAE","link":13}],
            "outputs":[{"name":"IMAGE","type":"IMAGE","links":[18]}],
            "widgets_values":[]}),
        json!({"id":7,"type":"SaveImage","mode":0,
            "inputs":[{"name":"images","type":"IMAGE","link":18}],
            "outputs":[],"widgets_values":["MyPrefix"]}),
    ];
    nodes.extend(extra_nodes);
    let mut links = vec![
        json!([10,1,0,5,0,"MODEL"]),
        json!([11,1,1,2,0,"CLIP"]),
        json!([12,1,1,3,0,"CLIP"]),
        json!([13,1,2,6,1,"VAE"]),
        json!([14,2,0,5,1,"CONDITIONING"]),
        json!([15,3,0,5,2,"CONDITIONING"]),
        json!([16,4,0,5,3,"LATENT"]),
        json!([17,5,0,6,0,"LATENT"]),
        json!([18,6,0,7,0,"IMAGE"]),
    ];
    links.extend(extra_links);
    json!({"nodes": nodes, "links": links, "version": 0.4})
}

/// A muted (mode==2) node and the links touching it are dropped from the typed
/// body, exactly like a bypassed (mode==4) node — the audit-parity fix.
#[test]
fn canvas_mode2_mute_node_is_dropped() {
    // Node 8 is a muted CLIPTextEncode; link 19 connects clip(1)->8, link 20
    // connects 8->(nowhere live). Both must be dropped along with the node.
    let muted = json!({"id":8,"type":"CLIPTextEncode","mode":2,
        "inputs":[{"name":"clip","type":"CLIP","link":19}],
        "outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[]}],
        "widgets_values":["MUTED dead prompt"]});
    let extra_link = json!([19,1,1,8,0,"CLIP"]);
    let wf = canvas_t2i(vec![muted], vec![extra_link]);

    let typed = comfy_ui_canvas_to_typed_body(&wf).expect("typed body builds");
    let nodes = typed["nodes"].as_array().unwrap();
    let edges = typed["edges"].as_array().unwrap();

    // 7 live nodes (node 8 dropped), 9 edges (the link touching node 8 dropped).
    assert_eq!(nodes.len(), 7, "muted node must be dropped: {nodes:?}");
    assert!(
        nodes.iter().all(|n| n["id"].as_i64() != Some(8)),
        "node 8 must not appear"
    );
    assert_eq!(edges.len(), 9, "link touching muted node must be dropped");
    assert!(
        edges.iter().all(|e| e["from"]["node"].as_i64() != Some(8)
            && e["to"]["node"].as_i64() != Some(8)),
        "no edge may reference the muted node"
    );

    // And the whole request still lowers cleanly (the unfixed oracle would 501).
    let mut req = json!({
        "model":"oracle-placeholder-model","prompt":"p","width":1024,"height":1024,
        "seed":42,"workflow": wf
    });
    lower_request(&mut req).expect("muted-node graph lowers with the mute fix");
    assert_eq!(
        req["workflow_source"].as_str(),
        Some("comfy_ui_canvas_graph")
    );
}

/// ComfySwitchNode lazy single-branch (audit-parity fix): with `switch`=true,
/// only the `on_true` branch must be wired; the `on_false` branch may be absent
/// and the graph still lowers. The pre-fix executor required BOTH branches and
/// would 501 here.
#[test]
fn comfy_switch_lazy_only_selected_branch_required() {
    // A typed {nodes, edges} body: model+clip from a checkpoint, a positive and
    // a negative CLIPTextEncode, an EmptyLatentImage, and a ComfySwitchNode that
    // (switch=true) selects the negative CONDITIONING into the KSampler's
    // `negative` input. The `on_false` branch is intentionally NOT wired.
    let body = json!({
        "nodes": [
            {"id":1,"type_id":"CheckpointLoaderSimple","fields":{"ckpt_name":"m.safetensors"}},
            {"id":2,"type_id":"CLIPTextEncode","fields":{"text":"a castle"}},
            {"id":3,"type_id":"CLIPTextEncode","fields":{"text":"watermark"}},
            {"id":4,"type_id":"EmptyLatentImage","fields":{"width":640,"height":480,"batch_size":1}},
            {"id":5,"type_id":"ComfySwitchNode","fields":{"switch":true}},
            {"id":6,"type_id":"KSampler","fields":{
                "seed":7,"steps":12,"cfg":5.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
            {"id":7,"type_id":"VAEDecode","fields":{}},
            {"id":8,"type_id":"SaveImage","fields":{"filename_prefix":"sw"}}
        ],
        "edges": [
            {"from":{"node":1,"port":"CLIP"},"to":{"node":2,"port":"clip"}},
            {"from":{"node":1,"port":"CLIP"},"to":{"node":3,"port":"clip"}},
            {"from":{"node":1,"port":"MODEL"},"to":{"node":6,"port":"model"}},
            {"from":{"node":1,"port":"VAE"},"to":{"node":7,"port":"vae"}},
            {"from":{"node":2,"port":"CONDITIONING"},"to":{"node":6,"port":"positive"}},
            // Negative branch goes THROUGH the switch's on_true input.
            {"from":{"node":3,"port":"CONDITIONING"},"to":{"node":5,"port":"on_true"}},
            {"from":{"node":5,"port":"output"},"to":{"node":6,"port":"negative"}},
            {"from":{"node":4,"port":"LATENT"},"to":{"node":6,"port":"latent_image"}},
            {"from":{"node":6,"port":"LATENT"},"to":{"node":7,"port":"samples"}},
            {"from":{"node":7,"port":"IMAGE"},"to":{"node":8,"port":"images"}}
        ]
    });
    let mut req = json!({
        "model":"m","prompt":"p","width":1024,"height":1024,"seed":1,"workflow": body
    });
    lower_request(&mut req).expect("switch with only the selected branch must lower");
    // The switch routed node 3's CONDITIONING text to the negative slot.
    assert_eq!(req["negative"].as_str(), Some("watermark"));
    assert_eq!(req["workflow_source"].as_str(), Some("typed_linked_graph"));
}

/// A bypassed (mode==4) node is also dropped (the original behavior, kept).
#[test]
fn canvas_mode4_bypass_node_is_dropped() {
    let bypassed = json!({"id":9,"type":"CLIPTextEncode","mode":4,
        "inputs":[{"name":"clip","type":"CLIP","link":21}],
        "outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[]}],
        "widgets_values":["bypassed"]});
    let extra_link = json!([21,1,1,9,0,"CLIP"]);
    let wf = canvas_t2i(vec![bypassed], vec![extra_link]);
    let typed = comfy_ui_canvas_to_typed_body(&wf).expect("typed body builds");
    assert_eq!(typed["nodes"].as_array().unwrap().len(), 7);
    assert_eq!(typed["edges"].as_array().unwrap().len(), 9);
}
