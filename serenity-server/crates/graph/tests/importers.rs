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

#[test]
fn flat_params_adapter_preserves_prompt_json_and_loras_alias() {
    let mut req = json!({
        "workflow": {
            "params": {
                "model": "ideogram4",
                "prompt_json": {
                    "caption": "bbox prompt",
                    "objects": [{"label": "package", "bbox": [128, 192, 768, 832]}]
                },
                "negative": "",
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "seed": 26061790,
                "cfg": 7.0,
                "sampler": "euler",
                "scheduler": "ideogram_logitnormal",
                "clip_skip": 2,
                "eta": 0.25,
                "sigma_min": 0.03,
                "sigma_max": 14.6,
                "restart_sampling": true,
                "vae": "Automatic",
                "loras": [{"name": "adapter.safetensors", "weight": 0.7}],
                "filename_prefix": "bbox_workflow"
            }
        }
    });
    lower_request(&mut req).expect("flat workflow params must lower");
    assert_eq!(req["model"].as_str(), Some("ideogram4"));
    assert_eq!(req["workflow_source"].as_str(), Some("flat_params_adapter"));
    assert_eq!(req["workflow_route_kind"].as_str(), Some("image"));
    assert_eq!(req["workflow_plan"]["route_kind"].as_str(), Some("image"));
    assert_eq!(req["workflow_save_prefix"].as_str(), Some("bbox_workflow"));
    assert_eq!(req["clip_skip"].as_i64(), Some(2));
    assert_eq!(req["eta"].as_f64(), Some(0.25));
    assert_eq!(req["sigma_min"].as_f64(), Some(0.03));
    assert_eq!(req["sigma_max"].as_f64(), Some(14.6));
    assert_eq!(req["restart_sampling"].as_bool(), Some(true));
    assert_eq!(req["vae"].as_str(), Some("Automatic"));
    assert_eq!(
        req["prompt_json"]["objects"][0]["bbox"][0].as_i64(),
        Some(128)
    );
    assert_eq!(req["lora"][0]["name"].as_str(), Some("adapter.safetensors"));
    assert!(
        req.get("loras").is_none(),
        "UI alias must not duplicate canonical lora field"
    );
}

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

fn rust_only_lowering_key(k: &str) -> bool {
    matches!(k, "workflow_route_kind" | "workflow_plan")
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
        if rust_only_lowering_key(k) {
            continue;
        }
        if !want.contains_key(k) {
            diffs.push(format!("  key '{k}': EXTRA (got {got_v}, absent in ref)"));
        }
    }
    assert!(
        diffs.is_empty(),
        "{name} parity failed:\n{}",
        diffs.join("\n")
    );
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

#[test]
fn comfy_canvas_min_t2i_records_image_route_plan() {
    let dir = refs_dir();
    let mut req: JsonValue = serde_json::from_slice(
        &fs::read(dir.join("comfy_canvas__min_t2i.request.json")).expect("read request"),
    )
    .expect("parse request");
    lower_request(&mut req).expect("canvas lowers");
    assert_eq!(req["workflow_route_kind"].as_str(), Some("image"));
    let terminals = req["workflow_plan"]["terminal_nodes"].as_array().unwrap();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0]["type"].as_str(), Some("SaveImage"));
    assert_eq!(terminals[0]["kind"].as_str(), Some("image"));
}

#[test]
fn ltx_no_vae_video_workflow_records_video_route_plan() {
    let mut req = json!({
        "workflow": {
            "1": {
                "class_type": "LTXVLoader",
                "inputs": {
                    "checkpoint_path": "ltx-2.3-22b-dev.safetensors",
                    "gemma_path": "gemma-3-12b-it",
                    "dtype": "bfloat16"
                }
            },
            "2": {
                "class_type": "LTXVSampler",
                "inputs": {
                    "ltxv_model": ["1", 0],
                    "prompt": "cinematic foggy forest",
                    "negative_prompt": "blurry",
                    "width": 768,
                    "height": 512,
                    "num_frames": 97,
                    "steps": 25,
                    "cfg": 3.0,
                    "seed": 42,
                    "frame_rate": 24,
                    "mode": "dev"
                }
            },
            "3": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["2", 1],
                    "filename_prefix": "ltx23_t2v",
                    "fps": 24,
                    "format": "mp4"
                }
            }
        }
    });
    lower_request(&mut req).expect("LTX no-VAE video graph lowers");
    assert_eq!(
        req["workflow_source"].as_str(),
        Some("comfy_api_prompt_graph")
    );
    assert_eq!(req["workflow_route_kind"].as_str(), Some("video"));
    assert_eq!(req["model"].as_str(), Some("ltx-2.3-22b-dev.safetensors"));
    assert_eq!(req["prompt"].as_str(), Some("cinematic foggy forest"));
    assert_eq!(req["num_frames"].as_i64(), Some(97));
    assert_eq!(req["workflow_plan"]["route_kind"].as_str(), Some("video"));
    let terminals = req["workflow_plan"]["terminal_nodes"].as_array().unwrap();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0]["type"].as_str(), Some("SaveVideo"));
    assert_eq!(terminals[0]["input_type"].as_str(), Some("VIDEO"));
}

#[test]
fn ltx_audio_video_workflow_records_audio_video_route_plan() {
    let mut req = json!({
        "workflow": {
            "1": {
                "class_type": "LTXVLoader",
                "inputs": {"checkpoint_path": "ltx-2.3-22b-dev.safetensors"}
            },
            "2": {
                "class_type": "LoadAudio",
                "inputs": {"audio": "input.mp3"}
            },
            "3": {
                "class_type": "LTXVSampler",
                "inputs": {
                    "ltxv_model": ["1", 0],
                    "audio": ["2", 0],
                    "prompt": "motion synchronized to audio",
                    "width": 768,
                    "height": 512,
                    "num_frames": 97,
                    "steps": 20,
                    "cfg": 3.0,
                    "seed": 42,
                    "frame_rate": 24
                }
            },
            "4": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["3", 1], "filename_prefix": "ltx23_a2v", "fps": 24}
            },
            "5": {
                "class_type": "SaveAudioOpus",
                "inputs": {"audio": ["3", 2], "filename_prefix": "ltx23_a2v"}
            }
        }
    });
    lower_request(&mut req).expect("LTX audio/video graph lowers");
    assert_eq!(req["workflow_route_kind"].as_str(), Some("audio_video"));
    let terminals = req["workflow_plan"]["terminal_nodes"].as_array().unwrap();
    assert_eq!(terminals.len(), 2);
    assert_eq!(terminals[0]["type"].as_str(), Some("SaveVideo"));
    assert_eq!(terminals[1]["type"].as_str(), Some("SaveAudioOpus"));
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

#[test]
fn ideogram4_export_accepts_prompt_json_bbox_override() {
    let dir = refs_dir();
    let mut req: JsonValue = serde_json::from_slice(
        &fs::read(dir.join("ideogram4__basic_txt2img_v3.request.json")).expect("read request"),
    )
    .expect("parse");
    let o = req.as_object_mut().unwrap();
    o.remove("prompt");
    o.remove("prompt_raw");
    o.insert(
        "prompt_json".to_string(),
        json!({
            "caption": "place the logo inside the marked package face",
            "objects": [
                {"label": "package", "bbox": [128, 192, 768, 832]}
            ]
        }),
    );

    lower_request(&mut req).unwrap_or_else(|e| panic!("prompt_json override must lower: {e}"));
    let prompt = req["prompt"].as_str().unwrap();
    assert!(
        prompt.contains("\"caption\""),
        "prompt_json lost caption: {prompt}"
    );
    assert!(
        prompt.contains("\"bbox\""),
        "prompt_json lost bbox: {prompt}"
    );
    assert_eq!(req["prompt_raw"].as_str(), Some(prompt));
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
        json!([10, 1, 0, 5, 0, "MODEL"]),
        json!([11, 1, 1, 2, 0, "CLIP"]),
        json!([12, 1, 1, 3, 0, "CLIP"]),
        json!([13, 1, 2, 6, 1, "VAE"]),
        json!([14, 2, 0, 5, 1, "CONDITIONING"]),
        json!([15, 3, 0, 5, 2, "CONDITIONING"]),
        json!([16, 4, 0, 5, 3, "LATENT"]),
        json!([17, 5, 0, 6, 0, "LATENT"]),
        json!([18, 6, 0, 7, 0, "IMAGE"]),
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
    let extra_link = json!([19, 1, 1, 8, 0, "CLIP"]);
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
        edges
            .iter()
            .all(|e| e["from"]["node"].as_i64() != Some(8) && e["to"]["node"].as_i64() != Some(8)),
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

// --- SamplerCustom ecosystem (ComfyUI-parity Phase 1) --------------------------

/// SamplerCustom — the single-output sibling of SamplerCustomAdvanced — lowers
/// through the SAME flat sampler/scheduler/steps/seed/cfg path. The cfg/seed come
/// from the node's own widgets (it has no CFGGuider/RandomNoise inputs); the
/// sampler name comes from a KSamplerSelect(euler) and the schedule from a
/// BasicScheduler(simple). Verified byte-identical to the Mojo oracle.
#[test]
fn sampler_custom_lowers_to_flat_params() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a red fox in snow"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "blurry, ugly"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "6": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 24, "denoise": 1.0}},
        "7": {"class_type": "SamplerCustom", "inputs": {"add_noise": true, "noise_seed": 12345, "cfg": 6.5, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "sampler": ["5", 0], "sigmas": ["6", 0], "latent_image": ["4", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "sctest"}}
    }});
    lower_request(&mut req).expect("SamplerCustom must lower cleanly");
    assert_eq!(req["sampler"].as_str(), Some("euler"));
    assert_eq!(req["scheduler"].as_str(), Some("simple"));
    assert_eq!(req["steps"].as_i64(), Some(24));
    assert_eq!(req["seed"].as_i64(), Some(12345));
    assert_eq!(req["cfg"].as_f64(), Some(6.5));
    assert_eq!(req["creativity"].as_f64(), Some(1.0));
    assert_eq!(req["prompt"].as_str(), Some("a red fox in snow"));
    assert_eq!(req["negative"].as_str(), Some("blurry, ugly"));
    assert_eq!(req["width"].as_i64(), Some(768));
    assert_eq!(req["height"].as_i64(), Some(512));
}

/// A named SAMPLER node (SamplerEulerAncestral → euler_ancestral) is NOT in the
/// zimage worker's supported list, so the lowering fails loud [501] rather than
/// silently substituting a different sampler.
#[test]
fn named_sampler_unsupported_is_rejected_loud() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a red fox"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "ugly"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}},
        "5": {"class_type": "SamplerEulerAncestral", "inputs": {"eta": 1.0, "s_noise": 1.0}},
        "6": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 24, "denoise": 1.0}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 777}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "cfg": 7.0}},
        "7": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["5", 0], "sigmas": ["6", 0], "latent_image": ["4", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "sctest"}}
    }});
    let err = lower_request(&mut req).expect_err("unsupported named sampler must 501");
    let msg = format!("{err}");
    assert!(msg.contains("euler_ancestral"), "msg = {msg}");
    assert!(msg.contains("unsupported sampler"), "msg = {msg}");
}

/// A named SIGMAS scheduler node (KarrasScheduler → karras) is NOT in the zimage
/// worker's supported list, so the lowering fails loud [501].
#[test]
fn named_scheduler_unsupported_is_rejected_loud() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a red fox"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "ugly"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "6": {"class_type": "KarrasScheduler", "inputs": {"steps": 24, "sigma_max": 14.6, "sigma_min": 0.03, "rho": 7.0}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 777}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "cfg": 7.0}},
        "7": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["5", 0], "sigmas": ["6", 0], "latent_image": ["4", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "sctest"}}
    }});
    let err = lower_request(&mut req).expect_err("unsupported named scheduler must 501");
    let msg = format!("{err}");
    assert!(msg.contains("karras"), "msg = {msg}");
    assert!(msg.contains("unsupported scheduler"), "msg = {msg}");
}

/// A bypassed (mode==4) node is also dropped (the original behavior, kept).
#[test]
fn canvas_mode4_bypass_node_is_dropped() {
    let bypassed = json!({"id":9,"type":"CLIPTextEncode","mode":4,
        "inputs":[{"name":"clip","type":"CLIP","link":21}],
        "outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[]}],
        "widgets_values":["bypassed"]});
    let extra_link = json!([21, 1, 1, 9, 0, "CLIP"]);
    let wf = canvas_t2i(vec![bypassed], vec![extra_link]);
    let typed = comfy_ui_canvas_to_typed_body(&wf).expect("typed body builds");
    assert_eq!(typed["nodes"].as_array().unwrap().len(), 7);
    assert_eq!(typed["edges"].as_array().unwrap().len(), 9);
}

// --- image/mask utility nodes (ComfyUI-parity Phase 2) -------------------------

/// LoadImageMask is an alias of LoadImage: its MASK output (slot 1) resolves to
/// the mask_image param exactly like LoadImage's MASK output. Here a
/// SetLatentNoiseMask consumes the LoadImageMask MASK so the mask_image is
/// written into the flat params, proving the alias resolves through.
#[test]
fn load_image_mask_alias_resolves_mask() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a cabin"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "blurry"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}},
        "10": {"class_type": "LoadImageMask", "inputs": {"image": "mymask.png"}},
        "11": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["4", 0], "mask": ["10", 1]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 9, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["11", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "lim"}}
    }});
    lower_request(&mut req).expect("LoadImageMask alias must lower cleanly");
    assert_eq!(req["mask_image"].as_str(), Some("mymask.png"));
    assert_eq!(req["prompt"].as_str(), Some("a cabin"));
    assert_eq!(
        req["workflow_source"].as_str(),
        Some("comfy_api_prompt_graph")
    );
}

/// LoadImageOutput is an alias of LoadImage: its IMAGE output (slot 0) resolves to
/// the init_image param through a VAEEncode → KSampler img2img chain.
#[test]
fn load_image_output_alias_resolves_init_image() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a city"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "ugly"}},
        "10": {"class_type": "LoadImageOutput", "inputs": {"image": "input.png"}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 3, "steps": 18, "cfg": 4.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.6, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["11", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "loo"}}
    }});
    lower_request(&mut req).expect("LoadImageOutput alias must lower cleanly");
    assert_eq!(req["init_image"].as_str(), Some("input.png"));
    assert_eq!(req["creativity"].as_f64(), Some(0.6));
    assert_eq!(req["prompt"].as_str(), Some("a city"));
}

/// GetImageSizeAndCount (KJ) is an alias of GetImageSize plus a leading IMAGE
/// passthrough (slot 0) and a count (slot 3). The IMAGE passthrough must carry
/// the source path through to a VAEEncode → KSampler init_image; the size INT
/// outputs are placeholders in the flat single-image model.
#[test]
fn get_image_size_and_count_passthrough_lowers() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a forest"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "blur"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "src.png"}},
        // GetImageSizeAndCount passes the image through on slot 0.
        "11": {"class_type": "GetImageSizeAndCount", "inputs": {"image": ["10", 0]}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.7, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["12", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "gisc"}}
    }});
    lower_request(&mut req).expect("GetImageSizeAndCount passthrough must lower");
    assert_eq!(req["init_image"].as_str(), Some("src.png"));
    assert_eq!(req["prompt"].as_str(), Some("a forest"));
}

/// ImageResizeKJ with explicit nonzero width/height (keep_proportion=false, no
/// get_image_size input) resolves the explicit dims into the flat width/height
/// and passes the IMAGE handle through to a VAEEncode → KSampler chain.
#[test]
fn image_resize_kj_explicit_dims_resolve() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a lake"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "noise"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}},
        "11": {"class_type": "ImageResizeKJ", "inputs": {"image": ["10", 0], "width": 1024, "height": 1024, "keep_proportion": false, "divisible_by": 2}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 2, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.5, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["12", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "irkj"}}
    }});
    lower_request(&mut req).expect("ImageResizeKJ explicit dims must lower");
    assert_eq!(req["width"].as_i64(), Some(1024));
    assert_eq!(req["height"].as_i64(), Some(1024));
    assert_eq!(req["init_image"].as_str(), Some("photo.png"));
}

/// ImageResizeKJ with keep_proportion=true derives a dimension from the
/// un-knowable source aspect → fail loud [501] (never silently emit a wrong size).
#[test]
fn image_resize_kj_keep_proportion_is_rejected_loud() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a lake"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "noise"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}},
        "11": {"class_type": "ImageResizeKJ", "inputs": {"image": ["10", 0], "width": 1024, "height": 0, "keep_proportion": true, "divisible_by": 2}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 2, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.5, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["12", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "irkj"}}
    }});
    let err = lower_request(&mut req).expect_err("keep_proportion must 501");
    let msg = format!("{err}");
    assert!(msg.contains("keep_proportion"), "msg = {msg}");
    assert!(msg.contains("not resolvable"), "msg = {msg}");
}

/// ImageResizeKJ with a zero target dimension means "keep the source dim", which
/// is not resolvable in the flat single-image model → fail loud [501].
#[test]
fn image_resize_kj_zero_dim_is_rejected_loud() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a lake"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "noise"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}},
        "11": {"class_type": "ImageResizeKJ", "inputs": {"image": ["10", 0], "width": 0, "height": 768, "keep_proportion": false, "divisible_by": 2}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 2, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.5, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["12", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "irkj"}}
    }});
    let err = lower_request(&mut req).expect_err("zero dim must 501");
    let msg = format!("{err}");
    assert!(msg.contains("keeps the source dimension"), "msg = {msg}");
}

/// ImageScaleBy multiplies the un-knowable source dims by scale_by, which cannot
/// be represented in the flat single-image model → fail loud [501] (the honest
/// choice; never silently emit a wrong size).
#[test]
fn image_scale_by_is_rejected_loud() {
    let mut req = json!({"workflow": {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "zimage_base"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a lake"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "noise"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}},
        "11": {"class_type": "ImageScaleBy", "inputs": {"image": ["10", 0], "upscale_method": "lanczos", "scale_by": 2.0}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 2, "steps": 20, "cfg": 5.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.5, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["12", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "isb"}}
    }});
    let err = lower_request(&mut req).expect_err("ImageScaleBy must 501");
    let msg = format!("{err}");
    assert!(msg.contains("ImageScaleBy"), "msg = {msg}");
    assert!(msg.contains("not resolvable"), "msg = {msg}");
}
