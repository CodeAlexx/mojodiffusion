//! Body-kind detection + importers that build a raw `{nodes, edges}` body from
//! the supported workflow request shapes. Port of the importer dispatch in
//! `apply_workflow_params` (Mojo line 2912) and the Comfy-API prompt importer
//! (`_comfy_api_prompt_to_typed_graph`, Mojo 1525 / `apply_comfy_api_prompt_graph`,
//! 1579).
//!
//! The only importer the t2i+lora corpus exercises is the Comfy-API prompt
//! graph (every `*_t2i` / `*_t2i_lora` ref lowers with
//! `workflow_source == "comfy_api_prompt_graph"`). The flat `params`/`genparams`
//! passthrough and the body-kind detection guards are ported for fidelity; the
//! Ideogram4 / Comfy-UI-canvas importers are out of scope for this stage and
//! fail loud (501) if a body matches them, rather than silently no-oping.

use serde_json::{json, Map, Value as JsonValue};

use crate::util::{
    is_bool_scalar_node, is_float_scalar_node, is_int_scalar_node, is_scalar_node as util_is_scalar,
    is_string_scalar_node, json_intish, node_id, node_mode, node_type, widget_bool, widget_float,
    widget_int, widget_string, wf_string,
};
use crate::{
    canonical_type_id, scalar_output_type, GraphError, GraphResult,
};

/// The detected request-body kind (mirrors the dispatch order in
/// `apply_workflow_params`, Mojo 2934-3011).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyKind {
    Ideogram4Export,
    ComfyApiPrompt,
    ComfyUiCanvas,
    FlatParams,
    FlatGenparams,
    /// `{nodes, edges}` typed graph (already lowerable directly).
    TypedGraph,
}

/// Flat keys copied verbatim from a `params`/`genparams` object onto the
/// request (Mojo 2948-2970 / 2979-3001).
pub const FLAT_PARAM_KEYS: &[&str] = &[
    "model", "prompt", "prompt_raw", "negative", "width", "height",
    "steps", "seed", "cfg", "cfg_override", "cfg_override_start_percent",
    "cfg_override_end_percent", "sampler", "scheduler", "sigma_shift",
    "variation_seed", "variation_strength", "images", "init_image", "creativity",
    "workflow_save_prefix",
    "mask_image", "reference_image", "reference_latent_method", "reference_latent_count",
    "inpaint_conditioning_image", "inpaint_conditioning_mask", "inpaint_conditioning_noise_mask",
    "qwen_edit_conditioning_image",
    "sample_caps_pos", "sample_caps_neg", "caps_pos", "caps_neg",
    "caps_positive", "caps_negative",
    "conditioning_mask_image", "conditioning_mask_channel",
    "conditioning_mask_strength", "conditioning_mask_set_area_to_bounds",
    "outpaint_left", "outpaint_top", "outpaint_right", "outpaint_bottom",
    "outpaint_feathering", "threshold_mask_value", "threshold_mask_operator",
    "lanpaint_mask_channel", "lanpaint_mask_blend_overlap", "lanpaint_num_steps",
    "lanpaint_lambda", "lanpaint_step_size", "lanpaint_beta", "lanpaint_friction",
    "lanpaint_prompt_mode", "lanpaint_inpainting_mode", "lanpaint_add_noise",
    "lanpaint_noise_seed", "lanpaint_start_at_step", "lanpaint_end_at_step",
    "lanpaint_return_with_leftover_noise", "lanpaint_early_stop",
    "lanpaint_inner_threshold", "lanpaint_inner_patience",
    "lora",
];

// ---------------------------------------------------------------------------
// Detection (mirrors looks_like_* in the Mojo file)
// ---------------------------------------------------------------------------

/// `_comfy_api_prompt_body` (Mojo 1311): unwrap a `prompt`/`comfy_prompt`
/// envelope around a bare Comfy-API prompt graph.
pub fn comfy_api_prompt_body(wf: &JsonValue) -> JsonValue {
    if let Some(obj) = wf.as_object() {
        if let Some(p) = obj.get("prompt") {
            if p.is_object() {
                return p.clone();
            }
        }
        if let Some(p) = obj.get("comfy_prompt") {
            if p.is_object() {
                return p.clone();
            }
        }
    }
    wf.clone()
}

/// `looks_like_ideogram4_comfy_ui_export` (Mojo 328): `nodes` + `links` +
/// `definitions.subgraphs` containing a subgraph named "...ideogram...".
pub fn looks_like_ideogram4_export(wf: &JsonValue) -> bool {
    let obj = match wf.as_object() {
        Some(o) => o,
        None => return false,
    };
    if !obj.get("nodes").map(JsonValue::is_array).unwrap_or(false) {
        return false;
    }
    if !obj.get("links").map(JsonValue::is_array).unwrap_or(false) {
        return false;
    }
    let defs = match obj.get("definitions").and_then(JsonValue::as_object) {
        Some(d) => d,
        None => return false,
    };
    let subgraphs = match defs.get("subgraphs").and_then(JsonValue::as_array) {
        Some(s) => s,
        None => return false,
    };
    for sg in subgraphs {
        let name = sg
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_lowercase();
        if name.contains("ideogram") {
            return true;
        }
    }
    false
}

/// `looks_like_comfy_api_prompt_graph` (Mojo 1319): every key parses as an int,
/// every node is an object with string `class_type` and object `inputs`.
pub fn looks_like_comfy_api_prompt_graph(wf: &JsonValue) -> bool {
    let graph = comfy_api_prompt_body(wf);
    let obj = match graph.as_object() {
        Some(o) => o,
        None => return false,
    };
    if obj.is_empty() {
        return false;
    }
    for (k, node) in obj {
        if k.parse::<i64>().is_err() {
            return false;
        }
        let node = match node.as_object() {
            Some(n) => n,
            None => return false,
        };
        if !node.get("class_type").map(JsonValue::is_string).unwrap_or(false) {
            return false;
        }
        if !node.get("inputs").map(JsonValue::is_object).unwrap_or(false) {
            return false;
        }
    }
    true
}

/// `looks_like_comfy_ui_canvas_graph` (Mojo 1015): `nodes` array + `links`
/// array, every node an object with a non-empty type.
pub fn looks_like_comfy_ui_canvas_graph(wf: &JsonValue) -> bool {
    let obj = match wf.as_object() {
        Some(o) => o,
        None => return false,
    };
    if !obj.get("nodes").map(JsonValue::is_array).unwrap_or(false) {
        return false;
    }
    if !obj.get("links").map(JsonValue::is_array).unwrap_or(false) {
        return false;
    }
    for node in obj["nodes"].as_array().unwrap() {
        let typ = node
            .get("type")
            .and_then(JsonValue::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| node.get("type_id").and_then(JsonValue::as_str))
            .unwrap_or("");
        if typ.is_empty() {
            return false;
        }
    }
    true
}

/// Detect the body kind for a workflow body, following the exact precedence of
/// `apply_workflow_params` (ideogram → comfy-api → comfy-canvas → params →
/// genparams → typed `{nodes,edges}`).
pub fn detect_body_kind(wf: &JsonValue) -> GraphResult<BodyKind> {
    if !wf.is_object() {
        return Err(GraphError::unsupported(
            "workflow graph body must be an object",
        ));
    }
    if looks_like_ideogram4_export(wf) {
        return Ok(BodyKind::Ideogram4Export);
    }
    if looks_like_comfy_api_prompt_graph(wf) {
        return Ok(BodyKind::ComfyApiPrompt);
    }
    if looks_like_comfy_ui_canvas_graph(wf) {
        return Ok(BodyKind::ComfyUiCanvas);
    }
    if wf.get("params").map(JsonValue::is_object).unwrap_or(false) {
        return Ok(BodyKind::FlatParams);
    }
    if wf.get("genparams").map(JsonValue::is_object).unwrap_or(false) {
        return Ok(BodyKind::FlatGenparams);
    }
    // A `nodes` body without `edges` is the legacy nodes-only fallback that the
    // audit explicitly drops: a nodes body without edges must 501 (not the
    // title-sniffing path).
    let has_nodes = wf.get("nodes").map(JsonValue::is_array).unwrap_or(false);
    let has_edges = wf.get("edges").map(JsonValue::is_array).unwrap_or(false);
    if has_nodes && has_edges {
        return Ok(BodyKind::TypedGraph);
    }
    if has_nodes && !has_edges {
        return Err(GraphError::unsupported(
            "workflow graph body needs edges for typed execution",
        ));
    }
    Err(GraphError::unsupported(
        "workflow graph body needs nodes or params/genparams",
    ))
}

// ---------------------------------------------------------------------------
// Comfy-API prompt → typed `{nodes, edges}` body
// ---------------------------------------------------------------------------

fn link_node_id(v: &JsonValue) -> GraphResult<i64> {
    let arr = v.as_array().filter(|a| a.len() >= 2).ok_or_else(|| {
        GraphError::unsupported("Comfy API prompt link must be [node_id, output_index]")
    })?;
    if let Some(n) = arr[0].as_i64() {
        return Ok(n);
    }
    if let Some(s) = arr[0].as_str() {
        if let Ok(n) = s.parse::<i64>() {
            return Ok(n);
        }
    }
    Err(GraphError::unsupported(
        "Comfy API prompt link node_id must be an integer",
    ))
}

fn link_output_slot(v: &JsonValue) -> GraphResult<i64> {
    let arr = v.as_array().filter(|a| a.len() >= 2).ok_or_else(|| {
        GraphError::unsupported("Comfy API prompt link must be [node_id, output_index]")
    })?;
    if let Some(n) = arr[1].as_i64() {
        return Ok(n);
    }
    if let Some(f) = arr[1].as_f64() {
        return Ok(f as i64);
    }
    Err(GraphError::unsupported(
        "Comfy API prompt link output_index must be an integer",
    ))
}

/// `_comfy_api_input_is_link` (Mojo 1364): `[node_ref, slot]` where node_ref is
/// int|string and slot is int|number.
fn input_is_link(v: &JsonValue) -> bool {
    let arr = match v.as_array() {
        Some(a) if a.len() >= 2 => a,
        _ => return false,
    };
    let node_ok = arr[0].is_i64() || arr[0].is_string();
    let slot_ok = arr[1].is_i64() || arr[1].is_number();
    node_ok && slot_ok
}

/// `_comfy_api_output_port` (Mojo 1372): resolve the typed port name a source
/// node produces on the given output slot.
fn output_port(graph: &Map<String, JsonValue>, src_id: i64, slot: i64) -> GraphResult<String> {
    let key = src_id.to_string();
    let node = graph
        .get(&key)
        .ok_or_else(|| GraphError::unsupported(format!(
            "Comfy API prompt link references missing node: {key}"
        )))?;
    let typ = canonical_type_id(
        node.get("class_type").and_then(JsonValue::as_str).unwrap_or(""),
    );

    if crate::is_scalar_node(&typ) && slot == 0 {
        if typ == "PrimitiveNode" {
            if let Some(inputs) = node.get("inputs").filter(|v| v.is_object()) {
                let pt = scalar_output_type(&typ, inputs);
                if !pt.is_empty() {
                    return Ok(pt);
                }
            }
        } else {
            let empty = json!({});
            let st = scalar_output_type(&typ, &empty);
            if !st.is_empty() {
                return Ok(st);
            }
        }
    }

    let port = match typ.as_str() {
        "CheckpointLoaderSimple" => match slot {
            0 => Some("MODEL"),
            1 => Some("CLIP"),
            2 => Some("VAE"),
            _ => None,
        },
        "UNETLoader" | "DiffusionModelLoader" => (slot == 0).then_some("MODEL"),
        "LoraLoaderModelOnly" | "ZImageLoraModelOnly" => (slot == 0).then_some("MODEL"),
        "LoraLoader" => match slot {
            0 => Some("MODEL"),
            1 => Some("CLIP"),
            _ => None,
        },
        "CLIPLoader" | "DualCLIPLoader" | "TripleCLIPLoader" => (slot == 0).then_some("CLIP"),
        "VAELoader" => (slot == 0).then_some("VAE"),
        "CLIPTextEncode" | "CLIPTextEncodeFlux" | "TextEncodeQwenImageEdit"
        | "TextEncodeQwenImageEditPlus" => (slot == 0).then_some("CONDITIONING"),
        "ConditioningZeroOut" => (slot == 0).then_some("CONDITIONING"),
        "ConditioningSetMask" => (slot == 0).then_some("CONDITIONING"),
        "LoadImage" => match slot {
            0 => Some("IMAGE"),
            1 => Some("MASK"),
            _ => None,
        },
        "ImageToMask" => (slot == 0).then_some("MASK"),
        "MaskToImage" => (slot == 0).then_some("IMAGE"),
        "EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage" => {
            (slot == 0).then_some("LATENT")
        }
        "VAEEncode" => (slot == 0).then_some("LATENT"),
        "SetLatentNoiseMask" => (slot == 0).then_some("LATENT"),
        "ImageScale" | "ImageScaleToTotalPixels" => (slot == 0).then_some("IMAGE"),
        "ImagePadForOutpaint" => match slot {
            0 => Some("IMAGE"),
            1 => Some("MASK"),
            _ => None,
        },
        "ThresholdMask" => (slot == 0).then_some("MASK"),
        "InpaintModelConditioning" => match slot {
            0 => Some("positive"),
            1 => Some("negative"),
            2 => Some("LATENT"),
            _ => None,
        },
        "GetImageSize" => match slot {
            0 => Some("width"),
            1 => Some("height"),
            2 => Some("batch_size"),
            _ => None,
        },
        "ReferenceLatent" => (slot == 0).then_some("CONDITIONING"),
        "FluxGuidance" => (slot == 0).then_some("CONDITIONING"),
        "ModelSamplingAuraFlow" | "ModelSamplingSD3" | "DifferentialDiffusion" => {
            (slot == 0).then_some("MODEL")
        }
        "KSampler" | "LanPaint_KSampler" | "LanPaint_KSamplerAdvanced" => {
            (slot == 0).then_some("LATENT")
        }
        "CFGGuider" => (slot == 0).then_some("GUIDER"),
        "BasicGuider" => (slot == 0).then_some("GUIDER"),
        "Flux2Scheduler" | "BasicScheduler" => (slot == 0).then_some("SIGMAS"),
        "RandomNoise" => (slot == 0).then_some("NOISE"),
        "KSamplerSelect" => (slot == 0).then_some("SAMPLER"),
        "SamplerCustomAdvanced" | "LanPaint_SamplerCustomAdvanced" => {
            (slot == 0 || slot == 1).then_some("LATENT")
        }
        "Reroute" => (slot == 0).then_some("REROUTE"),
        "SetNode" => (slot == 0).then_some("SET"),
        "GetNode" => (slot == 0).then_some("GET"),
        "ComfySwitchNode" => (slot == 0).then_some("output"),
        "LanPaint_MaskBlend" => (slot == 0).then_some("IMAGE"),
        "VAEDecode" => (slot == 0).then_some("IMAGE"),
        "SaveImage" => {
            return Err(GraphError::unsupported(
                "Comfy API prompt SaveImage has no supported output slot",
            ))
        }
        _ => {
            return Err(GraphError::unsupported(format!(
                "unsupported Comfy API prompt output node type: {typ}"
            ))
            .with_type(typ))
        }
    };
    match port {
        Some(p) => Ok(p.to_string()),
        None => Err(GraphError::unsupported(format!(
            "unsupported Comfy API prompt output node type: {typ}"
        ))
        .with_type(typ)),
    }
}

/// `_comfy_api_prompt_to_typed_graph` (Mojo 1525): convert a bare Comfy-API
/// prompt graph into a `{nodes, edges}` typed body. Node iteration follows the
/// graph's key order (serde_json preserves insertion order with the
/// `preserve_order` feature); edges are appended in input-key order per node.
pub fn comfy_api_prompt_to_typed_body(graph: &JsonValue) -> GraphResult<JsonValue> {
    let graph_obj = graph
        .as_object()
        .ok_or_else(|| GraphError::unsupported("Comfy API prompt graph must be an object"))?;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for (key, src) in graph_obj {
        let node_id = key.parse::<i64>().map_err(|_| {
            GraphError::unsupported("Comfy API prompt node id must be an integer")
        })?;
        let src = src
            .as_object()
            .ok_or_else(|| GraphError::unsupported("Comfy API prompt node must be an object"))?;
        let typ = canonical_type_id(
            src.get("class_type").and_then(JsonValue::as_str).unwrap_or(""),
        );
        if typ.is_empty() {
            return Err(GraphError::unsupported(
                "Comfy API prompt node missing class_type",
            )
            .with_node(node_id));
        }
        let inputs = src
            .get("inputs")
            .and_then(JsonValue::as_object)
            .ok_or_else(|| {
                GraphError::unsupported("Comfy API prompt node missing inputs object")
                    .with_node(node_id)
            })?;

        let mut fields = Map::new();
        for (input_name, input_value) in inputs {
            if input_is_link(input_value) {
                let source_id = link_node_id(input_value)?;
                let source_slot = link_output_slot(input_value)?;
                let from_port = output_port(graph_obj, source_id, source_slot)?;
                let mut to_port = input_name.clone();
                if typ == "Reroute" && to_port.is_empty() {
                    to_port = "input".to_string();
                }
                edges.push(json!({
                    "from": { "node": source_id, "port": from_port },
                    "to": { "node": node_id, "port": to_port },
                }));
            } else {
                fields.insert(input_name.clone(), input_value.clone());
            }
        }

        nodes.push(json!({
            "id": node_id,
            "type_id": typ,
            "fields": JsonValue::Object(fields),
        }));
    }

    Ok(json!({ "nodes": nodes, "edges": edges }))
}

// ---------------------------------------------------------------------------
// Ideogram4 Comfy-UI export -> flat backend params
// ---------------------------------------------------------------------------

/// `_set_if_missing` (Mojo 113): write `key`=`value` only if it is currently
/// absent or null (the graph fills a still-missing field; placeholders win when
/// already present).
fn set_if_missing(obj: &mut JsonValue, key: &str, value: JsonValue) {
    let o = obj.as_object_mut().expect("request is an object");
    let missing = o.get(key).map(JsonValue::is_null).unwrap_or(true);
    if missing {
        o.insert(key.to_string(), value);
    }
}

/// `_workflow_has_prompt_override` (Mojo 295): a non-empty top-level `prompt`
/// satisfies the gate; a non-empty `prompt_raw` is copied into `prompt` (if
/// missing) and also satisfies it.
fn has_prompt_override(obj: &mut JsonValue) -> bool {
    if let Some(s) = obj.get("prompt").and_then(JsonValue::as_str) {
        if !s.is_empty() {
            return true;
        }
    }
    if let Some(raw) = obj.get("prompt_raw").and_then(JsonValue::as_str) {
        if !raw.is_empty() {
            let raw_val = obj["prompt_raw"].clone();
            set_if_missing(obj, "prompt", raw_val);
            return true;
        }
    }
    false
}

/// `_workflow_set_seed_from_widget_if_missing` (Mojo 304): keep an existing
/// top-level seed; otherwise require a valid uint32 seed from the widget.
fn set_seed_from_widget_if_missing(obj: &mut JsonValue, seed: i64) -> GraphResult<()> {
    if obj
        .get("seed")
        .map(|v| !v.is_null())
        .unwrap_or(false)
    {
        return Ok(());
    }
    if seed < 0 {
        return Err(GraphError::unsupported(
            "Ideogram4 Comfy export uses randomized seed; provide top-level seed",
        ));
    }
    if seed > 4_294_967_295 {
        return Err(GraphError::unsupported(
            "Ideogram4 Comfy export seed exceeds the daemon uint32 seed range; provide top-level seed",
        ));
    }
    obj.as_object_mut()
        .unwrap()
        .insert("seed".to_string(), json!(seed));
    Ok(())
}

/// `_ideogram4_mode_steps` (Mojo 318): map the CustomCombo selection to a step
/// count (fail-loud on an unknown mode).
fn ideogram4_mode_steps(mode: &str) -> GraphResult<i64> {
    match mode {
        "Quality" => Ok(48),
        "Default" => Ok(20),
        "Turbo" => Ok(12),
        other => Err(GraphError::unsupported(format!(
            "unsupported Ideogram4 workflow mode: {other}"
        ))),
    }
}

/// Read a node's `widgets_values` array, or an empty array when absent.
fn node_widgets(node: &JsonValue) -> JsonValue {
    node.get("widgets_values")
        .filter(|v| v.is_array())
        .cloned()
        .unwrap_or_else(|| JsonValue::Array(Vec::new()))
}

/// `apply_ideogram4_comfy_ui_export` (Mojo 347): import the bounded Comfy-UI
/// Ideogram4 txt2img canvas. Writes flat backend params onto `obj` in place via
/// `set_if_missing` (graph values fill still-missing fields), validates the root
/// nodes and the "Text to Image (Ideogram v4)" subgraph, and records the
/// execution metadata. Requires a top-level prompt override (the prompt-builder
/// subgraph needs external Gemma/KJ execution).
pub fn apply_ideogram4_comfy_ui_export(obj: &mut JsonValue, wf: &JsonValue) -> GraphResult<()> {
    if !has_prompt_override(obj) {
        return Err(GraphError::unsupported(
            "Ideogram4 Comfy export uses a prompt-builder subgraph; provide top-level prompt or prompt_raw",
        ));
    }

    let root_nodes = wf
        .get("nodes")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| GraphError::unsupported("Ideogram4 Comfy export missing nodes array"))?;
    for node in root_nodes {
        if !node.is_object() {
            return Err(GraphError::unsupported(
                "Ideogram4 Comfy export root node must be an object",
            ));
        }
        let typ = node_type(node);
        let mode = node_mode(node);
        if matches!(
            typ.as_str(),
            "LoraLoader" | "LoraLoaderModelOnly" | "ZImageLoraModelOnly"
        ) {
            if mode != 4 {
                return Err(GraphError::unsupported(
                    "Ideogram4 Comfy export has active LoRA nodes, but the current Ideogram4 backend does not execute LoRA",
                ));
            }
        } else if matches!(typ.as_str(), "Seed (rgthree)") {
            let widgets = node_widgets(node);
            set_seed_from_widget_if_missing(obj, widget_int(&widgets, 0, -1))?;
        } else if matches!(
            typ.as_str(),
            "MarkdownNote"
                | "SaveImage"
                | "ResolutionSelector"
                | "PreviewAny"
                | "Ideogram4PromptBuilderKJ"
                | "83e6e004-48ea-408e-9024-eb49c3d7dc14"
                | "f5f04613-ee09-4cd9-9ada-a880360891d4"
        ) {
            // Allowed inert/UI root node.
        } else if mode != 4 {
            return Err(GraphError::unsupported(format!(
                "unsupported active Ideogram4 Comfy root node: {typ}"
            )));
        }
    }

    let sg_nodes = {
        let subgraphs = wf
            .get("definitions")
            .and_then(JsonValue::as_object)
            .and_then(|d| d.get("subgraphs"))
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                GraphError::unsupported(
                    "Ideogram4 Comfy export missing Text to Image (Ideogram v4) subgraph",
                )
            })?;
        let mut found: Option<&Vec<JsonValue>> = None;
        for sg in subgraphs {
            let name = wf_string(sg, "name").to_lowercase();
            if name.contains("text to image") && name.contains("ideogram") {
                let nodes = sg
                    .get("nodes")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| {
                        GraphError::unsupported("Ideogram4 Comfy export subgraph missing nodes")
                    })?;
                found = Some(nodes);
                break;
            }
        }
        found.ok_or_else(|| {
            GraphError::unsupported(
                "Ideogram4 Comfy export missing Text to Image (Ideogram v4) subgraph",
            )
        })?
    };

    let mut saw_empty_latent = false;
    let mut saw_cond_model = false;
    let mut saw_uncond_model = false;
    let mut saw_clip = false;
    let mut saw_vae = false;
    let mut saw_sampler = false;
    let mut saw_scheduler = false;
    let mut saw_aura = false;
    let mut saw_guider = false;
    let mut saw_cfg_override = false;
    let mut saw_prompt_encode = false;
    let mut selected_mode = "Default".to_string();

    for node in sg_nodes {
        if !node.is_object() {
            return Err(GraphError::unsupported(
                "Ideogram4 Comfy export subgraph node must be an object",
            ));
        }
        let typ = node_type(node);
        let mode = node_mode(node);
        if mode == 4 {
            continue;
        }
        let widgets = node_widgets(node);

        match typ.as_str() {
            "EmptyFlux2LatentImage" => {
                set_if_missing(obj, "width", json!(widget_int(&widgets, 0, 1024)));
                set_if_missing(obj, "height", json!(widget_int(&widgets, 1, 1024)));
                set_if_missing(obj, "images", json!(widget_int(&widgets, 2, 1)));
                saw_empty_latent = true;
            }
            "UNETLoader" => {
                let name = widget_string(&widgets, 0, "");
                let lower = name.to_lowercase();
                if lower.contains("ideogram4_unconditional") {
                    saw_uncond_model = true;
                } else if lower.contains("ideogram4") {
                    saw_cond_model = true;
                    set_if_missing(obj, "model", json!("ideogram-4-fp8"));
                }
            }
            "CLIPLoader" => {
                if widget_string(&widgets, 0, "").to_lowercase().contains("qwen3vl") {
                    saw_clip = true;
                }
            }
            "VAELoader" => {
                if widget_string(&widgets, 0, "").to_lowercase().contains("flux2") {
                    saw_vae = true;
                }
            }
            "CLIPTextEncode" => {
                saw_prompt_encode = true;
            }
            "KSamplerSelect" => {
                set_if_missing(
                    obj,
                    "sampler",
                    json!(widget_string(&widgets, 0, "euler")),
                );
                saw_sampler = true;
            }
            "BasicScheduler" => {
                set_if_missing(
                    obj,
                    "scheduler",
                    json!(widget_string(&widgets, 0, "simple")),
                );
                saw_scheduler = true;
            }
            "ModelSamplingAuraFlow" => {
                set_if_missing(obj, "sigma_shift", json!(widget_float(&widgets, 0, 5.0)));
                saw_aura = true;
            }
            "DualModelGuider" => {
                set_if_missing(obj, "cfg", json!(widget_float(&widgets, 0, 7.0)));
                saw_guider = true;
            }
            "CFGOverride" => {
                set_if_missing(obj, "cfg_override", json!(widget_float(&widgets, 0, 3.0)));
                set_if_missing(
                    obj,
                    "cfg_override_start_percent",
                    json!(widget_float(&widgets, 1, 0.7)),
                );
                set_if_missing(
                    obj,
                    "cfg_override_end_percent",
                    json!(widget_float(&widgets, 2, 1.0)),
                );
                saw_cfg_override = true;
            }
            "CustomCombo" => {
                selected_mode = widget_string(&widgets, 0, &selected_mode);
            }
            "ConditioningZeroOut" | "SamplerCustomAdvanced" | "VAEDecode" | "RandomNoise"
            | "PrimitiveInt" | "ComfyMathExpression" | "JsonExtractString" | "StringReplace"
            | "ComfyNumberConvert" => {
                // Allowed inert/auxiliary subgraph node.
            }
            other => {
                return Err(GraphError::unsupported(format!(
                    "unsupported Ideogram4 Comfy subgraph node: {other}"
                )));
            }
        }
    }

    set_if_missing(obj, "steps", json!(ideogram4_mode_steps(&selected_mode)?));
    let all_present = saw_empty_latent
        && saw_cond_model
        && saw_uncond_model
        && saw_clip
        && saw_vae
        && saw_sampler
        && saw_scheduler
        && saw_aura
        && saw_guider
        && saw_cfg_override
        && saw_prompt_encode;
    if !all_present {
        return Err(GraphError::unsupported(
            "Ideogram4 Comfy export is missing required txt2img sampler nodes",
        ));
    }

    let edge_count = wf
        .get("links")
        .and_then(JsonValue::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    crate::record_workflow_execution(
        obj,
        "ideogram4_comfy_ui_export",
        sg_nodes.len(),
        edge_count,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Comfy-UI canvas graph -> typed `{nodes, edges}` body
// ---------------------------------------------------------------------------

/// A ComfyUI node is inactive when bypassed (mode 4) OR muted (mode 2). The Mojo
/// importer only skipped bypass; the graph-runtime audit
/// (`graph_runtime_audit_2026-06-13.raw.json`) flags muted (mode 2) nodes being
/// emitted as live as a real-export parity gap — ComfyUI produces no output for
/// them. We treat BOTH as inactive in the node-emit loop and the node-index
/// lookup, dropping links that touch them (a dropped producer that orphans a
/// required consumer input is then caught by the typed executor's fail-loud
/// missing-input path).
fn canvas_node_inactive(node: &JsonValue) -> bool {
    matches!(node_mode(node), 2 | 4)
}

/// `_comfy_ui_lanpaint_mode` (Mojo 1118): canonicalize the LanPaint mode widget.
fn lanpaint_mode(widgets: &JsonValue, idx: usize) -> String {
    let raw = widget_string(widgets, idx, "");
    let lower = raw.to_lowercase();
    if lower.contains("video") {
        return "video".to_string();
    }
    if lower.contains("image") {
        return "image".to_string();
    }
    raw
}

/// `_comfy_ui_widget_fields` (Mojo 1128): translate a node's positional
/// `widgets_values` into named `fields` keyed by node type.
fn comfy_ui_widget_fields(type_id: &str, widgets: &JsonValue) -> JsonValue {
    let mut fields = Map::new();
    match type_id {
        "CheckpointLoaderSimple" => {
            fields.insert("ckpt_name".into(), json!(widget_string(widgets, 0, "")));
        }
        "UNETLoader" | "DiffusionModelLoader" => {
            fields.insert("unet_name".into(), json!(widget_string(widgets, 0, "")));
        }
        "LoraLoader" | "LoraLoaderModelOnly" | "ZImageLoraModelOnly" => {
            fields.insert("lora_name".into(), json!(widget_string(widgets, 0, "")));
            fields.insert("strength_model".into(), json!(widget_float(widgets, 1, 1.0)));
            if type_id == "LoraLoader" {
                fields.insert("strength_clip".into(), json!(widget_float(widgets, 2, 1.0)));
            }
        }
        "CLIPTextEncode" | "CLIPTextEncodeFlux" | "TextEncodeQwenImageEdit"
        | "TextEncodeQwenImageEditPlus" => {
            fields.insert("text".into(), json!(widget_string(widgets, 0, "")));
        }
        "LoadImage" => {
            fields.insert("image".into(), json!(widget_string(widgets, 0, "")));
        }
        "EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage" => {
            fields.insert("width".into(), json!(widget_int(widgets, 0, 512)));
            fields.insert("height".into(), json!(widget_int(widgets, 1, 512)));
            fields.insert("batch_size".into(), json!(widget_int(widgets, 2, 1)));
        }
        "KSampler" => {
            fields.insert("seed".into(), json!(widget_int(widgets, 0, -1)));
            fields.insert("steps".into(), json!(widget_int(widgets, 2, 20)));
            fields.insert("cfg".into(), json!(widget_float(widgets, 3, 4.5)));
            fields.insert("sampler_name".into(), json!(widget_string(widgets, 4, "euler")));
            fields.insert("scheduler".into(), json!(widget_string(widgets, 5, "simple")));
            fields.insert("denoise".into(), json!(widget_float(widgets, 6, 1.0)));
        }
        "LanPaint_KSampler" => {
            fields.insert("seed".into(), json!(widget_int(widgets, 0, -1)));
            fields.insert("steps".into(), json!(widget_int(widgets, 2, 20)));
            fields.insert("cfg".into(), json!(widget_float(widgets, 3, 4.5)));
            fields.insert("sampler_name".into(), json!(widget_string(widgets, 4, "euler")));
            fields.insert("scheduler".into(), json!(widget_string(widgets, 5, "simple")));
            fields.insert("denoise".into(), json!(widget_float(widgets, 6, 1.0)));
            fields.insert("LanPaint_NumSteps".into(), json!(widget_int(widgets, 7, -1)));
            fields.insert("LanPaint_PromptMode".into(), json!(widget_string(widgets, 8, "")));
            fields.insert("Inpainting_mode".into(), json!(lanpaint_mode(widgets, 10)));
        }
        "LanPaint_KSamplerAdvanced" => {
            fields.insert("add_noise".into(), json!(widget_string(widgets, 0, "")));
            fields.insert("noise_seed".into(), json!(widget_int(widgets, 1, -1)));
            fields.insert("seed".into(), json!(widget_int(widgets, 1, -1)));
            fields.insert("steps".into(), json!(widget_int(widgets, 3, 20)));
            fields.insert("cfg".into(), json!(widget_float(widgets, 4, 4.5)));
            fields.insert("sampler_name".into(), json!(widget_string(widgets, 5, "euler")));
            fields.insert("scheduler".into(), json!(widget_string(widgets, 6, "simple")));
            fields.insert("start_at_step".into(), json!(widget_int(widgets, 7, -1)));
            fields.insert("end_at_step".into(), json!(widget_int(widgets, 8, -1)));
            fields.insert(
                "return_with_leftover_noise".into(),
                json!(widget_string(widgets, 9, "")),
            );
            fields.insert("LanPaint_NumSteps".into(), json!(widget_int(widgets, 10, -1)));
            fields.insert("LanPaint_Lambda".into(), json!(widget_float(widgets, 11, -1.0)));
            fields.insert("LanPaint_StepSize".into(), json!(widget_float(widgets, 12, -1.0)));
            fields.insert("LanPaint_Beta".into(), json!(widget_float(widgets, 13, -1.0)));
            fields.insert("LanPaint_Friction".into(), json!(widget_float(widgets, 14, -1.0)));
            fields.insert("LanPaint_PromptMode".into(), json!(widget_string(widgets, 15, "")));
            fields.insert("LanPaint_EarlyStop".into(), json!(widget_int(widgets, 16, -1)));
            fields.insert("Inpainting_mode".into(), json!(lanpaint_mode(widgets, 18)));
        }
        "LanPaint_SamplerCustomAdvanced" => {
            fields.insert("LanPaint_NumSteps".into(), json!(widget_int(widgets, 0, -1)));
            fields.insert("LanPaint_Lambda".into(), json!(widget_float(widgets, 1, -1.0)));
            fields.insert("LanPaint_StepSize".into(), json!(widget_float(widgets, 2, -1.0)));
            fields.insert("LanPaint_Beta".into(), json!(widget_float(widgets, 3, -1.0)));
            fields.insert("LanPaint_Friction".into(), json!(widget_float(widgets, 4, -1.0)));
            fields.insert("LanPaint_PromptMode".into(), json!(widget_string(widgets, 5, "")));
            fields.insert("LanPaint_EarlyStop".into(), json!(widget_int(widgets, 6, -1)));
            fields.insert("LanPaint_InnerThreshold".into(), json!(widget_float(widgets, 8, -1.0)));
            fields.insert("LanPaint_InnerPatience".into(), json!(widget_int(widgets, 9, -1)));
        }
        "LanPaint_MaskBlend" => {
            fields.insert("blend_overlap".into(), json!(widget_int(widgets, 0, -1)));
        }
        "ImagePadForOutpaint" => {
            fields.insert("left".into(), json!(widget_int(widgets, 0, 0)));
            fields.insert("top".into(), json!(widget_int(widgets, 1, 0)));
            fields.insert("right".into(), json!(widget_int(widgets, 2, 0)));
            fields.insert("bottom".into(), json!(widget_int(widgets, 3, 0)));
            fields.insert("feathering".into(), json!(widget_int(widgets, 4, 40)));
        }
        "ThresholdMask" => {
            fields.insert("value".into(), json!(widget_float(widgets, 0, 0.5)));
        }
        "ConditioningSetMask" => {
            fields.insert("strength".into(), json!(widget_float(widgets, 0, 1.0)));
            fields.insert(
                "set_cond_area".into(),
                json!(widget_string(widgets, 1, "default")),
            );
        }
        "SetNode" | "GetNode" => {
            fields.insert("name".into(), json!(widget_string(widgets, 0, "")));
        }
        _ if is_int_scalar_node(type_id) => {
            fields.insert("value".into(), json!(widget_int(widgets, 0, 0)));
        }
        _ if is_float_scalar_node(type_id) => {
            fields.insert("value".into(), json!(widget_float(widgets, 0, 0.0)));
        }
        "StringConstant" | "StringConstantMultiline" => {
            fields.insert("string".into(), json!(widget_string(widgets, 0, "")));
            if type_id == "StringConstantMultiline" {
                fields.insert("strip_newlines".into(), json!(widget_bool(widgets, 1, true)));
            }
        }
        _ if is_string_scalar_node(type_id) => {
            fields.insert("value".into(), json!(widget_string(widgets, 0, "")));
        }
        _ if is_bool_scalar_node(type_id) => {
            fields.insert("value".into(), json!(widget_bool(widgets, 0, false)));
        }
        "PrimitiveNode" => {
            if let Some(a) = widgets.as_array() {
                if let Some(v) = a.first() {
                    fields.insert("value".into(), v.clone());
                }
            }
        }
        "InpaintModelConditioning" => {
            fields.insert("noise_mask".into(), json!(widget_bool(widgets, 0, true)));
        }
        "SaveImage" => {
            fields.insert(
                "filename_prefix".into(),
                json!(widget_string(widgets, 0, "ComfyUI")),
            );
        }
        "ImageToMask" => {
            fields.insert("channel".into(), json!(widget_string(widgets, 0, "")));
        }
        "RandomNoise" => {
            fields.insert("noise_seed".into(), json!(widget_int(widgets, 0, -1)));
        }
        "KSamplerSelect" => {
            fields.insert("sampler_name".into(), json!(widget_string(widgets, 0, "euler")));
        }
        "Flux2Scheduler" => {
            fields.insert("steps".into(), json!(widget_int(widgets, 0, 20)));
        }
        "BasicScheduler" => {
            fields.insert("scheduler".into(), json!(widget_string(widgets, 0, "simple")));
            fields.insert("steps".into(), json!(widget_int(widgets, 1, 20)));
            fields.insert("denoise".into(), json!(widget_float(widgets, 2, 1.0)));
        }
        "CFGGuider" | "FluxGuidance" => {
            fields.insert("cfg".into(), json!(widget_float(widgets, 0, 4.5)));
        }
        "ModelSamplingAuraFlow" | "ModelSamplingSD3" => {
            fields.insert("shift".into(), json!(widget_float(widgets, 0, 3.0)));
        }
        "ComfySwitchNode" => {
            fields.insert("switch".into(), json!(widget_bool(widgets, 0, false)));
        }
        _ => {}
    }
    JsonValue::Object(fields)
}

/// `_comfy_ui_node_index` (Mojo 1032): locate an active node by id, returning
/// `None` when missing or inactive (mute/bypass). Raises on a non-object node.
fn comfy_ui_node_index(
    nodes: &[JsonValue],
    target_id: i64,
    active_only: bool,
) -> GraphResult<Option<usize>> {
    for (i, node) in nodes.iter().enumerate() {
        if !node.is_object() {
            return Err(GraphError::unsupported(
                "Comfy UI canvas node must be an object",
            ));
        }
        if node_id(node)? == target_id {
            if active_only && canvas_node_inactive(node) {
                return Ok(None);
            }
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// `_comfy_ui_output_port` (Mojo 1044): resolve the typed port a source node
/// produces on `src_slot`.
fn comfy_ui_output_port(nodes: &[JsonValue], src_id: i64, src_slot: i64) -> GraphResult<String> {
    let idx = comfy_ui_node_index(nodes, src_id, true)?.ok_or_else(|| {
        GraphError::unsupported("Comfy UI canvas link references missing active source node")
    })?;
    let node = &nodes[idx];
    let outputs = node
        .get("outputs")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| GraphError::unsupported("Comfy UI canvas source node missing outputs"))?;
    if src_slot < 0 || src_slot as usize >= outputs.len() {
        return Err(GraphError::unsupported(
            "Comfy UI canvas source output slot out of range",
        ));
    }
    let out = &outputs[src_slot as usize];
    let typ = wf_string(out, "type");
    let name = wf_string(out, "name");
    let node_type_id = canonical_type_id(&node_type(node));

    if node_type_id == "SetNode" {
        return Ok("SET".to_string());
    }
    if node_type_id == "GetNode" {
        return Ok("GET".to_string());
    }
    if node_type_id == "Reroute" {
        return Ok("REROUTE".to_string());
    }
    if util_is_scalar(&node_type_id)
        && matches!(typ.as_str(), "INT" | "FLOAT" | "STRING" | "BOOLEAN")
    {
        return Ok(typ);
    }
    if node_type_id == "InpaintModelConditioning" {
        if name == "positive" || name == "negative" {
            return Ok(name);
        }
        if typ == "LATENT" || name == "latent" {
            return Ok("LATENT".to_string());
        }
    }
    if name == "CONDITIONING_1" {
        return Ok(name);
    }
    if typ == "INT" {
        if !name.is_empty() {
            return Ok(name);
        }
        return Ok("INT".to_string());
    }
    if matches!(
        typ.as_str(),
        "MODEL"
            | "CLIP"
            | "VAE"
            | "CONDITIONING"
            | "IMAGE"
            | "MASK"
            | "LATENT"
            | "GUIDER"
            | "SIGMAS"
            | "NOISE"
            | "SAMPLER"
    ) {
        return Ok(typ);
    }
    if !name.is_empty() {
        return Ok(name);
    }
    Err(GraphError::unsupported(
        "Comfy UI canvas source output missing type/name",
    ))
}

/// `_comfy_ui_input_port` (Mojo 1097): resolve the named input port on
/// `dst_slot`.
fn comfy_ui_input_port(nodes: &[JsonValue], dst_id: i64, dst_slot: i64) -> GraphResult<String> {
    let idx = comfy_ui_node_index(nodes, dst_id, true)?.ok_or_else(|| {
        GraphError::unsupported("Comfy UI canvas link references missing active target node")
    })?;
    let node = &nodes[idx];
    let inputs = node
        .get("inputs")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| GraphError::unsupported("Comfy UI canvas target node missing inputs"))?;
    if dst_slot < 0 || dst_slot as usize >= inputs.len() {
        return Err(GraphError::unsupported(
            "Comfy UI canvas target input slot out of range",
        ));
    }
    let port = wf_string(&inputs[dst_slot as usize], "name");
    let node_type_id = canonical_type_id(&node_type(node));
    if node_type_id == "SetNode" {
        return Ok("value".to_string());
    }
    if node_type_id == "Reroute" && port.is_empty() {
        return Ok("input".to_string());
    }
    if port.is_empty() {
        return Err(GraphError::unsupported(
            "Comfy UI canvas target input missing name",
        ));
    }
    Ok(port)
}

/// `comfy_ui_canvas_to_typed_graph` (Mojo 1253): convert a Comfy-UI canvas graph
/// (`nodes` + `links`) into the typed `{nodes, edges}` body the executor runs.
///
/// Audit parity fix: inactive nodes are BOTH bypassed (mode 4) AND muted (mode
/// 2) — see [`canvas_node_inactive`]. Links touching an inactive endpoint are
/// dropped (the index lookup returns `None`).
pub fn comfy_ui_canvas_to_typed_body(wf: &JsonValue) -> GraphResult<JsonValue> {
    let src_nodes = wf
        .get("nodes")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| GraphError::unsupported("Comfy UI canvas missing nodes array"))?;

    let mut nodes = Vec::new();
    for src in src_nodes {
        if !src.is_object() {
            return Err(GraphError::unsupported(
                "Comfy UI canvas node must be an object",
            ));
        }
        if canvas_node_inactive(src) {
            continue;
        }
        let type_id = canonical_type_id(&node_type(src));
        let widgets = node_widgets(src);
        let mut fields = comfy_ui_widget_fields(&type_id, &widgets);

        // GetNode/SetNode/scalar nodes carry their declared output type so the
        // executor can validate the named edge (Mojo 1269-1278).
        if type_id == "GetNode" || type_id == "SetNode" || util_is_scalar(&type_id) {
            if let Some(out0) = src
                .get("outputs")
                .and_then(JsonValue::as_array)
                .and_then(|a| a.first())
                .filter(|v| v.is_object())
            {
                let output_type = wf_string(out0, "type");
                if !output_type.is_empty() {
                    fields
                        .as_object_mut()
                        .unwrap()
                        .insert("output_type".to_string(), json!(output_type));
                }
            }
        }

        nodes.push(json!({
            "id": node_id(src)?,
            "type_id": type_id,
            "fields": fields,
        }));
    }

    let mut edges = Vec::new();
    if let Some(links) = wf.get("links").and_then(JsonValue::as_array) {
        for link in links {
            let arr = match link.as_array() {
                Some(a) if a.len() >= 6 => a,
                _ => continue,
            };
            let src_id = json_intish(&arr[1], "source node id")?;
            let src_slot = json_intish(&arr[2], "source output slot")?;
            let dst_id = json_intish(&arr[3], "target node id")?;
            let dst_slot = json_intish(&arr[4], "target input slot")?;
            if comfy_ui_node_index(src_nodes, src_id, true)?.is_none()
                || comfy_ui_node_index(src_nodes, dst_id, true)?.is_none()
            {
                continue;
            }
            edges.push(json!({
                "from": {
                    "node": src_id,
                    "port": comfy_ui_output_port(src_nodes, src_id, src_slot)?,
                },
                "to": {
                    "node": dst_id,
                    "port": comfy_ui_input_port(src_nodes, dst_id, dst_slot)?,
                },
            }));
        }
    }

    Ok(json!({ "nodes": nodes, "edges": edges }))
}
