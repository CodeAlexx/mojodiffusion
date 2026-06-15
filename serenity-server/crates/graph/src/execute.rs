//! `execute_typed_graph` — the topological executor body of
//! `apply_typed_workflow_graph` (Mojo 1714-2909).
//!
//! AUDIT FIX (ordering): instead of the Mojo rescan-until-stable loop, this uses
//! Kahn-style topological scheduling with PREBUILT maps:
//!   * `link_map: (to_node, to_port) -> WorkflowLink`  (find_input_link, O(1))
//!   * the [`ValueStore`] `(node, port) -> WorkflowValue` map (produced values)
//! A node fires in the first position where all its inputs are resolvable. To
//! preserve the Mojo first-writer-wins (`_set_if_missing`) outcome byte-for-byte
//! we break ties by ARRAY INDEX — the same order the Mojo rescan loop visits
//! nodes — so the same producer wins each scalar key. The result is a proper
//! topological order (every node after its producers) AND parity-faithful.

use std::collections::HashMap;

use serde_json::{json, Value as JsonValue};

use crate::util::*;
use crate::{
    GraphError, GraphResult, TypedGraph, ValuePayload, ValueStore, WorkflowLink, WorkflowNode,
    WorkflowValue,
};

// ---------------------------------------------------------------------------
// flat-output write helpers (mirror Mojo _set_if_missing / _copy_field_if_missing)
// ---------------------------------------------------------------------------

/// `_set_if_missing` (Mojo 113): write `key` only when absent/null.
fn set_if_missing(out: &mut JsonValue, key: &str, value: JsonValue) {
    let o = out.as_object_mut().expect("out is an object");
    let missing = o.get(key).map(JsonValue::is_null).unwrap_or(true);
    if missing {
        o.insert(key.to_string(), value);
    }
}

/// `_copy_field_if_missing` (Mojo 118): copy `src[src_key]` → `dst[dst_key]` when
/// source present/non-null and dst missing.
fn copy_field_if_missing(out: &mut JsonValue, src: &JsonValue, src_key: &str, dst_key: &str) {
    if let Some(o) = src.as_object() {
        if let Some(v) = o.get(src_key) {
            if !v.is_null() {
                set_if_missing(out, dst_key, v.clone());
            }
        }
    }
}

/// `_workflow_set_field_if_nonnegative_int` (Mojo 931).
fn set_field_if_nonneg_int(
    out: &mut JsonValue,
    fields: &JsonValue,
    src_key: &str,
    dst_key: &str,
) -> GraphResult<()> {
    let v = match fields.as_object().and_then(|o| o.get(src_key)) {
        Some(v) if !v.is_null() => v,
        _ => return Ok(()),
    };
    let n = v.as_i64().ok_or_else(|| {
        GraphError::unsupported(format!("workflow graph field {src_key} must be an integer"))
    })?;
    if n >= 0 {
        set_if_missing(out, dst_key, json!(n));
    }
    Ok(())
}

/// `_workflow_append_lora` (Mojo 176): append `{name, weight}` to `out["lora"]`,
/// skipping zero-weight; the float weight is emitted as JSON number.
fn append_lora(out: &mut JsonValue, name: &str, weight: f64) -> GraphResult<()> {
    if name.is_empty() {
        return Err(GraphError::unsupported(
            "workflow graph LoRA loader missing lora_name",
        ));
    }
    if weight == 0.0 {
        return Ok(());
    }
    let o = out.as_object_mut().expect("out is an object");
    let arr = match o.get("lora") {
        Some(JsonValue::Null) | None => Vec::new(),
        Some(JsonValue::Array(a)) => a.clone(),
        Some(_) => {
            return Err(GraphError::unsupported(
                "workflow graph lora metadata must be an array",
            ))
        }
    };
    let mut arr = arr;
    arr.push(json!({ "name": name, "weight": weight }));
    o.insert("lora".to_string(), JsonValue::Array(arr));
    Ok(())
}

/// `_workflow_copy_lanpaint_field_alias` (Mojo 975).
fn copy_lanpaint_alias(out: &mut JsonValue, fields: &JsonValue, src_key: &str, dst_key: &str) {
    copy_field_if_missing(out, fields, src_key, dst_key);
    copy_field_if_missing(out, fields, dst_key, dst_key);
}

/// `_workflow_copy_lanpaint_sampler_fields` (Mojo 982).
fn copy_lanpaint_sampler_fields(out: &mut JsonValue, fields: &JsonValue) {
    copy_lanpaint_alias(out, fields, "LanPaint_NumSteps", "lanpaint_num_steps");
    copy_lanpaint_alias(out, fields, "LanPaint_Lambda", "lanpaint_lambda");
    copy_lanpaint_alias(out, fields, "LanPaint_StepSize", "lanpaint_step_size");
    copy_lanpaint_alias(out, fields, "LanPaint_Beta", "lanpaint_beta");
    copy_lanpaint_alias(out, fields, "LanPaint_Friction", "lanpaint_friction");
    copy_lanpaint_alias(out, fields, "LanPaint_PromptMode", "lanpaint_prompt_mode");
    copy_lanpaint_alias(out, fields, "Inpainting_mode", "lanpaint_inpainting_mode");
    copy_lanpaint_alias(out, fields, "add_noise", "lanpaint_add_noise");
    copy_lanpaint_alias(out, fields, "noise_seed", "lanpaint_noise_seed");
}

// ---------------------------------------------------------------------------
// link map (audit fix: prebuilt (to_node, to_port) -> link)
// ---------------------------------------------------------------------------

/// Prebuilt input-link index. Built once from the edge list; each
/// `find_input_link` is then an O(1) map lookup. Multiple sources for one input
/// is a 422 (Mojo 564), detected at build time.
struct LinkMap {
    direct: HashMap<(i64, String), WorkflowLink>,
    /// All (to_node, to_port) targets, used by the Reroute/SetNode wildcard
    /// resolvers that match ANY accepted port on a node.
    by_node: HashMap<i64, Vec<(String, WorkflowLink)>>,
}

impl LinkMap {
    fn build(graph: &TypedGraph) -> GraphResult<Self> {
        let mut direct: HashMap<(i64, String), WorkflowLink> = HashMap::new();
        let mut by_node: HashMap<i64, Vec<(String, WorkflowLink)>> = HashMap::new();
        for edge in &graph.edges {
            let key = (edge.to.node, edge.to.port.clone());
            let link = WorkflowLink::found(edge.from.node, edge.from.port.clone());
            if direct.contains_key(&key) {
                return Err(GraphError::bad_request(format!(
                    "workflow graph input has multiple sources: {}",
                    edge.to.port
                ))
                .with_node(edge.to.node));
            }
            direct.insert(key, link.clone());
            by_node
                .entry(edge.to.node)
                .or_default()
                .push((edge.to.port.clone(), link));
        }
        Ok(LinkMap { direct, by_node })
    }

    /// `_workflow_find_input_link` (Mojo 553).
    fn input(&self, to_node: i64, to_port: &str) -> WorkflowLink {
        self.direct
            .get(&(to_node, to_port.to_string()))
            .cloned()
            .unwrap_or_else(WorkflowLink::none)
    }

    /// `_workflow_find_reroute_input_link` (Mojo 570): first edge into `to_node`
    /// whose port is one of input/""/"*"/"reroute"; multiple → 422.
    fn reroute_input(&self, to_node: i64) -> GraphResult<WorkflowLink> {
        let mut out = WorkflowLink::none();
        if let Some(targets) = self.by_node.get(&to_node) {
            for (port, link) in targets {
                if matches!(port.as_str(), "input" | "" | "*" | "reroute") {
                    if out.found {
                        return Err(GraphError::bad_request(
                            "workflow graph Reroute input has multiple sources",
                        )
                        .with_node(to_node));
                    }
                    out = link.clone();
                }
            }
        }
        Ok(out)
    }

    /// `_workflow_find_setnode_input_link` (Mojo 590): the bus-typed input edge.
    fn setnode_input(&self, to_node: i64) -> GraphResult<WorkflowLink> {
        let mut out = WorkflowLink::none();
        if let Some(targets) = self.by_node.get(&to_node) {
            for (port, link) in targets {
                let accepted = matches!(
                    port.as_str(),
                    "value" | "input" | "" | "*" | "MODEL" | "CLIP" | "VAE" | "CONDITIONING"
                        | "IMAGE" | "MASK" | "LATENT" | "GUIDER" | "SIGMAS" | "NOISE" | "SAMPLER"
                        | "INT" | "FLOAT" | "STRING" | "BOOLEAN"
                );
                if accepted {
                    if out.found {
                        return Err(GraphError::bad_request(
                            "workflow graph SetNode input has multiple sources",
                        )
                        .with_node(to_node));
                    }
                    out = link.clone();
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// payload accessors (replace the Mojo per-type side-table fetches)
// ---------------------------------------------------------------------------

/// `_workflow_require_value_type` (Mojo 772): resolved + type match, with the
/// COND_LATENT composite accepted for CONDITIONING/LATENT.
fn require_value_type(
    store: &ValueStore,
    link: &WorkflowLink,
    expected: &str,
    input_name: &str,
) -> GraphResult<()> {
    let v = store.get(link.node_id, &link.port).ok_or_else(|| {
        GraphError::bad_request(format!("workflow graph unresolved input: {input_name}"))
            .with_node(link.node_id)
    })?;
    let actual = &v.typ;
    let composite_ok =
        actual == "COND_LATENT" && (expected == "CONDITIONING" || expected == "LATENT");
    if actual != expected && !composite_ok {
        return Err(GraphError::bad_request(format!(
            "workflow graph input {input_name} expected {expected} from {}:{} but got {actual}",
            link.node_id, link.port
        ))
        .with_node(link.node_id));
    }
    Ok(())
}

fn model_name_of(store: &ValueStore, link: &WorkflowLink) -> GraphResult<String> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::Model { name }) => Ok(name.clone()),
        _ => Err(GraphError::bad_request("workflow graph model handle missing source")
            .with_node(link.node_id)),
    }
}

fn cond_text_of(store: &ValueStore, link: &WorkflowLink) -> GraphResult<String> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::Cond { text }) => Ok(text.clone()),
        _ => Err(
            GraphError::bad_request("workflow graph conditioning handle missing source")
                .with_node(link.node_id),
        ),
    }
}

fn image_path_of(store: &ValueStore, link: &WorkflowLink) -> GraphResult<String> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::Image { path, .. }) => Ok(path.clone()),
        Some(ValuePayload::Mask { path, .. }) => Ok(path.clone()),
        _ => Err(GraphError::bad_request("workflow graph image handle missing source")
            .with_node(link.node_id)),
    }
}

fn image_mask_source_of(store: &ValueStore, link: &WorkflowLink) -> GraphResult<String> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::Image { mask_source, .. }) => Ok(mask_source.clone().unwrap_or_default()),
        Some(ValuePayload::Mask { source, .. }) => Ok(source.clone().unwrap_or_default()),
        _ => Err(GraphError::bad_request("workflow graph handle metadata missing source")
            .with_node(link.node_id)),
    }
}

fn mask_source_of(store: &ValueStore, link: &WorkflowLink) -> GraphResult<String> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::Mask { source, .. }) => Ok(source.clone().unwrap_or_default()),
        Some(ValuePayload::Image { mask_source, .. }) => Ok(mask_source.clone().unwrap_or_default()),
        _ => Err(GraphError::bad_request("workflow graph handle metadata missing source")
            .with_node(link.node_id)),
    }
}

/// Read a scalar payload of the required type (Mojo _workflow_scalar_*).
fn scalar_int_of(store: &ValueStore, link: &WorkflowLink, name: &str) -> GraphResult<i64> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::ScalarInt(i)) => Ok(*i),
        _ => Err(scalar_type_err(link, "INT", name)),
    }
}
fn scalar_float_of(store: &ValueStore, link: &WorkflowLink, name: &str) -> GraphResult<f64> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::ScalarFloat(f)) => Ok(*f),
        _ => Err(scalar_type_err(link, "FLOAT", name)),
    }
}
fn scalar_string_of(store: &ValueStore, link: &WorkflowLink, name: &str) -> GraphResult<String> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::ScalarString(s)) => Ok(s.clone()),
        _ => Err(scalar_type_err(link, "STRING", name)),
    }
}
fn scalar_bool_of(store: &ValueStore, link: &WorkflowLink, name: &str) -> GraphResult<bool> {
    match store.get(link.node_id, &link.port).map(|v| &v.payload) {
        Some(ValuePayload::ScalarBool(b)) => Ok(*b),
        _ => Err(scalar_type_err(link, "BOOLEAN", name)),
    }
}
fn scalar_type_err(link: &WorkflowLink, expected: &str, name: &str) -> GraphError {
    GraphError::bad_request(format!(
        "workflow graph input {name} expected {expected} from {}:{} (scalar metadata)",
        link.node_id, link.port
    ))
    .with_node(link.node_id)
}

/// True if a value is present on the link target (Mojo _workflow_value_index>=0).
fn ready(store: &ValueStore, link: &WorkflowLink) -> bool {
    store.contains(link.node_id, &link.port)
}

/// `_workflow_optional_link_ready` (Mojo 764): absent link is "ready".
fn optional_ready(store: &ValueStore, link: &WorkflowLink) -> bool {
    !link.found || ready(store, link)
}

/// Look up the LATENT geometry payload behind a link (the Mojo latent_* row).
///
/// Resolution order mirrors the Mojo `latent_*` side-table:
/// 1. If the handle's intrinsic payload is a `Latent`, use it.
/// 2. Otherwise fall back to the explicit latent-geometry side-table, which is
///    populated only for a `ReferenceLatent` whose `CONDITIONING` output port
///    also carries copied latent geometry (Mojo workflow_graph.mojo:2431-2440).
/// This is what lets a downstream sampler reading a `ReferenceLatent`'s
/// `latent_image` re-emit the source latent's `init_image`/`mask_image`.
fn latent_payload<'a>(store: &'a ValueStore, link: &WorkflowLink) -> Option<&'a ValuePayload> {
    if let Some(v) = store.get(link.node_id, &link.port) {
        if matches!(v.payload, ValuePayload::Latent { .. }) {
            return Some(&v.payload);
        }
    }
    store.get_latent_geom(link.node_id, &link.port)
}

// ---------------------------------------------------------------------------
// the executor
// ---------------------------------------------------------------------------

/// Outcome of attempting to fire a node: it either progressed (deps were ready)
/// or it is still waiting on an upstream value.
enum Fire {
    Done,
    NotReady,
}

/// Execute a [`TypedGraph`] into a [`ValueStore`], writing flat backend params
/// into `out`. Port of the executor body of `apply_typed_workflow_graph`
/// (Mojo 1714-2909) with Kahn array-ordered scheduling (see module docs).
pub fn execute_typed_graph(graph: &TypedGraph, out: &mut JsonValue) -> GraphResult<ValueStore> {
    if !out.is_object() {
        *out = JsonValue::Object(Default::default());
    }
    let links = LinkMap::build(graph)?;
    let mut store = ValueStore::new();

    let n = graph.nodes.len();
    let mut done = vec![false; n];
    let mut remaining = n;
    let mut saw_prompt = false;
    let mut reference_latent_count: i64 = 0;

    // Kahn topo: each outer pass scans nodes in ARRAY order, firing the first
    // not-yet-done node whose inputs resolve (first-writer-wins parity tiebreak).
    while remaining > 0 {
        let mut progressed = false;
        for i in 0..n {
            if done[i] {
                continue;
            }
            let node = &graph.nodes[i];
            let fired = exec_node(
                node,
                &links,
                &mut store,
                out,
                &mut saw_prompt,
                &mut reference_latent_count,
            )?;
            if let Fire::Done = fired {
                done[i] = true;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            return Err(GraphError::bad_request(
                "workflow graph has unresolved or cyclic typed links",
            ));
        }
    }

    let prompt_missing = out
        .as_object()
        .and_then(|o| o.get("prompt"))
        .map(JsonValue::is_null)
        .unwrap_or(true);
    if !saw_prompt && prompt_missing {
        return Err(GraphError::unsupported(
            "workflow graph did not contain a prompt node",
        ));
    }
    if reference_latent_count > 0 {
        set_if_missing(out, "reference_latent_count", json!(reference_latent_count));
    }
    Ok(store)
}

/// Insert a produced value; duplicate output on (node, port) is a 422
/// (Mojo `_workflow_add_value`).
fn add_value(store: &mut ValueStore, node_id: i64, port: &str, payload: ValuePayload) -> GraphResult<()> {
    if store.contains(node_id, port) {
        return Err(GraphError::bad_request("workflow graph duplicate output value")
            .with_node(node_id));
    }
    store.insert(WorkflowValue::new(node_id, port, payload));
    Ok(())
}

/// Insert a typed value with an explicit `typ` override (for COND_LATENT etc.,
/// where the carried type string differs from the payload's intrinsic type_id).
fn add_value_typed(
    store: &mut ValueStore,
    node_id: i64,
    port: &str,
    typ: &str,
    payload: ValuePayload,
) -> GraphResult<()> {
    if store.contains(node_id, port) {
        return Err(GraphError::bad_request("workflow graph duplicate output value")
            .with_node(node_id));
    }
    store.insert(WorkflowValue {
        node_id,
        port: port.to_string(),
        typ: typ.to_string(),
        payload,
    });
    Ok(())
}

/// Execute one node (faithful per-type handler dispatch). Returns whether it
/// progressed. The big `match` mirrors the Mojo `if/elif` chain 1793-2900.
fn exec_node(
    node: &WorkflowNode,
    links: &LinkMap,
    store: &mut ValueStore,
    out: &mut JsonValue,
    saw_prompt: &mut bool,
    reference_latent_count: &mut i64,
) -> GraphResult<Fire> {
    let id = node.id;
    let t = node.type_id.as_str();
    let fields = &node.fields;

    // Scalar producer nodes (Mojo 1852).
    if is_scalar_node(t) {
        return exec_scalar(node, store);
    }

    match t {
        "CheckpointLoaderSimple" => {
            let model_name = loader_model_name(fields);
            if !model_name.is_empty() {
                set_if_missing(out, "model", json!(model_name));
            }
            add_value(store, id, "MODEL", ValuePayload::Model { name: model_name })?;
            add_value(store, id, "CLIP", ValuePayload::Clip)?;
            add_value(store, id, "VAE", ValuePayload::Vae)?;
            Ok(Fire::Done)
        }
        "UNETLoader" | "DiffusionModelLoader" => {
            let model_name = loader_model_name(fields);
            if !model_name.is_empty() {
                set_if_missing(out, "model", json!(model_name));
            }
            add_value(store, id, "MODEL", ValuePayload::Model { name: model_name })?;
            Ok(Fire::Done)
        }
        "LoraLoader" | "LoraLoaderModelOnly" | "ZImageLoraModelOnly" => {
            let model_link = links.input(id, "model");
            if !model_link.found {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} missing model input"
                ))
                .with_node(id));
            }
            let clip_link = links.input(id, "clip");
            if t == "LoraLoader" && !clip_link.found {
                return Err(GraphError::unsupported(
                    "workflow graph LoraLoader missing clip input",
                )
                .with_node(id));
            }
            let mut is_ready = ready(store, &model_link);
            if t == "LoraLoader" {
                is_ready = is_ready && ready(store, &clip_link);
            }
            if !is_ready {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &model_link, "MODEL", "model")?;
            if t == "LoraLoader" {
                require_value_type(store, &clip_link, "CLIP", "clip")?;
            }
            let model_name = model_name_of(store, &model_link)?;
            if !model_name.is_empty() {
                set_if_missing(out, "model", json!(model_name));
            }
            let lora_name = wf_string(fields, "lora_name");
            let strength = wf_float(fields, "strength_model", 1.0, -10.0, 10.0)?;
            let mut strength_clip = 0.0;
            if t == "LoraLoader" {
                strength_clip = wf_float(fields, "strength_clip", 1.0, -10.0, 10.0)?;
            }
            append_lora(out, &lora_name, strength)?;
            add_value(store, id, "MODEL", ValuePayload::Model { name: model_name })?;
            if t == "LoraLoader" {
                if strength_clip == 0.0 {
                    add_value(store, id, "CLIP", ValuePayload::Clip)?;
                } else {
                    add_value_typed(store, id, "CLIP", "CLIP_LORA_UNSUPPORTED", ValuePayload::Clip)?;
                }
            }
            Ok(Fire::Done)
        }
        "CLIPLoader" | "DualCLIPLoader" | "TripleCLIPLoader" => {
            add_value(store, id, "CLIP", ValuePayload::Clip)?;
            Ok(Fire::Done)
        }
        "VAELoader" => {
            add_value(store, id, "VAE", ValuePayload::Vae)?;
            Ok(Fire::Done)
        }
        "EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage" => {
            exec_empty_latent(node, links, store, out)
        }
        "CLIPTextEncode" | "CLIPTextEncodeFlux" => {
            let clip_link = links.input(id, "clip");
            let text_link = links.input(id, "text");
            if !clip_link.found {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} missing clip input"
                ))
                .with_node(id));
            }
            if !(ready(store, &clip_link) && optional_ready(store, &text_link)) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &clip_link, "CLIP", "clip")?;
            let mut text = conditioning_prompt_text(fields, t);
            if text_link.found {
                text = scalar_string_of(store, &text_link, "text")?;
            }
            add_value(store, id, "CONDITIONING", ValuePayload::Cond { text })?;
            Ok(Fire::Done)
        }
        "TextEncodeQwenImageEdit" | "TextEncodeQwenImageEditPlus" => {
            let clip_link = links.input(id, "clip");
            let image_link = links.input(id, "image");
            let text_link = links.input(id, "text");
            if !clip_link.found || !image_link.found {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} missing required typed input"
                ))
                .with_node(id));
            }
            if !(ready(store, &clip_link)
                && ready(store, &image_link)
                && optional_ready(store, &text_link))
            {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &clip_link, "CLIP", "clip")?;
            require_value_type(store, &image_link, "IMAGE", "image")?;
            let mut text = conditioning_prompt_text(fields, t);
            if text_link.found {
                text = scalar_string_of(store, &text_link, "text")?;
            }
            let edit_image = image_path_of(store, &image_link)?;
            set_if_missing(out, "qwen_edit_conditioning_image", json!(edit_image));
            add_value(store, id, "CONDITIONING", ValuePayload::Cond { text })?;
            Ok(Fire::Done)
        }
        "ConditioningZeroOut" => {
            let cond_link = links.input(id, "conditioning");
            if !cond_link.found {
                return Err(GraphError::unsupported(
                    "workflow graph ConditioningZeroOut missing conditioning input",
                )
                .with_node(id));
            }
            if !ready(store, &cond_link) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &cond_link, "CONDITIONING", "conditioning")?;
            add_value(store, id, "CONDITIONING", ValuePayload::Cond { text: String::new() })?;
            Ok(Fire::Done)
        }
        "ConditioningSetMask" => exec_conditioning_set_mask(node, links, store, out),
        "ConditioningConcat" => {
            // Comfy ConditioningConcat(conditioning_to, conditioning_from): the two
            // tensors are concatenated along the token axis. In this text-only
            // conditioning model a CONDITIONING handle carries prompt text, so the
            // faithful flat lowering is to join the two prompts: "to, from".
            let to_link = links.input(id, "conditioning_to");
            let from_link = links.input(id, "conditioning_from");
            if !to_link.found || !from_link.found {
                return Err(GraphError::unsupported(
                    "workflow graph ConditioningConcat missing required typed input",
                )
                .with_node(id));
            }
            if !(ready(store, &to_link) && ready(store, &from_link)) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &to_link, "CONDITIONING", "conditioning_to")?;
            require_value_type(store, &from_link, "CONDITIONING", "conditioning_from")?;
            let to_text = cond_text_of(store, &to_link)?;
            let from_text = cond_text_of(store, &from_link)?;
            let text = if to_text.is_empty() {
                from_text
            } else if from_text.is_empty() {
                to_text
            } else {
                format!("{to_text}, {from_text}")
            };
            add_value(store, id, "CONDITIONING", ValuePayload::Cond { text })?;
            Ok(Fire::Done)
        }
        "ConditioningCombine" | "ConditioningAverage" => {
            // Comfy ConditioningCombine batches both conditionings; ConditioningAverage
            // blends them by `conditioning_to_strength`. Neither is representable in a
            // single text prompt WITHOUT silently dropping the second conditioning (and
            // the blend weight) — which would render a subtly wrong image. Fail loud
            // instead; use ConditioningConcat to JOIN two prompts into one.
            Err(GraphError::unsupported(format!(
                "workflow graph {t} cannot be lowered to a single text prompt \
                 (it would silently drop the second conditioning / blend weight); \
                 use ConditioningConcat to join prompts"
            ))
            .with_node(id))
        }
        "FluxGuidance" => {
            let cond_link = links.input(id, "conditioning");
            if !cond_link.found {
                return Err(GraphError::unsupported(
                    "workflow graph FluxGuidance missing conditioning input",
                )
                .with_node(id));
            }
            if !ready(store, &cond_link) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &cond_link, "CONDITIONING", "conditioning")?;
            let text = cond_text_of(store, &cond_link)?;
            copy_field_if_missing(out, fields, "cfg", "cfg");
            add_value(store, id, "CONDITIONING", ValuePayload::Cond { text })?;
            Ok(Fire::Done)
        }
        "ModelSamplingAuraFlow" | "ModelSamplingSD3" | "DifferentialDiffusion" => {
            let model_link = links.input(id, "model");
            if !model_link.found {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} missing model input"
                ))
                .with_node(id));
            }
            if !ready(store, &model_link) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &model_link, "MODEL", "model")?;
            copy_field_if_missing(out, fields, "shift", "sigma_shift");
            let model_name = model_name_of(store, &model_link)?;
            add_value(store, id, "MODEL", ValuePayload::Model { name: model_name })?;
            Ok(Fire::Done)
        }
        "KSampler" | "KSamplerAdvanced" | "LanPaint_KSampler"
        | "LanPaint_KSamplerAdvanced" => {
            exec_ksampler(node, links, store, out, saw_prompt)
        }
        "KSamplerSelect" => {
            let mut sampler_name = wf_string(fields, "sampler_name");
            if sampler_name.is_empty() {
                sampler_name = wf_string(fields, "sampler");
            }
            if sampler_name.is_empty() {
                sampler_name = "euler".to_string();
            }
            set_if_missing(out, "sampler", json!(sampler_name));
            add_value(store, id, "SAMPLER", ValuePayload::Sampler { name: sampler_name })?;
            Ok(Fire::Done)
        }
        _ if crate::is_named_sampler_node(t) => {
            // Named SAMPLER producer: the sampler name is the node TYPE. Gate
            // against the worker's supported list; an unsupported name fails loud
            // rather than substituting a different sampler.
            let named = crate::named_sampler_name(t).to_string();
            if !crate::worker_supports_sampler(&named) {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} lowers to unsupported sampler '{named}'; \
                     the worker supports only euler/flowmatch_euler/dpmpp_2m/uni_pc/uni_pc_bh2"
                ))
                .with_node(id));
            }
            set_if_missing(out, "sampler", json!(named));
            add_value(store, id, "SAMPLER", ValuePayload::Sampler { name: named })?;
            Ok(Fire::Done)
        }
        _ if crate::is_named_scheduler_node(t) => exec_named_scheduler(node, links, store),
        "SamplerCustom" => exec_sampler_custom(node, links, store, out, saw_prompt),
        "RandomNoise" => {
            let seed_link = links.input(id, "noise_seed");
            if !optional_ready(store, &seed_link) {
                return Ok(Fire::NotReady);
            }
            let mut seed = opt_int(fields, "noise_seed", 0, 0, 4294967295)?;
            if seed_link.found {
                seed = scalar_int_of(store, &seed_link, "noise_seed")?;
            }
            if !(0..=4294967295).contains(&seed) {
                return Err(GraphError::unsupported(
                    "workflow graph RandomNoise scalar noise_seed out of range",
                )
                .with_node(id));
            }
            set_if_missing(out, "seed", json!(seed));
            add_value(store, id, "NOISE", ValuePayload::Noise { seed })?;
            Ok(Fire::Done)
        }
        "CFGGuider" => exec_cfg_guider(node, links, store, out, saw_prompt),
        "BasicGuider" => exec_basic_guider(node, links, store, out, saw_prompt),
        "Flux2Scheduler" => exec_flux2_scheduler(node, links, store, out),
        "BasicScheduler" => exec_basic_scheduler(node, links, store, out),
        "SamplerCustomAdvanced" | "LanPaint_SamplerCustomAdvanced" => {
            exec_sampler_custom_advanced(node, links, store, out)
        }
        "LoadImage" | "LoadImageOutput" | "LoadImageMask" => {
            // LoadImageOutput / LoadImageMask are aliases of LoadImage: they load
            // an image file and expose both IMAGE and MASK outputs that resolve to
            // init_image / mask_image downstream. Same path-resolution + handles.
            let mut image_path = wf_string(fields, "image");
            if image_path.is_empty() {
                image_path = wf_string(fields, "path");
            }
            if image_path.is_empty() {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} missing image path"
                ))
                .with_node(id));
            }
            add_value(
                store,
                id,
                "IMAGE",
                ValuePayload::Image { path: image_path.clone(), mask_source: None },
            )?;
            add_value(
                store,
                id,
                "MASK",
                ValuePayload::Mask {
                    path: image_path,
                    source: Some("load_image_mask".to_string()),
                },
            )?;
            Ok(Fire::Done)
        }
        "ImageToMask" => exec_image_to_mask(node, links, store, out),
        "MaskToImage" => exec_mask_to_image(node, links, store),
        "ThresholdMask" => exec_threshold_mask(node, links, store, out),
        "GetImageSize" | "GetImageSizeAndCount" => {
            // GetImageSizeAndCount (KJ) is GetImageSize plus a leading IMAGE
            // passthrough output and a `count` (batch) output. In the flat
            // single-image model the width/height are not resolvable (LoadImage
            // carries no dims), so the INT outputs are placeholders; the count is
            // the constant single-image batch of 1.
            let and_count = t == "GetImageSizeAndCount";
            let image_link = links.input(id, "image");
            if !image_link.found {
                return Err(GraphError::unsupported(format!(
                    "workflow graph {t} missing image input"
                ))
                .with_node(id));
            }
            if !ready(store, &image_link) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &image_link, "IMAGE", "image")?;
            if and_count {
                // Slot 0 is the unmodified IMAGE passthrough.
                let image_path = image_path_of(store, &image_link)?;
                let mask_source = image_mask_source_of(store, &image_link)?;
                add_value(
                    store,
                    id,
                    "IMAGE",
                    ValuePayload::Image { path: image_path, mask_source: opt_nonempty(&mask_source) },
                )?;
            }
            add_value(store, id, "width", ValuePayload::ScalarInt(0))?;
            add_value(store, id, "height", ValuePayload::ScalarInt(0))?;
            // GetImageSize: batch_size placeholder 0 (parity with the existing
            // handler). GetImageSizeAndCount: the count is the constant 1 of the
            // single-image model.
            let batch = if and_count { 1 } else { 0 };
            add_value(store, id, "batch_size", ValuePayload::ScalarInt(batch))?;
            Ok(Fire::Done)
        }
        "ImageScale" | "ImageScaleToTotalPixels" => exec_image_scale(node, links, store),
        "ImageScaleBy" => exec_image_scale_by(node, links, store),
        "ImageResizeKJ" => exec_image_resize_kj(node, links, store, out),
        "ImagePadForOutpaint" => exec_image_pad(node, links, store, out),
        "VAEEncode" => exec_vae_encode(node, links, store),
        "VAEEncodeForInpaint" => exec_vae_encode_for_inpaint(node, links, store, out),
        "RepeatLatentBatch" => exec_repeat_latent_batch(node, links, store, out),
        "SetLatentNoiseMask" => exec_set_latent_noise_mask(node, links, store, out),
        "InpaintModelConditioning" => exec_inpaint_model_conditioning(node, links, store, out),
        "f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca" => exec_lanpaint_preproc(node, links, store, out),
        "ReferenceLatent" => exec_reference_latent(node, links, store, out, reference_latent_count),
        "6007e698-2ebd-4917-84d8-299b35d7b7ab" => {
            exec_reference_conditioning(node, links, store, out, reference_latent_count)
        }
        "SetNode" => exec_setnode(node, links, store),
        "GetNode" => exec_getnode(node, store),
        "Reroute" => exec_reroute(node, links, store),
        "ComfySwitchNode" => exec_switch(node, links, store),
        "LanPaint_MaskBlend" => exec_mask_blend(node, links, store, out),
        "VAEDecode" => {
            let samples_link = links.input(id, "samples");
            let vae_link = links.input(id, "vae");
            if !samples_link.found || !vae_link.found {
                return Err(GraphError::unsupported(
                    "workflow graph VAEDecode missing required typed input",
                )
                .with_node(id));
            }
            if !(ready(store, &samples_link) && ready(store, &vae_link)) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &samples_link, "LATENT", "samples")?;
            require_value_type(store, &vae_link, "VAE", "vae")?;
            add_value(
                store,
                id,
                "IMAGE",
                ValuePayload::Image { path: String::new(), mask_source: None },
            )?;
            Ok(Fire::Done)
        }
        "SaveImage" => {
            let image_link = links.input(id, "images");
            let prefix_link = links.input(id, "filename_prefix");
            if !image_link.found {
                return Err(GraphError::unsupported(
                    "workflow graph SaveImage missing images input",
                )
                .with_node(id));
            }
            if !(ready(store, &image_link) && optional_ready(store, &prefix_link)) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &image_link, "IMAGE", "images")?;
            let mut prefix = wf_string(fields, "filename_prefix");
            if prefix_link.found {
                prefix = scalar_string_of(store, &prefix_link, "filename_prefix")?;
            }
            if !prefix.is_empty() {
                set_if_missing(out, "workflow_save_prefix", json!(prefix));
            }
            Ok(Fire::Done)
        }
        "PreviewImage" => {
            let mut image_link = links.input(id, "images");
            if !image_link.found {
                image_link = links.input(id, "image");
            }
            if !image_link.found {
                return Ok(Fire::Done);
            }
            if !ready(store, &image_link) {
                return Ok(Fire::NotReady);
            }
            require_value_type(store, &image_link, "IMAGE", "images")?;
            Ok(Fire::Done)
        }
        "MarkdownNote" | "Note" => Ok(Fire::Done),
        other => Err(GraphError::unsupported(format!(
            "unsupported workflow graph node type: {other}"
        ))
        .with_node(id)
        .with_type(other)),
    }
}

// --- scalar producer (Mojo 1852-1903) ------------------------------------------

fn exec_scalar(node: &WorkflowNode, store: &mut ValueStore) -> GraphResult<Fire> {
    let t = node.type_id.as_str();
    let fields = &node.fields;
    let scalar_type = scalar_output_type(t, fields);
    if scalar_type.is_empty() {
        return Err(GraphError::unsupported(
            "workflow graph primitive scalar missing supported output type",
        )
        .with_node(node.id));
    }
    let payload = match scalar_type.as_str() {
        "INT" => {
            let i = if fields.as_object().map(|o| o.contains_key("value")).unwrap_or(false) {
                wf_int(fields, "value", 0)?
            } else if fields.as_object().map(|o| o.contains_key("seed")).unwrap_or(false) {
                wf_int(fields, "seed", 0)?
            } else {
                0
            };
            ValuePayload::ScalarInt(i)
        }
        "FLOAT" => {
            let mut f = if fields.as_object().map(|o| o.contains_key("value")).unwrap_or(false) {
                wf_float(fields, "value", 0.0, -1.0e308, 1.0e308)?
            } else {
                0.0
            };
            if t == "FloatConstant" {
                f = round6(f);
            }
            ValuePayload::ScalarFloat(f)
        }
        "STRING" => {
            let mut s = wf_string(fields, "value");
            if s.is_empty() {
                s = wf_string(fields, "text");
            }
            if s.is_empty() {
                s = wf_string(fields, "string");
            }
            if t == "StringConstantMultiline" && wf_bool(fields, "strip_newlines", true)? {
                if s.contains('\n')
                    || s.contains('\r')
                    || s.starts_with(' ')
                    || s.starts_with('\t')
                    || s.ends_with(' ')
                    || s.ends_with('\t')
                {
                    return Err(GraphError::unsupported(
                        "workflow graph StringConstantMultiline strip_newlines transform is unsupported",
                    )
                    .with_node(node.id));
                }
            }
            ValuePayload::ScalarString(s)
        }
        "BOOLEAN" => ValuePayload::ScalarBool(wf_bool(fields, "value", false)?),
        _ => unreachable!(),
    };
    add_value(store, node.id, &scalar_type, payload)?;
    Ok(Fire::Done)
}

/// `_workflow_int` (Mojo 147): int field accepting int/number/parseable-string.
fn wf_int(obj: &JsonValue, key: &str, dflt: i64) -> GraphResult<i64> {
    let o = match obj.as_object() {
        Some(o) => o,
        None => return Ok(dflt),
    };
    let v = match o.get(key) {
        Some(v) if !v.is_null() => v,
        _ => return Ok(dflt),
    };
    if let Some(i) = v.as_i64() {
        Ok(i)
    } else if let Some(f) = v.as_f64() {
        Ok(f as i64)
    } else if let Some(s) = v.as_str() {
        s.parse::<i64>().map_err(|_| {
            GraphError::unsupported(format!("workflow graph field {key} must be an integer"))
        })
    } else {
        Err(GraphError::unsupported(format!(
            "workflow graph field {key} must be an integer"
        )))
    }
}

include!("execute_handlers.rs");
