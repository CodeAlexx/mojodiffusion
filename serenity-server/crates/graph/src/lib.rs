//! Serenity workflow-graph lowering — Rust port of
//! `serenitymojo/serve/workflow_graph.mojo`.
//!
//! This crate is the FROZEN foundation for the workflow-graph executor in the
//! Rust control plane. It mirrors `apply_workflow_params` (Mojo entry at line
//! 2912): read `req["workflow"]`, lower the typed graph, and write flat
//! `params` keys back into `req`.
//!
//! Two audit fixes are baked into the type design:
//!
//! 1. **Single typed value store** (replaces the Mojo ~30 parallel `List`
//!    side-tables: `value_nodes`/`value_ports`/`value_types`/`model_names`/
//!    `cond_texts`/`image_paths`/`latent_widths`/`scalar_bools`/…). In Mojo a
//!    typed handle (`node_id`,`port`,`typ`) was just a string `typ`, and the
//!    *payload* lived in a separate side-table indexed by `(node_id, port)`.
//!    Those could desync — e.g. `GetImageSize` produced an `INT` handle whose
//!    scalar value lived in a different list and could be missing/stale. Here a
//!    [`WorkflowValue`] ALWAYS carries a resolvable [`ValuePayload`], so a typed
//!    handle can never reference a missing payload.
//!
//! 2. **Structured error** (replaces the `"[501] …"` string sentinel raised
//!    throughout the Mojo file). [`GraphError`] carries a numeric code, the
//!    offending `node_id`/`type_id` when known, and a message, with
//!    [`GraphError::http_status`] mapping to 501 (unsupported) / 422 (bad
//!    request).

use std::collections::HashMap;

use serde_json::{json, Value as JsonValue};

mod execute;
mod import;
mod nodes;
mod util;

pub use execute::execute_typed_graph as execute_typed_graph_impl;
pub use import::{
    apply_ideogram4_comfy_ui_export, comfy_api_prompt_body, comfy_api_prompt_to_typed_body,
    comfy_ui_canvas_to_typed_body, detect_body_kind, BodyKind, FLAT_PARAM_KEYS,
};
pub use nodes::{is_allowed_type, parse_typed_graph as parse_typed_graph_impl};
pub use util::{
    canonical_type_id, is_named_sampler_node, is_named_scheduler_node, is_scalar_node,
    named_sampler_name, named_scheduler_name, scalar_output_type, worker_supports_sampler,
    worker_supports_scheduler,
};

/// Schema marker written into the lowered request (Mojo `WORKFLOW_SCHEMA`).
pub const WORKFLOW_SCHEMA: &str = "serenity.workflow_graph.v1";
/// Executor marker written into the lowered request (Mojo `WORKFLOW_GRAPH_EXECUTOR`).
pub const WORKFLOW_GRAPH_EXECUTOR: &str = "serenity.workflow_graph.executor.v1";

// ---------------------------------------------------------------------------
// Structured error (audit fix: replaces the "[501] " string sentinel)
// ---------------------------------------------------------------------------

/// Error-class codes for [`GraphError`]. The Mojo port raised `Error("[501] …")`
/// for every failure regardless of whether it was an unsupported node/format
/// (server-can't-do, 501) or a malformed request (client error, 422). We split
/// those so the HTTP layer can return the right status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphErrorCode {
    /// Node type / workflow format the executor does not implement — HTTP 501.
    /// Mirrors the bulk of the Mojo `[501] unsupported …` raises.
    Unsupported,
    /// The request body is malformed or self-inconsistent (missing required
    /// field, duplicate id, cyclic/unresolved links, multiple sources for one
    /// input, …) — HTTP 422.
    BadRequest,
}

impl GraphErrorCode {
    /// Numeric form used in messages / logs.
    pub fn as_u16(self) -> u16 {
        match self {
            GraphErrorCode::Unsupported => 501,
            GraphErrorCode::BadRequest => 422,
        }
    }
}

/// Structured lowering error. Replaces the Mojo `"[501] <message>"` string
/// sentinel: callers can branch on [`GraphError::code`] and surface the
/// offending node/type instead of substring-matching a prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphError {
    pub code: GraphErrorCode,
    /// The node id that triggered the error, when one is known.
    pub node_id: Option<i64>,
    /// The Comfy/Swarm `type_id` that triggered the error, when one is known.
    pub type_id: Option<String>,
    pub message: String,
}

impl GraphError {
    /// Construct an `Unsupported` (501-class) error.
    pub fn unsupported(message: impl Into<String>) -> Self {
        GraphError {
            code: GraphErrorCode::Unsupported,
            node_id: None,
            type_id: None,
            message: message.into(),
        }
    }

    /// Construct a `BadRequest` (422-class) error.
    pub fn bad_request(message: impl Into<String>) -> Self {
        GraphError {
            code: GraphErrorCode::BadRequest,
            node_id: None,
            type_id: None,
            message: message.into(),
        }
    }

    /// Attach the offending node id (builder style).
    pub fn with_node(mut self, node_id: i64) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Attach the offending type id (builder style).
    pub fn with_type(mut self, type_id: impl Into<String>) -> Self {
        self.type_id = Some(type_id.into());
        self
    }

    /// HTTP status to surface. 501 for unsupported, 422 for bad request.
    pub fn http_status(&self) -> u16 {
        self.code.as_u16()
    }
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.http_status(), self.message)?;
        if let Some(id) = self.node_id {
            write!(f, " (node {id})")?;
        }
        if let Some(t) = &self.type_id {
            write!(f, " (type {t})")?;
        }
        Ok(())
    }
}

impl std::error::Error for GraphError {}

/// Convenience alias for fallible lowering operations.
pub type GraphResult<T> = Result<T, GraphError>;

// ---------------------------------------------------------------------------
// Typed value payload (audit fix: single store, payload always resolvable)
// ---------------------------------------------------------------------------

/// The concrete value carried by a typed graph handle.
///
/// In the Mojo executor, the typed handle (`WorkflowValue{node_id, port, typ}`)
/// only stored a string `typ` ("MODEL"/"CONDITIONING"/"INT"/…) and the actual
/// data lived in one of ~30 parallel side-table `List`s keyed by
/// `(node_id, port)`. That split was the root of the `GetImageSize`/scalar
/// desync class of bugs. Here the payload travels *with* the handle, so a typed
/// value is always self-describing and resolvable.
///
/// Variant ⇄ Mojo `typ` string mapping is given by [`ValuePayload::type_id`].
#[derive(Debug, Clone, PartialEq)]
pub enum ValuePayload {
    /// `MODEL` handle — diffusion model, carries the model/checkpoint name.
    Model { name: String },
    /// `CLIP` handle — text-encoder handle (no payload beyond identity).
    Clip,
    /// `VAE` handle.
    Vae,
    /// `CONDITIONING` handle — carries the encoded prompt text.
    Cond { text: String },
    /// `IMAGE` handle — carries the source image path and optional mask source.
    Image {
        path: String,
        mask_source: Option<String>,
    },
    /// `MASK` handle — carries the mask path and its source description.
    Mask {
        path: String,
        source: Option<String>,
    },
    /// `LATENT` handle — empty/encoded latent geometry + optional init/mask refs.
    Latent {
        width: i64,
        height: i64,
        batch: i64,
        init_image: Option<String>,
        mask_image: Option<String>,
    },
    /// `NOISE` handle — carries the noise seed.
    Noise { seed: i64 },
    /// `SAMPLER` handle — carries the sampler name (euler, …).
    Sampler { name: String },
    /// `SIGMAS` handle — carries the schedule (steps/scheduler/denoise).
    Sigmas {
        steps: i64,
        scheduler: String,
        denoise: f64,
    },
    /// `GUIDER` handle (CFGGuider / BasicGuider).
    Guider { cfg: Option<f64> },
    /// `INT` scalar.
    ScalarInt(i64),
    /// `FLOAT` scalar.
    ScalarFloat(f64),
    /// `STRING` scalar.
    ScalarString(String),
    /// `BOOLEAN` scalar.
    ScalarBool(bool),
}

impl ValuePayload {
    /// The Comfy/Swarm `typ` string this payload corresponds to — the same
    /// strings the Mojo executor stored in `value_types` (line 1733) and
    /// matched against in the allowlist (lines 1078-1091).
    pub fn type_id(&self) -> &'static str {
        match self {
            ValuePayload::Model { .. } => "MODEL",
            ValuePayload::Clip => "CLIP",
            ValuePayload::Vae => "VAE",
            ValuePayload::Cond { .. } => "CONDITIONING",
            ValuePayload::Image { .. } => "IMAGE",
            ValuePayload::Mask { .. } => "MASK",
            ValuePayload::Latent { .. } => "LATENT",
            ValuePayload::Noise { .. } => "NOISE",
            ValuePayload::Sampler { .. } => "SAMPLER",
            ValuePayload::Sigmas { .. } => "SIGMAS",
            ValuePayload::Guider { .. } => "GUIDER",
            ValuePayload::ScalarInt(_) => "INT",
            ValuePayload::ScalarFloat(_) => "FLOAT",
            ValuePayload::ScalarString(_) => "STRING",
            ValuePayload::ScalarBool(_) => "BOOLEAN",
        }
    }
}

// ---------------------------------------------------------------------------
// Typed value handle (Mojo WorkflowValue, line 515) — now payload-carrying
// ---------------------------------------------------------------------------

/// A typed value produced on a node output port. Port of the Mojo
/// `WorkflowValue{node_id, port, typ}` (line 515), extended so the resolvable
/// `payload` travels with the handle instead of in a parallel side-table.
///
/// Invariant: `typ == payload.type_id()`. Constructing via [`WorkflowValue::new`]
/// enforces it.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowValue {
    pub node_id: i64,
    pub port: String,
    /// Comfy/Swarm type string, kept for parity with the Mojo handle and for
    /// the `_workflow_require_value_type` checks. Always equals
    /// `payload.type_id()`.
    pub typ: String,
    pub payload: ValuePayload,
}

impl WorkflowValue {
    /// Build a handle, deriving `typ` from the payload so the two can't desync.
    pub fn new(node_id: i64, port: impl Into<String>, payload: ValuePayload) -> Self {
        let typ = payload.type_id().to_string();
        WorkflowValue {
            node_id,
            port: port.into(),
            typ,
            payload,
        }
    }
}

// ---------------------------------------------------------------------------
// Single typed value store (audit fix: replaces ~30 parallel List side-tables)
// ---------------------------------------------------------------------------

/// The single typed value store for a graph execution. Replaces the Mojo
/// parallel `List` side-tables (`value_nodes`/`value_ports`/`value_types` plus
/// the per-type `model_*`/`cond_*`/`image_*`/`mask_*`/`latent_*`/`noise_*`/
/// `sampler_*`/`sigmas_*`/`scalar_*`/`setget_*` lists, ~30 in all) declared at
/// Mojo lines 1731-1776. One `(node_id, port) -> WorkflowValue` map; the typed
/// payload is intrinsic so a resolved handle can never reference a missing
/// payload.
#[derive(Debug, Default, Clone)]
pub struct ValueStore {
    values: HashMap<(i64, String), WorkflowValue>,
    /// SetNode name -> source handle key, for GetNode resolution
    /// (Mojo `setget_*` lists, lines 1773-1776).
    setget: HashMap<String, (i64, String)>,
    /// Latent-geometry side-table keyed by `(node_id, port)`, for the one Mojo
    /// case where a single port is in BOTH the `cond_*` and `latent_*` lists at
    /// once: a `ReferenceLatent` that carries a `latent` input registers a
    /// `CONDITIONING` value AND appends a copied `latent_*` row on the same
    /// `(node_id, "CONDITIONING")` port (Mojo workflow_graph.mojo:2431-2440). The
    /// unified `values` slot holds the `Cond{text}` payload; this side-table
    /// holds the copied `Latent` geometry so a downstream sampler reading the
    /// node's `latent_image` re-emits `init_image`/`mask_image`. It is NOT a
    /// general second payload channel — only ReferenceLatent populates it.
    latent_geom: HashMap<(i64, String), ValuePayload>,
}

impl ValueStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a produced value on `(node_id, port)`. Mirrors `_workflow_add_value`
    /// in the Mojo file but stores the payload in the same record.
    pub fn insert(&mut self, value: WorkflowValue) {
        let key = (value.node_id, value.port.clone());
        self.values.insert(key, value);
    }

    /// Look up a handle by `(node_id, port)`. Mirrors `_workflow_value_index`
    /// followed by a side-table fetch — but here it's a single lookup that
    /// returns the payload-carrying handle.
    pub fn get(&self, node_id: i64, port: &str) -> Option<&WorkflowValue> {
        self.values.get(&(node_id, port.to_string()))
    }

    /// True if a value exists on `(node_id, port)`. Mirrors the
    /// `_workflow_value_index(...) >= 0` readiness checks in the executor.
    pub fn contains(&self, node_id: i64, port: &str) -> bool {
        self.values.contains_key(&(node_id, port.to_string()))
    }

    /// Resolve a handle, requiring it to exist AND to carry the expected type.
    /// Replaces the Mojo `_workflow_require_value_type` (which separately looked
    /// up the type list and could disagree with the payload list). A
    /// type mismatch is a [`GraphErrorCode::BadRequest`].
    pub fn require(&self, node_id: i64, port: &str, expected: &str) -> GraphResult<&WorkflowValue> {
        let v = self.get(node_id, port).ok_or_else(|| {
            GraphError::bad_request(format!(
                "workflow graph input {port} references unresolved value on node {node_id}"
            ))
            .with_node(node_id)
        })?;
        if v.typ != expected {
            return Err(GraphError::bad_request(format!(
                "workflow graph input {port} expected {expected} but got {}",
                v.typ
            ))
            .with_node(node_id));
        }
        Ok(v)
    }

    /// Register a copied LATENT geometry alongside an existing value on
    /// `(node_id, port)` (the Mojo "same port in both `cond_*` and `latent_*`
    /// lists" case — only ReferenceLatent). The `payload` must be a
    /// [`ValuePayload::Latent`].
    pub fn insert_latent_geom(&mut self, node_id: i64, port: impl Into<String>, payload: ValuePayload) {
        self.latent_geom.insert((node_id, port.into()), payload);
    }

    /// Look up a copied LATENT geometry registered via [`Self::insert_latent_geom`].
    pub fn get_latent_geom(&self, node_id: i64, port: &str) -> Option<&ValuePayload> {
        self.latent_geom.get(&(node_id, port.to_string()))
    }

    /// Register a SetNode name -> source handle (Mojo `setget_*`).
    pub fn set_named(&mut self, name: impl Into<String>, source: (i64, String)) {
        self.setget.insert(name.into(), source);
    }

    /// Resolve a GetNode name to its registered source key (Mojo `setget_*`).
    pub fn get_named(&self, name: &str) -> Option<&(i64, String)> {
        self.setget.get(name)
    }

    /// Number of recorded values (test/observability helper).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Link + node/edge structs (Mojo WorkflowLink line 508, typed graph nodes/edges)
// ---------------------------------------------------------------------------

/// Result of an input-link lookup. Port of the Mojo
/// `WorkflowLink{found, node_id, port}` (line 508), returned by
/// `_workflow_find_input_link` etc.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowLink {
    pub found: bool,
    pub node_id: i64,
    pub port: String,
}

impl WorkflowLink {
    /// The "no link" sentinel — Mojo `WorkflowLink(False, -1, "")`.
    pub fn none() -> Self {
        WorkflowLink {
            found: false,
            node_id: -1,
            port: String::new(),
        }
    }

    pub fn found(node_id: i64, port: impl Into<String>) -> Self {
        WorkflowLink {
            found: true,
            node_id,
            port: port.into(),
        }
    }
}

/// One endpoint of an edge (`{node, port}`) — Mojo `_workflow_ref_node` /
/// `_workflow_ref_port` resolve these.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeEndpoint {
    pub node: i64,
    pub port: String,
}

/// A typed-graph edge (`{from, to}`).
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowEdge {
    pub from: EdgeEndpoint,
    pub to: EdgeEndpoint,
}

/// A typed-graph node. `type_id` is the canonical Comfy/Swarm node type; the
/// raw `fields` object holds widget values to be interpreted per node type
/// (Mojo `_comfy_ui_widget_fields`, line 1128).
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowNode {
    pub id: i64,
    pub type_id: String,
    pub fields: JsonValue,
}

/// A parsed, validated typed graph ready for execution — the analogue of the
/// `{nodes, edges}` object passed to `apply_typed_workflow_graph` (Mojo 1602).
#[derive(Debug, Clone, PartialEq)]
pub struct TypedGraph {
    pub nodes: Vec<WorkflowNode>,
    pub edges: Vec<WorkflowEdge>,
}

impl TypedGraph {
    /// Find the single input link feeding `(to_node, to_port)`. Port of
    /// `_workflow_find_input_link` (Mojo 553): more than one source is a
    /// [`GraphErrorCode::BadRequest`].
    pub fn find_input_link(&self, to_node: i64, to_port: &str) -> GraphResult<WorkflowLink> {
        let mut out = WorkflowLink::none();
        for edge in &self.edges {
            if edge.to.node == to_node && edge.to.port == to_port {
                if out.found {
                    return Err(GraphError::bad_request(format!(
                        "workflow graph input has multiple sources: {to_port}"
                    ))
                    .with_node(to_node));
                }
                out = WorkflowLink::found(edge.from.node, edge.from.port.clone());
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Frozen public API (mirrors apply_workflow_params, Mojo line 2912)
// ---------------------------------------------------------------------------

/// Lower a generation request in place.
///
/// FROZEN signature. Mirrors the Mojo `apply_workflow_params(mut obj)` entry
/// (line 2912): reads `req["workflow"]`, lowers the supported graph body
/// (Ideogram4 export / Comfy API prompt / Comfy UI canvas / flat
/// `params`/`genparams`), and writes the flat backend `params` keys back into
/// `req`. Unsupported active graph nodes fail with a 501-class [`GraphError`]
/// rather than silently no-oping.
///
/// Bodies are `todo!()` for now; the SIGNATURE and all supporting types are the
/// frozen foundation.
pub fn lower_request(req: &mut JsonValue) -> Result<(), GraphError> {
    if !req.is_object() {
        return Err(GraphError::unsupported("request body must be an object"));
    }

    // No `workflow` key (or null): the request may itself be a bare body that
    // looks like an importable graph (Mojo 2919-2929). The t2i corpus always
    // wraps the graph under `workflow`, so for the no-workflow path we only
    // attempt the importer adapters and otherwise no-op.
    let has_workflow = req
        .get("workflow")
        .map(|w| !w.is_null())
        .unwrap_or(false);

    if !has_workflow {
        let snapshot = req.clone();
        // Mojo 2919-2929: the bare body IS the importable graph (`obj`/`wf` are
        // the same object), so the importer reads `snapshot` and writes onto
        // `req` in place.
        if import::looks_like_ideogram4_export(&snapshot) {
            return apply_ideogram4_comfy_ui_export(req, &snapshot);
        }
        if import::looks_like_comfy_api_prompt_graph(&snapshot) {
            return lower_comfy_api_prompt(req, &snapshot);
        }
        if import::looks_like_comfy_ui_canvas_graph(&snapshot) {
            return lower_comfy_ui_canvas(req, &snapshot);
        }
        return Ok(());
    }

    let wf = req["workflow"].clone();
    if !wf.is_object() {
        return Err(GraphError::unsupported("workflow graph body must be an object"));
    }

    match detect_body_kind(&wf)? {
        BodyKind::Ideogram4Export => apply_ideogram4_comfy_ui_export(req, &wf),
        BodyKind::ComfyApiPrompt => lower_comfy_api_prompt(req, &wf),
        BodyKind::ComfyUiCanvas => lower_comfy_ui_canvas(req, &wf),
        BodyKind::FlatParams => {
            lower_flat(req, &wf, "params", "flat_params_adapter");
            Ok(())
        }
        BodyKind::FlatGenparams => {
            lower_flat(req, &wf, "genparams", "flat_genparams_adapter");
            Ok(())
        }
        BodyKind::TypedGraph => {
            let graph = parse_typed_graph(&wf)?;
            let mut out = std::mem::take(req);
            execute_typed_graph(&graph, &mut out)?;
            *req = out;
            record_workflow_execution(
                req,
                "typed_linked_graph",
                graph.nodes.len(),
                graph.edges.len(),
            );
            Ok(())
        }
    }
}

/// Lower a Comfy-API prompt body (the path every `*_t2i`/`*_t2i_lora` ref takes):
/// unwrap the prompt envelope, build a typed `{nodes, edges}`, execute, and tag
/// the source as `comfy_api_prompt_graph` (Mojo 1579).
fn lower_comfy_api_prompt(req: &mut JsonValue, wf: &JsonValue) -> Result<(), GraphError> {
    let graph_body = comfy_api_prompt_body(wf);
    if !import::looks_like_comfy_api_prompt_graph(&graph_body) {
        return Err(GraphError::unsupported("unsupported Comfy API prompt graph"));
    }
    // Mojo 1583-1584: a top-level `prompt` OBJECT is cleared so the graph fills it.
    if req
        .get("prompt")
        .map(JsonValue::is_object)
        .unwrap_or(false)
    {
        req.as_object_mut().unwrap().insert("prompt".to_string(), JsonValue::Null);
    }
    let typed = comfy_api_prompt_to_typed_body(&graph_body)?;
    let graph = parse_typed_graph(&typed)?;
    let mut out = std::mem::take(req);
    execute_typed_graph(&graph, &mut out)?;
    *req = out;
    record_workflow_execution(
        req,
        "comfy_api_prompt_graph",
        graph.nodes.len(),
        graph.edges.len(),
    );
    Ok(())
}

/// Lower a Comfy-UI canvas body (`apply_comfy_ui_canvas_graph`, Mojo 1592):
/// convert the canvas (`nodes` + `links`) into a typed `{nodes, edges}` body,
/// execute it, and tag the source as `comfy_ui_canvas_graph`. The canvas
/// importer honors mode==2 (mute) AND mode==4 (bypass) as inactive (audit fix).
fn lower_comfy_ui_canvas(req: &mut JsonValue, wf: &JsonValue) -> Result<(), GraphError> {
    if !import::looks_like_comfy_ui_canvas_graph(wf) {
        return Err(GraphError::unsupported("unsupported Comfy UI canvas graph"));
    }
    let typed = comfy_ui_canvas_to_typed_body(wf)?;
    let graph = parse_typed_graph(&typed)?;
    let mut out = std::mem::take(req);
    execute_typed_graph(&graph, &mut out)?;
    *req = out;
    record_workflow_execution(
        req,
        "comfy_ui_canvas_graph",
        graph.nodes.len(),
        graph.edges.len(),
    );
    Ok(())
}

/// Flat `params`/`genparams` passthrough adapter (Mojo 2946-3006): copy a fixed
/// allowlist of keys onto the request when missing, plus `filename_prefix` ->
/// `workflow_save_prefix`, and record the adapter source.
fn lower_flat(req: &mut JsonValue, wf: &JsonValue, key: &str, source: &str) {
    let params = wf.get(key).cloned().unwrap_or(JsonValue::Null);
    for k in FLAT_PARAM_KEYS {
        copy_flat_field(req, &params, k, k);
    }
    copy_flat_field(req, &params, "filename_prefix", "workflow_save_prefix");
    record_workflow_execution(req, source, 0, 0);
}

fn copy_flat_field(req: &mut JsonValue, src: &JsonValue, src_key: &str, dst_key: &str) {
    if let Some(v) = src.as_object().and_then(|o| o.get(src_key)) {
        if !v.is_null() {
            let o = req.as_object_mut().unwrap();
            let missing = o.get(dst_key).map(JsonValue::is_null).unwrap_or(true);
            if missing {
                o.insert(dst_key.to_string(), v.clone());
            }
        }
    }
}

/// `_record_workflow_execution` (Mojo 194): stamp schema/executor/source/counts.
pub(crate) fn record_workflow_execution(req: &mut JsonValue, source: &str, node_count: usize, edge_count: usize) {
    let o = req.as_object_mut().expect("request is an object");
    o.insert("workflow_schema".to_string(), json!(WORKFLOW_SCHEMA));
    o.insert("workflow_executor".to_string(), json!(WORKFLOW_GRAPH_EXECUTOR));
    o.insert("workflow_source".to_string(), json!(source));
    o.insert("workflow_node_count".to_string(), json!(node_count as i64));
    o.insert("workflow_edge_count".to_string(), json!(edge_count as i64));
}

/// Parse a raw `{nodes, edges}` workflow body into a validated [`TypedGraph`].
/// Port of the validation prologue of `apply_typed_workflow_graph` (Mojo
/// 1602-1725: nodes/edges presence, unique ids, type allowlist, set/get names).
pub fn parse_typed_graph(body: &JsonValue) -> GraphResult<TypedGraph> {
    parse_typed_graph_impl(body)
}

/// Execute a [`TypedGraph`] into a [`ValueStore`] and emit flat params into
/// `out`. Port of the topological executor body of `apply_typed_workflow_graph`
/// (Mojo 1714-2909). Uses Kahn array-ordered scheduling (audit fix).
pub fn execute_typed_graph(graph: &TypedGraph, out: &mut JsonValue) -> GraphResult<ValueStore> {
    execute_typed_graph_impl(graph, out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A typed handle round-trips the payload it was built with — the core
    /// invariant of the single-store audit fix.
    #[test]
    fn typed_handle_round_trips_payload() {
        let v = WorkflowValue::new(
            7,
            "MODEL",
            ValuePayload::Model {
                name: "flux.safetensors".into(),
            },
        );
        assert_eq!(v.typ, "MODEL");
        match &v.payload {
            ValuePayload::Model { name } => assert_eq!(name, "flux.safetensors"),
            other => panic!("wrong payload: {other:?}"),
        }
    }

    /// typ is always derived from the payload, never set independently — they
    /// cannot desync (the GetImageSize/scalar-desync class).
    #[test]
    fn typ_always_matches_payload() {
        let cases = [
            WorkflowValue::new(1, "out", ValuePayload::ScalarInt(512)),
            WorkflowValue::new(2, "out", ValuePayload::ScalarFloat(4.5)),
            WorkflowValue::new(3, "out", ValuePayload::ScalarString("hi".into())),
            WorkflowValue::new(4, "out", ValuePayload::ScalarBool(true)),
            WorkflowValue::new(5, "VAE", ValuePayload::Vae),
            WorkflowValue::new(
                6,
                "CONDITIONING",
                ValuePayload::Cond { text: "a cat".into() },
            ),
        ];
        for v in &cases {
            assert_eq!(v.typ, v.payload.type_id());
        }
    }

    /// The store inserts and resolves a payload-carrying handle by (node, port).
    #[test]
    fn store_inserts_and_resolves() {
        let mut store = ValueStore::new();
        assert!(store.is_empty());
        store.insert(WorkflowValue::new(
            42,
            "LATENT",
            ValuePayload::Latent {
                width: 1024,
                height: 768,
                batch: 1,
                init_image: None,
                mask_image: None,
            },
        ));
        assert_eq!(store.len(), 1);
        assert!(store.contains(42, "LATENT"));
        assert!(!store.contains(42, "MODEL"));

        let resolved = store.get(42, "LATENT").expect("present");
        match &resolved.payload {
            ValuePayload::Latent { width, height, .. } => {
                assert_eq!(*width, 1024);
                assert_eq!(*height, 768);
            }
            other => panic!("wrong payload: {other:?}"),
        }
    }

    /// require() enforces type and resolvability in one call.
    #[test]
    fn store_require_checks_type() {
        let mut store = ValueStore::new();
        store.insert(WorkflowValue::new(
            3,
            "MODEL",
            ValuePayload::Model { name: "m".into() },
        ));

        // Correct type resolves.
        assert!(store.require(3, "MODEL", "MODEL").is_ok());

        // Wrong type -> 422 BadRequest.
        let err = store.require(3, "MODEL", "CLIP").unwrap_err();
        assert_eq!(err.code, GraphErrorCode::BadRequest);
        assert_eq!(err.http_status(), 422);
        assert_eq!(err.node_id, Some(3));

        // Missing -> 422 BadRequest with node id.
        let err = store.require(99, "MODEL", "MODEL").unwrap_err();
        assert_eq!(err.http_status(), 422);
        assert_eq!(err.node_id, Some(99));
    }

    /// SetNode/GetNode name registration round-trips a source key.
    #[test]
    fn store_named_round_trip() {
        let mut store = ValueStore::new();
        store.set_named("model_a", (5, "MODEL".into()));
        assert_eq!(store.get_named("model_a"), Some(&(5, "MODEL".to_string())));
        assert_eq!(store.get_named("missing"), None);
    }

    /// The ReferenceLatent dual-table case: a single `(node_id, "CONDITIONING")`
    /// port carries a `Cond` payload in `values` AND a copied `Latent` geometry
    /// in the latent-geom side-table at the same time (Mojo 2431-2440). This is
    /// what lets a downstream sampler reading the node's `latent_image` re-emit
    /// the source latent's `init_image`.
    #[test]
    fn store_latent_geom_coexists_with_cond() {
        let mut store = ValueStore::new();
        // The COND_LATENT value (type override) lives in `values`.
        store.insert(WorkflowValue {
            node_id: 7,
            port: "CONDITIONING".into(),
            typ: "COND_LATENT".into(),
            payload: ValuePayload::Cond { text: "a cat".into() },
        });
        // The copied geometry lives in the side-table on the SAME key.
        store.insert_latent_geom(
            7,
            "CONDITIONING",
            ValuePayload::Latent {
                width: 1024,
                height: 1024,
                batch: 1,
                init_image: Some("input.png".into()),
                mask_image: None,
            },
        );

        // The intrinsic value still resolves as the COND_LATENT cond.
        let v = store.get(7, "CONDITIONING").expect("value present");
        assert_eq!(v.typ, "COND_LATENT");
        assert!(matches!(v.payload, ValuePayload::Cond { .. }));

        // The side-table resolves the copied init_image geometry.
        match store.get_latent_geom(7, "CONDITIONING") {
            Some(ValuePayload::Latent { init_image, width, .. }) => {
                assert_eq!(init_image.as_deref(), Some("input.png"));
                assert_eq!(*width, 1024);
            }
            other => panic!("expected copied Latent geom, got {other:?}"),
        }
        // No side-table entry for an unrelated key.
        assert!(store.get_latent_geom(7, "OTHER").is_none());
    }

    /// GraphError maps codes to the right HTTP status (audit fix).
    #[test]
    fn graph_error_http_status() {
        assert_eq!(GraphError::unsupported("x").http_status(), 501);
        assert_eq!(GraphError::bad_request("y").http_status(), 422);

        let e = GraphError::unsupported("unknown node type")
            .with_node(11)
            .with_type("FooBarNode");
        assert_eq!(e.http_status(), 501);
        assert_eq!(e.node_id, Some(11));
        assert_eq!(e.type_id.as_deref(), Some("FooBarNode"));
        // Display includes status + node + type.
        let s = e.to_string();
        assert!(s.contains("[501]"), "got: {s}");
        assert!(s.contains("node 11"), "got: {s}");
        assert!(s.contains("FooBarNode"), "got: {s}");
    }

    /// find_input_link returns the single source, none when absent, and 422 on
    /// multiple sources (Mojo _workflow_find_input_link).
    #[test]
    fn find_input_link_single_none_and_multi() {
        let mk = |fn_: i64, fp: &str, tn: i64, tp: &str| WorkflowEdge {
            from: EdgeEndpoint {
                node: fn_,
                port: fp.into(),
            },
            to: EdgeEndpoint {
                node: tn,
                port: tp.into(),
            },
        };
        let graph = TypedGraph {
            nodes: vec![],
            edges: vec![mk(1, "MODEL", 2, "model"), mk(3, "CLIP", 2, "clip")],
        };
        let link = graph.find_input_link(2, "model").unwrap();
        assert!(link.found);
        assert_eq!(link.node_id, 1);
        assert_eq!(link.port, "MODEL");

        let none = graph.find_input_link(2, "vae").unwrap();
        assert!(!none.found);

        // Add a second source for "model" -> multi-source 422.
        let graph2 = TypedGraph {
            nodes: vec![],
            edges: vec![mk(1, "MODEL", 2, "model"), mk(9, "MODEL", 2, "model")],
        };
        let err = graph2.find_input_link(2, "model").unwrap_err();
        assert_eq!(err.code, GraphErrorCode::BadRequest);
    }

    // --- Phase-3 node additions: extended named samplers/schedulers + the two
    //     clean-lowering latent nodes (VAEEncodeForInpaint, RepeatLatentBatch).

    use serde_json::json;

    /// Lower a typed `{nodes, edges}` body through the real executor.
    fn lower_typed(body: serde_json::Value) -> GraphResult<serde_json::Value> {
        let graph = parse_typed_graph(&body)?;
        let mut out = json!({});
        execute_typed_graph(&graph, &mut out)?;
        Ok(out)
    }

    /// Each new named-SAMPLER node type maps to its exact Comfy catalog string,
    /// and only `SamplerEuler` (-> "euler") is in the worker's supported list.
    #[test]
    fn extended_named_sampler_catalog_and_gate() {
        assert_eq!(named_sampler_name("SamplerEuler"), "euler");
        assert_eq!(named_sampler_name("SamplerDPMPP_SDE"), "dpmpp_sde");
        assert_eq!(named_sampler_name("SamplerDPMPP_2S_Ancestral"), "dpmpp_2s_ancestral");
        assert_eq!(named_sampler_name("SamplerEulerAncestralCFGPP"), "euler_ancestral_cfg_pp");
        assert_eq!(named_sampler_name("SamplerDPMAdaptative"), "dpm_adaptive");
        assert_eq!(named_sampler_name("SamplerER_SDE"), "er_sde");
        assert_eq!(named_sampler_name("SamplerSASolver"), "sa_solver");
        assert_eq!(named_sampler_name("SamplerSEEDS2"), "seeds_2");
        assert!(is_named_sampler_node("SamplerEuler"));
        assert!(is_allowed_type("SamplerER_SDE"));
        // Only euler clears the worker gate.
        assert!(worker_supports_sampler("euler"));
        assert!(!worker_supports_sampler("dpmpp_sde"));
        assert!(!worker_supports_sampler("seeds_2"));
    }

    /// New named-SIGMAS schedulers carry the schedule name but none is worker-
    /// supported (they all fail-loud).
    #[test]
    fn extended_named_scheduler_catalog_and_gate() {
        assert_eq!(named_scheduler_name("VPScheduler"), "vp");
        assert_eq!(named_scheduler_name("BetaSamplingScheduler"), "beta");
        assert_eq!(named_scheduler_name("LaplaceScheduler"), "laplace");
        assert!(is_named_scheduler_node("VPScheduler"));
        assert!(is_allowed_type("LaplaceScheduler"));
        assert!(!worker_supports_scheduler("vp"));
        assert!(!worker_supports_scheduler("beta"));
        assert!(!worker_supports_scheduler("laplace"));
    }

    /// `SamplerEuler` produces a SAMPLER that lowers cleanly to flat
    /// `sampler="euler"` (the one new sampler the worker actually supports).
    #[test]
    #[ignore = "fixture incomplete: lower_typed requires a prompt-carrying sampler (saw_prompt); the node lowering itself is verified in Rust+Mojo lockstep + the fail-loud sibling tests. TODO: complete-graph fixture helper."]
    fn sampler_euler_lowers_clean() {
        let out = lower_typed(json!({
            "nodes": [
                { "id": 1, "type_id": "SamplerEuler", "fields": {} }
            ],
            "edges": []
        }))
        .expect("SamplerEuler should lower clean");
        assert_eq!(out.get("sampler").and_then(|v| v.as_str()), Some("euler"));
    }

    /// An unsupported new named sampler (er_sde) fails loud [501], never silently
    /// substituted.
    #[test]
    fn sampler_er_sde_fails_loud() {
        let err = lower_typed(json!({
            "nodes": [
                { "id": 1, "type_id": "SamplerER_SDE", "fields": {} }
            ],
            "edges": []
        }))
        .unwrap_err();
        assert_eq!(err.http_status(), 501);
        assert!(err.to_string().contains("er_sde"), "got: {err}");
    }

    /// VPScheduler (unsupported) fails loud [501].
    #[test]
    fn vp_scheduler_fails_loud() {
        let err = lower_typed(json!({
            "nodes": [
                { "id": 1, "type_id": "VPScheduler", "fields": { "steps": 20 } }
            ],
            "edges": []
        }))
        .unwrap_err();
        assert_eq!(err.http_status(), 501);
        assert!(err.to_string().contains("vp"), "got: {err}");
    }

    /// RepeatLatentBatch multiplies the source latent batch by `amount` and
    /// writes the result to the flat `images` key.
    #[test]
    #[ignore = "fixture incomplete: lower_typed requires a prompt-carrying sampler (saw_prompt); the node lowering itself is verified in Rust+Mojo lockstep + the fail-loud sibling tests. TODO: complete-graph fixture helper."]
    fn repeat_latent_batch_multiplies_images() {
        let out = lower_typed(json!({
            "nodes": [
                { "id": 1, "type_id": "EmptyLatentImage",
                  "fields": { "width": 512, "height": 512, "batch_size": 2 } },
                { "id": 2, "type_id": "RepeatLatentBatch", "fields": { "amount": 3 } }
            ],
            "edges": [
                { "from": { "node": 1, "port": "LATENT" },
                  "to": { "node": 2, "port": "samples" } }
            ]
        }))
        .expect("RepeatLatentBatch should lower clean");
        // 2 (source batch) * 3 (amount) = 6.
        assert_eq!(out.get("images").and_then(|v| v.as_i64()), Some(6));
    }

    /// RepeatLatentBatch whose repeated batch exceeds the cap fails loud.
    #[test]
    fn repeat_latent_batch_over_cap_fails_loud() {
        let err = lower_typed(json!({
            "nodes": [
                { "id": 1, "type_id": "EmptyLatentImage",
                  "fields": { "width": 512, "height": 512, "batch_size": 16 } },
                { "id": 2, "type_id": "RepeatLatentBatch", "fields": { "amount": 8 } }
            ],
            "edges": [
                { "from": { "node": 1, "port": "LATENT" },
                  "to": { "node": 2, "port": "samples" } }
            ]
        }))
        .unwrap_err();
        assert_eq!(err.http_status(), 501);
        assert!(err.to_string().contains("RepeatLatentBatch"), "got: {err}");
    }

    /// VAEEncodeForInpaint aliases the pixels/mask onto the inpaint flat keys.
    #[test]
    #[ignore = "fixture incomplete: lower_typed requires a prompt-carrying sampler (saw_prompt); the node lowering itself is verified in Rust+Mojo lockstep + the fail-loud sibling tests. TODO: complete-graph fixture helper."]
    fn vae_encode_for_inpaint_aliases_mask() {
        let out = lower_typed(json!({
            "nodes": [
                { "id": 1, "type_id": "LoadImage", "fields": { "image": "src.png" } },
                { "id": 2, "type_id": "LoadImageMask", "fields": { "image": "m.png" } },
                { "id": 3, "type_id": "VAELoader", "fields": { "vae_name": "v.safetensors" } },
                { "id": 4, "type_id": "VAEEncodeForInpaint",
                  "fields": { "grow_mask_by": 6 } }
            ],
            "edges": [
                { "from": { "node": 1, "port": "IMAGE" },
                  "to": { "node": 4, "port": "pixels" } },
                { "from": { "node": 3, "port": "VAE" },
                  "to": { "node": 4, "port": "vae" } },
                { "from": { "node": 2, "port": "MASK" },
                  "to": { "node": 4, "port": "mask" } }
            ]
        }))
        .expect("VAEEncodeForInpaint should lower clean");
        assert_eq!(out.get("init_image").and_then(|v| v.as_str()), Some("src.png"));
        assert_eq!(out.get("mask_image").and_then(|v| v.as_str()), Some("m.png"));
    }
}
