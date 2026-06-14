//! `parse_typed_graph` — the validation prologue of `apply_typed_workflow_graph`
//! (Mojo 1602-1725): parse `{nodes, edges}`, require object nodes, enforce
//! unique ids and the type allowlist, parse edge endpoints, and validate
//! SetNode/GetNode names. Audit fixes: structured `GraphError`, exact membership.

use std::collections::HashSet;

use serde_json::Value as JsonValue;

use crate::util::{canonical_type_id, setget_name, wf_string};
use crate::{EdgeEndpoint, GraphError, GraphResult, TypedGraph, WorkflowEdge, WorkflowNode};

/// The set of node `type_id`s the executor accepts (Mojo allowlist 1621-1696).
/// An active node outside this set is a 501 (fail-loud), per the audit.
pub fn is_allowed_type(t: &str) -> bool {
    matches!(
        t,
        "CheckpointLoaderSimple"
            | "UNETLoader"
            | "DiffusionModelLoader"
            | "LoraLoader"
            | "LoraLoaderModelOnly"
            | "ZImageLoraModelOnly"
            | "CLIPLoader"
            | "DualCLIPLoader"
            | "TripleCLIPLoader"
            | "VAELoader"
            | "CLIPTextEncode"
            | "CLIPTextEncodeFlux"
            | "TextEncodeQwenImageEdit"
            | "TextEncodeQwenImageEditPlus"
            | "ConditioningZeroOut"
            | "ConditioningSetMask"
            | "Reroute"
            | "SetNode"
            | "GetNode"
            | "LoadImage"
            | "LoadImageOutput"
            | "LoadImageMask"
            | "ImageToMask"
            | "MaskToImage"
            | "EmptyLatentImage"
            | "EmptySD3LatentImage"
            | "EmptyFlux2LatentImage"
            | "VAEEncode"
            | "SetLatentNoiseMask"
            | "GetImageSize"
            | "GetImageSizeAndCount"
            | "ImageScale"
            | "ImageScaleToTotalPixels"
            | "ImageScaleBy"
            | "ImageResizeKJ"
            | "ImagePadForOutpaint"
            | "ThresholdMask"
            | "InpaintModelConditioning"
            | "ReferenceLatent"
            | "6007e698-2ebd-4917-84d8-299b35d7b7ab"
            | "f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca"
            | "ModelSamplingAuraFlow"
            | "ModelSamplingSD3"
            | "DifferentialDiffusion"
            | "KSampler"
            | "KSamplerAdvanced"
            | "LanPaint_KSampler"
            | "LanPaint_KSamplerAdvanced"
            | "ConditioningCombine"
            | "ConditioningConcat"
            | "ConditioningAverage"
            | "CFGGuider"
            | "BasicGuider"
            | "FluxGuidance"
            | "Flux2Scheduler"
            | "BasicScheduler"
            | "RandomNoise"
            | "KSamplerSelect"
            | "SamplerEulerAncestral"
            | "SamplerDPMPP_2M_SDE"
            | "SamplerDPMPP_3M_SDE"
            | "SamplerLMS"
            | "KarrasScheduler"
            | "ExponentialScheduler"
            | "PolyexponentialScheduler"
            | "SDTurboScheduler"
            | "SamplerCustom"
            | "SamplerCustomAdvanced"
            | "LanPaint_SamplerCustomAdvanced"
            | "LanPaint_MaskBlend"
            | "ComfySwitchNode"
            | "VAEDecode"
            | "SaveImage"
            | "PreviewImage"
            | "MarkdownNote"
            | "Note"
            | "PrimitiveInt"
            | "PrimitiveFloat"
            | "PrimitiveString"
            | "PrimitiveStringMultiline"
            | "PrimitiveBoolean"
            | "PrimitiveNode"
            | "INTConstant"
            | "FloatConstant"
            | "StringConstant"
            | "StringConstantMultiline"
            | "BOOLConstant"
            | "SeedNode"
            | "easy int"
            | "easy float"
            | "easy string"
    )
}

/// `_workflow_id` (Mojo 521): int or parseable-string node id.
fn node_id(node: &JsonValue) -> GraphResult<i64> {
    let o = node
        .as_object()
        .ok_or_else(|| GraphError::unsupported("workflow graph node must be an object"))?;
    let id = o
        .get("id")
        .ok_or_else(|| GraphError::unsupported("workflow graph node missing id"))?;
    if let Some(n) = id.as_i64() {
        return Ok(n);
    }
    if let Some(s) = id.as_str() {
        if let Ok(n) = s.parse::<i64>() {
            return Ok(n);
        }
    }
    Err(GraphError::unsupported(
        "workflow graph node id must be an integer",
    ))
}

/// `_workflow_type_id` (Mojo 42): canonical `type_id` field.
fn node_type_id(node: &JsonValue) -> String {
    canonical_type_id(&wf_string(node, "type_id"))
}

/// `_workflow_ref_node` (Mojo 534): int or parseable-string endpoint node.
fn endpoint_node(ep: &JsonValue) -> GraphResult<i64> {
    let o = ep
        .as_object()
        .ok_or_else(|| GraphError::unsupported("workflow graph edge endpoint missing node"))?;
    let n = o
        .get("node")
        .ok_or_else(|| GraphError::unsupported("workflow graph edge endpoint missing node"))?;
    if let Some(v) = n.as_i64() {
        return Ok(v);
    }
    if let Some(s) = n.as_str() {
        if let Ok(v) = s.parse::<i64>() {
            return Ok(v);
        }
    }
    Err(GraphError::unsupported(
        "workflow graph edge endpoint node must be an integer",
    ))
}

/// `_workflow_ref_port` (Mojo 547): string endpoint port (required).
fn endpoint_port(ep: &JsonValue) -> GraphResult<String> {
    ep.as_object()
        .and_then(|o| o.get("port"))
        .and_then(JsonValue::as_str)
        .map(|s| s.to_string())
        .ok_or_else(|| GraphError::unsupported("workflow graph edge endpoint missing port"))
}

/// Parse + validate a `{nodes, edges}` body into a [`TypedGraph`].
///
/// Port of `apply_typed_workflow_graph`'s prologue (Mojo 1604-1725):
/// 1. require `nodes` and `edges` arrays,
/// 2. object nodes, unique ids, non-empty allowed `type_id`,
/// 3. parse edges into typed endpoints,
/// 4. SetNode names unique; every GetNode name references an existing SetNode.
pub fn parse_typed_graph(body: &JsonValue) -> GraphResult<TypedGraph> {
    let nodes_json = body
        .get("nodes")
        .filter(|v| v.is_array())
        .ok_or_else(|| {
            GraphError::unsupported("workflow graph body needs nodes or params/genparams")
        })?
        .as_array()
        .unwrap();
    let edges_json = body
        .get("edges")
        .filter(|v| v.is_array())
        .ok_or_else(|| {
            GraphError::unsupported("workflow graph body needs edges for typed execution")
        })?
        .as_array()
        .unwrap();

    let mut nodes = Vec::with_capacity(nodes_json.len());
    let mut seen_ids: HashSet<i64> = HashSet::new();
    for node in nodes_json {
        if !node.is_object() {
            return Err(GraphError::unsupported("workflow graph node must be an object"));
        }
        let id = node_id(node)?;
        if !seen_ids.insert(id) {
            return Err(GraphError::unsupported(format!(
                "workflow graph duplicate node id: {id}"
            ))
            .with_node(id));
        }
        let type_id = node_type_id(node);
        if type_id.is_empty() {
            return Err(GraphError::unsupported(
                "unsupported workflow graph format: missing type_id",
            )
            .with_node(id));
        }
        if !is_allowed_type(&type_id) {
            return Err(GraphError::unsupported(format!(
                "unsupported workflow graph node type: {type_id}"
            ))
            .with_node(id)
            .with_type(type_id));
        }
        let fields = node
            .get("fields")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| JsonValue::Object(Default::default()));
        nodes.push(WorkflowNode { id, type_id, fields });
    }

    let mut edges = Vec::with_capacity(edges_json.len());
    for edge in edges_json {
        let eo = edge.as_object().filter(|o| o.contains_key("from") && o.contains_key("to"));
        let eo = eo.ok_or_else(|| {
            GraphError::unsupported("workflow graph edge must have from/to endpoints")
        })?;
        let from = &eo["from"];
        let to = &eo["to"];
        edges.push(WorkflowEdge {
            from: EdgeEndpoint {
                node: endpoint_node(from)?,
                port: endpoint_port(from)?,
            },
            to: EdgeEndpoint {
                node: endpoint_node(to)?,
                port: endpoint_port(to)?,
            },
        });
    }

    // SetNode name uniqueness (Mojo 1699-1712).
    let mut setnode_names: Vec<String> = Vec::new();
    for node in &nodes {
        if node.type_id == "SetNode" {
            let name = setget_name(&node.fields);
            if name.is_empty() {
                return Err(GraphError::unsupported("workflow graph SetNode missing name")
                    .with_node(node.id));
            }
            if setnode_names.iter().any(|n| n == &name) {
                return Err(GraphError::unsupported(format!(
                    "workflow graph duplicate SetNode name: {name}"
                ))
                .with_node(node.id));
            }
            setnode_names.push(name);
        }
    }
    // GetNode names must reference an existing SetNode (Mojo 1714-1725).
    for node in &nodes {
        if node.type_id == "GetNode" {
            let name = setget_name(&node.fields);
            if name.is_empty() {
                return Err(GraphError::unsupported("workflow graph GetNode missing name")
                    .with_node(node.id));
            }
            if !setnode_names.iter().any(|n| n == &name) {
                return Err(GraphError::unsupported(format!(
                    "workflow graph GetNode missing SetNode: {name}"
                ))
                .with_node(node.id));
            }
        }
    }

    Ok(TypedGraph { nodes, edges })
}
