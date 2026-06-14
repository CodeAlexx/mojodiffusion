//! Field-coercion and type-classification helpers shared by the importer and
//! executor. Faithful ports of the `_workflow_*` free functions in
//! `workflow_graph.mojo` (lines 17-260, 36-110, 651-683, 943-972).

use serde_json::Value as JsonValue;

use crate::{GraphError, GraphResult};

/// `_workflow_canonical_type_id` (Mojo 36): strip a leading `comfy/` namespace.
pub fn canonical_type_id(type_id: &str) -> String {
    type_id.strip_prefix("comfy/").unwrap_or(type_id).to_string()
}

/// `_workflow_string` (Mojo 28): string field or `""` (non-strings → "").
pub fn wf_string(obj: &JsonValue, key: &str) -> String {
    obj.as_object()
        .and_then(|o| o.get(key))
        .and_then(JsonValue::as_str)
        .unwrap_or("")
        .to_string()
}

// --- scalar node classification (Mojo 46-110) ----------------------------------

pub fn is_int_scalar_node(t: &str) -> bool {
    matches!(t, "PrimitiveInt" | "INTConstant" | "easy int" | "SeedNode")
}

pub fn is_float_scalar_node(t: &str) -> bool {
    matches!(t, "PrimitiveFloat" | "FloatConstant" | "easy float")
}

pub fn is_string_scalar_node(t: &str) -> bool {
    matches!(
        t,
        "PrimitiveString"
            | "PrimitiveStringMultiline"
            | "StringConstant"
            | "StringConstantMultiline"
            | "easy string"
    )
}

pub fn is_bool_scalar_node(t: &str) -> bool {
    matches!(t, "PrimitiveBoolean" | "BOOLConstant")
}

pub fn is_scalar_node(t: &str) -> bool {
    is_int_scalar_node(t)
        || is_float_scalar_node(t)
        || is_string_scalar_node(t)
        || is_bool_scalar_node(t)
        || t == "PrimitiveNode"
}

/// `_workflow_json_scalar_type` (Mojo 78).
pub fn json_scalar_type(v: &JsonValue) -> &'static str {
    if v.is_boolean() {
        "BOOLEAN"
    } else if v.is_i64() || v.is_u64() {
        "INT"
    } else if v.is_number() {
        "FLOAT"
    } else if v.is_string() {
        "STRING"
    } else {
        ""
    }
}

/// `_workflow_scalar_output_type` (Mojo 90).
pub fn scalar_output_type(type_id: &str, fields: &JsonValue) -> String {
    if is_int_scalar_node(type_id) {
        return "INT".to_string();
    }
    if is_float_scalar_node(type_id) {
        return "FLOAT".to_string();
    }
    if is_string_scalar_node(type_id) {
        return "STRING".to_string();
    }
    if is_bool_scalar_node(type_id) {
        return "BOOLEAN".to_string();
    }
    if type_id == "PrimitiveNode" {
        let mut declared = wf_string(fields, "output_type");
        if declared.is_empty() {
            if let Some(o) = fields.as_object() {
                if let Some(v) = o.get("value") {
                    declared = json_scalar_type(v).to_string();
                } else if let Some(v) = o.get("text") {
                    declared = json_scalar_type(v).to_string();
                } else if let Some(v) = o.get("string") {
                    declared = json_scalar_type(v).to_string();
                }
            }
        }
        if matches!(declared.as_str(), "INT" | "FLOAT" | "STRING" | "BOOLEAN") {
            return declared;
        }
    }
    String::new()
}

// --- node / widget readers for the canvas + ideogram4 importers ---------------
// (ports of `_workflow_node_type`/`_workflow_node_mode`/`_workflow_id`/
//  `_json_intish`/`_workflow_widget_*` in workflow_graph.mojo 204-276, 521, 1002).

/// `_workflow_node_type` (Mojo 204): `type` field, falling back to `type_id`.
pub fn node_type(node: &JsonValue) -> String {
    let typ = wf_string(node, "type");
    if typ.is_empty() {
        wf_string(node, "type_id")
    } else {
        typ
    }
}

/// `_workflow_node_mode` (Mojo 211): integer `mode` (default 0; floats truncate).
pub fn node_mode(node: &JsonValue) -> i64 {
    let v = match node.as_object().and_then(|o| o.get("mode")) {
        Some(v) if !v.is_null() => v,
        _ => return 0,
    };
    if let Some(i) = v.as_i64() {
        i
    } else if let Some(f) = v.as_f64() {
        f as i64
    } else {
        0
    }
}

/// `_workflow_id` (Mojo 521): int or parseable-string node id.
pub fn node_id(node: &JsonValue) -> GraphResult<i64> {
    let o = node
        .as_object()
        .ok_or_else(|| GraphError::unsupported("workflow graph node missing id"))?;
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

/// `_json_intish` (Mojo 1002): int, truncated number, or parseable string.
pub fn json_intish(v: &JsonValue, label: &str) -> GraphResult<i64> {
    if let Some(n) = v.as_i64() {
        return Ok(n);
    }
    if let Some(f) = v.as_f64() {
        return Ok(f as i64);
    }
    if let Some(s) = v.as_str() {
        if let Ok(n) = s.parse::<i64>() {
            return Ok(n);
        }
    }
    Err(GraphError::unsupported(format!(
        "Comfy UI canvas {label} must be an integer"
    )))
}

/// `_workflow_widget_string` (Mojo 221): widget at `idx` coerced to a string,
/// else `dflt`. Ints/floats stringify (floats via the same rust→`{}` path the
/// Mojo `String(Float64)` uses for the values we capture).
pub fn widget_string(widgets: &JsonValue, idx: usize, dflt: &str) -> String {
    let arr = match widgets.as_array() {
        Some(a) if idx < a.len() && !a[idx].is_null() => a,
        _ => return dflt.to_string(),
    };
    let v = &arr[idx];
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    if let Some(i) = v.as_i64() {
        return i.to_string();
    }
    if let Some(f) = v.as_f64() {
        return f.to_string();
    }
    dflt.to_string()
}

/// `_workflow_widget_int` (Mojo 233): widget at `idx` as int (number truncates,
/// string parses), else `dflt`.
pub fn widget_int(widgets: &JsonValue, idx: usize, dflt: i64) -> i64 {
    let arr = match widgets.as_array() {
        Some(a) if idx < a.len() && !a[idx].is_null() => a,
        _ => return dflt,
    };
    let v = &arr[idx];
    if let Some(i) = v.as_i64() {
        return i;
    }
    if let Some(f) = v.as_f64() {
        return f as i64;
    }
    if let Some(s) = v.as_str() {
        if let Ok(i) = s.parse::<i64>() {
            return i;
        }
    }
    dflt
}

/// `_workflow_widget_float` (Mojo 248): widget at `idx` as f64, else `dflt`.
pub fn widget_float(widgets: &JsonValue, idx: usize, dflt: f64) -> f64 {
    let arr = match widgets.as_array() {
        Some(a) if idx < a.len() && !a[idx].is_null() => a,
        _ => return dflt,
    };
    let v = &arr[idx];
    if let Some(f) = v.as_f64() {
        return f;
    }
    if let Some(i) = v.as_i64() {
        return i as f64;
    }
    if let Some(s) = v.as_str() {
        if let Ok(f) = s.parse::<f64>() {
            return f;
        }
    }
    dflt
}

/// `_workflow_widget_bool` (Mojo 263): widget at `idx` as bool, else `dflt`.
pub fn widget_bool(widgets: &JsonValue, idx: usize, dflt: bool) -> bool {
    let arr = match widgets.as_array() {
        Some(a) if idx < a.len() && !a[idx].is_null() => a,
        _ => return dflt,
    };
    let v = &arr[idx];
    if let Some(b) = v.as_bool() {
        return b;
    }
    if let Some(i) = v.as_i64() {
        return i != 0;
    }
    if let Some(s) = v.as_str() {
        let l = s.to_lowercase();
        if matches!(l.as_str(), "true" | "yes" | "1") {
            return true;
        }
        if matches!(l.as_str(), "false" | "no" | "0") {
            return false;
        }
    }
    dflt
}

// --- numeric / bool coercion (Mojo 125-292) ------------------------------------

/// `_opt_int` (Mojo 17): bounded int with default; non-int (incl. floats) is an
/// error. (NOTE: this is the stricter form used by the latent/pad handlers.)
pub fn opt_int(obj: &JsonValue, key: &str, dflt: i64, lo: i64, hi: i64) -> GraphResult<i64> {
    let o = match obj.as_object() {
        Some(o) => o,
        None => return Ok(dflt),
    };
    let v = match o.get(key) {
        Some(v) if !v.is_null() => v,
        _ => return Ok(dflt),
    };
    let n = v
        .as_i64()
        .ok_or_else(|| GraphError::bad_request(format!("'{key}' must be an integer")))?;
    if n < lo || n > hi {
        return Err(GraphError::bad_request(format!(
            "'{key}' out of range [{lo}..{hi}]"
        )));
    }
    Ok(n)
}

/// `_workflow_float` (Mojo 125): float field accepting number/int/parseable
/// string, range-checked.
pub fn wf_float(obj: &JsonValue, key: &str, dflt: f64, lo: f64, hi: f64) -> GraphResult<f64> {
    let o = match obj.as_object() {
        Some(o) => o,
        None => return Ok(dflt),
    };
    let v = match o.get(key) {
        Some(v) if !v.is_null() => v,
        _ => return Ok(dflt),
    };
    let n = if let Some(f) = v.as_f64() {
        f
    } else if let Some(i) = v.as_i64() {
        i as f64
    } else if let Some(s) = v.as_str() {
        s.parse::<f64>().map_err(|_| {
            GraphError::unsupported(format!("workflow graph field {key} must be numeric"))
        })?
    } else {
        return Err(GraphError::unsupported(format!(
            "workflow graph field {key} must be numeric"
        )));
    };
    if n < lo || n > hi {
        return Err(GraphError::unsupported(format!(
            "workflow graph field {key} out of range"
        )));
    }
    Ok(n)
}

/// `_workflow_bool` (Mojo 279).
pub fn wf_bool(obj: &JsonValue, key: &str, dflt: bool) -> GraphResult<bool> {
    let o = match obj.as_object() {
        Some(o) => o,
        None => return Ok(dflt),
    };
    let v = match o.get(key) {
        Some(v) if !v.is_null() => v,
        _ => return Ok(dflt),
    };
    if let Some(b) = v.as_bool() {
        return Ok(b);
    }
    if let Some(i) = v.as_i64() {
        return Ok(i != 0);
    }
    if let Some(s) = v.as_str() {
        let l = s.to_lowercase();
        if matches!(l.as_str(), "true" | "yes" | "1") {
            return Ok(true);
        }
        if matches!(l.as_str(), "false" | "no" | "0") {
            return Ok(false);
        }
    }
    Err(GraphError::unsupported(format!(
        "workflow graph field {key} must be a boolean"
    )))
}

/// `_workflow_round6` (Mojo 169): round-half-away-from-zero to 6 decimals.
pub fn round6(v: f64) -> f64 {
    let scaled = v * 1_000_000.0;
    if scaled >= 0.0 {
        ((scaled + 0.5) as i64) as f64 / 1_000_000.0
    } else {
        ((scaled - 0.5) as i64) as f64 / 1_000_000.0
    }
}

// --- per-handler field helpers (Mojo 943-972, 828-840) -------------------------

/// `_workflow_loader_model_name` (Mojo 943).
pub fn loader_model_name(fields: &JsonValue) -> String {
    let mut name = wf_string(fields, "ckpt_name");
    if name.is_empty() {
        name = wf_string(fields, "unet_name");
    }
    if name.is_empty() {
        name = wf_string(fields, "model_name");
    }
    name
}

/// `_workflow_conditioning_prompt_text` (Mojo 952).
pub fn conditioning_prompt_text(fields: &JsonValue, type_id: &str) -> String {
    let text = wf_string(fields, "text");
    if !text.is_empty() {
        return text;
    }
    if type_id == "CLIPTextEncodeFlux" {
        let t5 = wf_string(fields, "t5xxl");
        if !t5.is_empty() {
            return t5;
        }
        return wf_string(fields, "clip_l");
    }
    text
}

/// `_workflow_setget_name` (Mojo 964).
pub fn setget_name(fields: &JsonValue) -> String {
    let mut name = wf_string(fields, "name");
    if name.is_empty() {
        name = wf_string(fields, "variable");
    }
    if name.is_empty() {
        name = wf_string(fields, "key");
    }
    if name.is_empty() {
        name = wf_string(fields, "set_name");
    }
    name
}

/// `_workflow_imagetomask_channel` (Mojo 828).
pub fn imagetomask_channel(fields: &JsonValue) -> GraphResult<String> {
    let mut channel = wf_string(fields, "channel");
    if channel.is_empty() {
        channel = "red".to_string();
    }
    let lower = channel.to_lowercase();
    if matches!(lower.as_str(), "red" | "green" | "blue") {
        return Ok(lower);
    }
    if lower == "alpha" {
        return Err(GraphError::unsupported(
            "workflow graph ImageToMask alpha is unsupported on Comfy RGB IMAGE; use LoadImage MASK",
        ));
    }
    Err(GraphError::unsupported(format!(
        "workflow graph ImageToMask unsupported channel: {channel}"
    )))
}

/// `_workflow_type_accepts` (Mojo 651): EXACT comma-token membership (audit fix
/// — not substring). `declared` may be `""`/`"*"`/exact, or a comma-separated
/// list; membership tolerates a single space after a comma on either side.
pub fn type_accepts(declared: &str, actual: &str) -> bool {
    if declared.is_empty() || declared == "*" || declared == actual {
        return true;
    }
    let wrapped = format!(",{declared},");
    if wrapped.contains(&format!(",{actual},")) {
        return true;
    }
    if wrapped.contains(&format!(", {actual},")) {
        return true;
    }
    wrapped.contains(&format!(",{actual}, "))
}

/// `_workflow_setget_supported_type` (Mojo 665).
pub fn setget_supported_type(actual: &str) -> bool {
    matches!(
        actual,
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
            | "COND_LATENT"
            | "INT"
            | "FLOAT"
            | "STRING"
            | "BOOLEAN"
    )
}
