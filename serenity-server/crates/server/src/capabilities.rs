use serde_json::{json, Value as JsonValue};
use serenity_wire::JobParams;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    ZImage,
    Ideogram4,
    Sdxl,
    Anima,
    Sd3,
    Flux,
    Flux2,
    Sensenova,
}

impl ModelFamily {
    pub(crate) fn backend_key(self) -> &'static str {
        match self {
            ModelFamily::ZImage => "zimage",
            ModelFamily::Ideogram4 => "ideogram4",
            ModelFamily::Sdxl => "sdxl",
            ModelFamily::Anima => "anima",
            ModelFamily::Sd3 => "sd3",
            ModelFamily::Flux => "flux",
            ModelFamily::Flux2 => "flux2",
            ModelFamily::Sensenova => "sensenova",
        }
    }

    pub(crate) fn worker_binary_name(self) -> &'static str {
        match self {
            ModelFamily::ZImage => "serenity_worker_zimage",
            ModelFamily::Ideogram4 => "serenity_worker_ideogram4",
            ModelFamily::Sdxl => "serenity_worker_sdxl",
            ModelFamily::Anima => "serenity_worker_anima",
            ModelFamily::Sd3 => "serenity_worker_sd3",
            ModelFamily::Flux => "serenity_worker_flux",
            ModelFamily::Flux2 => "serenity_worker_klein",
            ModelFamily::Sensenova => "serenity_worker_sensenova",
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct BlockedModelInfo {
    backend: &'static str,
    production_status: &'static str,
    reason: &'static str,
}

fn blocked_model_info(normalized_model: &str) -> Option<BlockedModelInfo> {
    let m = normalized_model;
    if m.contains("qwen") && m.contains("edit") {
        return Some(BlockedModelInfo {
            backend: "qwenimage_edit",
            production_status: "blocked",
            reason: "Qwen-Image-Edit is known to the runtime docs, but the Rust server path has no production edit/image-conditioning gate yet",
        });
    }
    if m.contains("qwen") {
        return Some(BlockedModelInfo {
            backend: "qwenimage",
            production_status: "metadata/preflight-only",
            reason: "Qwen/Qwen-Image execution is metadata/preflight-only in serenity-server until a production artifact, timing, VRAM, and sampler gate passes",
        });
    }
    if m.contains("zimage_l2p")
        || m.contains("z-image-l2p")
        || m.contains("z_image_l2p")
        || m.contains("l2p")
    {
        return Some(BlockedModelInfo {
            backend: "zimage_l2p",
            production_status: "blocked",
            reason: "Z-Image L2P has Mojo smoke/runtime pieces, but no production serenity-server worker route yet",
        });
    }
    if m.contains("klein") && m.contains("4b") {
        return Some(BlockedModelInfo {
            backend: "flux2",
            production_status: "blocked",
            reason: "Klein/Flux2 4B is known to the runtime docs, but only the 9B txt2img route has current Rust-server product evidence",
        });
    }
    if (m.contains("flux2") || m.contains("flux-2") || m.contains("flux_2")) && !m.contains("klein")
    {
        return Some(BlockedModelInfo {
            backend: "flux2",
            production_status: "blocked",
            reason: "Flux2 generation is admitted only for the bounded Klein 9B txt2img route; generic Flux2 model names remain blocked",
        });
    }
    if m.contains("lens") || m.contains("microsoft_lens") || m.contains("microsoft-lens") {
        return Some(BlockedModelInfo {
            backend: "microsoft_lens",
            production_status: "blocked",
            reason: "Microsoft Lens is compiled experimentally, but is not production-admitted until the render/OOM gate passes",
        });
    }
    if m.contains("wan")
        || m.contains("lance")
        || m.contains("ltx")
        || m.contains("nava")
        || m.contains("video")
    {
        return Some(BlockedModelInfo {
            backend: "video",
            production_status: "bounded_elsewhere",
            reason: "video/audio model families are not admitted through /v1/generate; use the bounded video endpoints/gates instead",
        });
    }
    if m.contains("hidream") || m.contains("hi-dream") || m.contains("hi_dream") {
        return Some(BlockedModelInfo {
            backend: "hidream",
            production_status: "blocked",
            reason: "HiDream is not production-admitted in serenity-server yet",
        });
    }
    None
}

pub(crate) fn model_family(model: &str) -> Result<ModelFamily, String> {
    let m = model.trim().to_ascii_lowercase();
    if m.is_empty() || m.contains("select model") {
        return Err("model is required".to_string());
    }
    if let Some(info) = blocked_model_info(&m) {
        return Err(info.reason.to_string());
    }
    if m.contains("ideogram") {
        return Ok(ModelFamily::Ideogram4);
    }
    if m.contains("sdxl")
        || m.contains("sd_xl")
        || m.contains("sd-xl")
        || m.contains("sd xl")
        || m.contains("stable-diffusion-xl")
        || m.contains("animagine")
    {
        return Ok(ModelFamily::Sdxl);
    }
    if m.contains("anima") {
        return Ok(ModelFamily::Anima);
    }
    if m.contains("sd3") || m.contains("sd35") || m.contains("sd3.5") {
        return Ok(ModelFamily::Sd3);
    }
    if m.contains("klein") {
        return Ok(ModelFamily::Flux2);
    }
    if m.contains("flux") {
        return Ok(ModelFamily::Flux);
    }
    if m.contains("sensenova") || m.contains("sense_nova") || m.contains("sense-nova") {
        return Ok(ModelFamily::Sensenova);
    }
    if m.contains("zimage") || m.contains("z-image") || m.contains("z_image") {
        return Ok(ModelFamily::ZImage);
    }
    Err(format!(
        "unsupported model family for production generation: {model}; add an explicit Rust route, Mojo backend gate, and UI capability entry before exposing it"
    ))
}

fn request_model_name(obj: &JsonValue) -> &str {
    obj.get("model").and_then(JsonValue::as_str).unwrap_or("")
}

pub(crate) fn normalize_sampler_name(name: &str) -> String {
    let n = name.trim().to_ascii_lowercase();
    match n.as_str() {
        "" => String::new(),
        "flow_match_euler" | "flowmatch_euler" | "flow match euler" => {
            "flowmatch_euler".to_string()
        }
        "dpm++ 2m" | "dpmpp 2m" | "dpmpp_2m" => "dpmpp_2m".to_string(),
        "uni-pc" | "unipc" | "uni_pc" => "uni_pc".to_string(),
        "uni-pc bh2" | "unipc_bh2" | "uni_pc_bh2" => "uni_pc_bh2".to_string(),
        other => other.to_string(),
    }
}

pub(crate) fn normalize_scheduler_name(name: &str) -> String {
    let n = name.trim().to_ascii_lowercase();
    match n.as_str() {
        "" => String::new(),
        "flow_match" | "flowmatch" | "simple_flowmatch" => "simple".to_string(),
        "logitnormal" | "logit_normal" | "ideogram_logitnormal" | "ideogram4_logitnormal" => {
            "ideogram_logitnormal".to_string()
        }
        "qwen_flowmatch" => "simple".to_string(),
        other => other.to_string(),
    }
}

fn default_sampler_for_family(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::ZImage
        | ModelFamily::Ideogram4
        | ModelFamily::Sdxl
        | ModelFamily::Anima
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Sensenova => "euler",
    }
}

fn default_scheduler_for_family(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::ZImage
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Sensenova => "simple",
        ModelFamily::Ideogram4 => "ideogram_logitnormal",
        ModelFamily::Sdxl | ModelFamily::Anima => "normal",
    }
}

fn default_size_for_family(family: ModelFamily) -> (i64, i64) {
    match family {
        ModelFamily::Flux2 => (512, 512),
        _ => (1024, 1024),
    }
}

fn default_steps_for_family(family: ModelFamily) -> i64 {
    match family {
        ModelFamily::ZImage => 16,
        ModelFamily::Ideogram4 => 20,
        ModelFamily::Sdxl | ModelFamily::Anima | ModelFamily::Flux => 20,
        ModelFamily::Sd3 => 28,
        ModelFamily::Flux2 => 4,
        ModelFamily::Sensenova => 30,
    }
}

fn default_cfg_for_family(family: ModelFamily) -> f64 {
    match family {
        ModelFamily::ZImage => 5.0,
        ModelFamily::Ideogram4 | ModelFamily::Sdxl => 7.0,
        ModelFamily::Anima | ModelFamily::Sd3 => 4.5,
        ModelFamily::Flux | ModelFamily::Flux2 | ModelFamily::Sensenova => 4.0,
    }
}

const ZIMAGE_SIZES: &[(i64, i64)] = &[(512, 512), (1024, 1024)];
const KLEIN_SIZES: &[(i64, i64)] = &[(512, 512)];
const SENSENOVA_SIZES: &[(i64, i64)] = &[(512, 512), (1024, 1024)];
const SIZE_1024: &[(i64, i64)] = &[(1024, 1024)];
const SAMPLERS_EULER: &[&str] = &["euler"];
const SAMPLERS_EULER_FLOWMATCH: &[&str] = &["euler", "flowmatch_euler"];
const SAMPLERS_ZIMAGE: &[&str] = &[
    "euler",
    "flowmatch_euler",
    "dpmpp_2m",
    "uni_pc",
    "uni_pc_bh2",
];
const SCHEDULERS_SIMPLE: &[&str] = &["simple"];
const SCHEDULERS_NORMAL: &[&str] = &["normal"];
const SCHEDULERS_ZIMAGE: &[&str] = &["simple", "sgm_uniform"];
const SCHEDULERS_IDEOGRAM4: &[&str] = &["ideogram_logitnormal", "simple"];

#[derive(Debug, Copy, Clone)]
struct ResolutionPolicy {
    mode: &'static str,
    min_width: i64,
    max_width: i64,
    min_height: i64,
    max_height: i64,
    multiple: i64,
    square_only: bool,
    admitted_shapes: &'static [(i64, i64)],
    note: &'static str,
}

fn resolution_policy_for_family(family: ModelFamily) -> ResolutionPolicy {
    let admitted_shapes = production_sizes_for_family(family);
    match family {
        ModelFamily::ZImage => ResolutionPolicy {
            mode: "shape_dispatch",
            min_width: 512,
            max_width: 1024,
            min_height: 512,
            max_height: 1024,
            multiple: 512,
            square_only: true,
            admitted_shapes,
            note: "current product worker admits compiled 512 and 1024 square shapes",
        },
        ModelFamily::Ideogram4
        | ModelFamily::Sdxl
        | ModelFamily::Anima
        | ModelFamily::Sd3
        | ModelFamily::Flux => ResolutionPolicy {
            mode: "single_product_shape",
            min_width: 1024,
            max_width: 1024,
            min_height: 1024,
            max_height: 1024,
            multiple: 1024,
            square_only: true,
            admitted_shapes,
            note: "current product worker has one admitted shape; add worker shape dispatch before exposing more sizes",
        },
        ModelFamily::Flux2 => ResolutionPolicy {
            mode: "single_product_shape",
            min_width: 512,
            max_width: 512,
            min_height: 512,
            max_height: 512,
            multiple: 512,
            square_only: true,
            admitted_shapes,
            note: "current Klein product worker has one admitted shape; add worker shape dispatch before exposing more sizes",
        },
        ModelFamily::Sensenova => ResolutionPolicy {
            mode: "shape_dispatch",
            min_width: 512,
            max_width: 1024,
            min_height: 512,
            max_height: 1024,
            multiple: 512,
            square_only: true,
            admitted_shapes,
            note: "SenseNova worker dispatches concrete compiled image-token shapes; add specializations before exposing more workflow resolutions",
        },
    }
}

fn production_sizes_for_family(family: ModelFamily) -> &'static [(i64, i64)] {
    match family {
        ModelFamily::ZImage => ZIMAGE_SIZES,
        ModelFamily::Ideogram4
        | ModelFamily::Sdxl
        | ModelFamily::Anima
        | ModelFamily::Sd3
        | ModelFamily::Flux => SIZE_1024,
        ModelFamily::Flux2 => KLEIN_SIZES,
        ModelFamily::Sensenova => SENSENOVA_SIZES,
    }
}

fn supported_samplers_for_family(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::ZImage => SAMPLERS_ZIMAGE,
        ModelFamily::Ideogram4 | ModelFamily::Sd3 | ModelFamily::Flux => SAMPLERS_EULER_FLOWMATCH,
        ModelFamily::Flux2 | ModelFamily::Sensenova => SAMPLERS_EULER,
        ModelFamily::Sdxl | ModelFamily::Anima => SAMPLERS_EULER,
    }
}

fn supported_schedulers_for_family(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::ZImage => SCHEDULERS_ZIMAGE,
        ModelFamily::Ideogram4 => SCHEDULERS_IDEOGRAM4,
        ModelFamily::Sdxl | ModelFamily::Anima => SCHEDULERS_NORMAL,
        ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Sensenova => SCHEDULERS_SIMPLE,
    }
}

fn supports_negative_prompt(family: ModelFamily) -> bool {
    !matches!(
        family,
        ModelFamily::Ideogram4 | ModelFamily::Flux | ModelFamily::Flux2 | ModelFamily::Sensenova
    )
}

fn lora_limit_for_family(family: ModelFamily) -> Option<usize> {
    match family {
        ModelFamily::ZImage => None,
        ModelFamily::Flux | ModelFamily::Flux2 => Some(1),
        ModelFamily::Ideogram4
        | ModelFamily::Sdxl
        | ModelFamily::Anima
        | ModelFamily::Sd3
        | ModelFamily::Sensenova => Some(0),
    }
}

fn validate_lora_capability(params: &JobParams, family: ModelFamily) -> Result<(), String> {
    match lora_limit_for_family(family) {
        Some(0) => reject_loras(params, family),
        Some(limit) if params.loras.len() > limit => Err(format!(
            "{}: at most {limit} LoRA overlay is production-wired per job",
            family.backend_key()
        )),
        _ => Ok(()),
    }
}

pub(crate) fn requested_sampler(params: &JobParams, family: ModelFamily) -> String {
    let n = normalize_sampler_name(&params.sampler);
    if n.is_empty() {
        default_sampler_for_family(family).to_string()
    } else {
        n
    }
}

pub(crate) fn requested_scheduler(params: &JobParams, family: ModelFamily) -> String {
    let n = normalize_scheduler_name(&params.scheduler);
    if n.is_empty() {
        default_scheduler_for_family(family).to_string()
    } else {
        n
    }
}

fn one_of(value: &str, allowed: &[&str]) -> bool {
    allowed.iter().any(|v| *v == value)
}

pub(crate) fn has_text(value: &str) -> bool {
    !value.trim().is_empty()
}

pub(crate) fn has_vae_override(value: &str) -> bool {
    let v = value.trim();
    if v.is_empty() {
        return false;
    }
    let n = v.to_ascii_lowercase();
    !matches!(
        n.as_str(),
        "automatic" | "auto" | "default" | "baked" | "baked-in" | "baked_in"
    )
}

fn require_resolution(
    params: &JobParams,
    family: ModelFamily,
    policy: ResolutionPolicy,
) -> Result<(), String> {
    if policy.square_only && params.width != params.height {
        return Err(format!(
            "{}: unsupported size {}x{}; current product policy requires square output; admitted product shapes: {}",
            family.backend_key(),
            params.width,
            params.height,
            supported_size_string(policy.admitted_shapes)
        ));
    }
    if params.width < policy.min_width
        || params.width > policy.max_width
        || params.height < policy.min_height
        || params.height > policy.max_height
    {
        return Err(format!(
            "{}: unsupported size {}x{}; current product range is {}-{} wide by {}-{} high; admitted product shapes: {}",
            family.backend_key(),
            params.width,
            params.height,
            policy.min_width,
            policy.max_width,
            policy.min_height,
            policy.max_height,
            supported_size_string(policy.admitted_shapes)
        ));
    }
    if policy.multiple > 1
        && (params.width % policy.multiple != 0 || params.height % policy.multiple != 0)
    {
        return Err(format!(
            "{}: unsupported size {}x{}; current product policy requires dimensions divisible by {}; admitted product shapes: {}",
            family.backend_key(),
            params.width,
            params.height,
            policy.multiple,
            supported_size_string(policy.admitted_shapes)
        ));
    }
    if policy
        .admitted_shapes
        .iter()
        .any(|(w, h)| params.width == *w && params.height == *h)
    {
        return Ok(());
    }
    Err(format!(
        "{}: unsupported size {}x{}; admitted product shapes: {}",
        family.backend_key(),
        params.width,
        params.height,
        supported_size_string(policy.admitted_shapes)
    ))
}

fn supported_size_string(sizes: &[(i64, i64)]) -> String {
    sizes
        .iter()
        .map(|(w, h)| format!("{w}x{h}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn reject_loras(params: &JobParams, family: ModelFamily) -> Result<(), String> {
    if params.loras.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{}: LoRA is not production-wired for this backend; remove LoRA overlays",
            family.backend_key()
        ))
    }
}

fn reject_negative(params: &JobParams, family: ModelFamily) -> Result<(), String> {
    if has_text(&params.negative) {
        Err(format!(
            "{}: negative prompt is not supported by this production route",
            family.backend_key()
        ))
    } else {
        Ok(())
    }
}

fn reject_variation(params: &JobParams, family: ModelFamily) -> Result<(), String> {
    if params.variation_strength > 0.0 {
        Err(format!(
            "{}: variation noise is not supported by this production route",
            family.backend_key()
        ))
    } else {
        Ok(())
    }
}

fn validate_sampler_scheduler(
    params: &JobParams,
    family: ModelFamily,
    samplers: &[&str],
    schedulers: &[&str],
) -> Result<(String, String), String> {
    let sampler = requested_sampler(params, family);
    let scheduler = requested_scheduler(params, family);
    if !one_of(&sampler, samplers) {
        return Err(format!(
            "{}: unsupported sampler '{}'; supported: {}",
            family.backend_key(),
            if params.sampler.trim().is_empty() {
                "<default>"
            } else {
                params.sampler.trim()
            },
            samplers.join(", ")
        ));
    }
    if !one_of(&scheduler, schedulers) {
        return Err(format!(
            "{}: unsupported scheduler '{}'; supported: {}",
            family.backend_key(),
            if params.scheduler.trim().is_empty() {
                "<default>"
            } else {
                params.scheduler.trim()
            },
            schedulers.join(", ")
        ));
    }
    Ok((sampler, scheduler))
}

pub(crate) fn json_prompt_to_string(value: &JsonValue, field_name: &str) -> Result<String, String> {
    if let Some(s) = value.as_str() {
        return Ok(s.to_string());
    }
    if value.is_object() || value.is_array() {
        return serde_json::to_string(value)
            .map_err(|e| format!("{field_name} could not be serialized: {e}"));
    }
    Err(format!(
        "{field_name} must be a string or JSON object/array"
    ))
}

fn request_model_is_ideogram4(obj: &JsonValue) -> bool {
    request_model_name(obj)
        .to_ascii_lowercase()
        .contains("ideogram")
}

pub(crate) fn normalize_ideogram4_prompt_json(obj: &mut JsonValue) -> Result<(), String> {
    if !request_model_is_ideogram4(obj) {
        return Ok(());
    }
    let Some(map) = obj.as_object_mut() else {
        return Ok(());
    };
    let raw = if let Some(value) = map.get("prompt_json").filter(|v| !v.is_null()) {
        json_prompt_to_string(value, "prompt_json")?
    } else if let Some(value) = map.get("prompt_raw").and_then(JsonValue::as_str) {
        value.to_string()
    } else {
        return Ok(());
    };
    if raw.trim().is_empty() {
        return Err("Ideogram4 prompt_json/prompt_raw must be non-empty".to_string());
    }
    map.insert("prompt".to_string(), JsonValue::String(raw.clone()));
    map.insert("prompt_raw".to_string(), JsonValue::String(raw));
    Ok(())
}

fn raw_json_is_meaningful(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => false,
        JsonValue::Bool(v) => *v,
        JsonValue::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(true),
        JsonValue::String(s) => !s.trim().is_empty(),
        JsonValue::Array(items) => !items.is_empty(),
        JsonValue::Object(map) => !map.is_empty(),
    }
}

fn raw_string_is_set(map: &serde_json::Map<String, JsonValue>, key: &str) -> bool {
    map.get(key)
        .and_then(JsonValue::as_str)
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
}

fn raw_number_gt(map: &serde_json::Map<String, JsonValue>, key: &str, threshold: f64) -> bool {
    map.get(key)
        .and_then(JsonValue::as_f64)
        .map(|v| v.is_finite() && v > threshold)
        .unwrap_or(false)
}

fn raw_number_gte(map: &serde_json::Map<String, JsonValue>, key: &str, threshold: f64) -> bool {
    map.get(key)
        .and_then(JsonValue::as_f64)
        .map(|v| v.is_finite() && v >= threshold)
        .unwrap_or(false)
}

fn raw_bool_is_true(map: &serde_json::Map<String, JsonValue>, key: &str) -> bool {
    map.get(key).and_then(JsonValue::as_bool).unwrap_or(false)
}

fn raw_any_string_is_set(map: &serde_json::Map<String, JsonValue>, keys: &[&str]) -> bool {
    keys.iter().any(|key| raw_string_is_set(map, key))
}

fn raw_any_bool_is_true(map: &serde_json::Map<String, JsonValue>, keys: &[&str]) -> bool {
    keys.iter().any(|key| raw_bool_is_true(map, key))
}

fn raw_any_number_gte(
    map: &serde_json::Map<String, JsonValue>,
    keys: &[&str],
    threshold: f64,
) -> bool {
    keys.iter().any(|key| raw_number_gte(map, key, threshold))
}

pub(crate) fn reject_disabled_raw_surfaces(obj: &JsonValue) -> Result<(), String> {
    let Some(map) = obj.as_object() else {
        return Ok(());
    };

    if map.get("prompt_json").is_some_and(|v| !v.is_null()) && !request_model_is_ideogram4(obj) {
        return Err(
            "prompt_json/bbox prompts are admitted only by the Ideogram4 production route"
                .to_string(),
        );
    }

    if raw_string_is_set(map, "mask_image")
        || raw_any_string_is_set(
            map,
            &["inpaint_conditioning_image", "inpaint_conditioning_mask"],
        )
        || raw_bool_is_true(map, "inpaint_conditioning_noise_mask")
    {
        return Err(
            "inpaint is not production-admitted in the current /v1/generate route".to_string(),
        );
    }
    if raw_string_is_set(map, "init_image") {
        return Err(
            "image-to-image is not production-admitted in the current /v1/generate route"
                .to_string(),
        );
    }
    if raw_any_string_is_set(
        map,
        &[
            "conditioning_mask_image",
            "qwen_edit_conditioning_image",
            "reference_image",
        ],
    ) || raw_any_number_gte(map, &["conditioning_mask_strength"], 0.0)
        || raw_any_bool_is_true(map, &["conditioning_mask_set_area_to_bounds"])
    {
        return Err(
            "image conditioning is not production-admitted in the current /v1/generate route"
                .to_string(),
        );
    }
    if raw_any_string_is_set(
        map,
        &[
            "sample_caps_pos",
            "sample_caps_neg",
            "caps_pos",
            "caps_neg",
            "caps_positive",
            "caps_negative",
        ],
    ) {
        return Err(
            "conditioning caps are not production-admitted in the current /v1/generate route"
                .to_string(),
        );
    }
    if map
        .get("vae")
        .and_then(JsonValue::as_str)
        .is_some_and(has_vae_override)
    {
        return Err(
            "VAE override is not production-wired for /v1/generate; current routes use the baked local VAE from each model manifest".to_string(),
        );
    }
    if raw_number_gt(map, "hires_scale", 1.0) {
        return Err(
            "hires two-pass currently depends on img2img refine and is disabled in the production /v1/generate path".to_string(),
        );
    }
    if raw_number_gt(map, "images", 1.0) {
        return Err(
            "serenity-server currently admits one image per /v1/generate job; batch fanout must be wired before exposing images>1".to_string(),
        );
    }

    for key in ["controlnet", "refiner", "upscaler", "outpaint"] {
        if map.get(key).is_some_and(raw_json_is_meaningful) {
            return Err(format!(
                "{key} is not production-admitted in the current /v1/generate route"
            ));
        }
    }
    if raw_bool_is_true(map, "outpaint_enabled") {
        return Err(
            "outpaint is not production-admitted in the current /v1/generate route".to_string(),
        );
    }
    if raw_any_number_gte(
        map,
        &[
            "outpaint_left",
            "outpaint_top",
            "outpaint_right",
            "outpaint_bottom",
            "outpaint_feathering",
            "threshold_mask_value",
            "lanpaint_mask_blend_overlap",
            "lanpaint_num_steps",
            "lanpaint_lambda",
            "lanpaint_step_size",
            "lanpaint_beta",
            "lanpaint_friction",
            "lanpaint_noise_seed",
            "lanpaint_start_at_step",
            "lanpaint_end_at_step",
            "lanpaint_early_stop",
            "lanpaint_inner_threshold",
            "lanpaint_inner_patience",
        ],
        0.0,
    ) || raw_any_string_is_set(
        map,
        &[
            "threshold_mask_operator",
            "lanpaint_prompt_mode",
            "lanpaint_inpainting_mode",
            "lanpaint_add_noise",
            "lanpaint_return_with_leftover_noise",
        ],
    ) {
        return Err(
            "outpaint/LanPaint is not production-admitted in the current /v1/generate route"
                .to_string(),
        );
    }
    if raw_string_is_set(map, "refiner_model")
        || raw_number_gt(map, "refiner_steps", 0.0)
        || raw_number_gte(map, "refiner_cfg", 0.0)
        || raw_string_is_set(map, "refiner_method")
        || raw_number_gte(map, "refiner_control", 0.0)
        || raw_bool_is_true(map, "refiner_tiling")
    {
        return Err(
            "refiner is not production-admitted in the current /v1/generate route".to_string(),
        );
    }
    if raw_string_is_set(map, "upscaler_model") || raw_number_gt(map, "upscale_by", 1.0) {
        return Err(
            "upscale is not production-admitted in the current /v1/generate route".to_string(),
        );
    }
    if let Some(denoise) = map.get("denoise").and_then(JsonValue::as_f64) {
        if denoise.is_finite() && (denoise - 1.0).abs() > f64::EPSILON {
            return Err(
                "denoise/img2img creativity is not admitted in the current /v1/generate route"
                    .to_string(),
            );
        }
    }

    Ok(())
}

pub(crate) fn raw_surface_preflight_report(error: String, obj: &JsonValue) -> serde_json::Value {
    let model = request_model_name(obj);
    json!({
        "schema": "serenity.generate.preflight.v1",
        "admitted": false,
        "error": error,
        "model": model,
        "same_gate_as_generate": true,
        "production_gate": "validate_generate_prequeue",
        "capability_profile": capability_profile_for_model(model),
        "limits": {
            "capabilities_route": "/v1/capabilities",
            "unsupported_policy": "fail_loud",
        },
    })
}

pub(crate) fn raw_surface_generate_error_report(
    error: String,
    obj: &JsonValue,
) -> serde_json::Value {
    let mut report = raw_surface_preflight_report(error, obj);
    if let Some(map) = report.as_object_mut() {
        map.insert(
            "schema".to_string(),
            JsonValue::String("serenity.generate.error.v1".to_string()),
        );
        map.insert("same_gate_as_preflight".to_string(), JsonValue::Bool(true));
        map.insert("enqueue_blocked".to_string(), JsonValue::Bool(true));
    }
    report
}

fn attach_workflow_rejection_context(report: &mut JsonValue, obj: &JsonValue, stage: &str) {
    if let Some(map) = report.as_object_mut() {
        map.insert(
            "production_gate".to_string(),
            JsonValue::String("workflow_lowering_then_validate_generate_prequeue".to_string()),
        );
        map.insert(
            "rejection_stage".to_string(),
            JsonValue::String(stage.to_string()),
        );
        if let Some(route) = workflow_route_kind(obj) {
            map.insert("workflow_route_kind".to_string(), JsonValue::String(route));
        }
        if let Some(plan) = obj.get("workflow_plan") {
            map.insert("workflow_plan".to_string(), plan.clone());
        }
    }
}

pub(crate) fn workflow_preflight_report(error: String, obj: &JsonValue) -> serde_json::Value {
    let mut report = raw_surface_preflight_report(error, obj);
    attach_workflow_rejection_context(&mut report, obj, "workflow_lowering");
    report
}

pub(crate) fn workflow_generate_error_report(error: String, obj: &JsonValue) -> serde_json::Value {
    let mut report = workflow_preflight_report(error, obj);
    if let Some(map) = report.as_object_mut() {
        map.insert(
            "schema".to_string(),
            JsonValue::String("serenity.generate.error.v1".to_string()),
        );
        map.insert("same_gate_as_preflight".to_string(), JsonValue::Bool(true));
        map.insert("enqueue_blocked".to_string(), JsonValue::Bool(true));
    }
    report
}

pub(crate) fn workflow_feature_preflight_report(
    error: String,
    obj: &JsonValue,
) -> serde_json::Value {
    let mut report = raw_surface_preflight_report(error, obj);
    attach_workflow_rejection_context(&mut report, obj, "workflow_capability");
    report
}

pub(crate) fn workflow_feature_generate_error_report(
    error: String,
    obj: &JsonValue,
) -> serde_json::Value {
    let mut report = workflow_feature_preflight_report(error, obj);
    if let Some(map) = report.as_object_mut() {
        map.insert(
            "schema".to_string(),
            JsonValue::String("serenity.generate.error.v1".to_string()),
        );
        map.insert("same_gate_as_preflight".to_string(), JsonValue::Bool(true));
        map.insert("enqueue_blocked".to_string(), JsonValue::Bool(true));
    }
    report
}

pub(crate) fn workflow_route_kind(obj: &JsonValue) -> Option<String> {
    obj.get("workflow_route_kind")
        .and_then(JsonValue::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            obj.get("workflow_plan")
                .and_then(|p| p.get("route_kind"))
                .and_then(JsonValue::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.to_string())
        })
}

pub(crate) fn reject_unsupported_workflow_route(obj: &JsonValue) -> Result<(), String> {
    let Some(route) = workflow_route_kind(obj) else {
        return Ok(());
    };
    if route == "image" {
        return Ok(());
    }
    if route == "unknown" {
        return Err(
            "workflow IR did not resolve to an executable terminal route; add a supported terminal node or route executor".to_string(),
        );
    }
    Err(format!(
        "workflow route '{route}' is not executable through the image /v1/generate job queue; dispatch must use a workflow route executor for this terminal kind"
    ))
}

pub(crate) fn workflow_route_preflight_report(error: String, obj: &JsonValue) -> serde_json::Value {
    let mut report = raw_surface_preflight_report(error, obj);
    if let Some(map) = report.as_object_mut() {
        map.insert(
            "production_gate".to_string(),
            JsonValue::String("workflow_route_dispatch".to_string()),
        );
        map.insert(
            "rejection_stage".to_string(),
            JsonValue::String("workflow_route".to_string()),
        );
        if let Some(route) = workflow_route_kind(obj) {
            map.insert("workflow_route_kind".to_string(), JsonValue::String(route));
        }
        if let Some(plan) = obj.get("workflow_plan") {
            map.insert("workflow_plan".to_string(), plan.clone());
        }
    }
    report
}

pub(crate) fn workflow_route_generate_error_report(
    error: String,
    obj: &JsonValue,
) -> serde_json::Value {
    let mut report = workflow_route_preflight_report(error, obj);
    if let Some(map) = report.as_object_mut() {
        map.insert(
            "schema".to_string(),
            JsonValue::String("serenity.generate.error.v1".to_string()),
        );
        map.insert("same_gate_as_preflight".to_string(), JsonValue::Bool(true));
        map.insert("enqueue_blocked".to_string(), JsonValue::Bool(true));
    }
    report
}

pub(crate) fn validate_generate_prequeue(
    params: &JobParams,
    hires_scale: f64,
) -> Result<ModelFamily, String> {
    let family = model_family(&params.model)?;
    if !has_text(&params.prompt) {
        return Err("prompt is required".to_string());
    }
    if params.width <= 0 || params.height <= 0 {
        return Err("width and height must be positive".to_string());
    }
    if params.steps < 1 {
        return Err("steps must be >= 1".to_string());
    }
    if !params.cfg.is_finite() || params.cfg < 0.0 {
        return Err("cfg must be finite and non-negative".to_string());
    }
    if !params.creativity.is_finite() || !(0.0..=1.0).contains(&params.creativity) {
        return Err("creativity/denoise must be finite and in [0, 1]".to_string());
    }
    if !params.variation_strength.is_finite() || !(0.0..=1.0).contains(&params.variation_strength) {
        return Err("variation_strength must be finite and in [0, 1]".to_string());
    }
    if params.images != 1 || params.image_count != 1 || params.image_index != 0 {
        return Err(
            "serenity-server currently admits one image per /v1/generate job; batch fanout must be wired before exposing images>1".to_string(),
        );
    }
    if has_text(&params.init_image) || has_text(&params.mask_image) {
        return Err(
            "image-to-image, inpaint, and image conditioning are not in the current production /v1/generate scope".to_string(),
        );
    }
    if has_vae_override(&params.vae) {
        return Err(
            "VAE override is not production-wired for /v1/generate; current routes use the baked local VAE from each model manifest".to_string(),
        );
    }
    if hires_scale > 1.0 {
        return Err(
            "hires two-pass currently depends on img2img refine and is disabled in the production /v1/generate path".to_string(),
        );
    }

    require_resolution(params, family, resolution_policy_for_family(family))?;
    validate_sampler_scheduler(
        params,
        family,
        supported_samplers_for_family(family),
        supported_schedulers_for_family(family),
    )?;
    if !supports_negative_prompt(family) {
        reject_negative(params, family)?;
    }
    validate_lora_capability(params, family)?;

    match family {
        ModelFamily::Ideogram4 => {
            reject_variation(params, family)?;
            if (params.creativity - 0.5).abs() > f64::EPSILON {
                return Err("ideogram4: creativity/denoise must remain at 0.5 in the bounded production route".to_string());
            }
            if params.cfg_override >= 0.0 {
                return Err(
                    "ideogram4: cfg_override is not admitted in the bounded production route"
                        .to_string(),
                );
            }
        }
        ModelFamily::Anima | ModelFamily::Sensenova => {
            reject_variation(params, family)?;
        }
        ModelFamily::ZImage
        | ModelFamily::Sdxl
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2 => {}
    }
    Ok(family)
}

fn admitted_feature() -> JsonValue {
    json!({
        "supported": true,
        "policy": "admit",
    })
}

fn unsupported_feature(reason: &str) -> JsonValue {
    json!({
        "supported": false,
        "policy": "fail_loud",
        "reason": reason,
    })
}

fn blocked_feature_set(reason: &str) -> JsonValue {
    json!({
        "text_to_image": unsupported_feature(reason),
        "cfg": unsupported_feature(reason),
        "negative_prompt": unsupported_feature(reason),
        "bbox_prompt_json": unsupported_feature(reason),
        "prompt_weights": unsupported_feature(reason),
        "lora": unsupported_feature(reason),
        "multi_lora": unsupported_feature(reason),
        "image_to_image": unsupported_feature(reason),
        "inpaint": unsupported_feature(reason),
        "image_conditioning": unsupported_feature(reason),
        "vae_override": unsupported_feature(reason),
        "hires_two_pass": unsupported_feature(reason),
        "refiner": unsupported_feature(reason),
        "upscale": unsupported_feature(reason),
        "outpaint": unsupported_feature(reason),
        "controlnet": unsupported_feature(reason),
        "video": unsupported_feature(reason),
    })
}

fn size_limits_for_family(family: ModelFamily) -> Vec<JsonValue> {
    production_sizes_for_family(family)
        .iter()
        .map(|(width, height)| json!({"width": width, "height": height}))
        .collect()
}

fn resolution_policy_json(family: ModelFamily) -> JsonValue {
    let policy = resolution_policy_for_family(family);
    json!({
        "mode": policy.mode,
        "min_width": policy.min_width,
        "max_width": policy.max_width,
        "min_height": policy.min_height,
        "max_height": policy.max_height,
        "multiple": policy.multiple,
        "square_only": policy.square_only,
        "admitted_product_shapes": size_limits_for_family(family),
        "unsupported_policy": "fail_loud",
        "note": policy.note,
    })
}

fn lora_feature_for_family(family: ModelFamily) -> JsonValue {
    match lora_limit_for_family(family) {
        Some(0) => unsupported_feature("LoRA overlays are not production-wired for this backend"),
        Some(limit) => json!({
            "supported": true,
            "policy": "admit",
            "max_count": limit,
        }),
        None => json!({
            "supported": true,
            "policy": "admit",
            "max_count": JsonValue::Null,
        }),
    }
}

fn multi_lora_feature_for_family(family: ModelFamily) -> JsonValue {
    match lora_limit_for_family(family) {
        None => admitted_feature(),
        Some(limit) if limit > 1 => admitted_feature(),
        Some(1) => unsupported_feature("this backend admits at most one LoRA overlay per job"),
        _ => unsupported_feature("LoRA overlays are not production-wired for this backend"),
    }
}

fn capability_for_family(family: ModelFamily) -> JsonValue {
    let (default_width, default_height) = default_size_for_family(family);
    let negative_prompt = if supports_negative_prompt(family) {
        admitted_feature()
    } else {
        unsupported_feature("negative prompt is not supported by this production route")
    };
    let bbox_prompt_json = if family == ModelFamily::Ideogram4 {
        json!({
            "supported": true,
            "policy": "admit",
            "schema": "ideogram4 prompt_json with bbox arrays",
        })
    } else {
        unsupported_feature("bbox prompt JSON is currently admitted only by the Ideogram4 route")
    };

    json!({
        "backend": family.backend_key(),
        "model_family": family.backend_key(),
        "production_status": "admitted",
        "worker_binary": family.worker_binary_name(),
        "defaults": {
            "width": default_width,
            "height": default_height,
            "steps": default_steps_for_family(family),
            "cfg": default_cfg_for_family(family),
            "sampler": default_sampler_for_family(family),
            "scheduler": default_scheduler_for_family(family),
        },
        "limits": {
            "sizes": size_limits_for_family(family),
            "resolution": resolution_policy_json(family),
            "one_image_per_job": true,
            "txt2img_only": true,
            "runtime_dependency_on_external_repos": false,
        },
        "samplers": {
            "supported_samplers": supported_samplers_for_family(family),
            "supported_schedulers": supported_schedulers_for_family(family),
            "unsupported_policy": "fail_loud",
            "accepted_sampler_parity": false,
        },
        "features": {
            "text_to_image": admitted_feature(),
            "cfg": admitted_feature(),
            "negative_prompt": negative_prompt,
            "bbox_prompt_json": bbox_prompt_json,
            "prompt_weights": unsupported_feature("weighted prompt conditioning math is not product-admitted yet"),
            "lora": lora_feature_for_family(family),
            "multi_lora": multi_lora_feature_for_family(family),
            "image_to_image": unsupported_feature("image-to-image is not admitted in the current production /v1/generate route"),
            "inpaint": unsupported_feature("inpaint depends on image conditioning and is not admitted in the current production /v1/generate route"),
            "image_conditioning": unsupported_feature("image conditioning is not admitted in the current production /v1/generate route"),
            "vae_override": unsupported_feature("VAE override is not production-wired for /v1/generate"),
            "hires_two_pass": unsupported_feature("hires two-pass depends on img2img refine and is disabled"),
            "refiner": unsupported_feature("refiner is not production-admitted in this route"),
            "upscale": unsupported_feature("upscale is not production-admitted in this route"),
            "outpaint": unsupported_feature("outpaint is not production-admitted in this route"),
            "controlnet": unsupported_feature("ControlNet is not production-admitted in this route"),
            "video": unsupported_feature("video models use separate bounded video endpoints/gates, not /v1/generate"),
        },
    })
}

pub(crate) fn capability_profile_for_model(model: &str) -> JsonValue {
    let normalized = model.trim().to_ascii_lowercase();
    if let Ok(family) = model_family(model) {
        let mut profile = capability_for_family(family);
        if let Some(obj) = profile.as_object_mut() {
            obj.insert(
                "schema".to_string(),
                JsonValue::String("serenity.capability_profile.v1".to_string()),
            );
            obj.insert(
                "selected_model".to_string(),
                JsonValue::String(model.to_string()),
            );
            obj.insert(
                "source_route".to_string(),
                JsonValue::String("/v1/capabilities".to_string()),
            );
        }
        return profile;
    }

    let (backend, production_status, reason) = if normalized.is_empty()
        || normalized.contains("select model")
    {
        ("", "invalid_request", "model is required".to_string())
    } else if let Some(info) = blocked_model_info(&normalized) {
        (
            info.backend,
            info.production_status,
            info.reason.to_string(),
        )
    } else {
        (
            "",
            "unsupported",
            format!(
                "unsupported model family for production generation: {model}; add an explicit Rust route, Mojo backend gate, and UI capability entry before exposing it"
            ),
        )
    };

    json!({
        "schema": "serenity.capability_profile.v1",
        "selected_model": model,
        "source_route": "/v1/capabilities",
        "backend": backend,
        "model_family": backend,
        "production_status": production_status,
        "policy": "fail_loud",
        "reason": reason,
        "limits": {
            "sizes": [],
            "resolution": {
                "mode": "unsupported",
                "admitted_product_shapes": [],
                "unsupported_policy": "fail_loud",
            },
            "one_image_per_job": true,
            "txt2img_only": true,
            "runtime_dependency_on_external_repos": false,
        },
        "samplers": {
            "supported_samplers": [],
            "supported_schedulers": [],
            "unsupported_policy": "fail_loud",
            "accepted_sampler_parity": false,
        },
        "features": blocked_feature_set(&reason),
    })
}

pub(crate) fn generate_capabilities_v1() -> JsonValue {
    let families = [
        ModelFamily::ZImage,
        ModelFamily::Ideogram4,
        ModelFamily::Sdxl,
        ModelFamily::Anima,
        ModelFamily::Sd3,
        ModelFamily::Flux,
        ModelFamily::Flux2,
        ModelFamily::Sensenova,
    ];
    let backends: Vec<JsonValue> = families
        .iter()
        .map(|family| capability_for_family(*family))
        .collect();
    json!({
        "schema": "serenity.capabilities.v1",
        "product_route": "/v1/generate",
        "preflight_route": "/v1/preflight",
        "same_gate_as_generate": true,
        "production_gate": "validate_generate_prequeue",
        "unsupported_policy": "fail_loud",
        "sampler_registry_route": "/v1/samplers",
        "output_contract": {
            "root_kind": "ui_workflow_gallery",
            "default_relative_root": "output/run_serenity_ui",
            "server_override_env": "SERENITY_OUT_DIR",
            "generate_result_field": "output_path",
            "location_field": "output_location",
            "result_sidecar_suffix": ".serenity_server_result.json",
            "worker_sidecar_suffix_pattern": ".<backend>_daemon_result.json",
        },
        "global_limits": {
            "one_image_per_job": true,
            "txt2img_only": true,
            "image_to_image": false,
            "vae_override": false,
            "runtime_dependency_on_external_repos": false,
        },
        "backends": backends,
        "blocked_families": [
            {
                "backend": "qwenimage",
                "production_status": "metadata/preflight-only",
                "policy": "fail_loud",
                "reason": "Qwen-Image generation is rejected before enqueue until artifact, timing, VRAM, and sampler gates pass."
            },
            {
                "backend": "zimage_l2p",
                "production_status": "blocked",
                "policy": "fail_loud",
                "reason": "Z-Image L2P has runtime pieces but no production serenity-server worker route yet."
            },
            {
                "backend": "video",
                "production_status": "bounded_elsewhere",
                "policy": "fail_loud",
                "reason": "Video/audio model families are not admitted through /v1/generate."
            }
        ],
        "non_claims": [
            "Capabilities describe current product admission, not full SwarmUI/Comfy parity.",
            "accepted_sampler_parity remains false until each exposed sampler/scheduler pair has artifact, timing, and VRAM evidence.",
            "Unsupported features must remain hidden or disabled in the UI and fail before enqueue if posted directly."
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_surface_guard_rejects_disabled_feature_fields() {
        let base = json!({
            "model": "zimage",
            "prompt": "raw surface guard",
            "width": 1024,
            "height": 1024,
            "steps": 1,
            "cfg": 4.0,
            "sampler": "euler",
            "scheduler": "simple"
        });
        let cases = [
            (
                json!({"prompt_json": {"caption": "bbox prompt", "objects": [{"bbox": [0, 0, 1000, 1000]}]}}),
                "prompt_json/bbox",
            ),
            (json!({"mask_image": "/tmp/mask.png"}), "inpaint"),
            (json!({"init_image": "/tmp/init.png"}), "image-to-image"),
            (
                json!({"inpaint_conditioning_image": "/tmp/init.png"}),
                "inpaint",
            ),
            (
                json!({"conditioning_mask_image": "/tmp/mask.png"}),
                "image conditioning",
            ),
            (
                json!({"reference_image": "/tmp/ref.png"}),
                "image conditioning",
            ),
            (
                json!({"sample_caps_pos": "/tmp/caps.json"}),
                "conditioning caps",
            ),
            (json!({"vae": "sdxl_vae.safetensors"}), "VAE override"),
            (json!({"hires_scale": 2.0}), "hires two-pass"),
            (json!({"images": 2}), "one image per"),
            (json!({"controlnet": {"enabled": true}}), "controlnet"),
            (json!({"refiner": {"enabled": true}}), "refiner"),
            (json!({"refiner_model": "sdxl-refiner"}), "refiner"),
            (json!({"upscaler": {"model": "4x"}}), "upscaler"),
            (json!({"upscale_by": 2.0}), "upscale"),
            (json!({"outpaint_enabled": true}), "outpaint"),
            (json!({"outpaint_left": 64}), "outpaint"),
            (json!({"threshold_mask_value": 0.5}), "outpaint"),
            (json!({"lanpaint_num_steps": 16}), "outpaint"),
            (json!({"denoise": 0.5}), "denoise/img2img"),
        ];

        for (extra, expected) in cases {
            let mut req = base.clone();
            req.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let error = reject_disabled_raw_surfaces(&req).unwrap_err();
            assert!(
                error.contains(expected),
                "expected {expected:?} in error {error:?}"
            );
        }
    }

    #[test]
    fn raw_surface_guard_allows_txt2img_sentinels_and_model_scoped_prompt_json() {
        let mut txt2img = json!({
            "model": "zimage",
            "prompt": "raw surface guard",
            "controlnet": null,
            "refiner": null,
            "upscaler": null,
            "outpaint": null,
            "outpaint_enabled": false,
            "refiner_model": "",
            "refiner_steps": 0,
            "refiner_cfg": -1,
            "refiner_method": "",
            "refiner_control": -1,
            "refiner_tiling": false,
            "upscaler_model": "",
            "upscale_by": 1.0,
            "denoise": 1.0
        });
        let sentinels = [
            ("init_image", json!("")),
            ("mask_image", json!("")),
            ("inpaint_conditioning_image", json!("")),
            ("inpaint_conditioning_mask", json!("")),
            ("inpaint_conditioning_noise_mask", json!(false)),
            ("conditioning_mask_image", json!("")),
            ("conditioning_mask_strength", json!(-1)),
            ("conditioning_mask_set_area_to_bounds", json!(false)),
            ("qwen_edit_conditioning_image", json!("")),
            ("reference_image", json!("")),
            ("sample_caps_pos", json!("")),
            ("sample_caps_neg", json!("")),
            ("caps_pos", json!("")),
            ("caps_neg", json!("")),
            ("caps_positive", json!("")),
            ("caps_negative", json!("")),
            ("vae", json!("Automatic")),
            ("hires_scale", json!(1.0)),
            ("images", json!(1)),
            ("outpaint_left", json!(-1)),
            ("outpaint_top", json!(-1)),
            ("outpaint_right", json!(-1)),
            ("outpaint_bottom", json!(-1)),
            ("outpaint_feathering", json!(-1)),
            ("threshold_mask_value", json!(-1)),
            ("threshold_mask_operator", json!("")),
            ("lanpaint_mask_blend_overlap", json!(-1)),
            ("lanpaint_num_steps", json!(-1)),
            ("lanpaint_lambda", json!(-1)),
            ("lanpaint_step_size", json!(-1)),
            ("lanpaint_beta", json!(-1)),
            ("lanpaint_friction", json!(-1)),
            ("lanpaint_noise_seed", json!(-1)),
            ("lanpaint_start_at_step", json!(-1)),
            ("lanpaint_end_at_step", json!(-1)),
            ("lanpaint_early_stop", json!(-1)),
            ("lanpaint_inner_threshold", json!(-1)),
            ("lanpaint_inner_patience", json!(-1)),
            ("lanpaint_prompt_mode", json!("")),
            ("lanpaint_inpainting_mode", json!("")),
            ("lanpaint_add_noise", json!("")),
            ("lanpaint_return_with_leftover_noise", json!("")),
        ];
        txt2img.as_object_mut().unwrap().extend(
            sentinels
                .into_iter()
                .map(|(key, value)| (key.to_string(), value)),
        );
        reject_disabled_raw_surfaces(&txt2img).unwrap();

        let ideogram = json!({
            "model": "ideogram4",
            "prompt_json": {
                "caption": "a product label",
                "objects": [{"label": "package", "bbox": [128, 192, 768, 832]}]
            }
        });
        reject_disabled_raw_surfaces(&ideogram).unwrap();
    }

    #[test]
    fn raw_surface_preflight_report_embeds_selected_capability_profile() {
        let req = json!({
            "model": "zimage",
            "prompt": "raw surface guard",
            "controlnet": {"enabled": true}
        });
        let report =
            raw_surface_preflight_report("controlnet is not production-admitted".to_string(), &req);
        assert_eq!(report["schema"], "serenity.generate.preflight.v1");
        assert_eq!(report["admitted"], false);
        assert_eq!(report["model"], "zimage");
        assert_eq!(
            report["capability_profile"]["schema"],
            "serenity.capability_profile.v1"
        );
        assert_eq!(report["capability_profile"]["backend"], "zimage");
        assert_eq!(
            report["capability_profile"]["features"]["controlnet"]["supported"],
            false
        );
        assert_eq!(
            report["capability_profile"]["features"]["text_to_image"]["supported"],
            true
        );
    }

    #[test]
    fn raw_surface_generate_error_report_is_capability_aware() {
        let req = json!({
            "model": "zimage",
            "prompt": "raw surface guard",
            "controlnet": {"enabled": true}
        });
        let report = raw_surface_generate_error_report(
            "controlnet is not production-admitted".to_string(),
            &req,
        );
        assert_eq!(report["schema"], "serenity.generate.error.v1");
        assert_eq!(report["admitted"], false);
        assert_eq!(report["same_gate_as_preflight"], true);
        assert_eq!(report["enqueue_blocked"], true);
        assert_eq!(
            report["capability_profile"]["schema"],
            "serenity.capability_profile.v1"
        );
        assert_eq!(report["capability_profile"]["backend"], "zimage");
    }

    #[test]
    fn workflow_preflight_report_embeds_selected_capability_profile() {
        let req = json!({
            "model": "zimage",
            "workflow": {
                "nodes": [{"id": 1, "type_id": "comfy/ControlNetApply", "fields": {}}],
                "edges": []
            }
        });
        let report = workflow_preflight_report(
            "[501] unsupported workflow graph node type 'comfy/ControlNetApply'".to_string(),
            &req,
        );
        assert_eq!(report["schema"], "serenity.generate.preflight.v1");
        assert_eq!(report["admitted"], false);
        assert_eq!(report["rejection_stage"], "workflow_lowering");
        assert_eq!(
            report["production_gate"],
            "workflow_lowering_then_validate_generate_prequeue"
        );
        assert_eq!(report["capability_profile"]["backend"], "zimage");
        assert_eq!(
            report["capability_profile"]["features"]["controlnet"]["supported"],
            false
        );
    }

    #[test]
    fn workflow_generate_error_report_marks_lowering_stage() {
        let req = json!({
            "model": "zimage",
            "workflow": {
                "nodes": [{"id": 1, "type_id": "comfy/ControlNetApply", "fields": {}}],
                "edges": []
            }
        });
        let report = workflow_generate_error_report(
            "[501] unsupported workflow graph node type 'comfy/ControlNetApply'".to_string(),
            &req,
        );
        assert_eq!(report["schema"], "serenity.generate.error.v1");
        assert_eq!(report["admitted"], false);
        assert_eq!(report["rejection_stage"], "workflow_lowering");
        assert_eq!(report["same_gate_as_preflight"], true);
        assert_eq!(report["enqueue_blocked"], true);
        assert_eq!(report["capability_profile"]["backend"], "zimage");
    }

    #[test]
    fn workflow_feature_report_preserves_route_plan_context() {
        let req = json!({
            "model": "zimage",
            "prompt": "workflow-derived img2img",
            "init_image": "/tmp/init.png",
            "workflow_route_kind": "image",
            "workflow_plan": {
                "schema": "serenity.workflow_plan.v1",
                "route_kind": "image",
                "terminal_nodes": [{"id": 8, "type": "SaveImage"}]
            }
        });
        let report = workflow_feature_generate_error_report(
            "image-to-image is not production-admitted".to_string(),
            &req,
        );
        assert_eq!(report["schema"], "serenity.generate.error.v1");
        assert_eq!(report["admitted"], false);
        assert_eq!(report["rejection_stage"], "workflow_capability");
        assert_eq!(report["workflow_route_kind"], "image");
        assert_eq!(report["workflow_plan"]["route_kind"], "image");
        assert_eq!(report["same_gate_as_preflight"], true);
        assert_eq!(report["enqueue_blocked"], true);
        assert_eq!(report["capability_profile"]["backend"], "zimage");
    }

    #[test]
    fn workflow_route_gate_allows_image_and_blocks_video_before_image_jobparams() {
        let image = json!({
            "workflow_route_kind": "image",
            "workflow_plan": {"schema": "serenity.workflow_plan.v1", "route_kind": "image"}
        });
        reject_unsupported_workflow_route(&image).unwrap();

        let video = json!({
            "model": "ltx-2.3-22b-dev.safetensors",
            "workflow_route_kind": "video",
            "workflow_plan": {
                "schema": "serenity.workflow_plan.v1",
                "route_kind": "video",
                "terminal_nodes": [{"node_id": 3, "type": "SaveVideo", "kind": "video"}]
            }
        });
        let err = reject_unsupported_workflow_route(&video).unwrap_err();
        assert!(err.contains("workflow route 'video'"), "got: {err}");
        let report = workflow_route_generate_error_report(err, &video);
        assert_eq!(report["schema"], "serenity.generate.error.v1");
        assert_eq!(report["rejection_stage"], "workflow_route");
        assert_eq!(report["workflow_route_kind"], "video");
        assert_eq!(report["workflow_plan"]["route_kind"], "video");
        assert_eq!(report["enqueue_blocked"], true);
    }
}
