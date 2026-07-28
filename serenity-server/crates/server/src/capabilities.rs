use serde_json::{Value as JsonValue, json};
use serenity_wire::JobParams;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    ZImage,
    QwenImage,
    Ideogram4,
    Sdxl,
    Anima,
    Sd3,
    Flux,
    Flux2,
    Sensenova,
    Krea2,
    Chroma,
    Lens,
}

impl ModelFamily {
    pub(crate) fn backend_key(self) -> &'static str {
        match self {
            ModelFamily::ZImage => "zimage",
            ModelFamily::QwenImage => "qwenimage",
            ModelFamily::Ideogram4 => "ideogram4",
            ModelFamily::Sdxl => "sdxl",
            ModelFamily::Anima => "anima",
            ModelFamily::Sd3 => "sd3",
            ModelFamily::Flux => "flux",
            ModelFamily::Flux2 => "flux2",
            ModelFamily::Sensenova => "sensenova",
            ModelFamily::Krea2 => "krea2",
            ModelFamily::Chroma => "chroma",
            ModelFamily::Lens => "lens",
        }
    }

    pub(crate) fn worker_binary_name(self) -> &'static str {
        match self {
            ModelFamily::ZImage => "serenity_worker_zimage",
            ModelFamily::QwenImage => "serenity_worker_qwenimage",
            ModelFamily::Ideogram4 => "serenity_worker_ideogram4",
            ModelFamily::Sdxl => "serenity_worker_sdxl",
            ModelFamily::Anima => "serenity_worker_anima",
            ModelFamily::Sd3 => "serenity_worker_sd3",
            ModelFamily::Flux => "serenity_worker_flux",
            ModelFamily::Flux2 => "serenity_worker_klein",
            ModelFamily::Sensenova => "serenity_worker_sensenova",
            ModelFamily::Krea2 => "serenity_worker_krea2",
            ModelFamily::Chroma => "serenity_worker_chroma",
            ModelFamily::Lens => "serenity_worker_lens",
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
    if (m.contains("flux2") || m.contains("flux-2") || m.contains("flux_2")) && !m.contains("klein")
    {
        return Some(BlockedModelInfo {
            backend: "flux2",
            production_status: "blocked",
            reason: "Flux2 generation is admitted only for the bounded Klein 9B and 4B routes; generic Flux2 model names remain blocked",
        });
    }
    // Flux.1-dev: the 2026-06-17 22.3 GiB OOM block was against the old
    // whole-resident worker; flux_backend.mojo now streams the DiT (BlockLoader)
    // with per-job encoder load->use->free, so admission defers to the worker's
    // own fail-loud preflights.
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

pub(crate) fn model_family_for_arch(arch: &str) -> Option<ModelFamily> {
    match arch.trim().to_ascii_lowercase().as_str() {
        "ideogram4" => Some(ModelFamily::Ideogram4),
        "qwen-image" | "qwenimage" => Some(ModelFamily::QwenImage),
        "sdxl" => Some(ModelFamily::Sdxl),
        "anima" => Some(ModelFamily::Anima),
        "sd3" => Some(ModelFamily::Sd3),
        "flux-2" | "flux-2/klein" | "klein" => Some(ModelFamily::Flux2),
        "flux" => Some(ModelFamily::Flux),
        "sensenova" => Some(ModelFamily::Sensenova),
        "krea2" => Some(ModelFamily::Krea2),
        "chroma" => Some(ModelFamily::Chroma),
        "lens" | "microsoft_lens" | "microsoft-lens" => Some(ModelFamily::Lens),
        "zimage" => Some(ModelFamily::ZImage),
        _ => None,
    }
}

pub(crate) fn model_family(model: &str) -> Result<ModelFamily, String> {
    let m = model.trim().to_ascii_lowercase();
    if m.is_empty() || m.contains("select model") {
        return Err("model is required".to_string());
    }
    if let Some(info) = blocked_model_info(&m) {
        return Err(info.reason.to_string());
    }
    if let Some(arch) = crate::models::architecture_for_model(model) {
        return model_family_for_arch(&arch).ok_or_else(|| {
            format!("unsupported model architecture for production generation: {arch} ({model})")
        });
    }
    if m.contains("ideogram") {
        return Ok(ModelFamily::Ideogram4);
    }
    if m.contains("qwen") {
        return Ok(ModelFamily::QwenImage);
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
    if m.contains("krea") {
        return Ok(ModelFamily::Krea2);
    }
    if m.contains("chroma") {
        return Ok(ModelFamily::Chroma);
    }
    if m.contains("lens") {
        return Ok(ModelFamily::Lens);
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
        | ModelFamily::QwenImage
        | ModelFamily::Ideogram4
        | ModelFamily::Sdxl
        | ModelFamily::Anima
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Sensenova
        | ModelFamily::Krea2
        | ModelFamily::Chroma
        | ModelFamily::Lens => "euler",
    }
}

fn default_scheduler_for_family(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::ZImage
        | ModelFamily::QwenImage
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Sensenova
        | ModelFamily::Krea2
        | ModelFamily::Lens => "simple",
        ModelFamily::Chroma => "beta",
        ModelFamily::Ideogram4 => "ideogram_logitnormal",
        ModelFamily::Sdxl | ModelFamily::Anima => "normal",
    }
}

fn default_size_for_family(_family: ModelFamily) -> (i64, i64) {
    (1024, 1024)
}

pub(crate) fn default_steps_for_family(family: ModelFamily) -> i64 {
    match family {
        // Z-Image Base is undistilled. The publisher recommends 28-50 steps
        // and the local sampler contract pins the product default at 28.
        ModelFamily::ZImage => 28,
        // Qwen-Image's published pipeline default and the Mojo scheduler
        // contract are both 50 inference steps.
        ModelFamily::QwenImage => 50,
        // Ideogram's quality preset is the published default. The 20-step
        // profile is explicitly the speed-oriented V4_DEFAULT_20 preset.
        ModelFamily::Ideogram4 => 48,
        ModelFamily::Sdxl => 50,
        ModelFamily::Anima => 30,
        ModelFamily::Sd3 => 28,
        ModelFamily::Flux => 50,
        ModelFamily::Chroma => 40,
        ModelFamily::Flux2 => 4,
        ModelFamily::Sensenova => 50,
        ModelFamily::Krea2 => 52,
        ModelFamily::Lens => 20,
    }
}

pub(crate) fn default_cfg_for_family(family: ModelFamily) -> f64 {
    match family {
        ModelFamily::ZImage => 4.0,
        ModelFamily::QwenImage => 4.0,
        ModelFamily::Ideogram4 | ModelFamily::Sdxl => 7.0,
        ModelFamily::Anima | ModelFamily::Sd3 => 4.5,
        ModelFamily::Flux => 3.5,
        ModelFamily::Sensenova => 4.0,
        // BFL's production Klein checkpoints are guidance-distilled. The
        // undistilled Base checkpoints override this to 4.0 below.
        ModelFamily::Flux2 => 1.0,
        ModelFamily::Krea2 => 3.5,
        ModelFamily::Chroma => 3.0,
        ModelFamily::Lens => 5.0,
    }
}

pub(crate) fn default_steps_for_model(model: &str, family: ModelFamily) -> i64 {
    let normalized = model.to_ascii_lowercase();
    if family == ModelFamily::Flux2 && normalized.contains("base") {
        // Official FLUX.2 Klein Base inference profile: undistilled, 50 steps.
        50
    } else if family == ModelFamily::ZImage && normalized.contains("turbo") {
        8
    } else if family == ModelFamily::Anima && normalized.contains("turbo") {
        10
    } else if family == ModelFamily::Ideogram4 && normalized.contains("turbo") {
        12
    } else if family == ModelFamily::Sensenova
        && (normalized.contains("8step") || normalized.contains("8-step"))
    {
        8
    } else if family == ModelFamily::Krea2 && normalized.contains("turbo") {
        8
    } else if family == ModelFamily::Lens && normalized.contains("turbo") {
        4
    } else if family == ModelFamily::Lens && normalized.contains("base") {
        50
    } else {
        default_steps_for_family(family)
    }
}

pub(crate) fn default_cfg_for_model(model: &str, family: ModelFamily) -> f64 {
    let normalized = model.to_ascii_lowercase();
    if family == ModelFamily::Flux2 && normalized.contains("base") {
        // Base is not guidance-distilled; BFL's model card specifies CFG 4.0.
        4.0
    } else if family == ModelFamily::ZImage && normalized.contains("turbo") {
        0.0
    } else if family == ModelFamily::Anima && normalized.contains("turbo") {
        1.0
    } else if family == ModelFamily::Sensenova
        && (normalized.contains("8step") || normalized.contains("8-step"))
    {
        1.0
    } else if family == ModelFamily::Krea2 && normalized.contains("turbo") {
        0.0
    } else if family == ModelFamily::Lens && normalized.contains("turbo") {
        1.0
    } else {
        default_cfg_for_family(family)
    }
}

/// Exact defaults attached to one registry model card. Family capabilities
/// describe the backend; this record preserves checkpoint variants such as
/// Base, Turbo, and distilled without frontend filename guesses.
pub(crate) fn generation_defaults_for_model_arch(model: &str, arch: &str) -> Option<JsonValue> {
    let family = model_family_for_arch(arch)?;
    Some(json!({
        "source": "server_model_profile",
        "steps": default_steps_for_model(model, family),
        "cfg": default_cfg_for_model(model, family),
        "sampler": default_sampler_for_family(family),
        "scheduler": default_scheduler_for_family(family),
    }))
}

// Z-Image supports the proven 512 square path plus the shared seven-shape 1MP
// ladder. The backend evicts resident DiT blocks when decode headroom is low,
// then dispatches the exact finite specializations below. Keep this synchronized
// with zimage_backend.mojo and DEFAULT_ASPECT_LADDER_X100.
const ZIMAGE_SIZES: &[(i64, i64)] = &[
    (512, 512),
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
];
// Klein's Mojo worker has finite comptime specializations for 512 square plus
// the same parity-gated 1024px-area ladder used by the image backends. Keep
// this explicit list synchronized with klein_runtime_backend.mojo; product
// admission still requires real artifact/VRAM evidence for every exposed arm.
const KLEIN_SIZES: &[(i64, i64)] = &[
    (512, 512),
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
];
const SENSENOVA_SIZES: &[(i64, i64)] = &[(512, 512), (1024, 1024)];
const LENS_SIZES: &[(i64, i64)] = &[(1024, 1024)];
// Exact 1024px-area ladder compiled by the Krea and Qwen image workers. Krea
// also shares it with training/cache. Keep this wire contract synchronized with
// serenitymojo.training.aspect_buckets.DEFAULT_ASPECT_LADDER_X100.
// 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3.
const IMAGE_1024_ASPECT_SIZES: &[(i64, i64)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1344, 768),
    (768, 1344),
    (1280, 832),
    (832, 1280),
];
const SAMPLERS_EULER: &[&str] = &["euler"];
const SAMPLERS_EULER_FLOWMATCH: &[&str] = &["euler", "flowmatch_euler"];
const SAMPLERS_FLOWMATCH_DPM2M: &[&str] = &["euler", "flowmatch_euler", "dpmpp_2m"];
const SAMPLERS_SDXL: &[&str] = &["euler", "ddim", "dpmpp_2m"];
const SAMPLERS_ZIMAGE: &[&str] = &[
    "euler",
    "flowmatch_euler",
    "dpmpp_2m",
    "uni_pc",
    "uni_pc_bh2",
];
const SCHEDULERS_SIMPLE: &[&str] = &["simple"];
// Flux2/klein: the worker accepts "flux2" (Flux2Scheduler graph node lowers to
// it; klein_runtime maps it onto its flow-match schedule) alongside "simple".
const SCHEDULERS_FLUX2: &[&str] = &["simple", "flux2"];
const SCHEDULERS_NORMAL: &[&str] = &["normal"];
const SCHEDULERS_SDXL: &[&str] = &["normal", "karras", "exponential", "simple", "ddim_uniform"];
const SCHEDULERS_ZIMAGE: &[&str] = &["simple", "sgm_uniform"];
const SCHEDULERS_IDEOGRAM4: &[&str] = &["ideogram_logitnormal", "simple"];
const SCHEDULERS_SWARM_FLUX: &[&str] = &[
    "normal",
    "karras",
    "exponential",
    "simple",
    "ddim_uniform",
    "sgm_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
];

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
            max_width: 1344,
            min_height: 512,
            max_height: 1344,
            multiple: 64,
            square_only: false,
            admitted_shapes,
            note: "Z-Image dispatches 512x512 plus the compiled seven-shape 1MP ladder; low-headroom decode evicts resident DiT blocks before VAE allocation",
        },
        ModelFamily::QwenImage => ResolutionPolicy {
            mode: "shape_dispatch",
            min_width: 768,
            max_width: 1344,
            min_height: 768,
            max_height: 1344,
            multiple: 64,
            square_only: false,
            admitted_shapes,
            note: "Qwen-Image worker dispatches the compiled 1024px-area image aspect ladder",
        },
        ModelFamily::Sdxl
        | ModelFamily::Anima
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Krea2
        | ModelFamily::Chroma => ResolutionPolicy {
            mode: "shape_dispatch",
            min_width: 768,
            max_width: 1344,
            min_height: 768,
            max_height: 1344,
            multiple: 64,
            square_only: false,
            admitted_shapes,
            note: "This worker dispatches the compiled seven-shape 1MP image aspect ladder",
        },
        ModelFamily::Ideogram4 => ResolutionPolicy {
            mode: "shape_dispatch",
            min_width: 768,
            max_width: 1344,
            min_height: 768,
            max_height: 1344,
            multiple: 64,
            square_only: false,
            admitted_shapes,
            note: "Ideogram4 dispatches the compiled seven-shape 1MP image aspect ladder",
        },
        ModelFamily::Flux2 => ResolutionPolicy {
            mode: "shape_dispatch",
            min_width: 512,
            max_width: 1344,
            min_height: 512,
            max_height: 1344,
            multiple: 64,
            square_only: false,
            admitted_shapes,
            note: "The Klein 9B and 4B worker dispatches 512x512 plus the compiled seven-shape 1MP image aspect ladder",
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
        ModelFamily::Lens => ResolutionPolicy {
            mode: "fixed_shape",
            min_width: 1024,
            max_width: 1024,
            min_height: 1024,
            max_height: 1024,
            multiple: 1024,
            square_only: true,
            admitted_shapes,
            note: "Microsoft Lens is compiled for the fixed 1024x1024 DiT, RoPE, and SDPA geometry",
        },
    }
}

fn production_sizes_for_family(family: ModelFamily) -> &'static [(i64, i64)] {
    match family {
        ModelFamily::ZImage => ZIMAGE_SIZES,
        ModelFamily::QwenImage => IMAGE_1024_ASPECT_SIZES,
        ModelFamily::Sdxl => IMAGE_1024_ASPECT_SIZES,
        ModelFamily::Anima => IMAGE_1024_ASPECT_SIZES,
        ModelFamily::Ideogram4 | ModelFamily::Sd3 | ModelFamily::Chroma => IMAGE_1024_ASPECT_SIZES,
        ModelFamily::Flux => IMAGE_1024_ASPECT_SIZES,
        ModelFamily::Flux2 => KLEIN_SIZES,
        ModelFamily::Sensenova => SENSENOVA_SIZES,
        ModelFamily::Krea2 => IMAGE_1024_ASPECT_SIZES,
        ModelFamily::Lens => LENS_SIZES,
    }
}

fn supported_samplers_for_family(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::ZImage => SAMPLERS_ZIMAGE,
        ModelFamily::QwenImage | ModelFamily::Sd3 => SAMPLERS_FLOWMATCH_DPM2M,
        ModelFamily::Flux | ModelFamily::Chroma => SAMPLERS_FLOWMATCH_DPM2M,
        ModelFamily::Ideogram4 => SAMPLERS_EULER_FLOWMATCH,
        ModelFamily::Flux2 | ModelFamily::Sensenova | ModelFamily::Lens => SAMPLERS_EULER,
        ModelFamily::Sdxl => SAMPLERS_SDXL,
        ModelFamily::Anima => SAMPLERS_EULER,
        ModelFamily::Krea2 => SAMPLERS_EULER,
    }
}

fn supported_schedulers_for_family(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::ZImage => SCHEDULERS_ZIMAGE,
        ModelFamily::Ideogram4 => SCHEDULERS_IDEOGRAM4,
        ModelFamily::Flux | ModelFamily::Chroma => SCHEDULERS_SWARM_FLUX,
        ModelFamily::Sdxl => SCHEDULERS_SDXL,
        ModelFamily::Anima => SCHEDULERS_NORMAL,
        ModelFamily::Flux2 => SCHEDULERS_FLUX2,
        ModelFamily::QwenImage
        | ModelFamily::Sd3
        | ModelFamily::Sensenova
        | ModelFamily::Krea2
        | ModelFamily::Lens => SCHEDULERS_SIMPLE,
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
        ModelFamily::ZImage | ModelFamily::Sdxl => None,
        ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Krea2
        | ModelFamily::Chroma
        | ModelFamily::Sd3
        | ModelFamily::Anima
        | ModelFamily::QwenImage
        | ModelFamily::Ideogram4 => Some(1),
        ModelFamily::Sensenova | ModelFamily::Lens => Some(0),
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

fn has_lanpaint_params(params: &JobParams) -> bool {
    params.lanpaint_mask_blend_overlap >= 0
        || params.lanpaint_context_expand >= 0
        || params.lanpaint_num_steps >= 0
        || params.lanpaint_lambda >= 0.0
        || params.lanpaint_step_size >= 0.0
        || params.lanpaint_beta >= 0.0
        || params.lanpaint_friction >= 0.0
        || has_text(&params.lanpaint_prompt_mode)
        || has_text(&params.lanpaint_inpainting_mode)
        || has_text(&params.lanpaint_add_noise)
        || params.lanpaint_noise_seed >= 0
        || params.lanpaint_start_at_step >= 0
        || params.lanpaint_end_at_step >= 0
        || has_text(&params.lanpaint_return_with_leftover_noise)
        || params.lanpaint_early_stop >= 0
        || params.lanpaint_inner_threshold >= 0.0
        || params.lanpaint_inner_patience >= 0
}

fn validate_krea_lanpaint(params: &JobParams) -> Result<(), String> {
    if !has_text(&params.init_image) || !has_text(&params.mask_image) {
        return Err("krea2 LanPaint requires both init_image and mask_image".to_string());
    }
    if params.width != 1024 || params.height != 1024 {
        return Err(format!(
            "krea2 LanPaint: unsupported size {}x{}; choose the compiled 1024x1024 profile",
            params.width, params.height
        ));
    }
    if !has_text(&params.lanpaint_mask_channel) {
        return Err("krea2 LanPaint requires an explicit mask channel".to_string());
    }
    if params.lanpaint_num_steps < 0 {
        return Err("krea2 LanPaint requires lanpaint_num_steps >= 0".to_string());
    }
    if !(0..=256).contains(&params.lanpaint_context_expand) {
        return Err(
            "krea2 LanPaint context expand must be an image-pixel value in 0..=256".to_string(),
        );
    }
    for (name, value) in [
        ("lanpaint_lambda", params.lanpaint_lambda),
        ("lanpaint_step_size", params.lanpaint_step_size),
        ("lanpaint_beta", params.lanpaint_beta),
        ("lanpaint_friction", params.lanpaint_friction),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("krea2 LanPaint requires {name} > 0"));
        }
    }
    let prompt_mode = params.lanpaint_prompt_mode.trim().to_ascii_lowercase();
    if !matches!(prompt_mode.as_str(), "image first" | "prompt first") {
        return Err("krea2 LanPaint prompt mode must be Image First or Prompt First".to_string());
    }
    if params
        .lanpaint_inpainting_mode
        .to_ascii_lowercase()
        .contains("video")
    {
        return Err("krea2 LanPaint admits image inpainting only".to_string());
    }
    if params.lanpaint_add_noise.eq_ignore_ascii_case("disable") {
        return Err("krea2 LanPaint add_noise=disable is not supported".to_string());
    }
    if params.lanpaint_start_at_step > 0 {
        return Err("krea2 LanPaint start_at_step must be 0/full schedule".to_string());
    }
    if params.lanpaint_end_at_step >= 0 && params.lanpaint_end_at_step < params.steps {
        return Err("krea2 LanPaint early end_at_step is not supported".to_string());
    }
    if params
        .lanpaint_return_with_leftover_noise
        .eq_ignore_ascii_case("enable")
    {
        return Err("krea2 LanPaint leftover-noise output is not supported".to_string());
    }
    if params.lanpaint_early_stop < 0 {
        return Err("krea2 LanPaint requires lanpaint_early_stop >= 0".to_string());
    }
    if params.lanpaint_inner_threshold > 0.0 {
        return Err(
            "krea2 LanPaint semantic inner-threshold stopping is not supported".to_string(),
        );
    }
    if params.lanpaint_mask_blend_overlap == 0
        || params.lanpaint_mask_blend_overlap > 51
        || (params.lanpaint_mask_blend_overlap > 0 && params.lanpaint_mask_blend_overlap % 2 == 0)
    {
        return Err("krea2 LanPaint mask blend overlap must be an odd value in 1..=51".to_string());
    }
    if (params.creativity - 1.0).abs() > f64::EPSILON {
        return Err("krea2 LanPaint requires denoise/creativity=1.0".to_string());
    }
    if params.outpaint_left >= 0
        || params.outpaint_top >= 0
        || params.outpaint_right >= 0
        || params.outpaint_bottom >= 0
        || params.outpaint_feathering >= 0
        || params.threshold_mask_value >= 0.0
        || has_text(&params.threshold_mask_operator)
    {
        return Err(
            "krea2 LanPaint outpaint preprocessing is not supported in this profile".to_string(),
        );
    }
    Ok(())
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

fn reject_qwen_runtime_overrides(params: &JobParams) -> Result<(), String> {
    if params.cfg_override >= 0.0 {
        return Err("qwenimage: cfg_override is not supported yet".to_string());
    }
    if (params.cfg_override_start_percent - 0.0).abs() > f64::EPSILON
        || (params.cfg_override_end_percent - 1.0).abs() > f64::EPSILON
    {
        return Err("qwenimage: cfg_override percent window is not supported yet".to_string());
    }
    if (params.sigma_shift - 3.0).abs() > f64::EPSILON {
        return Err("qwenimage: sigma_shift override is not supported yet".to_string());
    }
    if (params.creativity - 0.5).abs() > f64::EPSILON {
        return Err("qwenimage: creativity/partial denoise is not supported yet".to_string());
    }
    if !params.sample_caps_pos.trim().is_empty() || !params.sample_caps_neg.trim().is_empty() {
        return Err(
            "qwenimage: pre-encoded sample caps are not supported by the live Qwen text encoder path"
                .to_string(),
        );
    }
    if params.clip_skip > 0 {
        return Err(
            "qwenimage: advanced sampling parameter 'clip_skip' is not supported yet".to_string(),
        );
    }
    if params.eta >= 0.0 {
        return Err(
            "qwenimage: advanced sampling parameter 'eta' is not supported yet".to_string(),
        );
    }
    if params.sigma_min >= 0.0 {
        return Err(
            "qwenimage: advanced sampling parameter 'sigma_min' is not supported yet".to_string(),
        );
    }
    if params.sigma_max >= 0.0 {
        return Err(
            "qwenimage: advanced sampling parameter 'sigma_max' is not supported yet".to_string(),
        );
    }
    if params.restart_sampling {
        return Err(
            "qwenimage: advanced sampling parameter 'restart_sampling' is not supported yet"
                .to_string(),
        );
    }
    Ok(())
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

fn request_model_is_zimage(obj: &JsonValue) -> bool {
    model_family(request_model_name(obj)) == Ok(ModelFamily::ZImage)
}

fn request_model_is_krea2(obj: &JsonValue) -> bool {
    model_family(request_model_name(obj)) == Ok(ModelFamily::Krea2)
}

/// Klein ReferenceLatent edit: the product route consumes one reference_image;
/// the preserved two-reference legacy graph additionally carries ref A in
/// init_image. Only these bounded shapes are carved out of the raw
/// image-conditioning/img2img rejections below.
fn request_is_klein_reference_edit(obj: &JsonValue) -> bool {
    let model = request_model_name(obj).to_ascii_lowercase();
    let count = obj
        .as_object()
        .and_then(|m| m.get("reference_latent_count"))
        .and_then(JsonValue::as_i64)
        .unwrap_or(0);
    (model.contains("klein") || model.contains("flux2")) && (1..=2).contains(&count)
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

    let zimage_edit = request_model_is_zimage(obj);
    let has_init_image = raw_string_is_set(map, "init_image");
    let has_mask_image = raw_string_is_set(map, "mask_image");
    let krea_lanpaint = request_model_is_krea2(obj)
        && has_init_image
        && has_mask_image
        && raw_number_gte(map, "lanpaint_num_steps", 0.0);
    if (has_mask_image && !zimage_edit && !krea_lanpaint)
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
    if has_mask_image && !has_init_image {
        return Err(
            "SetLatentNoiseMask/LanPaint requires init_image/VAEEncode source pixels".to_string(),
        );
    }
    if has_init_image && !zimage_edit && !krea_lanpaint && !request_is_klein_reference_edit(obj) {
        return Err(
            "image-to-image is not production-admitted in the current /v1/generate route"
                .to_string(),
        );
    }
    let ref_gate_keys: &[&str] = if request_is_klein_reference_edit(obj) {
        &["conditioning_mask_image", "qwen_edit_conditioning_image"]
    } else {
        &[
            "conditioning_mask_image",
            "qwen_edit_conditioning_image",
            "reference_image",
        ]
    };
    if raw_any_string_is_set(map, ref_gate_keys)
        || raw_any_number_gte(map, &["conditioning_mask_strength"], 0.0)
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
        ],
        0.0,
    ) || raw_string_is_set(map, "threshold_mask_operator")
    {
        return Err(
            "outpaint preprocessing is not production-admitted in the current /v1/generate route"
                .to_string(),
        );
    }
    if (raw_any_number_gte(
        map,
        &[
            "lanpaint_mask_blend_overlap",
            "lanpaint_context_expand",
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
            "lanpaint_prompt_mode",
            "lanpaint_inpainting_mode",
            "lanpaint_add_noise",
            "lanpaint_return_with_leftover_noise",
        ],
    )) && !krea_lanpaint
    {
        return Err(
            "LanPaint is production-admitted only for the bounded Krea2 init+mask route"
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
            if !(zimage_edit && has_init_image) {
                return Err(
                    "denoise/img2img creativity is admitted only for Z-Image requests with init_image"
                        .to_string(),
                );
            }
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
    // Klein ReferenceLatent edits carry the product source in reference_image.
    // The preserved count==2 legacy contract additionally carries ref A in
    // init_image; both bounded shapes are exempt from the img2img rejection.
    let klein_ref_edit = (params.model.to_ascii_lowercase().contains("klein")
        || params.model.to_ascii_lowercase().contains("flux2"))
        && (1..=2).contains(&params.reference_latent_count);
    let zimage_edit = family == ModelFamily::ZImage && has_text(&params.init_image);
    let lanpaint_requested = has_lanpaint_params(params);
    let krea_lanpaint = family == ModelFamily::Krea2 && lanpaint_requested;
    if lanpaint_requested {
        if family != ModelFamily::Krea2 {
            return Err(
                "LanPaint is production-admitted only by the bounded Krea2 route".to_string(),
            );
        }
        validate_krea_lanpaint(params)?;
    }
    if (has_text(&params.init_image)
        && !klein_ref_edit
        && family != ModelFamily::ZImage
        && !krea_lanpaint)
        || (has_text(&params.mask_image) && !zimage_edit && !krea_lanpaint)
    {
        return Err(
            "image-to-image/inpaint is admitted only for Z-Image, except the bounded Krea2 LanPaint route; Klein ReferenceLatent remains separate".to_string(),
        );
    }
    if has_text(&params.mask_image) && !has_text(&params.init_image) {
        return Err(
            "SetLatentNoiseMask/LanPaint requires init_image/VAEEncode source pixels".to_string(),
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

    // Krea2 FlowEdit has explicit 512² and 1024² Mojo geometry arms. Keep this
    // narrower than the t2i aspect ladder because edit is square-only.
    if krea_lanpaint {
        require_resolution(
            params,
            family,
            ResolutionPolicy {
                mode: "shape_dispatch",
                min_width: 1024,
                max_width: 1024,
                min_height: 1024,
                max_height: 1024,
                multiple: 1024,
                square_only: true,
                admitted_shapes: &[(1024, 1024)],
                note: "Krea2 LanPaint worker mode is compiled at 1024x1024",
            },
        )?;
    } else if !params.edit_src_image.is_empty() && family == ModelFamily::Krea2 {
        require_resolution(
            params,
            family,
            ResolutionPolicy {
                mode: "shape_dispatch",
                min_width: 512,
                max_width: 1024,
                min_height: 512,
                max_height: 1024,
                multiple: 512,
                square_only: true,
                admitted_shapes: &[(512, 512), (1024, 1024)],
                note: "krea2 FlowEdit worker mode is compiled at 512x512 and 1024x1024",
            },
        )?;
    } else {
        require_resolution(params, family, resolution_policy_for_family(family))?;
    }
    let (sampler, _) = validate_sampler_scheduler(
        params,
        family,
        supported_samplers_for_family(family),
        supported_schedulers_for_family(family),
    )?;
    if zimage_edit && matches!(sampler.as_str(), "uni_pc" | "uni_pc_bh2") {
        return Err(
            "zimage: UniPC img2img/inpaint is not admitted; use Euler/flowmatch_euler or DPM++ 2M"
                .to_string(),
        );
    }
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
            // Ideogram-4 is trained on structured JSON captions; the ComfyUI
            // workflow builds them with a Gemma prompt-builder subgraph this
            // server does not execute. A plain-text prompt sails through the
            // model but the conditioning is off-distribution and the output is
            // incoherent (e2e-observed 2026-07-15). Mirror the trainer's
            // ideogram4_encode_sample_prompt: require a JSON object caption.
            let trimmed = params.prompt.trim();
            let is_json_caption = trimmed.starts_with('{')
                && serde_json::from_str::<serde_json::Value>(trimmed)
                    .map(|v| v.is_object())
                    .unwrap_or(false);
            if !is_json_caption {
                return Err(
                    "ideogram4: prompt must be a structured JSON caption object (plain text \
                     produces incoherent output; build one with /v1/magic_prompt or an \
                     ideogram4 caption template)"
                        .to_string(),
                );
            }
            if !params.edit_src_image.is_empty() {
                let src_trimmed = params.edit_src_prompt.trim();
                let src_is_json = src_trimmed.starts_with('{')
                    && serde_json::from_str::<serde_json::Value>(src_trimmed)
                        .map(|v| v.is_object())
                        .unwrap_or(false);
                if !src_is_json {
                    return Err(
                        "ideogram4: FlowEdit src prompt must also be a structured JSON \
                         caption object"
                            .to_string(),
                    );
                }
            }
        }
        ModelFamily::QwenImage => {
            reject_qwen_runtime_overrides(params)?;
        }
        ModelFamily::Anima | ModelFamily::Sensenova | ModelFamily::Lens => {
            reject_variation(params, family)?;
        }
        ModelFamily::ZImage
        | ModelFamily::Sdxl
        | ModelFamily::Sd3
        | ModelFamily::Flux
        | ModelFamily::Flux2
        | ModelFamily::Krea2
        | ModelFamily::Chroma => {}
    }
    // FlowEdit (edit_src_image marks an edit job) is implemented by the Krea2
    // and Ideogram4 Mojo workers. Other families would ignore these edit params,
    // so they stay fail-loud.
    if !params.edit_src_image.is_empty()
        && !matches!(family, ModelFamily::Krea2 | ModelFamily::Ideogram4)
    {
        return Err(format!(
            "{}: FlowEdit is only worker-admitted for krea2 and ideogram4; this \
             family's worker would ignore the edit params",
            params.model
        ));
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
        "instruction_edit": unsupported_feature(reason),
        "image_conditioning": unsupported_feature(reason),
        "vae_override": unsupported_feature(reason),
        "hires_two_pass": unsupported_feature(reason),
        "refiner": unsupported_feature(reason),
        "upscale": unsupported_feature(reason),
        "outpaint": unsupported_feature(reason),
        "controlnet": unsupported_feature(reason),
        "video": unsupported_feature(reason),
        "advanced_sampling": {
            "supported": false,
            "policy": "fail_loud",
            "reason": reason,
            "parameters": {},
        },
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

fn advanced_sampling_feature_for_family(family: ModelFamily) -> JsonValue {
    let sigma_shift = if matches!(family, ModelFamily::ZImage | ModelFamily::Ideogram4) {
        json!({
            "supported": true,
            "policy": "admit",
            "default": 3.0,
            "min": 0.01,
            "max": 100.0,
            "step": 0.01,
            "reason": "the selected Mojo worker applies sigma_shift when it constructs the flow schedule",
        })
    } else {
        unsupported_feature("the selected Mojo worker does not admit a user sigma-shift override")
    };
    json!({
        "supported": matches!(family, ModelFamily::ZImage | ModelFamily::Ideogram4),
        "policy": "partial",
        "parameters": {
            "sigma_shift": sigma_shift,
            "clip_skip": unsupported_feature("CLIP layer skipping is not production-admitted by this model runtime"),
            "eta": unsupported_feature("sampler eta is not production-admitted by this model runtime"),
            "sigma_min": unsupported_feature("custom sigma minimum is not production-admitted by this model runtime"),
            "sigma_max": unsupported_feature("custom sigma maximum is not production-admitted by this model runtime"),
            "restart_sampling": unsupported_feature("restart sampling is not production-admitted by this model runtime"),
            "vae": unsupported_feature("VAE override is not production-wired for /v1/generate"),
        },
    })
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
    let image_to_image = if family == ModelFamily::ZImage {
        json!({
            "supported": true,
            "policy": "admit",
            "backend": "zimage",
            "creativity_range": [0.0, 1.0],
            "note": "VAE-encoded init image with sliced FlowMatch denoise schedule",
        })
    } else {
        unsupported_feature("image-to-image is admitted only by the Z-Image production route")
    };
    let inpaint = if family == ModelFamily::ZImage {
        json!({
            "supported": true,
            "policy": "admit",
            "backend": "zimage",
            "mask_contract": "SetLatentNoiseMask preserve semantics",
            "note": "Mask-aware Z-Image denoise plus bounded final mask blend; full LanPaint inner-loop controls remain blocked",
        })
    } else if family == ModelFamily::Krea2 {
        json!({
            "supported": true,
            "policy": "admit",
            "backend": "krea2",
            "engine": "lanpaint",
            "sizes": [[1024, 1024]],
            "mask_contract": "LanPaint latent preserve mask plus final feathered MaskBlend",
            "lora": true,
            "note": "Mojo-native damped LanPaint sampler for Krea2 Raw and Turbo",
        })
    } else {
        unsupported_feature("mask-aware inpaint is admitted by Z-Image and bounded Krea2 LanPaint")
    };
    let instruction_edit = if matches!(family, ModelFamily::Krea2 | ModelFamily::Ideogram4) {
        json!({
            "supported": true,
            "policy": "admit",
            "engine": "flowedit",
            "note": if family == ModelFamily::Krea2 { "compiled 512x512 and 1024x1024 FlowEdit" } else { "compiled 1024x1024 FlowEdit with structured JSON captions" },
        })
    } else if family == ModelFamily::Flux2 {
        json!({
            "supported": true,
            "policy": "admit",
            "engine": "reference_latent",
            "sizes": [[1024, 1024]],
            "source_images": 1,
            "note": "Klein 9B and 4B native ReferenceLatent instruction edit",
        })
    } else {
        unsupported_feature(
            "instruction editing is admitted by Krea2/Ideogram4 FlowEdit and Klein ReferenceLatent",
        )
    };
    let image_conditioning = if family == ModelFamily::Flux2 {
        json!({
            "supported": true,
            "policy": "admit",
            "engine": "reference_latent",
            "max_reference_images": 1,
            "note": "one source image is VAE-encoded and attached to the positive conditioning as target-plus-reference image tokens",
        })
    } else {
        unsupported_feature(
            "image conditioning is not admitted in the current production /v1/generate route",
        )
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
            "txt2img_only": !matches!(family, ModelFamily::ZImage | ModelFamily::Krea2 | ModelFamily::Flux2),
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
            "image_to_image": image_to_image,
            "inpaint": inpaint,
            "instruction_edit": instruction_edit,
            "image_conditioning": image_conditioning,
            "vae_override": unsupported_feature("VAE override is not production-wired for /v1/generate"),
            "hires_two_pass": unsupported_feature("hires two-pass depends on img2img refine and is disabled"),
            "refiner": unsupported_feature("refiner is not production-admitted in this route"),
            "upscale": unsupported_feature("upscale is not production-admitted in this route"),
            "outpaint": unsupported_feature("outpaint is not production-admitted in this route"),
            "controlnet": unsupported_feature("ControlNet is not production-admitted in this route"),
            "video": unsupported_feature("video models use separate bounded video endpoints/gates, not /v1/generate"),
            "advanced_sampling": advanced_sampling_feature_for_family(family),
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
            if let Some(defaults) = obj.get_mut("defaults").and_then(JsonValue::as_object_mut) {
                defaults.insert(
                    "steps".to_string(),
                    JsonValue::from(default_steps_for_model(model, family)),
                );
                defaults.insert(
                    "cfg".to_string(),
                    JsonValue::from(default_cfg_for_model(model, family)),
                );
            }
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
        ModelFamily::QwenImage,
        ModelFamily::Ideogram4,
        ModelFamily::Sdxl,
        ModelFamily::Anima,
        ModelFamily::Sd3,
        ModelFamily::Flux,
        ModelFamily::Flux2,
        ModelFamily::Sensenova,
        ModelFamily::Krea2,
        ModelFamily::Chroma,
        ModelFamily::Lens,
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
            "txt2img_only": false,
            "image_to_image": true,
            "vae_override": false,
            "runtime_dependency_on_external_repos": false,
        },
        "backends": backends,
        "blocked_families": [
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
            "Capabilities describe current product admission, not full Comfy parity.",
            "accepted_sampler_parity remains false until each exposed sampler/scheduler pair has artifact, timing, and VRAM evidence.",
            "Unsupported features must remain hidden or disabled in the UI and fail before enqueue if posted directly."
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_aspect_capabilities_match_measured_worker_shapes() {
        let expected = [
            (1024, 1024),
            (1152, 896),
            (896, 1152),
            (1344, 768),
            (768, 1344),
            (1280, 832),
            (832, 1280),
        ];
        assert_eq!(production_sizes_for_family(ModelFamily::Krea2), expected);
        assert_eq!(
            production_sizes_for_family(ModelFamily::QwenImage),
            expected
        );

        let profile = capability_for_family(ModelFamily::QwenImage);
        assert_eq!(profile["defaults"]["steps"], 50);
        assert_eq!(profile["defaults"]["cfg"], 4.0);
        assert_eq!(profile["limits"]["resolution"]["mode"], "shape_dispatch");
        assert_eq!(profile["limits"]["resolution"]["square_only"], false);
        assert_eq!(profile["limits"]["sizes"].as_array().unwrap().len(), 7);

        let zimage = capability_for_family(ModelFamily::ZImage);
        assert_eq!(
            production_sizes_for_family(ModelFamily::ZImage),
            [
                (512, 512),
                (1024, 1024),
                (1152, 896),
                (896, 1152),
                (1344, 768),
                (768, 1344),
                (1280, 832),
                (832, 1280),
            ]
        );
        assert_eq!(zimage["limits"]["sizes"].as_array().unwrap().len(), 8);
        assert_eq!(zimage["limits"]["resolution"]["max_width"], 1344);
        assert_eq!(zimage["limits"]["resolution"]["square_only"], false);

        for family in [
            ModelFamily::QwenImage,
            ModelFamily::Sdxl,
            ModelFamily::Anima,
            ModelFamily::Ideogram4,
            ModelFamily::Sd3,
            ModelFamily::Flux,
            ModelFamily::Krea2,
            ModelFamily::Chroma,
        ] {
            assert_eq!(production_sizes_for_family(family), expected);
        }
        let sdxl = capability_for_family(ModelFamily::Sdxl);
        assert_eq!(sdxl["limits"]["resolution"]["mode"], "shape_dispatch");
        assert_eq!(sdxl["limits"]["sizes"].as_array().unwrap().len(), 7);
        assert_eq!(production_sizes_for_family(ModelFamily::Flux2).len(), 8);
    }

    #[test]
    fn advanced_sampling_capabilities_publish_only_worker_honored_controls() {
        let zimage = capability_for_family(ModelFamily::ZImage);
        let zimage_advanced = &zimage["features"]["advanced_sampling"];
        assert_eq!(zimage_advanced["supported"], json!(true));
        assert_eq!(
            zimage_advanced["parameters"]["sigma_shift"]["supported"],
            json!(true)
        );
        assert_eq!(
            zimage_advanced["parameters"]["sigma_shift"]["default"],
            json!(3.0)
        );
        assert_eq!(
            zimage_advanced["parameters"]["clip_skip"]["supported"],
            json!(false)
        );

        let qwen = capability_for_family(ModelFamily::QwenImage);
        let qwen_advanced = &qwen["features"]["advanced_sampling"];
        assert_eq!(qwen_advanced["supported"], json!(false));
        assert_eq!(
            qwen_advanced["parameters"]["sigma_shift"]["supported"],
            json!(false)
        );
        assert_eq!(
            qwen_advanced["parameters"]["vae"]["supported"],
            json!(false)
        );
    }

    #[test]
    fn canvas_image_sizes_are_server_capability_driven() {
        let model_utils = include_str!("../../../canvas/js/model-utils.js");
        let generate = include_str!("../../../canvas/js/generate.js");
        let simple = include_str!("../../../canvas/js/simple.js");

        assert!(model_utils.contains("fetch('/v1/capabilities'"));
        assert!(model_utils.contains("function aspectsForArch(capabilities, arch)"));
        assert!(generate.contains("ModelUtils.aspectsForArch(state.capabilities, arch)"));
        assert!(simple.contains("ModelUtils.aspectsForArch(state.capabilities, state.arch)"));

        for mapping in [
            "zimage: 'zimage'",
            "qwen: 'qwenimage'",
            "ideogram4: 'ideogram4'",
            "sdxl: 'sdxl'",
            "anima: 'anima'",
            "sd3: 'sd3'",
            "flux: 'flux'",
            "klein: 'flux2'",
            "sensenova: 'sensenova'",
            "krea2: 'krea2'",
            "chroma: 'chroma'",
            "lens: 'lens'",
        ] {
            assert!(
                model_utils.contains(mapping),
                "missing browser mapping {mapping}"
            );
        }

        for stale_table in [
            "image1024AspectLadder",
            "sdxlRuntimeAspects",
            "var imageAspects",
            "IMAGE_RESOLUTIONS",
        ] {
            assert!(!generate.contains(stale_table), "{stale_table} in Generate");
            assert!(!simple.contains(stale_table), "{stale_table} in Simple");
            assert!(
                !model_utils.contains(stale_table),
                "{stale_table} in model-utils"
            );
        }
    }

    #[test]
    fn raw_surface_guard_rejects_disabled_feature_fields() {
        let base = json!({
            "model": "sdxl",
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
            (json!({"lanpaint_num_steps": 16}), "LanPaint"),
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
            ("lanpaint_context_expand", json!(-1)),
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
    fn zimage_raw_surface_admits_init_image_mask_and_creativity_only() {
        let img2img = json!({
            "model": "zimage",
            "prompt": "replace the sky",
            "init_image": "/tmp/init.png",
            "denoise": 0.65
        });
        reject_disabled_raw_surfaces(&img2img).unwrap();

        let inpaint = json!({
            "model": "zimage",
            "prompt": "replace the sky",
            "init_image": "/tmp/init.png",
            "mask_image": "/tmp/mask.png",
            "denoise": 0.65
        });
        reject_disabled_raw_surfaces(&inpaint).unwrap();

        let missing_init = json!({
            "model": "zimage",
            "prompt": "replace the sky",
            "mask_image": "/tmp/mask.png"
        });
        assert!(
            reject_disabled_raw_surfaces(&missing_init)
                .unwrap_err()
                .contains("requires init_image")
        );

        let full_lanpaint = json!({
            "model": "zimage",
            "prompt": "replace the sky",
            "init_image": "/tmp/init.png",
            "mask_image": "/tmp/mask.png",
            "lanpaint_num_steps": 16
        });
        assert!(
            reject_disabled_raw_surfaces(&full_lanpaint)
                .unwrap_err()
                .contains("LanPaint")
        );
    }

    #[test]
    fn lens_profile_admits_only_the_compiled_txt2img_surface() {
        let profile = capability_profile_for_model("microsoft_lens");
        assert_eq!(profile["backend"], "lens");
        assert_eq!(profile["production_status"], "admitted");
        assert_eq!(profile["worker_binary"], "serenity_worker_lens");
        assert_eq!(profile["defaults"]["steps"], 20);
        assert_eq!(profile["defaults"]["cfg"], 5.0);
        assert_eq!(profile["limits"]["sizes"].as_array().unwrap().len(), 1);
        assert_eq!(profile["limits"]["sizes"][0]["width"], 1024);
        assert_eq!(profile["limits"]["sizes"][0]["height"], 1024);
        assert_eq!(profile["features"]["negative_prompt"]["supported"], true);
        assert_eq!(profile["features"]["lora"]["supported"], false);

        let mut params = JobParams::default();
        params.model = "microsoft_lens".to_string();
        params.prompt = "a red ceramic teapot".to_string();
        params.width = 1024;
        params.height = 1024;
        params.steps = 1;
        params.cfg = 5.0;
        params.sampler = "euler".to_string();
        params.scheduler = "simple".to_string();
        assert_eq!(
            validate_generate_prequeue(&params, 1.0).unwrap(),
            ModelFamily::Lens
        );

        params.width = 896;
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("admitted product shapes")
        );
        params.width = 1024;
        params
            .loras
            .push(serenity_wire::LoraSpec::new("unsupported".to_string(), 1.0));
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("LoRA")
        );
    }

    #[test]
    fn zimage_prequeue_admits_bounded_img2img_and_inpaint() {
        let mut params = JobParams::default();
        params.model = "zimage".to_string();
        params.prompt = "replace the sky".to_string();
        params.width = 512;
        params.height = 512;
        params.steps = 4;
        params.cfg = 1.0;
        params.sampler = "flowmatch_euler".to_string();
        params.scheduler = "simple".to_string();
        params.creativity = 0.65;
        params.init_image = "/tmp/init.png".to_string();
        assert_eq!(
            validate_generate_prequeue(&params, 1.0).unwrap(),
            ModelFamily::ZImage
        );

        params.mask_image = "/tmp/mask.png".to_string();
        assert_eq!(
            validate_generate_prequeue(&params, 1.0).unwrap(),
            ModelFamily::ZImage
        );

        params.sampler = "uni_pc".to_string();
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("UniPC img2img/inpaint")
        );

        params.model = "sdxl".to_string();
        params.sampler = "euler".to_string();
        params.scheduler = "normal".to_string();
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("admitted only for Z-Image")
        );
    }

    #[test]
    fn krea2_flowedit_prequeue_admits_both_compiled_square_profiles() {
        let mut params = JobParams::default();
        params.model = "krea2_turbo".to_string();
        params.prompt = "the same subject in polished chrome".to_string();
        params.edit_src_prompt = "a portrait beside a pool".to_string();
        params.edit_src_image = "/tmp/source.png".to_string();
        params.steps = 28;
        params.cfg = 5.5;
        params.sampler = "euler".to_string();
        params.scheduler = "simple".to_string();

        for size in [512, 1024] {
            params.width = size;
            params.height = size;
            assert_eq!(
                validate_generate_prequeue(&params, 1.0).unwrap(),
                ModelFamily::Krea2
            );
        }

        params.width = 768;
        params.height = 768;
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("admitted product shapes")
        );
    }

    #[test]
    fn krea2_lanpaint_raw_and_prequeue_admit_only_complete_1024_profile() {
        let raw = json!({
            "model": "krea2_turbo",
            "prompt": "replace the hand with a red glove",
            "width": 1024,
            "height": 1024,
            "init_image": "/tmp/source.png",
            "mask_image": "/tmp/source.png",
            "lanpaint_mask_channel": "load_image_mask",
            "lanpaint_num_steps": 5,
            "lanpaint_lambda": 16.0,
            "lanpaint_step_size": 0.2,
            "lanpaint_beta": 1.0,
            "lanpaint_friction": 15.0,
            "lanpaint_prompt_mode": "Image First",
            "lanpaint_inpainting_mode": "Image Inpainting",
            "lanpaint_early_stop": 1,
            "lanpaint_mask_blend_overlap": 9,
            "lanpaint_context_expand": 96,
            "denoise": 1.0
        });
        reject_disabled_raw_surfaces(&raw).unwrap();

        let mut params = JobParams::default();
        params.model = "krea2_turbo".to_string();
        params.prompt = "replace the hand with a red glove".to_string();
        params.width = 1024;
        params.height = 1024;
        params.steps = 8;
        params.cfg = 1.0;
        params.sampler = "euler".to_string();
        params.scheduler = "simple".to_string();
        params.creativity = 1.0;
        params.init_image = "/tmp/source.png".to_string();
        params.mask_image = "/tmp/source.png".to_string();
        params.lanpaint_mask_channel = "load_image_mask".to_string();
        params.lanpaint_num_steps = 5;
        params.lanpaint_lambda = 16.0;
        params.lanpaint_step_size = 0.2;
        params.lanpaint_beta = 1.0;
        params.lanpaint_friction = 15.0;
        params.lanpaint_prompt_mode = "Image First".to_string();
        params.lanpaint_inpainting_mode = "Image Inpainting".to_string();
        params.lanpaint_early_stop = 1;
        params.lanpaint_inner_threshold = 0.0;
        params.lanpaint_inner_patience = 1;
        params.lanpaint_mask_blend_overlap = 9;
        params.lanpaint_context_expand = 96;
        assert_eq!(
            validate_generate_prequeue(&params, 1.0).unwrap(),
            ModelFamily::Krea2
        );

        params.width = 512;
        params.height = 512;
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("1024x1024")
        );
        params.width = 1024;
        params.height = 1024;
        params.lanpaint_friction = -1.0;
        assert!(
            validate_generate_prequeue(&params, 1.0)
                .unwrap_err()
                .contains("lanpaint_friction")
        );
    }

    #[test]
    fn capability_document_exposes_only_verified_edit_backends() {
        let zimage = capability_for_family(ModelFamily::ZImage);
        assert_eq!(zimage["limits"]["txt2img_only"], false);
        assert_eq!(zimage["features"]["image_to_image"]["supported"], true);
        assert_eq!(zimage["features"]["inpaint"]["supported"], true);

        let krea2 = capability_for_family(ModelFamily::Krea2);
        let ideogram4 = capability_for_family(ModelFamily::Ideogram4);
        let flux2 = capability_for_family(ModelFamily::Flux2);
        let sdxl = capability_for_family(ModelFamily::Sdxl);
        assert_eq!(krea2["features"]["instruction_edit"]["supported"], true);
        assert_eq!(ideogram4["features"]["instruction_edit"]["supported"], true);
        assert_eq!(flux2["features"]["instruction_edit"]["supported"], true);
        assert_eq!(
            flux2["features"]["instruction_edit"]["engine"],
            "reference_latent"
        );
        assert_eq!(flux2["features"]["image_conditioning"]["supported"], true);
        assert_eq!(flux2["limits"]["txt2img_only"], false);
        assert_eq!(sdxl["features"]["instruction_edit"]["supported"], false);
        assert_eq!(sdxl["features"]["image_to_image"]["supported"], false);
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

    #[test]
    fn krea_turbo_selected_profile_uses_creator_defaults() {
        let profile = capability_profile_for_model("krea2-turbo");
        assert_eq!(profile["backend"], "krea2");
        assert_eq!(profile["defaults"]["steps"], 8);
        assert_eq!(profile["defaults"]["cfg"], 0.0);
        assert_eq!(default_steps_for_model("krea2-raw", ModelFamily::Krea2), 52);
        assert_eq!(default_cfg_for_model("krea2-raw", ModelFamily::Krea2), 3.5);
    }

    #[test]
    fn klein_selected_profile_separates_base_from_distilled_defaults() {
        let base = capability_profile_for_model("flux-2-klein-base-9b");
        assert_eq!(base["backend"], "flux2");
        assert_eq!(base["defaults"]["steps"], 50);
        assert_eq!(base["defaults"]["cfg"], 4.0);

        let distilled = capability_profile_for_model("flux-2-klein-9b");
        assert_eq!(distilled["defaults"]["steps"], 4);
        assert_eq!(distilled["defaults"]["cfg"], 1.0);

        let four_b = capability_profile_for_model("flux-2-klein-base-4b");
        assert_eq!(four_b["backend"], "flux2");
        assert_eq!(four_b["production_status"], "admitted");
        assert_eq!(four_b["features"]["instruction_edit"]["supported"], true);
    }

    #[test]
    fn every_image_family_uses_a_publisher_aligned_generation_profile() {
        let cases = [
            ("zimage_base", "zimage", 28, 4.0),
            ("z_image_turbo_bf16", "zimage", 8, 0.0),
            ("qwen-image-2512", "qwen-image", 50, 4.0),
            ("ideogram-4-fp8", "ideogram4", 48, 7.0),
            ("sd_xl_base_1.0", "sdxl", 50, 7.0),
            ("anima", "anima", 30, 4.5),
            ("sd3.5_large", "sd3", 28, 4.5),
            ("flux1-dev", "flux", 50, 3.5),
            ("flux-2-klein-base-9b", "flux-2/klein", 50, 4.0),
            ("flux-2-klein-9b", "flux-2/klein", 4, 1.0),
            ("sensenova-u1", "sensenova", 50, 4.0),
            ("sensenova-u1-8step-preview", "sensenova", 8, 1.0),
            ("krea2-raw", "krea2", 52, 3.5),
            ("krea2-turbo", "krea2", 8, 0.0),
            ("chroma1_hd_bf16", "chroma", 40, 3.0),
            ("microsoft_lens", "lens", 20, 5.0),
            ("microsoft_lens_turbo", "lens", 4, 1.0),
            ("microsoft_lens_base", "lens", 50, 5.0),
        ];
        for (model, arch, steps, cfg) in cases {
            let defaults = generation_defaults_for_model_arch(model, arch)
                .unwrap_or_else(|| panic!("missing defaults for {model} ({arch})"));
            assert_eq!(defaults["steps"], steps, "wrong step default for {model}");
            assert_eq!(defaults["cfg"], cfg, "wrong CFG default for {model}");
            assert_eq!(defaults["sampler"], "euler");
        }
    }
}
