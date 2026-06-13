# serenitymojo.serve.ipc_codec — Phase-5 worker IPC message (de)serialization.
#
# The transport for process isolation: newline-framed JSON objects between the
# parent ProcessIsolatedBackend and a child worker. This module owns the wire
# schema so neither side re-implements it. Reuses the same json lib + field set
# as the daemon's parse_generate / job serializers (no NEW contract — process
# isolation is a transport change). See PHASE5_PROCESS_ISOLATION_DESIGN.md.
#
# Messages (one JSON object per line):
#   parent->child : {"cmd":"start", <all JobParams fields>}  |  {"cmd":"cancel"}
#   child->parent : {"ev":"ready"}
#                   {"ev":"progress","step":N,"total":M,"phase":"..."}
#                   {"ev":"done","output_path":"..."}
#                   {"ev":"failed","error":"..."}
#                   {"ev":"cancelled"}

from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue

from serenitymojo.serve.backend import JobParams, LoraSpec, StepResult


# ── parent -> child ──────────────────────────────────────────────────────────

def encode_start(p: JobParams) raises -> String:
    """Serialize a fully-built JobParams (as the daemon already has it, with
    params_json populated) into a "start" command line. ALL fields ride,
    including job_id / out_dir / loras / params_json, so the child reconstructs
    an identical JobParams without re-running HTTP validation."""
    var o = JSONValue.new_object()
    o.set("cmd", JSONValue.from_string(String("start")))
    o.set("job_id", JSONValue.from_string(p.job_id))
    o.set("model", JSONValue.from_string(p.model))
    o.set("prompt", JSONValue.from_string(p.prompt))
    o.set("negative", JSONValue.from_string(p.negative))
    o.set("width", JSONValue.from_int(p.width))
    o.set("height", JSONValue.from_int(p.height))
    o.set("steps", JSONValue.from_int(p.steps))
    o.set("seed", JSONValue.from_int(p.seed))
    o.set("cfg", JSONValue.from_float(p.cfg))
    o.set("cfg_override", JSONValue.from_float(p.cfg_override))
    o.set("cfg_override_start_percent", JSONValue.from_float(p.cfg_override_start_percent))
    o.set("cfg_override_end_percent", JSONValue.from_float(p.cfg_override_end_percent))
    o.set("sampler", JSONValue.from_string(p.sampler))
    o.set("scheduler", JSONValue.from_string(p.scheduler))
    o.set("sigma_shift", JSONValue.from_float(p.sigma_shift))
    o.set("variation_seed", JSONValue.from_int(p.variation_seed))
    o.set("variation_strength", JSONValue.from_float(p.variation_strength))
    o.set("images", JSONValue.from_int(p.images))
    o.set("image_index", JSONValue.from_int(p.image_index))
    o.set("image_count", JSONValue.from_int(p.image_count))
    o.set("init_image", JSONValue.from_string(p.init_image))
    o.set("mask_image", JSONValue.from_string(p.mask_image))
    o.set("lanpaint_mask_channel", JSONValue.from_string(p.lanpaint_mask_channel))
    o.set("lanpaint_mask_blend_overlap", JSONValue.from_int(p.lanpaint_mask_blend_overlap))
    o.set("lanpaint_num_steps", JSONValue.from_int(p.lanpaint_num_steps))
    o.set("lanpaint_lambda", JSONValue.from_float(p.lanpaint_lambda))
    o.set("lanpaint_step_size", JSONValue.from_float(p.lanpaint_step_size))
    o.set("lanpaint_beta", JSONValue.from_float(p.lanpaint_beta))
    o.set("lanpaint_friction", JSONValue.from_float(p.lanpaint_friction))
    o.set("lanpaint_prompt_mode", JSONValue.from_string(p.lanpaint_prompt_mode))
    o.set("lanpaint_inpainting_mode", JSONValue.from_string(p.lanpaint_inpainting_mode))
    o.set("lanpaint_add_noise", JSONValue.from_string(p.lanpaint_add_noise))
    o.set("lanpaint_noise_seed", JSONValue.from_int(p.lanpaint_noise_seed))
    o.set("lanpaint_start_at_step", JSONValue.from_int(p.lanpaint_start_at_step))
    o.set("lanpaint_end_at_step", JSONValue.from_int(p.lanpaint_end_at_step))
    o.set("lanpaint_return_with_leftover_noise", JSONValue.from_string(p.lanpaint_return_with_leftover_noise))
    o.set("lanpaint_early_stop", JSONValue.from_int(p.lanpaint_early_stop))
    o.set("lanpaint_inner_threshold", JSONValue.from_float(p.lanpaint_inner_threshold))
    o.set("lanpaint_inner_patience", JSONValue.from_int(p.lanpaint_inner_patience))
    o.set("reference_image", JSONValue.from_string(p.reference_image))
    o.set("reference_latent_method", JSONValue.from_string(p.reference_latent_method))
    o.set("reference_latent_count", JSONValue.from_int(p.reference_latent_count))
    o.set("creativity", JSONValue.from_float(p.creativity))
    o.set("out_dir", JSONValue.from_string(p.out_dir))
    o.set("params_json", JSONValue.from_string(p.params_json))
    var la = JSONValue.new_array()
    for i in range(len(p.loras)):
        var lo = JSONValue.new_object()
        lo.set("name", JSONValue.from_string(p.loras[i].name))
        lo.set("weight", JSONValue.from_float(p.loras[i].weight))
        la.append(lo^)
    o.set("lora", la^)
    return dumps(o)


def decode_start(obj: JSONValue) raises -> JobParams:
    """Reconstruct JobParams from a parsed "start" object (inverse of
    encode_start). Reads fields directly — no defaulting/validation, the parent
    already validated at HTTP time."""
    var p = JobParams()
    p.job_id = obj["job_id"].as_string()
    p.model = obj["model"].as_string()
    p.prompt = obj["prompt"].as_string()
    p.negative = obj["negative"].as_string()
    p.width = Int(obj["width"].as_float())
    p.height = Int(obj["height"].as_float())
    p.steps = Int(obj["steps"].as_float())
    p.seed = Int(obj["seed"].as_float())
    p.cfg = obj["cfg"].as_float()
    if obj.contains("cfg_override") and not obj["cfg_override"].is_null():
        p.cfg_override = obj["cfg_override"].as_float()
    if obj.contains("cfg_override_start_percent") and not obj["cfg_override_start_percent"].is_null():
        p.cfg_override_start_percent = obj["cfg_override_start_percent"].as_float()
    if obj.contains("cfg_override_end_percent") and not obj["cfg_override_end_percent"].is_null():
        p.cfg_override_end_percent = obj["cfg_override_end_percent"].as_float()
    p.sampler = obj["sampler"].as_string()
    p.scheduler = obj["scheduler"].as_string()
    p.sigma_shift = obj["sigma_shift"].as_float()
    p.variation_seed = Int(obj["variation_seed"].as_float())
    p.variation_strength = obj["variation_strength"].as_float()
    p.images = Int(obj["images"].as_float())
    p.image_index = Int(obj["image_index"].as_float())
    p.image_count = Int(obj["image_count"].as_float())
    p.init_image = obj["init_image"].as_string()
    if obj.contains("mask_image") and not obj["mask_image"].is_null():
        p.mask_image = obj["mask_image"].as_string()
    if obj.contains("lanpaint_mask_channel") and not obj["lanpaint_mask_channel"].is_null():
        p.lanpaint_mask_channel = obj["lanpaint_mask_channel"].as_string()
    if obj.contains("lanpaint_mask_blend_overlap") and not obj["lanpaint_mask_blend_overlap"].is_null():
        p.lanpaint_mask_blend_overlap = Int(obj["lanpaint_mask_blend_overlap"].as_float())
    if obj.contains("lanpaint_num_steps") and not obj["lanpaint_num_steps"].is_null():
        p.lanpaint_num_steps = Int(obj["lanpaint_num_steps"].as_float())
    if obj.contains("lanpaint_lambda") and not obj["lanpaint_lambda"].is_null():
        p.lanpaint_lambda = obj["lanpaint_lambda"].as_float()
    if obj.contains("lanpaint_step_size") and not obj["lanpaint_step_size"].is_null():
        p.lanpaint_step_size = obj["lanpaint_step_size"].as_float()
    if obj.contains("lanpaint_beta") and not obj["lanpaint_beta"].is_null():
        p.lanpaint_beta = obj["lanpaint_beta"].as_float()
    if obj.contains("lanpaint_friction") and not obj["lanpaint_friction"].is_null():
        p.lanpaint_friction = obj["lanpaint_friction"].as_float()
    if obj.contains("lanpaint_prompt_mode") and not obj["lanpaint_prompt_mode"].is_null():
        p.lanpaint_prompt_mode = obj["lanpaint_prompt_mode"].as_string()
    if obj.contains("lanpaint_inpainting_mode") and not obj["lanpaint_inpainting_mode"].is_null():
        p.lanpaint_inpainting_mode = obj["lanpaint_inpainting_mode"].as_string()
    if obj.contains("lanpaint_add_noise") and not obj["lanpaint_add_noise"].is_null():
        p.lanpaint_add_noise = obj["lanpaint_add_noise"].as_string()
    if obj.contains("lanpaint_noise_seed") and not obj["lanpaint_noise_seed"].is_null():
        p.lanpaint_noise_seed = Int(obj["lanpaint_noise_seed"].as_float())
    if obj.contains("lanpaint_start_at_step") and not obj["lanpaint_start_at_step"].is_null():
        p.lanpaint_start_at_step = Int(obj["lanpaint_start_at_step"].as_float())
    if obj.contains("lanpaint_end_at_step") and not obj["lanpaint_end_at_step"].is_null():
        p.lanpaint_end_at_step = Int(obj["lanpaint_end_at_step"].as_float())
    if obj.contains("lanpaint_return_with_leftover_noise") and not obj["lanpaint_return_with_leftover_noise"].is_null():
        p.lanpaint_return_with_leftover_noise = obj["lanpaint_return_with_leftover_noise"].as_string()
    if obj.contains("lanpaint_early_stop") and not obj["lanpaint_early_stop"].is_null():
        p.lanpaint_early_stop = Int(obj["lanpaint_early_stop"].as_float())
    if obj.contains("lanpaint_inner_threshold") and not obj["lanpaint_inner_threshold"].is_null():
        p.lanpaint_inner_threshold = obj["lanpaint_inner_threshold"].as_float()
    if obj.contains("lanpaint_inner_patience") and not obj["lanpaint_inner_patience"].is_null():
        p.lanpaint_inner_patience = Int(obj["lanpaint_inner_patience"].as_float())
    if obj.contains("reference_image") and not obj["reference_image"].is_null():
        p.reference_image = obj["reference_image"].as_string()
    if obj.contains("reference_latent_method") and not obj["reference_latent_method"].is_null():
        p.reference_latent_method = obj["reference_latent_method"].as_string()
    if obj.contains("reference_latent_count") and not obj["reference_latent_count"].is_null():
        p.reference_latent_count = Int(obj["reference_latent_count"].as_float())
    p.creativity = obj["creativity"].as_float()
    p.out_dir = obj["out_dir"].as_string()
    p.params_json = obj["params_json"].as_string()
    if obj.contains("lora") and obj["lora"].is_array():
        var arr = obj["lora"]
        for i in range(arr.length()):
            var ent = arr[i]
            p.loras.append(LoraSpec(ent["name"].as_string(), ent["weight"].as_float()))
    return p^


def encode_cancel() raises -> String:
    var o = JSONValue.new_object()
    o.set("cmd", JSONValue.from_string(String("cancel")))
    return dumps(o)


# ── child -> parent ──────────────────────────────────────────────────────────

def encode_ready() raises -> String:
    var o = JSONValue.new_object()
    o.set("ev", JSONValue.from_string(String("ready")))
    return dumps(o)


def encode_ev(r: StepResult) raises -> String:
    """Serialize a StepResult into a child->parent event line."""
    var o = JSONValue.new_object()
    if r.done:
        o.set("ev", JSONValue.from_string(String("done")))
        o.set("output_path", JSONValue.from_string(r.output_path))
    elif r.failed:
        o.set("ev", JSONValue.from_string(String("failed")))
        o.set("error", JSONValue.from_string(r.error))
    elif r.cancelled:
        o.set("ev", JSONValue.from_string(String("cancelled")))
    else:
        o.set("ev", JSONValue.from_string(String("progress")))
        o.set("step", JSONValue.from_int(r.step))
        o.set("total", JSONValue.from_int(r.total))
        o.set("phase", JSONValue.from_string(r.phase))
        o.set("preview", JSONValue.from_string(r.preview))
    return dumps(o)


def decode_ev(obj: JSONValue) raises -> StepResult:
    """Inverse of encode_ev: a child->parent event line into a StepResult.
    A "ready" event maps to a plain no-progress StepResult (step=0)."""
    var r = StepResult()
    var ev = obj["ev"].as_string()
    if ev == "done":
        r.done = True
        r.output_path = obj["output_path"].as_string()
    elif ev == "failed":
        r.failed = True
        r.error = obj["error"].as_string()
    elif ev == "cancelled":
        r.cancelled = True
    elif ev == "progress":
        r.step = Int(obj["step"].as_float())
        r.total = Int(obj["total"].as_float())
        r.phase = obj["phase"].as_string()
        r.preview = obj["preview"].as_string()
    # "ready" -> empty StepResult (no-op tick)
    return r^
