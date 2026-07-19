# serenitymojo.serve.workflow_graph — typed workflow graph adapter/executor.
#
# This is the daemon-side contract the SerenityUI graph editor can target:
# `workflow.nodes` + `workflow.edges` describe a typed value graph, this module
# validates and topologically executes the supported Comfy/Swarm t2i subset,
# then writes the resulting backend fields onto the request object. Tensor
# execution remains in the model backends; this module owns graph semantics,
# typed handles, fail-loud unsupported nodes, and import adapters.

from json.serialize import dumps
from json.value import JSONValue


comptime WORKFLOW_GRAPH_EXECUTOR = "serenity.workflow_graph.executor.v1"
comptime WORKFLOW_SCHEMA = "serenity.workflow_graph.v1"


def _opt_int(obj: JSONValue, key: String, dflt: Int, lo: Int, hi: Int) raises -> Int:
    if not obj.contains(key) or obj[key].is_null():
        return dflt
    if not obj[key].is_int():
        raise Error("'" + key + "' must be an integer")
    var n = obj[key].as_int()
    if n < lo or n > hi:
        raise Error("'" + key + "' out of range [" + String(lo) + ".." + String(hi) + "]")
    return n


def _workflow_string(obj: JSONValue, key: String) raises -> String:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return String("")
    if not obj[key].is_string():
        return String("")
    return obj[key].as_string()


def _workflow_canonical_type_id(var type_id: String) -> String:
    if type_id.startswith("comfy/"):
        type_id = String(type_id.removeprefix("comfy/"))
    return type_id^


def _workflow_type_id(node: JSONValue) raises -> String:
    return _workflow_canonical_type_id(_workflow_string(node, String("type_id")))


def _workflow_is_denoise_node(type_id: String) -> Bool:
    return (
        type_id == "KSampler"
        or type_id == "KSamplerAdvanced"
        or type_id == "LanPaint_KSampler"
        or type_id == "LanPaint_KSamplerAdvanced"
        or type_id == "SamplerCustom"
        or type_id == "SamplerCustomAdvanced"
        or type_id == "LanPaint_SamplerCustomAdvanced"
    )


def _workflow_reject_multi_output_topology(nodes_json: JSONValue) raises:
    """The current product executor lowers to one flat JobParams object.

    Multi-denoise or multi-SaveImage Comfy graphs need real per-node tensor
    outputs. Until that exists, fail loud instead of accepting the graph and
    silently using whichever sampler/save-prefix writes first.
    """
    var denoise_count = 0
    var first_denoise = -1
    var second_denoise = -1
    var save_count = 0
    var first_save = -1
    var second_save = -1
    for i in range(nodes_json.length()):
        var node = nodes_json[i]
        var node_id = _workflow_id(node)
        var type_id = _workflow_type_id(node)
        if _workflow_is_denoise_node(type_id):
            denoise_count += 1
            if first_denoise < 0:
                first_denoise = node_id
            elif second_denoise < 0:
                second_denoise = node_id
        if type_id == "SaveImage":
            save_count += 1
            if first_save < 0:
                first_save = node_id
            elif second_save < 0:
                second_save = node_id

    if denoise_count > 1:
        raise Error(
            "[501] workflow graph has multiple sampler/output branches; "
            + "flat single-job execution supports one sampler node "
            + "(first=" + String(first_denoise)
            + ", second=" + String(second_denoise) + ")"
        )
    if save_count > 1:
        raise Error(
            "[501] workflow graph has multiple SaveImage outputs; "
            + "flat single-job execution supports one SaveImage node "
            + "(first=" + String(first_save)
            + ", second=" + String(second_save) + ")"
        )


# --- SamplerCustom ecosystem: named SAMPLER / SIGMAS node tables ---------------
#
# ComfyUI exposes per-algorithm SAMPLER producers (SamplerEulerAncestral, ...)
# and per-schedule SIGMAS producers (KarrasScheduler, ...). Each carries its
# sampler/scheduler NAME implicitly in the node TYPE, so an unsupported one is a
# definite error (unlike KSamplerSelect, where the user types a free string).
# These lower to the SAME flat sampler=/scheduler= path, GATED on the worker's
# supported list — an unsupported name fails loud [501] (never substituted).


def _workflow_named_sampler_name(type_id: String) -> String:
    """Map a named-SAMPLER node type to its Comfy sampler catalog name (or "").

    Names are the exact comfy.samplers catalog strings each
    nodes_custom_sampler.py class passes to KSAMPLER/sampler_object. Only
    SamplerEuler (-> "euler") is in the worker's supported list; the rest gate
    fail-loud [501] (see _workflow_worker_supports_sampler).
    """
    if type_id == "SamplerEuler":
        return String("euler")
    if type_id == "SamplerEulerAncestral":
        return String("euler_ancestral")
    if type_id == "SamplerEulerAncestralCFGPP":
        return String("euler_ancestral_cfg_pp")
    if type_id == "SamplerDPMPP_2M_SDE":
        return String("dpmpp_2m_sde")
    if type_id == "SamplerDPMPP_3M_SDE":
        return String("dpmpp_3m_sde")
    if type_id == "SamplerDPMPP_SDE":
        return String("dpmpp_sde")
    if type_id == "SamplerDPMPP_2S_Ancestral":
        return String("dpmpp_2s_ancestral")
    if type_id == "SamplerDPMAdaptative":
        return String("dpm_adaptive")
    if type_id == "SamplerLMS":
        return String("lms")
    if type_id == "SamplerER_SDE":
        return String("er_sde")
    if type_id == "SamplerSASolver":
        return String("sa_solver")
    if type_id == "SamplerSEEDS2":
        return String("seeds_2")
    return String("")


def _workflow_is_named_sampler_node(type_id: String) -> Bool:
    return _workflow_named_sampler_name(type_id) != ""


def _workflow_named_scheduler_name(type_id: String) -> String:
    """Map a named-SIGMAS node type to its Comfy scheduler catalog name (or "").

    vp/laplace/polyexponential/turbo are not SCHEDULER_NAMES combo entries (they
    are produced by their own get_sigmas_* functions), but the node TYPE still
    names the schedule, so we carry that name and let the worker gate decide:
    none of these is in the worker's supported list, so they fail-loud [501]
    (beta IS a combo name but still absent from the worker).
    """
    if type_id == "KarrasScheduler":
        return String("karras")
    if type_id == "ExponentialScheduler":
        return String("exponential")
    if type_id == "PolyexponentialScheduler":
        return String("polyexponential")
    if type_id == "SDTurboScheduler":
        return String("turbo")
    if type_id == "VPScheduler":
        return String("vp")
    if type_id == "BetaSamplingScheduler":
        return String("beta")
    if type_id == "LaplaceScheduler":
        return String("laplace")
    return String("")


def _workflow_is_named_scheduler_node(type_id: String) -> Bool:
    return _workflow_named_scheduler_name(type_id) != ""


def _workflow_worker_supports_sampler(name: String) -> Bool:
    """Gate against the zimage worker's supported sampler list.

    Mirrors sampler_registry.swarmui_sampler_registry_json zimage_supported_samplers
    /home/alex/mojodiffusion/serenitymojo/sampling/sampler_registry.mojo.
    """
    var n = String(name.lower())
    return (
        n == "euler"
        or n == "flowmatch_euler"
        or n == "flow_match_euler"
        or n == "dpmpp_2m"
        or n == "dpm++ 2m"
        or n == "uni_pc"
        or n == "uni_pc_bh2"
    )


def _workflow_worker_supports_scheduler(name: String) -> Bool:
    """Gate against the zimage worker's supported scheduler list.

    Mirrors sampler_registry.swarmui_sampler_registry_json zimage_supported_schedulers.
    """
    var n = String(name.lower())
    return (
        n == "simple"
        or n == "flowmatch"
        or n == "flow_match"
        or n == "sgm_uniform"
    )


def _workflow_is_int_scalar_node(type_id: String) -> Bool:
    return type_id == "PrimitiveInt" or type_id == "INTConstant" or type_id == "easy int" or type_id == "SeedNode"


def _workflow_is_float_scalar_node(type_id: String) -> Bool:
    return type_id == "PrimitiveFloat" or type_id == "FloatConstant" or type_id == "easy float"


def _workflow_is_string_scalar_node(type_id: String) -> Bool:
    return (
        type_id == "PrimitiveString"
        or type_id == "PrimitiveStringMultiline"
        or type_id == "StringConstant"
        or type_id == "StringConstantMultiline"
        or type_id == "easy string"
    )


def _workflow_is_bool_scalar_node(type_id: String) -> Bool:
    return type_id == "PrimitiveBoolean" or type_id == "BOOLConstant"


def _workflow_is_scalar_node(type_id: String) -> Bool:
    return (
        _workflow_is_int_scalar_node(type_id)
        or _workflow_is_float_scalar_node(type_id)
        or _workflow_is_string_scalar_node(type_id)
        or _workflow_is_bool_scalar_node(type_id)
        or type_id == "PrimitiveNode"
    )


def _workflow_json_scalar_type(value: JSONValue) -> String:
    if value.is_bool():
        return String("BOOLEAN")
    if value.is_int():
        return String("INT")
    if value.is_number():
        return String("FLOAT")
    if value.is_string():
        return String("STRING")
    return String("")


def _workflow_scalar_output_type(type_id: String, fields: JSONValue) raises -> String:
    if _workflow_is_int_scalar_node(type_id):
        return String("INT")
    if _workflow_is_float_scalar_node(type_id):
        return String("FLOAT")
    if _workflow_is_string_scalar_node(type_id):
        return String("STRING")
    if _workflow_is_bool_scalar_node(type_id):
        return String("BOOLEAN")
    if type_id == "PrimitiveNode":
        var declared = _workflow_string(fields, String("output_type"))
        if declared == "":
            if fields.contains("value"):
                declared = _workflow_json_scalar_type(fields["value"])
            elif fields.contains("text"):
                declared = _workflow_json_scalar_type(fields["text"])
            elif fields.contains("string"):
                declared = _workflow_json_scalar_type(fields["string"])
        if declared == "INT" or declared == "FLOAT" or declared == "STRING" or declared == "BOOLEAN":
            return declared^
    return String("")


def _set_if_missing(mut obj: JSONValue, key: String, value: JSONValue) raises:
    if not obj.contains(key) or obj[key].is_null():
        obj.set(key, value.copy())


def _copy_field_if_missing(
    mut dst: JSONValue, src: JSONValue, src_key: String, dst_key: String,
) raises:
    if src.is_object() and src.contains(src_key) and not src[src_key].is_null():
        _set_if_missing(dst, dst_key, src[src_key])


def _workflow_float(
    obj: JSONValue, key: String, dflt: Float64, lo: Float64, hi: Float64,
) raises -> Float64:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return dflt
    var n: Float64
    if obj[key].is_number():
        n = obj[key].as_float()
    elif obj[key].is_int():
        n = Float64(obj[key].as_int())
    elif obj[key].is_string():
        try:
            n = Float64(obj[key].as_string())
        except:
            raise Error("[501] workflow graph field " + key + " must be numeric")
    else:
        raise Error("[501] workflow graph field " + key + " must be numeric")
    if n < lo or n > hi:
        raise Error("[501] workflow graph field " + key + " out of range")
    return n


def _workflow_int(
    obj: JSONValue, key: String, dflt: Int, lo: Int, hi: Int,
) raises -> Int:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return dflt
    var n: Int
    if obj[key].is_int():
        n = obj[key].as_int()
    elif obj[key].is_number():
        n = Int(obj[key].as_float())
    elif obj[key].is_string():
        try:
            n = Int(obj[key].as_string())
        except:
            raise Error("[501] workflow graph field " + key + " must be an integer")
    else:
        raise Error("[501] workflow graph field " + key + " must be an integer")
    if n < lo or n > hi:
        raise Error("[501] workflow graph field " + key + " out of range")
    return n


def _workflow_round6(v: Float64) -> Float64:
    var scaled = v * 1000000.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 1000000.0
    return Float64(Int(scaled - 0.5)) / 1000000.0


def _workflow_append_lora(mut obj: JSONValue, name: String, weight: Float64) raises:
    if name == "":
        raise Error("[501] workflow graph LoRA loader missing lora_name")
    if weight == 0.0:
        return
    var arr = JSONValue.new_array()
    if obj.contains("lora") and not obj["lora"].is_null():
        if not obj["lora"].is_array():
            raise Error("[501] workflow graph lora metadata must be an array")
        for i in range(obj["lora"].length()):
            arr.append(obj["lora"][i].copy())
    var ent = JSONValue.new_object()
    ent.set("name", JSONValue.from_string(name))
    ent.set("weight", JSONValue.from_float(weight))
    arr.append(ent^)
    obj.set("lora", arr^)


def _record_workflow_execution(
    mut obj: JSONValue, source: String, node_count: Int, edge_count: Int,
) raises:
    obj.set("workflow_schema", JSONValue.from_string(String(WORKFLOW_SCHEMA)))
    obj.set("workflow_executor", JSONValue.from_string(String(WORKFLOW_GRAPH_EXECUTOR)))
    obj.set("workflow_source", JSONValue.from_string(source))
    obj.set("workflow_node_count", JSONValue.from_int(node_count))
    obj.set("workflow_edge_count", JSONValue.from_int(edge_count))


def _workflow_node_type(node: JSONValue) raises -> String:
    var typ = _workflow_string(node, String("type"))
    if typ == "":
        typ = _workflow_string(node, String("type_id"))
    return typ^


def _workflow_node_mode(node: JSONValue) raises -> Int:
    if not node.is_object() or not node.contains("mode") or node["mode"].is_null():
        return 0
    if node["mode"].is_int():
        return node["mode"].as_int()
    if node["mode"].is_number():
        return Int(node["mode"].as_float())
    return 0


def _workflow_widget_string(widgets: JSONValue, idx: Int, dflt: String) raises -> String:
    if not widgets.is_array() or idx < 0 or idx >= widgets.length() or widgets[idx].is_null():
        return dflt
    if widgets[idx].is_string():
        return widgets[idx].as_string()
    if widgets[idx].is_int():
        return String(widgets[idx].as_int())
    if widgets[idx].is_number():
        return String(widgets[idx].as_float())
    return dflt


def _workflow_widget_int(widgets: JSONValue, idx: Int, dflt: Int) raises -> Int:
    if not widgets.is_array() or idx < 0 or idx >= widgets.length() or widgets[idx].is_null():
        return dflt
    if widgets[idx].is_int():
        return widgets[idx].as_int()
    if widgets[idx].is_number():
        return Int(widgets[idx].as_float())
    if widgets[idx].is_string():
        try:
            return Int(widgets[idx].as_string())
        except:
            pass
    return dflt


def _workflow_widget_float(widgets: JSONValue, idx: Int, dflt: Float64) raises -> Float64:
    if not widgets.is_array() or idx < 0 or idx >= widgets.length() or widgets[idx].is_null():
        return dflt
    if widgets[idx].is_number():
        return widgets[idx].as_float()
    if widgets[idx].is_int():
        return Float64(widgets[idx].as_int())
    if widgets[idx].is_string():
        try:
            return Float64(widgets[idx].as_string())
        except:
            pass
    return dflt


def _workflow_widget_bool(widgets: JSONValue, idx: Int, dflt: Bool) raises -> Bool:
    if not widgets.is_array() or idx < 0 or idx >= widgets.length() or widgets[idx].is_null():
        return dflt
    if widgets[idx].is_bool():
        return widgets[idx].as_bool()
    if widgets[idx].is_int():
        return widgets[idx].as_int() != 0
    if widgets[idx].is_string():
        var lower = String(widgets[idx].as_string().lower())
        if lower == "true" or lower == "yes" or lower == "1":
            return True
        if lower == "false" or lower == "no" or lower == "0":
            return False
    return dflt


def _workflow_bool(obj: JSONValue, key: String, dflt: Bool) raises -> Bool:
    if not obj.is_object() or not obj.contains(key) or obj[key].is_null():
        return dflt
    if obj[key].is_bool():
        return obj[key].as_bool()
    if obj[key].is_int():
        return obj[key].as_int() != 0
    if obj[key].is_string():
        var lower = String(obj[key].as_string().lower())
        if lower == "true" or lower == "yes" or lower == "1":
            return True
        if lower == "false" or lower == "no" or lower == "0":
            return False
    raise Error("[501] workflow graph field " + key + " must be a boolean")


def _workflow_has_prompt_override(mut obj: JSONValue) raises -> Bool:
    if obj.contains("prompt") and obj["prompt"].is_string() and obj["prompt"].as_string() != "":
        return True
    if obj.contains("prompt_raw") and obj["prompt_raw"].is_string() and obj["prompt_raw"].as_string() != "":
        _set_if_missing(obj, String("prompt"), obj["prompt_raw"])
        return True
    if obj.contains("prompt_json") and not obj["prompt_json"].is_null():
        var raw = String("")
        if obj["prompt_json"].is_string():
            raw = obj["prompt_json"].as_string()
        elif obj["prompt_json"].is_object() or obj["prompt_json"].is_array():
            raw = dumps(obj["prompt_json"])
        else:
            raise Error("[501] Ideogram4 Comfy export prompt_json must be a string or JSON object/array")
        if raw == "":
            raise Error("[501] Ideogram4 Comfy export prompt_json must be non-empty")
        _set_if_missing(obj, String("prompt"), JSONValue.from_string(raw))
        _set_if_missing(obj, String("prompt_raw"), JSONValue.from_string(raw))
        return True
    return False


def _workflow_set_seed_from_widget_if_missing(mut obj: JSONValue, seed: Int) raises:
    if obj.contains("seed") and not obj["seed"].is_null():
        return
    if seed < 0:
        raise Error(
            "[501] Ideogram4 Comfy export uses randomized seed; provide top-level seed"
        )
    if seed > 4294967295:
        raise Error(
            "[501] Ideogram4 Comfy export seed exceeds the daemon uint32 seed range; provide top-level seed"
        )
    _set_if_missing(obj, String("seed"), JSONValue.from_int(seed))


def _ideogram4_mode_steps(mode: String) raises -> Int:
    if mode == "Quality":
        return 48
    if mode == "Default":
        return 20
    if mode == "Turbo":
        return 12
    raise Error("[501] unsupported Ideogram4 workflow mode: " + mode)


def looks_like_ideogram4_comfy_ui_export(wf: JSONValue) raises -> Bool:
    if not wf.is_object() or not wf.contains("nodes") or not wf["nodes"].is_array():
        return False
    if not wf.contains("links") or not wf["links"].is_array():
        return False
    if not wf.contains("definitions") or not wf["definitions"].is_object():
        return False
    var defs = wf["definitions"]
    if not defs.contains("subgraphs") or not defs["subgraphs"].is_array():
        return False
    var subgraphs = defs["subgraphs"]
    for i in range(subgraphs.length()):
        var sg = subgraphs[i]
        var name = String(_workflow_string(sg, String("name")).lower())
        if name.find("ideogram") >= 0:
            return True
    return False


def apply_ideogram4_comfy_ui_export(mut obj: JSONValue, wf: JSONValue) raises:
    """Import the bounded Comfy UI Ideogram4 txt2img canvas shape.

    The prompt-builder subgraph requires external Gemma/KJ execution, so a real
    prompt override is required. The sampler/scheduler fields are extracted
    through the graph adapter only for the known Ideogram4 text-to-image
    subgraph.
    """
    if not _workflow_has_prompt_override(obj):
        raise Error(
            "[501] Ideogram4 Comfy export uses a prompt-builder subgraph; provide top-level prompt, prompt_raw, or prompt_json"
        )

    var root_nodes = wf["nodes"]
    for i in range(root_nodes.length()):
        var node = root_nodes[i]
        if not node.is_object():
            raise Error("[501] Ideogram4 Comfy export root node must be an object")
        var typ = _workflow_node_type(node)
        var mode = _workflow_node_mode(node)
        if typ == "LoraLoader" or typ == "LoraLoaderModelOnly" or typ == "ZImageLoraModelOnly":
            if mode != 4:
                raise Error(
                    "[501] Ideogram4 Comfy export has active LoRA nodes, but the current Ideogram4 backend does not execute LoRA"
                )
        elif typ == "Seed (rgthree)":
            var widgets = JSONValue.new_array()
            if node.contains("widgets_values") and node["widgets_values"].is_array():
                widgets = node["widgets_values"]
            _workflow_set_seed_from_widget_if_missing(obj, _workflow_widget_int(widgets, 0, -1))
        elif (
            typ == "MarkdownNote"
            or typ == "SaveImage"
            or typ == "ResolutionSelector"
            or typ == "PreviewAny"
            or typ == "Ideogram4PromptBuilderKJ"
            or typ == "83e6e004-48ea-408e-9024-eb49c3d7dc14"
            or typ == "f5f04613-ee09-4cd9-9ada-a880360891d4"
        ):
            pass
        else:
            if mode != 4:
                raise Error("[501] unsupported active Ideogram4 Comfy root node: " + typ)

    var defs = wf["definitions"]
    var subgraphs = defs["subgraphs"]
    var found = False
    var sg_nodes = JSONValue.new_array()
    for i in range(subgraphs.length()):
        var sg = subgraphs[i]
        var name = String(_workflow_string(sg, String("name")).lower())
        if name.find("text to image") >= 0 and name.find("ideogram") >= 0:
            if not sg.contains("nodes") or not sg["nodes"].is_array():
                raise Error("[501] Ideogram4 Comfy export subgraph missing nodes")
            sg_nodes = sg["nodes"]
            found = True
            break
    if not found:
        raise Error("[501] Ideogram4 Comfy export missing Text to Image (Ideogram v4) subgraph")

    var saw_empty_latent = False
    var saw_cond_model = False
    var saw_uncond_model = False
    var saw_clip = False
    var saw_vae = False
    var saw_sampler = False
    var saw_scheduler = False
    var saw_aura = False
    var saw_guider = False
    var saw_cfg_override = False
    var saw_prompt_encode = False
    var selected_mode = String("Default")

    for i in range(sg_nodes.length()):
        var node = sg_nodes[i]
        if not node.is_object():
            raise Error("[501] Ideogram4 Comfy export subgraph node must be an object")
        var typ = _workflow_node_type(node)
        var mode = _workflow_node_mode(node)
        if mode == 4:
            continue
        var widgets = JSONValue.new_array()
        if node.contains("widgets_values") and node["widgets_values"].is_array():
            widgets = node["widgets_values"]

        if typ == "EmptyFlux2LatentImage":
            var batch = _workflow_widget_int(widgets, 2, 1)
            if batch != 1:
                raise Error(
                    "[501] workflow graph EmptyFlux2LatentImage batch_size>1 "
                    + "requires real Comfy latent-batch execution; use flat images=N "
                    + "for serial product fanout"
                )
            _set_if_missing(obj, String("width"), JSONValue.from_int(_workflow_widget_int(widgets, 0, 1024)))
            _set_if_missing(obj, String("height"), JSONValue.from_int(_workflow_widget_int(widgets, 1, 1024)))
            _set_if_missing(obj, String("images"), JSONValue.from_int(1))
            saw_empty_latent = True
        elif typ == "UNETLoader":
            var name = _workflow_widget_string(widgets, 0, String(""))
            var lower = String(name.lower())
            if lower.find("ideogram4_unconditional") >= 0:
                saw_uncond_model = True
            elif lower.find("ideogram4") >= 0:
                saw_cond_model = True
                _set_if_missing(obj, String("model"), JSONValue.from_string(String("ideogram-4-fp8")))
        elif typ == "CLIPLoader":
            var clip_name = String(_workflow_widget_string(widgets, 0, String("")).lower())
            if clip_name.find("qwen3vl") >= 0:
                saw_clip = True
        elif typ == "VAELoader":
            var vae_name = String(_workflow_widget_string(widgets, 0, String("")).lower())
            if vae_name.find("flux2") >= 0:
                saw_vae = True
        elif typ == "CLIPTextEncode":
            saw_prompt_encode = True
        elif typ == "KSamplerSelect":
            _set_if_missing(obj, String("sampler"), JSONValue.from_string(_workflow_widget_string(widgets, 0, String("euler"))))
            saw_sampler = True
        elif typ == "BasicScheduler":
            _set_if_missing(obj, String("scheduler"), JSONValue.from_string(_workflow_widget_string(widgets, 0, String("simple"))))
            saw_scheduler = True
        elif typ == "ModelSamplingAuraFlow":
            _set_if_missing(obj, String("sigma_shift"), JSONValue.from_float(_workflow_widget_float(widgets, 0, 5.0)))
            saw_aura = True
        elif typ == "DualModelGuider":
            _set_if_missing(obj, String("cfg"), JSONValue.from_float(_workflow_widget_float(widgets, 0, 7.0)))
            saw_guider = True
        elif typ == "CFGOverride":
            _set_if_missing(obj, String("cfg_override"), JSONValue.from_float(_workflow_widget_float(widgets, 0, 3.0)))
            _set_if_missing(obj, String("cfg_override_start_percent"), JSONValue.from_float(_workflow_widget_float(widgets, 1, 0.7)))
            _set_if_missing(obj, String("cfg_override_end_percent"), JSONValue.from_float(_workflow_widget_float(widgets, 2, 1.0)))
            saw_cfg_override = True
        elif typ == "CustomCombo":
            selected_mode = _workflow_widget_string(widgets, 0, selected_mode)
        elif (
            typ == "ConditioningZeroOut"
            or typ == "SamplerCustomAdvanced"
            or typ == "VAEDecode"
            or typ == "RandomNoise"
            or typ == "PrimitiveInt"
            or typ == "ComfyMathExpression"
            or typ == "JsonExtractString"
            or typ == "StringReplace"
            or typ == "ComfyNumberConvert"
        ):
            pass
        else:
            raise Error("[501] unsupported Ideogram4 Comfy subgraph node: " + typ)

    _set_if_missing(obj, String("steps"), JSONValue.from_int(_ideogram4_mode_steps(selected_mode)))
    if not (
        saw_empty_latent
        and saw_cond_model
        and saw_uncond_model
        and saw_clip
        and saw_vae
        and saw_sampler
        and saw_scheduler
        and saw_aura
        and saw_guider
        and saw_cfg_override
        and saw_prompt_encode
    ):
        raise Error("[501] Ideogram4 Comfy export is missing required txt2img sampler nodes")
    _record_workflow_execution(obj, String("ideogram4_comfy_ui_export"), sg_nodes.length(), wf["links"].length())


@fieldwise_init
struct WorkflowLink(Copyable, Movable):
    var found: Bool
    var node_id: Int
    var port: String


@fieldwise_init
struct WorkflowValue(Copyable, Movable):
    var node_id: Int
    var port: String
    var typ: String


def _workflow_id(obj: JSONValue) raises -> Int:
    if not obj.is_object() or not obj.contains("id"):
        raise Error("[501] workflow graph node missing id")
    if obj["id"].is_int():
        return obj["id"].as_int()
    if obj["id"].is_string():
        try:
            return Int(obj["id"].as_string())
        except:
            pass
    raise Error("[501] workflow graph node id must be an integer")


def _workflow_ref_node(obj: JSONValue) raises -> Int:
    if not obj.is_object() or not obj.contains("node"):
        raise Error("[501] workflow graph edge endpoint missing node")
    if obj["node"].is_int():
        return obj["node"].as_int()
    if obj["node"].is_string():
        try:
            return Int(obj["node"].as_string())
        except:
            pass
    raise Error("[501] workflow graph edge endpoint node must be an integer")


def _workflow_ref_port(obj: JSONValue) raises -> String:
    if not obj.is_object() or not obj.contains("port") or not obj["port"].is_string():
        raise Error("[501] workflow graph edge endpoint missing port")
    return obj["port"].as_string()


def _workflow_find_input_link(edges: JSONValue, to_node: Int, to_port: String) raises -> WorkflowLink:
    var out = WorkflowLink(False, -1, String(""))
    for i in range(edges.length()):
        var edge = edges[i]
        if not edge.is_object() or not edge.contains("from") or not edge.contains("to"):
            raise Error("[501] workflow graph edge must have from/to endpoints")
        var to = edge["to"]
        var dst_node = _workflow_ref_node(to)
        var dst_port = _workflow_ref_port(to)
        if dst_node == to_node and dst_port == to_port:
            if out.found:
                raise Error("[501] workflow graph input has multiple sources: " + to_port)
            var src = edge["from"]
            out = WorkflowLink(True, _workflow_ref_node(src), _workflow_ref_port(src))
    return out^


def _workflow_find_reroute_input_link(edges: JSONValue, to_node: Int) raises -> WorkflowLink:
    var out = WorkflowLink(False, -1, String(""))
    for i in range(edges.length()):
        var edge = edges[i]
        if not edge.is_object() or not edge.contains("from") or not edge.contains("to"):
            raise Error("[501] workflow graph edge must have from/to endpoints")
        var to = edge["to"]
        var dst_node = _workflow_ref_node(to)
        var dst_port = _workflow_ref_port(to)
        if (
            dst_node == to_node
            and (dst_port == "input" or dst_port == "" or dst_port == "*" or dst_port == "reroute")
        ):
            if out.found:
                raise Error("[501] workflow graph Reroute input has multiple sources")
            var src = edge["from"]
            out = WorkflowLink(True, _workflow_ref_node(src), _workflow_ref_port(src))
    return out^


def _workflow_find_setnode_input_link(edges: JSONValue, to_node: Int) raises -> WorkflowLink:
    var out = WorkflowLink(False, -1, String(""))
    for i in range(edges.length()):
        var edge = edges[i]
        if not edge.is_object() or not edge.contains("from") or not edge.contains("to"):
            raise Error("[501] workflow graph edge must have from/to endpoints")
        var to = edge["to"]
        var dst_node = _workflow_ref_node(to)
        var dst_port = _workflow_ref_port(to)
        var accepted = (
            dst_port == "value"
            or dst_port == "input"
            or dst_port == ""
            or dst_port == "*"
            or dst_port == "MODEL"
            or dst_port == "CLIP"
            or dst_port == "VAE"
            or dst_port == "CONDITIONING"
            or dst_port == "IMAGE"
            or dst_port == "MASK"
            or dst_port == "LATENT"
            or dst_port == "GUIDER"
            or dst_port == "SIGMAS"
            or dst_port == "NOISE"
            or dst_port == "SAMPLER"
            or dst_port == "INT"
            or dst_port == "FLOAT"
            or dst_port == "STRING"
            or dst_port == "BOOLEAN"
        )
        if dst_node == to_node and accepted:
            if out.found:
                raise Error("[501] workflow graph SetNode input has multiple sources")
            var src = edge["from"]
            out = WorkflowLink(True, _workflow_ref_node(src), _workflow_ref_port(src))
    return out^


def _workflow_has_node_id(ids: List[Int], id: Int) -> Bool:
    for i in range(len(ids)):
        if ids[i] == id:
            return True
    return False


def _workflow_value_index(
    nodes: List[Int], ports: List[String], node_id: Int, port: String,
) -> Int:
    for i in range(len(nodes)):
        if nodes[i] == node_id and ports[i] == port:
            return i
    return -1


def _workflow_string_index(values: List[String], target: String) -> Int:
    for i in range(len(values)):
        if values[i] == target:
            return i
    return -1


def _workflow_type_accepts(declared: String, actual: String) -> Bool:
    if declared == "" or declared == "*" or declared == actual:
        return True
    var wrapped = String(",") + declared + String(",")
    var token = String(",") + actual + String(",")
    if wrapped.find(token) >= 0:
        return True
    token = String(", ") + actual + String(",")
    if wrapped.find(token) >= 0:
        return True
    token = String(",") + actual + String(", ")
    return wrapped.find(token) >= 0


def _workflow_setget_supported_type(actual: String) -> Bool:
    return (
        actual == "MODEL"
        or actual == "CLIP"
        or actual == "VAE"
        or actual == "CONDITIONING"
        or actual == "IMAGE"
        or actual == "MASK"
        or actual == "LATENT"
        or actual == "GUIDER"
        or actual == "SIGMAS"
        or actual == "NOISE"
        or actual == "SAMPLER"
        or actual == "COND_LATENT"
        or actual == "INT"
        or actual == "FLOAT"
        or actual == "STRING"
        or actual == "BOOLEAN"
    )


def _workflow_add_value(
    mut nodes: List[Int], mut ports: List[String], mut types: List[String],
    node_id: Int, port: String, typ: String,
) raises:
    if _workflow_value_index(nodes, ports, node_id, port) >= 0:
        raise Error("[501] workflow graph duplicate output value")
    nodes.append(node_id)
    ports.append(port)
    types.append(typ)


def _workflow_add_scalar(
    mut nodes: List[Int], mut ports: List[String], mut types: List[String],
    mut ints: List[Int], mut floats: List[Float64], mut strings: List[String],
    mut bools: List[Bool],
    node_id: Int, port: String, typ: String, int_value: Int, float_value: Float64,
    string_value: String, bool_value: Bool,
) raises:
    if _workflow_value_index(nodes, ports, node_id, port) >= 0:
        raise Error("[501] workflow graph duplicate scalar metadata")
    nodes.append(node_id)
    ports.append(port.copy())
    types.append(typ.copy())
    ints.append(int_value)
    floats.append(float_value)
    strings.append(string_value.copy())
    bools.append(bool_value)


def _workflow_require_scalar_type(
    nodes: List[Int], ports: List[String], types: List[String],
    link: WorkflowLink, expected: String, input_name: String,
) raises -> Int:
    var idx = _workflow_value_index(nodes, ports, link.node_id, link.port)
    if idx < 0:
        raise Error("[501] workflow graph scalar metadata missing source: " + input_name)
    var actual = types[idx]
    if actual != expected:
        raise Error(
            "[501] workflow graph input " + input_name + " expected "
            + expected + " from " + String(link.node_id) + ":" + link.port
            + " but got " + actual
        )
    return idx


def _workflow_scalar_int(
    nodes: List[Int], ports: List[String], types: List[String], ints: List[Int],
    link: WorkflowLink, input_name: String,
) raises -> Int:
    var idx = _workflow_require_scalar_type(nodes, ports, types, link, String("INT"), input_name)
    return ints[idx]


def _workflow_scalar_float(
    nodes: List[Int], ports: List[String], types: List[String], floats: List[Float64],
    link: WorkflowLink, input_name: String,
) raises -> Float64:
    var idx = _workflow_require_scalar_type(nodes, ports, types, link, String("FLOAT"), input_name)
    return floats[idx]


def _workflow_scalar_string(
    nodes: List[Int], ports: List[String], types: List[String], strings: List[String],
    link: WorkflowLink, input_name: String,
) raises -> String:
    var idx = _workflow_require_scalar_type(nodes, ports, types, link, String("STRING"), input_name)
    return strings[idx].copy()


def _workflow_scalar_bool(
    nodes: List[Int], ports: List[String], types: List[String], bools: List[Bool],
    link: WorkflowLink, input_name: String,
) raises -> Bool:
    var idx = _workflow_require_scalar_type(nodes, ports, types, link, String("BOOLEAN"), input_name)
    return bools[idx]


def _workflow_optional_link_ready(
    nodes: List[Int], ports: List[String], link: WorkflowLink,
) -> Bool:
    if not link.found:
        return True
    return _workflow_value_index(nodes, ports, link.node_id, link.port) >= 0


def _workflow_require_value_type(
    nodes: List[Int], ports: List[String], types: List[String],
    link: WorkflowLink, expected: String, input_name: String,
) raises:
    var idx = _workflow_value_index(nodes, ports, link.node_id, link.port)
    if idx < 0:
        raise Error("[501] workflow graph unresolved input: " + input_name)
    var actual = types[idx]
    var composite_ok = (
        actual == "COND_LATENT"
        and (expected == "CONDITIONING" or expected == "LATENT")
    )
    if actual != expected and not composite_ok:
        raise Error(
            "[501] workflow graph input " + input_name + " expected "
            + expected + " from " + String(link.node_id) + ":" + link.port
            + " but got " + actual
        )


def _workflow_model_name(
    nodes: List[Int], ports: List[String], names: List[String], link: WorkflowLink,
) raises -> String:
    for i in range(len(nodes)):
        if nodes[i] == link.node_id and ports[i] == link.port:
            return names[i].copy()
    raise Error("[501] workflow graph model handle missing source")


def _workflow_conditioning_text(
    nodes: List[Int], ports: List[String], texts: List[String], link: WorkflowLink,
) raises -> String:
    for i in range(len(nodes)):
        if nodes[i] == link.node_id and ports[i] == link.port:
            return texts[i].copy()
    raise Error("[501] workflow graph conditioning handle missing source")


def _workflow_image_path(
    nodes: List[Int], ports: List[String], paths: List[String], link: WorkflowLink,
) raises -> String:
    for i in range(len(nodes)):
        if nodes[i] == link.node_id and ports[i] == link.port:
            return paths[i].copy()
    raise Error("[501] workflow graph image handle missing source")


def _workflow_source_meta(
    nodes: List[Int], ports: List[String], metas: List[String], link: WorkflowLink,
) raises -> String:
    for i in range(len(nodes)):
        if nodes[i] == link.node_id and ports[i] == link.port:
            return metas[i].copy()
    raise Error("[501] workflow graph handle metadata missing source")


def _workflow_imagetomask_channel(fields: JSONValue) raises -> String:
    var channel = _workflow_string(fields, String("channel"))
    if channel == "":
        channel = String("red")
    var lower = String(channel.lower())
    if lower == "red" or lower == "green" or lower == "blue":
        return lower^
    if lower == "alpha":
        raise Error(
            "[501] workflow graph ImageToMask alpha is unsupported on Comfy RGB IMAGE; use LoadImage MASK"
        )
    raise Error(String("[501] workflow graph ImageToMask unsupported channel: ") + channel)


def _workflow_latent_index(
    nodes: List[Int], ports: List[String], link: WorkflowLink,
) -> Int:
    for i in range(len(nodes)):
        if nodes[i] == link.node_id and ports[i] == link.port:
            return i
    return -1


def _workflow_copy_value_metadata(
    source: WorkflowLink, dst_node: Int, dst_port: String, actual: String,
    mut model_nodes: List[Int], mut model_ports: List[String], mut model_names: List[String],
    mut cond_nodes: List[Int], mut cond_ports: List[String], mut cond_texts: List[String],
    mut image_nodes: List[Int], mut image_ports: List[String], mut image_paths: List[String],
    mut image_mask_sources: List[String],
    mut mask_nodes: List[Int], mut mask_ports: List[String], mut mask_paths: List[String],
    mut mask_sources: List[String],
    mut latent_nodes: List[Int], mut latent_ports: List[String], mut latent_widths: List[Int],
    mut latent_heights: List[Int], mut latent_images: List[Int],
    mut latent_init_images: List[String], mut latent_mask_images: List[String],
    mut noise_nodes: List[Int], mut noise_ports: List[String], mut noise_seeds: List[Int],
    mut sampler_nodes: List[Int], mut sampler_ports: List[String], mut sampler_names: List[String],
    mut sigmas_nodes: List[Int], mut sigmas_ports: List[String], mut sigmas_steps: List[Int],
    mut sigmas_schedulers: List[String], mut sigmas_denoises: List[Float64],
    mut scalar_nodes: List[Int], mut scalar_ports: List[String], mut scalar_types: List[String],
    mut scalar_ints: List[Int], mut scalar_floats: List[Float64], mut scalar_strings: List[String],
    mut scalar_bools: List[Bool],
) raises:
    if actual == "MODEL":
        model_nodes.append(dst_node); model_ports.append(dst_port.copy())
        model_names.append(_workflow_model_name(model_nodes, model_ports, model_names, source))
    elif actual == "CONDITIONING" or actual == "COND_LATENT":
        cond_nodes.append(dst_node); cond_ports.append(dst_port.copy())
        cond_texts.append(_workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, source))
    if actual == "IMAGE":
        image_nodes.append(dst_node); image_ports.append(dst_port.copy())
        image_paths.append(_workflow_image_path(image_nodes, image_ports, image_paths, source))
        image_mask_sources.append(_workflow_source_meta(image_nodes, image_ports, image_mask_sources, source))
    elif actual == "MASK":
        mask_nodes.append(dst_node); mask_ports.append(dst_port.copy())
        mask_paths.append(_workflow_image_path(mask_nodes, mask_ports, mask_paths, source))
        mask_sources.append(_workflow_source_meta(mask_nodes, mask_ports, mask_sources, source))
    if actual == "LATENT" or actual == "COND_LATENT":
        var latent_idx = _workflow_latent_index(latent_nodes, latent_ports, source)
        latent_nodes.append(dst_node); latent_ports.append(dst_port.copy())
        if latent_idx >= 0:
            latent_widths.append(latent_widths[latent_idx])
            latent_heights.append(latent_heights[latent_idx])
            latent_images.append(latent_images[latent_idx])
            latent_init_images.append(latent_init_images[latent_idx])
            latent_mask_images.append(latent_mask_images[latent_idx])
        else:
            latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
            latent_init_images.append(String(""))
            latent_mask_images.append(String(""))
    elif actual == "NOISE":
        for j in range(len(noise_nodes)):
            if noise_nodes[j] == source.node_id and noise_ports[j] == source.port:
                noise_nodes.append(dst_node); noise_ports.append(dst_port.copy()); noise_seeds.append(noise_seeds[j])
                break
    elif actual == "SAMPLER":
        for j in range(len(sampler_nodes)):
            if sampler_nodes[j] == source.node_id and sampler_ports[j] == source.port:
                sampler_nodes.append(dst_node); sampler_ports.append(dst_port.copy()); sampler_names.append(sampler_names[j])
                break
    elif actual == "SIGMAS":
        for j in range(len(sigmas_nodes)):
            if sigmas_nodes[j] == source.node_id and sigmas_ports[j] == source.port:
                sigmas_nodes.append(dst_node); sigmas_ports.append(dst_port.copy())
                sigmas_steps.append(sigmas_steps[j])
                sigmas_schedulers.append(sigmas_schedulers[j])
                sigmas_denoises.append(sigmas_denoises[j])
                break
    elif actual == "INT" or actual == "FLOAT" or actual == "STRING" or actual == "BOOLEAN":
        var scalar_idx = _workflow_value_index(scalar_nodes, scalar_ports, source.node_id, source.port)
        if scalar_idx < 0:
            raise Error("[501] workflow graph scalar metadata missing source")
        var int_value = scalar_ints[scalar_idx]
        var float_value = scalar_floats[scalar_idx]
        var string_value = scalar_strings[scalar_idx].copy()
        var bool_value = scalar_bools[scalar_idx]
        _workflow_add_scalar(
            scalar_nodes, scalar_ports, scalar_types,
            scalar_ints, scalar_floats, scalar_strings, scalar_bools,
            dst_node, dst_port, actual,
            int_value, float_value, string_value, bool_value,
        )


def _workflow_set_field_if_nonnegative_int(
    mut obj: JSONValue, fields: JSONValue, src_key: String, dst_key: String,
) raises:
    if not fields.is_object() or not fields.contains(src_key) or fields[src_key].is_null():
        return
    if not fields[src_key].is_int():
        raise Error("[501] workflow graph field " + src_key + " must be an integer")
    var v = fields[src_key].as_int()
    if v >= 0:
        _set_if_missing(obj, dst_key, JSONValue.from_int(v))


def _workflow_loader_model_name(fields: JSONValue) raises -> String:
    var model_name = _workflow_string(fields, String("ckpt_name"))
    if model_name == "":
        model_name = _workflow_string(fields, String("unet_name"))
    if model_name == "":
        model_name = _workflow_string(fields, String("model_name"))
    return model_name^


def _workflow_conditioning_prompt_text(fields: JSONValue, type_id: String) raises -> String:
    var text = _workflow_string(fields, String("text"))
    if text != "":
        return text^
    if type_id == "CLIPTextEncodeFlux":
        text = _workflow_string(fields, String("t5xxl"))
        if text != "":
            return text^
        text = _workflow_string(fields, String("clip_l"))
    return text^


def _workflow_setget_name(fields: JSONValue) raises -> String:
    var name = _workflow_string(fields, String("name"))
    if name == "":
        name = _workflow_string(fields, String("variable"))
    if name == "":
        name = _workflow_string(fields, String("key"))
    if name == "":
        name = _workflow_string(fields, String("set_name"))
    return name^


def _workflow_copy_lanpaint_field_alias(
    mut obj: JSONValue, fields: JSONValue, src_key: String, dst_key: String,
) raises:
    _copy_field_if_missing(obj, fields, src_key, dst_key)
    _copy_field_if_missing(obj, fields, dst_key, dst_key)


def _workflow_copy_lanpaint_sampler_fields(
    mut obj: JSONValue, fields: JSONValue,
) raises:
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_NumSteps"), String("lanpaint_num_steps"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_Lambda"), String("lanpaint_lambda"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_StepSize"), String("lanpaint_step_size"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_Beta"), String("lanpaint_beta"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_Friction"), String("lanpaint_friction"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_PromptMode"), String("lanpaint_prompt_mode"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("Inpainting_mode"), String("lanpaint_inpainting_mode"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("add_noise"), String("lanpaint_add_noise"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("noise_seed"), String("lanpaint_noise_seed"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("start_at_step"), String("lanpaint_start_at_step"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("end_at_step"), String("lanpaint_end_at_step"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("return_with_leftover_noise"), String("lanpaint_return_with_leftover_noise"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_EarlyStop"), String("lanpaint_early_stop"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_InnerThreshold"), String("lanpaint_inner_threshold"))
    _workflow_copy_lanpaint_field_alias(obj, fields, String("LanPaint_InnerPatience"), String("lanpaint_inner_patience"))


def _json_intish(v: JSONValue, label: String) raises -> Int:
    if v.is_int():
        return v.as_int()
    if v.is_number():
        return Int(v.as_float())
    if v.is_string():
        try:
            return Int(v.as_string())
        except:
            pass
    raise Error("[501] Comfy UI canvas " + label + " must be an integer")


def looks_like_comfy_ui_canvas_graph(wf: JSONValue) raises -> Bool:
    if not wf.is_object():
        return False
    if not wf.contains("nodes") or not wf["nodes"].is_array():
        return False
    if not wf.contains("links") or not wf["links"].is_array():
        return False
    var nodes = wf["nodes"]
    for i in range(nodes.length()):
        var node = nodes[i]
        if not node.is_object():
            return False
        if _workflow_node_type(node) == "":
            return False
    return True


def _comfy_ui_node_index(nodes: JSONValue, node_id: Int, active_only: Bool) raises -> Int:
    for i in range(nodes.length()):
        var node = nodes[i]
        if not node.is_object():
            raise Error("[501] Comfy UI canvas node must be an object")
        if _workflow_id(node) == node_id:
            if active_only and _workflow_node_mode(node) == 4:
                return -1
            return i
    return -1


def _comfy_ui_output_port(nodes: JSONValue, src_id: Int, src_slot: Int) raises -> String:
    var idx = _comfy_ui_node_index(nodes, src_id, True)
    if idx < 0:
        raise Error("[501] Comfy UI canvas link references missing active source node")
    var node = nodes[idx]
    if not node.contains("outputs") or not node["outputs"].is_array():
        raise Error("[501] Comfy UI canvas source node missing outputs")
    var outputs = node["outputs"]
    if src_slot < 0 or src_slot >= outputs.length():
        raise Error("[501] Comfy UI canvas source output slot out of range")
    var out = outputs[src_slot]
    var typ = _workflow_string(out, String("type"))
    var name = _workflow_string(out, String("name"))
    var node_type = _workflow_canonical_type_id(_workflow_node_type(node))
    if node_type == "SetNode":
        return String("SET")
    if node_type == "GetNode":
        return String("GET")
    if node_type == "Reroute":
        return String("REROUTE")
    if _workflow_is_scalar_node(node_type):
        if typ == "INT" or typ == "FLOAT" or typ == "STRING" or typ == "BOOLEAN":
            return typ^
    if node_type == "InpaintModelConditioning":
        if name == "positive" or name == "negative":
            return name^
        if typ == "LATENT" or name == "latent":
            return String("LATENT")
    if name == "CONDITIONING_1":
        return name^
    if typ == "INT":
        if name != "":
            return name^
        return String("INT")
    if (
        typ == "MODEL"
        or typ == "CLIP"
        or typ == "VAE"
        or typ == "CONDITIONING"
        or typ == "IMAGE"
        or typ == "MASK"
        or typ == "LATENT"
        or typ == "GUIDER"
        or typ == "SIGMAS"
        or typ == "NOISE"
        or typ == "SAMPLER"
    ):
        return typ^
    if name != "":
        return name^
    raise Error("[501] Comfy UI canvas source output missing type/name")


def _comfy_ui_input_port(nodes: JSONValue, dst_id: Int, dst_slot: Int) raises -> String:
    var idx = _comfy_ui_node_index(nodes, dst_id, True)
    if idx < 0:
        raise Error("[501] Comfy UI canvas link references missing active target node")
    var node = nodes[idx]
    if not node.contains("inputs") or not node["inputs"].is_array():
        raise Error("[501] Comfy UI canvas target node missing inputs")
    var inputs = node["inputs"]
    if dst_slot < 0 or dst_slot >= inputs.length():
        raise Error("[501] Comfy UI canvas target input slot out of range")
    var port = _workflow_string(inputs[dst_slot], String("name"))
    var node_type = _workflow_canonical_type_id(_workflow_node_type(node))
    if node_type == "SetNode":
        return String("value")
    if node_type == "Reroute" and port == "":
        return String("input")
    if port == "":
        raise Error("[501] Comfy UI canvas target input missing name")
    return port^


def _comfy_ui_lanpaint_mode(widgets: JSONValue, idx: Int) raises -> String:
    var raw = _workflow_widget_string(widgets, idx, String(""))
    var lower = String(raw.lower())
    if lower.find("video") >= 0:
        return String("video")
    if lower.find("image") >= 0:
        return String("image")
    return raw^


def _comfy_ui_widget_fields(type_id: String, widgets: JSONValue) raises -> JSONValue:
    var fields = JSONValue.new_object()
    if type_id == "CheckpointLoaderSimple":
        fields.set("ckpt_name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "UNETLoader" or type_id == "DiffusionModelLoader":
        fields.set("unet_name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "LoraLoader" or type_id == "LoraLoaderModelOnly" or type_id == "ZImageLoraModelOnly":
        fields.set("lora_name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
        fields.set("strength_model", JSONValue.from_float(_workflow_widget_float(widgets, 1, 1.0)))
        if type_id == "LoraLoader":
            fields.set("strength_clip", JSONValue.from_float(_workflow_widget_float(widgets, 2, 1.0)))
    elif (
        type_id == "CLIPTextEncode"
        or type_id == "CLIPTextEncodeFlux"
        or type_id == "TextEncodeQwenImageEdit"
        or type_id == "TextEncodeQwenImageEditPlus"
    ):
        fields.set("text", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "LoadImage" or type_id == "LoadImageOutput" or type_id == "LoadImageMask":
        fields.set("image", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "ImageScaleBy":
        # Comfy widget order: [upscale_method(combo), scale_by(float)].
        fields.set("scale_by", JSONValue.from_float(_workflow_widget_float(widgets, 1, 1.0)))
    elif type_id == "ImageResizeKJ":
        # KJ widget order: [width, height, upscale_method, keep_proportion,
        # divisible_by]. width/height carry the explicit target dims;
        # keep_proportion/divisible_by gate fail-loud in the executor (they need
        # the un-knowable source dims to compute the result).
        fields.set("width", JSONValue.from_int(_workflow_widget_int(widgets, 0, 512)))
        fields.set("height", JSONValue.from_int(_workflow_widget_int(widgets, 1, 512)))
        fields.set("keep_proportion", JSONValue.from_bool(_workflow_widget_bool(widgets, 3, False)))
        fields.set("divisible_by", JSONValue.from_int(_workflow_widget_int(widgets, 4, 2)))
    elif type_id == "EmptyLatentImage" or type_id == "EmptySD3LatentImage" or type_id == "EmptyFlux2LatentImage":
        fields.set("width", JSONValue.from_int(_workflow_widget_int(widgets, 0, 512)))
        fields.set("height", JSONValue.from_int(_workflow_widget_int(widgets, 1, 512)))
        fields.set("batch_size", JSONValue.from_int(_workflow_widget_int(widgets, 2, 1)))
    elif type_id == "KSampler":
        fields.set("seed", JSONValue.from_int(_workflow_widget_int(widgets, 0, -1)))
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 2, 20)))
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 3, 4.5)))
        fields.set("sampler_name", JSONValue.from_string(_workflow_widget_string(widgets, 4, String("euler"))))
        fields.set("scheduler", JSONValue.from_string(_workflow_widget_string(widgets, 5, String("simple"))))
        fields.set("denoise", JSONValue.from_float(_workflow_widget_float(widgets, 6, 1.0)))
    elif type_id == "KSamplerAdvanced":
        fields.set("add_noise", JSONValue.from_string(_workflow_widget_string(widgets, 0, String("enable"))))
        fields.set("noise_seed", JSONValue.from_int(_workflow_widget_int(widgets, 1, -1)))
        fields.set("seed", JSONValue.from_int(_workflow_widget_int(widgets, 1, -1)))
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 3, 20)))
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 4, 4.5)))
        fields.set("sampler_name", JSONValue.from_string(_workflow_widget_string(widgets, 5, String("euler"))))
        fields.set("scheduler", JSONValue.from_string(_workflow_widget_string(widgets, 6, String("simple"))))
        fields.set("start_at_step", JSONValue.from_int(_workflow_widget_int(widgets, 7, 0)))
        fields.set("end_at_step", JSONValue.from_int(_workflow_widget_int(widgets, 8, 10000)))
        fields.set("return_with_leftover_noise", JSONValue.from_string(_workflow_widget_string(widgets, 9, String("disable"))))
    elif type_id == "LanPaint_KSampler":
        fields.set("seed", JSONValue.from_int(_workflow_widget_int(widgets, 0, -1)))
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 2, 20)))
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 3, 4.5)))
        fields.set("sampler_name", JSONValue.from_string(_workflow_widget_string(widgets, 4, String("euler"))))
        fields.set("scheduler", JSONValue.from_string(_workflow_widget_string(widgets, 5, String("simple"))))
        fields.set("denoise", JSONValue.from_float(_workflow_widget_float(widgets, 6, 1.0)))
        fields.set("LanPaint_NumSteps", JSONValue.from_int(_workflow_widget_int(widgets, 7, -1)))
        fields.set("LanPaint_PromptMode", JSONValue.from_string(_workflow_widget_string(widgets, 8, String(""))))
        fields.set("Inpainting_mode", JSONValue.from_string(_comfy_ui_lanpaint_mode(widgets, 10)))
    elif type_id == "LanPaint_KSamplerAdvanced":
        fields.set("add_noise", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
        fields.set("noise_seed", JSONValue.from_int(_workflow_widget_int(widgets, 1, -1)))
        fields.set("seed", JSONValue.from_int(_workflow_widget_int(widgets, 1, -1)))
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 3, 20)))
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 4, 4.5)))
        fields.set("sampler_name", JSONValue.from_string(_workflow_widget_string(widgets, 5, String("euler"))))
        fields.set("scheduler", JSONValue.from_string(_workflow_widget_string(widgets, 6, String("simple"))))
        fields.set("start_at_step", JSONValue.from_int(_workflow_widget_int(widgets, 7, -1)))
        fields.set("end_at_step", JSONValue.from_int(_workflow_widget_int(widgets, 8, -1)))
        fields.set("return_with_leftover_noise", JSONValue.from_string(_workflow_widget_string(widgets, 9, String(""))))
        fields.set("LanPaint_NumSteps", JSONValue.from_int(_workflow_widget_int(widgets, 10, -1)))
        fields.set("LanPaint_Lambda", JSONValue.from_float(_workflow_widget_float(widgets, 11, -1.0)))
        fields.set("LanPaint_StepSize", JSONValue.from_float(_workflow_widget_float(widgets, 12, -1.0)))
        fields.set("LanPaint_Beta", JSONValue.from_float(_workflow_widget_float(widgets, 13, -1.0)))
        fields.set("LanPaint_Friction", JSONValue.from_float(_workflow_widget_float(widgets, 14, -1.0)))
        fields.set("LanPaint_PromptMode", JSONValue.from_string(_workflow_widget_string(widgets, 15, String(""))))
        fields.set("LanPaint_EarlyStop", JSONValue.from_int(_workflow_widget_int(widgets, 16, -1)))
        fields.set("Inpainting_mode", JSONValue.from_string(_comfy_ui_lanpaint_mode(widgets, 18)))
    elif type_id == "LanPaint_SamplerCustomAdvanced":
        fields.set("LanPaint_NumSteps", JSONValue.from_int(_workflow_widget_int(widgets, 0, -1)))
        fields.set("LanPaint_Lambda", JSONValue.from_float(_workflow_widget_float(widgets, 1, -1.0)))
        fields.set("LanPaint_StepSize", JSONValue.from_float(_workflow_widget_float(widgets, 2, -1.0)))
        fields.set("LanPaint_Beta", JSONValue.from_float(_workflow_widget_float(widgets, 3, -1.0)))
        fields.set("LanPaint_Friction", JSONValue.from_float(_workflow_widget_float(widgets, 4, -1.0)))
        fields.set("LanPaint_PromptMode", JSONValue.from_string(_workflow_widget_string(widgets, 5, String(""))))
        fields.set("LanPaint_EarlyStop", JSONValue.from_int(_workflow_widget_int(widgets, 6, -1)))
        fields.set("LanPaint_InnerThreshold", JSONValue.from_float(_workflow_widget_float(widgets, 8, -1.0)))
        fields.set("LanPaint_InnerPatience", JSONValue.from_int(_workflow_widget_int(widgets, 9, -1)))
    elif type_id == "LanPaint_MaskBlend":
        fields.set("blend_overlap", JSONValue.from_int(_workflow_widget_int(widgets, 0, -1)))
    elif type_id == "ImagePadForOutpaint":
        fields.set("left", JSONValue.from_int(_workflow_widget_int(widgets, 0, 0)))
        fields.set("top", JSONValue.from_int(_workflow_widget_int(widgets, 1, 0)))
        fields.set("right", JSONValue.from_int(_workflow_widget_int(widgets, 2, 0)))
        fields.set("bottom", JSONValue.from_int(_workflow_widget_int(widgets, 3, 0)))
        fields.set("feathering", JSONValue.from_int(_workflow_widget_int(widgets, 4, 40)))
    elif type_id == "ThresholdMask":
        fields.set("value", JSONValue.from_float(_workflow_widget_float(widgets, 0, 0.5)))
    elif type_id == "ConditioningSetMask":
        fields.set("strength", JSONValue.from_float(_workflow_widget_float(widgets, 0, 1.0)))
        fields.set("set_cond_area", JSONValue.from_string(_workflow_widget_string(widgets, 1, String("default"))))
    elif type_id == "SetNode" or type_id == "GetNode":
        fields.set("name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif _workflow_is_int_scalar_node(type_id):
        fields.set("value", JSONValue.from_int(_workflow_widget_int(widgets, 0, 0)))
    elif _workflow_is_float_scalar_node(type_id):
        fields.set("value", JSONValue.from_float(_workflow_widget_float(widgets, 0, 0.0)))
    elif type_id == "StringConstant" or type_id == "StringConstantMultiline":
        fields.set("string", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
        if type_id == "StringConstantMultiline":
            fields.set("strip_newlines", JSONValue.from_bool(_workflow_widget_bool(widgets, 1, True)))
    elif _workflow_is_string_scalar_node(type_id):
        fields.set("value", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif _workflow_is_bool_scalar_node(type_id):
        fields.set("value", JSONValue.from_bool(_workflow_widget_bool(widgets, 0, False)))
    elif type_id == "PrimitiveNode":
        if widgets.is_array() and widgets.length() > 0:
            fields.set("value", widgets[0].copy())
    elif type_id == "InpaintModelConditioning":
        fields.set("noise_mask", JSONValue.from_bool(_workflow_widget_bool(widgets, 0, True)))
    elif type_id == "VAEEncodeForInpaint":
        # Comfy widget order: [grow_mask_by(int)]. Carried for parity but the
        # flat model has no mask-grow control; the handler aliases to inpaint_*.
        fields.set("grow_mask_by", JSONValue.from_int(_workflow_widget_int(widgets, 0, 6)))
    elif type_id == "RepeatLatentBatch":
        # Comfy widget order: [amount(int)] — the batch-repeat count.
        fields.set("amount", JSONValue.from_int(_workflow_widget_int(widgets, 0, 1)))
    elif type_id == "SaveImage":
        fields.set("filename_prefix", JSONValue.from_string(_workflow_widget_string(widgets, 0, String("ComfyUI"))))
    elif type_id == "ImageToMask":
        fields.set("channel", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "RandomNoise":
        fields.set("noise_seed", JSONValue.from_int(_workflow_widget_int(widgets, 0, -1)))
    elif type_id == "KSamplerSelect":
        fields.set("sampler_name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String("euler"))))
    elif type_id == "Flux2Scheduler":
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 0, 20)))
    elif type_id == "BasicScheduler":
        fields.set("scheduler", JSONValue.from_string(_workflow_widget_string(widgets, 0, String("simple"))))
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 1, 20)))
        fields.set("denoise", JSONValue.from_float(_workflow_widget_float(widgets, 2, 1.0)))
    elif (
        type_id == "KarrasScheduler"
        or type_id == "ExponentialScheduler"
        or type_id == "PolyexponentialScheduler"
        or type_id == "VPScheduler"
        or type_id == "LaplaceScheduler"
    ):
        # Comfy widget order: [steps, <shape params: sigma_max/min/rho or
        # beta_d/beta_min/eps_s or mu/sigma>]. Only steps has a flat slot; the
        # shape params have no flat representation (the scheduler name itself
        # gates fail-loud below — none of these is worker-supported).
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 0, 20)))
    elif type_id == "BetaSamplingScheduler":
        # Comfy widget order: [steps, alpha, beta]. Only steps is flat.
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 0, 20)))
    elif type_id == "SDTurboScheduler":
        # Comfy widget order: [steps, denoise].
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 0, 20)))
        fields.set("denoise", JSONValue.from_float(_workflow_widget_float(widgets, 1, 1.0)))
    elif type_id == "SamplerCustom":
        # Comfy widget order: [add_noise(bool), noise_seed(int),
        # control_after_generate, cfg(float)].
        fields.set("add_noise", JSONValue.from_bool(_workflow_widget_bool(widgets, 0, True)))
        fields.set("noise_seed", JSONValue.from_int(_workflow_widget_int(widgets, 1, -1)))
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 3, 4.5)))
    elif type_id == "CFGGuider" or type_id == "FluxGuidance":
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 0, 4.5)))
    elif type_id == "ModelSamplingAuraFlow" or type_id == "ModelSamplingSD3":
        fields.set("shift", JSONValue.from_float(_workflow_widget_float(widgets, 0, 3.0)))
    elif type_id == "ComfySwitchNode":
        fields.set("switch", JSONValue.from_bool(_workflow_widget_bool(widgets, 0, False)))
    return fields^


def comfy_ui_canvas_to_typed_graph(wf: JSONValue) raises -> JSONValue:
    var out = JSONValue.new_object()
    var nodes = JSONValue.new_array()
    var edges = JSONValue.new_array()
    var src_nodes = wf["nodes"]
    for i in range(src_nodes.length()):
        var src = src_nodes[i]
        if not src.is_object():
            raise Error("[501] Comfy UI canvas node must be an object")
        if _workflow_node_mode(src) == 4:
            continue
        var type_id = _workflow_canonical_type_id(_workflow_node_type(src))
        var widgets = JSONValue.new_array()
        if src.contains("widgets_values") and src["widgets_values"].is_array():
            widgets = src["widgets_values"]
        var fields = _comfy_ui_widget_fields(type_id, widgets)
        if (
            (type_id == "GetNode" or type_id == "SetNode" or _workflow_is_scalar_node(type_id))
            and src.contains("outputs")
            and src["outputs"].is_array()
            and src["outputs"].length() > 0
            and src["outputs"][0].is_object()
        ):
            var output_type = _workflow_string(src["outputs"][0], String("type"))
            if output_type != "":
                fields.set("output_type", JSONValue.from_string(output_type))
        var dst = JSONValue.new_object()
        dst.set("id", JSONValue.from_int(_workflow_id(src)))
        dst.set("type_id", JSONValue.from_string(type_id))
        dst.set("fields", fields.copy())
        nodes.append(dst.copy())

    var links = wf["links"]
    for i in range(links.length()):
        var link = links[i]
        if not link.is_array() or link.length() < 6:
            continue
        var src_id = _json_intish(link[1], String("source node id"))
        var src_slot = _json_intish(link[2], String("source output slot"))
        var dst_id = _json_intish(link[3], String("target node id"))
        var dst_slot = _json_intish(link[4], String("target input slot"))
        if _comfy_ui_node_index(src_nodes, src_id, True) < 0 or _comfy_ui_node_index(src_nodes, dst_id, True) < 0:
            continue
        var edge = JSONValue.new_object()
        var from_ref = JSONValue.new_object()
        var to_ref = JSONValue.new_object()
        from_ref.set("node", JSONValue.from_int(src_id))
        from_ref.set("port", JSONValue.from_string(_comfy_ui_output_port(src_nodes, src_id, src_slot)))
        to_ref.set("node", JSONValue.from_int(dst_id))
        to_ref.set("port", JSONValue.from_string(_comfy_ui_input_port(src_nodes, dst_id, dst_slot)))
        edge.set("from", from_ref.copy())
        edge.set("to", to_ref.copy())
        edges.append(edge.copy())
    out.set("nodes", nodes.copy())
    out.set("edges", edges.copy())
    return out^


def _comfy_api_prompt_body(wf: JSONValue) raises -> JSONValue:
    if wf.is_object() and wf.contains("prompt") and wf["prompt"].is_object():
        return wf["prompt"]
    if wf.is_object() and wf.contains("comfy_prompt") and wf["comfy_prompt"].is_object():
        return wf["comfy_prompt"]
    return wf.copy()


def looks_like_comfy_api_prompt_graph(wf: JSONValue) raises -> Bool:
    if not wf.is_object():
        return False
    var graph = _comfy_api_prompt_body(wf)
    if not graph.is_object() or graph.length() == 0:
        return False
    var keys = graph.keys()
    for i in range(len(keys)):
        try:
            _ = Int(keys[i])
        except:
            return False
        var node = graph[keys[i]]
        if not node.is_object():
            return False
        if not node.contains("class_type") or not node["class_type"].is_string():
            return False
        if not node.contains("inputs") or not node["inputs"].is_object():
            return False
    return True


def _comfy_api_link_node_id(v: JSONValue) raises -> Int:
    if not v.is_array() or v.length() < 2:
        raise Error("[501] Comfy API prompt link must be [node_id, output_index]")
    if v[0].is_int():
        return v[0].as_int()
    if v[0].is_string():
        try:
            return Int(v[0].as_string())
        except:
            pass
    raise Error("[501] Comfy API prompt link node_id must be an integer")


def _comfy_api_link_output_slot(v: JSONValue) raises -> Int:
    if not v.is_array() or v.length() < 2:
        raise Error("[501] Comfy API prompt link must be [node_id, output_index]")
    if v[1].is_int():
        return v[1].as_int()
    if v[1].is_number():
        return Int(v[1].as_float())
    raise Error("[501] Comfy API prompt link output_index must be an integer")


def _comfy_api_input_is_link(v: JSONValue) -> Bool:
    if not v.is_array() or v.length() < 2:
        return False
    if not (v[0].is_int() or v[0].is_string()):
        return False
    return v[1].is_int() or v[1].is_number()


def _comfy_api_output_port(graph: JSONValue, src_id: Int, slot: Int) raises -> String:
    var key = String(src_id)
    if not graph.contains(key):
        raise Error("[501] Comfy API prompt link references missing node: " + key)
    var node = graph[key]
    var typ = _workflow_canonical_type_id(_workflow_string(node, String("class_type")))
    if _workflow_is_scalar_node(typ):
        if slot == 0:
            if typ == "PrimitiveNode" and node.contains("inputs") and node["inputs"].is_object():
                var primitive_type = _workflow_scalar_output_type(typ, node["inputs"])
                if primitive_type != "":
                    return primitive_type^
            else:
                var empty_fields = JSONValue.new_object()
                var scalar_type = _workflow_scalar_output_type(typ, empty_fields)
                if scalar_type != "":
                    return scalar_type^
    if typ == "CheckpointLoaderSimple":
        if slot == 0:
            return String("MODEL")
        if slot == 1:
            return String("CLIP")
        if slot == 2:
            return String("VAE")
    elif typ == "UNETLoader" or typ == "DiffusionModelLoader":
        if slot == 0:
            return String("MODEL")
    elif typ == "LoraLoaderModelOnly" or typ == "ZImageLoraModelOnly":
        if slot == 0:
            return String("MODEL")
    elif typ == "LoraLoader":
        if slot == 0:
            return String("MODEL")
        if slot == 1:
            return String("CLIP")
    elif typ == "CLIPLoader" or typ == "DualCLIPLoader" or typ == "TripleCLIPLoader":
        if slot == 0:
            return String("CLIP")
    elif typ == "VAELoader":
        if slot == 0:
            return String("VAE")
    elif (
        typ == "CLIPTextEncode"
        or typ == "CLIPTextEncodeFlux"
        or typ == "TextEncodeQwenImageEdit"
        or typ == "TextEncodeQwenImageEditPlus"
    ):
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "ConditioningZeroOut":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "ConditioningSetMask":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "ConditioningConcat" or typ == "ConditioningCombine" or typ == "ConditioningAverage":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "LoadImage" or typ == "LoadImageOutput" or typ == "LoadImageMask":
        if slot == 0:
            return String("IMAGE")
        if slot == 1:
            return String("MASK")
    elif typ == "ImageToMask":
        if slot == 0:
            return String("MASK")
    elif typ == "MaskToImage":
        if slot == 0:
            return String("IMAGE")
    elif typ == "EmptyLatentImage" or typ == "EmptySD3LatentImage" or typ == "EmptyFlux2LatentImage":
        if slot == 0:
            return String("LATENT")
    elif typ == "VAEEncode" or typ == "VAEEncodeForInpaint" or typ == "RepeatLatentBatch":
        if slot == 0:
            return String("LATENT")
    elif typ == "SetLatentNoiseMask":
        if slot == 0:
            return String("LATENT")
    elif typ == "ImageScale" or typ == "ImageScaleToTotalPixels" or typ == "ImageScaleBy":
        if slot == 0:
            return String("IMAGE")
    elif typ == "ImageResizeKJ":
        # ImageResizeKJ (KJ): outputs (IMAGE, width, height).
        if slot == 0:
            return String("IMAGE")
        if slot == 1:
            return String("width")
        if slot == 2:
            return String("height")
    elif typ == "ImagePadForOutpaint":
        if slot == 0:
            return String("IMAGE")
        if slot == 1:
            return String("MASK")
    elif typ == "ThresholdMask":
        if slot == 0:
            return String("MASK")
    elif typ == "InpaintModelConditioning":
        if slot == 0:
            return String("positive")
        if slot == 1:
            return String("negative")
        if slot == 2:
            return String("LATENT")
    elif typ == "GetImageSize":
        if slot == 0:
            return String("width")
        if slot == 1:
            return String("height")
        if slot == 2:
            return String("batch_size")
    elif typ == "GetImageSizeAndCount":
        # GetImageSizeAndCount (KJ): outputs (image, width, height, count). The
        # leading IMAGE passthrough shifts width/height to slots 1/2 and adds a
        # 4th `count` slot (constant 1 in the single-image model).
        if slot == 0:
            return String("IMAGE")
        if slot == 1:
            return String("width")
        if slot == 2:
            return String("height")
        if slot == 3:
            return String("batch_size")
    elif typ == "ReferenceLatent":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "FluxGuidance":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "ModelSamplingAuraFlow" or typ == "ModelSamplingSD3" or typ == "DifferentialDiffusion":
        if slot == 0:
            return String("MODEL")
    elif typ == "KSampler" or typ == "KSamplerAdvanced" or typ == "LanPaint_KSampler" or typ == "LanPaint_KSamplerAdvanced":
        if slot == 0:
            return String("LATENT")
    elif typ == "CFGGuider":
        if slot == 0:
            return String("GUIDER")
    elif typ == "BasicGuider":
        if slot == 0:
            return String("GUIDER")
    elif typ == "Flux2Scheduler" or typ == "BasicScheduler":
        if slot == 0:
            return String("SIGMAS")
    elif _workflow_is_named_scheduler_node(typ):
        if slot == 0:
            return String("SIGMAS")
    elif typ == "RandomNoise":
        if slot == 0:
            return String("NOISE")
    elif typ == "KSamplerSelect":
        if slot == 0:
            return String("SAMPLER")
    elif _workflow_is_named_sampler_node(typ):
        if slot == 0:
            return String("SAMPLER")
    elif typ == "SamplerCustomAdvanced" or typ == "LanPaint_SamplerCustomAdvanced":
        if slot == 0 or slot == 1:
            return String("LATENT")
    elif typ == "SamplerCustom":
        if slot == 0 or slot == 1:
            return String("LATENT")
    elif typ == "Reroute":
        if slot == 0:
            return String("REROUTE")
    elif typ == "SetNode":
        if slot == 0:
            return String("SET")
    elif typ == "GetNode":
        if slot == 0:
            return String("GET")
    elif typ == "ComfySwitchNode":
        if slot == 0:
            return String("output")
    elif typ == "LanPaint_MaskBlend":
        if slot == 0:
            return String("IMAGE")
    elif typ == "VAEDecode":
        if slot == 0:
            return String("IMAGE")
    elif typ == "SaveImage":
        raise Error("[501] Comfy API prompt SaveImage has no supported output slot")
    raise Error("[501] unsupported Comfy API prompt output node type: " + typ)


def _comfy_api_prompt_to_typed_graph(graph: JSONValue) raises -> JSONValue:
    var out = JSONValue.new_object()
    var nodes = JSONValue.new_array()
    var edges = JSONValue.new_array()
    var keys = graph.keys()
    for i in range(len(keys)):
        var key = keys[i]
        var node_id: Int
        try:
            node_id = Int(key)
        except:
            raise Error("[501] Comfy API prompt node id must be an integer")
        var src = graph[key]
        if not src.is_object():
            raise Error("[501] Comfy API prompt node must be an object")
        var typ = _workflow_canonical_type_id(_workflow_string(src, String("class_type")))
        if typ == "":
            raise Error("[501] Comfy API prompt node missing class_type")
        if not src.contains("inputs") or not src["inputs"].is_object():
            raise Error("[501] Comfy API prompt node missing inputs object")
        var inputs = src["inputs"]
        var fields = JSONValue.new_object()
        var input_keys = inputs.keys()
        for j in range(len(input_keys)):
            var input_name = input_keys[j]
            var input_value = inputs[input_name]
            if _comfy_api_input_is_link(input_value):
                var source_id = _comfy_api_link_node_id(input_value)
                var source_slot = _comfy_api_link_output_slot(input_value)
                var edge = JSONValue.new_object()
                var from_ref = JSONValue.new_object()
                var to_ref = JSONValue.new_object()
                from_ref.set("node", JSONValue.from_int(source_id))
                from_ref.set("port", JSONValue.from_string(_comfy_api_output_port(graph, source_id, source_slot)))
                to_ref.set("node", JSONValue.from_int(node_id))
                var to_port = input_name.copy()
                if typ == "Reroute" and to_port == "":
                    to_port = String("input")
                to_ref.set("port", JSONValue.from_string(to_port))
                edge.set("from", from_ref.copy())
                edge.set("to", to_ref.copy())
                edges.append(edge.copy())
            else:
                fields.set(input_name, input_value.copy())
        var dst = JSONValue.new_object()
        dst.set("id", JSONValue.from_int(node_id))
        dst.set("type_id", JSONValue.from_string(typ))
        dst.set("fields", fields.copy())
        nodes.append(dst.copy())
    out.set("nodes", nodes.copy())
    out.set("edges", edges.copy())
    return out^


def apply_comfy_api_prompt_graph(mut obj: JSONValue, wf: JSONValue) raises:
    var graph = _comfy_api_prompt_body(wf)
    if not looks_like_comfy_api_prompt_graph(graph):
        raise Error("[501] unsupported Comfy API prompt graph")
    if obj.contains("prompt") and obj["prompt"].is_object():
        obj.set("prompt", JSONValue.null())
    var typed = _comfy_api_prompt_to_typed_graph(graph)
    apply_typed_workflow_graph(obj, typed)
    _record_workflow_execution(
        obj, String("comfy_api_prompt_graph"), typed["nodes"].length(), typed["edges"].length(),
    )


def apply_comfy_ui_canvas_graph(mut obj: JSONValue, wf: JSONValue) raises:
    if not looks_like_comfy_ui_canvas_graph(wf):
        raise Error("[501] unsupported Comfy UI canvas graph")
    var typed = comfy_ui_canvas_to_typed_graph(wf)
    apply_typed_workflow_graph(obj, typed)
    _record_workflow_execution(
        obj, String("comfy_ui_canvas_graph"), typed["nodes"].length(), typed["edges"].length(),
    )


def apply_typed_workflow_graph(mut obj: JSONValue, wf: JSONValue) raises:
    """Typed topological executor for the supported Comfy/Swarm t2i subset."""
    if not wf.contains("nodes") or not wf["nodes"].is_array():
        raise Error("[501] workflow graph body needs nodes or params/genparams")
    if not wf.contains("edges") or not wf["edges"].is_array():
        raise Error("[501] workflow graph body needs edges for typed execution")
    var nodes_json = wf["nodes"]
    var edges = wf["edges"]
    var ids = List[Int]()
    for i in range(nodes_json.length()):
        var node = nodes_json[i]
        if not node.is_object():
            raise Error("[501] workflow graph node must be an object")
        var id = _workflow_id(node)
        if _workflow_has_node_id(ids, id):
            raise Error("[501] workflow graph duplicate node id: " + String(id))
        var type_id = _workflow_type_id(node)
        if type_id == "":
            raise Error("[501] unsupported workflow graph format: missing type_id")
        if not (
            type_id == "CheckpointLoaderSimple"
            or type_id == "UNETLoader"
            or type_id == "DiffusionModelLoader"
            or type_id == "LoraLoader"
            or type_id == "LoraLoaderModelOnly"
            or type_id == "ZImageLoraModelOnly"
            or type_id == "CLIPLoader"
            or type_id == "DualCLIPLoader"
            or type_id == "TripleCLIPLoader"
            or type_id == "VAELoader"
            or type_id == "CLIPTextEncode"
            or type_id == "CLIPTextEncodeFlux"
            or type_id == "TextEncodeQwenImageEdit"
            or type_id == "TextEncodeQwenImageEditPlus"
            or type_id == "ConditioningZeroOut"
            or type_id == "ConditioningSetMask"
            or type_id == "ConditioningConcat"
            or type_id == "ConditioningCombine"
            or type_id == "ConditioningAverage"
            or type_id == "Reroute"
            or type_id == "SetNode"
            or type_id == "GetNode"
            or type_id == "LoadImage"
            or type_id == "LoadImageOutput"
            or type_id == "LoadImageMask"
            or type_id == "ImageToMask"
            or type_id == "MaskToImage"
            or type_id == "EmptyLatentImage"
            or type_id == "EmptySD3LatentImage"
            or type_id == "EmptyFlux2LatentImage"
            or type_id == "VAEEncode"
            or type_id == "VAEEncodeForInpaint"
            or type_id == "RepeatLatentBatch"
            or type_id == "SetLatentNoiseMask"
            or type_id == "GetImageSize"
            or type_id == "GetImageSizeAndCount"
            or type_id == "ImageScale"
            or type_id == "ImageScaleToTotalPixels"
            or type_id == "ImageScaleBy"
            or type_id == "ImageResizeKJ"
            or type_id == "ImagePadForOutpaint"
            or type_id == "ThresholdMask"
            or type_id == "InpaintModelConditioning"
            or type_id == "ReferenceLatent"
            or type_id == "6007e698-2ebd-4917-84d8-299b35d7b7ab"
            or type_id == "f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca"
            or type_id == "ModelSamplingAuraFlow"
            or type_id == "ModelSamplingSD3"
            or type_id == "DifferentialDiffusion"
            or type_id == "KSampler"
            or type_id == "KSamplerAdvanced"
            or type_id == "LanPaint_KSampler"
            or type_id == "LanPaint_KSamplerAdvanced"
            or type_id == "CFGGuider"
            or type_id == "BasicGuider"
            or type_id == "FluxGuidance"
            or type_id == "Flux2Scheduler"
            or type_id == "BasicScheduler"
            or type_id == "RandomNoise"
            or type_id == "KSamplerSelect"
            or type_id == "SamplerEuler"
            or type_id == "SamplerEulerAncestral"
            or type_id == "SamplerEulerAncestralCFGPP"
            or type_id == "SamplerDPMPP_2M_SDE"
            or type_id == "SamplerDPMPP_3M_SDE"
            or type_id == "SamplerDPMPP_SDE"
            or type_id == "SamplerDPMPP_2S_Ancestral"
            or type_id == "SamplerDPMAdaptative"
            or type_id == "SamplerLMS"
            or type_id == "SamplerER_SDE"
            or type_id == "SamplerSASolver"
            or type_id == "SamplerSEEDS2"
            or type_id == "KarrasScheduler"
            or type_id == "ExponentialScheduler"
            or type_id == "PolyexponentialScheduler"
            or type_id == "SDTurboScheduler"
            or type_id == "VPScheduler"
            or type_id == "BetaSamplingScheduler"
            or type_id == "LaplaceScheduler"
            or type_id == "SamplerCustom"
            or type_id == "SamplerCustomAdvanced"
            or type_id == "LanPaint_SamplerCustomAdvanced"
            or type_id == "LanPaint_MaskBlend"
            or type_id == "ComfySwitchNode"
            or type_id == "VAEDecode"
            or type_id == "SaveImage"
            or type_id == "PreviewImage"
            or type_id == "MarkdownNote"
            or type_id == "Note"
            or type_id == "PrimitiveInt"
            or type_id == "PrimitiveFloat"
            or type_id == "PrimitiveString"
            or type_id == "PrimitiveStringMultiline"
            or type_id == "PrimitiveBoolean"
            or type_id == "PrimitiveNode"
            or type_id == "INTConstant"
            or type_id == "FloatConstant"
            or type_id == "StringConstant"
            or type_id == "StringConstantMultiline"
            or type_id == "BOOLConstant"
            or type_id == "SeedNode"
            or type_id == "easy int"
            or type_id == "easy float"
            or type_id == "easy string"
        ):
            raise Error(String("[501] unsupported workflow graph node type: ") + type_id)
        ids.append(id)

    _workflow_reject_multi_output_topology(nodes_json)

    var setnode_names = List[String]()
    for i in range(nodes_json.length()):
        var node = nodes_json[i]
        var type_id = _workflow_type_id(node)
        var fields = JSONValue.new_object()
        if node.contains("fields") and node["fields"].is_object():
            fields = node["fields"]
        if type_id == "SetNode":
            var name = _workflow_setget_name(fields)
            if name == "":
                raise Error("[501] workflow graph SetNode missing name")
            if _workflow_string_index(setnode_names, name) >= 0:
                raise Error("[501] workflow graph duplicate SetNode name: " + name)
            setnode_names.append(name)

    for i in range(nodes_json.length()):
        var node = nodes_json[i]
        var type_id = _workflow_type_id(node)
        if type_id == "GetNode":
            var fields = JSONValue.new_object()
            if node.contains("fields") and node["fields"].is_object():
                fields = node["fields"]
            var name = _workflow_setget_name(fields)
            if name == "":
                raise Error("[501] workflow graph GetNode missing name")
            if _workflow_string_index(setnode_names, name) < 0:
                raise Error("[501] workflow graph GetNode missing SetNode: " + name)

    var done = List[Bool]()
    for _ in range(nodes_json.length()):
        done.append(False)

    var value_nodes = List[Int]()
    var value_ports = List[String]()
    var value_types = List[String]()
    var model_nodes = List[Int]()
    var model_ports = List[String]()
    var model_names = List[String]()
    var cond_nodes = List[Int]()
    var cond_ports = List[String]()
    var cond_texts = List[String]()
    var image_nodes = List[Int]()
    var image_ports = List[String]()
    var image_paths = List[String]()
    var image_mask_sources = List[String]()
    var mask_nodes = List[Int]()
    var mask_ports = List[String]()
    var mask_paths = List[String]()
    var mask_sources = List[String]()
    var latent_nodes = List[Int]()
    var latent_ports = List[String]()
    var latent_widths = List[Int]()
    var latent_heights = List[Int]()
    var latent_images = List[Int]()
    var latent_init_images = List[String]()
    var latent_mask_images = List[String]()
    var noise_nodes = List[Int]()
    var noise_ports = List[String]()
    var noise_seeds = List[Int]()
    var sampler_nodes = List[Int]()
    var sampler_ports = List[String]()
    var sampler_names = List[String]()
    var sigmas_nodes = List[Int]()
    var sigmas_ports = List[String]()
    var sigmas_steps = List[Int]()
    var sigmas_schedulers = List[String]()
    var sigmas_denoises = List[Float64]()
    var scalar_nodes = List[Int]()
    var scalar_ports = List[String]()
    var scalar_types = List[String]()
    var scalar_ints = List[Int]()
    var scalar_floats = List[Float64]()
    var scalar_strings = List[String]()
    var scalar_bools = List[Bool]()
    var setget_names = List[String]()
    var setget_nodes = List[Int]()
    var setget_ports = List[String]()
    var setget_types = List[String]()
    var reference_latent_count = 0

    var remaining = nodes_json.length()
    var saw_prompt = False
    while remaining > 0:
        var progressed = False
        for i in range(nodes_json.length()):
            if done[i]:
                continue
            var node = nodes_json[i]
            var node_id = _workflow_id(node)
            var type_id = _workflow_type_id(node)
            var fields = JSONValue.new_object()
            if node.contains("fields") and node["fields"].is_object():
                fields = node["fields"]

            if type_id == "CheckpointLoaderSimple":
                var model_name = _workflow_loader_model_name(fields)
                if model_name != "":
                    _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                else:
                    model_name = String("")
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MODEL"), String("MODEL"))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CLIP"), String("CLIP"))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("VAE"), String("VAE"))
                model_nodes.append(node_id); model_ports.append(String("MODEL")); model_names.append(model_name)
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "UNETLoader" or type_id == "DiffusionModelLoader":
                var model_name = _workflow_loader_model_name(fields)
                if model_name != "":
                    _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MODEL"), String("MODEL"))
                model_nodes.append(node_id); model_ports.append(String("MODEL")); model_names.append(model_name)
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "LoraLoader" or type_id == "LoraLoaderModelOnly" or type_id == "ZImageLoraModelOnly":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                if not model_link.found:
                    if type_id == "LoraLoaderModelOnly":
                        raise Error("[501] workflow graph LoraLoaderModelOnly missing model input")
                    if type_id == "ZImageLoraModelOnly":
                        raise Error("[501] workflow graph ZImageLoraModelOnly missing model input")
                    raise Error("[501] workflow graph " + type_id + " missing model input")
                var clip_link = _workflow_find_input_link(edges, node_id, String("clip"))
                if type_id == "LoraLoader" and not clip_link.found:
                    raise Error("[501] workflow graph LoraLoader missing clip input")
                var ready = _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                if type_id == "LoraLoader":
                    ready = ready and _workflow_value_index(value_nodes, value_ports, clip_link.node_id, clip_link.port) >= 0
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    if type_id == "LoraLoader":
                        _workflow_require_value_type(value_nodes, value_ports, value_types, clip_link, String("CLIP"), String("clip"))
                    var model_name = _workflow_model_name(model_nodes, model_ports, model_names, model_link)
                    if model_name != "":
                        _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                    var lora_name = _workflow_string(fields, String("lora_name"))
                    var strength = _workflow_float(fields, String("strength_model"), 1.0, -10.0, 10.0)
                    var strength_clip = 0.0
                    if type_id == "LoraLoader":
                        strength_clip = _workflow_float(fields, String("strength_clip"), 1.0, -10.0, 10.0)
                    _workflow_append_lora(obj, lora_name, strength)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MODEL"), String("MODEL"))
                    model_nodes.append(node_id); model_ports.append(String("MODEL")); model_names.append(model_name)
                    if type_id == "LoraLoader":
                        if strength_clip == 0.0:
                            _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CLIP"), String("CLIP"))
                        else:
                            _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CLIP"), String("CLIP_LORA_UNSUPPORTED"))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "CLIPLoader" or type_id == "DualCLIPLoader" or type_id == "TripleCLIPLoader":
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CLIP"), String("CLIP"))
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "VAELoader":
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("VAE"), String("VAE"))
                done[i] = True; remaining -= 1; progressed = True
            elif _workflow_is_scalar_node(type_id):
                var scalar_type = _workflow_scalar_output_type(type_id, fields)
                if scalar_type == "":
                    raise Error("[501] workflow graph primitive scalar missing supported output type")
                var int_value = 0
                var float_value = 0.0
                var string_value = String("")
                var bool_value = False
                if scalar_type == "INT":
                    if fields.contains("value"):
                        int_value = _workflow_int(fields, String("value"), 0, -9223372036854775807, 9223372036854775807)
                    elif fields.contains("seed"):
                        int_value = _workflow_int(fields, String("seed"), 0, -9223372036854775807, 9223372036854775807)
                    else:
                        int_value = 0
                    float_value = Float64(int_value)
                elif scalar_type == "FLOAT":
                    if fields.contains("value"):
                        float_value = _workflow_float(fields, String("value"), 0.0, -1.0e308, 1.0e308)
                    else:
                        float_value = 0.0
                    if type_id == "FloatConstant":
                        float_value = _workflow_round6(float_value)
                    int_value = Int(float_value)
                elif scalar_type == "STRING":
                    string_value = _workflow_string(fields, String("value"))
                    if string_value == "":
                        string_value = _workflow_string(fields, String("text"))
                    if string_value == "":
                        string_value = _workflow_string(fields, String("string"))
                    if type_id == "StringConstantMultiline" and _workflow_bool(fields, String("strip_newlines"), True):
                        if (
                            string_value.find(String("\n")) >= 0
                            or string_value.find(String("\r")) >= 0
                            or string_value.startswith(String(" "))
                            or string_value.startswith(String("\t"))
                            or string_value.endswith(String(" "))
                            or string_value.endswith(String("\t"))
                        ):
                            raise Error("[501] workflow graph StringConstantMultiline strip_newlines transform is unsupported")
                elif scalar_type == "BOOLEAN":
                    bool_value = _workflow_bool(fields, String("value"), False)
                    int_value = 1 if bool_value else 0
                    float_value = Float64(int_value)
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, scalar_type.copy(), scalar_type.copy())
                _workflow_add_scalar(
                    scalar_nodes, scalar_ports, scalar_types,
                    scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                    node_id, scalar_type, scalar_type,
                    int_value, float_value, string_value, bool_value,
                )
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "EmptyLatentImage" or type_id == "EmptySD3LatentImage" or type_id == "EmptyFlux2LatentImage":
                var width_link = _workflow_find_input_link(edges, node_id, String("width"))
                var height_link = _workflow_find_input_link(edges, node_id, String("height"))
                var batch_link = _workflow_find_input_link(edges, node_id, String("batch_size"))
                var ready = (
                    _workflow_optional_link_ready(value_nodes, value_ports, width_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, height_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, batch_link)
                )
                if ready:
                    var width = _opt_int(fields, "width", 512, 16, 2048)
                    var height = _opt_int(fields, "height", 512, 16, 2048)
                    var images = _opt_int(fields, "batch_size", 1, 1, 64)
                    if width_link.found:
                        width = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            width_link, String("width"),
                        )
                    if height_link.found:
                        height = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            height_link, String("height"),
                        )
                    if batch_link.found:
                        images = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            batch_link, String("batch_size"),
                        )
                    if width < 16 or width > 2048 or height < 16 or height > 2048:
                        raise Error("[501] workflow graph EmptyLatentImage scalar dimensions out of range")
                    if images < 1 or images > 64:
                        raise Error("[501] workflow graph EmptyLatentImage scalar batch_size out of range")
                    if images != 1:
                        raise Error(
                            "[501] workflow graph EmptyLatentImage batch_size>1 "
                            + "requires real Comfy latent-batch execution; use flat images=N "
                            + "for serial product fanout"
                        )
                    _set_if_missing(obj, String("width"), JSONValue.from_int(width))
                    _set_if_missing(obj, String("height"), JSONValue.from_int(height))
                    _set_if_missing(obj, String("images"), JSONValue.from_int(1))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    latent_widths.append(width); latent_heights.append(height); latent_images.append(1)
                    latent_init_images.append(String(""))
                    latent_mask_images.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "CLIPTextEncode" or type_id == "CLIPTextEncodeFlux":
                var clip_link = _workflow_find_input_link(edges, node_id, String("clip"))
                var text_link = _workflow_find_input_link(edges, node_id, String("text"))
                if clip_link.found:
                    var ready = (
                        _workflow_value_index(value_nodes, value_ports, clip_link.node_id, clip_link.port) >= 0
                        and _workflow_optional_link_ready(value_nodes, value_ports, text_link)
                    )
                    if ready:
                        _workflow_require_value_type(value_nodes, value_ports, value_types, clip_link, String("CLIP"), String("clip"))
                        var text = _workflow_conditioning_prompt_text(fields, type_id)
                        if text_link.found:
                            text = _workflow_scalar_string(
                                scalar_nodes, scalar_ports, scalar_types, scalar_strings,
                                text_link, String("text"),
                            )
                        _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                        cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                        done[i] = True; remaining -= 1; progressed = True
                else:
                    raise Error("[501] workflow graph " + type_id + " missing clip input")
            elif type_id == "TextEncodeQwenImageEdit" or type_id == "TextEncodeQwenImageEditPlus":
                var clip_link = _workflow_find_input_link(edges, node_id, String("clip"))
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                var text_link = _workflow_find_input_link(edges, node_id, String("text"))
                if not clip_link.found or not image_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, clip_link.node_id, clip_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, text_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, clip_link, String("CLIP"), String("clip"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    var text = _workflow_conditioning_prompt_text(fields, type_id)
                    if text_link.found:
                        text = _workflow_scalar_string(
                            scalar_nodes, scalar_ports, scalar_types, scalar_strings,
                            text_link, String("text"),
                        )
                    var edit_image = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    _set_if_missing(obj, String("qwen_edit_conditioning_image"), JSONValue.from_string(edit_image))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ConditioningZeroOut":
                var cond_link = _workflow_find_input_link(edges, node_id, String("conditioning"))
                if not cond_link.found:
                    raise Error("[501] workflow graph ConditioningZeroOut missing conditioning input")
                var idx = _workflow_value_index(value_nodes, value_ports, cond_link.node_id, cond_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, cond_link, String("CONDITIONING"), String("conditioning"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ConditioningSetMask":
                var cond_link = _workflow_find_input_link(edges, node_id, String("conditioning"))
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not cond_link.found or not mask_link.found:
                    raise Error("[501] workflow graph ConditioningSetMask missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, cond_link.node_id, cond_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, cond_link, String("CONDITIONING"), String("conditioning"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, cond_link)
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    var strength = _workflow_float(fields, String("strength"), 1.0, 0.0, 10.0)
                    var set_cond_area = String(_workflow_string(fields, String("set_cond_area")).lower())
                    var set_area_to_bounds = False
                    if set_cond_area != "" and set_cond_area != "default":
                        set_area_to_bounds = True
                    _set_if_missing(obj, String("conditioning_mask_image"), JSONValue.from_string(mask_path))
                    _set_if_missing(obj, String("conditioning_mask_channel"), JSONValue.from_string(mask_source))
                    _set_if_missing(obj, String("conditioning_mask_strength"), JSONValue.from_float(strength))
                    _set_if_missing(obj, String("conditioning_mask_set_area_to_bounds"), JSONValue.from_bool(set_area_to_bounds))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ConditioningConcat":
                # Comfy ConditioningConcat(conditioning_to, conditioning_from): the two
                # tensors are concatenated along the token axis. In this text-only
                # conditioning model a CONDITIONING handle carries prompt text, so the
                # faithful flat lowering is to join the two prompts: "to, from".
                var to_link = _workflow_find_input_link(edges, node_id, String("conditioning_to"))
                var from_link = _workflow_find_input_link(edges, node_id, String("conditioning_from"))
                if not to_link.found or not from_link.found:
                    raise Error("[501] workflow graph ConditioningConcat missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, to_link.node_id, to_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, from_link.node_id, from_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, to_link, String("CONDITIONING"), String("conditioning_to"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, from_link, String("CONDITIONING"), String("conditioning_from"))
                    var to_text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, to_link)
                    var from_text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, from_link)
                    var text: String
                    if to_text == "":
                        text = from_text^
                    elif from_text == "":
                        text = to_text^
                    else:
                        text = to_text + String(", ") + from_text
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ConditioningCombine" or type_id == "ConditioningAverage":
                # Comfy ConditioningCombine batches both conditionings; ConditioningAverage
                # blends them by `conditioning_to_strength`. Neither is representable in a
                # single text prompt WITHOUT silently dropping the second conditioning (and
                # the blend weight) — which would render a subtly wrong image. Fail loud
                # instead; use ConditioningConcat to JOIN two prompts into one.
                raise Error(
                    "[501] workflow graph " + type_id + " cannot be lowered to a single"
                    " text prompt (it would silently drop the second conditioning / blend"
                    " weight); use ConditioningConcat to join prompts"
                )
            elif type_id == "FluxGuidance":
                var cond_link = _workflow_find_input_link(edges, node_id, String("conditioning"))
                if not cond_link.found:
                    raise Error("[501] workflow graph FluxGuidance missing conditioning input")
                var idx = _workflow_value_index(value_nodes, value_ports, cond_link.node_id, cond_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, cond_link, String("CONDITIONING"), String("conditioning"))
                    var text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, cond_link)
                    _copy_field_if_missing(obj, fields, String("cfg"), String("cfg"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "SetNode":
                var set_link = _workflow_find_setnode_input_link(edges, node_id)
                if not set_link.found:
                    raise Error("[501] workflow graph SetNode missing input")
                var value_idx = _workflow_value_index(value_nodes, value_ports, set_link.node_id, set_link.port)
                if value_idx >= 0:
                    var name = _workflow_setget_name(fields)
                    var actual = value_types[value_idx].copy()
                    if not _workflow_setget_supported_type(actual):
                        raise Error("[501] workflow graph SetNode unsupported bus type: " + actual)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SET"), actual.copy())
                    _workflow_copy_value_metadata(
                        set_link, node_id, String("SET"), actual,
                        model_nodes, model_ports, model_names,
                        cond_nodes, cond_ports, cond_texts,
                        image_nodes, image_ports, image_paths, image_mask_sources,
                        mask_nodes, mask_ports, mask_paths, mask_sources,
                        latent_nodes, latent_ports, latent_widths, latent_heights, latent_images,
                        latent_init_images, latent_mask_images,
                        noise_nodes, noise_ports, noise_seeds,
                        sampler_nodes, sampler_ports, sampler_names,
                        sigmas_nodes, sigmas_ports, sigmas_steps, sigmas_schedulers, sigmas_denoises,
                        scalar_nodes, scalar_ports, scalar_types,
                        scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                    )
                    setget_names.append(name)
                    setget_nodes.append(node_id)
                    setget_ports.append(String("SET"))
                    setget_types.append(actual.copy())
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "GetNode":
                var name = _workflow_setget_name(fields)
                var bus_idx = _workflow_string_index(setget_names, name)
                if bus_idx >= 0:
                    var actual = setget_types[bus_idx].copy()
                    var declared = _workflow_string(fields, String("output_type"))
                    if not _workflow_type_accepts(declared, actual):
                        raise Error("[501] workflow graph GetNode output type mismatch: " + declared + " vs " + actual)
                    var source = WorkflowLink(True, setget_nodes[bus_idx], setget_ports[bus_idx].copy())
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("GET"), actual.copy())
                    _workflow_copy_value_metadata(
                        source, node_id, String("GET"), actual,
                        model_nodes, model_ports, model_names,
                        cond_nodes, cond_ports, cond_texts,
                        image_nodes, image_ports, image_paths, image_mask_sources,
                        mask_nodes, mask_ports, mask_paths, mask_sources,
                        latent_nodes, latent_ports, latent_widths, latent_heights, latent_images,
                        latent_init_images, latent_mask_images,
                        noise_nodes, noise_ports, noise_seeds,
                        sampler_nodes, sampler_ports, sampler_names,
                        sigmas_nodes, sigmas_ports, sigmas_steps, sigmas_schedulers, sigmas_denoises,
                        scalar_nodes, scalar_ports, scalar_types,
                        scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                    )
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "Reroute":
                var input_link = _workflow_find_reroute_input_link(edges, node_id)
                if not input_link.found:
                    raise Error("[501] workflow graph Reroute missing input")
                var value_idx = _workflow_value_index(value_nodes, value_ports, input_link.node_id, input_link.port)
                if value_idx >= 0:
                    var actual = value_types[value_idx].copy()
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("REROUTE"), actual.copy())
                    _workflow_copy_value_metadata(
                        input_link, node_id, String("REROUTE"), actual,
                        model_nodes, model_ports, model_names,
                        cond_nodes, cond_ports, cond_texts,
                        image_nodes, image_ports, image_paths, image_mask_sources,
                        mask_nodes, mask_ports, mask_paths, mask_sources,
                        latent_nodes, latent_ports, latent_widths, latent_heights, latent_images,
                        latent_init_images, latent_mask_images,
                        noise_nodes, noise_ports, noise_seeds,
                        sampler_nodes, sampler_ports, sampler_names,
                        sigmas_nodes, sigmas_ports, sigmas_steps, sigmas_schedulers, sigmas_denoises,
                        scalar_nodes, scalar_ports, scalar_types,
                        scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                    )
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ComfySwitchNode":
                var false_link = _workflow_find_input_link(edges, node_id, String("on_false"))
                var true_link = _workflow_find_input_link(edges, node_id, String("on_true"))
                var switch_link = _workflow_find_input_link(edges, node_id, String("switch"))
                if not false_link.found or not true_link.found:
                    raise Error("[501] workflow graph ComfySwitchNode missing required typed input")
                var ready = _workflow_optional_link_ready(value_nodes, value_ports, switch_link)
                var switch_value = _workflow_bool(fields, String("switch"), False)
                if ready:
                    if switch_link.found:
                        switch_value = _workflow_scalar_bool(
                            scalar_nodes, scalar_ports, scalar_types, scalar_bools,
                            switch_link, String("switch"),
                        )
                    var selected = false_link.copy()
                    if switch_value:
                        selected = true_link.copy()
                    var selected_idx = _workflow_value_index(value_nodes, value_ports, selected.node_id, selected.port)
                    if selected_idx >= 0:
                        var actual = value_types[selected_idx].copy()
                        _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("output"), actual.copy())
                        _workflow_copy_value_metadata(
                            selected, node_id, String("output"), actual,
                            model_nodes, model_ports, model_names,
                            cond_nodes, cond_ports, cond_texts,
                            image_nodes, image_ports, image_paths, image_mask_sources,
                            mask_nodes, mask_ports, mask_paths, mask_sources,
                            latent_nodes, latent_ports, latent_widths, latent_heights, latent_images,
                            latent_init_images, latent_mask_images,
                            noise_nodes, noise_ports, noise_seeds,
                            sampler_nodes, sampler_ports, sampler_names,
                            sigmas_nodes, sigmas_ports, sigmas_steps, sigmas_schedulers, sigmas_denoises,
                            scalar_nodes, scalar_ports, scalar_types,
                            scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                        )
                        done[i] = True; remaining -= 1; progressed = True
            elif type_id == "LoadImage" or type_id == "LoadImageOutput" or type_id == "LoadImageMask":
                # LoadImageOutput / LoadImageMask are aliases of LoadImage: they
                # load an image file and expose both IMAGE and MASK outputs that
                # resolve to init_image / mask_image downstream.
                var image_path = _workflow_string(fields, String("image"))
                if image_path == "":
                    image_path = _workflow_string(fields, String("path"))
                if image_path == "":
                    raise Error("[501] workflow graph " + type_id + " missing image path")
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                image_mask_sources.append(String(""))
                mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(image_path)
                mask_sources.append(String("load_image_mask"))
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ImageToMask":
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph ImageToMask missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    var requested_channel = _workflow_imagetomask_channel(fields)
                    var mask_source = _workflow_source_meta(image_nodes, image_ports, image_mask_sources, image_link)
                    if mask_source == "":
                        mask_source = requested_channel
                    _set_if_missing(obj, String("lanpaint_mask_channel"), JSONValue.from_string(mask_source))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                    mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(image_path)
                    mask_sources.append(mask_source)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "MaskToImage":
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not mask_link.found:
                    raise Error("[501] workflow graph MaskToImage missing mask input")
                var idx = _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(mask_path)
                    image_mask_sources.append(mask_source)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ThresholdMask":
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not mask_link.found:
                    raise Error("[501] workflow graph ThresholdMask missing mask input")
                var idx = _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    var threshold = _workflow_float(fields, String("value"), 0.5, 0.0, 1.0)
                    _set_if_missing(obj, String("threshold_mask_value"), JSONValue.from_float(threshold))
                    _set_if_missing(obj, String("threshold_mask_operator"), JSONValue.from_string(String("gt")))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                    mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(mask_path)
                    mask_sources.append(mask_source)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "GetImageSize" or type_id == "GetImageSizeAndCount":
                # GetImageSizeAndCount (KJ) is GetImageSize plus a leading IMAGE
                # passthrough output and a `count` (batch) output. In the flat
                # single-image model the width/height are not resolvable (LoadImage
                # carries no dims), so the INT outputs are placeholders; the count
                # is the constant single-image batch of 1.
                var and_count = type_id == "GetImageSizeAndCount"
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    if and_count:
                        # Slot 0 is the unmodified IMAGE passthrough.
                        var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                        var mask_source = _workflow_source_meta(image_nodes, image_ports, image_mask_sources, image_link)
                        _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                        image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                        image_mask_sources.append(mask_source)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("width"), String("INT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("height"), String("INT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("batch_size"), String("INT"))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ImageScaleBy":
                # ImageScaleBy multiplies the SOURCE image dims by `scale_by`
                # (out_w = round(src_w * scale_by)). In the flat single-image model
                # the source image dims are NOT resolvable (LoadImage carries only a
                # path, never dims), so the scaled output dims cannot be represented
                # on a worker-supported grid. Fail loud [501] — never silently emit
                # a wrong size. The scale_by widget is range-validated first so the
                # error is precise.
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph ImageScaleBy missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    var _scale_by = _workflow_float(fields, String("scale_by"), 1.0, 0.01, 8.0)
                    raise Error("[501] workflow graph ImageScaleBy scales the source image dims by scale_by, but the source image dimensions are not resolvable in the flat single-image model (LoadImage carries no dims); use ImageResizeKJ with explicit width/height, or an EmptyLatentImage, to set a worker-supported size")
            elif type_id == "ImageResizeKJ":
                # ImageResizeKJ resizes an image to EXPLICIT width/height widgets.
                # Unlike ImageScaleBy the target dims are knowable — but only when
                # the resize is a plain explicit resize: keep_proportion=false, both
                # width AND height nonzero, and no get_image_size IMAGE input (all
                # three make the output dims depend on the un-knowable source dims).
                # In the representable case we resolve the explicit width/height into
                # the flat params (range-validated like the ImageScale scalar path),
                # pass the IMAGE handle through, and emit width/height INT outputs
                # (slots 1/2). Otherwise fail loud [501].
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph ImageResizeKJ missing image input")
                var width_link = _workflow_find_input_link(edges, node_id, String("width"))
                var height_link = _workflow_find_input_link(edges, node_id, String("height"))
                var get_size_link = _workflow_find_input_link(edges, node_id, String("get_image_size"))
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, width_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, height_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, get_size_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    if get_size_link.found:
                        raise Error("[501] workflow graph ImageResizeKJ get_image_size input copies the source image dims, which are not resolvable in the flat single-image model")
                    if _workflow_bool(fields, String("keep_proportion"), False):
                        raise Error("[501] workflow graph ImageResizeKJ keep_proportion=true derives a dimension from the source image aspect, which is not resolvable in the flat single-image model; set explicit width and height instead")
                    var width = _opt_int(fields, "width", 512, 0, 8192)
                    var height = _opt_int(fields, "height", 512, 0, 8192)
                    if width_link.found:
                        width = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            width_link, String("width"),
                        )
                    if height_link.found:
                        height = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            height_link, String("height"),
                        )
                    if width == 0 or height == 0:
                        raise Error("[501] workflow graph ImageResizeKJ width/height of 0 keeps the source dimension, which is not resolvable in the flat single-image model; set explicit nonzero width and height")
                    if width < 1 or width > 8192 or height < 1 or height > 8192:
                        raise Error("[501] workflow graph ImageResizeKJ width/height out of range")
                    _set_if_missing(obj, String("width"), JSONValue.from_int(width))
                    _set_if_missing(obj, String("height"), JSONValue.from_int(height))
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    var mask_source = _workflow_source_meta(image_nodes, image_ports, image_mask_sources, image_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                    image_mask_sources.append(mask_source)
                    # Resolved width/height INT outputs (slots 1/2) carry concrete
                    # values (the explicit target dims), readable downstream.
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("width"), String("INT"))
                    _workflow_add_scalar(
                        scalar_nodes, scalar_ports, scalar_types,
                        scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                        node_id, String("width"), String("INT"),
                        width, Float64(width), String(""), False,
                    )
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("height"), String("INT"))
                    _workflow_add_scalar(
                        scalar_nodes, scalar_ports, scalar_types,
                        scalar_ints, scalar_floats, scalar_strings, scalar_bools,
                        node_id, String("height"), String("INT"),
                        height, Float64(height), String(""), False,
                    )
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ImageScale" or type_id == "ImageScaleToTotalPixels":
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing image input")
                var width_link = _workflow_find_input_link(edges, node_id, String("width"))
                var height_link = _workflow_find_input_link(edges, node_id, String("height"))
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, width_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, height_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    if width_link.found:
                        var width = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            width_link, String("width"),
                        )
                        if width < 1 or width > 8192:
                            raise Error("[501] workflow graph ImageScale scalar width out of range")
                    if height_link.found:
                        var height = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            height_link, String("height"),
                        )
                        if height < 1 or height > 8192:
                            raise Error("[501] workflow graph ImageScale scalar height out of range")
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    var mask_source = _workflow_source_meta(image_nodes, image_ports, image_mask_sources, image_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                    image_mask_sources.append(mask_source)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ImagePadForOutpaint":
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph ImagePadForOutpaint missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    var left = _opt_int(fields, "left", 0, 0, 4096)
                    var top = _opt_int(fields, "top", 0, 0, 4096)
                    var right = _opt_int(fields, "right", 0, 0, 4096)
                    var bottom = _opt_int(fields, "bottom", 0, 0, 4096)
                    var feathering = _opt_int(fields, "feathering", 40, 0, 4096)
                    _set_if_missing(obj, String("outpaint_left"), JSONValue.from_int(left))
                    _set_if_missing(obj, String("outpaint_top"), JSONValue.from_int(top))
                    _set_if_missing(obj, String("outpaint_right"), JSONValue.from_int(right))
                    _set_if_missing(obj, String("outpaint_bottom"), JSONValue.from_int(bottom))
                    _set_if_missing(obj, String("outpaint_feathering"), JSONValue.from_int(feathering))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                    image_mask_sources.append(String(""))
                    mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(image_path)
                    mask_sources.append(String("image_pad_for_outpaint"))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "VAEEncode":
                var pixels_link = _workflow_find_input_link(edges, node_id, String("pixels"))
                var vae_link = _workflow_find_input_link(edges, node_id, String("vae"))
                if not pixels_link.found or not vae_link.found:
                    raise Error("[501] workflow graph VAEEncode missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, pixels_link.node_id, pixels_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, vae_link.node_id, vae_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pixels_link, String("IMAGE"), String("pixels"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, vae_link, String("VAE"), String("vae"))
                    var init_path = _workflow_image_path(image_nodes, image_ports, image_paths, pixels_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                    latent_init_images.append(init_path)
                    latent_mask_images.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "VAEEncodeForInpaint":
                # Encode pixels to a LATENT + attach the inpaint mask. Same flat
                # effect as InpaintModelConditioning's mask half (aliases to
                # inpaint_* params); no conditioning here. grow_mask_by has no flat
                # representation and is read only to validate range.
                var pixels_link = _workflow_find_input_link(edges, node_id, String("pixels"))
                var vae_link = _workflow_find_input_link(edges, node_id, String("vae"))
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not pixels_link.found or not vae_link.found or not mask_link.found:
                    raise Error("[501] workflow graph VAEEncodeForInpaint missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, pixels_link.node_id, pixels_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, vae_link.node_id, vae_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pixels_link, String("IMAGE"), String("pixels"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, vae_link, String("VAE"), String("vae"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    # grow_mask_by: read only for range validation; no flat key carries it.
                    _ = _opt_int(fields, "grow_mask_by", 6, 0, 64)
                    var init_path = _workflow_image_path(image_nodes, image_ports, image_paths, pixels_link)
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    _set_if_missing(obj, String("init_image"), JSONValue.from_string(init_path))
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                    _set_if_missing(obj, String("lanpaint_mask_channel"), JSONValue.from_string(mask_source))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                    latent_init_images.append(init_path)
                    latent_mask_images.append(mask_path)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "RepeatLatentBatch":
                # RepeatLatentBatch mutates a Comfy latent tensor batch. The
                # daemon's flat `images=N` is serial fanout, not latent-batch
                # execution, so fail loud until a real batched latent path exists.
                var samples_link = _workflow_find_input_link(edges, node_id, String("samples"))
                var amount_link = _workflow_find_input_link(edges, node_id, String("amount"))
                if not samples_link.found:
                    raise Error("[501] workflow graph RepeatLatentBatch missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, samples_link.node_id, samples_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, amount_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, samples_link, String("LATENT"), String("samples"))
                    var amount = _opt_int(fields, "amount", 1, 1, 64)
                    if amount_link.found:
                        amount = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            amount_link, String("amount"),
                        )
                    if amount < 1 or amount > 64:
                        raise Error("[501] workflow graph RepeatLatentBatch scalar amount out of range")
                    raise Error(
                        "[501] workflow graph RepeatLatentBatch requires real Comfy "
                        + "latent-batch execution; use flat images=N for serial "
                        + "product fanout"
                    )
            elif type_id == "SetLatentNoiseMask":
                var samples_link = _workflow_find_input_link(edges, node_id, String("samples"))
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not samples_link.found or not mask_link.found:
                    raise Error("[501] workflow graph SetLatentNoiseMask missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, samples_link.node_id, samples_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, samples_link, String("LATENT"), String("samples"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var latent_idx = _workflow_latent_index(latent_nodes, latent_ports, samples_link)
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                    _set_if_missing(obj, String("lanpaint_mask_channel"), JSONValue.from_string(mask_source))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    if latent_idx >= 0:
                        latent_widths.append(latent_widths[latent_idx])
                        latent_heights.append(latent_heights[latent_idx])
                        latent_images.append(latent_images[latent_idx])
                        latent_init_images.append(latent_init_images[latent_idx])
                        latent_mask_images.append(mask_path)
                    else:
                        latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                        latent_init_images.append(String(""))
                        latent_mask_images.append(mask_path)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "InpaintModelConditioning":
                var pos_link = _workflow_find_input_link(edges, node_id, String("positive"))
                var neg_link = _workflow_find_input_link(edges, node_id, String("negative"))
                var vae_link = _workflow_find_input_link(edges, node_id, String("vae"))
                var pixels_link = _workflow_find_input_link(edges, node_id, String("pixels"))
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not pos_link.found or not neg_link.found or not vae_link.found or not pixels_link.found or not mask_link.found:
                    raise Error("[501] workflow graph InpaintModelConditioning missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, pos_link.node_id, pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, neg_link.node_id, neg_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, vae_link.node_id, vae_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, pixels_link.node_id, pixels_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pos_link, String("CONDITIONING"), String("positive"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, neg_link, String("CONDITIONING"), String("negative"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, vae_link, String("VAE"), String("vae"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pixels_link, String("IMAGE"), String("pixels"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var positive_text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, pos_link)
                    var negative_text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, neg_link)
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, pixels_link)
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    var noise_mask = _workflow_bool(fields, String("noise_mask"), True)
                    # Comfy also attaches concat conditioning metadata; real
                    # backends must implement that path before accepting it.
                    _set_if_missing(obj, String("init_image"), JSONValue.from_string(image_path))
                    _set_if_missing(obj, String("inpaint_conditioning_image"), JSONValue.from_string(image_path))
                    _set_if_missing(obj, String("inpaint_conditioning_mask"), JSONValue.from_string(mask_path))
                    _set_if_missing(obj, String("inpaint_conditioning_noise_mask"), JSONValue.from_bool(noise_mask))
                    if noise_mask:
                        _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                        _set_if_missing(obj, String("lanpaint_mask_channel"), JSONValue.from_string(mask_source))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("positive"), String("CONDITIONING"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("negative"), String("CONDITIONING"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    cond_nodes.append(node_id); cond_ports.append(String("positive")); cond_texts.append(positive_text)
                    cond_nodes.append(node_id); cond_ports.append(String("negative")); cond_texts.append(negative_text)
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                    latent_init_images.append(image_path)
                    if noise_mask:
                        latent_mask_images.append(mask_path)
                    else:
                        latent_mask_images.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca":
                var vae_link = _workflow_find_input_link(edges, node_id, String("vae"))
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not vae_link.found or not image_link.found or not mask_link.found:
                    raise Error("[501] workflow graph LanPaint preprocessing subgraph missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, vae_link.node_id, vae_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, vae_link, String("VAE"), String("vae"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var init_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    _set_if_missing(obj, String("init_image"), JSONValue.from_string(init_path))
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                    _set_if_missing(obj, String("lanpaint_mask_channel"), JSONValue.from_string(mask_source))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                    latent_init_images.append(init_path)
                    latent_mask_images.append(mask_path)
                    mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(mask_path)
                    mask_sources.append(mask_source)
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(init_path)
                    image_mask_sources.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ReferenceLatent":
                var cond_link = _workflow_find_input_link(edges, node_id, String("conditioning"))
                if not cond_link.found:
                    raise Error("[501] workflow graph ReferenceLatent missing conditioning input")
                var latent_link = _workflow_find_input_link(edges, node_id, String("latent"))
                var cond_ready = _workflow_value_index(value_nodes, value_ports, cond_link.node_id, cond_link.port) >= 0
                var latent_ready = True
                if latent_link.found:
                    latent_ready = _workflow_value_index(value_nodes, value_ports, latent_link.node_id, latent_link.port) >= 0
                if cond_ready and latent_ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, cond_link, String("CONDITIONING"), String("conditioning"))
                    var text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, cond_link)
                    if latent_link.found:
                        _workflow_require_value_type(value_nodes, value_ports, value_types, latent_link, String("LATENT"), String("latent"))
                        reference_latent_count += 1
                        _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("COND_LATENT"))
                        cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                        var latent_idx = _workflow_latent_index(latent_nodes, latent_ports, latent_link)
                        latent_nodes.append(node_id); latent_ports.append(String("CONDITIONING"))
                        if latent_idx >= 0:
                            if latent_init_images[latent_idx] != "":
                                _set_if_missing(obj, String("reference_image"), JSONValue.from_string(latent_init_images[latent_idx]))
                                _set_if_missing(obj, String("reference_latent_method"), JSONValue.from_string(String("index")))
                            latent_widths.append(latent_widths[latent_idx])
                            latent_heights.append(latent_heights[latent_idx])
                            latent_images.append(latent_images[latent_idx])
                            latent_init_images.append(latent_init_images[latent_idx])
                            latent_mask_images.append(latent_mask_images[latent_idx])
                        else:
                            latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                            latent_init_images.append(String(""))
                            latent_mask_images.append(String(""))
                    else:
                        _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                        cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "6007e698-2ebd-4917-84d8-299b35d7b7ab":
                var pos_link = _workflow_find_input_link(edges, node_id, String("conditioning"))
                var neg_link = _workflow_find_input_link(edges, node_id, String("conditioning_1"))
                var pixels_link = _workflow_find_input_link(edges, node_id, String("pixels"))
                var vae_link = _workflow_find_input_link(edges, node_id, String("vae"))
                if not pos_link.found or not neg_link.found or not pixels_link.found or not vae_link.found:
                    raise Error("[501] workflow graph Reference Conditioning subgraph missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, pos_link.node_id, pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, neg_link.node_id, neg_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, pixels_link.node_id, pixels_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, vae_link.node_id, vae_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pos_link, String("CONDITIONING"), String("conditioning"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, neg_link, String("CONDITIONING"), String("conditioning_1"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pixels_link, String("IMAGE"), String("pixels"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, vae_link, String("VAE"), String("vae"))
                    var pos_text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, pos_link)
                    var neg_text = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, neg_link)
                    var reference_path = _workflow_image_path(image_nodes, image_ports, image_paths, pixels_link)
                    _set_if_missing(obj, String("reference_image"), JSONValue.from_string(reference_path))
                    _set_if_missing(obj, String("reference_latent_method"), JSONValue.from_string(String("index")))
                    reference_latent_count += 2
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("COND_LATENT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING_1"), String("COND_LATENT"))
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(pos_text)
                    cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING_1")); cond_texts.append(neg_text)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ModelSamplingAuraFlow" or type_id == "ModelSamplingSD3" or type_id == "DifferentialDiffusion":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                if not model_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing model input")
                var ready = _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    _copy_field_if_missing(obj, fields, String("shift"), String("sigma_shift"))
                    var model_name = _workflow_model_name(model_nodes, model_ports, model_names, model_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MODEL"), String("MODEL"))
                    model_nodes.append(node_id); model_ports.append(String("MODEL")); model_names.append(model_name)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "CFGGuider":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                var pos_link = _workflow_find_input_link(edges, node_id, String("positive"))
                var neg_link = _workflow_find_input_link(edges, node_id, String("negative"))
                var cfg_link = _workflow_find_input_link(edges, node_id, String("cfg"))
                if not model_link.found or not pos_link.found or not neg_link.found:
                    raise Error("[501] workflow graph CFGGuider missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, pos_link.node_id, pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, neg_link.node_id, neg_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, cfg_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pos_link, String("CONDITIONING"), String("positive"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, neg_link, String("CONDITIONING"), String("negative"))
                    var model_name = _workflow_model_name(model_nodes, model_ports, model_names, model_link)
                    if model_name != "":
                        _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                    var prompt = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, pos_link)
                    _set_if_missing(obj, String("prompt"), JSONValue.from_string(prompt))
                    saw_prompt = True
                    var negative = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, neg_link)
                    _set_if_missing(obj, String("negative"), JSONValue.from_string(negative))
                    if cfg_link.found:
                        _set_if_missing(
                            obj, String("cfg"),
                            JSONValue.from_float(
                                _workflow_scalar_float(
                                    scalar_nodes, scalar_ports, scalar_types, scalar_floats,
                                    cfg_link, String("cfg"),
                                )
                            ),
                        )
                    else:
                        _copy_field_if_missing(obj, fields, String("cfg"), String("cfg"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("GUIDER"), String("GUIDER"))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "BasicGuider":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                var cond_link = _workflow_find_input_link(edges, node_id, String("conditioning"))
                if not model_link.found or not cond_link.found:
                    raise Error("[501] workflow graph BasicGuider missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, cond_link.node_id, cond_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, cond_link, String("CONDITIONING"), String("conditioning"))
                    var model_name = _workflow_model_name(model_nodes, model_ports, model_names, model_link)
                    if model_name != "":
                        _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                    var prompt = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, cond_link)
                    _set_if_missing(obj, String("prompt"), JSONValue.from_string(prompt))
                    saw_prompt = True
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("GUIDER"), String("GUIDER"))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "Flux2Scheduler":
                var steps_link = _workflow_find_input_link(edges, node_id, String("steps"))
                var ready = _workflow_optional_link_ready(value_nodes, value_ports, steps_link)
                if ready:
                    var steps = _opt_int(fields, "steps", 20, 1, 4096)
                    if steps_link.found:
                        steps = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            steps_link, String("steps"),
                        )
                    if steps < 1 or steps > 4096:
                        raise Error("[501] workflow graph Flux2Scheduler scalar steps out of range")
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SIGMAS"), String("SIGMAS"))
                    sigmas_nodes.append(node_id); sigmas_ports.append(String("SIGMAS"))
                    sigmas_steps.append(steps)
                    sigmas_schedulers.append(String("flux2"))
                    sigmas_denoises.append(1.0)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "BasicScheduler":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                if not model_link.found:
                    raise Error("[501] workflow graph BasicScheduler missing model input")
                var scheduler_link = _workflow_find_input_link(edges, node_id, String("scheduler"))
                var steps_link = _workflow_find_input_link(edges, node_id, String("steps"))
                var denoise_link = _workflow_find_input_link(edges, node_id, String("denoise"))
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, scheduler_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, steps_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, denoise_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    var scheduler = _workflow_string(fields, String("scheduler"))
                    if scheduler == "":
                        scheduler = String("simple")
                    var steps = _opt_int(fields, "steps", 20, 1, 4096)
                    var denoise = _workflow_float(fields, String("denoise"), 1.0, 0.0, 1.0)
                    if scheduler_link.found:
                        scheduler = _workflow_scalar_string(
                            scalar_nodes, scalar_ports, scalar_types, scalar_strings,
                            scheduler_link, String("scheduler"),
                        )
                    if steps_link.found:
                        steps = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            steps_link, String("steps"),
                        )
                    if denoise_link.found:
                        denoise = _workflow_scalar_float(
                            scalar_nodes, scalar_ports, scalar_types, scalar_floats,
                            denoise_link, String("denoise"),
                        )
                    if steps < 1 or steps > 4096:
                        raise Error("[501] workflow graph BasicScheduler scalar steps out of range")
                    if denoise < 0.0 or denoise > 1.0:
                        raise Error("[501] workflow graph BasicScheduler scalar denoise out of range")
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SIGMAS"), String("SIGMAS"))
                    sigmas_nodes.append(node_id); sigmas_ports.append(String("SIGMAS"))
                    sigmas_steps.append(steps)
                    sigmas_schedulers.append(scheduler)
                    sigmas_denoises.append(denoise)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "RandomNoise":
                var seed_link = _workflow_find_input_link(edges, node_id, String("noise_seed"))
                var ready = _workflow_optional_link_ready(value_nodes, value_ports, seed_link)
                if ready:
                    var seed = _opt_int(fields, "noise_seed", 0, 0, 4294967295)
                    if seed_link.found:
                        seed = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            seed_link, String("noise_seed"),
                        )
                    if seed < 0 or seed > 4294967295:
                        raise Error("[501] workflow graph RandomNoise scalar noise_seed out of range")
                    _set_if_missing(obj, String("seed"), JSONValue.from_int(seed))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("NOISE"), String("NOISE"))
                    noise_nodes.append(node_id); noise_ports.append(String("NOISE")); noise_seeds.append(seed)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "KSamplerSelect":
                var sampler_name = _workflow_string(fields, String("sampler_name"))
                if sampler_name == "":
                    sampler_name = _workflow_string(fields, String("sampler"))
                if sampler_name == "":
                    sampler_name = String("euler")
                _set_if_missing(obj, String("sampler"), JSONValue.from_string(sampler_name))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SAMPLER"), String("SAMPLER"))
                sampler_nodes.append(node_id); sampler_ports.append(String("SAMPLER")); sampler_names.append(sampler_name)
                done[i] = True; remaining -= 1; progressed = True
            elif _workflow_is_named_sampler_node(type_id):
                # Named SAMPLER producer: the sampler name is the node TYPE. Gate
                # against the worker's supported list; an unsupported name fails
                # loud rather than substituting a different sampler.
                var named_sampler = _workflow_named_sampler_name(type_id)
                if not _workflow_worker_supports_sampler(named_sampler):
                    raise Error(
                        "[501] workflow graph " + type_id + " lowers to unsupported sampler '"
                        + named_sampler + "'; the worker supports only "
                        + "euler/flowmatch_euler/dpmpp_2m/uni_pc/uni_pc_bh2"
                    )
                _set_if_missing(obj, String("sampler"), JSONValue.from_string(named_sampler))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SAMPLER"), String("SAMPLER"))
                sampler_nodes.append(node_id); sampler_ports.append(String("SAMPLER")); sampler_names.append(named_sampler)
                done[i] = True; remaining -= 1; progressed = True
            elif _workflow_is_named_scheduler_node(type_id):
                # Named SIGMAS producer: the scheduler name is the node TYPE. Like
                # BasicScheduler, lowers to scheduler= + steps (+ denoise for
                # SDTurboScheduler). Gate against the worker's supported list; an
                # unsupported name fails loud rather than substituting.
                var named_scheduler = _workflow_named_scheduler_name(type_id)
                if not _workflow_worker_supports_scheduler(named_scheduler):
                    raise Error(
                        "[501] workflow graph " + type_id + " lowers to unsupported scheduler '"
                        + named_scheduler + "'; the worker supports only "
                        + "simple/flowmatch/flow_match/sgm_uniform"
                    )
                var named_steps_link = _workflow_find_input_link(edges, node_id, String("steps"))
                var named_sched_ready = _workflow_optional_link_ready(value_nodes, value_ports, named_steps_link)
                if named_sched_ready:
                    var named_steps = _opt_int(fields, "steps", 20, 1, 4096)
                    if named_steps_link.found:
                        named_steps = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            named_steps_link, String("steps"),
                        )
                    if named_steps < 1 or named_steps > 4096:
                        raise Error("[501] workflow graph " + type_id + " scalar steps out of range")
                    var named_denoise = 1.0
                    if type_id == "SDTurboScheduler":
                        named_denoise = _workflow_float(fields, String("denoise"), 1.0, 0.0, 1.0)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SIGMAS"), String("SIGMAS"))
                    sigmas_nodes.append(node_id); sigmas_ports.append(String("SIGMAS"))
                    sigmas_steps.append(named_steps)
                    sigmas_schedulers.append(named_scheduler)
                    sigmas_denoises.append(named_denoise)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "KSampler" or type_id == "KSamplerAdvanced" or type_id == "LanPaint_KSampler" or type_id == "LanPaint_KSamplerAdvanced":
                var advanced = type_id == "KSamplerAdvanced"
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                var pos_link = _workflow_find_input_link(edges, node_id, String("positive"))
                var neg_link = _workflow_find_input_link(edges, node_id, String("negative"))
                var latent_link = _workflow_find_input_link(edges, node_id, String("latent_image"))
                # KSamplerAdvanced seeds from `noise_seed`; KSampler from `seed`.
                # Resolve the seed input link from whichever field the node carries.
                var seed_link = _workflow_find_input_link(edges, node_id, String("seed"))
                if advanced:
                    var noise_seed_link = _workflow_find_input_link(edges, node_id, String("noise_seed"))
                    if noise_seed_link.found:
                        seed_link = noise_seed_link^
                var steps_link = _workflow_find_input_link(edges, node_id, String("steps"))
                var cfg_link = _workflow_find_input_link(edges, node_id, String("cfg"))
                var sampler_name_link = _workflow_find_input_link(edges, node_id, String("sampler_name"))
                var scheduler_link = _workflow_find_input_link(edges, node_id, String("scheduler"))
                var denoise_link = _workflow_find_input_link(edges, node_id, String("denoise"))
                if not model_link.found or not pos_link.found or not neg_link.found or not latent_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, pos_link.node_id, pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, neg_link.node_id, neg_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, latent_link.node_id, latent_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, seed_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, steps_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, cfg_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, sampler_name_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, scheduler_link)
                    and _workflow_optional_link_ready(value_nodes, value_ports, denoise_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, pos_link, String("CONDITIONING"), String("positive"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, neg_link, String("CONDITIONING"), String("negative"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, latent_link, String("LATENT"), String("latent_image"))
                    var model_name = _workflow_model_name(model_nodes, model_ports, model_names, model_link)
                    if model_name != "":
                        _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                    var prompt = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, pos_link)
                    _set_if_missing(obj, String("prompt"), JSONValue.from_string(prompt))
                    saw_prompt = True
                    var negative = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, neg_link)
                    _set_if_missing(obj, String("negative"), JSONValue.from_string(negative))
                    var latent_idx = _workflow_latent_index(latent_nodes, latent_ports, latent_link)
                    if latent_idx >= 0:
                        if latent_widths[latent_idx] > 0:
                            _set_if_missing(obj, String("width"), JSONValue.from_int(latent_widths[latent_idx]))
                        if latent_heights[latent_idx] > 0:
                            _set_if_missing(obj, String("height"), JSONValue.from_int(latent_heights[latent_idx]))
                        _set_if_missing(obj, String("images"), JSONValue.from_int(latent_images[latent_idx]))
                        if latent_init_images[latent_idx] != "":
                            _set_if_missing(obj, String("init_image"), JSONValue.from_string(latent_init_images[latent_idx]))
                        if latent_mask_images[latent_idx] != "":
                            _set_if_missing(obj, String("mask_image"), JSONValue.from_string(latent_mask_images[latent_idx]))
                    # Track the resolved step count so KSamplerAdvanced can derive
                    # `creativity` from its `start_at_step`/`steps` denoise window.
                    # 0 == no usable step count.
                    var resolved_steps = 0
                    if steps_link.found:
                        var steps = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            steps_link, String("steps"),
                        )
                        if steps < 1 or steps > 4096:
                            raise Error("[501] workflow graph KSampler scalar steps out of range")
                        _set_if_missing(obj, String("steps"), JSONValue.from_int(steps))
                        resolved_steps = steps
                    else:
                        _copy_field_if_missing(obj, fields, String("steps"), String("steps"))
                        if fields.is_object() and fields.contains("steps") and fields["steps"].is_int():
                            resolved_steps = fields["steps"].as_int()
                    if seed_link.found:
                        var seed_name = String("seed")
                        if advanced:
                            seed_name = String("noise_seed")
                        var seed = _workflow_scalar_int(
                            scalar_nodes, scalar_ports, scalar_types, scalar_ints,
                            seed_link, seed_name,
                        )
                        if seed >= 0:
                            _set_if_missing(obj, String("seed"), JSONValue.from_int(seed))
                    elif advanced:
                        # KSamplerAdvanced names the seed `noise_seed`; fall back to `seed`.
                        if fields.is_object() and fields.contains("noise_seed") and not fields["noise_seed"].is_null():
                            _workflow_set_field_if_nonnegative_int(obj, fields, String("noise_seed"), String("seed"))
                        else:
                            _workflow_set_field_if_nonnegative_int(obj, fields, String("seed"), String("seed"))
                    else:
                        _workflow_set_field_if_nonnegative_int(obj, fields, String("seed"), String("seed"))
                    if cfg_link.found:
                        _set_if_missing(
                            obj, String("cfg"),
                            JSONValue.from_float(
                                _workflow_scalar_float(
                                    scalar_nodes, scalar_ports, scalar_types, scalar_floats,
                                    cfg_link, String("cfg"),
                                )
                            ),
                        )
                    else:
                        _copy_field_if_missing(obj, fields, String("cfg"), String("cfg"))
                    if sampler_name_link.found:
                        _set_if_missing(
                            obj, String("sampler"),
                            JSONValue.from_string(
                                _workflow_scalar_string(
                                    scalar_nodes, scalar_ports, scalar_types, scalar_strings,
                                    sampler_name_link, String("sampler_name"),
                                )
                            ),
                        )
                    else:
                        _copy_field_if_missing(obj, fields, String("sampler_name"), String("sampler"))
                    if scheduler_link.found:
                        _set_if_missing(
                            obj, String("scheduler"),
                            JSONValue.from_string(
                                _workflow_scalar_string(
                                    scalar_nodes, scalar_ports, scalar_types, scalar_strings,
                                    scheduler_link, String("scheduler"),
                                )
                            ),
                        )
                    else:
                        _copy_field_if_missing(obj, fields, String("scheduler"), String("scheduler"))
                    if advanced:
                        # KSamplerAdvanced has no `denoise` field. Its denoise window is
                        # the [start_at_step, end_at_step] slice of the `steps` schedule,
                        # with `add_noise` toggling whether the latent is re-noised at the
                        # start. zimage's flat model only represents the START of the
                        # window (creativity = denoise start). An early `end_at_step`
                        # (return_with_leftover_noise) or `add_noise="disable"` (continue
                        # an existing latent without re-noising) CANNOT be faithfully
                        # lowered — so fail loud rather than silently emit a full-txt2img
                        # creativity and render the wrong image.
                        var adv_steps = 0
                        if resolved_steps > 0:
                            adv_steps = resolved_steps
                        var add_noise = String(_workflow_string(fields, String("add_noise")).lower())
                        if add_noise == "disable":
                            raise Error(
                                "[501] workflow graph KSamplerAdvanced add_noise=disable"
                                " (continue without re-noising) is not representable in the"
                                " zimage flat denoise model"
                            )
                        # Default end_at_step to a large sentinel (Comfy's "run to the
                        # end"); only an explicit early end is rejected.
                        var end_at_step = _opt_int(fields, "end_at_step", 1000000, 0, 10000000)
                        if adv_steps > 0 and end_at_step < adv_steps:
                            raise Error(
                                "[501] workflow graph KSamplerAdvanced early end_at_step"
                                " (return_with_leftover_noise) is not representable in the"
                                " zimage flat denoise model"
                            )
                        var start_at_step = _opt_int(fields, "start_at_step", 0, 0, 4096)
                        var creativity = 1.0
                        if adv_steps > 0:
                            var raw = 1.0 - Float64(start_at_step) / Float64(adv_steps)
                            creativity = raw.clamp(0.0, 1.0)
                        _set_if_missing(obj, String("creativity"), JSONValue.from_float(creativity))
                    elif denoise_link.found:
                        var denoise = _workflow_scalar_float(
                            scalar_nodes, scalar_ports, scalar_types, scalar_floats,
                            denoise_link, String("denoise"),
                        )
                        if denoise < 0.0 or denoise > 1.0:
                            raise Error("[501] workflow graph KSampler scalar denoise out of range")
                        _set_if_missing(obj, String("creativity"), JSONValue.from_float(denoise))
                    else:
                        _copy_field_if_missing(obj, fields, String("denoise"), String("creativity"))
                    if type_id == "LanPaint_KSampler" or type_id == "LanPaint_KSamplerAdvanced":
                        _workflow_copy_lanpaint_sampler_fields(obj, fields)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    if latent_idx >= 0:
                        latent_widths.append(latent_widths[latent_idx])
                        latent_heights.append(latent_heights[latent_idx])
                        latent_images.append(latent_images[latent_idx])
                        latent_init_images.append(latent_init_images[latent_idx])
                        latent_mask_images.append(latent_mask_images[latent_idx])
                    else:
                        latent_widths.append(512); latent_heights.append(512); latent_images.append(1)
                        latent_init_images.append(String(""))
                        latent_mask_images.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "SamplerCustomAdvanced" or type_id == "LanPaint_SamplerCustomAdvanced":
                var noise_link = _workflow_find_input_link(edges, node_id, String("noise"))
                var guider_link = _workflow_find_input_link(edges, node_id, String("guider"))
                var sampler_link = _workflow_find_input_link(edges, node_id, String("sampler"))
                var sigmas_link = _workflow_find_input_link(edges, node_id, String("sigmas"))
                var latent_link = _workflow_find_input_link(edges, node_id, String("latent_image"))
                if not noise_link.found or not guider_link.found or not sampler_link.found or not sigmas_link.found or not latent_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, noise_link.node_id, noise_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, guider_link.node_id, guider_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sampler_link.node_id, sampler_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sigmas_link.node_id, sigmas_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, latent_link.node_id, latent_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, noise_link, String("NOISE"), String("noise"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, guider_link, String("GUIDER"), String("guider"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sampler_link, String("SAMPLER"), String("sampler"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sigmas_link, String("SIGMAS"), String("sigmas"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, latent_link, String("LATENT"), String("latent_image"))
                    for j in range(len(noise_nodes)):
                        if noise_nodes[j] == noise_link.node_id and noise_ports[j] == noise_link.port:
                            _set_if_missing(obj, String("seed"), JSONValue.from_int(noise_seeds[j]))
                    for j in range(len(sampler_nodes)):
                        if sampler_nodes[j] == sampler_link.node_id and sampler_ports[j] == sampler_link.port:
                            _set_if_missing(obj, String("sampler"), JSONValue.from_string(sampler_names[j]))
                    for j in range(len(sigmas_nodes)):
                        if sigmas_nodes[j] == sigmas_link.node_id and sigmas_ports[j] == sigmas_link.port:
                            _set_if_missing(obj, String("steps"), JSONValue.from_int(sigmas_steps[j]))
                            _set_if_missing(obj, String("scheduler"), JSONValue.from_string(sigmas_schedulers[j]))
                            _set_if_missing(obj, String("creativity"), JSONValue.from_float(sigmas_denoises[j]))
                    if type_id == "LanPaint_SamplerCustomAdvanced":
                        _workflow_copy_lanpaint_sampler_fields(obj, fields)
                    var latent_idx = _workflow_latent_index(latent_nodes, latent_ports, latent_link)
                    if latent_idx >= 0:
                        if latent_widths[latent_idx] > 0:
                            _set_if_missing(obj, String("width"), JSONValue.from_int(latent_widths[latent_idx]))
                        if latent_heights[latent_idx] > 0:
                            _set_if_missing(obj, String("height"), JSONValue.from_int(latent_heights[latent_idx]))
                        _set_if_missing(obj, String("images"), JSONValue.from_int(latent_images[latent_idx]))
                        if latent_init_images[latent_idx] != "":
                            _set_if_missing(obj, String("init_image"), JSONValue.from_string(latent_init_images[latent_idx]))
                        if latent_mask_images[latent_idx] != "":
                            _set_if_missing(obj, String("mask_image"), JSONValue.from_string(latent_mask_images[latent_idx]))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    if latent_idx >= 0:
                        latent_widths.append(latent_widths[latent_idx])
                        latent_heights.append(latent_heights[latent_idx])
                        latent_images.append(latent_images[latent_idx])
                        latent_init_images.append(latent_init_images[latent_idx])
                        latent_mask_images.append(latent_mask_images[latent_idx])
                    else:
                        latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                        latent_init_images.append(String(""))
                        latent_mask_images.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "SamplerCustom":
                # Single-output sibling of SamplerCustomAdvanced. Unlike the
                # Advanced node it has NO noise/guider input ports — it builds CFG
                # guidance and noise itself from its model/positive/negative inputs
                # and add_noise/noise_seed/cfg widgets, then samples with the
                # SAMPLER + SIGMAS inputs. Reuses the SAME flat
                # sampler/scheduler/steps/seed/cfg path.
                var sc_model_link = _workflow_find_input_link(edges, node_id, String("model"))
                var sc_pos_link = _workflow_find_input_link(edges, node_id, String("positive"))
                var sc_neg_link = _workflow_find_input_link(edges, node_id, String("negative"))
                var sc_sampler_link = _workflow_find_input_link(edges, node_id, String("sampler"))
                var sc_sigmas_link = _workflow_find_input_link(edges, node_id, String("sigmas"))
                var sc_latent_link = _workflow_find_input_link(edges, node_id, String("latent_image"))
                if (
                    not sc_model_link.found
                    or not sc_pos_link.found
                    or not sc_neg_link.found
                    or not sc_sampler_link.found
                    or not sc_sigmas_link.found
                    or not sc_latent_link.found
                ):
                    raise Error("[501] workflow graph SamplerCustom missing required typed input")
                var sc_ready = (
                    _workflow_value_index(value_nodes, value_ports, sc_model_link.node_id, sc_model_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sc_pos_link.node_id, sc_pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sc_neg_link.node_id, sc_neg_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sc_sampler_link.node_id, sc_sampler_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sc_sigmas_link.node_id, sc_sigmas_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, sc_latent_link.node_id, sc_latent_link.port) >= 0
                )
                if sc_ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sc_model_link, String("MODEL"), String("model"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sc_pos_link, String("CONDITIONING"), String("positive"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sc_neg_link, String("CONDITIONING"), String("negative"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sc_sampler_link, String("SAMPLER"), String("sampler"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sc_sigmas_link, String("SIGMAS"), String("sigmas"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, sc_latent_link, String("LATENT"), String("latent_image"))
                    var sc_model_name = _workflow_model_name(model_nodes, model_ports, model_names, sc_model_link)
                    if sc_model_name != "":
                        _set_if_missing(obj, String("model"), JSONValue.from_string(sc_model_name))
                    var sc_prompt = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, sc_pos_link)
                    _set_if_missing(obj, String("prompt"), JSONValue.from_string(sc_prompt))
                    saw_prompt = True
                    var sc_negative = _workflow_conditioning_text(cond_nodes, cond_ports, cond_texts, sc_neg_link)
                    _set_if_missing(obj, String("negative"), JSONValue.from_string(sc_negative))
                    # cfg/seed come from the node's own widgets (SamplerCustom has
                    # no CFGGuider/RandomNoise inputs).
                    _copy_field_if_missing(obj, fields, String("cfg"), String("cfg"))
                    _workflow_set_field_if_nonnegative_int(obj, fields, String("noise_seed"), String("seed"))
                    for j in range(len(sampler_nodes)):
                        if sampler_nodes[j] == sc_sampler_link.node_id and sampler_ports[j] == sc_sampler_link.port:
                            _set_if_missing(obj, String("sampler"), JSONValue.from_string(sampler_names[j]))
                    for j in range(len(sigmas_nodes)):
                        if sigmas_nodes[j] == sc_sigmas_link.node_id and sigmas_ports[j] == sc_sigmas_link.port:
                            _set_if_missing(obj, String("steps"), JSONValue.from_int(sigmas_steps[j]))
                            _set_if_missing(obj, String("scheduler"), JSONValue.from_string(sigmas_schedulers[j]))
                            _set_if_missing(obj, String("creativity"), JSONValue.from_float(sigmas_denoises[j]))
                    var sc_latent_idx = _workflow_latent_index(latent_nodes, latent_ports, sc_latent_link)
                    if sc_latent_idx >= 0:
                        if latent_widths[sc_latent_idx] > 0:
                            _set_if_missing(obj, String("width"), JSONValue.from_int(latent_widths[sc_latent_idx]))
                        if latent_heights[sc_latent_idx] > 0:
                            _set_if_missing(obj, String("height"), JSONValue.from_int(latent_heights[sc_latent_idx]))
                        _set_if_missing(obj, String("images"), JSONValue.from_int(latent_images[sc_latent_idx]))
                        if latent_init_images[sc_latent_idx] != "":
                            _set_if_missing(obj, String("init_image"), JSONValue.from_string(latent_init_images[sc_latent_idx]))
                        if latent_mask_images[sc_latent_idx] != "":
                            _set_if_missing(obj, String("mask_image"), JSONValue.from_string(latent_mask_images[sc_latent_idx]))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    if sc_latent_idx >= 0:
                        latent_widths.append(latent_widths[sc_latent_idx])
                        latent_heights.append(latent_heights[sc_latent_idx])
                        latent_images.append(latent_images[sc_latent_idx])
                        latent_init_images.append(latent_init_images[sc_latent_idx])
                        latent_mask_images.append(latent_mask_images[sc_latent_idx])
                    else:
                        latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                        latent_init_images.append(String(""))
                        latent_mask_images.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "LanPaint_MaskBlend":
                var image1_link = _workflow_find_input_link(edges, node_id, String("image1"))
                var image2_link = _workflow_find_input_link(edges, node_id, String("image2"))
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not image1_link.found or not image2_link.found or not mask_link.found:
                    raise Error("[501] workflow graph LanPaint_MaskBlend missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, image1_link.node_id, image1_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, image2_link.node_id, image2_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image1_link, String("IMAGE"), String("image1"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image2_link, String("IMAGE"), String("image2"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image2_link)
                    if image_path == "":
                        image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image1_link)
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    var mask_source = _workflow_source_meta(mask_nodes, mask_ports, mask_sources, mask_link)
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                    _set_if_missing(obj, String("lanpaint_mask_channel"), JSONValue.from_string(mask_source))
                    _copy_field_if_missing(obj, fields, String("blend_overlap"), String("lanpaint_mask_blend_overlap"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                    image_mask_sources.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "VAEDecode":
                var samples_link = _workflow_find_input_link(edges, node_id, String("samples"))
                var vae_link = _workflow_find_input_link(edges, node_id, String("vae"))
                if not samples_link.found or not vae_link.found:
                    raise Error("[501] workflow graph VAEDecode missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, samples_link.node_id, samples_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, vae_link.node_id, vae_link.port) >= 0
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, samples_link, String("LATENT"), String("samples"))
                    _workflow_require_value_type(value_nodes, value_ports, value_types, vae_link, String("VAE"), String("vae"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(String(""))
                    image_mask_sources.append(String(""))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "SaveImage":
                var image_link = _workflow_find_input_link(edges, node_id, String("images"))
                var prefix_link = _workflow_find_input_link(edges, node_id, String("filename_prefix"))
                if not image_link.found:
                    raise Error("[501] workflow graph SaveImage missing images input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port) >= 0
                    and _workflow_optional_link_ready(value_nodes, value_ports, prefix_link)
                )
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("images"))
                    var prefix = _workflow_string(fields, String("filename_prefix"))
                    if prefix_link.found:
                        prefix = _workflow_scalar_string(
                            scalar_nodes, scalar_ports, scalar_types, scalar_strings,
                            prefix_link, String("filename_prefix"),
                        )
                    if prefix != "":
                        _set_if_missing(obj, String("workflow_save_prefix"), JSONValue.from_string(prefix))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "PreviewImage":
                var image_link = _workflow_find_input_link(edges, node_id, String("images"))
                if not image_link.found:
                    image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    done[i] = True; remaining -= 1; progressed = True
                else:
                    var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                    if idx >= 0:
                        _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("images"))
                        done[i] = True; remaining -= 1; progressed = True
            elif type_id == "MarkdownNote" or type_id == "Note":
                done[i] = True; remaining -= 1; progressed = True

        if not progressed:
            raise Error("[501] workflow graph has unresolved or cyclic typed links")

    if not saw_prompt and (not obj.contains("prompt") or obj["prompt"].is_null()):
        raise Error("[501] workflow graph did not contain a prompt node")
    if reference_latent_count > 0:
        _set_if_missing(obj, String("reference_latent_count"), JSONValue.from_int(reference_latent_count))
    _record_workflow_execution(obj, String("typed_linked_graph"), nodes_json.length(), edges.length())


def apply_workflow_params(mut obj: JSONValue) raises:
    """Map a Serenity/Comfy/Swarm workflow request to backend genparams.

    Supported linked graph bodies run through `apply_typed_workflow_graph`.
    Field-only `params`/`genparams` bodies remain compatibility adapters, and
    unknown active graph nodes fail with 501 instead of silently no-oping.
    """
    if not obj.contains("workflow") or obj["workflow"].is_null():
        if looks_like_ideogram4_comfy_ui_export(obj):
            var wf_copy = obj.copy()
            apply_ideogram4_comfy_ui_export(obj, wf_copy)
        elif looks_like_comfy_api_prompt_graph(obj):
            var wf_copy = obj.copy()
            apply_comfy_api_prompt_graph(obj, wf_copy)
        elif looks_like_comfy_ui_canvas_graph(obj):
            var wf_copy = obj.copy()
            apply_comfy_ui_canvas_graph(obj, wf_copy)
        return
    var wf = obj["workflow"]
    if not wf.is_object():
        raise Error("[501] workflow graph body must be an object")

    if looks_like_ideogram4_comfy_ui_export(wf):
        apply_ideogram4_comfy_ui_export(obj, wf)
        return

    if looks_like_comfy_api_prompt_graph(wf):
        apply_comfy_api_prompt_graph(obj, wf)
        return

    if looks_like_comfy_ui_canvas_graph(wf):
        apply_comfy_ui_canvas_graph(obj, wf)
        return

    if wf.contains("params") and wf["params"].is_object():
        var params = wf["params"]
        var keys: List[String] = [
            "model", "prompt", "prompt_raw", "prompt_json", "negative", "width", "height",
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
        ]
        for i in range(len(keys)):
            _copy_field_if_missing(obj, params, keys[i], keys[i])
        _copy_field_if_missing(obj, params, String("filename_prefix"), String("workflow_save_prefix"))
        _record_workflow_execution(obj, String("flat_params_adapter"), 0, 0)
        return

    if wf.contains("genparams") and wf["genparams"].is_object():
        var params = wf["genparams"]
        var keys: List[String] = [
            "model", "prompt", "prompt_raw", "prompt_json", "negative", "width", "height",
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
        ]
        for i in range(len(keys)):
            _copy_field_if_missing(obj, params, keys[i], keys[i])
        _copy_field_if_missing(obj, params, String("filename_prefix"), String("workflow_save_prefix"))
        _record_workflow_execution(obj, String("flat_genparams_adapter"), 0, 0)
        return

    if not wf.contains("nodes") or not wf["nodes"].is_array():
        raise Error("[501] workflow graph body needs nodes or params/genparams")
    if wf.contains("edges") and wf["edges"].is_array():
        apply_typed_workflow_graph(obj, wf)
        return

    raise Error("[501] workflow graph body needs edges for typed execution")
