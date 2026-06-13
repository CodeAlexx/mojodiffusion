# serenitymojo.serve.workflow_graph — typed workflow graph adapter/executor.
#
# This is the daemon-side contract the SerenityUI graph editor can target:
# `workflow.nodes` + `workflow.edges` describe a typed value graph, this module
# validates and topologically executes the supported Comfy/Swarm t2i subset,
# then writes the resulting backend fields onto the request object. Tensor
# execution remains in the model backends; this module owns graph semantics,
# typed handles, fail-loud unsupported nodes, and import adapters.

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


def _workflow_append_lora(mut obj: JSONValue, name: String, weight: Float64) raises:
    if name == "":
        raise Error("[501] workflow graph LoraLoaderModelOnly missing lora_name")
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


def _workflow_has_prompt_override(mut obj: JSONValue) raises -> Bool:
    if obj.contains("prompt") and obj["prompt"].is_string() and obj["prompt"].as_string() != "":
        return True
    if obj.contains("prompt_raw") and obj["prompt_raw"].is_string() and obj["prompt_raw"].as_string() != "":
        _set_if_missing(obj, String("prompt"), obj["prompt_raw"])
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
            "[501] Ideogram4 Comfy export uses a prompt-builder subgraph; provide top-level prompt or prompt_raw"
        )

    var root_nodes = wf["nodes"]
    for i in range(root_nodes.length()):
        var node = root_nodes[i]
        if not node.is_object():
            raise Error("[501] Ideogram4 Comfy export root node must be an object")
        var typ = _workflow_node_type(node)
        var mode = _workflow_node_mode(node)
        if typ == "LoraLoaderModelOnly":
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
            _set_if_missing(obj, String("width"), JSONValue.from_int(_workflow_widget_int(widgets, 0, 1024)))
            _set_if_missing(obj, String("height"), JSONValue.from_int(_workflow_widget_int(widgets, 1, 1024)))
            _set_if_missing(obj, String("images"), JSONValue.from_int(_workflow_widget_int(widgets, 2, 1)))
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


def _workflow_add_value(
    mut nodes: List[Int], mut ports: List[String], mut types: List[String],
    node_id: Int, port: String, typ: String,
) raises:
    if _workflow_value_index(nodes, ports, node_id, port) >= 0:
        raise Error("[501] workflow graph duplicate output value")
    nodes.append(node_id)
    ports.append(port)
    types.append(typ)


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


def _workflow_latent_index(
    nodes: List[Int], ports: List[String], link: WorkflowLink,
) -> Int:
    for i in range(len(nodes)):
        if nodes[i] == link.node_id and ports[i] == link.port:
            return i
    return -1


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
    elif type_id == "LoraLoaderModelOnly":
        fields.set("lora_name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
        fields.set("strength_model", JSONValue.from_float(_workflow_widget_float(widgets, 1, 1.0)))
    elif type_id == "CLIPTextEncode" or type_id == "CLIPTextEncodeFlux":
        fields.set("text", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "LoadImage":
        fields.set("image", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
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
    elif type_id == "ImageToMask":
        fields.set("channel", JSONValue.from_string(_workflow_widget_string(widgets, 0, String(""))))
    elif type_id == "RandomNoise":
        fields.set("noise_seed", JSONValue.from_int(_workflow_widget_int(widgets, 0, -1)))
    elif type_id == "KSamplerSelect":
        fields.set("sampler_name", JSONValue.from_string(_workflow_widget_string(widgets, 0, String("euler"))))
    elif type_id == "Flux2Scheduler":
        fields.set("steps", JSONValue.from_int(_workflow_widget_int(widgets, 0, 20)))
    elif type_id == "CFGGuider" or type_id == "FluxGuidance":
        fields.set("cfg", JSONValue.from_float(_workflow_widget_float(widgets, 0, 4.5)))
    elif type_id == "ModelSamplingAuraFlow" or type_id == "ModelSamplingSD3":
        fields.set("shift", JSONValue.from_float(_workflow_widget_float(widgets, 0, 3.0)))
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
    elif typ == "LoraLoaderModelOnly":
        if slot == 0:
            return String("MODEL")
    elif typ == "CLIPLoader" or typ == "DualCLIPLoader" or typ == "TripleCLIPLoader":
        if slot == 0:
            return String("CLIP")
    elif typ == "VAELoader":
        if slot == 0:
            return String("VAE")
    elif typ == "CLIPTextEncode" or typ == "CLIPTextEncodeFlux":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "ConditioningZeroOut":
        if slot == 0:
            return String("CONDITIONING")
    elif typ == "LoadImage":
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
    elif typ == "VAEEncode":
        if slot == 0:
            return String("LATENT")
    elif typ == "SetLatentNoiseMask":
        if slot == 0:
            return String("LATENT")
    elif typ == "ImageScale" or typ == "ImageScaleToTotalPixels":
        if slot == 0:
            return String("IMAGE")
    elif typ == "GetImageSize":
        if slot == 0:
            return String("width")
        if slot == 1:
            return String("height")
        if slot == 2:
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
    elif typ == "KSampler" or typ == "LanPaint_KSampler" or typ == "LanPaint_KSamplerAdvanced":
        if slot == 0:
            return String("LATENT")
    elif typ == "CFGGuider":
        if slot == 0:
            return String("GUIDER")
    elif typ == "BasicGuider":
        if slot == 0:
            return String("GUIDER")
    elif typ == "Flux2Scheduler":
        if slot == 0:
            return String("SIGMAS")
    elif typ == "RandomNoise":
        if slot == 0:
            return String("NOISE")
    elif typ == "KSamplerSelect":
        if slot == 0:
            return String("SAMPLER")
    elif typ == "SamplerCustomAdvanced" or typ == "LanPaint_SamplerCustomAdvanced":
        if slot == 0 or slot == 1:
            return String("LATENT")
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
                to_ref.set("port", JSONValue.from_string(input_name))
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
            or type_id == "LoraLoaderModelOnly"
            or type_id == "CLIPLoader"
            or type_id == "DualCLIPLoader"
            or type_id == "TripleCLIPLoader"
            or type_id == "VAELoader"
            or type_id == "CLIPTextEncode"
            or type_id == "CLIPTextEncodeFlux"
            or type_id == "ConditioningZeroOut"
            or type_id == "LoadImage"
            or type_id == "ImageToMask"
            or type_id == "MaskToImage"
            or type_id == "EmptyLatentImage"
            or type_id == "EmptySD3LatentImage"
            or type_id == "EmptyFlux2LatentImage"
            or type_id == "VAEEncode"
            or type_id == "SetLatentNoiseMask"
            or type_id == "GetImageSize"
            or type_id == "ImageScale"
            or type_id == "ImageScaleToTotalPixels"
            or type_id == "ReferenceLatent"
            or type_id == "6007e698-2ebd-4917-84d8-299b35d7b7ab"
            or type_id == "f07d2d08-2bc5-4dd8-a9f0-f2347c6b5cca"
            or type_id == "ModelSamplingAuraFlow"
            or type_id == "ModelSamplingSD3"
            or type_id == "DifferentialDiffusion"
            or type_id == "KSampler"
            or type_id == "LanPaint_KSampler"
            or type_id == "LanPaint_KSamplerAdvanced"
            or type_id == "CFGGuider"
            or type_id == "BasicGuider"
            or type_id == "FluxGuidance"
            or type_id == "Flux2Scheduler"
            or type_id == "RandomNoise"
            or type_id == "KSamplerSelect"
            or type_id == "SamplerCustomAdvanced"
            or type_id == "LanPaint_SamplerCustomAdvanced"
            or type_id == "LanPaint_MaskBlend"
            or type_id == "VAEDecode"
            or type_id == "SaveImage"
            or type_id == "PreviewImage"
            or type_id == "MarkdownNote"
            or type_id == "Note"
        ):
            raise Error(String("[501] unsupported workflow graph node type: ") + type_id)
        ids.append(id)

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
    var mask_nodes = List[Int]()
    var mask_ports = List[String]()
    var mask_paths = List[String]()
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
            elif type_id == "LoraLoaderModelOnly":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                if not model_link.found:
                    raise Error("[501] workflow graph LoraLoaderModelOnly missing model input")
                var ready = _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                if ready:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, model_link, String("MODEL"), String("model"))
                    var model_name = _workflow_model_name(model_nodes, model_ports, model_names, model_link)
                    if model_name != "":
                        _set_if_missing(obj, String("model"), JSONValue.from_string(model_name))
                    var lora_name = _workflow_string(fields, String("lora_name"))
                    var strength = _workflow_float(fields, String("strength_model"), 1.0, -10.0, 10.0)
                    _workflow_append_lora(obj, lora_name, strength)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MODEL"), String("MODEL"))
                    model_nodes.append(node_id); model_ports.append(String("MODEL")); model_names.append(model_name)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "CLIPLoader" or type_id == "DualCLIPLoader" or type_id == "TripleCLIPLoader":
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CLIP"), String("CLIP"))
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "VAELoader":
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("VAE"), String("VAE"))
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "EmptyLatentImage" or type_id == "EmptySD3LatentImage" or type_id == "EmptyFlux2LatentImage":
                var width = _opt_int(fields, "width", 512, 16, 2048)
                var height = _opt_int(fields, "height", 512, 16, 2048)
                var images = _opt_int(fields, "batch_size", 1, 1, 64)
                _set_if_missing(obj, String("width"), JSONValue.from_int(width))
                _set_if_missing(obj, String("height"), JSONValue.from_int(height))
                _set_if_missing(obj, String("images"), JSONValue.from_int(images))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                latent_widths.append(width); latent_heights.append(height); latent_images.append(images)
                latent_init_images.append(String(""))
                latent_mask_images.append(String(""))
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "CLIPTextEncode" or type_id == "CLIPTextEncodeFlux":
                var clip_link = _workflow_find_input_link(edges, node_id, String("clip"))
                if clip_link.found:
                    var idx = _workflow_value_index(value_nodes, value_ports, clip_link.node_id, clip_link.port)
                    if idx >= 0:
                        _workflow_require_value_type(value_nodes, value_ports, value_types, clip_link, String("CLIP"), String("clip"))
                        var text = _workflow_conditioning_prompt_text(fields, type_id)
                        _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("CONDITIONING"), String("CONDITIONING"))
                        cond_nodes.append(node_id); cond_ports.append(String("CONDITIONING")); cond_texts.append(text)
                        done[i] = True; remaining -= 1; progressed = True
                else:
                    raise Error("[501] workflow graph " + type_id + " missing clip input")
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
            elif type_id == "LoadImage":
                var image_path = _workflow_string(fields, String("image"))
                if image_path == "":
                    image_path = _workflow_string(fields, String("path"))
                if image_path == "":
                    raise Error("[501] workflow graph LoadImage missing image path")
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
                mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(image_path)
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ImageToMask":
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph ImageToMask missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    _copy_field_if_missing(obj, fields, String("channel"), String("lanpaint_mask_channel"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                    mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(image_path)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "MaskToImage":
                var mask_link = _workflow_find_input_link(edges, node_id, String("mask"))
                if not mask_link.found:
                    raise Error("[501] workflow graph MaskToImage missing mask input")
                var idx = _workflow_value_index(value_nodes, value_ports, mask_link.node_id, mask_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, mask_link, String("MASK"), String("mask"))
                    var mask_path = _workflow_image_path(mask_nodes, mask_ports, mask_paths, mask_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(mask_path)
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "GetImageSize":
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph GetImageSize missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("width"), String("INT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("height"), String("INT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("batch_size"), String("INT"))
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "ImageScale" or type_id == "ImageScaleToTotalPixels":
                var image_link = _workflow_find_input_link(edges, node_id, String("image"))
                if not image_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing image input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("image"))
                    var image_path = _workflow_image_path(image_nodes, image_ports, image_paths, image_link)
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
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
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
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
                    _set_if_missing(obj, String("init_image"), JSONValue.from_string(init_path))
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("LATENT"), String("LATENT"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("MASK"), String("MASK"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    latent_nodes.append(node_id); latent_ports.append(String("LATENT"))
                    latent_widths.append(0); latent_heights.append(0); latent_images.append(1)
                    latent_init_images.append(init_path)
                    latent_mask_images.append(mask_path)
                    mask_nodes.append(node_id); mask_ports.append(String("MASK")); mask_paths.append(mask_path)
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(init_path)
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
                if not model_link.found or not pos_link.found or not neg_link.found:
                    raise Error("[501] workflow graph CFGGuider missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, pos_link.node_id, pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, neg_link.node_id, neg_link.port) >= 0
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
                _copy_field_if_missing(obj, fields, String("steps"), String("steps"))
                _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("SIGMAS"), String("SIGMAS"))
                sigmas_nodes.append(node_id); sigmas_ports.append(String("SIGMAS"))
                sigmas_steps.append(_opt_int(fields, "steps", 20, 1, 4096))
                done[i] = True; remaining -= 1; progressed = True
            elif type_id == "RandomNoise":
                var seed = _opt_int(fields, "noise_seed", 0, 0, 4294967295)
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
            elif type_id == "KSampler" or type_id == "LanPaint_KSampler" or type_id == "LanPaint_KSamplerAdvanced":
                var model_link = _workflow_find_input_link(edges, node_id, String("model"))
                var pos_link = _workflow_find_input_link(edges, node_id, String("positive"))
                var neg_link = _workflow_find_input_link(edges, node_id, String("negative"))
                var latent_link = _workflow_find_input_link(edges, node_id, String("latent_image"))
                if not model_link.found or not pos_link.found or not neg_link.found or not latent_link.found:
                    raise Error("[501] workflow graph " + type_id + " missing required typed input")
                var ready = (
                    _workflow_value_index(value_nodes, value_ports, model_link.node_id, model_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, pos_link.node_id, pos_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, neg_link.node_id, neg_link.port) >= 0
                    and _workflow_value_index(value_nodes, value_ports, latent_link.node_id, latent_link.port) >= 0
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
                    _copy_field_if_missing(obj, fields, String("steps"), String("steps"))
                    _workflow_set_field_if_nonnegative_int(obj, fields, String("seed"), String("seed"))
                    _copy_field_if_missing(obj, fields, String("cfg"), String("cfg"))
                    _copy_field_if_missing(obj, fields, String("sampler_name"), String("sampler"))
                    _copy_field_if_missing(obj, fields, String("scheduler"), String("scheduler"))
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
                    _set_if_missing(obj, String("scheduler"), JSONValue.from_string(String("flux2")))
                    _set_if_missing(obj, String("creativity"), JSONValue.from_float(1.0))
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
                    _set_if_missing(obj, String("mask_image"), JSONValue.from_string(mask_path))
                    _copy_field_if_missing(obj, fields, String("blend_overlap"), String("lanpaint_mask_blend_overlap"))
                    _workflow_add_value(value_nodes, value_ports, value_types, node_id, String("IMAGE"), String("IMAGE"))
                    image_nodes.append(node_id); image_ports.append(String("IMAGE")); image_paths.append(image_path)
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
                    done[i] = True; remaining -= 1; progressed = True
            elif type_id == "SaveImage":
                var image_link = _workflow_find_input_link(edges, node_id, String("images"))
                if not image_link.found:
                    raise Error("[501] workflow graph SaveImage missing images input")
                var idx = _workflow_value_index(value_nodes, value_ports, image_link.node_id, image_link.port)
                if idx >= 0:
                    _workflow_require_value_type(value_nodes, value_ports, value_types, image_link, String("IMAGE"), String("images"))
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
            "model", "prompt", "prompt_raw", "negative", "width", "height",
            "steps", "seed", "cfg", "cfg_override", "cfg_override_start_percent",
            "cfg_override_end_percent", "sampler", "scheduler", "sigma_shift",
            "variation_seed", "variation_strength", "images", "init_image", "creativity",
            "mask_image", "reference_image", "reference_latent_method", "reference_latent_count",
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
        _record_workflow_execution(obj, String("flat_params_adapter"), 0, 0)
        return

    if wf.contains("genparams") and wf["genparams"].is_object():
        var params = wf["genparams"]
        var keys: List[String] = [
            "model", "prompt", "prompt_raw", "negative", "width", "height",
            "steps", "seed", "cfg", "cfg_override", "cfg_override_start_percent",
            "cfg_override_end_percent", "sampler", "scheduler", "sigma_shift",
            "variation_seed", "variation_strength", "images", "init_image", "creativity",
            "mask_image", "reference_image", "reference_latent_method", "reference_latent_count",
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
        _record_workflow_execution(obj, String("flat_genparams_adapter"), 0, 0)
        return

    if not wf.contains("nodes") or not wf["nodes"].is_array():
        raise Error("[501] workflow graph body needs nodes or params/genparams")
    if wf.contains("edges") and wf["edges"].is_array():
        apply_typed_workflow_graph(obj, wf)
        return

    var nodes = wf["nodes"]
    var saw_prompt = False
    for i in range(nodes.length()):
        var node = nodes[i]
        if not node.is_object():
            raise Error("[501] workflow graph node must be an object")
        var type_id = _workflow_type_id(node)
        var title = _workflow_string(node, String("title"))
        var fields = JSONValue.new_object()
        if node.contains("fields") and node["fields"].is_object():
            fields = node["fields"]

        if type_id == "CheckpointLoaderSimple":
            _copy_field_if_missing(obj, fields, String("ckpt_name"), String("model"))
        elif type_id == "CLIPTextEncode":
            var text = _workflow_string(fields, String("text"))
            if text != "":
                var lower_title = String(title.lower())
                if lower_title.find("negative") >= 0:
                    _set_if_missing(obj, String("negative"), JSONValue.from_string(text))
                else:
                    _set_if_missing(obj, String("prompt"), JSONValue.from_string(text))
                    saw_prompt = True
        elif type_id == "EmptyLatentImage" or type_id == "EmptySD3LatentImage":
            _copy_field_if_missing(obj, fields, String("width"), String("width"))
            _copy_field_if_missing(obj, fields, String("height"), String("height"))
            _copy_field_if_missing(obj, fields, String("batch_size"), String("images"))
        elif type_id == "KSampler":
            _copy_field_if_missing(obj, fields, String("steps"), String("steps"))
            _copy_field_if_missing(obj, fields, String("seed"), String("seed"))
            _copy_field_if_missing(obj, fields, String("cfg"), String("cfg"))
            _copy_field_if_missing(obj, fields, String("sampler_name"), String("sampler"))
            _copy_field_if_missing(obj, fields, String("scheduler"), String("scheduler"))
            _copy_field_if_missing(obj, fields, String("denoise"), String("creativity"))
        elif type_id == "ModelSamplingAuraFlow":
            _copy_field_if_missing(obj, fields, String("shift"), String("sigma_shift"))
        elif (
            type_id == "VAEDecode"
            or type_id == "SaveImage"
            or type_id == ""
        ):
            if type_id == "":
                raise Error("[501] unsupported workflow graph format: missing type_id")
        else:
            raise Error(
                String("[501] unsupported workflow graph node type: ") + type_id
            )

    if not saw_prompt and (not obj.contains("prompt") or obj["prompt"].is_null()):
        raise Error("[501] workflow graph did not contain a prompt node")
    _record_workflow_execution(obj, String("field_only_graph_adapter"), nodes.length(), 0)
