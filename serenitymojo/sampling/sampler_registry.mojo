# serenitymojo.sampling.sampler_registry — SwarmUI/Comfy sampler discovery.
#
# This is a product-path registry, not a checker-only inventory. The daemon uses
# it for UI discovery and backend admission; backends still mark
# accepted_sampler_parity=false until each distinct algorithm has artifact,
# timing, and VRAM evidence.

struct SamplerAdmission(Copyable, Movable):
    var supported: Bool
    var backend: String
    var requested: String
    var normalized: String
    var executed: String
    var reason: String

    def __init__(
        out self,
        supported: Bool,
        backend: String,
        requested: String,
        normalized: String,
        executed: String,
        reason: String,
    ):
        self.supported = supported
        self.backend = backend
        self.requested = requested
        self.normalized = normalized
        self.executed = executed
        self.reason = reason


def _byte_string(c: UInt8) raises -> String:
    var b = List[UInt8]()
    b.append(c)
    return String(from_utf8=b)


def _json_escape_local(s: String) raises -> String:
    var out = String("")
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch == 0x22:
            out += String("\\\"")
        elif ch == 0x5C:
            out += String("\\\\")
        elif ch == 0x0A:
            out += String("\\n")
        elif ch == 0x0D:
            out += String("\\r")
        elif ch == 0x09:
            out += String("\\t")
        else:
            out += _byte_string(ch)
    return out^


def _backend_key(backend_name: String) -> String:
    var b = String(backend_name.lower())
    if b == "qwen" or b == "qwenimage" or b.find("qwen") >= 0:
        return String("qwenimage")
    if (
        b == "zimage"
        or b == "z-image"
        or b == "z_image"
        or b.find("zimage") >= 0
        or b.find("z-image") >= 0
        or b.find("z_image") >= 0
    ):
        return String("zimage")
    if b == "dispatch" or b == "isolated":
        return String("zimage")
    if b == "stub":
        return String("stub")
    return b^


def sampler_backend_for_model(model_name: String, default_backend: String) -> String:
    var m = String(model_name.lower())
    if m.find("qwen") >= 0:
        return String("qwenimage")
    if m.find("zimage") >= 0 or m.find("z-image") >= 0 or m.find("z_image") >= 0:
        return String("zimage")
    if m == "" or m == "dispatch" or m == "isolated":
        return _backend_key(default_backend)
    return _backend_key(default_backend)


def default_generation_model(default_backend: String) -> String:
    var b = _backend_key(default_backend)
    if b == "qwenimage":
        return String("qwen-image-2512")
    if b == "zimage":
        return String("zimage_base")
    return default_backend


def default_sampler_for_backend(backend_name: String) -> String:
    var b = _backend_key(backend_name)
    if b == "zimage" or b == "qwenimage":
        return String("euler")
    return String("euler")


def default_scheduler_for_backend(backend_name: String) -> String:
    var b = _backend_key(backend_name)
    if b == "zimage" or b == "qwenimage":
        return String("simple")
    return String("normal")


def normalize_sampler_name(name: String) -> String:
    var n = String(name.lower())
    if n == "":
        return String("")
    if n == "flow_match_euler" or n == "flowmatch_euler":
        return String("flowmatch_euler")
    if n == "dpm++ 2m" or n == "dpmpp 2m":
        return String("dpmpp_2m")
    if n == "uni-pc" or n == "unipc":
        return String("uni_pc")
    if n == "uni-pc bh2" or n == "unipc_bh2":
        return String("uni_pc_bh2")
    return n^


def normalize_scheduler_name(name: String) -> String:
    var n = String(name.lower())
    if n == "":
        return String("")
    if n == "flow_match" or n == "flowmatch" or n == "simple_flowmatch":
        return String("simple")
    if n == "qwen_flowmatch":
        return String("qwen")
    return n^


def sampler_admission_for_backend(
    backend_name: String, sampler_name: String
) -> SamplerAdmission:
    var b = _backend_key(backend_name)
    var requested = sampler_name.copy()
    var normalized = normalize_sampler_name(sampler_name)
    if normalized == "":
        normalized = default_sampler_for_backend(b)
    if b == "zimage":
        if normalized == "euler" or normalized == "flowmatch_euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("flowmatch_euler"),
                String("backend executes the verified rectified-flow Euler path"),
            )
        if normalized == "dpmpp_2m":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("dpmpp_2m"),
                String(
                    "backend executes the bounded Z-Image DPM++ 2M path on "
                    + "the simple flow-match sigma schedule"
                ),
            )
        if normalized == "uni_pc_bh2":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("uni_pc_bh2"),
                String(
                    "backend executes the bounded Z-Image UniPC bh2/order-2 "
                    + "path on the simple flow-match sigma schedule"
                ),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Z-Image currently supports only euler/flowmatch_euler and "
                + "bounded dpmpp_2m/uni_pc_bh2 aliases; generic UniPC, "
                + "ancestral/SDE/CFG++ catalog names "
                + "remain fail-loud until their distinct denoise loops have "
                + "artifact evidence"
            ),
        )
    if b == "qwenimage":
        if normalized == "euler" or normalized == "flowmatch_euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("flowmatch_euler"),
                String("backend executes the verified Qwen flow-match Euler path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Qwen-Image currently supports only euler/flowmatch_euler aliases; "
                + "DPM++/UniPC/ancestral/SDE catalog names remain fail-loud until "
                + "their distinct denoise loops have artifact evidence"
            ),
        )
    return SamplerAdmission(
        True,
        b,
        requested,
        normalized,
        normalized,
        String("stub/non-image backend accepts sampler metadata only"),
    )


def scheduler_admission_for_backend(
    backend_name: String, scheduler_name: String
) -> SamplerAdmission:
    var b = _backend_key(backend_name)
    var requested = scheduler_name.copy()
    var normalized = normalize_scheduler_name(scheduler_name)
    if normalized == "":
        normalized = default_scheduler_for_backend(b)
    if b == "zimage":
        if normalized == "simple":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("simple_flowmatch"),
                String("backend executes the verified Z-Image simple flow-match schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Z-Image currently supports only the simple flow-match schedule; "
                + "normal/karras/beta/turbo/align_your_steps/flux2/ltxv remain fail-loud here"
            ),
        )
    if b == "qwenimage":
        if normalized == "simple" or normalized == "qwen":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("qwen_flowmatch"),
                String("backend executes the verified Qwen dynamic flow-match schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Qwen-Image currently supports only simple/qwen flow-match scheduler aliases; "
                + "normal/karras/beta/turbo/align_your_steps/flux2/ltxv remain fail-loud here"
            ),
        )
    return SamplerAdmission(
        True,
        b,
        requested,
        normalized,
        normalized,
        String("stub/non-image backend accepts scheduler metadata only"),
    )


def _generic_unipc_blocked_detail_json() -> String:
    return String(
        '{"name":"uni_pc",'
        + '"normalized":"uni_pc",'
        + '"status":"blocked",'
        + '"not_alias_of":"uni_pc_bh2",'
        + '"comfy_dispatch":"sample_unipc",'
        + '"required_variant":"bh1",'
        + '"required_order":"min(3,len(sigmas)-2)",'
        + '"required_schedule":"SigmaConvert",'
        + '"reason":"generic Comfy uni_pc has bh1/order<=3/SigmaConvert/final-zero-replacement semantics; current runtime evidence covers only uni_pc_bh2 bh2/order-2 flow semantics"}'
    )


def _backend_json(
    backend_name: String,
    family: String,
    executed_sampler: String,
    executed_scheduler: String,
    supported_samplers: String,
    supported_schedulers: String,
    blocked_samplers: String,
    reason: String,
) raises -> String:
    var out = String("{")
    out += String('"backend":"') + _json_escape_local(backend_name) + String('",')
    out += String('"model_family":"') + _json_escape_local(family) + String('",')
    out += String('"accepted_sampler_parity":false,')
    out += String('"executed_sampler":"') + _json_escape_local(executed_sampler) + String('",')
    out += String('"executed_scheduler":"') + _json_escape_local(executed_scheduler) + String('",')
    out += String('"supported_samplers":') + supported_samplers + String(",")
    out += String('"supported_schedulers":') + supported_schedulers + String(",")
    out += String('"blocked_samplers":') + blocked_samplers + String(",")
    out += String('"unsupported_policy":"fail_loud",')
    out += String('"reason":"') + _json_escape_local(reason) + String('"}')
    return out^


def swarmui_sampler_registry_json() raises -> String:
    """Versioned /v1/samplers response.

    Catalog names are pinned to the local SwarmUI Comfy backend:
    /home/alex/SwarmUI/dlbackend/ComfyUI/comfy/samplers.py
    /home/alex/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs
    """
    var comfy_samplers = String(
        '["euler","euler_cfg_pp","euler_ancestral","euler_ancestral_cfg_pp",'
        + '"heun","heunpp2","exp_heun_2_x0","exp_heun_2_x0_sde","dpm_2",'
        + '"dpm_2_ancestral","lms","dpm_fast","dpm_adaptive",'
        + '"dpmpp_2s_ancestral","dpmpp_2s_ancestral_cfg_pp","dpmpp_sde",'
        + '"dpmpp_sde_gpu","dpmpp_2m","dpmpp_2m_cfg_pp","dpmpp_2m_sde",'
        + '"dpmpp_2m_sde_gpu","dpmpp_2m_sde_heun","dpmpp_2m_sde_heun_gpu",'
        + '"dpmpp_3m_sde","dpmpp_3m_sde_gpu","ddpm","lcm","ipndm","ipndm_v",'
        + '"deis","res_multistep","res_multistep_cfg_pp",'
        + '"res_multistep_ancestral","res_multistep_ancestral_cfg_pp",'
        + '"gradient_estimation","gradient_estimation_cfg_pp","er_sde",'
        + '"seeds_2","seeds_3","sa_solver","sa_solver_pece","ddim","uni_pc",'
        + '"uni_pc_bh2","flowmatch_euler"]'
    )
    var comfy_schedulers = String(
        '["simple","sgm_uniform","karras","exponential","ddim_uniform","beta",'
        + '"normal","linear_quadratic","kl_optimal","turbo","align_your_steps",'
        + '"flux2","ltxv","ltxv-image","flowmatch","flow_match","qwen"]'
    )
    var zimage_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler","dpmpp_2m","dpm++ 2m","uni_pc_bh2"]')
    var qwen_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler"]')
    var zimage_supported_schedulers = String('["simple","flowmatch","flow_match"]')
    var qwen_supported_schedulers = String('["simple","flowmatch","flow_match","qwen"]')
    var out = String("{\n")
    out += String('  "schema":"serenity.samplers.v1",\n')
    out += String('  "source":"local SwarmUI Comfy sampler catalog",\n')
    out += String('  "accepted_sampler_parity":false,\n')
    out += String('  "catalog":{\n')
    out += String('    "samplers":') + comfy_samplers + String(",\n")
    out += String('    "schedulers":') + comfy_schedulers + String("\n")
    out += String("  },\n")
    out += String('  "backends":[\n    ')
    out += _backend_json(
        String("zimage"),
        String("zimage"),
        String("flowmatch_euler"),
        String("simple_flowmatch"),
        zimage_supported_samplers,
        zimage_supported_schedulers,
        String("[") + _generic_unipc_blocked_detail_json() + String("]"),
        String("Z-Image daemon runs verified rectified-flow Euler/simple plus bounded DPM++ 2M and UniPC bh2/simple flow-match paths; generic uni_pc remains blocked because Comfy maps it to bh1/order<=3/SigmaConvert semantics, not the bh2/order-2 flow path."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("qwenimage"),
        String("qwenimage"),
        String("flowmatch_euler"),
        String("qwen_flowmatch"),
        qwen_supported_samplers,
        qwen_supported_schedulers,
        String("[") + _generic_unipc_blocked_detail_json() + String("]"),
        String("Qwen-Image daemon runs the verified dynamic flow-match Euler path; full Qwen generation remains separately gated."),
    )
    out += String("\n  ],\n")
    out += String('  "non_claims":[\n')
    out += String('    "The registry is a product admission/discovery surface, not proof that every catalog sampler executes.",\n')
    out += String('    "Per-algorithm sampler parity still requires real artifacts, requested/executed metadata, timings, and peak VRAM.",\n')
    out += String('    "Unsupported sampler/scheduler names must fail loud per backend."\n')
    out += String("  ]\n")
    out += String("}\n")
    return out^
