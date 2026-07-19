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
    if b == "sdxl" or b.find("sdxl") >= 0 or b.find("stable-diffusion-xl") >= 0 or b.find("animagine") >= 0:
        return String("sdxl")
    if b == "anima" or b.find("anima") >= 0:
        return String("anima")
    if b == "sd3" or b == "sd35" or b.find("sd3") >= 0 or b.find("sd35") >= 0:
        return String("sd3")
    if b == "flux2" or b == "flux-2" or b.find("klein") >= 0 or b.find("flux2") >= 0 or b.find("flux-2") >= 0:
        return String("flux2")
    if b == "flux" or b == "flux1" or b == "flux-1" or b == "flux_1" or b.find("flux") >= 0:
        return String("flux")
    if b == "qwen" or b == "qwenimage" or b.find("qwen") >= 0:
        return String("qwenimage")
    if b == "ideogram" or b == "ideogram4" or b.find("ideogram") >= 0:
        return String("ideogram4")
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
    if (
        m.find("wan2.2") >= 0
        or m.find("wan 2.2") >= 0
        or m.find("wan-2.2") >= 0
        or m.find("wan_2_2") >= 0
        or m.find("wan22") >= 0
        or m.find("hidream") >= 0
        or m.find("hi-dream") >= 0
        or m.find("hi_dream") >= 0
        or m.find("sensenova") >= 0
        or m.find("sense_nova") >= 0
        or m.find("sense-nova") >= 0
        or m.find("zimage_l2p") >= 0
        or m.find("z-image-l2p") >= 0
        or m.find("z_image_l2p") >= 0
        or m.find("l2p") >= 0
        or m.find("ltx") >= 0
        or (m.find("qwen") >= 0 and m.find("edit") >= 0)
    ):
        return String("disabled")
    if m.find("qwen") >= 0:
        return String("qwenimage")
    if m.find("ideogram") >= 0:
        return String("ideogram4")
    if m.find("sdxl") >= 0 or m.find("stable-diffusion-xl") >= 0 or m.find("animagine") >= 0:
        return String("sdxl")
    if m.find("anima") >= 0:
        return String("anima")
    if m.find("sd3") >= 0 or m.find("sd35") >= 0:
        return String("sd3")
    if m.find("flux2-dev") >= 0 or m.find("flux-2-dev") >= 0 or m.find("flux2_dev") >= 0:
        return String("disabled")
    if m.find("klein") >= 0 or m.find("flux2") >= 0 or m.find("flux-2") >= 0:
        return String("flux2")
    if m.find("flux") >= 0:
        return String("flux")
    if m.find("zimage") >= 0 or m.find("z-image") >= 0 or m.find("z_image") >= 0:
        return String("zimage")
    if m == "" or m == "dispatch" or m == "isolated":
        return _backend_key(default_backend)
    return _backend_key(default_backend)


def default_generation_model(default_backend: String) -> String:
    var b = _backend_key(default_backend)
    if b == "qwenimage":
        return String("qwen-image-2512")
    if b == "ideogram4":
        return String("ideogram-4-fp8")
    if b == "sdxl":
        return String("sdxl")
    if b == "anima":
        return String("anima")
    if b == "sd3":
        return String("sd3.5-large")
    if b == "flux":
        return String("flux-dev")
    if b == "zimage":
        return String("zimage_base")
    return default_backend


def default_sampler_for_backend(backend_name: String) -> String:
    var b = _backend_key(backend_name)
    if b == "zimage" or b == "qwenimage" or b == "ideogram4" or b == "sdxl" or b == "anima" or b == "sd3" or b == "flux" or b == "disabled":
        return String("euler")
    return String("euler")


def default_scheduler_for_backend(backend_name: String) -> String:
    var b = _backend_key(backend_name)
    if b == "ideogram4":
        return String("logitnormal")
    if b == "flux2":
        return String("flux2")
    if b == "sdxl" or b == "anima":
        return String("normal")
    if b == "sd3" or b == "flux":
        return String("simple")
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
        return String("simple")
    if n == "logitnormal" or n == "logit_normal" or n == "ideogram_logitnormal" or n == "ideogram4_logitnormal":
        return String("ideogram_logitnormal")
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
                    "backend executes the bounded Z-Image Comfy UniPC bh2/"
                    + "order<=3 path over simple or sgm_uniform flow-match sigmas"
                ),
            )
        if normalized == "uni_pc":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("uni_pc"),
                String(
                    "backend executes the bounded Z-Image generic Comfy UniPC "
                    + "bh1/order<=3 path on the simple flow-match sigma schedule"
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
                + "bounded dpmpp_2m/uni_pc/uni_pc_bh2 aliases; "
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
                String("qwenimage_flowmatch_euler"),
                String("backend executes the bounded Qwen-Image Euler/simple flow-match path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Qwen-Image currently supports only euler/flowmatch_euler over "
                + "its simple flow-match schedule; other Comfy sampler catalog "
                + "names remain fail-loud until artifact evidence exists"
            ),
        )
    if b == "ideogram4":
        if normalized == "euler" or normalized == "flowmatch_euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("ideogram4_logitnormal_euler"),
                String("backend executes the bounded Ideogram-4 logit-normal Euler path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Ideogram-4 currently supports only euler/flowmatch_euler sampler "
                + "aliases over its explicit logit-normal Euler loop; DPM++/UniPC/"
                + "ancestral/SDE catalog names remain fail-loud until artifact evidence exists"
            ),
        )
    if b == "sdxl":
        if normalized == "euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("sdxl_euler_fixed_cli"),
                String("existing SDXL sample CLI executes the fixed 1024x1024 30-step Euler path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "SDXL daemon route currently admits only Comfy euler over normal scheduler through the existing sample CLI; DPM++/UniPC/ancestral/SDE/flowmatch aliases remain fail-loud"
            ),
        )
    if b == "anima":
        if normalized == "euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("anima_euler_fixed_cli"),
                String("existing Anima sample CLI executes the fixed 1024x1024 30-step Euler path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Anima daemon route currently admits only euler through the existing sample CLI; other Comfy sampler catalog names remain fail-loud"
            ),
        )
    if b == "sd3":
        if normalized == "euler" or normalized == "flowmatch_euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("sd3_flowmatch_euler"),
                String("existing SD3.5 Mojo worker executes the bounded 1024x1024 Euler/simple flow-match path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "SD3.5 currently supports only euler/flowmatch_euler over the simple flow-match schedule; other Comfy sampler catalog names remain fail-loud"
            ),
        )
    if b == "flux":
        if normalized == "euler" or normalized == "flowmatch_euler":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("flux_flowmatch_euler"),
                String("existing Flux Mojo worker executes the bounded 1024x1024 Euler/simple flow-match path"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Flux currently supports only euler/flowmatch_euler over the simple flow-match schedule; other Comfy sampler catalog names remain fail-loud"
            ),
        )
    if b == "flux2":
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Flux2/Klein daemon backend route exists, but sampler execution "
                + "is blocked until the Qwen3 cap-cache/ReferenceLatent bridge "
                + "feeds the existing Klein staged sampler"
            ),
        )
    if b == "disabled":
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("model family is known but execution is disabled for this slice"),
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
        if normalized == "sgm_uniform":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("sgm_uniform_flowmatch"),
                String(
                    "backend executes the bounded Z-Image Comfy sgm_uniform "
                    + "flow-match schedule for Euler/DPM++/both UniPC variants"
                ),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Z-Image currently supports simple and bounded sgm_uniform flow-match schedules; "
                + "normal/karras/beta/turbo/align_your_steps/flux2/ltxv remain fail-loud here"
            ),
        )
    if b == "qwenimage":
        if normalized == "simple":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("qwenimage_simple_flowmatch"),
                String("backend executes the bounded Qwen-Image simple flow-match schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("Qwen-Image currently admits only the simple flow-match scheduler"),
        )
    if b == "ideogram4":
        if normalized == "ideogram_logitnormal":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("ideogram4_logitnormal"),
                String("backend executes the Ideogram-4 logit-normal schedule"),
            )
        if normalized == "simple":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("ideogram4_simple_flowmatch"),
                String("backend executes the bounded Comfy simple AuraFlow schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Ideogram-4 currently supports only logitnormal/ideogram_logitnormal "
                + "and simple/flowmatch scheduler aliases in the production route; "
                + "normal/karras/beta/turbo/align_your_steps/flux2/ltxv remain fail-loud here"
            ),
        )
    if b == "sdxl":
        if normalized == "normal":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("sdxl_normal_fixed_cli"),
                String("existing SDXL sample CLI executes its fixed normal Euler schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("SDXL sample CLI route currently admits only normal scheduler metadata"),
        )
    if b == "anima":
        if normalized == "normal":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("anima_normal_fixed_cli"),
                String("existing Anima sample CLI executes its fixed normal Euler schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("Anima sample CLI route currently admits only normal scheduler metadata"),
        )
    if b == "sd3":
        if normalized == "simple":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("sd3_simple_flowmatch"),
                String("existing SD3.5 Mojo worker executes its simple flow-match schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("SD3.5 currently admits only the simple flow-match scheduler"),
        )
    if b == "flux":
        if normalized == "simple":
            return SamplerAdmission(
                True,
                b,
                requested,
                normalized,
                String("flux_simple_flowmatch"),
                String("existing Flux Mojo worker executes its simple flow-match schedule"),
            )
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("Flux currently admits only the simple flow-match scheduler"),
        )
    if b == "flux2":
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String(
                "Flux2/Klein scheduler metadata is imported from Swarm/Comfy, "
                + "but the daemon backend currently admission-fails before "
                + "executing it because cap-cache/ReferenceLatent inputs are not wired"
            ),
        )
    if b == "disabled":
        return SamplerAdmission(
            False,
            b,
            requested,
            normalized,
            String(""),
            String("model family is known but scheduler execution is disabled for this slice"),
        )
    return SamplerAdmission(
        True,
        b,
        requested,
        normalized,
        normalized,
        String("stub/non-image backend accepts scheduler metadata only"),
    )


def _qwen_generic_unipc_blocked_detail_json() -> String:
    return String(
        '{"name":"uni_pc",'
        + '"normalized":"uni_pc",'
        + '"status":"blocked",'
        + '"not_alias_of":"uni_pc_bh2",'
        + '"comfy_dispatch":"sample_unipc",'
        + '"required_variant":"bh1",'
        + '"required_order":"min(3,len(sigmas)-2)",'
        + '"required_schedule":"SigmaConvert",'
        + '"reason":"Qwen-Image has no generic Comfy uni_pc runtime evidence yet"}'
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
    var zimage_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler","dpmpp_2m","dpm++ 2m","uni_pc","uni_pc_bh2"]')
    var qwen_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler"]')
    var ideogram_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler"]')
    var sdxl_supported_samplers = String('["euler"]')
    var anima_supported_samplers = String('["euler"]')
    var sd3_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler"]')
    var flux_supported_samplers = String('["euler","flowmatch_euler","flow_match_euler"]')
    var zimage_supported_schedulers = String('["simple","flowmatch","flow_match","sgm_uniform"]')
    var qwen_supported_schedulers = String('["simple","flowmatch","flow_match"]')
    var ideogram_supported_schedulers = String('["logitnormal","logit_normal","ideogram_logitnormal","ideogram4_logitnormal","simple","flowmatch","flow_match","simple_flowmatch"]')
    var sdxl_supported_schedulers = String('["normal"]')
    var anima_supported_schedulers = String('["normal"]')
    var sd3_supported_schedulers = String('["simple","flowmatch","flow_match"]')
    var flux_supported_schedulers = String('["simple","flowmatch","flow_match"]')
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
        String("[]"),
        String("Z-Image daemon runs SwarmUI/Comfy-aligned rectified-flow Euler/simple sigmas, bounded Euler/DPM++/both-UniPC sgm_uniform sigmas, plus bounded DPM++ 2M, generic UniPC bh1/order<=3, and UniPC bh2/order<=3 over Comfy SigmaConvert. Generic uni_pc is not an alias for uni_pc_bh2."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("qwenimage"),
        String("qwenimage"),
        String("qwenimage_flowmatch_euler"),
        String("qwenimage_simple_flowmatch"),
        qwen_supported_samplers,
        qwen_supported_schedulers,
        String("[]"),
        String("Qwen-Image daemon admits a bounded 1024x1024 txt2img route over Euler/simple flow-match; edit, LoRA, img2img, and broad sampler/scheduler aliases remain fail-loud."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("ideogram4"),
        String("ideogram4"),
        String("ideogram4_logitnormal_euler"),
        String("ideogram4_logitnormal"),
        ideogram_supported_samplers,
        ideogram_supported_schedulers,
        String("[]"),
        String("Ideogram-4 daemon runs a bounded 1024x1024 txt2img path with explicit logit-normal Euler semantics; non-Euler sampler and broad scheduler aliases remain fail-loud."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("sdxl"),
        String("sdxl"),
        String("sdxl_euler_fixed_cli"),
        String("sdxl_normal_fixed_cli"),
        sdxl_supported_samplers,
        sdxl_supported_schedulers,
        String("[]"),
        String("SDXL daemon route delegates to the existing Mojo SDXL sample CLI: fixed 1024x1024, 30 steps, cfg 7.5, seed 42, pre-encoded caps_pos sidecar required. This is a bounded route, not full Comfy sampler parity."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("anima"),
        String("anima"),
        String("anima_euler_fixed_cli"),
        String("anima_normal_fixed_cli"),
        anima_supported_samplers,
        anima_supported_schedulers,
        String("[]"),
        String("Anima daemon route delegates to the existing Mojo Anima sample CLI: fixed 1024x1024, 30 steps, cfg 4.5, pre-encoded caps_pos/caps_neg sidecars required. This is a bounded route, not full Comfy sampler parity."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("sd3"),
        String("sd3"),
        String("sd3_flowmatch_euler"),
        String("sd3_simple_flowmatch"),
        sd3_supported_samplers,
        sd3_supported_schedulers,
        String("[]"),
        String("SD3.5 worker admits the bounded 1024x1024 Euler/simple flow-match route backed by the streamed MMDiT and embedded VAE path. LoRA and image-conditioning remain fail-loud."),
    )
    out += String(",\n    ")
    out += _backend_json(
        String("flux"),
        String("flux"),
        String("flux_flowmatch_euler"),
        String("flux_simple_flowmatch"),
        flux_supported_samplers,
        flux_supported_schedulers,
        String("[]"),
        String("Flux worker admits the bounded 1024x1024 Euler/simple flow-match route; negative prompt, image-conditioning, and multiple LoRAs remain fail-loud."),
    )
    out += String("\n  ],\n")
    out += String('  "non_claims":[\n')
    out += String('    "The registry is a product admission/discovery surface, not proof that every catalog sampler executes.",\n')
    out += String('    "Per-algorithm sampler parity still requires real artifacts, requested/executed metadata, timings, and peak VRAM.",\n')
    out += String('    "Unsupported sampler/scheduler names must fail loud per backend."\n')
    out += String("  ]\n")
    out += String("}\n")
    return out^
