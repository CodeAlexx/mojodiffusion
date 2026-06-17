from serenitymojo.sampling.sampler_registry import (
    sampler_admission_for_backend,
    sampler_backend_for_model,
    scheduler_admission_for_backend,
    swarmui_sampler_registry_json,
)


def require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)


def main() raises:
    var qwen = sampler_admission_for_backend(String("qwenimage"), String("euler"))
    require(not qwen.supported, String("qwen euler should stay preflight-only"))
    require(
        qwen.reason.find("metadata/preflight-only") >= 0,
        String("qwen sampler rejection should explain preflight-only status"),
    )

    var qwen_sched = scheduler_admission_for_backend(String("qwenimage"), String("qwen_flowmatch"))
    require(not qwen_sched.supported, String("qwen scheduler should stay preflight-only"))
    require(qwen_sched.normalized == "simple", String("qwen scheduler alias should normalize to simple"))

    var sd3 = sampler_admission_for_backend(String("sd3"), String("flow_match_euler"))
    require(sd3.supported, String("sd3 flow-match euler should be admitted"))
    require(sd3.executed == "sd3_flowmatch_euler", String("sd3 execution key mismatch"))

    var flux_backend = sampler_backend_for_model(String("flux-dev"), String("zimage"))
    require(flux_backend == "flux", String("flux-dev should resolve to flux backend"))
    var flux_sched = scheduler_admission_for_backend(String("flux"), String("flow_match"))
    require(flux_sched.supported, String("flux flow_match scheduler should be admitted"))
    require(flux_sched.normalized == "simple", String("flux scheduler alias should normalize to simple"))

    var ideogram_simple = scheduler_admission_for_backend(String("ideogram4"), String("simple"))
    require(ideogram_simple.supported, String("ideogram simple scheduler should be admitted as bounded AuraFlow"))
    require(ideogram_simple.normalized == "simple", String("ideogram simple scheduler should normalize to simple"))
    require(
        ideogram_simple.executed == "ideogram4_simple_flowmatch",
        String("ideogram simple scheduler execution key mismatch"),
    )
    var ideogram_karras = scheduler_admission_for_backend(String("ideogram4"), String("karras"))
    require(not ideogram_karras.supported, String("ideogram karras scheduler should stay blocked"))

    var registry = swarmui_sampler_registry_json()
    require(registry.find(String('"backend":"qwenimage"')) >= 0, String("registry missing qwenimage"))
    require(registry.find(String('"backend":"ideogram4"')) >= 0, String("registry missing ideogram4"))
    require(registry.find(String('"simple_flowmatch"')) >= 0, String("registry missing ideogram simple alias"))
    require(registry.find(String('"backend":"sd3"')) >= 0, String("registry missing sd3"))
    require(registry.find(String('"backend":"flux"')) >= 0, String("registry missing flux"))
    print("sampler_registry_smoke ok")
