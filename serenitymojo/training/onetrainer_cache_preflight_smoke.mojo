# onetrainer_cache_preflight_smoke.mojo
#
# Run:
#   timeout 180 prlimit --as=12000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_cache_preflight_smoke.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    onetrainer_cache_preflight_summary,
    validate_onetrainer_cache_preflight_plan,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("onetrainer_cache_preflight_smoke FAILED: ") + msg)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("cache preflight smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("cache preflight smoke: short write to ") + path)


def _write_config(path: String, model_type: String, only_cache: Bool) raises:
    var content = String("{")
    content += '"model_type":"' + model_type + String('",')
    content += '"workspace_dir":"/tmp/ot-cache-preflight-workspace",'
    content += '"cache_dir":"/tmp/ot-cache-preflight-cache",'
    content += '"training_method":"LORA",'
    content += '"sample_after":0,'
    content += '"sample_after_unit":"NEVER",'
    if only_cache:
        content += '"only_cache":true'
    else:
        content += '"only_cache":false'
    content += String("}")
    _write_file(path, content)


def _check_model(model_type: String) raises:
    var path = String("/tmp/ot_cache_preflight_") + model_type + String(".json")
    _write_config(path, model_type, False)
    var cfg = read_model_config(path)
    var plan = create_onetrainer_cache_preflight_plan(cfg)
    validate_onetrainer_cache_preflight_plan(plan)
    _check(plan.model_type == model_type, model_type + String(" model type"))
    _check(plan.text_train_required_fields != String(""), model_type + String(" train fields"))
    _check(plan.text_sample_required_fields != String(""), model_type + String(" sample fields"))
    _check(plan.vae_cache_channels > 0, model_type + String(" VAE cache channels"))
    _check(plan.vae_prepared_channels > 0, model_type + String(" VAE prepared channels"))


def _expect_model_raises(model_type: String) raises:
    var path = String("/tmp/ot_cache_preflight_blocked_") + model_type + String(".json")
    _write_config(path, model_type, False)
    var cfg = read_model_config(path)
    var raised = False
    try:
        var plan = create_onetrainer_cache_preflight_plan(cfg)
        validate_onetrainer_cache_preflight_plan(plan)
    except e:
        raised = True
        print("  model blocked as expected [", model_type, "]:", String(e))
    if not raised:
        raise Error(String("cache preflight smoke: expected model block for ") + model_type)


def _expect_only_cache_raw_status(model_type: String, should_pass: Bool) raises:
    var path = String("/tmp/ot_cache_preflight_only_") + model_type + String(".json")
    _write_config(path, model_type, True)
    var cfg = read_model_config(path)
    var raised = False
    try:
        var plan = create_onetrainer_cache_preflight_plan(cfg)
        validate_onetrainer_cache_preflight_plan(plan)
        if should_pass:
            _check(plan.raw_vae_cache_ready(), model_type + String(" raw cache ready"))
            print("  only_cache raw VAE ready:", onetrainer_cache_preflight_summary(plan))
    except e:
        raised = True
        print("  only_cache blocked as expected [", model_type, "]:", String(e))
    if should_pass and raised:
        raise Error(String("cache preflight smoke: unexpected only_cache block for ") + model_type)
    if (not should_pass) and not raised:
        raise Error(String("cache preflight smoke: expected only_cache block for ") + model_type)


def main() raises:
    print("==== OneTrainer product cache preflight smoke ====")
    _check_model(String("qwenimage"))
    _check_model(String("ernie_image"))
    _check_model(String("anima"))
    _check_model(String("klein"))
    _check_model(String("zimage"))
    _check_model(String("chroma"))
    _check_model(String("flux"))
    _check_model(String("FLUX_2_DEV"))
    _check_model(String("STABLE_DIFFUSION_35"))
    _check_model(String("STABLE_DIFFUSION_XL_10_BASE"))
    _expect_model_raises(String("STABLE_DIFFUSION_3"))

    _expect_only_cache_raw_status(String("qwenimage"), True)
    _expect_only_cache_raw_status(String("STABLE_DIFFUSION_35"), True)
    _expect_only_cache_raw_status(String("klein"), False)
    _expect_only_cache_raw_status(String("FLUX_2_DEV"), False)
    _expect_only_cache_raw_status(String("flux"), False)
    _expect_only_cache_raw_status(String("chroma"), False)

    print("onetrainer_cache_preflight_smoke PASS")
