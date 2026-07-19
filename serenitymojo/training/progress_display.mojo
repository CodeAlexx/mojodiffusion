# Shared pure-Mojo trainer/sample progress display.
#
# Keep trainer UIs here so every trainer prints the same operator-facing line:
#   [Name] step k/total | epoch e/E | loss ... | grad_norm ... | speed | elapsed | ETA
#
# This is display-only. Machine logs and Python replay helpers are optional dev
# tools; trainer runtime should call these Mojo functions directly.


def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)


def _pad4(n: Int) -> String:
    if n < 10:
        return String("000") + String(n)
    if n < 100:
        return String("00") + String(n)
    if n < 1000:
        return String("0") + String(n)
    return String(n)


def _fmt_fixed(x: Float64, decimals: Int) -> String:
    var sign = String("")
    var v = x
    if v < 0.0:
        sign = String("-")
        v = -v
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(v * Float64(scale) + 0.5)
    var whole = scaled // scale
    var frac = scaled % scale
    if decimals == 1:
        return sign + String(whole) + String(".") + String(frac)
    if decimals == 4:
        return sign + String(whole) + String(".") + _pad4(frac)
    return sign + String(whole)


def _hms(seconds: Float64) -> String:
    var total = Int(seconds + 0.5)
    if total < 0:
        total = 0
    var h = total // 3600
    var m = (total % 3600) // 60
    var s = total % 60
    return String(h) + String(":") + _pad2(m) + String(":") + _pad2(s)


def print_trainer_progress(
    name: String,
    step: Int,
    total: Int,
    samples_per_epoch: Int,
    loss: Float32,
    grad_norm: Float64,
    step_secs: Float64,
    noise_speed: Float64,
    elapsed_secs: Float64,
):
    var safe_samples = samples_per_epoch
    if safe_samples <= 0:
        safe_samples = 1
    var ep = ((step - 1) // safe_samples) + 1
    var ep_total = (total + safe_samples - 1) // safe_samples
    var eta = step_secs * Float64(total - step)
    print(
        String("[") + name + String("] step ") + String(step) + String("/") + String(total)
        + String(" | epoch ") + String(ep) + String("/") + String(ep_total)
        + String(" | loss ") + _fmt_fixed(Float64(loss), 4)
        + String(" | grad_norm ") + _fmt_fixed(grad_norm, 4)
        + String(" | ") + _fmt_fixed(step_secs, 1) + String("s/step")
        + String(" | elapsed ") + _hms(elapsed_secs)
        + String(" | ETA ") + _hms(eta)
    )


def print_sample_setup(
    name: String, model_name: String, steps: Int, cfg_scale: Float32,
    image_tokens: Int, blocks: Int,
):
    print(
        String("[") + name + String("] setup model=") + model_name
        + String(" steps=") + String(steps)
        + String(" cfg=") + _fmt_fixed(Float64(cfg_scale), 1)
        + String(" N_IMG=") + String(image_tokens)
        + String(" blocks=") + String(blocks)
    )


def print_sample_step(
    name: String, step: Int, total: Int, sigma: Float32,
    secs: Float64, steps_per_sec: Float64,
):
    print(
        String("[") + name + String("] denoise step ") + String(step)
        + String("/") + String(total)
        + String(" | sigma ") + _fmt_fixed(Float64(sigma), 4)
        + String(" | ") + _fmt_fixed(secs, 1) + String("s/step")
        + String(" | ") + _fmt_fixed(steps_per_sec, 4) + String(" steps/s")
    )


def print_sample_saved(name: String, path: String):
    print(String("[") + name + String("] saved ") + path)
