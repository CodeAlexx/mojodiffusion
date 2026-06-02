# Checks cached per-step modulation weights against the legacy loader path.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_step_mod_cache_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.models.klein.weights import (
    build_klein_vec_silu,
    build_klein_double_modvecs,
    build_klein_single_modvecs,
    load_klein_step_mod_weights,
    build_klein_step_mods_cached,
    build_klein_step_mods_device_cached,
)


comptime KLEIN4B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-4b.safetensors"
comptime TIMESTEP_DIM = 256
comptime D = 3072


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("klein_step_mod_cache_smoke: length mismatch")
    var m = 0.0
    for i in range(len(a)):
        var d = Float64(a[i] - b[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def _check(name: String, a: List[Float32], b: List[Float32]) raises:
    var m = _max_abs(a, b)
    print(name, " max_abs=", m)
    if m > 0.000001:
        raise Error(name + " mismatch")


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(KLEIN4B_PATH)
    var sigma = Float32(0.423)

    var tvals = List[Float32]()
    tvals.append(sigma * Float32(1000.0))
    var ts = Tensor.from_host(tvals, [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, TIMESTEP_DIM, D, ctx)
    var ref_img = build_klein_double_modvecs(st, vec_silu.copy(), String("img"), D, ctx)
    var ref_txt = build_klein_double_modvecs(st, vec_silu.copy(), String("txt"), D, ctx)
    var ref_single = build_klein_single_modvecs(st, vec_silu.copy(), D, ctx)

    var cached_weights = load_klein_step_mod_weights(st, D, ctx)
    var cached = build_klein_step_mods_cached(
        cached_weights, sigma, TIMESTEP_DIM, D, ctx
    )
    var got_img = cached[0].copy()
    var got_txt = cached[1].copy()
    var got_single = cached[2].copy()

    var cached_dev = build_klein_step_mods_device_cached(
        cached_weights, sigma, TIMESTEP_DIM, D, ctx
    )
    var got_img_dev = cached_dev[0].copy()
    var got_txt_dev = cached_dev[1].copy()
    var got_single_dev = cached_dev[2].copy()

    _check(String("img.shift1"), ref_img.shift1, got_img.shift1)
    _check(String("img.scale1"), ref_img.scale1, got_img.scale1)
    _check(String("img.gate1"), ref_img.gate1, got_img.gate1)
    _check(String("img.shift2"), ref_img.shift2, got_img.shift2)
    _check(String("img.scale2"), ref_img.scale2, got_img.scale2)
    _check(String("img.gate2"), ref_img.gate2, got_img.gate2)
    _check(String("img.dev.shift1"), ref_img.shift1, got_img_dev.shift1[].to_host(ctx))
    _check(String("img.dev.scale1"), ref_img.scale1, got_img_dev.scale1[].to_host(ctx))
    _check(String("img.dev.gate1"), ref_img.gate1, got_img_dev.gate1[].to_host(ctx))
    _check(String("img.dev.shift2"), ref_img.shift2, got_img_dev.shift2[].to_host(ctx))
    _check(String("img.dev.scale2"), ref_img.scale2, got_img_dev.scale2[].to_host(ctx))
    _check(String("img.dev.gate2"), ref_img.gate2, got_img_dev.gate2[].to_host(ctx))

    _check(String("txt.shift1"), ref_txt.shift1, got_txt.shift1)
    _check(String("txt.scale1"), ref_txt.scale1, got_txt.scale1)
    _check(String("txt.gate1"), ref_txt.gate1, got_txt.gate1)
    _check(String("txt.shift2"), ref_txt.shift2, got_txt.shift2)
    _check(String("txt.scale2"), ref_txt.scale2, got_txt.scale2)
    _check(String("txt.gate2"), ref_txt.gate2, got_txt.gate2)
    _check(String("txt.dev.shift1"), ref_txt.shift1, got_txt_dev.shift1[].to_host(ctx))
    _check(String("txt.dev.scale1"), ref_txt.scale1, got_txt_dev.scale1[].to_host(ctx))
    _check(String("txt.dev.gate1"), ref_txt.gate1, got_txt_dev.gate1[].to_host(ctx))
    _check(String("txt.dev.shift2"), ref_txt.shift2, got_txt_dev.shift2[].to_host(ctx))
    _check(String("txt.dev.scale2"), ref_txt.scale2, got_txt_dev.scale2[].to_host(ctx))
    _check(String("txt.dev.gate2"), ref_txt.gate2, got_txt_dev.gate2[].to_host(ctx))

    _check(String("single.shift"), ref_single.shift, got_single.shift)
    _check(String("single.scale"), ref_single.scale, got_single.scale)
    _check(String("single.gate"), ref_single.gate, got_single.gate)
    _check(String("single.dev.shift"), ref_single.shift, got_single_dev.shift[].to_host(ctx))
    _check(String("single.dev.scale"), ref_single.scale, got_single_dev.scale[].to_host(ctx))
    _check(String("single.dev.gate"), ref_single.gate, got_single_dev.gate[].to_host(ctx))

    print("KLEIN STEP MOD CACHE GATE PASSED")
