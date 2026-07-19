# models/dit/parity/krea2_attn_timing.mojo — BEFORE/AFTER timing of krea2_attention
# at the PRODUCTION 1024² scale (L=4608 = LPAD_POS), tiled-F32 vs cuDNN-flash.
#
# This isolates the site-A SDPA change the lead nsys-measured (tiled SDPA = 54% of
# 1024² GPU time). Random weights at the production arch (heads=48, kvheads=12,
# headdim=128, features=6144); NREP timed reps after a warmup. NOT a parity gate
# (that's chunk 3 / 7a / 7b) — purely wall-clock of the two attention paths.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#   -Xlinker -lcuda -Xlinker -rpath -Xlinker \
#   /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -lm \
#   serenitymojo/models/dit/parity/krea2_attn_timing.mojo

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.models.dit.krea2_dit import (
    krea2_attention,
    build_krea2_rope,
    build_krea2_text_mask,
)


comptime FEATURES = 6144
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime LFULL = 4588   # production 1024² positive (LT_POS=492 + IMGLEN=4096)
comptime LPAD = 4608    # ceil(LFULL/256)*256
comptime NREP = 5


def _rand(var shape: List[Int], dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    # cheap deterministic fill via zeros (timing is dominated by the GEMM/softmax
    # shape, not the values).
    return zeros_device(shape^, dt, ctx)


def main() raises:
    var ctx = DeviceContext()

    var x = _rand([1, LPAD, FEATURES], STDtype.BF16, ctx)
    var wq = _rand([HEADS * HEADDIM, FEATURES], STDtype.BF16, ctx)
    var wk = _rand([KVHEADS * HEADDIM, FEATURES], STDtype.BF16, ctx)
    var wv = _rand([KVHEADS * HEADDIM, FEATURES], STDtype.BF16, ctx)
    var gate_w = _rand([FEATURES, FEATURES], STDtype.BF16, ctx)
    var wo = _rand([FEATURES, FEATURES], STDtype.BF16, ctx)
    var qn = _rand([HEADDIM], STDtype.F32, ctx)
    var kn = _rand([HEADDIM], STDtype.F32, ctx)

    # rope tables [LPAD, HEADDIM/2] and the pad-to-LPAD additive mask (tiled path).
    var pos_host = List[Float32]()
    for i in range(LPAD):
        pos_host.append(Float32(i)); pos_host.append(Float32(0.0)); pos_host.append(Float32(0.0))
    var pos = Tensor.from_host(pos_host^, [LPAD * 3], STDtype.F32, ctx)
    var axes = [32, 48, 48]
    var rope = build_krea2_rope(pos, axes, Float32(1.0e3), ctx, STDtype.F32)

    var keep_host = List[Float32]()
    for i in range(LPAD):
        keep_host.append(Float32(1.0) if i < LFULL else Float32(0.0))
    var keep = Tensor.from_host(keep_host^, [LPAD], STDtype.F32, ctx)
    var mask = build_krea2_text_mask(keep, HEADS, LPAD, ctx, STDtype.BF16)
    var mask_opt = Optional[Tensor](mask^)
    var none_mask = Optional[Tensor](None)
    var rl_none = Optional[Int](None)
    var rl_full = Optional[Int](LFULL)

    print("krea2_attention timing @ L=", LPAD, " (1024² scale), heads=", HEADS,
          " kvheads=", KVHEADS, " headdim=", HEADDIM, " — ", NREP, " reps after warmup")

    # ---- TILED F32 (before) ----
    var t0w = krea2_attention[LPAD, HEADS, KVHEADS, HEADDIM](
        x, wq, wk, wv, gate_w, wo, qn, kn, rope[0], rope[1], mask_opt, rl_none, ctx)
    _ = t0w.to_host(ctx)  # force sync (warmup)
    var t_tiled_start = perf_counter_ns()
    for _r in range(NREP):
        var y = krea2_attention[LPAD, HEADS, KVHEADS, HEADDIM](
            x, wq, wk, wv, gate_w, wo, qn, kn, rope[0], rope[1], mask_opt, rl_none, ctx)
        _ = y.to_host(ctx)
    var t_tiled = Float64(perf_counter_ns() - t_tiled_start) / 1.0e6 / Float64(NREP)

    # ---- cuDNN FLASH (after) ----
    var t1w = krea2_attention[LPAD, HEADS, KVHEADS, HEADDIM](
        x, wq, wk, wv, gate_w, wo, qn, kn, rope[0], rope[1], none_mask, rl_full, ctx)
    _ = t1w.to_host(ctx)  # force sync (warmup)
    var t_flash_start = perf_counter_ns()
    for _r in range(NREP):
        var y2 = krea2_attention[LPAD, HEADS, KVHEADS, HEADDIM](
            x, wq, wk, wv, gate_w, wo, qn, kn, rope[0], rope[1], none_mask, rl_full, ctx)
        _ = y2.to_host(ctx)
    var t_flash = Float64(perf_counter_ns() - t_flash_start) / 1.0e6 / Float64(NREP)

    print("  TILED F32  : ", t_tiled, " ms / call")
    print("  cuDNN FLASH: ", t_flash, " ms / call")
    print("  speedup    : ", t_tiled / (t_flash + 1.0e-9), "x  (per single attention call)")
    print("  per-1024²-step (28 blocks, 1 block tiled + 27 flash) attention-only:")
    print("    all-tiled (before): ", 28.0 * t_tiled, " ms")
    print("    split     (after) : ", 1.0 * t_tiled + 27.0 * t_flash, " ms")
