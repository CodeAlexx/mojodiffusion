# serenitymojo/models/zimage/parity/lora_step_smoke.mojo
#
# END-TO-END LoRA TRAINING STEP smoke for Z-Image (NextDiT) — mirrors the Ernie
# LoRA step proof. Uses the SAME small parity dims (NR=1/CR=1/MAIN=2, S=10, reduced
# F) and the SAME base inputs (stack_oracle.py .bin files) so it is cheap on the
# shared 3090, but the pipeline is the REAL one: build adapters (B=0 init) ->
# zimage_stack_lora_forward -> oracle d_out upstream -> zimage_stack_lora_backward ->
# global-norm clip -> zimage_lora_adamw_step -> confirm LoRA-B moves 0 -> nonzero
# (the adapter is LEARNING, ratio=1.0) -> save_zimage_lora -> load_zimage_lora_resume
# -> assert A/B BYTE-EXACT round-trip. The "complete LoRA training STEP" deliverable.
#
# Run (oracle FIRST for the base inputs; SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/zimage/parity/stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/lora_step_smoke.mojo -o /tmp/zimage_lora_step
#   /tmp/zimage_lora_step

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import alloc, ArcPointer
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.lora_block import ZIMAGE_SLOTS
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads, build_zimage_lora_set,
    zimage_stack_lora_forward, zimage_stack_lora_backward,
    zimage_lora_adamw_step, save_zimage_lora, load_zimage_lora_resume,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime N_IMG = 6
comptime N_TXT = 4
comptime S = N_IMG + N_TXT # 10
comptime F = 96
comptime OUT_CH = 16
comptime HALF = Dh // 2
comptime EPS = Float32(1e-05)
comptime FINAL_EPS = Float32(1e-06)
comptime NUM_NR = 1
comptime NUM_CR = 1
comptime NUM_MAIN = 2
comptime RANK = 8
comptime ALPHA = Float32(16.0)
comptime SAVE_PATH = "/tmp/zimage_lora_smoke.safetensors"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run stack_oracle.py first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _t1(vals: List[Float32], n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [n], STDtype.F32, ctx))


def _t2(vals: List[Float32], a: Int, b: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [a, b], STDtype.F32, ctx))


def _load_block(prefix: String, ctx: DeviceContext) raises -> ZImageBlockWeights:
    return ZImageBlockWeights(
        _t1(_in(prefix + "_n1"), D, ctx),
        _t2(_in(prefix + "_wq"), D, D, ctx),
        _t2(_in(prefix + "_wk"), D, D, ctx),
        _t2(_in(prefix + "_wv"), D, D, ctx),
        _t2(_in(prefix + "_wo"), D, D, ctx),
        _t1(_in(prefix + "_q_norm"), Dh, ctx),
        _t1(_in(prefix + "_k_norm"), Dh, ctx),
        _t1(_in(prefix + "_n2"), D, ctx),
        _t1(_in(prefix + "_fn1"), D, ctx),
        _t2(_in(prefix + "_w1"), F, D, ctx),
        _t2(_in(prefix + "_w3"), F, D, ctx),
        _t2(_in(prefix + "_w2"), D, F, ctx),
        _t1(_in(prefix + "_fn2"), D, ctx),
    )


def _load_mod(prefix: String) raises -> ZImageModVecs:
    return ZImageModVecs(
        _in(prefix + "_scale_msa"), _in(prefix + "_gate_msa"),
        _in(prefix + "_scale_mlp"), _in(prefix + "_gate_mlp"),
    )


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _global_norm(grads: ZImageLoraGrads) -> Float32:
    var ss = Float32(0.0)
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: ZImageLoraGrads, max_norm: Float32):
    var gn = _global_norm(grads)
    if gn <= max_norm or gn == 0.0:
        return
    var s = max_norm / gn
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s


def main() raises:
    var ctx = DeviceContext()
    print("==== zimage LoRA STEP smoke (build -> fwd -> bwd -> clip -> AdamW -> save/load) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F,
          " NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", NUM_MAIN,
          " RANK=", RANK, " ALPHA=", ALPHA)

    var x_seq = _in("sin_x_seq")
    var cap_seq = _in("sin_cap_seq")
    var f_scale = _in("sin_f_scale")
    var final_lin_w = Tensor.from_host(_in("sin_final_lin"), [OUT_CH, D], STDtype.F32, ctx)
    var final_lin_b = Tensor.from_host(_in("sin_final_lin_b"), [OUT_CH], STDtype.F32, ctx)

    var nr_blocks = List[ZImageBlockWeights]()
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_blocks.append(_load_block(String("sin_nr") + String(i), ctx))
        nr_mod.append(_load_mod(String("sin_nr") + String(i)))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(_load_block(String("sin_cr") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    var main_mod = List[ZImageModVecs]()
    for i in range(NUM_MAIN):
        main_blocks.append(_load_block(String("sin_main") + String(i), ctx))
        main_mod.append(_load_mod(String("sin_main") + String(i)))

    var x_cos = Tensor.from_host(_in("sin_x_cos"), [N_IMG * H, HALF], STDtype.F32, ctx)
    var x_sin = Tensor.from_host(_in("sin_x_sin"), [N_IMG * H, HALF], STDtype.F32, ctx)
    var cap_cos = Tensor.from_host(_in("sin_cap_cos"), [N_TXT * H, HALF], STDtype.F32, ctx)
    var cap_sin = Tensor.from_host(_in("sin_cap_sin"), [N_TXT * H, HALF], STDtype.F32, ctx)
    var uni_cos = Tensor.from_host(_in("sin_uni_cos"), [S * H, HALF], STDtype.F32, ctx)
    var uni_sin = Tensor.from_host(_in("sin_uni_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── build the LoRA set (B=0 init -> adapter identity at step 0) ──
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, NUM_MAIN, D, F, RANK, ALPHA)
    var n_adapters = (NUM_NR + NUM_CR + NUM_MAIN) * ZIMAGE_SLOTS
    print("")
    print("adapter count =", n_adapters, " (7 slots x", NUM_NR + NUM_CR + NUM_MAIN, "blocks)")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── forward ──
    var fwd = zimage_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        x_seq.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )

    var d_out = _in("sin_d_out")   # [N_IMG, OUT_CH] deterministic upstream

    # ── backward ──
    var grads = zimage_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), final_lin_w,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin, fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    print("nonfinite_lora_grads =", grads.nonfinite_lora_grads, " (expect 0)")

    var da_absum = Float32(0.0)
    var db_absum = Float32(0.0)
    for i in range(n_adapters):
        da_absum += _absum(grads.d_a[i])
        db_absum += _absum(grads.d_b[i])
    print("grad |dA|_1 =", da_absum, "  |dB|_1 =", db_absum)

    # ── global-norm clip (max_norm = 1.0) ──
    var gn_before = _global_norm(grads)
    _clip(grads, Float32(1.0))
    var gn_after = _global_norm(grads)
    print("global grad norm: before =", gn_before, " after clip(1.0) =", gn_after)

    # ── AdamW step ──
    zimage_lora_adamw_step(lora, grads, 1, Float32(1.0e-3), ctx)

    var b_nonzero_slots = 0
    var b_absum_after = Float32(0.0)
    for i in range(n_adapters):
        var s = _absum(lora.ad[i].b)
        b_absum_after += s
        if s > 0.0:
            b_nonzero_slots += 1
    print("")
    print("LoRA-B |.|_1 after AdamW =", b_absum_after)
    print("LoRA-B nonzero slots =", b_nonzero_slots, "/", n_adapters,
          " ratio =", Float32(b_nonzero_slots) / Float32(n_adapters))

    var trains = (b_absum_init == 0.0) and (b_absum_after > 0.0) and (b_nonzero_slots == n_adapters)

    # ── save -> load BYTE-EXACT round-trip ──
    var npairs = save_zimage_lora(lora, SAVE_PATH, ctx)
    print("")
    print("save_zimage_lora wrote", npairs, "adapter pairs to", SAVE_PATH)
    var reloaded = load_zimage_lora_resume(NUM_NR, NUM_CR, NUM_MAIN, RANK, ALPHA, SAVE_PATH, ctx)

    var max_abs_diff = Float32(0.0)
    for i in range(n_adapters):
        if len(lora.ad[i].a) != len(reloaded.ad[i].a) or len(lora.ad[i].b) != len(reloaded.ad[i].b):
            raise Error("round-trip shape mismatch")
        for j in range(len(lora.ad[i].a)):
            var d = lora.ad[i].a[j] - reloaded.ad[i].a[j]
            d = d if d >= 0.0 else -d
            if d > max_abs_diff:
                max_abs_diff = d
        for j in range(len(lora.ad[i].b)):
            var d = lora.ad[i].b[j] - reloaded.ad[i].b[j]
            d = d if d >= 0.0 else -d
            if d > max_abs_diff:
                max_abs_diff = d
    print("save/load max_abs_diff (A+B over all adapters) =", max_abs_diff,
          "  ", "BYTE-EXACT" if max_abs_diff == 0.0 else "DIVERGED")

    print("")
    var byte_exact = (max_abs_diff == Float32(0.0))
    if trains and byte_exact and grads.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — Z-Image LoRA step trains (B 0->nonzero), grads finite, save/load byte-exact")
    else:
        print("VERDICT: FAIL — trains=", trains, " byte_exact=", byte_exact,
              " nonfinite=", grads.nonfinite_lora_grads)
