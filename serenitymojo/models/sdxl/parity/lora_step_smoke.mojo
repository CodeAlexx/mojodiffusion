# serenitymojo/models/sdxl/parity/lora_step_smoke.mojo
#
# END-TO-END LoRA TRAINING STEP smoke for the SDXL SpatialTransformer (mirrors the
# Klein/Ernie LoRA proof). Uses the SAME small parity dims (depth=2, small spatial)
# and the SAME base weights (lora_stack_oracle.py bw_*.bin) so it is cheap on the
# shared 3090, but the pipeline is the REAL one: build adapters (B=0 init) ->
# sdxl_st_lora_forward -> upstream grad (oracle go) -> sdxl_st_lora_backward ->
# global-norm clip -> sdxl_lora_adamw_step -> confirm LoRA-B moves 0 -> nonzero
# (the adapter is LEARNING) -> save_sdxl_lora -> load_sdxl_lora_resume -> assert A/B
# BYTE-EXACT round-trip. This is the "complete LoRA training STEP" deliverable.
#
# Run (oracle FIRST for the base weights; SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sdxl/parity/lora_step_smoke.mojo -o /tmp/sdxl_lora_step
#   /tmp/sdxl_lora_step

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.memory import alloc, ArcPointer
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.sdxl.spatial_transformer import (
    SpatialTransformerWeights, BasicTransformerBlockWeights, AttnWeights,
)
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, SdxlStLoraGrads, build_sdxl_lora_set,
    sdxl_st_lora_forward, sdxl_st_lora_backward,
    sdxl_lora_adamw_step, save_sdxl_lora, load_sdxl_lora_resume,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/"

comptime B = 1
comptime HSP = 4
comptime WSP = 4
comptime C = 64
comptime Dh = 32
comptime Hh = C // Dh        # 2
comptime N = HSP * WSP       # 16
comptime NKV = 7
comptime CCTX = 16
comptime CFF = 32
comptime G = 16
comptime DEPTH = 2
comptime RANK = 8
comptime ALPHA = Float32(16.0)
comptime ST_PREFIX = "input_blocks.4.1"
comptime SAVE_PATH = "/tmp/sdxl_lora_smoke.safetensors"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run lora_stack_oracle.py first): ") + path)
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


def _attn(pre: String, kind: String, ctx: DeviceContext) raises -> AttnWeights:
    var kv_in = C if kind == "1" else CCTX
    return AttnWeights(
        _t2(_in(pre + String("q") + kind), C, C, ctx),
        _t2(_in(pre + String("k") + kind), C, kv_in, ctx),
        _t2(_in(pre + String("v") + kind), C, kv_in, ctx),
        _t2(_in(pre + String("o") + kind), C, C, ctx),
        _t1(_in(pre + String("o") + kind + String("b")), C, ctx),
    )


def _load_block(j: Int, ctx: DeviceContext) raises -> BasicTransformerBlockWeights:
    var pre = String("bw_b") + String(j) + String("_")
    return BasicTransformerBlockWeights(
        _t1(_in(pre + String("n1w")), C, ctx), _t1(_in(pre + String("n1b")), C, ctx),
        _attn(pre, String("1"), ctx),
        _t1(_in(pre + String("n2w")), C, ctx), _t1(_in(pre + String("n2b")), C, ctx),
        _attn(pre, String("2"), ctx),
        _t1(_in(pre + String("n3w")), C, ctx), _t1(_in(pre + String("n3b")), C, ctx),
        _t2(_in(pre + String("fpw")), 2 * CFF, C, ctx), _t1(_in(pre + String("fpb")), 2 * CFF, ctx),
        _t2(_in(pre + String("fow")), C, CFF, ctx), _t1(_in(pre + String("fob")), C, ctx),
    )


def _load_st(ctx: DeviceContext) raises -> SpatialTransformerWeights:
    var blocks = List[BasicTransformerBlockWeights]()
    for j in range(DEPTH):
        blocks.append(_load_block(j, ctx))
    return SpatialTransformerWeights(
        _t1(_in("bw_gn_w"), C, ctx), _t1(_in("bw_gn_b"), C, ctx),
        _t2(_in("bw_proj_in_w"), C, C, ctx), _t1(_in("bw_proj_in_b"), C, ctx),
        blocks^,
        _t2(_in("bw_proj_out_w"), C, C, ctx), _t1(_in("bw_proj_out_b"), C, ctx),
    )


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


# host global L2 norm over the flat LoRA grads (the clip basis).
def _global_norm(grads: SdxlStLoraGrads) -> Float32:
    var ss = Float32(0.0)
    var n = len(grads.d_a)
    for i in range(n):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: SdxlStLoraGrads, max_norm: Float32):
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
    print("==== sdxl LoRA STEP smoke (build -> fwd -> bwd -> clip -> AdamW -> save/load) ====")
    print("C=", C, " Dh=", Dh, " Hh=", Hh, " N=", N, " NKV=", NKV, " CCTX=", CCTX,
          " CFF=", CFF, " DEPTH=", DEPTH, " RANK=", RANK, " ALPHA=", ALPHA)

    var x = Tensor.from_host(_in("bw_x"), [B, HSP, WSP, C], STDtype.F32, ctx)
    var context = Tensor.from_host(_in("bw_context"), [B, NKV, CCTX], STDtype.F32, ctx)
    var st = _load_st(ctx)

    # ── build the LoRA set (B=0 init -> adapter identity at step 0) ──
    var lora = build_sdxl_lora_set(DEPTH, C, CCTX, CFF, RANK, ALPHA)
    var n_adapters = DEPTH * SDXL_SLOTS
    print("")
    print("adapter count =", n_adapters, " (10 slots x", DEPTH, "blocks)")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── forward ──
    var fwd = sdxl_st_lora_forward[B, HSP, WSP, C, NKV, CCTX, Hh, Dh, CFF, G, DEPTH](
        x.clone(ctx), context.clone(ctx), st, lora, ctx,
    )

    # ── backward (oracle go as a finite, deterministic upstream) ──
    var go = Tensor.from_host(_in("bw_go"), [B, HSP, WSP, C], STDtype.F32, ctx)
    var grads = sdxl_st_lora_backward[B, HSP, WSP, C, NKV, CCTX, Hh, Dh, CFF, G, DEPTH](
        go, fwd.acts, st, lora, ctx,
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
    sdxl_lora_adamw_step(lora, grads, 1, Float32(1.0e-3), ctx)

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
    var npairs = save_sdxl_lora(lora, ST_PREFIX, SAVE_PATH, ctx)
    print("")
    print("save_sdxl_lora wrote", npairs, "adapter pairs to", SAVE_PATH)
    var reloaded = load_sdxl_lora_resume(ST_PREFIX, DEPTH, RANK, ALPHA, C, CCTX, CFF, SAVE_PATH, ctx)

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
        print("VERDICT: PASS — SDXL LoRA step trains (B 0->nonzero), grads finite, save/load byte-exact")
    else:
        print("VERDICT: FAIL — trains=", trains, " byte_exact=", byte_exact,
              " nonfinite=", grads.nonfinite_lora_grads)
