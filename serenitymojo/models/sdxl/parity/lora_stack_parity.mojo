# serenitymojo/models/sdxl/parity/lora_stack_parity.mojo
#
# LoRA COMPOSITION PARITY GATE for the SDXL SpatialTransformer *WITH LoRA* on all
# 10 projections per BasicTransformerBlock (models/sdxl/sdxl_unet_stack_lora.mojo).
# Loads the EXACT base weights (bw_*) + LoRA A/B inits (lin_*) + torch-autograd
# LoRA grads (lref_*) written by lora_stack_oracle.py, builds an SdxlLoraSet from
# those SAME A/B, runs sdxl_st_lora_forward + sdxl_st_lora_backward at depth=2 /
# small dims, and compares at cos >= 0.999:
#   * forward output (out) — LoRA-modified
#   * d_x  (full-chain proof: threads through the summed LoRA d_x into the ST input)
#   * d_context (cross-attn K/V LoRA d_x summed across blocks back to context)
#   * ALL 10×depth LoRA A/B grads (the deliverable: every adapter's d_A/d_B vs torch)
# Base-no-regression is covered separately by unet_stack_parity (adapters absent =>
# bit-for-bit base ST); here every adapter is LIVE (B!=0) so the LoRA path is proven.
#
# Run (oracle FIRST, SEPARATE command):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sdxl/parity/lora_stack_parity.mojo -o /tmp/sdxl_lora_parity
#   /tmp/sdxl_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.sdxl.spatial_transformer import (
    SpatialTransformerWeights, BasicTransformerBlockWeights, AttnWeights,
)
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, SdxlStLoraGrads, sdxl_st_lora_forward, sdxl_st_lora_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/"

# MUST match lora_stack_oracle.py
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


# slot order MUST match lora_block.mojo SLOT_* and the oracle SLOTS list.
def _slot_name(s: Int) -> String:
    if s == 0:
        return String("a1_to_q")
    elif s == 1:
        return String("a1_to_k")
    elif s == 2:
        return String("a1_to_v")
    elif s == 3:
        return String("a1_to_out")
    elif s == 4:
        return String("a2_to_q")
    elif s == 5:
        return String("a2_to_k")
    elif s == 6:
        return String("a2_to_v")
    elif s == 7:
        return String("a2_to_out")
    elif s == 8:
        return String("ff_proj")
    return String("ff_out")


def _slot_in(s: Int) -> Int:
    if s == 5 or s == 6:   # a2_to_k / a2_to_v: in=Cctx
        return CCTX
    if s == 9:             # ff_out: in=Cff
        return CFF
    return C


def _slot_out(s: Int) -> Int:
    if s == 8:             # ff_proj: out=2*Cff
        return 2 * CFF
    return C


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
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


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _attn(pre: String, kind: String, ctx: DeviceContext) raises -> AttnWeights:
    # kind "1" = self (k/v in=C); "2" = cross (k/v in=Cctx)
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


# Build SdxlLoraSet from the oracle's lin_*.bin A/B (so inits are identical).
def _load_lora_set(ctx: DeviceContext) raises -> SdxlLoraSet:
    var scale = ALPHA / Float32(RANK)
    var ad = List[LoraAdapter]()
    for j in range(DEPTH):
        for s in range(SDXL_SLOTS):
            var in_f = _slot_in(s)
            var out_f = _slot_out(s)
            var pre = String("lin_b") + String(j) + String("_") + _slot_name(s)
            var a = _in(pre + String("_A"))   # [rank, in]
            var b = _in(pre + String("_B"))   # [out, rank]
            ad.append(LoraAdapter(
                a^, b^, RANK, in_f, out_f, scale,
                _zeros(RANK * in_f), _zeros(RANK * in_f),
                _zeros(out_f * RANK), _zeros(out_f * RANK),
            ))
    return SdxlLoraSet(ad^, DEPTH, RANK)


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== sdxl LoRA stack_parity (SpatialTransformer + 10-slot LoRA vs torch) ====")
    print("C=", C, " Dh=", Dh, " Hh=", Hh, " N=", N, " NKV=", NKV, " CCTX=", CCTX,
          " CFF=", CFF, " DEPTH=", DEPTH, " RANK=", RANK, " ALPHA=", ALPHA)

    var x = Tensor.from_host(_in("bw_x"), [B, HSP, WSP, C], STDtype.F32, ctx)
    var context = Tensor.from_host(_in("bw_context"), [B, NKV, CCTX], STDtype.F32, ctx)
    var st = _load_st(ctx)
    var lora = _load_lora_set(ctx)

    var fwd = sdxl_st_lora_forward[B, HSP, WSP, C, NKV, CCTX, Hh, Dh, CFF, G, DEPTH](
        x.clone(ctx), context.clone(ctx), st, lora, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch (LoRA-modified) ----")
    _check(harness, "out", fwd.out.to_host(ctx), _in("lref_out"), allok)

    var go = Tensor.from_host(_in("bw_go"), [B, HSP, WSP, C], STDtype.F32, ctx)
    var g = sdxl_st_lora_backward[B, HSP, WSP, C, NKV, CCTX, Hh, Dh, CFF, G, DEPTH](
        go, fwd.acts, st, lora, ctx,
    )

    print("")
    print("---- load-bearing input grads vs torch (full-chain proof through LoRA d_x) ----")
    _check(harness, "d_x      ", g.d_x, _in("lref_d_x"), allok)
    _check(harness, "d_context", g.d_context, _in("lref_d_context"), allok)

    print("")
    print("---- ALL 10-slot LoRA A/B grads, every block vs torch (the deliverable) ----")
    for j in range(DEPTH):
        for s in range(SDXL_SLOTS):
            var flat = j * SDXL_SLOTS + s
            var nm = String("b") + String(j) + String("_") + _slot_name(s)
            var pre = String("lref_") + nm
            _check(harness, nm + String("_dA"), g.d_a[flat], _in(pre + String("_dA")), allok)
            _check(harness, nm + String("_dB"), g.d_b[flat], _in(pre + String("_dB")), allok)

    print("")
    print("nonfinite_lora_grads =", g.nonfinite_lora_grads)
    if allok and g.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — SDXL LoRA composition fwd+bwd (all A/B grads) matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
