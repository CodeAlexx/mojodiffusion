# svdquant_synth_parity.mojo — SYNTHETIC self-consistent gate for the SVDQuant
# Class-A int4 linear (ops/svdquant.mojo). NO checkpoint required.
#
# The companion oracle (svdquant_synth_oracle.py) builds a known BF16 weight,
# quantizes it with the SAME convention the Mojo kernel decodes (group-64
# symmetric int4, lo_even + twos_complement by default), and dumps qweight /
# wscales / lora_down / lora_up / smooth / bias / x plus the torch reference
# ref_W (dequantized dense weight) and ref_y (full contract output) to a
# .safetensors fixture. This gate loads the fixture, runs:
#     W_mojo = svdquant_reconstruct_weight(...)          # int4 → BF16 dequant
#     y_mojo = svdquant_linear(x, weights, ctx)          # full contract
# and asserts cos(W_mojo, ref_W) >= 0.999 AND cos(y_mojo, ref_y) >= 0.999.
#
# Two-step run (regenerate fixture, then gate):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/svdquant_synth_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/ops/parity/svdquant_synth_parity.mojo -o /tmp/svdq_synth
#   /tmp/svdq_synth
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.svdquant import (
    SvdquantLinearA,
    svdquant_reconstruct_weight,
    svdquant_linear,
)


comptime FIXTURE = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/svdq_synth_fixture.safetensors"
)
comptime COS_BAR = 0.999


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Load a BF16/F32 tensor verbatim (compute dtype preserved)."""
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_raw(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Load raw bytes preserving the on-disk dtype (I8 qweight)."""
    var tv = st.tensor_view(name)
    return Tensor.from_view_raw(tv, ctx)


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("_cos: length mismatch ")
            + String(len(a))
            + " vs "
            + String(len(b))
        )
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        raise Error("_cos: zero-norm vector")
    return dot / denom


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(FIXTURE))

    # ── Load fixture tensors ─────────────────────────────────────────────────
    var qweight = _load_raw(st, String("qweight"), ctx)      # I8   [out, in/2]
    var wscales = _load_bf16(st, String("wscales"), ctx)     # BF16 [in/64, out]
    var lora_down = _load_bf16(st, String("lora_down"), ctx) # BF16 [in, rank]
    var lora_up = _load_bf16(st, String("lora_up"), ctx)     # BF16 [out, rank]
    var smooth = _load_bf16(st, String("smooth"), ctx)       # BF16 [in]
    var bias = _load_bf16(st, String("bias"), ctx)           # BF16 [out]
    var x = _load_bf16(st, String("x"), ctx)                 # BF16 [M, in]
    var ref_y = _load_bf16(st, String("ref_y"), ctx)         # F32  [M, out]
    var ref_W = _load_bf16(st, String("ref_W"), ctx)         # F32  [out, in]

    # ── Derive geometry from shapes ──────────────────────────────────────────
    var qw_shape = qweight.shape()
    var out_f = qw_shape[0]
    var in_f = qw_shape[1] * 2
    var rank = lora_down.shape()[1]
    var m = x.shape()[0]

    print("[svdq-synth] geometry: out=", out_f, " in=", in_f, " rank=", rank, " M=", m)
    print("[svdq-synth] qweight dtype:", qweight.dtype().name(),
          " wscales dtype:", wscales.dtype().name())

    if qweight.dtype() != STDtype.I8 and qweight.dtype() != STDtype.U8:
        raise Error(
            String("[svdq-synth] expected qweight I8/U8, got ")
            + qweight.dtype().name()
        )

    # ── Build the weight bundle ──────────────────────────────────────────────
    var w = SvdquantLinearA(
        qweight^, wscales^, lora_down^, lora_up^, smooth^, bias^,
        in_f, out_f, rank,
    )

    # ── (1) Weight reconstruction parity ─────────────────────────────────────
    var W_mojo = svdquant_reconstruct_weight(w, ctx)
    var w_h = W_mojo.to_host(ctx)
    var ref_w_h = ref_W.to_host(ctx)
    var w_cos = _cos(w_h, ref_w_h)
    print("[svdq-synth] W reconstruction cos =", w_cos)

    # ── (2) Full-contract output parity ──────────────────────────────────────
    var y_mojo = svdquant_linear(x, w, ctx)
    var y_h = y_mojo.to_host(ctx)
    var ref_y_h = ref_y.to_host(ctx)
    var y_cos = _cos(y_h, ref_y_h)
    print("[svdq-synth] y (full contract) cos =", y_cos)

    # ── Verdict ──────────────────────────────────────────────────────────────
    var w_ok = w_cos >= COS_BAR
    var y_ok = y_cos >= COS_BAR
    if w_ok and y_ok:
        print("[svdq-synth] PASS (W cos", w_cos, ">=", COS_BAR,
              "AND y cos", y_cos, ">=", COS_BAR, ")")
    else:
        print("[svdq-synth] FAIL — W_ok=", w_ok, " y_ok=", y_ok,
              " (bar", COS_BAR, ")")
        raise Error("svdquant synthetic parity FAILED")
