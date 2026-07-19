# svdquant_ltx2_layer_parity.mojo — REAL-WEIGHT gate for the SVDQuant int4 codec.
#
# Loads a fixture produced by scripts/svdquant_selfquant.py (a REAL LTX2 bf16
# linear self-quantized to weight-only W4A16 SVDQuant in the CLEAN row-major
# layout: SVD rank-32 low-rank + group-64 int4 residual). Runs the Mojo codec
# (ops/svdquant.mojo) and asserts the reconstruction matches the ORIGINAL bf16
# weight and the ideal bf16 output — i.e. real int4 fidelity, not a self-
# consistent synthetic round-trip.
#
#   W_mojo = svdquant_reconstruct_weight(...)   vs  W_orig (ground-truth bf16)
#   y_mojo = svdquant_linear(x, w, ctx)         vs  y_true (x @ W_orig^T)
#
# Bar 0.99 (int4+r32 on a real trained weight ≈ 0.9995 W / 0.9967 y, measured).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python scripts/svdquant_selfquant.py \
#       <src.safetensors> <weight_key> serenitymojo/ops/parity/svdq_ltx2_fixture.safetensors
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/ops/parity/svdquant_ltx2_layer_parity.mojo -o /tmp/svdq_ltx2
#   /tmp/svdq_ltx2

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.svdquant import (
    SvdquantLinearA,
    svdquant_reconstruct_weight,
    svdquant_linear,
)


comptime FIXTURE = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/svdq_ltx2_fixture.safetensors"
)
comptime COS_BAR = 0.99


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_raw(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_raw(tv, ctx)


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(String("_cos: length mismatch ") + String(len(a)) + " vs " + String(len(b)))
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        var av = Float64(a[i]); var bv = Float64(b[i])
        dot += av * bv; na += av * av; nb += bv * bv
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        raise Error("_cos: zero-norm vector")
    return dot / denom


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(FIXTURE))

    var qweight = _load_raw(st, String("qweight"), ctx)
    var wscales = _load_bf16(st, String("wscales"), ctx)
    var lora_down = _load_bf16(st, String("lora_down"), ctx)
    var lora_up = _load_bf16(st, String("lora_up"), ctx)
    var smooth = _load_bf16(st, String("smooth"), ctx)
    var bias = _load_bf16(st, String("bias"), ctx)
    var x = _load_bf16(st, String("x"), ctx)
    var y_true = _load_bf16(st, String("y_true"), ctx)     # F32 [M,out]
    var W_orig = _load_bf16(st, String("W_orig"), ctx)     # F32 [out,in]

    var qw_shape = qweight.shape()
    var out_f = qw_shape[0]
    var in_f = qw_shape[1] * 2
    var rank = lora_down.shape()[1]
    var m = x.shape()[0]
    print("[svdq-ltx2] REAL LTX2 layer: out=", out_f, " in=", in_f, " rank=", rank, " M=", m)

    if qweight.dtype() != STDtype.I8 and qweight.dtype() != STDtype.U8:
        raise Error(String("[svdq-ltx2] expected qweight I8/U8, got ") + qweight.dtype().name())

    var w = SvdquantLinearA(
        qweight^, wscales^, lora_down^, lora_up^, smooth^, bias^, in_f, out_f, rank,
    )

    var W_mojo = svdquant_reconstruct_weight(w, ctx)
    var w_cos = _cos(W_mojo.to_host(ctx), W_orig.to_host(ctx))
    print("[svdq-ltx2] W_recon vs ORIGINAL bf16 cos =", w_cos)

    var y_mojo = svdquant_linear(x, w, ctx)
    var y_cos = _cos(y_mojo.to_host(ctx), y_true.to_host(ctx))
    print("[svdq-ltx2] y vs bf16-ideal cos =", y_cos)

    var w_ok = w_cos >= COS_BAR
    var y_ok = y_cos >= COS_BAR
    if w_ok and y_ok:
        print("[svdq-ltx2] PASS (real int4+r32: W cos", w_cos, ", y cos", y_cos, ">=", COS_BAR, ")")
    else:
        print("[svdq-ltx2] FAIL — W_ok=", w_ok, " y_ok=", y_ok, " (bar", COS_BAR, ")")
        raise Error("svdquant LTX2 real-layer parity FAILED")
