# Krea2 TextFusion BF16 base-linear backend probe.
#
# Diagnostic only: compares the existing MAX `linear` backend and the cuBLAS
# BF16 NT shim against PyTorch BF16 debug-oracle projections. This file does not
# change production model code or weaken the strict TextFusion LoRA parity gate.
#
# Run:
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/parity/krea2_text_fusion_base_linear_backend_probe.mojo

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident_unfused,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.cublas_gemm import cublas_gemm_bf16_nt
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, reshape_owned
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]
comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_text_fusion_lora_oracle.safetensors"
comptime DEBUG_ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_text_fusion_lora_debug_oracle.safetensors"
comptime BATCH = 1
comptime LT = 16
comptime NLAYERS = 12
comptime TXTDIM = 2560
comptime HEADS = 20
comptime HEADDIM = 128
comptime RANK = 2
comptime LSCALE = Float32(1.0)


struct ProbeStats(Copyable, Movable):
    var cos: Float64
    var max_abs: Float64
    var rms: Float64
    var mean_abs: Float64
    var n: Int

    def __init__(
        out self,
        cos: Float64,
        max_abs: Float64,
        rms: Float64,
        mean_abs: Float64,
        n: Int,
    ):
        self.cos = cos
        self.max_abs = max_abs
        self.rms = rms
        self.mean_abs = mean_abs
        self.n = n


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _shape_text(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _require_bf16_tensor(st: SafeTensors, key: String, expected_shape: List[Int]) raises:
    _require(key in st.tensors, String("missing tensor ") + key)
    var info = st.tensor_info(key)
    _require(
        info.dtype == STDtype.BF16,
        key + String(" dtype mismatch got=") + info.dtype.name() + String(" expected=BF16"),
    )
    _require(
        _same_shape(info.shape, expected_shape),
        key + String(" shape mismatch got=") + _shape_text(info.shape)
        + String(" expected=") + _shape_text(expected_shape),
    )
    _require(
        info.size == _numel(expected_shape) * STDtype.BF16.byte_size(),
        key + String(" byte-size mismatch"),
    )


def _load_bf16(st: SafeTensors, key: String, expected_shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    _require_bf16_tensor(st, key, expected_shape)
    var info = st.tensor_info(key)
    var bytes = st.tensor_bytes(key)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    _require(t.dtype() == STDtype.BF16, key + String(" device boundary changed"))
    return t^


def _arc_bf16(st: SafeTensors, key: String, expected_shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(_load_bf16(st, key, expected_shape, ctx))


def _host_ref(st: SafeTensors, key: String, expected_shape: List[Int], ctx: DeviceContext) raises -> List[Float32]:
    var t = _load_bf16(st, key, expected_shape, ctx)
    return t.to_host(ctx)


def _adapter(
    st: SafeTensors,
    label: String,
    slot: String,
    in_f: Int,
    out_f: Int,
    ctx: DeviceContext,
) raises -> Optional[LoraAdapterDevice]:
    var prefix = label + String(".") + slot + String(".")
    var a = _arc_bf16(st, prefix + "A", _shape2(RANK, in_f), ctx)
    var b = _arc_bf16(st, prefix + "B", _shape2(out_f, RANK), ctx)
    return Optional[LoraAdapterDevice](LoraAdapterDevice(a^, b^, RANK, in_f, out_f, LSCALE))


def _stats(actual: List[Float32], expected: List[Float32]) raises -> ProbeStats:
    if len(actual) != len(expected):
        raise Error(
            String("stats length mismatch actual=")
            + String(len(actual))
            + String(" expected=")
            + String(len(expected))
        )
    if len(actual) == 0:
        raise Error("stats on empty tensors")
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var sse: Float64 = 0.0
    var abs_sum: Float64 = 0.0
    var max_abs: Float64 = 0.0
    for i in range(len(actual)):
        var a = Float64(actual[i])
        var b = Float64(expected[i])
        dot += a * b
        na += a * a
        nb += b * b
        var d = a - b
        var ad = d
        if ad < 0.0:
            ad = -ad
        if ad > max_abs:
            max_abs = ad
        sse += d * d
        abs_sum += ad
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float64
    if denom == 0.0:
        cos = 1.0 if (na == 0.0 and nb == 0.0) else 0.0
    else:
        cos = dot / denom
    return ProbeStats(cos, max_abs, sqrt(sse / Float64(len(actual))), abs_sum / Float64(len(actual)), len(actual))


def _tensor_stats(t: Tensor, expected: List[Float32], ctx: DeviceContext) raises -> ProbeStats:
    return _stats(t.to_host(ctx), expected)


def _pair_stats(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> ProbeStats:
    return _stats(a.to_host(ctx), b.to_host(ctx))


def _print_stats(label: String, backend: String, s: ProbeStats):
    print(
        "RESULT ", label, " ", backend,
        " cos=", s.cos,
        " max_abs=", s.max_abs,
        " rms=", s.rms,
        " mean_abs=", s.mean_abs,
        " n=", s.n,
    )


def _better_text(label: String, max_s: ProbeStats, cub_s: ProbeStats):
    var by_rms = max_s.rms / cub_s.rms if cub_s.rms > 0.0 else (999999.0 if max_s.rms > 0.0 else 1.0)
    var by_max = max_s.max_abs / cub_s.max_abs if cub_s.max_abs > 0.0 else (999999.0 if max_s.max_abs > 0.0 else 1.0)
    var verdict = String("tie")
    if cub_s.rms < max_s.rms:
        verdict = String("cuBLAS better")
    elif cub_s.rms > max_s.rms:
        verdict = String("MAX better")
    print(
        "VERDICT ", label,
        " ", verdict,
        " rms_ratio_MAX_over_cuBLAS=", by_rms,
        " max_abs_ratio_MAX_over_cuBLAS=", by_max,
    )


def _cublas_linear_bf16_nt(x: Tensor, weight: Tensor, ctx: DeviceContext) raises -> Tensor:
    _require(x.dtype() == STDtype.BF16, "cuBLAS probe expects BF16 activation input")
    _require(weight.dtype() == STDtype.BF16, "cuBLAS probe expects BF16 weight input")
    var xshape = x.shape()
    var wshape = weight.shape()
    _require(len(xshape) >= 1, "cuBLAS probe x rank must be >= 1")
    _require(len(wshape) == 2, "cuBLAS probe weight rank must be 2")
    var k = xshape[len(xshape) - 1]
    var n = wshape[0]
    _require(
        wshape[1] == k,
        String("cuBLAS probe K mismatch x_last=")
        + String(k)
        + String(" weight_in=")
        + String(wshape[1]),
    )
    var m = 1
    var out_shape = List[Int]()
    for i in range(len(xshape) - 1):
        m *= xshape[i]
        out_shape.append(xshape[i])
    out_shape.append(n)

    # F32 is internal compute/output from cublasGemmEx, matching linear.mojo's
    # GEMM accumulator. Cast back to BF16 before comparing to the BF16 oracle.
    var c_buf = ctx.enqueue_create_buffer[DType.uint8](m * n * STDtype.F32.byte_size())
    cublas_gemm_bf16_nt(x.buf, weight.buf, c_buf, m, n, k, ctx)
    var c_f32 = Tensor(c_buf^, out_shape.copy(), STDtype.F32)
    return cast_tensor(c_f32, STDtype.BF16, ctx)


def _probe_refiner_wq(st: SafeTensors, dbg: SafeTensors, ctx: DeviceContext) raises:
    print("=== refiner0.wq.base pure base-linear probe M=16 K=2560 N=2560 ===")
    var x = _load_bf16(dbg, "refiner0.xn", _shape3(BATCH, LT, TXTDIM), ctx)
    var w = _load_bf16(st, "refiner0.wq.W", _shape2(TXTDIM, TXTDIM), ctx)
    var expected = _host_ref(dbg, "refiner0.wq.base", _shape3(BATCH, LT, TXTDIM), ctx)
    var nb = Optional[Tensor](None)
    var max_y = linear(x, w, nb^, ctx)
    var cub_y = _cublas_linear_bf16_nt(x, w, ctx)
    ctx.synchronize()

    _require(max_y.dtype() == STDtype.BF16, "MAX refiner0.wq output is not BF16")
    _require(cub_y.dtype() == STDtype.BF16, "cuBLAS refiner0.wq output is not BF16")
    var max_s = _tensor_stats(max_y, expected, ctx)
    var cub_s = _tensor_stats(cub_y, expected, ctx)
    _print_stats("refiner0.wq.base", "MAX.linear", max_s)
    _print_stats("refiner0.wq.base", "cuBLAS.bf16_nt", cub_s)
    _print_stats("refiner0.wq.base", "cuBLAS_vs_MAX", _pair_stats(cub_y, max_y, ctx))
    _better_text("refiner0.wq.base", max_s, cub_s)


def _probe_layerwise_wq(st: SafeTensors, dbg: SafeTensors, ctx: DeviceContext) raises:
    print("=== layerwise0.wq output contrast M=192 K=2560 N=2560 (base + LoRA delta vs q_pre) ===")
    var x = _load_bf16(dbg, "layerwise0.xn", _shape3(LT, NLAYERS, TXTDIM), ctx)
    var w = _load_bf16(st, "layerwise0.wq.W", _shape2(TXTDIM, TXTDIM), ctx)
    var lo = _adapter(st, "layerwise0", "wq", TXTDIM, TXTDIM, ctx)
    if not lo:
        raise Error("missing layerwise0.wq LoRA adapter")
    var expected = _host_ref(dbg, "layerwise0.q_pre", _shape4(LT, NLAYERS, HEADS, HEADDIM), ctx)
    var nb = Optional[Tensor](None)
    var max_base = linear(x, w, nb^, ctx)
    var cub_base = _cublas_linear_bf16_nt(x, w, ctx)
    var delta = klein_lora_fwd_device_resident_unfused(x, lo.value(), LT * NLAYERS, ctx)
    var max_out = add(max_base, delta, ctx)
    var cub_out = add(cub_base, delta, ctx)
    var max_q = reshape_owned(max_out^, _shape4(LT, NLAYERS, HEADS, HEADDIM))
    var cub_q = reshape_owned(cub_out^, _shape4(LT, NLAYERS, HEADS, HEADDIM))
    ctx.synchronize()

    _require(max_q.dtype() == STDtype.BF16, "MAX layerwise0.wq output is not BF16")
    _require(cub_q.dtype() == STDtype.BF16, "cuBLAS layerwise0.wq output is not BF16")
    var max_s = _tensor_stats(max_q, expected, ctx)
    var cub_s = _tensor_stats(cub_q, expected, ctx)
    _print_stats("layerwise0.wq.out", "MAX.linear_plus_lora", max_s)
    _print_stats("layerwise0.wq.out", "cuBLAS.bf16_nt_plus_lora", cub_s)
    _print_stats("layerwise0.wq.out", "cuBLAS_vs_MAX", _pair_stats(cub_q, max_q, ctx))
    _better_text("layerwise0.wq.out", max_s, cub_s)


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(ORACLE))
    var dbg = SafeTensors.open(String(DEBUG_ORACLE))
    print("Krea2 TextFusion BF16 base-linear backend probe")
    print("Inputs/weights/oracle tensors remain BF16; F32 appears only inside GEMM/host comparison.")
    _probe_refiner_wq(st, dbg, ctx)
    _probe_layerwise_wq(st, dbg, ctx)
    print("DONE: Krea2 TextFusion base-linear backend probe")
