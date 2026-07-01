# Krea2 TextFusion native SDPA vs PyTorch-CUDNN oracle.
#
# Diagnostic gate for the TextFusion LoRA parity blocker.  The oracle is written
# by krea2_text_fusion_sdpa_cudnn_oracle.py from PyTorch
# `SDPBackend.CUDNN_ATTENTION`; this Mojo gate compares the local BF16 native
# cuDNN wrappers at the exact TextFusion no-mask shapes.
#
# Run:
#   python3 serenitymojo/models/krea2/parity/krea2_text_fusion_sdpa_cudnn_oracle.py
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/parity/krea2_text_fusion_sdpa_cudnn_parity.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.attention_flash import (
    sdpa_flash_backward_native,
    sdpa_flash_train_fwd_native,
)
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_text_fusion_sdpa_cudnn_oracle.safetensors"


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


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


def _host_ref(st: SafeTensors, key: String, expected_shape: List[Int], ctx: DeviceContext) raises -> List[Float32]:
    var t = _load_bf16(st, key, expected_shape, ctx)
    return t.to_host(ctx)


def _check(
    mut harness: ParityHarness,
    st: SafeTensors,
    name: String,
    actual: Tensor,
    shape: List[Int],
    ok: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    var expected = _host_ref(st, name, shape.copy(), ctx)
    var r = harness.compare(actual, expected, ctx)
    print("  cos(", name, ") =", r.cos, " max_abs=", r.max_abs, " n=", r.n)
    if not r.passed:
        return False
    return ok


def _run_case[
    B: Int, S: Int, H: Int, Dh: Int
](st: SafeTensors, prefix: String, ctx: DeviceContext) raises -> Bool:
    var shape = _shape4(B, S, H, Dh)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    print("=== Krea2 TextFusion PyTorch-CUDNN SDPA case", prefix, "[", B, ",", S, ",", H, ",", Dh, "] ===")

    var q = _load_bf16(st, prefix + String(".q"), shape.copy(), ctx)
    var k = _load_bf16(st, prefix + String(".k"), shape.copy(), ctx)
    var v = _load_bf16(st, prefix + String(".v"), shape.copy(), ctx)
    var d_out = _load_bf16(st, prefix + String(".d_out"), shape.copy(), ctx)

    var fwd = sdpa_flash_train_fwd_native[B, S, H, Dh](q, k, v, scale, ctx)
    var bwd = sdpa_flash_backward_native[B, S, H, Dh](fwd, d_out, scale, ctx)
    ctx.synchronize()

    _require(fwd.o.dtype() == STDtype.BF16, prefix + String(".o dtype changed"))
    _require(bwd.d_q.dtype() == STDtype.BF16, prefix + String(".d_q dtype changed"))
    _require(bwd.d_k.dtype() == STDtype.BF16, prefix + String(".d_k dtype changed"))
    _require(bwd.d_v.dtype() == STDtype.BF16, prefix + String(".d_v dtype changed"))

    var harness = ParityHarness(0.9999)
    var ok = True
    ok = _check(harness, st, prefix + String(".o"), fwd.o, shape.copy(), ok, ctx)
    ok = _check(harness, st, prefix + String(".d_q"), bwd.d_q, shape.copy(), ok, ctx)
    ok = _check(harness, st, prefix + String(".d_k"), bwd.d_k, shape.copy(), ok, ctx)
    ok = _check(harness, st, prefix + String(".d_v"), bwd.d_v, shape.copy(), ok, ctx)
    return ok


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(ORACLE))
    _require(st.count() == 16, "Krea2 TextFusion SDPA oracle tensor count changed")
    var ok = True
    ok = _run_case[16, 12, 20, 128](st, String("layerwise"), ctx) and ok
    ok = _run_case[1, 16, 20, 128](st, String("refiner"), ctx) and ok
    if not ok:
        raise Error("Krea2 TextFusion PyTorch-CUDNN native SDPA parity failed")
    print("PASS: Krea2 TextFusion native SDPA matches PyTorch-CUDNN BF16 oracle")
