# models/klein/parity/klein_direct_double_block_identity_smoke.mojo
#
# Direct DoRA/OFT double-block plumbing gate. With no direct slots present, the
# new direct forward must match the established no-adapter LoRA block path.

from std.collections import List, Optional
from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights,
    ModVecs, modvecs_to_device,
    StreamLoraDevice, DoubleBlockLoraDevice,
    double_block_lora_forward_device_resident_scratch,
    double_block_lora_backward_device_resident_scratch,
    double_block_direct_dora_forward_device_resident_scratch,
    double_block_direct_dora_backward_device_resident_scratch,
    double_block_direct_oft_forward_device_resident_scratch,
    double_block_direct_oft_backward_device_resident_scratch,
)
from serenitymojo.models.klein.klein_direct_lycoris_stack import (
    KleinStreamDirectDoRA, KleinDoubleDirectDoRA,
    KleinStreamDirectOFT, KleinDoubleDirectOFT, KleinDirectOFTDeviceSlot,
)
from serenitymojo.training.dora_substitution_device import DoRAAdapterDevice


comptime TArc = ArcPointer[Tensor]
comptime H = 2
comptime Dh = 128
comptime D = H * Dh
comptime F = 512
comptime N_IMG = 64
comptime N_TXT = 64
comptime S = N_IMG + N_TXT
comptime EPS = Float32(1.0e-6)
comptime COS_BAR = 0.999999
comptime NREL_BAR = 2.0e-5


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32 = Float32(0.0)) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale + bias)
    return out^


def _fill(n: Int, value: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(value)
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("nrel: len mismatch")
    var d = 0.0
    var n = 0.0
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _check(name: String, got: List[Float32], expected: List[Float32]) raises:
    var c = _cos(got, expected)
    var n = _nrel(got, expected)
    print("  ", name, " cos=", c, " nrel=", n)
    if c < COS_BAR or n > NREL_BAR:
        raise Error(String("identity mismatch: ") + name)


def _stream_weights(seed: UInt64, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _randn(3 * D * D, seed + 1, 0.02),
        _randn(D * D, seed + 2, 0.02),
        _randn(2 * F * D, seed + 3, 0.02),
        _randn(D * F, seed + 4, 0.02),
        _randn(Dh, seed + 5, 0.02, Float32(1.0)),
        _randn(Dh, seed + 6, 0.02, Float32(1.0)),
        D, F, Dh, ctx,
    )


def _mod(seed: UInt64) -> ModVecs:
    return ModVecs(
        _randn(D, seed + 1, 0.02), _randn(D, seed + 2, 0.02),
        _randn(D, seed + 3, 0.02), _randn(D, seed + 4, 0.02),
        _randn(D, seed + 5, 0.02), _randn(D, seed + 6, 0.02),
    )


def _empty_lora_stream() -> StreamLoraDevice:
    return StreamLoraDevice(
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None),
    )


def _empty_dora_stream() -> KleinStreamDirectDoRA:
    return KleinStreamDirectDoRA(
        Optional[DoRAAdapterDevice](None),
        Optional[DoRAAdapterDevice](None),
        Optional[DoRAAdapterDevice](None),
        Optional[DoRAAdapterDevice](None),
        Optional[DoRAAdapterDevice](None),
        Optional[DoRAAdapterDevice](None),
    )


def _empty_oft_stream() -> KleinStreamDirectOFT:
    return KleinStreamDirectOFT(
        Optional[KleinDirectOFTDeviceSlot](None),
        Optional[KleinDirectOFTDeviceSlot](None),
        Optional[KleinDirectOFTDeviceSlot](None),
        Optional[KleinDirectOFTDeviceSlot](None),
        Optional[KleinDirectOFTDeviceSlot](None),
        Optional[KleinDirectOFTDeviceSlot](None),
    )


def main() raises:
    print("=== klein direct double-block identity smoke ===")
    var ctx = DeviceContext()
    var scratch = ScratchRingAllocator(ctx, 256 * 1024 * 1024, 2)
    var norm_ones = Tensor.from_host(_fill(D, Float32(1.0)), [D], STDtype.F32, ctx)
    var norm_zeros = Tensor.from_host(_fill(D, Float32(0.0)), [D], STDtype.F32, ctx)
    var cos = Tensor.from_host(_fill(S * H * (Dh // 2), Float32(1.0)), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_fill(S * H * (Dh // 2), Float32(0.0)), [S * H, Dh // 2], STDtype.F32, ctx)

    var img = TArc(Tensor.from_host(_randn(N_IMG * D, 10, 0.05), [N_IMG, D], STDtype.F32, ctx))
    var txt = TArc(Tensor.from_host(_randn(N_TXT * D, 20, 0.05), [N_TXT, D], STDtype.F32, ctx))
    var w = DoubleBlockWeights(_stream_weights(100, ctx), _stream_weights(200, ctx))
    var img_mod = modvecs_to_device(_mod(300), D, ctx)
    var txt_mod = modvecs_to_device(_mod(400), D, ctx)
    var lora = DoubleBlockLoraDevice(_empty_lora_stream(), _empty_lora_stream())
    var dora = KleinDoubleDirectDoRA(_empty_dora_stream(), _empty_dora_stream())
    var oft = KleinDoubleDirectOFT(_empty_oft_stream(), _empty_oft_stream())

    var lf = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
        img, txt, w, img_mod, txt_mod, lora, cos, sin,
        D, F, EPS, norm_ones, norm_zeros, ctx, scratch,
    )
    scratch.reset()
    var df = double_block_direct_dora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
        img, txt, w, img_mod, txt_mod, dora, cos, sin,
        D, F, EPS, norm_ones, norm_zeros, ctx, scratch,
    )
    scratch.reset()
    var of = double_block_direct_oft_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
        img, txt, w, img_mod, txt_mod, oft, cos, sin,
        D, F, EPS, norm_ones, norm_zeros, ctx, scratch,
    )

    _check("dora img", df.img_out[].to_host(ctx), lf.img_out[].to_host(ctx))
    _check("dora txt", df.txt_out[].to_host(ctx), lf.txt_out[].to_host(ctx))
    _check("oft img", of.img_out[].to_host(ctx), lf.img_out[].to_host(ctx))
    _check("oft txt", of.txt_out[].to_host(ctx), lf.txt_out[].to_host(ctx))

    var d_img = TArc(Tensor.from_host(_randn(N_IMG * D, 500, 0.03), [N_IMG, D], STDtype.F32, ctx))
    var d_txt = TArc(Tensor.from_host(_randn(N_TXT * D, 600, 0.03), [N_TXT, D], STDtype.F32, ctx))
    scratch.reset()
    var lb = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
        d_img, d_txt, w, img_mod, txt_mod, lora, lf.saved, cos, sin,
        D, F, EPS, norm_ones, ctx, scratch, False,
    )
    scratch.reset()
    var db = double_block_direct_dora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
        d_img, d_txt, w, img_mod, txt_mod, dora, df.saved, cos, sin,
        D, F, EPS, norm_ones, ctx, scratch, False,
    )
    scratch.reset()
    var ob = double_block_direct_oft_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S](
        d_img, d_txt, w, img_mod, txt_mod, oft, of.saved, cos, sin,
        D, F, EPS, norm_ones, ctx, scratch, False,
    )
    _check("dora img dx", db.img.d_x[].to_host(ctx), lb.img.d_x[].to_host(ctx))
    _check("dora txt dx", db.txt.d_x[].to_host(ctx), lb.txt.d_x[].to_host(ctx))
    _check("oft img dx", ob.img.d_x[].to_host(ctx), lb.img.d_x[].to_host(ctx))
    _check("oft txt dx", ob.txt.d_x[].to_host(ctx), lb.txt.d_x[].to_host(ctx))
    print("PASS -- Klein direct double-block no-slot forward/backward matches no-adapter LoRA path")
