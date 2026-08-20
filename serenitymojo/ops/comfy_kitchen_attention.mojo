# Direct Mojo surface for Comfy Kitchen's Ampere INT8 tensor-core attention.
#
# BF16 [B,S,H,128] Q/K/V enter a same-stream C ABI bridge. The upstream CUDA
# launchers apply a fixed H128 Hadamard rotation to Q/K, per-thread signed-INT8
# quantization, per-channel signed-INT8 V quantization, and unsigned-INT8 P x
# signed-INT8 V tensor-core attention with FP32 online-softmax accumulation.
# The returned BF16 tensor preserves Mojo's [B,S,H,D] layout. No Python code or
# tensor wrapper runs in the hot path.

from std.ffi import external_call
from std.math import ceildiv
from std.memory import ArcPointer
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host._nvidia_cuda import CUDA

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr
from serenitymojo.tensor import Tensor


comptime _HEAD_DIM = 128
comptime _CTA_Q = 128
comptime _CTA_K = 128
comptime _Q_SCALES_PER_CTA = 32
comptime _K_SCALES_PER_CTA = 4


def _ck_ptr(t: Tensor) -> BytePtr:
    return BytePtr(unsafe_from_address=Int(t.buf.unsafe_ptr()))


def comfy_kitchen_attention_available() -> Bool:
    """True when the configured Comfy Kitchen CUDA launcher DSO is usable."""
    return Int(
        external_call["serenity_comfy_kitchen_available", Int32]()
    ) == 1


struct ComfyKitchenAttentionScratch(Copyable, Movable):
    """Preallocated, zero-device-allocation scratch for H3 attention.

    Output is scratch-backed just like ``SageInt8Scratch``: callers must
    consume it on the same stream before the next call overwrites it.
    """

    var q8: ArcPointer[Tensor]
    var k8: ArcPointer[Tensor]
    var qs: ArcPointer[Tensor]
    var ks: ArcPointer[Tensor]
    var v8: ArcPointer[Tensor]
    var vs: ArcPointer[Tensor]
    var anchor: ArcPointer[Tensor]
    var out: ArcPointer[Tensor]
    var max_s: Int
    var max_s_pad: Int
    var heads: Int

    def __init__(
        out self,
        max_s: Int,
        heads: Int,
        ctx: DeviceContext,
    ) raises:
        if max_s <= 0 or heads <= 0:
            raise Error(
                "ComfyKitchenAttentionScratch requires positive max_s and heads"
            )
        var elems = heads * max_s * _HEAD_DIM
        self.max_s_pad = ceildiv(max_s, _CTA_K) * _CTA_K
        var qscale_elems = (
            heads * ceildiv(max_s, _CTA_Q) * _Q_SCALES_PER_CTA
        )
        var kscale_elems = (
            heads * ceildiv(max_s, _CTA_K) * _K_SCALES_PER_CTA
        )
        var v8_elems = heads * _HEAD_DIM * self.max_s_pad

        var q8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
        var k8_buf = ctx.enqueue_create_buffer[DType.uint8](elems)
        var qs_buf = ctx.enqueue_create_buffer[DType.uint8](qscale_elems * 4)
        var ks_buf = ctx.enqueue_create_buffer[DType.uint8](kscale_elems * 4)
        var v8_buf = ctx.enqueue_create_buffer[DType.uint8](v8_elems)
        var vs_buf = ctx.enqueue_create_buffer[DType.uint8](
            heads * _HEAD_DIM * 4
        )
        var anchor_buf = ctx.enqueue_create_buffer[DType.uint8](heads * 4)
        var out_buf = ctx.enqueue_create_buffer[DType.uint8](elems * 2)
        self.q8 = ArcPointer(Tensor(q8_buf^, [elems], STDtype.I8))
        self.k8 = ArcPointer(Tensor(k8_buf^, [elems], STDtype.I8))
        self.qs = ArcPointer(
            Tensor(qs_buf^, [qscale_elems], STDtype.F32)
        )
        self.ks = ArcPointer(
            Tensor(ks_buf^, [kscale_elems], STDtype.F32)
        )
        self.v8 = ArcPointer(Tensor(v8_buf^, [v8_elems], STDtype.I8))
        self.vs = ArcPointer(
            Tensor(vs_buf^, [heads * _HEAD_DIM], STDtype.F32)
        )
        self.anchor = ArcPointer(
            Tensor(anchor_buf^, [heads], STDtype.I32)
        )
        self.out = ArcPointer(Tensor(out_buf^, [elems], STDtype.BF16))
        self.max_s = max_s
        self.heads = heads

    def resident_bytes(self) -> Int:
        return (
            self.q8[].nbytes() + self.k8[].nbytes()
            + self.qs[].nbytes() + self.ks[].nbytes()
            + self.v8[].nbytes() + self.vs[].nbytes()
            + self.anchor[].nbytes() + self.out[].nbytes()
        )


def comfy_kitchen_attention_fwd_scratch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float32,
    scratch: ComfyKitchenAttentionScratch,
    ctx: DeviceContext,
) raises -> Tensor:
    """Run the Comfy Kitchen INT8 attention launch chain on MAX's stream."""
    if q.dtype() != STDtype.BF16 or k.dtype() != STDtype.BF16 \
            or v.dtype() != STDtype.BF16:
        raise Error("comfy_kitchen_attention requires BF16 Q/K/V")
    var shape = q.shape()
    if len(shape) != 4 or k.shape() != shape or v.shape() != shape:
        raise Error("comfy_kitchen_attention Q/K/V shape mismatch")
    var B = shape[0]
    var S = shape[1]
    var H = shape[2]
    var D = shape[3]
    if B != 1 or D != _HEAD_DIM:
        raise Error("comfy_kitchen_attention currently requires B=1,D=128")
    if H != scratch.heads or S > scratch.max_s:
        raise Error("comfy_kitchen_attention exceeds scratch geometry")
    if not comfy_kitchen_attention_available():
        raise Error(
            "Comfy Kitchen CUDA backend unavailable; set "
            "SERENITY_COMFY_KITCHEN_CUDA to its _C.abi3.so"
        )

    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["serenity_comfy_kitchen_sage_bf16", Int32](
        _ck_ptr(q), _ck_ptr(k), _ck_ptr(v), _ck_ptr(scratch.out[]),
        _ck_ptr(scratch.q8[]), _ck_ptr(scratch.qs[]),
        _ck_ptr(scratch.k8[]), _ck_ptr(scratch.ks[]),
        _ck_ptr(scratch.v8[]), _ck_ptr(scratch.vs[]),
        _ck_ptr(scratch.anchor[]),
        Int32(B), Int32(S), Int32(H), Int32(D), scale, stream,
    ))
    if rc != 0:
        raise Error(
            String("comfy_kitchen_attention launcher rc=") + String(rc)
            + String(" (B=") + String(B) + String(" S=") + String(S)
            + String(" H=") + String(H) + String(" D=") + String(D)
            + String(")")
        )

    var elems = B * S * H * D
    var out_view = DeviceBuffer[DType.uint8](
        ctx, scratch.out[].buf.unsafe_ptr(), elems * 2, owning=False
    )
    return Tensor(out_view^, shape^, STDtype.BF16)
