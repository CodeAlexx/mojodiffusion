# serenitymojo/models/minimax_h3/h3_lora_overlay.mojo
#
# MiniMax-H3 inference LoRA OVERLAY: W' = W + multiplier·(alpha/rank)·(up@down)
# applied to the RAW checkpoint block weights at load time — LoRA is ADDED as
# an overlay, NEVER fused into a saved model (house rule). Applies BEFORE the
# loader's qkv-deinterleave / fc1-swap transforms, because trained adapters
# live in the RAW layout (the trainer's convention).
#
# File format: the trainer's own save (upstream-standard keys) —
#   lora_unet_blocks_{i}_{attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2}
#     .lora_down.weight [rank, in]   .lora_up.weight [out, rank]   .alpha [1]
#
# Overlay math (same formula the reference applies at runtime): delta in F32
# (up_f32 @ down_f32, scaled), added to the F32 upcast of W, ONE rounding
# back to bf16.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, mul_scalar

comptime TArc = ArcPointer[Tensor]
comptime H3_OVERLAY_BLOCKS = 50
comptime H3_OVERLAY_SLOTS = 4


def h3_overlay_slot_of(raw_suffix: String) -> Int:
    """RAW block tensor suffix -> overlay slot (-1 = never adapted)."""
    if raw_suffix == String("attn.qkv_proj.weight"):
        return 0
    if raw_suffix == String("attn.out_proj.weight"):
        return 1
    if raw_suffix == String("mlp.fc1.weight"):
        return 2
    if raw_suffix == String("mlp.fc2.weight"):
        return 3
    return -1


def _slot_key(slot: Int) raises -> String:
    if slot == 0:
        return String("attn_qkv_proj")
    if slot == 1:
        return String("attn_out_proj")
    if slot == 2:
        return String("mlp_fc1")
    if slot == 3:
        return String("mlp_fc2")
    raise Error("h3 overlay: bad slot")


struct H3LoraOverlay(Copyable, Movable):
    """Parsed adapter set. down/up are F32 device tensors; scale folds
    multiplier·(alpha/rank). Missing slots carry scale 0 and no tensors."""
    var down: List[Optional[TArc]]  # [rank, in] F32
    var up: List[Optional[TArc]]    # [out, rank] F32
    var scale: List[Float32]
    var adapters: Int

    def __init__(
        out self,
        var down: List[Optional[TArc]], var up: List[Optional[TArc]],
        var scale: List[Float32], adapters: Int,
    ):
        self.down = down^
        self.up = up^
        self.scale = scale^
        self.adapters = adapters

    @staticmethod
    def load(
        path: String, multiplier: Float32, ctx: DeviceContext
    ) raises -> H3LoraOverlay:
        var st = SafeTensors.open(path)
        var down = List[Optional[TArc]]()
        var up = List[Optional[TArc]]()
        var scale = List[Float32]()
        for _ in range(H3_OVERLAY_BLOCKS * H3_OVERLAY_SLOTS):
            down.append(Optional[TArc](None))
            up.append(Optional[TArc](None))
            scale.append(Float32(0.0))
        var n = 0
        for b in range(H3_OVERLAY_BLOCKS):
            for s in range(H3_OVERLAY_SLOTS):
                var base = (
                    String("lora_unet_blocks_") + String(b) + "_" + _slot_key(s)
                )
                if not st.has_tensor(base + ".lora_down.weight"):
                    continue
                var d = _f32(st, base + ".lora_down.weight", ctx)
                var u = _f32(st, base + ".lora_up.weight", ctx)
                var rank = d.shape()[0]
                if u.shape()[1] != rank:
                    raise Error("h3 overlay: down/up rank mismatch at " + base)
                var alpha = Float32(rank)  # upstream default when .alpha absent
                if st.has_tensor(base + ".alpha"):
                    var ah = _f32(st, base + ".alpha", ctx).to_host(ctx)
                    alpha = Float32(ah[0])
                var idx = b * H3_OVERLAY_SLOTS + s
                down[idx] = Optional[TArc](TArc(d^))
                up[idx] = Optional[TArc](TArc(u^))
                scale[idx] = multiplier * alpha / Float32(rank)
                n += 1
        if n == 0:
            raise Error("h3 overlay: no lora_unet_blocks_* adapters in " + path)
        return H3LoraOverlay(down^, up^, scale^, n)

    def has(self, layer: Int, slot: Int) -> Bool:
        if slot < 0 or layer < 0 or layer >= H3_OVERLAY_BLOCKS:
            return False
        return Bool(self.down[layer * H3_OVERLAY_SLOTS + slot])

    def apply_raw(
        self, layer: Int, slot: Int, w_raw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """W' (bf16) for a RAW-layout block weight. Caller checks has()."""
        var idx = layer * H3_OVERLAY_SLOTS + slot
        var d = self.down[idx].value()
        var u = self.up[idx].value()
        var no_bias = Optional[Tensor](None)
        # delta = up @ down  (F32 GEMM): x [out, r] @ weight [r, in]
        var delta = linear(u[], d[], no_bias^, ctx, transpose_b=False)
        var scaled = mul_scalar(delta, self.scale[idx], ctx)
        var w32 = cast_tensor(w_raw, STDtype.F32, ctx)
        var sum = add(w32, scaled, ctx)
        return cast_tensor(sum, STDtype.BF16, ctx)


def _f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var t = Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )
    if t.dtype() == STDtype.F32:
        return t^
    return cast_tensor(t, STDtype.F32, ctx)
