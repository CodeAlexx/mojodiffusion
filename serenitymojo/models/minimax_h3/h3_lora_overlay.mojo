# serenitymojo/models/minimax_h3/h3_lora_overlay.mojo
#
# MiniMax-H3 inference LoRA OVERLAY. Product inference keeps the quantized base
# weights untouched and applies y += multiplier·(alpha/rank)·B(A(x)) after
# each base projection, exactly like training. This is important: folding the
# small LoRA delta into BF16 and then groupwise-INT8 quantizing W+delta erased
# most of a real trained adapter. apply_raw/delta_raw remain compatibility and
# parity surfaces; product pipelines use activation_delta.
#
# New trainer artifacts use the stack's canonical PEFT/AI-Toolkit-style keys:
#   diffusion_model.blocks.{i}.<module>.lora_A.weight [rank, in]
#   diffusion_model.blocks.{i}.<module>.lora_B.weight [out, rank]
# Legacy Musubi/Kohya lora_unet_* down/up/alpha files remain loadable.
#
# Product math is the training formula y += scale·B(A(x)) in BF16 activation
# space. The raw-weight helpers remain only for non-QKV compatibility gates;
# product inference never calls them.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, gather_rows, mul_scalar, slice
from serenitymojo.models.minimax_h3.h3_lora_format import (
    h3_lora_peft_prefix,
    h3_lora_bare_peft_prefix,
    h3_lora_token_refiner_peft_prefix,
    h3_lora_token_refiner_bare_peft_prefix,
    h3_lora_legacy_prefix,
)

comptime TArc = ArcPointer[Tensor]
comptime H3_OVERLAY_BLOCKS = 50
comptime H3_OVERLAY_SLOTS = 4
comptime H3_OVERLAY_REFINER_BLOCKS = 2
comptime H3_OVERLAY_MAIN_COUNT = H3_OVERLAY_BLOCKS * H3_OVERLAY_SLOTS
comptime H3_OVERLAY_TOTAL_COUNT = (
    H3_OVERLAY_MAIN_COUNT + H3_OVERLAY_REFINER_BLOCKS * H3_OVERLAY_SLOTS
)


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


struct H3LoraOverlay(Copyable, Movable):
    """Parsed adapter set in BF16, the training/runtime compute dtype.

    `up` retains canonical module-output row order for compatibility gates.
    QKV LoRA is already `[q_all;k_all;v_all]`, so it is unchanged. `runtime_up`
    only swaps FC1 from the trainer/Musubi `[gate;value]` convention to the
    product block's `[value;gate]` convention. Other slots share the same
    ArcPointer. Missing slots carry scale 0 and no tensors.
    """
    var down: List[Optional[TArc]]        # [rank, in] BF16
    var up: List[Optional[TArc]]          # [raw_out, rank] BF16
    var runtime_up: List[Optional[TArc]]  # [runtime_out, rank] BF16
    var scale: List[Float32]
    var adapters: Int

    def __init__(
        out self,
        var down: List[Optional[TArc]], var up: List[Optional[TArc]],
        var runtime_up: List[Optional[TArc]],
        var scale: List[Float32], adapters: Int,
    ):
        self.down = down^
        self.up = up^
        self.runtime_up = runtime_up^
        self.scale = scale^
        self.adapters = adapters

    @staticmethod
    def load(
        path: String, multiplier: Float32, ctx: DeviceContext
    ) raises -> H3LoraOverlay:
        var st = SafeTensors.open(path)
        var down = List[Optional[TArc]]()
        var up = List[Optional[TArc]]()
        var runtime_up = List[Optional[TArc]]()
        var scale = List[Float32]()
        for _ in range(H3_OVERLAY_TOTAL_COUNT):
            down.append(Optional[TArc](None))
            up.append(Optional[TArc](None))
            runtime_up.append(Optional[TArc](None))
            scale.append(Float32(0.0))
        var n = 0
        for b in range(H3_OVERLAY_BLOCKS):
            for s in range(H3_OVERLAY_SLOTS):
                var canonical = h3_lora_peft_prefix(b, s)
                var bare = h3_lora_bare_peft_prefix(b, s)
                var legacy = h3_lora_legacy_prefix(b, s)
                var base = String("")
                var dkey = String("")
                var ukey = String("")
                var alpha_key = String("")

                var canonical_a = st.has_tensor(canonical + ".lora_A.weight")
                var canonical_b = st.has_tensor(canonical + ".lora_B.weight")
                if canonical_a and not canonical_b or canonical_b and not canonical_a:
                    raise Error("h3 overlay: incomplete PEFT A/B pair at " + canonical)
                var bare_a = st.has_tensor(bare + ".lora_A.weight")
                var bare_b = st.has_tensor(bare + ".lora_B.weight")
                if bare_a and not bare_b or bare_b and not bare_a:
                    raise Error("h3 overlay: incomplete bare PEFT A/B pair at " + bare)
                var legacy_down = st.has_tensor(legacy + ".lora_down.weight")
                var legacy_up = st.has_tensor(legacy + ".lora_up.weight")
                if legacy_down and not legacy_up or legacy_up and not legacy_down:
                    raise Error("h3 overlay: incomplete legacy down/up pair at " + legacy)

                if canonical_a:
                    base = canonical
                    dkey = canonical + ".lora_A.weight"
                    ukey = canonical + ".lora_B.weight"
                    alpha_key = canonical + ".alpha"
                elif bare_a:
                    base = bare
                    dkey = bare + ".lora_A.weight"
                    ukey = bare + ".lora_B.weight"
                    alpha_key = bare + ".alpha"
                elif legacy_down:
                    base = legacy
                    dkey = legacy + ".lora_down.weight"
                    ukey = legacy + ".lora_up.weight"
                    alpha_key = legacy + ".alpha"
                else:
                    continue

                var d = _bf16(st, dkey, ctx)
                var u = _bf16(st, ukey, ctx)
                var d_shape = d.shape()
                var u_shape = u.shape()
                if len(d_shape) != 2 or len(u_shape) != 2:
                    raise Error("h3 overlay: A/B tensors must be 2-D at " + base)
                var rank = d_shape[0]
                if rank <= 0 or u_shape[1] != rank:
                    raise Error("h3 overlay: down/up rank mismatch at " + base)
                var alpha = Float32(rank)  # upstream default when .alpha absent
                if st.has_tensor(alpha_key):
                    var ah = _f32(st, alpha_key, ctx).to_host(ctx)
                    alpha = Float32(ah[0])
                var idx = b * H3_OVERLAY_SLOTS + s
                var d_arc = TArc(d^)
                var u_arc = TArc(u^)
                down[idx] = Optional[TArc](d_arc^)
                if s == 2:
                    var transformed = _runtime_up_bf16(u_arc[], s, ctx)
                    up[idx] = Optional[TArc](u_arc^)
                    runtime_up[idx] = Optional[TArc](TArc(transformed^))
                else:
                    up[idx] = Optional[TArc](u_arc.copy())
                    runtime_up[idx] = Optional[TArc](u_arc^)
                scale[idx] = multiplier * alpha / Float32(rank)
                n += 1
        for b in range(H3_OVERLAY_REFINER_BLOCKS):
            for s in range(H3_OVERLAY_SLOTS):
                var canonical = h3_lora_token_refiner_peft_prefix(b, s)
                var bare = h3_lora_token_refiner_bare_peft_prefix(b, s)
                var base = String("")
                var dkey = String("")
                var ukey = String("")
                var alpha_key = String("")

                var canonical_a = st.has_tensor(canonical + ".lora_A.weight")
                var canonical_b = st.has_tensor(canonical + ".lora_B.weight")
                if canonical_a and not canonical_b or canonical_b and not canonical_a:
                    raise Error("h3 overlay: incomplete PEFT A/B pair at " + canonical)
                var bare_a = st.has_tensor(bare + ".lora_A.weight")
                var bare_b = st.has_tensor(bare + ".lora_B.weight")
                if bare_a and not bare_b or bare_b and not bare_a:
                    raise Error("h3 overlay: incomplete bare PEFT A/B pair at " + bare)
                if canonical_a:
                    base = canonical
                    dkey = canonical + ".lora_A.weight"
                    ukey = canonical + ".lora_B.weight"
                    alpha_key = canonical + ".alpha"
                elif bare_a:
                    base = bare
                    dkey = bare + ".lora_A.weight"
                    ukey = bare + ".lora_B.weight"
                    alpha_key = bare + ".alpha"
                else:
                    continue

                var d = _bf16(st, dkey, ctx)
                var u = _bf16(st, ukey, ctx)
                var d_shape = d.shape()
                var u_shape = u.shape()
                if len(d_shape) != 2 or len(u_shape) != 2:
                    raise Error("h3 overlay: A/B tensors must be 2-D at " + base)
                var rank = d_shape[0]
                if rank <= 0 or u_shape[1] != rank:
                    raise Error("h3 overlay: down/up rank mismatch at " + base)
                var alpha = Float32(rank)
                if st.has_tensor(alpha_key):
                    var ah = _f32(st, alpha_key, ctx).to_host(ctx)
                    alpha = Float32(ah[0])
                var idx = H3_OVERLAY_MAIN_COUNT + b * H3_OVERLAY_SLOTS + s
                var d_arc = TArc(d^)
                var u_arc = TArc(u^)
                down[idx] = Optional[TArc](d_arc^)
                if s == 2:
                    var transformed = _runtime_up_bf16(u_arc[], s, ctx)
                    up[idx] = Optional[TArc](u_arc^)
                    runtime_up[idx] = Optional[TArc](TArc(transformed^))
                else:
                    up[idx] = Optional[TArc](u_arc.copy())
                    runtime_up[idx] = Optional[TArc](u_arc^)
                scale[idx] = multiplier * alpha / Float32(rank)
                n += 1
        if n == 0:
            raise Error(
                "h3 overlay: no main-block/token-refiner PEFT or legacy "
                "lora_unet_blocks_* adapters in " + path
            )
        return H3LoraOverlay(down^, up^, runtime_up^, scale^, n)

    def has(self, layer: Int, slot: Int) -> Bool:
        if slot < 0 or layer < 0 or layer >= H3_OVERLAY_BLOCKS:
            return False
        return Bool(self.down[layer * H3_OVERLAY_SLOTS + slot])

    def has_layer(self, layer: Int) -> Bool:
        if layer < 0 or layer >= H3_OVERLAY_BLOCKS:
            return False
        for slot in range(H3_OVERLAY_SLOTS):
            if self.has(layer, slot):
                return True
        return False

    def has_refiner(self, layer: Int, slot: Int) -> Bool:
        if (
            slot < 0 or layer < 0
            or layer >= H3_OVERLAY_REFINER_BLOCKS
        ):
            return False
        return Bool(self.down[
            H3_OVERLAY_MAIN_COUNT + layer * H3_OVERLAY_SLOTS + slot
        ])

    def activation_delta(
        self, layer: Int, slot: Int, x: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """Scaled B(A(x)) in transformed runtime row order."""
        var idx = layer * H3_OVERLAY_SLOTS + slot
        var no_bias = Optional[Tensor](None)
        var hidden = linear(x, self.down[idx].value()[], no_bias^, ctx)
        var delta = linear(
            hidden, self.runtime_up[idx].value()[], no_bias^, ctx
        )
        return mul_scalar(delta, self.scale[idx], ctx)

    def activation_delta_rows(
        self, layer: Int, slot: Int, x: Tensor,
        row_start: Int, row_count: Int, ctx: DeviceContext,
    ) raises -> Tensor:
        """A row slice of B(A(x)); used by memory-bounded split projections."""
        var idx = layer * H3_OVERLAY_SLOTS + slot
        var b = slice(
            self.runtime_up[idx].value()[], 0, row_start, row_count, ctx
        )
        var no_bias = Optional[Tensor](None)
        var hidden = linear(x, self.down[idx].value()[], no_bias^, ctx)
        var delta = linear(hidden, b, no_bias^, ctx)
        return mul_scalar(delta, self.scale[idx], ctx)

    def refiner_activation_delta(
        self, layer: Int, slot: Int, x: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """Scaled B(A(x)) for a token-refiner projection."""
        var idx = H3_OVERLAY_MAIN_COUNT + layer * H3_OVERLAY_SLOTS + slot
        var no_bias = Optional[Tensor](None)
        var hidden = linear(x, self.down[idx].value()[], no_bias^, ctx)
        var delta = linear(
            hidden, self.runtime_up[idx].value()[], no_bias^, ctx
        )
        return mul_scalar(delta, self.scale[idx], ctx)

    def delta_raw(
        self, layer: Int, slot: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """Scaled delta alone (bf16 [out, in]): mult·(alpha/rank)·(up@down).
        For build paths that add into an existing staged weight (e.g. the
        int8 resident store) before quantization."""
        if slot == 0:
            raise Error(
                "h3 overlay: raw QKV merge is disabled; canonical LoRA B is "
                "[q_all;k_all;v_all] while checkpoint base rows are per-head"
            )
        var idx = layer * H3_OVERLAY_SLOTS + slot
        var no_bias = Optional[Tensor](None)
        # bf16 GEMM (F32-accumulated) keeps the transient at one bf16
        # [out,in] instead of two F32 copies — the fat F32 chain OOM'd
        # next to a fully built resident store.
        var delta = linear(
            self.up[idx].value()[], self.down[idx].value()[], no_bias^, ctx,
            transpose_b=False,
        )
        return mul_scalar(delta, self.scale[idx], ctx)

    def apply_raw(
        self, layer: Int, slot: Int, w_raw: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        """W' (bf16) for a RAW-layout block weight. Caller checks has()."""
        if slot == 0:
            raise Error(
                "h3 overlay: raw QKV merge is disabled; use activation_delta"
            )
        var idx = layer * H3_OVERLAY_SLOTS + slot
        var d = cast_tensor(self.down[idx].value()[], STDtype.F32, ctx)
        var u = cast_tensor(self.up[idx].value()[], STDtype.F32, ctx)
        var no_bias = Optional[Tensor](None)
        # delta = up @ down  (F32 GEMM): x [out, r] @ weight [r, in]
        var delta = linear(u, d, no_bias^, ctx, transpose_b=False)
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


def _bf16(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var t = Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )
    if t.dtype() == STDtype.BF16:
        return t^
    return cast_tensor(t, STDtype.BF16, ctx)


def _runtime_up_bf16(up: Tensor, slot: Int, ctx: DeviceContext) raises -> Tensor:
    """Transform only FC1 B rows to the product block's [value;gate] order.

    QKV B is the output of the logical qkv_proj module and is already
    [q_all;k_all;v_all], independent of the released base checkpoint's packed
    storage. A addresses unchanged projection inputs and is never permuted.
    """
    var shape = up.shape()
    if len(shape) != 2:
        raise Error("h3 overlay: B must be rank-2")
    var ids = List[Int]()
    if slot == 2:
        comptime ffn = 14336
        if shape[0] != 2 * ffn:
            raise Error("h3 overlay: fc1 B output does not match released H3")
        for row in range(ffn):
            ids.append(ffn + row)
        for row in range(ffn):
            ids.append(row)
    else:
        return up.clone(ctx)
    return gather_rows(up, ids, ctx)
