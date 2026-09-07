# MiniMax-H3 Fun ControlNet-Union -- native inference runtime.
#
# The released Alibaba checkpoint is a five-block H3 side transformer.  It
# replaces only the target-video rows of the packed T2VA sequence, advances at
# base layers 0/10/20/30/40, zeros its audio residual rows, and adds the scaled
# residual to the base stream.  The checkpoint remains memory mapped on the
# host; one side block is materialized on the GPU at a time.

from std.collections import Dict, List, Optional
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    full_device,
    mul_scalar,
    reshape_owned,
    slice,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    minimax_h3_block_forward_dynamic,
    minimax_h3_block_prefix,
    minimax_h3_check_block_weights,
)
from serenitymojo.ops.sage_attention_int8 import SageInt8Scratch
from serenitymojo.ops.comfy_kitchen_attention import ComfyKitchenAttentionScratch
from serenitymojo.ops.evg_attention_int8 import EVGH3RaggedLayout


comptime TArc = ArcPointer[Tensor]
comptime MINIMAX_H3_CONTROL_BLOCKS = 5
comptime MINIMAX_H3_CONTROL_INPUT_CHANNELS = 49
comptime MINIMAX_H3_CONTROL_PATCH_VOLUME = 4
comptime MINIMAX_H3_CONTROL_PATCH_DIM = (
    MINIMAX_H3_CONTROL_INPUT_CHANNELS * MINIMAX_H3_CONTROL_PATCH_VOLUME
)


def minimax_h3_control_injection_layer(index: Int) raises -> Int:
    if index < 0 or index >= MINIMAX_H3_CONTROL_BLOCKS:
        raise Error("MiniMax-H3 ControlNet block index must be in [0,5)")
    return index * 10


def minimax_h3_control_block_prefix(index: Int) raises -> String:
    _ = minimax_h3_control_injection_layer(index)
    return String("control_blocks.") + String(index) + String(".")


def _control_load(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _control_check(
    st: SafeTensors,
    name: String,
    dtype: STDtype,
    shape: List[Int],
) raises:
    if not st.has_tensor(name):
        raise Error(String("MiniMax-H3 ControlNet missing tensor ") + name)
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("MiniMax-H3 ControlNet tensor ") + name + String(" has dtype ")
            + info.dtype.name() + String(", expected ") + dtype.name()
        )
    if info.shape != shape:
        raise Error(
            String("MiniMax-H3 ControlNet tensor ") + name
            + String(" has the wrong shape")
        )


def minimax_h3_controlnet_preflight(
    path: String, config: MiniMaxH3DiTConfig
) raises:
    """Header-only validation of the official dense Union checkpoint."""
    config.validate()
    var st = SafeTensors.open(path)
    var names = st.names()
    for ref name in names:
        var info = st.tensor_info(name)
        if (
            name.endswith(".comfy_quant")
            or name.endswith(".weight_scale")
            or name.endswith(".input_scale")
            or info.dtype == STDtype.I8
            or info.dtype == STDtype.U8
        ):
            raise Error(
                String("MiniMax-H3 ControlNet packed/quantized tensor ") + name
                + String(" is unsupported; install the official dense Union checkpoint")
            )

    var hidden = config.hidden_size
    var inner = config.inner_dim()
    var ffn = config.ffn_hidden_size
    var adaln = config.adaln_out_features
    var temb = config.time_embed_dim
    _control_check(
        st, String("control_proj_in.weight"), STDtype.F32,
        [hidden, MINIMAX_H3_CONTROL_PATCH_DIM],
    )
    _control_check(
        st, String("control_proj_in.bias"), STDtype.F32, [hidden]
    )
    for index in range(MINIMAX_H3_CONTROL_BLOCKS):
        var p = minimax_h3_control_block_prefix(index)
        _control_check(st, p + "adaln_proj.linear.weight", STDtype.BF16, [adaln, temb])
        _control_check(st, p + "adaln_proj.linear.bias", STDtype.BF16, [adaln])
        _control_check(st, p + "norm1.weight", STDtype.BF16, [hidden])
        _control_check(st, p + "norm2.weight", STDtype.BF16, [hidden])
        _control_check(st, p + "attn.norm_q.weight", STDtype.BF16, [config.attention_head_dim])
        _control_check(st, p + "attn.norm_k.weight", STDtype.BF16, [config.attention_head_dim])
        _control_check(st, p + "attn.to_q.weight", STDtype.BF16, [inner, hidden])
        _control_check(st, p + "attn.to_k.weight", STDtype.BF16, [inner, hidden])
        _control_check(st, p + "attn.to_v.weight", STDtype.BF16, [inner, hidden])
        _control_check(st, p + "attn.to_out.0.weight", STDtype.BF16, [hidden, inner])
        _control_check(st, p + "ff.net.0.proj.weight", STDtype.BF16, [2 * ffn, hidden])
        _control_check(st, p + "ff.net.2.weight", STDtype.BF16, [hidden, ffn])
        _control_check(st, p + "after_proj.weight", STDtype.BF16, [hidden, hidden])
        _control_check(st, p + "after_proj.bias", STDtype.BF16, [hidden])
        if index == 0:
            _control_check(st, p + "before_proj.weight", STDtype.BF16, [hidden, hidden])
            _control_check(st, p + "before_proj.bias", STDtype.BF16, [hidden])
    if st.has_tensor(String("control_blocks.5.after_proj.weight")):
        raise Error("MiniMax-H3 ControlNet checkpoint has more than five side blocks")


@fieldwise_init
struct MiniMaxH3ControlInput(Copyable, Movable):
    """One prepared `[target_video_rows,196]` F32 guide and its schedule."""

    var rows: TArc
    var strength: Float32
    var start_percent: Float32
    var end_percent: Float32


@fieldwise_init
struct MiniMaxH3ControlModCache(Movable):
    var block_mod: List[TArc]
    var distinct_timesteps: Int

    def total_bytes(self) -> Int:
        var total = 0
        for i in range(len(self.block_mod)):
            total += self.block_mod[i][].nbytes()
        return total


struct MiniMaxH3ControlRuntime(Movable):
    """Request-owned checkpoint mapping, resident projections, and inputs."""

    var checkpoint: SafeTensors
    var proj_w: TArc
    var proj_b: TArc
    var before_w: TArc
    var before_b: TArc
    var modcache: MiniMaxH3ControlModCache
    var inputs: List[MiniMaxH3ControlInput]

    def __init__(
        out self,
        var checkpoint: SafeTensors,
        var proj_w: Tensor,
        var proj_b: Tensor,
        var before_w: Tensor,
        var before_b: Tensor,
        var modcache: MiniMaxH3ControlModCache,
        var inputs: List[MiniMaxH3ControlInput],
    ):
        self.checkpoint = checkpoint^
        self.proj_w = TArc(proj_w^)
        self.proj_b = TArc(proj_b^)
        self.before_w = TArc(before_w^)
        self.before_b = TArc(before_b^)
        self.modcache = modcache^
        self.inputs = inputs^

    def active(self, progress: Float32) -> Bool:
        for i in range(len(self.inputs)):
            if (
                self.inputs[i].strength != Float32(0.0)
                and self.inputs[i].start_percent <= progress
                and progress <= self.inputs[i].end_percent
            ):
                return True
        return False


def _control_build_modcache(
    st: SafeTensors,
    temb: Tensor,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3ControlModCache:
    var shape = temb.shape()
    if (
        len(shape) != 2
        or shape[0] <= 0
        or shape[1] != config.time_embed_dim
        or temb.dtype() != STDtype.F32
    ):
        raise Error("MiniMax-H3 ControlNet temb must be [N,2688] F32")
    var activated_f32 = silu(temb, ctx)
    var activated = cast_tensor(activated_f32, STDtype.BF16, ctx)
    var block_mod = List[TArc]()
    for index in range(MINIMAX_H3_CONTROL_BLOCKS):
        var p = minimax_h3_control_block_prefix(index)
        var w = _control_load(st, p + "adaln_proj.linear.weight", ctx)
        var b = _control_load(st, p + "adaln_proj.linear.bias", ctx)
        var wide = linear_bias(activated, w, b, ctx)
        var mod = reshape_owned(
            wide^, [shape[0] * config.adaln_rows_per_timestep(), 6 * config.hidden_size]
        )
        block_mod.append(TArc(mod^))
        ctx.synchronize()
        st.release_tensor(p + "adaln_proj.linear.weight")
        st.release_tensor(p + "adaln_proj.linear.bias")
    return MiniMaxH3ControlModCache(block_mod^, shape[0])


def minimax_h3_control_runtime(
    path: String,
    temb: Tensor,
    config: MiniMaxH3DiTConfig,
    var inputs: List[MiniMaxH3ControlInput],
    ctx: DeviceContext,
) raises -> MiniMaxH3ControlRuntime:
    if len(inputs) < 1 or len(inputs) > 4:
        raise Error("MiniMax-H3 ControlNet requires from one through four controls")
    minimax_h3_controlnet_preflight(path, config)
    for i in range(len(inputs)):
        var rs = inputs[i].rows[].shape()
        if (
            len(rs) != 2
            or rs[0] <= 0
            or rs[1] != MINIMAX_H3_CONTROL_PATCH_DIM
            or inputs[i].rows[].dtype() != STDtype.F32
        ):
            raise Error("MiniMax-H3 ControlNet rows must be [Nv,196] F32")
        if (
            inputs[i].start_percent < Float32(0.0)
            or inputs[i].end_percent > Float32(1.0)
            or inputs[i].start_percent > inputs[i].end_percent
        ):
            raise Error("MiniMax-H3 ControlNet range must satisfy 0 <= start <= end <= 1")

    var st = SafeTensors.open(path)
    var proj_w = _control_load(st, String("control_proj_in.weight"), ctx)
    var proj_b = _control_load(st, String("control_proj_in.bias"), ctx)
    var p0 = minimax_h3_control_block_prefix(0)
    var before_w = _control_load(st, p0 + "before_proj.weight", ctx)
    var before_b = _control_load(st, p0 + "before_proj.bias", ctx)
    var modcache = _control_build_modcache(st, temb, config, ctx)
    ctx.synchronize()
    st.release_tensor(String("control_proj_in.weight"))
    st.release_tensor(String("control_proj_in.bias"))
    st.release_tensor(p0 + "before_proj.weight")
    st.release_tensor(p0 + "before_proj.bias")
    return MiniMaxH3ControlRuntime(
        st^, proj_w^, proj_b^, before_w^, before_b^, modcache^, inputs^
    )


def minimax_h3_control_init_hidden(
    hidden: Tensor,
    rows: Tensor,
    runtime: MiniMaxH3ControlRuntime,
    target_video_rows: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """Replace the packed video suffix, then `before_proj(c) + hidden`."""
    var hs = hidden.shape()
    if len(hs) != 3 or hs[0] != 1 or hs[2] != config.hidden_size:
        raise Error("MiniMax-H3 ControlNet hidden must be [1,S,H]")
    if rows.shape()[0] != target_video_rows:
        raise Error("MiniMax-H3 ControlNet guide row count != target video rows")
    var video_start = hs[1] - target_video_rows
    if video_start <= 0:
        raise Error("MiniMax-H3 ControlNet packed T2VA prefix is empty")
    var projected_f32 = linear_bias(rows, runtime.proj_w[], runtime.proj_b[], ctx)
    var projected_bf16 = cast_tensor(projected_f32, STDtype.BF16, ctx)
    var projected = reshape_owned(
        projected_bf16^, [1, target_video_rows, config.hidden_size]
    )
    var prefix = slice(hidden, 1, 0, video_start, ctx)
    var control = concat(1, ctx, prefix, projected)
    var before = linear_bias(control, runtime.before_w[], runtime.before_b[], ctx)
    return add(before, hidden, ctx)


def _control_load_block(
    runtime: MiniMaxH3ControlRuntime,
    index: Int,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> Dict[String, TArc]:
    """Map one official VideoX-Fun side block to the native H3 block ABI."""
    var layer = minimax_h3_control_injection_layer(index)
    var source = minimax_h3_control_block_prefix(index)
    var target = minimax_h3_block_prefix(layer)
    var weights = Dict[String, TArc]()
    weights[target + "norm1.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "norm1.weight", ctx)^
    )
    weights[target + "norm2.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "norm2.weight", ctx)^
    )
    weights[target + "attn.q_norm.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "attn.norm_q.weight", ctx)^
    )
    weights[target + "attn.k_norm.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "attn.norm_k.weight", ctx)^
    )

    # Official checkpoint stores separate [Q,K,V].  The native block ABI
    # consumes contiguous Q/K/V thirds, so concat is already the transformed
    # representation; applying the base loader's de-interleave here would be
    # a silent double permutation.
    var q = _control_load(runtime.checkpoint, source + "attn.to_q.weight", ctx)
    var k = _control_load(runtime.checkpoint, source + "attn.to_k.weight", ctx)
    var v = _control_load(runtime.checkpoint, source + "attn.to_v.weight", ctx)
    var qkv = concat(0, ctx, q, k, v)
    weights[target + "attn.qkv_proj.weight"] = TArc(qkv^)
    weights[target + "attn.out_proj.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "attn.to_out.0.weight", ctx)^
    )
    # FC1 needs NO reorder here, and applying one silently swaps the SwiGLU
    # gate and value halves.  Three facts compose:
    #   * the base checkpoint stores `mlp.fc1` as [gate; value], and the base
    #     device loader rewrites it with `_minimax_h3_fc1_swap_bf16_device`,
    #     so the native block ABI is [value; gate];
    #   * the released Union checkpoint stores `ff.net.0.proj` in the diffusers
    #     order, which is already [value; gate] -- the product reference
    #     (`serenityflow/models/minimax_h3/control.py::_load_local_tensor`)
    #     has to swap it to reach its own base-layout `mlp.fc1`;
    #   * measured on the real checkpoints, `control_blocks.0.ff.net.0.proj`
    #     matches `blocks.0.mlp.fc1` only ACROSS halves (cos 0.9995 crossed,
    #     -0.006 uncrossed), confirming the two orders are opposites.
    # Native = swap(base) and control = swap(base), so control is already the
    # native order and is handed to the block ABI unchanged.
    weights[target + "mlp.fc1.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "ff.net.0.proj.weight", ctx)^
    )
    weights[target + "mlp.fc2.weight"] = TArc(
        _control_load(runtime.checkpoint, source + "ff.net.2.weight", ctx)^
    )
    weights[String("__control_after.weight")] = TArc(
        _control_load(runtime.checkpoint, source + "after_proj.weight", ctx)^
    )
    weights[String("__control_after.bias")] = TArc(
        _control_load(runtime.checkpoint, source + "after_proj.bias", ctx)^
    )

    # The native block guard requires proof that qkv and fc1 are in its ABI.
    # Reuse the already-owned norm Arc as a presence-only marker.
    weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = (
        weights[target + "norm1.weight"].copy()
    )
    weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = (
        weights[target + "norm1.weight"].copy()
    )
    minimax_h3_check_block_weights(weights, layer, config)
    return weights^


def _control_release_block(
    runtime: MiniMaxH3ControlRuntime, index: Int
) raises:
    var source = minimax_h3_control_block_prefix(index)
    for suffix in [
        String("norm1.weight"), String("norm2.weight"),
        String("attn.norm_q.weight"), String("attn.norm_k.weight"),
        String("attn.to_q.weight"), String("attn.to_k.weight"),
        String("attn.to_v.weight"), String("attn.to_out.0.weight"),
        String("ff.net.0.proj.weight"), String("ff.net.2.weight"),
        String("after_proj.weight"), String("after_proj.bias"),
    ]:
        runtime.checkpoint.release_tensor(source + suffix)


def minimax_h3_control_inject_active[
    Heads: Int, HeadDim: Int
](
    var hidden: Tensor,
    mut control_hidden: List[TArc],
    active_inputs: List[Int],
    runtime: MiniMaxH3ControlRuntime,
    control_index: Int,
    config: MiniMaxH3DiTConfig,
    block_adaln_indices: List[Int],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    text_rows: Int,
    audio_rows: Int,
    attention_backend: Int,
    sage_scratch: Optional[SageInt8Scratch],
    comfy_kitchen_scratch: Optional[ComfyKitchenAttentionScratch],
    evg_layout: Optional[ArcPointer[EVGH3RaggedLayout]],
    step_index: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Advance all active controls through one shared streamed side block."""
    if len(active_inputs) != len(control_hidden):
        raise Error("MiniMax-H3 ControlNet active input/state count mismatch")
    var layer = minimax_h3_control_injection_layer(control_index)
    var weights = _control_load_block(runtime, control_index, config, ctx)
    for stream in range(len(active_inputs)):
        var c = minimax_h3_block_forward_dynamic[Heads, HeadDim](
            control_hidden[stream][], weights, layer, config,
            runtime.modcache.block_mod[control_index][], block_adaln_indices,
            cos, sin, rotary_dim, ctx, attention_backend, sage_scratch,
            evg_layout=evg_layout, evg_step=step_index,
            comfy_kitchen_scratch=comfy_kitchen_scratch,
        )
        var skip = linear_bias(
            c, weights[String("__control_after.weight")][],
            weights[String("__control_after.bias")][], ctx,
        )
        # The published implementation suppresses control residuals on audio
        # rows only; text and target-video residuals remain live.
        var prefix = slice(skip, 1, 0, text_rows, ctx)
        var zeros = full_device(
            [1, audio_rows, config.hidden_size], Float32(0.0), STDtype.BF16, ctx
        )
        var suffix_start = text_rows + audio_rows
        var suffix = slice(
            skip, 1, suffix_start, skip.shape()[1] - suffix_start, ctx
        )
        var prefix_audio = concat(1, ctx, prefix, zeros)
        var filtered = concat(1, ctx, prefix_audio, suffix)
        var scaled = mul_scalar(
            filtered, runtime.inputs[active_inputs[stream]].strength, ctx
        )
        hidden = add(hidden, scaled, ctx)
        control_hidden[stream] = TArc(c^)
    # Side-block weights are streamed by design. Fence their last use before
    # dropping the device Arcs and the checkpoint's host mapping windows.
    ctx.synchronize()
    weights.clear()
    _control_release_block(runtime, control_index)
    return hidden^
