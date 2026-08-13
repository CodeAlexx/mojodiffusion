# serenitymojo/models/dit/minimax_h3_frontend_probe.mojo
#
# GPU smoke probe for minimax_h3_frontend.mojo — proves the module BUILDS and
# RUNS end to end on device at the real released config's dimensions (hidden
# 5376, heads 56, head_dim 128, inner 7168 != hidden, ffn 14336). Weights are
# synthetic (uniform-constant device fill, no host staging — see
# ops/tensor_algebra.full_device) since the 61.7 GiB checkpoint is out of
# scope for a probe.
#
# THIS IS NOT A PARITY CHECK. There is no reference output to compare
# against; it verifies shapes, dtypes (the 12-fp32/rest-bf16 split), and that
# every kernel launches and returns finite-shaped tensors. The final_layer
# call below feeds `embed.hidden` (the frontend's OWN pre-stack output)
# straight in as a stand-in for "the 50-block stack's output" purely to
# exercise that code path — it does NOT stand in for the real block stack
# numerically.
#
# Run: pixi run mojo run -I . serenitymojo/models/dit/minimax_h3_frontend_probe.mojo

from std.collections import Dict, List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
)
from serenitymojo.models.dit.minimax_h3_frontend import (
    MiniMaxH3FrontendEmbed,
    MiniMaxH3FrontendOutput,
    minimax_h3_video_patchify,
    minimax_h3_frontend_embed,
    minimax_h3_final_layer,
)
from serenitymojo.ops.tensor_algebra import full_device


def _put(
    mut w: Dict[String, ArcPointer[Tensor]],
    name: String,
    var shape: List[Int],
    value: Float32,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    w[name] = ArcPointer(full_device(shape^, value, dtype, ctx))


def main() raises:
    var ctx = DeviceContext()
    var config = minimax_h3_released_config()
    config.validate()

    var hidden = config.hidden_size  # 5376
    var inner = config.inner_dim()  # 7168 (!= hidden — the trap the header warns about)
    var ffn = config.ffn_hidden_size  # 14336
    var head_dim = config.attention_head_dim  # 128
    var video_pd = config.video_patch_dim()  # 96

    if inner == hidden:
        raise Error("minimax_h3_frontend probe: expected inner_dim != hidden_size")

    var w = Dict[String, ArcPointer[Tensor]]()

    # ── fp32 tensors: the 12-tensor dtype trap; all 12 live in this module ──
    # (ORIGINAL checkpoint key spelling — see minimax_h3_frontend.mojo's
    # revised "WEIGHT-KEY CONTRACT" header, team-lead correction 2026-08-02.)
    _put(w, "video_patch_proj.weight", [hidden, video_pd], 0.01, STDtype.F32, ctx)
    _put(w, "video_patch_proj.bias", [hidden], 0.0, STDtype.F32, ctx)
    _put(w, "audio_patch_proj.weight", [hidden, config.audio_latents_dim], 0.01, STDtype.F32, ctx)
    _put(w, "audio_patch_proj.bias", [hidden], 0.0, STDtype.F32, ctx)
    _put(w, "time_embedder.proj_in.weight", [config.time_embed_dim, config.timestep_input_dim], 0.01, STDtype.F32, ctx)
    _put(w, "time_embedder.proj_in.bias", [config.time_embed_dim], 0.0, STDtype.F32, ctx)
    _put(w, "time_embedder.proj_out.weight", [config.time_embed_dim, config.time_embed_dim], 0.001, STDtype.F32, ctx)
    _put(w, "time_embedder.proj_out.bias", [config.time_embed_dim], 0.0, STDtype.F32, ctx)
    _put(w, "final_layer.video_out.weight", [video_pd, hidden], 0.01, STDtype.F32, ctx)
    _put(w, "final_layer.video_out.bias", [video_pd], 0.0, STDtype.F32, ctx)
    _put(w, "final_layer.audio_out.weight", [config.audio_latents_dim, hidden], 0.01, STDtype.F32, ctx)
    _put(w, "final_layer.audio_out.bias", [config.audio_latents_dim], 0.0, STDtype.F32, ctx)

    # ── bf16 tensors: everything else. attn.qkv_proj and mlp.fc1 are ONE
    # fused tensor each (loader-de-interleaved / loader-swapped values, key
    # unchanged) — sliced by row-range inside the frontend, not split here. ──
    _put(w, "condition_proj.weight", [hidden, config.text_dim], 0.001, STDtype.BF16, ctx)
    _put(w, "condition_proj.bias", [hidden], 0.0, STDtype.BF16, ctx)
    for layer in range(config.token_refiner_num_layers):
        var p = String("token_refiner.blocks.") + String(layer)
        _put(w, p + ".norm1.weight", [hidden], 1.0, STDtype.BF16, ctx)
        _put(w, p + ".attn.qkv_proj.weight", [3 * inner, hidden], 0.002, STDtype.BF16, ctx)
        _put(w, p + ".attn.q_norm.weight", [head_dim], 1.0, STDtype.BF16, ctx)
        _put(w, p + ".attn.k_norm.weight", [head_dim], 1.0, STDtype.BF16, ctx)
        _put(w, p + ".attn.out_proj.weight", [hidden, inner], 0.002, STDtype.BF16, ctx)
        _put(w, p + ".norm2.weight", [hidden], 1.0, STDtype.BF16, ctx)
        _put(w, p + ".mlp.fc1.weight", [2 * ffn, hidden], 0.001, STDtype.BF16, ctx)
        _put(w, p + ".mlp.fc2.weight", [hidden, ffn], 0.001, STDtype.BF16, ctx)
    _put(w, "token_refiner.final_norm.weight", [hidden], 1.0, STDtype.BF16, ctx)
    _put(w, "final_layer.norm.weight", [hidden], 1.0, STDtype.BF16, ctx)

    if len(w) != 12 + 2 + 2 * 8 + 2:
        raise Error("minimax_h3_frontend probe: unexpected weight-dict size")

    # ── synthetic activations at a tiny grid ──
    comptime TR_S = 6  # text rows
    comptime TR_H = 56  # num_attention_heads
    comptime TR_DH = 128  # attention_head_dim
    var video_latents = full_device([config.latents_dim, 2, 4, 4], 0.1, STDtype.F32, ctx)  # [24,2,4,4]
    var video_rows = minimax_h3_video_patchify(video_latents, ctx)  # -> [8, 96]
    var audio_rows = full_device([4, config.audio_latents_dim], 0.1, STDtype.F32, ctx)  # [4, 32]
    var text_rows = full_device([TR_S, config.text_dim], 0.05, STDtype.BF16, ctx)  # [6, 5120]
    var timesteps = full_device([1], 0.5, STDtype.F32, ctx)  # [1]

    if video_rows.shape()[0] != 8 or video_rows.shape()[1] != video_pd:
        raise Error("minimax_h3_frontend probe: video_patchify shape mismatch")

    var text_indices = List[Int]()
    for i in range(TR_S):
        text_indices.append(i)
    var audio_indices = List[Int]()
    for i in range(4):
        audio_indices.append(TR_S + i)
    var video_indices = List[Int]()
    for i in range(8):
        video_indices.append(TR_S + 4 + i)
    var sequence_length = TR_S + 4 + 8  # 18

    var embed = minimax_h3_frontend_embed[TR_S, TR_H, TR_DH](
        video_rows, audio_rows, text_rows, timesteps,
        video_indices, audio_indices, text_indices,
        sequence_length, w, config, ctx,
    )

    if embed.hidden.shape()[0] != sequence_length or embed.hidden.shape()[1] != hidden:
        raise Error("minimax_h3_frontend probe: hidden shape mismatch")
    if embed.hidden.dtype() != STDtype.BF16:
        raise Error("minimax_h3_frontend probe: hidden dtype must be bf16")
    if embed.temb.shape()[0] != 1 or embed.temb.shape()[1] != config.time_embed_dim:
        raise Error("minimax_h3_frontend probe: temb shape mismatch")
    if embed.temb.dtype() != STDtype.F32:
        raise Error("minimax_h3_frontend probe: temb dtype must be F32 (fp32 prefix)")

    # ── final layer: `embed.hidden` stands in for the 50-block stack's
    # output (NOT a parity substitute — see file header) purely to run the
    # code path. ──
    var final_modulation = full_device([1, 2 * hidden], 0.01, STDtype.F32, ctx)
    var timestep_indices = List[Int]()
    for _ in range(sequence_length):
        timestep_indices.append(0)

    var out = minimax_h3_final_layer(
        embed.hidden, final_modulation, timestep_indices,
        video_indices, audio_indices, w, config, ctx,
    )

    if out.video_out.shape()[0] != 8 or out.video_out.shape()[1] != video_pd:
        raise Error("minimax_h3_frontend probe: video_out shape mismatch")
    if out.video_out.dtype() != STDtype.F32:
        raise Error("minimax_h3_frontend probe: video_out dtype must be F32 (fp32 prefix)")
    if out.audio_out.shape()[0] != 4 or out.audio_out.shape()[1] != config.audio_latents_dim:
        raise Error("minimax_h3_frontend probe: audio_out shape mismatch")
    if out.audio_out.dtype() != STDtype.F32:
        raise Error("minimax_h3_frontend probe: audio_out dtype must be F32 (fp32 prefix)")

    var vh = out.video_out.to_host(ctx)
    var ah = out.audio_out.to_host(ctx)
    print("minimax_h3_frontend probe: hidden    ", embed.hidden.shape()[0], "x", embed.hidden.shape()[1])
    print("minimax_h3_frontend probe: temb      ", embed.temb.shape()[0], "x", embed.temb.shape()[1])
    print("minimax_h3_frontend probe: video_out ", out.video_out.shape()[0], "x", out.video_out.shape()[1], " sample=", vh[0])
    print("minimax_h3_frontend probe: audio_out ", out.audio_out.shape()[0], "x", out.audio_out.shape()[1], " sample=", ah[0])
    print("PROBE PASS (shapes/dtypes only — NOT a parity check)")
