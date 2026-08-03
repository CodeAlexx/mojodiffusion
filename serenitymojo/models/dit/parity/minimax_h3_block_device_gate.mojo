# serenitymojo/models/dit/parity/minimax_h3_block_device_gate.mojo
#
# MiniMax-H3 DEVICE block-forward parity gate — block 0 of the tiny
# random-weight fixture, GPU bf16, against the already-gated host-f32 oracle.
#
# Runtime under test: `models/dit/minimax_h3_dit.mojo::minimax_h3_block_forward`.
# Oracle:              `models/minimax_h3/block_forward.mojo::_transformer_block`,
#                       already gated end-to-end (whole 2-layer stack) at
#                       max_abs 5.96e-08 by `minimax_h3_block_parity.mojo`.
# Fixture:              `output/minimax_h3_block/block_ref.safetensors`, written
#                       by `scripts/minimax_h3_block_oracle.py` — REUSED verbatim,
#                       not regenerated or reshaped here.
#
# WHY THIS FILE ALSO REPLICATES `minimax_h3_forward` STEPS 1-5: the fixture only
# stores the model's INPUTS (raw video/audio/text rows, before the patch
# projections) and its FINAL outputs (after the whole 2-layer stack + final
# norm + both heads) — it has no recorded "hidden state entering block 0"
# tensor, because gating the whole forward never needed one. Isolating ONE
# block means reconstructing that boundary tensor. This file does that by
# calling the SAME already-gated functions `minimax_h3_forward` itself calls —
# `minimax_h3_rope_inv_freq`, `minimax_h3_rope_table`, `linear_bias`,
# `_token_refiner`, `minimax_h3_timestep_projection`, `_modulation`,
# `minimax_h3_adaln_indices` — in the same order, on the same fixture inputs.
# It does not reimplement or edit a single one of them; every number a device
# vs. host comparison could blame on THIS file, rather than on the block under
# test, is produced by oracle code that already has its own gate.
#
# WEIGHT CONVENTION: the device block wants ORIGINAL-checkpoint-shaped tensors,
# already rewritten to the device's post-loader convention (`minimax_h3_dit.mojo`
# header): `attn.qkv_proj.weight` = contiguous `[q_all; k_all; v_all]`, and
# `mlp.fc1.weight` = `[value; gate]` (loader's `minimax_h3_swap_fc1_bf16`
# target, confirmed by reading `minimax_h3_loader_device.mojo` directly rather
# than trusting either file's prose comment — see the FINDINGS section below,
# they disagree with each other). The fixture's DIFFUSERS-layout weights
# (`w.transformer_blocks.0.attn.to_q/to_k/to_v.weight`, `w...ff.net.0.proj.weight`)
# are ALREADY in exactly those shapes/conventions — `to_q;to_k;to_v` stacked is
# byte-for-byte the same permutation `minimax_h3_deinterleave_qkv_bf16` produces
# from the raw checkpoint, and diffusers' `ff.net.0.proj.weight` IS the
# `[value; gate]` layout (oracle script's own comment: "diffusers holds
# [value; gate]; the checkpoint holds [gate; value]"). So this gate builds the
# device weight Dict directly from the fixture's `w.*` tensors rather than
# round-tripping through the real bf16 loader — equivalent bytes, no checkpoint
# needed, and it keeps this file from depending on a second unit under test.
#
# BAR: cos >= 0.999, with max_abs and the L2 magnitude ratio |mine|/|ref|
# reported alongside — cos alone is magnitude-blind.
#
# Run — `mojo run` (JIT) CANNOT resolve the cuDNN shim's `flame_cudnn_sdpa_bf16`
# external symbol bare (`mojo run -I .` alone fails: "JIT session error:
# Symbols not found: [ flame_cudnn_sdpa_bf16 ]"); build a binary instead,
# exactly as scripts/build_wan22.sh does for the same shim:
#
#   cd /home/alex/mojodiffusion
#   pixi run scripts/mem_safe.sh mojo build --optimization-level 2 -j 1 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/dit/parity/minimax_h3_block_device_gate.mojo \
#     -o output/checks/minimax_h3_block_device_gate \
#   && output/checks/minimax_h3_block_device_gate
#
# CORRECTION (verified 2026-08-02): `-Xlinker` flags DO reach `mojo run`'s
# JIT linker — the earlier claim above that JIT has "no -Xlinker support" is
# wrong. The minimal flag set that actually resolves the shim (no -lcuda, no
# cudnn_stubs, no rpath needed) works for BOTH `mojo build` and `mojo run`:
#   pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/dit/parity/minimax_h3_block_device_gate.mojo
# (LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib is also needed at RUN time for
# a separately-built binary, since it isn't rpath'd by the minimal command.)
#
# GEOMETRY: `minimax_h3_block_forward[S, Heads, HeadDim]` takes heads/head_dim
# as its OWN comptime parameters (defaulted to the released 56/128) — this
# gate instantiates it at the FIXTURE's own geometry, `[S, 2, 16]`, which is
# exactly why that generalization had to land before this gate could measure
# anything (the earlier hardcoded-56/128 version raised unconditionally on
# any config that wasn't the release, before a single line of block math ran).

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.minimax_h3.block_forward import (
    MiniMaxH3BlockConfig,
    MiniMaxH3Weights,
    linear_bias,
    _silu,
    _token_refiner,
    _modulation,
    _transformer_block,
)
from serenitymojo.models.minimax_h3.dit_frontend import (
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
    minimax_h3_timestep_projection,
    minimax_h3_adaln_indices,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    minimax_h3_block_prefix,
    minimax_h3_block_forward,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/block_ref.safetensors"

# The reference fixture's own packed layout (scripts/minimax_h3_block_oracle.py).
comptime NUM_TEXT_TOKENS = 4
comptime NUM_AUDIO_TOKENS = 6
comptime NUM_VIDEO_TOKENS = 8
comptime S = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS  # 18


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_f64_as_f64(ref st: SafeTensors, name: String) raises -> List[Float64]:
    """position_ids are dumped float32 by the fixture; the port's rope builder
    takes float64 as the packing modules emit, so widen on the way in."""
    var values = _load_f32(st, name)
    var out = List[Float64]()
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _weight_names() -> List[String]:
    """Every parameter the tiny model exposes, in the reference's own naming —
    identical to `minimax_h3_block_parity.mojo`'s list (same fixture)."""
    var names = List[String]()
    names.append(String("proj_in.weight"))
    names.append(String("proj_in.bias"))
    names.append(String("audio_proj_in.weight"))
    names.append(String("audio_proj_in.bias"))
    names.append(String("context_embedder.weight"))
    names.append(String("context_embedder.bias"))
    names.append(String("time_embedder.linear_1.weight"))
    names.append(String("time_embedder.linear_1.bias"))
    names.append(String("time_embedder.linear_2.weight"))
    names.append(String("time_embedder.linear_2.bias"))
    names.append(String("token_refiner.final_norm.weight"))
    names.append(String("norm_out.norm.weight"))
    names.append(String("norm_out.linear.weight"))
    names.append(String("norm_out.linear.bias"))
    names.append(String("proj_out.weight"))
    names.append(String("proj_out.bias"))
    names.append(String("audio_proj_out.weight"))
    names.append(String("audio_proj_out.bias"))
    for layer in range(2):
        var p = String("transformer_blocks.") + String(layer)
        names.append(p + ".norm1.weight")
        names.append(p + ".norm2.weight")
        names.append(p + ".attn.to_q.weight")
        names.append(p + ".attn.to_k.weight")
        names.append(p + ".attn.to_v.weight")
        names.append(p + ".attn.norm_q.weight")
        names.append(p + ".attn.norm_k.weight")
        names.append(p + ".attn.to_out.0.weight")
        names.append(p + ".ff.net.0.proj.weight")
        names.append(p + ".ff.net.2.weight")
        names.append(p + ".adaln_proj.linear.weight")
        names.append(p + ".adaln_proj.linear.bias")
    for layer in range(2):
        var p = String("token_refiner.refiner_blocks.") + String(layer)
        names.append(p + ".norm1.weight")
        names.append(p + ".norm2.weight")
        names.append(p + ".attn.to_q.weight")
        names.append(p + ".attn.to_k.weight")
        names.append(p + ".attn.to_v.weight")
        names.append(p + ".attn.norm_q.weight")
        names.append(p + ".attn.norm_k.weight")
        names.append(p + ".attn.to_out.0.weight")
        names.append(p + ".ff.net.0.proj.weight")
        names.append(p + ".ff.net.2.weight")
    return names^


def main() raises:
    print("MiniMax-H3 DEVICE block-0 parity gate (tiny fixture, bf16 GPU)")
    print("  fixture:", REF)
    var st = SafeTensors.open(String(REF))

    # The reference fixture's own tiny config (scripts/minimax_h3_block_oracle.py
    # INIT / minimax_h3_block_parity.mojo). heads=2, head_dim=16 -> inner=32,
    # WIDER than hidden=24 but nowhere near the released 56/128 geometry.
    var host_config = MiniMaxH3BlockConfig(
        2,      # num_attention_heads
        16,     # attention_head_dim   -> inner 32
        24,     # hidden_size
        2,      # num_layers
        2,      # num_refiner_layers
        32,     # ffn_dim
        4,      # in_channels          -> video patch dim 16
        6,      # audio_in_channels
        8,      # text_dim
        8,      # freq_dim
        16,     # time_embed_dim
        2,      # rope_freq_dim        -> rotary_dim 12 < head_dim 16 (partial rope)
        Float32(1.0e-5),
        Float32(1.0e-5),
        Float32(1.0e-5),
    )

    var names = _weight_names()
    var values = List[List[Float32]]()
    for i in range(len(names)):
        values.append(_load_f32(st, String("w.") + names[i]))
    var weights = MiniMaxH3Weights(names^, values^)

    var video_rows = _load_f32(st, "in.hidden_states")
    var audio_rows = _load_f32(st, "in.audio_hidden_states")
    var text_rows = _load_f32(st, "in.encoder_hidden_states")
    var timesteps = _load_f32(st, "in.timestep")
    var timestep_indices = _load_i64(st, "in.timestep_indices")
    var token_tags = _load_i64(st, "in.token_tags")
    var position_ids = _load_f64_as_f64(st, "in.position_ids")
    var video_indices = _load_i64(st, "in.video_indices")
    var audio_indices = _load_i64(st, "in.audio_indices")
    var text_indices = _load_i64(st, "in.text_indices")

    var sequence_length = len(token_tags)
    if sequence_length != S:
        raise Error(
            String("fixture sequence_length ") + String(sequence_length)
            + " != comptime S " + String(S)
            + " — the fixture layout changed; update S to match before"
            " chasing a shape error downstream"
        )
    var hidden_size = host_config.hidden_size
    var num_timesteps = len(timesteps)

    # ─── Replicate minimax_h3_forward steps 1-5 (rope table, patch/text
    # embeds + token refiner, packed-buffer scatter, timestep embedding,
    # adaLN row map) — calling the SAME already-gated functions, same order,
    # same inputs, to reconstruct the tensor entering block 0. ───
    var inv_freq = minimax_h3_rope_inv_freq(host_config.rope_freq_dim)
    var rope = minimax_h3_rope_table(position_ids, sequence_length, inv_freq)
    var rotary_dim = rope.rotary_dim

    var video_embeds = linear_bias(
        video_rows, len(video_indices), host_config.video_patch_dim(),
        weights.get("proj_in.weight"), weights.get("proj_in.bias"), hidden_size,
    )
    var audio_embeds = linear_bias(
        audio_rows, len(audio_indices), host_config.audio_in_channels,
        weights.get("audio_proj_in.weight"), weights.get("audio_proj_in.bias"), hidden_size,
    )
    var text_embeds = linear_bias(
        text_rows, len(text_indices), host_config.text_dim,
        weights.get("context_embedder.weight"), weights.get("context_embedder.bias"), hidden_size,
    )
    text_embeds = _token_refiner(text_embeds, len(text_indices), host_config, weights)

    var hidden = List[Float32]()
    for _ in range(sequence_length * hidden_size):
        hidden.append(Float32(0.0))
    for r in range(len(text_indices)):
        for i in range(hidden_size):
            hidden[text_indices[r] * hidden_size + i] = text_embeds[r * hidden_size + i]
    for r in range(len(video_indices)):
        for i in range(hidden_size):
            hidden[video_indices[r] * hidden_size + i] = video_embeds[r * hidden_size + i]
    for r in range(len(audio_indices)):
        for i in range(hidden_size):
            hidden[audio_indices[r] * hidden_size + i] = audio_embeds[r * hidden_size + i]

    var projected = minimax_h3_timestep_projection(timesteps, host_config.freq_dim)
    var temb_hidden_bias = weights.get("time_embedder.linear_1.bias")
    var temb_hidden = linear_bias(
        projected, num_timesteps, host_config.freq_dim,
        weights.get("time_embedder.linear_1.weight"), temb_hidden_bias, len(temb_hidden_bias),
    )
    for i in range(len(temb_hidden)):
        temb_hidden[i] = _silu(temb_hidden[i])
    var temb = linear_bias(
        temb_hidden, num_timesteps, len(temb_hidden_bias),
        weights.get("time_embedder.linear_2.weight"), weights.get("time_embedder.linear_2.bias"),
        host_config.time_embed_dim,
    )

    var adaln_indices = minimax_h3_adaln_indices(timestep_indices, token_tags)

    # ─── block 0's own modulation table, then the HOST reference block ───
    var modulation0 = _modulation(
        temb, num_timesteps, host_config,
        weights.get("transformer_blocks.0.adaln_proj.linear.weight"),
        weights.get("transformer_blocks.0.adaln_proj.linear.bias"),
    )

    var hidden_in = hidden.copy()   # block-0 INPUT — shared by both sides
    var hidden_ref = hidden.copy()
    _transformer_block(
        hidden_ref, sequence_length, host_config, weights,
        "transformer_blocks.0", modulation0, adaln_indices,
        rope.cos, rope.sin, rotary_dim,
    )
    # hidden_ref is now the host-oracle's block-0 OUTPUT — the target.

    print("")
    print("[fixture]")
    print(
        "  S=", sequence_length, " hidden=", hidden_size,
        " heads=", host_config.num_attention_heads,
        " head_dim=", host_config.attention_head_dim,
        " rotary_dim=", rotary_dim, " num_timesteps=", num_timesteps,
    )

    # ─── DEVICE side ───
    print("")
    print("[device] building block-0 weights/inputs and DeviceContext ...")
    var ctx = DeviceContext()

    # `MiniMaxH3DiTConfig` field order: hidden_size, num_layers,
    # token_refiner_num_layers, num_attention_heads, attention_head_dim,
    # ffn_hidden_size, latents_dim, audio_latents_dim, text_dim,
    # timestep_input_dim, time_embed_dim, adaln_out_features,
    # final_adaln_out_features, rope_inv_freq_len, norm_eps, qk_norm_eps,
    # final_norm_eps. timestep_input_dim/adaln_out_features/
    # final_adaln_out_features are not read by minimax_h3_block_forward itself
    # (only by minimax_h3_check_block_weights/.validate(), not called here) —
    # filled in consistently anyway rather than left as dead placeholders.
    var device_config = MiniMaxH3DiTConfig(
        hidden_size,
        host_config.num_layers,
        host_config.num_refiner_layers,
        host_config.num_attention_heads,
        host_config.attention_head_dim,
        host_config.ffn_dim,
        host_config.in_channels,
        host_config.audio_in_channels,
        host_config.text_dim,
        host_config.freq_dim,
        host_config.time_embed_dim,
        6 * hidden_size * 3,
        2 * hidden_size,
        host_config.rope_freq_dim,
        Float32(1.0e-5),
        Float32(1.0e-5),
        Float32(1.0e-5),
    )

    var prefix = minimax_h3_block_prefix(0)          # "blocks.0."
    var w_prefix = String("transformer_blocks.0.")   # fixture's diffusers naming
    var inner = host_config.inner_dim()

    # qkv_proj: diffusers' to_q;to_k;to_v stacked IS the de-interleaved
    # [q_all;k_all;v_all] convention the device block expects — see header.
    var qkv = weights.get(w_prefix + "attn.to_q.weight")
    qkv.extend(weights.get(w_prefix + "attn.to_k.weight"))
    qkv.extend(weights.get(w_prefix + "attn.to_v.weight"))

    var dev_weights = Dict[String, ArcPointer[Tensor]]()
    dev_weights[prefix + "norm1.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "norm1.weight"), [hidden_size], STDtype.BF16, ctx)
    )
    dev_weights[prefix + "norm2.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "norm2.weight"), [hidden_size], STDtype.BF16, ctx)
    )
    dev_weights[prefix + "attn.qkv_proj.weight"] = ArcPointer(
        Tensor.from_host(qkv, [3 * inner, hidden_size], STDtype.BF16, ctx)
    )
    dev_weights[prefix + "attn.q_norm.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "attn.norm_q.weight"), [host_config.attention_head_dim], STDtype.BF16, ctx)
    )
    dev_weights[prefix + "attn.k_norm.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "attn.norm_k.weight"), [host_config.attention_head_dim], STDtype.BF16, ctx)
    )
    dev_weights[prefix + "attn.out_proj.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "attn.to_out.0.weight"), [hidden_size, inner], STDtype.BF16, ctx)
    )
    # fc1: diffusers' ff.net.0.proj.weight IS [value;gate] — the loader's
    # documented device target (see header + FINDINGS).
    dev_weights[prefix + "mlp.fc1.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "ff.net.0.proj.weight"), [2 * host_config.ffn_dim, hidden_size], STDtype.BF16, ctx)
    )
    dev_weights[prefix + "mlp.fc2.weight"] = ArcPointer(
        Tensor.from_host(weights.get(w_prefix + "ff.net.2.weight"), [hidden_size, host_config.ffn_dim], STDtype.BF16, ctx)
    )
    # minimax_h3_block_forward now REQUIRES these two markers (minimax_h3_dit
    # .mojo::minimax_h3_require_transformed_weights) — real production weights
    # get them from minimax_h3_load_block_device, but this gate builds
    # dev_weights directly from the fixture's already-in-the-right-convention
    # diffusers tensors (see header), which IS the transformed layout, just
    # assembled by a different, non-loader code path. Stamping here is
    # truthful, not a bypass: this gate's own passing cos result is the
    # evidence the row order is correct.
    dev_weights[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(
        Tensor.from_host([Float32(0.0)], [1], STDtype.BF16, ctx)
    )
    dev_weights[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(
        Tensor.from_host([Float32(0.0)], [1], STDtype.BF16, ctx)
    )

    var x = Tensor.from_host(hidden_in.copy(), [1, sequence_length, hidden_size], STDtype.BF16, ctx)
    var mod_tensor = Tensor.from_host(modulation0.copy(), [num_timesteps * 3, 6 * hidden_size], STDtype.F32, ctx)
    var cos_tensor = Tensor.from_host(rope.cos.copy(), [sequence_length, rotary_dim], STDtype.F32, ctx)
    var sin_tensor = Tensor.from_host(rope.sin.copy(), [sequence_length, rotary_dim], STDtype.F32, ctx)

    print("")
    print(
        "[device] invoking minimax_h3_block_forward[", S, ",",
        host_config.num_attention_heads, ",", host_config.attention_head_dim, "] ...",
    )
    var out = minimax_h3_block_forward[S, 2, 16](
        x, dev_weights, 0, device_config, mod_tensor, adaln_indices,
        cos_tensor, sin_tensor, rotary_dim, ctx,
    )
    var out_f32 = cast_tensor(out, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)
    var r = harness.compare(out_f32, hidden_ref, ctx)

    var actual = out_f32.to_host(ctx)
    var norm_actual = Float64(0.0)
    var norm_ref = Float64(0.0)
    for i in range(len(actual)):
        norm_actual += Float64(actual[i]) * Float64(actual[i])
    for i in range(len(hidden_ref)):
        norm_ref += Float64(hidden_ref[i]) * Float64(hidden_ref[i])
    norm_actual = sqrt(norm_actual)
    norm_ref = sqrt(norm_ref)
    var ratio = Float64(-1.0)
    if norm_ref != 0.0:
        ratio = norm_actual / norm_ref

    print("")
    print("[result]", r)
    print("  |mine| =", norm_actual, " |ref| =", norm_ref, " ratio |mine|/|ref| =", ratio)
    if r.passed:
        print("GATE PASS blockGateCos=", r.cos)
    else:
        raise Error("MiniMax-H3 DEVICE block-0 parity gate failed")
