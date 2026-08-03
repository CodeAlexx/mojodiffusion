# serenitymojo/models/minimax_h3/loader.mojo
#
# MiniMax-H3 transformer loader — unit 12 of the H3 port.
#
# Turns the ORIGINAL checkpoint layout into the tensor layout
# `block_forward.mojo` consumes. This is the only place in the port allowed to
# know about checkpoint spelling, which is what keeps the forward pass provably
# about arithmetic and this file provably about placement.
#
# Specification: the diffusers converter's own key plan, run with no weights
# present and saved at /home/alex/minimax_h3_ref/transformer_key_plan.txt —
# 535 original keys -> 638 tensors, 1 dropped, 12 float32, 61.73 GiB total.
#   scripts/convert_minimax_h3_to_diffusers.py
#     reorder_interleaved_qkv  :110-132
#     split_fused_qkv          :135-151
#     convert_transformer_key  :231-273
#
# THREE TRANSFORMS AND NOTHING ELSE. The converter states it outright: "There
# are no transposes anywhere."
#
#   1. FUSED QKV IS PER-HEAD INTERLEAVED in the raw shards —
#      `[head0: q(D) k(D) v(D), head1: q k v, ...]`. De-interleave to
#      `[q_all; k_all; v_all]`, then split contiguous thirds. The reference
#      applies exactly this at load time (`_reorder_grouped_qkv_to_qkv`, one
#      head per query group, so plain MHA rather than GQA). A naive
#      three-contiguous-thirds split loads without complaint and emits noise —
#      the single most dangerous mistake available in this checkpoint.
#   2. `mlp.fc1` stores `[gate; value]` and the reference computes
#      `fc2(silu(gate) * value)`. diffusers' SwiGLU wants `[value; gate]`, so
#      the halves swap. Same transform recurs in the video VAE's `ff.w1`.
#   3. `rope.inv_freq` is dropped and recomputed — the converter asserts the
#      recomputed tensor is bitwise equal to the shipped one in BOTH released
#      variants. `dit_frontend.mojo` does that recompute, gated in unit 5.
#
# DTYPE: the checkpoint is mixed precision. Exactly these original prefixes are
# float32 — `video_patch_proj.`, `audio_patch_proj.`, `time_embedder.`,
# `final_layer.video_out.`, `final_layer.audio_out.` — and everything else is
# bfloat16, INCLUDING the adaLN projections. `is_fp32_key` below is that rule,
# and the dry run counts exactly 12 such tensors in the released config.

from std.collections import List


comptime MINIMAX_H3_FP32_PREFIXES_COUNT = 5


def minimax_h3_is_fp32_key(name: String) -> Bool:
    """Whether an ORIGINAL key is stored float32 rather than bfloat16.

    Substring-at-start match on the five prefixes the converter lists. Note
    `final_layer.adaln_proj` is NOT among them: the adaLN projections are
    bfloat16 like the block stack, despite sitting next to the fp32 heads."""
    if name.startswith("video_patch_proj."):
        return True
    if name.startswith("audio_patch_proj."):
        return True
    if name.startswith("time_embedder."):
        return True
    if name.startswith("final_layer.video_out."):
        return True
    if name.startswith("final_layer.audio_out."):
        return True
    return False


def minimax_h3_is_dropped_key(name: String) -> Bool:
    """`rope.inv_freq` is recomputed rather than loaded."""
    return name == "rope.inv_freq"


def minimax_h3_rename_key(name: String) raises -> String:
    """ORIGINAL key -> the name `block_forward.mojo` looks up.

    Pure renaming; the three tensor transforms are separate functions. Keys
    whose tensor is SPLIT (`attn.qkv_proj.weight`) are rejected here, because
    they map to three names rather than one."""
    if name.endswith(".attn.qkv_proj.weight"):
        raise Error(
            "MiniMax-H3 loader: qkv_proj maps to three tensors; use"
            " minimax_h3_split_qkv rather than a rename"
        )

    var renamed = name
    # `replace` on the leading segment rather than a slice: Mojo strings index
    # by byte with an explicit keyword, and the prefixes here are ASCII and
    # occur only at the start, so this is equivalent and reads better.
    if renamed.startswith("token_refiner.blocks."):
        renamed = renamed.replace(
            "token_refiner.blocks.", "token_refiner.refiner_blocks."
        )
    elif renamed.startswith("blocks."):
        renamed = renamed.replace("blocks.", "transformer_blocks.")

    renamed = renamed.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    renamed = renamed.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    renamed = renamed.replace("video_patch_proj.", "proj_in.")
    renamed = renamed.replace("audio_patch_proj.", "audio_proj_in.")
    renamed = renamed.replace("condition_proj.", "context_embedder.")
    renamed = renamed.replace("final_layer.norm.", "norm_out.norm.")
    renamed = renamed.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    renamed = renamed.replace("final_layer.video_out.", "proj_out.")
    renamed = renamed.replace("final_layer.audio_out.", "audio_proj_out.")
    renamed = renamed.replace(".attn.q_norm.", ".attn.norm_q.")
    renamed = renamed.replace(".attn.k_norm.", ".attn.norm_k.")
    renamed = renamed.replace(".attn.out_proj.", ".attn.to_out.0.")
    renamed = renamed.replace(".mlp.fc1.", ".ff.net.0.proj.")
    renamed = renamed.replace(".mlp.fc2.", ".ff.net.2.")
    return renamed


def minimax_h3_deinterleave_qkv(
    fused: List[Float32], heads: Int, head_dim: Int, in_features: Int
) raises -> List[Float32]:
    """Per-head-interleaved fused QKV -> `[q_all; k_all; v_all]`.

    Raw rows are `[head0: q(head_dim) k(head_dim) v(head_dim), head1: ...]`.
    Reading them as three contiguous thirds gives a model that runs and
    produces noise, so this is gated rather than assumed."""
    var expected = heads * 3 * head_dim
    if len(fused) != expected * in_features:
        raise Error(
            "MiniMax-H3 loader: fused qkv has the wrong row count for this"
            " head configuration"
        )

    var out = List[Float32]()
    for _ in range(len(fused)):
        out.append(Float32(0.0))
    for part in range(3):
        for h in range(heads):
            for d in range(head_dim):
                var source_row = h * 3 * head_dim + part * head_dim + d
                var dest_row = part * heads * head_dim + h * head_dim + d
                for i in range(in_features):
                    out[dest_row * in_features + i] = fused[source_row * in_features + i]
    return out^


def minimax_h3_split_qkv(
    reordered: List[Float32], heads: Int, head_dim: Int, in_features: Int, part: Int
) raises -> List[Float32]:
    """One contiguous third of a de-interleaved QKV: 0 = q, 1 = k, 2 = v."""
    if part < 0 or part > 2:
        raise Error("MiniMax-H3 loader: qkv part must be 0, 1 or 2")
    var inner = heads * head_dim
    var out = List[Float32]()
    for row in range(inner):
        var source = (part * inner + row) * in_features
        for i in range(in_features):
            out.append(reordered[source + i])
    return out^


def minimax_h3_swap_fc1(
    fc1: List[Float32], ffn_dim: Int, in_features: Int
) raises -> List[Float32]:
    """`[gate; value]` -> `[value; gate]`.

    The checkpoint stores the gate first and the reference computes
    `fc2(silu(gate) * value)`; the SwiGLU this port's forward uses reads the
    value first. Getting this backwards swaps which half is squashed by the
    non-linearity, which is not visible in any shape."""
    if len(fc1) != 2 * ffn_dim * in_features:
        raise Error("MiniMax-H3 loader: fc1 has the wrong shape for this ffn_dim")
    var out = List[Float32]()
    for row in range(ffn_dim):
        var source = (ffn_dim + row) * in_features
        for i in range(in_features):
            out.append(fc1[source + i])
    for row in range(ffn_dim):
        var source = row * in_features
        for i in range(in_features):
            out.append(fc1[source + i])
    return out^
