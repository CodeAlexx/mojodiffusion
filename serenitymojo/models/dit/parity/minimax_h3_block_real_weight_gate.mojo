# serenitymojo/models/dit/parity/minimax_h3_block_real_weight_gate.mojo
#
# MiniMax-H3 DEVICE block-0 parity gate at the RELEASED geometry (56 heads x
# 128 head_dim, hidden 5376, ffn 14336), against REAL bf16 checkpoint bytes —
# the companion to minimax_h3_block_device_gate.mojo (tiny 2x16 fixture,
# random weights). That gate proved the block's ARITHMETIC; this one proves
# the block correctly consumes the ACTUAL released weights at the ACTUAL
# geometry, which is a categorically different risk: a wrong permutation or
# a swapped convention can be invisible on random weights (any consistent
# relabeling of random numbers still "looks like a tensor") but is exactly
# the class of bug real weights expose.
#
# WEIGHTS: real, loaded through the PRODUCTION loader —
# `minimax_h3_load_block_device` on `model-00001-of-00013.safetensors`
# (the shard block 0/1 live in, per model.safetensors.index.json) — not
# hand-assembled. This is deliberate: `minimax_h3_block_forward`'s sentinel
# guard (`minimax_h3_require_transformed_weights`) now REJECTS a hand-built
# Dict outright, and routing through the real loader is what this gate is
# FOR — it is the only thing in this port so far that has run the model's
# own math on the model's own weights (the loader's earlier self-check only
# proved the PERMUTATION was byte-correct, never that the forward pass
# consumes it right).
#
# The FL2VA download is 7/13 shards as of this gate (shard 2 and 9-13
# missing); `ShardedSafeTensors.open` on the DIRECTORY eagerly opens every
# shard the index lists and fails on the first missing one — confirmed by
# running minimax_h3_loader_device.mojo's own probe. Passing the SHARD FILE
# PATH directly instead (not the directory) hits `ShardedSafeTensors.open`'s
# documented single-file fallback (sharded.mojo: "Direct single-file input")
# and maps exactly that shard's 54 tensors with no index lookup and no
# missing-file error — confirmed working via a standalone probe before this
# gate was written. No new files, no symlinks, no trimmed index.
#
# INPUTS/REFERENCE: hidden_states (block-0 input), the AdaLN modulation table
# `mod`, rope cos/sin, adaln_indices, and the reference output are all from
# minimax_h3_block_real_weight_oracle.py, which runs diffusers' OWN
# `MiniMaxH3TransformerBlock.forward` (GPU bf16) with these SAME real weights
# — not a hand transcription of the math. `hidden_states`/`temb` are seeded
# random activations (adaln_proj is intentionally NOT loaded from the
# checkpoint, matching production — see that script's header), so `mod` is
# fed to this gate as an opaque input exactly as `minimax_h3_block_forward`
# consumes it in production, never recomputed here.
#
# BAR: cos >= 0.999, reported alongside max_abs AND the L2 magnitude ratio
# |mine|/|ref| — cos is magnitude-blind. EXPECT A LOOSER max_abs than the
# tiny fixture's 0.0068: these GEMMs are 5376/7168/14336-wide instead of
# 24/32/32-wide, so bf16 rounding accumulates over ~300x more terms per
# output element. Judge PASS/FAIL on cos and the magnitude ratio, not on
# max_abs matching the tiny gate's number — an absolute threshold copied
# from a different width would not mean what it looks like it means.
#
# Run:
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/dit/parity/minimax_h3_block_real_weight_oracle.py
#   pixi run scripts/mem_safe.sh mojo build --optimization-level 2 -j 1 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/dit/parity/minimax_h3_block_real_weight_gate.mojo \
#     -o output/checks/minimax_h3_block_real_weight_gate \
#   && output/checks/minimax_h3_block_real_weight_gate
#   (mojo run/JIT cannot resolve flame_cudnn_sdpa_bf16 — build a binary.)

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor

from serenitymojo.models.dit.minimax_h3_dit import (
    minimax_h3_released_config,
    minimax_h3_block_forward,
)
from serenitymojo.models.dit.minimax_h3_loader_device import minimax_h3_load_block_device

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/block0_real_weight_ref.safetensors"

# The shard FILE (not the directory) — see header for why: the directory has
# 6/13 shards missing and ShardedSafeTensors.open on a directory opens every
# shard the index names, so it fails before ever reaching block 0's shard.
comptime SHARD1 = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/"
    "model-00001-of-00013.safetensors"
)

comptime S = 18  # same packed S as the tiny fixture — see oracle header


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


def _load_i64_as_int(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64_as_int: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def main() raises:
    print("MiniMax-H3 DEVICE block-0 parity gate — REAL WEIGHTS, released geometry (56/128)")
    print("  reference:", REF)
    print("  checkpoint shard:", SHARD1)

    var st_ref = SafeTensors.open(String(REF))
    var x_h = _load_f32(st_ref, "in.hidden_states")
    var mod_h = _load_f32(st_ref, "in.mod")
    var cos_h = _load_f32(st_ref, "in.cos")
    var sin_h = _load_f32(st_ref, "in.sin")
    var adaln_indices = _load_i64_as_int(st_ref, "in.adaln_indices")
    var ref_out = _load_f32(st_ref, "out.hidden_states")

    var config = minimax_h3_released_config()
    var hidden = config.hidden_size          # 5376
    var mod_rows = len(mod_h) // (6 * hidden)
    var rotary_dim = len(cos_h) // S

    if len(adaln_indices) != S:
        raise Error(
            String("fixture sequence_length ") + String(len(adaln_indices))
            + " != comptime S " + String(S)
        )
    if len(x_h) != S * hidden:
        raise Error("in.hidden_states size does not match S*hidden_size")
    if len(ref_out) != S * hidden:
        raise Error("out.hidden_states size does not match S*hidden_size")

    print("")
    print("[fixture]")
    print(
        "  S=", S, " hidden=", hidden, " heads=", config.num_attention_heads,
        " head_dim=", config.attention_head_dim, " ffn=", config.ffn_hidden_size,
        " rotary_dim=", rotary_dim, " mod_rows=", mod_rows,
    )

    # ─── Load block 0 through the REAL production loader, on the REAL shard ───
    print("")
    print("[device] loading block 0 via minimax_h3_load_block_device (real bf16 checkpoint bytes) ...")
    var ctx = DeviceContext()
    var shard = ShardedSafeTensors.open(String(SHARD1))
    print("  shard opened:", shard.num_shards(), "file(s),", shard.num_tensors(), "tensors")
    var weights = minimax_h3_load_block_device(shard, 0, config, ctx)
    print("  loaded", len(weights), "entries (8 real tensors + 2 sentinel markers)")

    var x = Tensor.from_host(x_h.copy(), [1, S, hidden], STDtype.BF16, ctx)
    var mod_tensor = Tensor.from_host(mod_h.copy(), [mod_rows, 6 * hidden], STDtype.F32, ctx)
    var cos_tensor = Tensor.from_host(cos_h.copy(), [S, rotary_dim], STDtype.F32, ctx)
    var sin_tensor = Tensor.from_host(sin_h.copy(), [S, rotary_dim], STDtype.F32, ctx)

    print("")
    print("[device] invoking minimax_h3_block_forward[", S, ",", config.num_attention_heads, ",", config.attention_head_dim, "] ...")
    var out = minimax_h3_block_forward[S, 56, 128](
        x, weights, 0, config, mod_tensor, adaln_indices,
        cos_tensor, sin_tensor, rotary_dim, ctx,
    )
    var out_f32 = cast_tensor(out, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)
    var r = harness.compare(out_f32, ref_out, ctx)

    var actual = out_f32.to_host(ctx)
    var norm_actual = Float64(0.0)
    var norm_ref = Float64(0.0)
    for i in range(len(actual)):
        norm_actual += Float64(actual[i]) * Float64(actual[i])
    for i in range(len(ref_out)):
        norm_ref += Float64(ref_out[i]) * Float64(ref_out[i])
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
        print("GATE FAIL blockGateCos=", r.cos)
        raise Error("MiniMax-H3 DEVICE block-0 REAL-WEIGHT parity gate failed")
