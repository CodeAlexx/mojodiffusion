# serenitymojo/models/dit/parity/minimax_h3_stack_real_weight_gate.mojo
#
# MiniMax-H3 2-LAYER STACK parity gate against REAL checkpoint bytes — drives
# `minimax_h3_run_stack` (models/dit/minimax_h3_stack.mojo) over blocks 0/1's
# REAL weights, released geometry (56/128), and compares the STACK'S output
# (not one block in isolation) to a reference that runs the SAME two real
# layers sequentially. Companion to:
#   minimax_h3_block_real_weight_gate.mojo     — one real block, released geometry
#   minimax_h3_modcache_real_weight_gate.mojo  — real adaln_proj bytes, modcache math
# This is the first thing in the port to compose real weights ACROSS layers
# rather than testing one block or one weight-consuming unit at a time — it
# is the closest thing to "the actual denoise loop" that real checkpoint
# bytes on disk currently allow.
#
# SCOPE, DELIBERATELY NARROW (h3-block's suggestion, confirmed with
# team-lead): this gate does NOT call `minimax_h3_build_modulation_cache` —
# that unit is already verified separately
# (minimax_h3_modcache_real_weight_gate.mojo, cos=1.0 on real blocks.0/1
# adaln_proj bytes). `MiniMaxH3ModCache` is `@fieldwise_init`, so this gate
# builds one directly from the oracle's own dumped per-layer modulation
# tensors — the SAME approach minimax_h3_block_real_weight_gate.mojo already
# used for `mod` (an opaque input, never recomputed here). That keeps this
# gate isolated to STACK COMPOSITION + BLOCK MATH across two real-weight
# layers, not a second exercise of modcache-building as a side effect.
# `final_mod` is unused by `minimax_h3_run_stack` (a separate final-layer
# unit); a placeholder tensor is supplied and never read.
#
# WEIGHTS: real, loaded through the PRODUCTION streaming path —
# `minimax_h3_run_stack` calls `minimax_h3_load_block_device` once per layer
# internally, exactly as it would for the real 50-layer stack — not
# hand-assembled here. `st` is the SAME shard-1-opened `ShardedSafeTensors`
# the block gate already validated works via `ShardedSafeTensors.open`'s
# documented single-file fallback (sharded.mojo) — no directory open (which
# fails on the 6 missing shards), no symlink, no scratch dir. Both blocks
# 0 and 1 are 100% contained in shard 1, per model.safetensors.index.json.
#
# ORACLE: minimax_h3_stack_real_weight_oracle.py runs diffusers' OWN
# `MiniMaxH3TransformerBlock` TWICE, sequentially (h1 = block0(h0), h2 =
# block1(h1)), each loaded with that layer's real weights — the actual
# composition under test, not two isolated single-block runs compared
# separately.
#
# BAR: cos >= 0.999, max_abs AND magnitude ratio |mine|/|ref| reported
# alongside. EXPECT max_abs looser than even the single-block real-weight
# gate's 100.5 — output magnitude compounds across layers when adaln_proj is
# untrained (std ~690 after 1 layer, ~831 after 2, per both oracles' own
# printed stats) — judge PASS/FAIL on cos and the magnitude ratio.
#
# Run:
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/dit/parity/minimax_h3_stack_real_weight_oracle.py
#   pixi run scripts/mem_safe.sh mojo build --optimization-level 2 -j 1 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/dit/parity/minimax_h3_stack_real_weight_gate.mojo \
#     -o output/checks/minimax_h3_stack_real_weight_gate \
#   && output/checks/minimax_h3_stack_real_weight_gate
#   (mojo run/JIT cannot resolve flame_cudnn_sdpa_bf16 — build a binary.)

from std.collections import List
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import full_device

from serenitymojo.models.dit.minimax_h3_dit import minimax_h3_released_config
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache
from serenitymojo.models.dit.minimax_h3_stack import minimax_h3_run_stack

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/stack2_real_weight_ref.safetensors"
comptime SHARD1 = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/"
    "model-00001-of-00013.safetensors"
)

comptime S = 18            # same packed S as the block-0 gate — see oracle header
comptime NUM_LAYERS = 2    # blocks 0 and 1 -- the only fully-present real bytes


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
    print("MiniMax-H3 2-LAYER STACK parity gate — REAL WEIGHTS, released geometry (56/128)")
    print("  reference:", REF)
    print("  checkpoint shard:", SHARD1)

    var st_ref = SafeTensors.open(String(REF))
    var x_h = _load_f32(st_ref, "in.hidden_states")
    var cos_h = _load_f32(st_ref, "in.cos")
    var sin_h = _load_f32(st_ref, "in.sin")
    var adaln_indices = _load_i64_as_int(st_ref, "in.adaln_indices")
    var mod0_h = _load_f32(st_ref, "in.mod.0")
    var mod1_h = _load_f32(st_ref, "in.mod.1")
    var ref_out = _load_f32(st_ref, "out.hidden_states")

    var config = minimax_h3_released_config()
    var hidden = config.hidden_size          # 5376
    var rotary_dim = len(cos_h) // S
    var mod_rows = len(mod0_h) // (6 * hidden)

    if len(adaln_indices) != S:
        raise Error("fixture sequence_length != comptime S")
    if len(x_h) != S * hidden or len(ref_out) != S * hidden:
        raise Error("in/out hidden_states size does not match S*hidden_size")

    print("")
    print("[fixture]")
    print(
        "  S=", S, " hidden=", hidden, " heads=", config.num_attention_heads,
        " head_dim=", config.attention_head_dim, " rotary_dim=", rotary_dim,
        " mod_rows=", mod_rows, " num_layers=", NUM_LAYERS,
    )

    var ctx = DeviceContext()

    print("")
    print("[device] opening shard 1 (both blocks 0 and 1 are fully contained in it) ...")
    var st = ShardedSafeTensors.open(String(SHARD1))
    print("  shard opened:", st.num_shards(), "file(s),", st.num_tensors(), "tensors")

    # ── Hand-built modcache from the oracle's own per-layer mod dump — see
    # header for why this gate does not call minimax_h3_build_modulation_cache. ──
    var block_mod = List[ArcPointer[Tensor]]()
    block_mod.append(
        ArcPointer(Tensor.from_host(mod0_h.copy(), [mod_rows, 6 * hidden], STDtype.F32, ctx))
    )
    block_mod.append(
        ArcPointer(Tensor.from_host(mod1_h.copy(), [mod_rows, 6 * hidden], STDtype.F32, ctx))
    )
    # final_mod is unused by minimax_h3_run_stack (a separate final-layer unit);
    # a minimal placeholder, never read.
    var final_shape: List[Int] = [1, 2 * hidden]
    var final_mod = ArcPointer(full_device(final_shape^, Float32(0.0), STDtype.F32, ctx))
    var distinct_timesteps = mod_rows // 3
    var modcache = MiniMaxH3ModCache(block_mod^, final_mod^, distinct_timesteps)

    var hidden_in = Tensor.from_host(x_h.copy(), [S, hidden], STDtype.BF16, ctx)
    var cos_tensor = Tensor.from_host(cos_h.copy(), [S, rotary_dim], STDtype.F32, ctx)
    var sin_tensor = Tensor.from_host(sin_h.copy(), [S, rotary_dim], STDtype.F32, ctx)

    print("")
    print("[device] minimax_h3_run_stack[", S, ",56,128] over", NUM_LAYERS, "real streamed layers ...")
    var out = minimax_h3_run_stack[S, 56, 128](
        hidden_in^, st, modcache, adaln_indices, cos_tensor, sin_tensor,
        rotary_dim, config, ctx, NUM_LAYERS,
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
        print("GATE PASS stackGateCos=", r.cos)
    else:
        print("GATE FAIL stackGateCos=", r.cos)
        raise Error("MiniMax-H3 2-layer STACK real-weight parity gate failed")
