# serenitymojo/models/dit/parity/minimax_h3_final_layer_real_weight_gate.mojo
#
# MiniMax-H3 FINAL-LAYER parity gate against REAL checkpoint bytes — shard 13
# (the last missing shard) landed, so this is the first thing in the port to
# run real `final_layer.*` bytes through anything. Two things under test,
# chained exactly as production would chain them:
#
#   [1] `minimax_h3_build_modulation_cache`'s FINAL-LAYER path
#       (models/dit/minimax_h3_modcache.mojo) — real
#       `final_layer.adaln_proj.linear.{weight,bias}` -> `cache.final_mod`.
#       This is the piece minimax_h3_modcache_real_weight_gate.mojo had to
#       stub with a random stand-in file, because shard 13 did not exist yet.
#   [2] `minimax_h3_final_layer` (models/dit/minimax_h3_frontend.mojo) fed
#       `cache.final_mod` DIRECTLY — real `final_layer.norm.weight` (BF16)
#       and real `final_layer.video_out`/`audio_out.{weight,bias}` (F32, NOT
#       downcast) — against diffusers' own final-layer chain
#       (`MiniMaxH3AdaLayerNormOut.forward(...).to(proj_out.weight.dtype)`
#       then the two real F32 `nn.Linear` heads, transformer_minimax_h3.py
#       :638-640).
#
# TWO REAL SHARDS, NO STAND-IN: `config.validate()` (called first thing
# inside `minimax_h3_build_modulation_cache`) rejects `num_layers<=0`
# outright, so this gate cannot skip the per-block preflight by zeroing
# `num_layers` the way it first tried. Instead it uses `num_layers=1` and
# merges shard 1 (`blocks.0.adaln_proj.linear.*`, real bytes, already
# validated by the earlier block/modcache gates) with shard 13 (every
# `final_layer.*` tensor) via `ShardedSafeTensors`'s own public constructor —
# unlike `minimax_h3_modcache_real_weight_gate.mojo`, which needed a random
# stand-in file for the final layer because shard 13 didn't exist yet, THIS
# gate needs no synthetic bytes anywhere: both halves of the merge are real.
#
# THE THING MOST LIKELY TO BE WRONG (team-lead's own framing, verified by
# reading the code before running it — see the two files' docstrings):
# diffusers computes RMSNorm + the timestep-indexed shift/scale modulation
# ENTIRELY IN BF16 (both `norm_out.norm` and `norm_out.linear` are bf16
# checkpoint tensors) and casts to F32 ONLY AFTER modulation, immediately
# before the F32 output heads (transformer_minimax_h3.py:638). But
# `ops/linear.mojo`'s own docstring says `linear`/`linear_bias` "returns
# ...x's dtype" — and `minimax_h3_build_modulation_cache` calls
# `linear_bias(activated, ...)` where `activated` is explicitly cast to BF16
# (minimax_h3_modcache.mojo:311) BEFORE that call. So `cache.final_mod`'s
# ACTUAL runtime dtype is BF16, not the F32 its own docstring claims
# ("kept F32 through this port") — while `minimax_h3_final_layer` explicitly
# upcasts `final_normed` to F32 BEFORE combining it with the modulation table
# (minimax_h3_frontend.mojo:442, `final_normed_f32 = cast_tensor(...)`),
# because it assumes an F32 modulation table. `ops/tensor_algebra.mojo`'s
# `_binary` (the shared add/mul launcher) REQUIRES `a.dtype() == b.dtype()`
# and raises a named error otherwise (tensor_algebra.mojo:338-348) — it does
# NOT silently promote. This gate feeds the REAL modcache output into the
# REAL final-layer function to find out, empirically, whether that
# type-level assumption mismatch is live or dormant, rather than asserting
# it either way.
#
# BAR: cos >= 0.999, max_abs AND magnitude ratio |mine|/|ref| for BOTH
# checks. If check [2] raises instead of producing a number, that IS the
# result — report the exact error, do not paper over it by hand-casting
# `cache.final_mod` before this gate calls `minimax_h3_final_layer` (that
# would hide exactly the integration gap this gate exists to find).
#
# Run:
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/dit/parity/minimax_h3_final_layer_real_weight_oracle.py
#   pixi run scripts/mem_safe.sh mojo build --optimization-level 2 -j 1 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/dit/parity/minimax_h3_final_layer_real_weight_gate.mojo \
#     -o output/checks/minimax_h3_final_layer_real_weight_gate \
#   && output/checks/minimax_h3_final_layer_real_weight_gate
#   (no cuDNN shim needed — this gate never calls attention.)

from std.collections import Dict, List
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

from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.models.dit.minimax_h3_modcache import minimax_h3_build_modulation_cache
from serenitymojo.models.dit.minimax_h3_frontend import minimax_h3_final_layer

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/final_layer_real_weight_ref.safetensors"
comptime SHARD1 = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/"
    "model-00001-of-00013.safetensors"
)
comptime SHARD13 = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/"
    "model-00013-of-00013.safetensors"
)

comptime S = 18  # same packed S as the other real-weight gates


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


def _open_shard1_plus_shard13() raises -> ShardedSafeTensors:
    """Merge two REAL single-file shards via ShardedSafeTensors' own public
    constructor — shard 1 (blocks.0.adaln_proj, needed only so
    config.num_layers=1 passes preflight) and shard 13 (every final_layer.*
    tensor). No directory open (fails on the 4 still-missing shards), no
    dummy/stand-in bytes anywhere — both halves are real."""
    var s1 = SafeTensors.open(String(SHARD1))
    var s13 = SafeTensors.open(String(SHARD13))
    var shards = List[ArcPointer[SafeTensors]]()
    var name_to_shard = Dict[String, Int]()
    for ref nm in s1.names():
        name_to_shard[nm] = 0
    shards.append(ArcPointer(s1^))
    for ref nm in s13.names():
        name_to_shard[nm] = 1
    shards.append(ArcPointer(s13^))
    return ShardedSafeTensors(shards^, name_to_shard^)


def _report(
    label: String, got: Tensor, ref_flat: List[Float32], ctx: DeviceContext
) raises -> Bool:
    var out_f32 = cast_tensor(got, STDtype.F32, ctx)
    var harness = ParityHarness(0.999)
    var r = harness.compare(out_f32, ref_flat, ctx)

    var actual = out_f32.to_host(ctx)
    var norm_actual = Float64(0.0)
    var norm_ref = Float64(0.0)
    for i in range(len(actual)):
        norm_actual += Float64(actual[i]) * Float64(actual[i])
    for i in range(len(ref_flat)):
        norm_ref += Float64(ref_flat[i]) * Float64(ref_flat[i])
    norm_actual = sqrt(norm_actual)
    norm_ref = sqrt(norm_ref)
    var ratio = Float64(-1.0)
    if norm_ref != 0.0:
        ratio = norm_actual / norm_ref

    print("  ", label, r)
    print("     |mine| =", norm_actual, " |ref| =", norm_ref, " ratio |mine|/|ref| =", ratio)
    return r.passed


def main() raises:
    print("MiniMax-H3 FINAL-LAYER parity gate — REAL adaln_proj/norm/video_out/audio_out bytes")
    print("  reference:", REF)
    print("  checkpoint shard:", SHARD13)

    var st_ref = SafeTensors.open(String(REF))
    var hidden_h = _load_f32(st_ref, "in.hidden_states")
    var temb_h = _load_f32(st_ref, "in.temb")
    var timestep_indices = _load_i64_as_int(st_ref, "in.timestep_indices")
    var video_indices = _load_i64_as_int(st_ref, "in.video_indices")
    var audio_indices = _load_i64_as_int(st_ref, "in.audio_indices")
    var ref_final_mod = _load_f32(st_ref, "out.final_mod")
    var ref_video_out = _load_f32(st_ref, "out.video_out")
    var ref_audio_out = _load_f32(st_ref, "out.audio_out")

    var hidden = 5376
    var time_embed_dim = 2688
    var num_timesteps = len(temb_h) // time_embed_dim

    print("")
    print("[fixture] S=", len(timestep_indices), " hidden=", hidden, " num_timesteps=", num_timesteps)
    print("  video_indices:", len(video_indices), " audio_indices:", len(audio_indices))

    # num_layers=1: config.validate() (called first thing inside
    # minimax_h3_build_modulation_cache) rejects num_layers<=0 outright, so 0
    # is not available as a "skip the per-block preflight" trick. Instead:
    # blocks.0.adaln_proj.linear.{weight,bias} are ALSO real bytes, already
    # on shard 1 (verified by the earlier block/modcache gates) — so this
    # gate merges shard 1 + shard 13 (both real, no stand-in file anywhere)
    # and lets minimax_h3_build_modulation_cache do its normal 1-layer loop
    # plus its unconditional final-layer check, both against real bytes.
    var config = MiniMaxH3DiTConfig(
        hidden,                # hidden_size
        1,                     # num_layers -- 1 real block (blocks.0, shard 1)
        2,                     # token_refiner_num_layers (unused here)
        56, 128, 14336,        # heads, head_dim, ffn (unused here)
        24, 32, 5120,          # latents_dim, audio_latents_dim, text_dim (unused here)
        256,                   # timestep_input_dim (unused here)
        time_embed_dim,        # time_embed_dim
        6 * hidden * 3,        # adaln_out_features (unused when num_layers=0)
        2 * hidden,            # final_adaln_out_features (10752)
        16,                    # rope_inv_freq_len (unused here)
        Float32(1.0e-5),
        Float32(1.0e-5),
        Float32(1.0e-5),
    )

    var ctx = DeviceContext()

    print("")
    print("[device] opening shard 1 + shard 13 (both real, no stand-in) ...")
    var shards = _open_shard1_plus_shard13()
    print("  shards:", shards.num_shards(), " tensors:", shards.num_tensors())

    var temb = Tensor.from_host(temb_h.copy(), [num_timesteps, time_embed_dim], STDtype.F32, ctx)

    print("")
    print("[1] minimax_h3_build_modulation_cache — real final_layer.adaln_proj bytes")
    var cache = minimax_h3_build_modulation_cache(shards, temb, config, ctx)
    print("  cache.num_layers()=", cache.num_layers(), " (0, as configured) final_mod dtype=", cache.final_mod[].dtype().name())
    var mod_ok = _report("final_mod (modcache projection alone)", cache.final_mod[], ref_final_mod, ctx)

    var w = Dict[String, ArcPointer[Tensor]]()
    w["final_layer.norm.weight"] = ArcPointer(Tensor.from_view(shards.tensor_view("final_layer.norm.weight"), ctx))
    w["final_layer.video_out.weight"] = ArcPointer(Tensor.from_view(shards.tensor_view("final_layer.video_out.weight"), ctx))
    w["final_layer.video_out.bias"] = ArcPointer(Tensor.from_view(shards.tensor_view("final_layer.video_out.bias"), ctx))
    w["final_layer.audio_out.weight"] = ArcPointer(Tensor.from_view(shards.tensor_view("final_layer.audio_out.weight"), ctx))
    w["final_layer.audio_out.bias"] = ArcPointer(Tensor.from_view(shards.tensor_view("final_layer.audio_out.bias"), ctx))
    print(
        "  final_layer.norm.weight dtype=", w["final_layer.norm.weight"][].dtype().name(),
        " video_out.weight dtype=", w["final_layer.video_out.weight"][].dtype().name(),
        " audio_out.weight dtype=", w["final_layer.audio_out.weight"][].dtype().name(),
    )

    var hidden_bf16 = Tensor.from_host(hidden_h.copy(), [S, hidden], STDtype.BF16, ctx)

    print("")
    print("[2] minimax_h3_final_layer — fed cache.final_mod DIRECTLY (no hand-cast, no stub)")
    var video_ok = False
    var audio_ok = False
    var crashed = False
    try:
        var out = minimax_h3_final_layer(
            hidden_bf16, cache.final_mod[], timestep_indices, video_indices, audio_indices, w, config, ctx,
        )
        video_ok = _report("video_out", out.video_out, ref_video_out, ctx)
        audio_ok = _report("audio_out", out.audio_out, ref_audio_out, ctx)
    except e:
        crashed = True
        print("   CRASHED:", e)
        print("   This is a real integration defect, not a numeric miss — see this")
        print("   file's header for the exact file:line diagnosis. NOT worked")
        print("   around here; reported as-is.")

    if crashed:
        print("")
        print("[2b] DIAGNOSTIC ONLY (not a fix, not production code, not applied to")
        print("     minimax_h3_frontend.mojo or minimax_h3_modcache.mojo): explicitly")
        print("     upcasting cache.final_mod to F32 in THIS GATE ALONE, purely to see")
        print("     whether the rest of the pipeline agrees with the reference once the")
        print("     dtype crash is bypassed, or whether a SECOND (precision-order) issue")
        print("     is also hiding behind it.")
        var final_mod_f32 = cast_tensor(cache.final_mod[], STDtype.F32, ctx)
        var out2 = minimax_h3_final_layer(
            hidden_bf16, final_mod_f32, timestep_indices, video_indices, audio_indices, w, config, ctx,
        )
        _ = _report("  [diagnostic] video_out (final_mod force-cast to F32)", out2.video_out, ref_video_out, ctx)
        _ = _report("  [diagnostic] audio_out (final_mod force-cast to F32)", out2.audio_out, ref_audio_out, ctx)

    print("")
    if mod_ok and video_ok and audio_ok:
        print("ALL CHECKS PASS")
    elif crashed:
        raise Error(
            "MiniMax-H3 final-layer real-weight gate: minimax_h3_final_layer"
            " crashed on the REAL modcache output — see [2]'s printed error"
            " and this file's header for the diagnosis. This is the result,"
            " not a gate bug."
        )
    else:
        raise Error("MiniMax-H3 final-layer real-weight gate: one or more checks FAILED")
