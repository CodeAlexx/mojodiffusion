# serenitymojo/models/dit/parity/minimax_h3_modcache_real_weight_gate.mojo
#
# MiniMax-H3 AdaLN modulation-cache parity gate against REAL checkpoint bytes
# — `blocks.0/1.adaln_proj.linear.{weight,bias}`, 13.04B params / 39% of the
# model over the full 50-layer stack, and until this gate the largest
# unverified surface in the whole port: nothing had ever run real adaln_proj
# bytes through anything. Companion to minimax_h3_block_real_weight_gate.mojo
# (the OTHER 8 block tensors); together they cover every real bf16 tensor
# blocks 0/1 own.
#
# UNIT UNDER TEST: `minimax_h3_build_modulation_cache`
# (models/dit/minimax_h3_modcache.mojo) — the REAL, UNMODIFIED entry point,
# not its constituent ops re-called by hand. That function's preflight
# (`minimax_h3_check_modcache_weights`) unconditionally checks EVERY layer up
# to `config.num_layers` AND the final layer, before loading anything — so
# this gate:
#   (a) uses a `num_layers=2` config (blocks 0/1 are the only complete real
#       adaln_proj bytes on disk right now — shards 2 and 9-13 are missing)
#   (b) supplies a shape/dtype-correct STAND-IN for `final_layer.adaln_proj
#       .linear.*` (real bytes live in the missing shard 13) so the real
#       preflight passes without faking the block-0/1 result this gate
#       actually measures. See minimax_h3_modcache_real_weight_oracle.py's
#       `write_dummy_final_layer_shard` — random bf16, correct shape, value
#       NEVER read by anything that compares numbers. `final_mod` is computed
#       (the real function always computes it) and immediately IGNORED here.
# This gate does not weaken the guard or the preflight — it feeds real data
# to a real function and stands in only for bytes that are out of scope and
# genuinely absent from disk.
#
# WHY MERGE TWO SHARDS INSTEAD OF POINTING AT THE CHECKPOINT DIRECTORY:
# `ShardedSafeTensors.open(directory)` eagerly opens EVERY shard the index
# references and fails on the first missing one (confirmed against this
# exact checkpoint by minimax_h3_block_real_weight_gate.mojo). This gate
# builds a `ShardedSafeTensors` directly from its own public constructor
# (`ShardedSafeTensors.__init__(shards, name_to_shard)`, sharded.mojo) over
# TWO independently-opened single files: the real shard 1
# (`model-00001-of-00013.safetensors`, via the documented single-file
# fallback also used by the block gate) and the dummy final-layer file. No
# directory open, no missing-shard error, no symlink, no hand-edited index.
#
# ORACLE: `minimax_h3_modcache_real_weight_oracle.py` runs diffusers' OWN
# `MiniMaxH3AdaLayerNormModulation` (not a hand transcription) on the SAME
# real blocks.0/1 weights, GPU bf16.
#
# ALSO CHECKED (not just the numeric parity):
#   [2] the ACTIVATION-PRECISION GUARD actually fires on a bf16 `temb` — the
#       oracle script's own SiLU-order probe (max diff 0.03125 over a
#       [-6,6] grid) already proves the F32-vs-bf16 SiLU order is
#       discriminating, not tautological; this gate proves the Mojo function
#       refuses the wrong order rather than silently computing it.
#   [3] row differentiation ON REAL BYTES: video/text/audio rows at the same
#       timestep must genuinely differ (an earlier skeptic measured 2.4-2.7
#       on SYNTHETIC weights; this repeats the measurement on the real
#       blocks.0/1 adaln_proj output via `MiniMaxH3ModCache.modulation_row`).
#
# BAR: cos >= 0.999, max_abs AND magnitude ratio |mine|/|ref| reported
# alongside — same reasoning as the block gate.
#
# Run:
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/dit/parity/minimax_h3_modcache_real_weight_oracle.py
#   pixi run scripts/mem_safe.sh mojo build --optimization-level 2 -j 1 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/dit/parity/minimax_h3_modcache_real_weight_gate.mojo \
#     -o output/checks/minimax_h3_modcache_real_weight_gate \
#   && output/checks/minimax_h3_modcache_real_weight_gate
#   (no cuDNN shim needed here — this gate never calls attention.)

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

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/modcache_real_weight_ref.safetensors"
comptime SHARD1 = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/"
    "model-00001-of-00013.safetensors"
)
comptime DUMMY_FINAL_LAYER = (
    "/home/alex/mojodiffusion/output/minimax_h3_block/h3_dummy_final_layer.safetensors"
)

comptime NUM_LAYERS_TESTED = 2


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


def _open_shard1_plus_final_layer_stub() raises -> ShardedSafeTensors:
    """Merge real shard 1 with the dummy final-layer file into one
    `ShardedSafeTensors`, via the struct's own public constructor — no
    directory open (which would fail on the missing shards), no symlink, no
    hand-edited index.json."""
    var real = SafeTensors.open(String(SHARD1))
    var dummy = SafeTensors.open(String(DUMMY_FINAL_LAYER))

    var shards = List[ArcPointer[SafeTensors]]()
    var name_to_shard = Dict[String, Int]()
    for ref nm in real.names():
        name_to_shard[nm] = 0
    shards.append(ArcPointer(real^))
    for ref nm in dummy.names():
        name_to_shard[nm] = 1
    shards.append(ArcPointer(dummy^))
    return ShardedSafeTensors(shards^, name_to_shard^)


def _compare_layer(
    label: String,
    got: Tensor,
    ref_flat: List[Float32],
    ctx: DeviceContext,
) raises:
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
    if not r.passed:
        raise Error(String("MiniMax-H3 modcache real-weight gate: ") + label + " FAILED")


def main() raises:
    print("MiniMax-H3 modulation-cache parity gate — REAL adaln_proj bytes (blocks 0/1)")
    print("  reference:", REF)
    print("  checkpoint shard:", SHARD1)
    print("  final-layer stand-in (out of scope, shape-only):", DUMMY_FINAL_LAYER)

    var st_ref = SafeTensors.open(String(REF))
    var temb_h = _load_f32(st_ref, "in.temb")
    var ref0 = _load_f32(st_ref, "out.block_mod.0")
    var ref1 = _load_f32(st_ref, "out.block_mod.1")

    var hidden = 5376
    var time_embed_dim = 2688
    var num_timesteps = len(temb_h) // time_embed_dim
    if num_timesteps * time_embed_dim != len(temb_h):
        raise Error("in.temb size is not a multiple of time_embed_dim")

    print("")
    print("[fixture] num_timesteps=", num_timesteps, " hidden=", hidden, " time_embed_dim=", time_embed_dim)

    # Released geometry, but num_layers=2 — the only real adaln_proj bytes
    # currently on disk. Same fields minimax_h3_released_config() hardcodes,
    # otherwise; config.validate() (called inside the unit under test) still
    # must pass, and does: adaln_out_features/final_adaln_out_features are
    # geometry-derived, not layer-count-derived.
    var config = MiniMaxH3DiTConfig(
        hidden,                # hidden_size
        NUM_LAYERS_TESTED,     # num_layers (NOT 50 — see header)
        2,                     # token_refiner_num_layers
        56,                    # num_attention_heads
        128,                   # attention_head_dim
        14336,                 # ffn_hidden_size
        24,                    # latents_dim
        32,                    # audio_latents_dim
        5120,                  # text_dim
        256,                   # timestep_input_dim
        time_embed_dim,        # time_embed_dim
        6 * hidden * 3,        # adaln_out_features (96768)
        2 * hidden,            # final_adaln_out_features (10752)
        16,                    # rope_inv_freq_len
        Float32(1.0e-5),
        Float32(1.0e-5),
        Float32(1.0e-5),
    )

    var ctx = DeviceContext()

    print("")
    print("[device] building the merged ShardedSafeTensors (real shard 1 + final-layer stand-in) ...")
    var shards = _open_shard1_plus_final_layer_stub()
    print("  shards:", shards.num_shards(), " tensors:", shards.num_tensors())

    var temb = Tensor.from_host(temb_h.copy(), [num_timesteps, time_embed_dim], STDtype.F32, ctx)

    print("")
    print("[device] minimax_h3_build_modulation_cache(shards, temb F32, config[num_layers=2]) ...")
    var cache = minimax_h3_build_modulation_cache(shards, temb, config, ctx)
    print("  cache.num_layers() =", cache.num_layers(), " distinct_timesteps =", cache.distinct_timesteps)
    print("  (final_mod computed as the real function always does; its bytes are a")
    print("   stand-in and are NOT compared against anything — out of scope)")

    print("")
    print("[1] numeric parity vs the diffusers-class oracle, per real layer")
    _compare_layer("layer 0", cache.block_mod[0][], ref0, ctx)
    _compare_layer("layer 1", cache.block_mod[1][], ref1, ctx)

    print("")
    print("[2] activation-precision GUARD: a pre-downcast (BF16) temb must be rejected")
    var temb_bf16 = cast_tensor(temb, STDtype.BF16, ctx)
    var guard_fired = False
    try:
        _ = minimax_h3_build_modulation_cache(shards, temb_bf16, config, ctx)
    except e:
        guard_fired = True
        print("   ok, raised:", e)
    if not guard_fired:
        raise Error("MiniMax-H3 modcache real-weight gate: guard did NOT fire on a bf16 temb")

    print("")
    print("[3] row differentiation on REAL bytes: video/text/audio rows at timestep 0 must genuinely differ")
    for layer in range(NUM_LAYERS_TESTED):
        var video = cache.modulation_row(layer, 0, 0, ctx).to_host(ctx)
        var text = cache.modulation_row(layer, 0, 1, ctx).to_host(ctx)
        var audio = cache.modulation_row(layer, 0, 2, ctx).to_host(ctx)
        var d_vt = Float64(0.0)
        var d_va = Float64(0.0)
        var d_ta = Float64(0.0)
        for i in range(len(video)):
            var vt = Float64(video[i]) - Float64(text[i])
            var va = Float64(video[i]) - Float64(audio[i])
            var ta = Float64(text[i]) - Float64(audio[i])
            d_vt += vt * vt
            d_va += va * va
            d_ta += ta * ta
        print(
            "   layer", layer, "row L2 distances @ t0: video<->text=", sqrt(d_vt),
            " video<->audio=", sqrt(d_va), " text<->audio=", sqrt(d_ta),
        )
        if sqrt(d_vt) < 1.0 or sqrt(d_va) < 1.0 or sqrt(d_ta) < 1.0:
            raise Error(
                "MiniMax-H3 modcache real-weight gate: modality rows at the"
                " same timestep are suspiciously close — row addressing may"
                " be broken"
            )

    print("")
    print("ALL CHECKS PASS")
