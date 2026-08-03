# models/dit/parity/minimax_h3_modcache_modality_skeptic_probe.mojo
#
# SKEPTIC probe attacking one specific claim in minimax_h3_modcache_probe.mojo:
# "row-addressing self-consistency max_diff: 0.0".
#
# That check only proves `MiniMaxH3ModCache.modulation_row(layer, ti, tag)`
# agrees with MANUALLY indexing the same row of the same flat cache tensor at
# ONE FIXED (ti=2, tag=1) — i.e. it proves the ACCESSOR's slice matches a
# by-hand slice at the identical row index. It does NOT prove that different
# modalities (video=0, text=1, audio=2) at the SAME timestep actually carry
# DIFFERENT modulation content — a broken reshape or a broken adaln-row
# formula that mapped every tag to the same underlying bytes at a given
# timestep would pass that check too, since both sides of the comparison read
# through the identical row-index formula.
#
# This probe builds a cache from a REAL (if toy) checkpoint, then:
#   1. pulls modulation_row(layer, ti, tag) for tag in {0,1,2} at the SAME ti
#      and proves they are NOT bit-identical to each other (a positive
#      discrimination test, not just a self-consistency echo);
#   2. independently recomputes the RAW GEMM output (silu(temb) @ w.T + b)
#      itself, in THIS file, and manually slices the three
#      [tag*block_width : (tag+1)*block_width] column ranges of timestep ti's
#      96768-wide row, and checks modulation_row(layer, ti, tag) equals THAT
#      slice bit-exact for all three tags — proving the cache's per-modality
#      addressing is not just self-consistent but actually correct against an
#      independently-recomputed value.
#   3. also checks two DIFFERENT timesteps at the SAME tag differ (the
#      complementary axis — this one WAS already implicitly covered by phase 2's
#      byte-count check, but confirmed here directly on real content too).
#
#   pixi run mojo run -I . serenitymojo/models/dit/parity/minimax_h3_modcache_modality_skeptic_probe.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_remove
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear_bias

from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig, minimax_h3_adaln_row
from serenitymojo.models.dit.minimax_h3_modcache import (
    minimax_h3_block_adaln_weight_key,
    minimax_h3_block_adaln_bias_key,
    minimax_h3_check_modcache_weights,
    minimax_h3_build_modulation_cache,
    MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY,
    MINIMAX_H3_FINAL_ADALN_BIAS_KEY,
)

comptime TArc = ArcPointer[Tensor]
comptime CKPT = "/tmp/minimax_h3_modality_skeptic_toy.safetensors"


def _toy_config() -> MiniMaxH3DiTConfig:
    var hidden = 8
    return MiniMaxH3DiTConfig(
        hidden, 2, 1, 1, hidden, 16, 1, 1, 4, 4,
        6,                # time_embed_dim
        6 * hidden * 3,   # adaln_out_features
        2 * hidden,       # final_adaln_out_features
        4,
        Float32(1.0e-5), Float32(1.0e-5), Float32(1.0e-5),
    )


def _pattern(seed: Int, n: Int) -> List[Float32]:
    """Deterministic non-degenerate F32 values in [-1.001, 1.001] — SAME
    generator shape as the shipped probe's `_pattern`, different seeds so this
    checkpoint's content is independent of the shipped probe's toy fixture."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        var v = ((seed * 2654435761 + i * 40503 + 11) % 2003) - 1001
        out.append(Float32(v) * Float32(0.001))
    return out^


def _write_toy_checkpoint(cfg: MiniMaxH3DiTConfig, ctx: DeviceContext) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    for layer in range(cfg.num_layers):
        var w_host = _pattern(layer * 2 + 101, cfg.adaln_out_features * cfg.time_embed_dim)
        names.append(minimax_h3_block_adaln_weight_key(layer))
        tensors.append(TArc(Tensor.from_host(
            w_host, [cfg.adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
        )))
        var b_host = _pattern(layer * 2 + 102, cfg.adaln_out_features)
        names.append(minimax_h3_block_adaln_bias_key(layer))
        tensors.append(TArc(Tensor.from_host(
            b_host, [cfg.adaln_out_features], STDtype.BF16, ctx
        )))
    var fw_host = _pattern(197, cfg.final_adaln_out_features * cfg.time_embed_dim)
    names.append(MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY)
    tensors.append(TArc(Tensor.from_host(
        fw_host, [cfg.final_adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
    )))
    var fb_host = _pattern(198, cfg.final_adaln_out_features)
    names.append(MINIMAX_H3_FINAL_ADALN_BIAS_KEY)
    tensors.append(TArc(Tensor.from_host(
        fb_host, [cfg.final_adaln_out_features], STDtype.BF16, ctx
    )))
    save_safetensors(names, tensors, String(CKPT), ctx)


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("probe: length mismatch")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < Float32(0.0):
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    var cfg = _toy_config()
    var distinct_timesteps = 4
    _write_toy_checkpoint(cfg, ctx)

    var temb_host = _pattern(900, distinct_timesteps * cfg.time_embed_dim)
    var temb = Tensor.from_host(
        temb_host, [distinct_timesteps, cfg.time_embed_dim], STDtype.F32, ctx
    )

    var shards = ShardedSafeTensors.open(String(CKPT))
    minimax_h3_check_modcache_weights(shards, cfg)
    var cache = minimax_h3_build_modulation_cache(shards, temb, cfg, ctx)

    var block_width = 6 * cfg.hidden_size  # 48

    # ── 1+2: independent recomputation of the RAW [distinct_timesteps, 96768]
    # GEMM output, done HERE (not trusting the cache's own internal call),
    # then manual per-modality column slicing. Uses the SAME dtype path as
    # production (bf16-rounded activation x native-bf16 weights, per the
    # ACTIVATION PRECISION TRAP the module documents) rather than an F32
    # reference — an F32 reference would legitimately differ from production
    # by that (already separately gated, by the shipped probe) rounding
    # effect, which would contaminate THIS test's specific target: row
    # addressing, not activation precision.
    var w_bf16 = Tensor.from_view(
        shards.tensor_view(minimax_h3_block_adaln_weight_key(0)), ctx
    )
    var b_bf16 = Tensor.from_view(
        shards.tensor_view(minimax_h3_block_adaln_bias_key(0)), ctx
    )
    var activated_f32 = silu(temb, ctx)
    var activated_bf16 = cast_tensor(activated_f32, STDtype.BF16, ctx)
    var raw = linear_bias(activated_bf16, w_bf16, b_bf16, ctx)  # [distinct_timesteps, 3*block_width], F32-accum
    var raw_host = raw.to_host(ctx)
    var raw_width = cfg.adaln_out_features  # 3*block_width = 144

    var ti = 2
    print("checking layer=0, timestep_index=", ti, "block_width=", block_width,
          "raw_width=", raw_width)

    var rows_by_tag = List[List[Float32]]()
    for tag in range(3):
        var expected_row = minimax_h3_adaln_row(ti, tag)
        var manual = List[Float32](capacity=block_width)
        for c in range(block_width):
            manual.append(raw_host[ti * raw_width + tag * block_width + c])
        var accessor = cache.modulation_row(0, ti, tag, ctx).to_host(ctx)
        var diff = _max_abs_diff(accessor, manual)
        print("  tag=", tag, " adaln_row=", expected_row,
              " accessor-vs-independent-recompute max_diff=", diff,
              " (expect EXACTLY 0.0 — this is the real correctness check,"
              " not mere self-consistency)")
        if diff != Float32(0.0):
            raise Error("probe: FAIL modulation_row diverges from an independently recomputed slice")
        rows_by_tag.append(manual^)

    # ── The actual DISCRIMINATION test: are the three modality rows at the
    # SAME timestep genuinely different from each other? A broken reshape or
    # a formula collapsing all three tags to the same row would sail through
    # every check above (both sides would read the SAME wrong bytes) but MUST
    # fail this one, because the underlying GEMM output legitimately differs
    # per 32256-wide (here 16-wide toy) column block — different weight rows,
    # different bias entries, different silu(temb) columns feed each block.
    var d01 = _max_abs_diff(rows_by_tag[0], rows_by_tag[1])
    var d02 = _max_abs_diff(rows_by_tag[0], rows_by_tag[2])
    var d12 = _max_abs_diff(rows_by_tag[1], rows_by_tag[2])
    print("  video(0) vs text(1) max_diff:", d01)
    print("  video(0) vs audio(2) max_diff:", d02)
    print("  text(1) vs audio(2) max_diff:", d12)
    var MIN_DISCRIMINATION = Float32(1.0e-4)
    if d01 < MIN_DISCRIMINATION or d02 < MIN_DISCRIMINATION or d12 < MIN_DISCRIMINATION:
        raise Error(
            "probe: FAIL the three modality rows at the same timestep are"
            " suspiciously close/identical — the modality dimension may be"
            " silently collapsed"
        )

    # ── Complementary axis: two DIFFERENT timesteps at the SAME tag must also
    # differ (content-level check, not just the byte-count check phase 2 of
    # the shipped probe already does).
    var tag_fixed = 1
    var row_t0 = cache.modulation_row(0, 0, tag_fixed, ctx).to_host(ctx)
    var row_t3 = cache.modulation_row(0, 3, tag_fixed, ctx).to_host(ctx)
    var dt = _max_abs_diff(row_t0, row_t3)
    print("  text(1) @ t=0 vs text(1) @ t=3 max_diff:", dt)
    if dt < MIN_DISCRIMINATION:
        raise Error("probe: FAIL different timesteps at the same modality look suspiciously identical")

    _ = sys_remove(String(CKPT))
    print("minimax_h3_modcache_modality_skeptic_probe PASS")
    print("(modality dimension genuinely discriminates, cross-verified against an")
    print(" independent recomputation of the raw GEMM output, not just self-consistency)")
