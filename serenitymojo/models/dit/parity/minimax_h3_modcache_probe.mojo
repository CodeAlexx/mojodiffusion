# models/dit/parity/minimax_h3_modcache_probe.mojo — compile+run gate for
# models/dit/minimax_h3_modcache.minimax_h3_build_modulation_cache.
#
# Two phases:
#
#   1. CORRECTNESS (toy config, hidden_size=8). Builds the modulation table
#      two independent ways and compares them:
#        (a) the PRODUCTION path — minimax_h3_build_modulation_cache itself:
#            silu(temb) at F32, cast the RESULT to bf16, bf16 x bf16
#            linear_bias.
#        (b) a REFERENCE computed by the SAME ops/linear.mojo +
#            ops/activations.mojo GPU kernels, but never leaving F32 (the
#            adaLN weight bytes are the SAME bf16-quantized values the
#            checkpoint stores — upcast losslessly via Tensor.from_view_as_f32
#            — only the activation-rounding step differs).
#      Per house rule ([[no-cpu-parity-for-gpu]]: a GPU kernel's reference
#      must itself be GPU-generated, never CPU), (b) is NOT
#      block_forward.mojo's host-F32 oracle — it is a SECOND GPU computation.
#      Comparing (a) vs (b) isolates exactly the ACTIVATION PRECISION TRAP
#      (see minimax_h3_modcache.mojo's header) and, since (a)/(b) diverge
#      only there, a tight cosine bound is the right metric, not bit-exact.
#      A separate BIT-EXACT self-check (row-addressing) proves the
#      distinct_timesteps*3 reshape lines up with minimax_h3_adaln_row.
#
#   2. REAL-SHAPE SMOKE. The RELEASED config's real dimensions
#      (hidden_size=5376, time_embed_dim=2688, adaln_out_features=96768,
#      final_adaln_out_features=10752 — minimax_h3_released_config(), reused
#      verbatim) and the REAL ref2va@50 row count
#      (minimax_h3_adaln_distinct_timesteps(50, True, True) = 102, reused
#      from fp8_policy.mojo), but only 2 and 4 of the 50 blocks — every block
#      is shaped and streamed identically and independently, so this proves
#      the per-block behavior the full 50 would repeat, without writing the
#      ~26 GiB a full synthetic 50-block checkpoint would take (the real
#      checkpoint is still downloading — see the report). Runs the SAME
#      shard file at num_layers=2 and num_layers=4 and checks
#      cache.total_bytes() scales by EXACTLY 2 more [306,32256] bf16 block
#      tables — real production dimensions, real dtype, exact arithmetic,
#      no shape/dtype surprise at scale.
#
#      This phase does NOT measure device memory at runtime (an earlier
#      version sampled cuMemGetInfo_v2 before/after each build to show the
#      per-block weight isn't accumulating — that needs -Xlinker -lcuda,
#      which plain `mojo run -I .` cannot supply, so it was dropped rather
#      than block the house-standard gate invocation). The "one block's
#      weight resident at a time" claim rests on code structure instead:
#      `w`/`b` in minimax_h3_build_modulation_cache's loop are function-local
#      and are freed (Mojo end-of-scope Tensor destructor) before the next
#      iteration allocates the next block's — see that module's MEMORY
#      DISCIPLINE section. That is an architectural argument, not a
#      measurement; it was runtime-verified once (2-vs-4-block device-memory
#      delta, ~0 vs one block's 496 MiB) with the -lcuda build, which remains
#      valid to rerun by hand when you want a live number, just not as part
#      of this gate.
#
#      The full-50-block byte accounting (task deliverable (c)) is pure
#      arithmetic over the released config's real shapes, using
#      fp8_policy.mojo's ALREADY-GATED MiniMaxH3Fp8Budget — reused, not
#      reimplemented — since running 50 real-sized blocks isn't needed to
#      total the sizes any more than 2 real ones' worth of measurement plus
#      multiplication is.
#
#   pixi run mojo run -I . serenitymojo/models/dit/parity/minimax_h3_modcache_probe.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_remove
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.activations import silu
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.tensor_algebra import reshape_owned, zeros_device

from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
    minimax_h3_adaln_row,
)
from serenitymojo.models.dit.minimax_h3_modcache import (
    minimax_h3_block_adaln_weight_key,
    minimax_h3_block_adaln_bias_key,
    minimax_h3_check_modcache_weights,
    minimax_h3_build_modulation_cache,
    MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY,
    MINIMAX_H3_FINAL_ADALN_BIAS_KEY,
)
from serenitymojo.models.minimax_h3.fp8_policy import (
    minimax_h3_adaln_distinct_timesteps,
    MiniMaxH3Fp8Budget,
)

comptime TArc = ArcPointer[Tensor]
comptime TOY_CKPT = "/tmp/minimax_h3_modcache_probe_toy.safetensors"
comptime REAL_CKPT = "/tmp/minimax_h3_modcache_probe_real.safetensors"


# ── phase 1: toy config ───────────────────────────────────────────────────────
def _toy_config() -> MiniMaxH3DiTConfig:
    """Small config satisfying MiniMaxH3DiTConfig.validate()'s relationships
    (adaln_out_features == 6*hidden*3, final_adaln_out_features == 2*hidden).
    Every OTHER field is a placeholder — this module never reads them."""
    var hidden = 8
    return MiniMaxH3DiTConfig(
        hidden,           # hidden_size
        3,                # num_layers (blocks in the toy checkpoint)
        1, 1, hidden, 16, 1, 1, 4, 4,   # unused-here placeholders
        6,                # time_embed_dim
        6 * hidden * 3,   # adaln_out_features
        2 * hidden,       # final_adaln_out_features
        4,                # rope_inv_freq_len (unused here)
        Float32(1.0e-5), Float32(1.0e-5), Float32(1.0e-5),
    )


def _pattern(seed: Int, n: Int) -> List[Float32]:
    """Deterministic pseudo-random-looking F32 values in [-1.001, 1.001]."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        var v = ((seed * 1103515245 + i * 12345 + 7) % 2003) - 1001
        out.append(Float32(v) * Float32(0.001))
    return out^


def _write_toy_checkpoint(cfg: MiniMaxH3DiTConfig, ctx: DeviceContext) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    for layer in range(cfg.num_layers):
        var w_host = _pattern(layer * 2 + 1, cfg.adaln_out_features * cfg.time_embed_dim)
        names.append(minimax_h3_block_adaln_weight_key(layer))
        tensors.append(TArc(Tensor.from_host(
            w_host, [cfg.adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
        )))
        var b_host = _pattern(layer * 2 + 2, cfg.adaln_out_features)
        names.append(minimax_h3_block_adaln_bias_key(layer))
        tensors.append(TArc(Tensor.from_host(
            b_host, [cfg.adaln_out_features], STDtype.BF16, ctx
        )))
    var fw_host = _pattern(97, cfg.final_adaln_out_features * cfg.time_embed_dim)
    names.append(MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY)
    tensors.append(TArc(Tensor.from_host(
        fw_host, [cfg.final_adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
    )))
    var fb_host = _pattern(98, cfg.final_adaln_out_features)
    names.append(MINIMAX_H3_FINAL_ADALN_BIAS_KEY)
    tensors.append(TArc(Tensor.from_host(
        fb_host, [cfg.final_adaln_out_features], STDtype.BF16, ctx
    )))
    save_safetensors(names, tensors, String(TOY_CKPT), ctx)


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("probe: _max_abs_diff length mismatch")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < Float32(0.0):
            d = -d
        if d > m:
            m = d
    return m


def _run_correctness_phase(ctx: DeviceContext) raises:
    var cfg = _toy_config()
    var distinct_timesteps = 5
    _write_toy_checkpoint(cfg, ctx)

    var temb_host = _pattern(500, distinct_timesteps * cfg.time_embed_dim)
    var temb = Tensor.from_host(
        temb_host, [distinct_timesteps, cfg.time_embed_dim], STDtype.F32, ctx
    )

    var shards = ShardedSafeTensors.open(String(TOY_CKPT))
    minimax_h3_check_modcache_weights(shards, cfg)  # must not raise

    # (a) production path.
    var cache = minimax_h3_build_modulation_cache(shards, temb, cfg, ctx)

    var tags = cfg.adaln_rows_per_timestep()
    var block_width = 6 * cfg.hidden_size
    if cache.num_layers() != cfg.num_layers:
        raise Error("probe: FAIL cache.num_layers() mismatch")
    if cache.distinct_timesteps != distinct_timesteps:
        raise Error("probe: FAIL cache.distinct_timesteps mismatch")
    for layer in range(cfg.num_layers):
        if cache.block_mod[layer][].shape() != [distinct_timesteps * tags, block_width]:
            raise Error(String("probe: FAIL block ") + String(layer) + " cache shape")
    if cache.final_mod[].shape() != [distinct_timesteps, cfg.final_adaln_out_features]:
        raise Error("probe: FAIL final cache shape")

    # (b) GPU F32 reference: SAME temb (borrowed by (a), still valid — Tensor
    # is not consumed by a borrowed `temb: Tensor` parameter), same silu op,
    # SAME bf16-quantized weight VALUES upcast losslessly to F32, but no
    # activation-precision rounding in between.
    var ref_activated = silu(temb, ctx)
    var harness = ParityHarness(cos_threshold=0.999)

    for layer in range(cfg.num_layers):
        var wf = Tensor.from_view_as_f32(
            shards.tensor_view(minimax_h3_block_adaln_weight_key(layer)), ctx
        )
        var bf = Tensor.from_view_as_f32(
            shards.tensor_view(minimax_h3_block_adaln_bias_key(layer)), ctx
        )
        var ref_wide = linear_bias(ref_activated, wf, bf, ctx)
        var ref_mod = reshape_owned(ref_wide^, [distinct_timesteps * tags, block_width])
        var ref_host = ref_mod.to_host(ctx)
        var result = harness.compare(cache.block_mod[layer][], ref_host, ctx)
        print("  block", layer, "vs GPU-F32 reference:", result)
        if not result.passed:
            raise Error("probe: FAIL block modulation diverges from the GPU F32 reference")

    var fwf = Tensor.from_view_as_f32(
        shards.tensor_view(MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY), ctx
    )
    var fbf = Tensor.from_view_as_f32(
        shards.tensor_view(MINIMAX_H3_FINAL_ADALN_BIAS_KEY), ctx
    )
    var ref_final = linear_bias(ref_activated, fwf, fbf, ctx)
    var ref_final_host = ref_final.to_host(ctx)
    var final_result = harness.compare(cache.final_mod[], ref_final_host, ctx)
    print("  final layer vs GPU-F32 reference:", final_result)
    if not final_result.passed:
        raise Error("probe: FAIL final modulation diverges from the GPU F32 reference")

    # Bit-exact self-check: modulation_row(layer, ti, tag) must equal the
    # manually-addressed row inside the flat cache tensor — proves the
    # distinct_timesteps*3 reshape + minimax_h3_adaln_row agree (this is a
    # pure device-tensor byte extraction on both sides, so max_diff must be
    # EXACTLY zero, not merely close).
    var ti = 2
    var tag = 1
    var row_via_accessor = cache.modulation_row(0, ti, tag, ctx).to_host(ctx)
    var expected_row = minimax_h3_adaln_row(ti, tag)
    var full = cache.block_mod[0][].to_host(ctx)
    var manual_row = List[Float32](capacity=block_width)
    for c in range(block_width):
        manual_row.append(full[expected_row * block_width + c])
    var addr_diff = _max_abs_diff(row_via_accessor, manual_row)
    print("  row-addressing self-consistency max_diff:", addr_diff)
    if addr_diff != Float32(0.0):
        raise Error("probe: FAIL modulation_row does not match the manually-addressed row")

    _ = sys_remove(String(TOY_CKPT))
    print("phase 1 (correctness, toy config) PASS")


# ── phase 2: real-shape / memory-discipline smoke ─────────────────────────────
def _real_prefix_config(blocks: Int) raises -> MiniMaxH3DiTConfig:
    var real = minimax_h3_released_config()
    return MiniMaxH3DiTConfig(
        real.hidden_size, blocks, real.token_refiner_num_layers,
        real.num_attention_heads, real.attention_head_dim, real.ffn_hidden_size,
        real.latents_dim, real.audio_latents_dim, real.text_dim,
        real.timestep_input_dim, real.time_embed_dim, real.adaln_out_features,
        real.final_adaln_out_features, real.rope_inv_freq_len,
        real.norm_eps, real.qk_norm_eps, real.final_norm_eps,
    )


def _write_real_shape_checkpoint(cfg: MiniMaxH3DiTConfig, ctx: DeviceContext) raises:
    """Real released dimensions, `cfg.num_layers` blocks — VALUES are zeros
    (shape/dtype/memory-only smoke; correctness at these dtypes/dimensions is
    already proven in phase 1). zeros_device builds the tensors device-side
    with no per-element host loop, so this stays cheap even at real per-block
    size (~495.6 MiB bf16 weight per block)."""
    var names = List[String]()
    var tensors = List[TArc]()
    for layer in range(cfg.num_layers):
        names.append(minimax_h3_block_adaln_weight_key(layer))
        tensors.append(TArc(zeros_device(
            [cfg.adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
        )))
        names.append(minimax_h3_block_adaln_bias_key(layer))
        tensors.append(TArc(zeros_device([cfg.adaln_out_features], STDtype.BF16, ctx)))
    names.append(MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY)
    tensors.append(TArc(zeros_device(
        [cfg.final_adaln_out_features, cfg.time_embed_dim], STDtype.BF16, ctx
    )))
    names.append(MINIMAX_H3_FINAL_ADALN_BIAS_KEY)
    tensors.append(TArc(zeros_device([cfg.final_adaln_out_features], STDtype.BF16, ctx)))
    save_safetensors(names, tensors, String(REAL_CKPT), ctx)


def _build_and_total_bytes(
    shards: ShardedSafeTensors, temb: Tensor, cfg: MiniMaxH3DiTConfig, ctx: DeviceContext,
) raises -> Int:
    """Build the cache at real production dimensions and return
    cache.total_bytes(); the cache itself falls out of scope at return."""
    var cache = minimax_h3_build_modulation_cache(shards, temb, cfg, ctx)
    return cache.total_bytes()


def _run_real_shape_phase(ctx: DeviceContext) raises:
    var real = minimax_h3_released_config()
    var distinct_timesteps = minimax_h3_adaln_distinct_timesteps(50, True, True)
    print("  released config: hidden_size=", real.hidden_size, " num_layers=",
        real.num_layers, " adaln_out_features=", real.adaln_out_features,
        " final_adaln_out_features=", real.final_adaln_out_features)
    print("  ref2va@50 distinct_timesteps=", distinct_timesteps)

    var cfg4 = _real_prefix_config(4)
    _write_real_shape_checkpoint(cfg4, ctx)
    var shards = ShardedSafeTensors.open(String(REAL_CKPT))
    minimax_h3_check_modcache_weights(shards, cfg4)      # 4-block file, full check
    var cfg2 = _real_prefix_config(2)
    minimax_h3_check_modcache_weights(shards, cfg2)      # 2-block prefix also valid

    var temb = zeros_device([distinct_timesteps, real.time_embed_dim], STDtype.F32, ctx)

    var cache2_bytes = _build_and_total_bytes(shards, temb, cfg2, ctx)
    var cache4_bytes = _build_and_total_bytes(shards, temb, cfg4, ctx)
    print("  N=2 blocks: cache.total_bytes()=", cache2_bytes)
    print("  N=4 blocks: cache.total_bytes()=", cache4_bytes)

    var one_block_weight_bytes = real.adaln_out_features * real.time_embed_dim * 2
    var cache_delta = cache4_bytes - cache2_bytes
    print("  delta(4-2 blocks): cache.total_bytes()=", cache_delta,
        " -- one block's WEIGHT alone=", one_block_weight_bytes)

    # cache_delta is pure arithmetic (2 more [306,32256] bf16 block tables) at
    # REAL production dimensions and must be EXACT — this is the check that
    # actually proves the real-scale shapes/dtypes/reshape are right; see this
    # file's header for why a runtime device-memory measurement is not also
    # part of this gate.
    var expected_cache_delta = 2 * (distinct_timesteps * 3 * 6 * real.hidden_size) * 2
    if cache_delta != expected_cache_delta:
        raise Error("probe: FAIL cache_delta arithmetic mismatch")

    _ = sys_remove(String(REAL_CKPT))
    print("phase 2 (real-shape smoke, N=2 vs N=4 of 50) PASS")


# ── phase 3: full-50-block byte accounting (pure arithmetic, already-gated formula) ──
def _report_full_scale_accounting() raises:
    var real = minimax_h3_released_config()
    var distinct_timesteps = minimax_h3_adaln_distinct_timesteps(50, True, True)
    var budget = MiniMaxH3Fp8Budget()
    for layer in range(real.num_layers):
        budget.add(
            minimax_h3_block_adaln_weight_key(layer),
            real.adaln_out_features, real.time_embed_dim,
        )
        budget.add(minimax_h3_block_adaln_bias_key(layer), real.adaln_out_features, 0)
    budget.add(
        MINIMAX_H3_FINAL_ADALN_WEIGHT_KEY,
        real.final_adaln_out_features, real.time_embed_dim,
    )
    budget.add(MINIMAX_H3_FINAL_ADALN_BIAS_KEY, real.final_adaln_out_features, 0)

    var weight_bytes = budget.adaln_resident_bytes()
    var cache_bytes = budget.resident_bytes(distinct_timesteps)
    var weight_gib = Float64(weight_bytes) / Float64(1024 * 1024 * 1024)
    var cache_gib = Float64(cache_bytes) / Float64(1024 * 1024 * 1024)
    print("  50 blocks + final, ref2va@50 (distinct_timesteps=", distinct_timesteps, "):")
    print("    adaLN weight bytes replaced:", weight_bytes, "(", weight_gib, "GiB)")
    print("    modulation cache bytes kept:", cache_bytes, "(", cache_gib, "GiB)")


def main() raises:
    var ctx = DeviceContext()
    print("== phase 1: correctness (toy config) ==")
    _run_correctness_phase(ctx)
    print("== phase 2: real-shape smoke ==")
    _run_real_shape_phase(ctx)
    print("== phase 3: full-50-block byte accounting (arithmetic) ==")
    _report_full_scale_accounting()
    print("minimax_h3_modcache_probe PASS")
