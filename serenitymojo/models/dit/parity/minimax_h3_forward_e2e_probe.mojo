# serenitymojo/models/dit/parity/minimax_h3_forward_e2e_probe.mojo
#
# WIRING GATE for `minimax_h3_frontend_embed -> minimax_h3_run_stack ->
# minimax_h3_final_layer`, end to end on REAL block weights (blocks 0 and 1,
# via the real checkpoint's shard 1) and REAL frontend weights (video/audio/
# condition patch-proj, time_embedder, token_refiner blocks 0-1 — all also
# in shard 1). `final_layer.*` (norm, video_out, audio_out, adaln_proj) is
# ENTIRELY in shard 13, which has not landed — every final_layer weight AND
# the final-adaLN modulation table are synthetic, and said so at each use.
#
# THIS IS NOT A NUMERIC PARITY GATE. No oracle is compared against here —
# h3-block-gate owns the real-weight oracles (adaLN, then the 2-layer stack).
# This file only proves the SEAMS between three independently-gated units
# are wired correctly: shapes, index round-trips, and that per-row
# differentiation (adaLN) actually propagates through the full composition,
# not just within one block in isolation. Per the porting doctrine — "gate
# every piece, THEN wire; the integration run IS the wiring test" — every
# piece here is independently gated/probed elsewhere; the risk this file
# targets is a wrong seam (a swapped index, a transposed boundary, a
# mis-scattered row) that no per-piece gate could see.
#
# FOUR CHECKS:
#   [1] SCATTER BIT-EXACT ROUND TRIP. `minimax_h3_frontend_embed` scatters
#       video/audio/text embeds into one packed buffer via
#       `index_select_backward` reused as a plain scatter (disjoint index
#       sets). Independently recompute video/audio/text embeds via the SAME
#       public sub-functions frontend_embed calls internally, then
#       `gather_rows(hidden, indices)` at each modality's OWN index list and
#       diff against the independently-computed embeds. Any nonzero diff
#       means the scatter placed the wrong row somewhere — max_abs reported,
#       not just asserted zero.
#   [2] PERMUTATION INVARIANCE (the transposition/off-by-one detector).
#       Route the SAME per-row content through TWO different physical
#       packed-position assignments (same index SETS per modality, reversed
#       WITHIN each modality) with a uniform single timestep and
#       num_layers=0 (pure reshape passthrough — no rope/attention, so
#       physical position carries no information). If scatter and
#       final_layer's gather consistently pair "logical row k" to whatever
#       physical position it was routed through, the two runs' OUTPUTS must
#       be IDENTICAL. A transposed or off-by-one index in either direction
#       breaks this while leaving shapes and NaN-checks perfectly clean.
#   [3] ADALN DIFFERENTIATION SURVIVES THE FULL COMPOSITION. Same packed
#       sequence, REAL 2-layer stack, run TWICE — once with the real
#       per-(timestep,tag) adaln_indices, once with adaln_indices forced flat
#       (every row reads modulation row 0). The two runs' stack outputs must
#       DIFFER. (This extends — does not repeat — the single-block
#       differential check from minimax_h3_block_device_probe.mojo: that one
#       proved one block CAN differentiate; this proves the composition
#       actually WIRES the per-row table through frontend -> 2 real blocks.)
#   [4] SHAPES AT EVERY SEAM + NaN/Inf-free + MAGNITUDES, on the real 2-layer
#       full run. Every boundary is asserted explicitly (named error on
#       mismatch), not left to a silent broadcast; magnitudes are PRINTED,
#       not just checked non-NaN.
#
# Run (package-relative imports need -I .; sdpa_flash_infer_fwd needs the
# cuDNN shim explicitly linked — bare `mojo run -I .` fails with
# "JIT session error: Symbols not found: [ flame_cudnn_sdpa_bf16 ]"):
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -Xlinker -lm \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/dit/parity/minimax_h3_forward_e2e_probe.mojo

from std.collections import Dict, List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import gather_rows
from serenitymojo.models.minimax_h3.dit_frontend import (
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
    minimax_h3_adaln_rows,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
    minimax_h3_load_qkv_device,
    minimax_h3_load_fc1_device,
)
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache
from serenitymojo.models.dit.minimax_h3_stack import minimax_h3_run_stack
from serenitymojo.models.dit.minimax_h3_frontend import (
    MiniMaxH3FrontendEmbed,
    MiniMaxH3FrontendOutput,
    minimax_h3_frontend_embed,
    minimax_h3_video_patch_embed,
    minimax_h3_audio_patch_embed,
    minimax_h3_condition_embed,
    minimax_h3_token_refiner,
    minimax_h3_final_layer,
)

comptime CHECKPOINT_FILE = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/model-00001-of-00013.safetensors"
comptime TR_S = 3
comptime TR_H = 56
comptime TR_DH = 128
comptime S = 8
comptime NV = 3
comptime NA = 2
comptime NT = 3


def _mag(t: Tensor, ctx: DeviceContext) raises -> Tuple[Float32, Float32, Int]:
    """(mean_abs, max_abs, nan_count) — magnitudes REPORTED, not just
    asserted finite."""
    var host = t.to_host(ctx)
    var sum_abs = Float32(0.0)
    var max_abs = Float32(0.0)
    var n_bad = 0
    for i in range(len(host)):
        var v = host[i]
        if v != v:
            n_bad += 1
            continue
        var av = v if v >= Float32(0.0) else -v
        sum_abs += av
        if av > max_abs:
            max_abs = av
    var mean_abs = sum_abs / Float32(max(len(host), 1))
    return (mean_abs, max_abs, n_bad)


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("_max_abs_diff: numel mismatch")
    var m = Float32(0.0)
    for i in range(len(ha)):
        var d = ha[i] - hb[i]
        var ad = d if d >= Float32(0.0) else -d
        if ad > m:
            m = ad
    return m


def _build_frontend_weights(
    st: ShardedSafeTensors, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    """REAL video/audio/condition patch-proj + time_embedder + token_refiner
    weights from shard 1 (verified present against the real index.json's
    weight_map before writing this — not guessed). token_refiner's own
    qkv_proj/fc1 need the SAME de-interleave/swap as the main blocks
    (`_minimax_h3_token_refiner_block`'s own header says so) — reuses
    `minimax_h3_load_qkv_device`/`minimax_h3_load_fc1_device` (generic over
    any tensor name, not block-specific) rather than re-deriving that
    transform a third time.

    `final_layer.*` is SYNTHETIC — shard 13 (where it lives) has not
    downloaded. That includes `final_layer.norm.weight`, NOT only the adaLN
    table team-lead flagged: `final_layer.video_out`/`audio_out` are in
    shard 13 too, confirmed against the real index.json."""
    var w = Dict[String, ArcPointer[Tensor]]()
    var direct = List[String]()
    direct.append("video_patch_proj.weight")
    direct.append("video_patch_proj.bias")
    direct.append("audio_patch_proj.weight")
    direct.append("audio_patch_proj.bias")
    direct.append("condition_proj.weight")
    direct.append("condition_proj.bias")
    direct.append("time_embedder.proj_in.weight")
    direct.append("time_embedder.proj_in.bias")
    direct.append("time_embedder.proj_out.weight")
    direct.append("time_embedder.proj_out.bias")
    direct.append("token_refiner.final_norm.weight")
    for i in range(len(direct)):
        var n = direct[i]
        w[n] = ArcPointer(Tensor.from_view(st.tensor_view(n), ctx))

    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var hidden = config.hidden_size
    var ffn = config.ffn_hidden_size
    for layer in range(2):
        var p = String("token_refiner.blocks.") + String(layer) + "."
        w[p + "norm1.weight"] = ArcPointer(Tensor.from_view(st.tensor_view(p + "norm1.weight"), ctx))
        w[p + "norm2.weight"] = ArcPointer(Tensor.from_view(st.tensor_view(p + "norm2.weight"), ctx))
        w[p + "attn.q_norm.weight"] = ArcPointer(Tensor.from_view(st.tensor_view(p + "attn.q_norm.weight"), ctx))
        w[p + "attn.k_norm.weight"] = ArcPointer(Tensor.from_view(st.tensor_view(p + "attn.k_norm.weight"), ctx))
        w[p + "attn.out_proj.weight"] = ArcPointer(Tensor.from_view(st.tensor_view(p + "attn.out_proj.weight"), ctx))
        w[p + "mlp.fc2.weight"] = ArcPointer(Tensor.from_view(st.tensor_view(p + "mlp.fc2.weight"), ctx))
        w[p + "attn.qkv_proj.weight"] = ArcPointer(
            minimax_h3_load_qkv_device(st, p + "attn.qkv_proj.weight", heads, head_dim, hidden, ctx)
        )
        w[p + "mlp.fc1.weight"] = ArcPointer(
            minimax_h3_load_fc1_device(st, p + "mlp.fc1.weight", ffn, hidden, ctx)
        )

    # SYNTHETIC final_layer (shard 13 not downloaded).
    var norm_shape: List[Int] = [hidden]
    w["final_layer.norm.weight"] = ArcPointer(randn(norm_shape^, 501, STDtype.BF16, ctx))
    var vo_w_shape: List[Int] = [config.video_patch_dim(), hidden]
    var vo_b_shape: List[Int] = [config.video_patch_dim()]
    w["final_layer.video_out.weight"] = ArcPointer(randn(vo_w_shape^, 502, STDtype.F32, ctx))
    w["final_layer.video_out.bias"] = ArcPointer(randn(vo_b_shape^, 503, STDtype.F32, ctx))
    var ao_w_shape: List[Int] = [config.audio_latents_dim, hidden]
    var ao_b_shape: List[Int] = [config.audio_latents_dim]
    w["final_layer.audio_out.weight"] = ArcPointer(randn(ao_w_shape^, 504, STDtype.F32, ctx))
    w["final_layer.audio_out.bias"] = ArcPointer(randn(ao_b_shape^, 505, STDtype.F32, ctx))
    return w^


def main() raises:
    var ctx = DeviceContext()
    var config = minimax_h3_released_config()
    config.validate()

    var st = ShardedSafeTensors.open(String(CHECKPOINT_FILE))
    print("checkpoint:", st.num_shards(), "shard(s),", st.num_tensors(), "tensors (shard 1 only)")
    var w = _build_frontend_weights(st, config, ctx)
    print("frontend weights built:", len(w), "tensors (real video/audio/condition/time_embedder/token_refiner + synthetic final_layer)")

    var hidden_size = config.hidden_size
    var n_fail = 0

    # ── shared rope table over S=8 ──
    var inv_freq = minimax_h3_rope_inv_freq(config.rope_inv_freq_len)
    var position_ids = List[Float64]()
    for r in range(S):
        position_ids.append(Float64(r))
        position_ids.append(Float64(0))
        position_ids.append(Float64(0))
    var rope = minimax_h3_rope_table(position_ids, S, inv_freq)
    var cos_shape: List[Int] = [S, rope.rotary_dim]
    var sin_shape: List[Int] = [S, rope.rotary_dim]
    var cos_t = Tensor.from_host(rope.cos, cos_shape^, STDtype.F32, ctx)
    var sin_t = Tensor.from_host(rope.sin, sin_shape^, STDtype.F32, ctx)

    # ── shared source content (same for every check below) ──
    var video_shape: List[Int] = [NV, config.video_patch_dim()]
    var video_rows = randn(video_shape^, 11, STDtype.F32, ctx)
    var audio_shape: List[Int] = [NA, config.audio_latents_dim]
    var audio_rows = randn(audio_shape^, 12, STDtype.F32, ctx)
    var text_shape: List[Int] = [NT, config.text_dim]
    var text_rows = randn(text_shape^, 13, STDtype.BF16, ctx)
    var ts_shape: List[Int] = [1]
    var timesteps = randn(ts_shape^, 14, STDtype.F32, ctx)  # num_timesteps=1

    # Assignment A: interleaved, not contiguous blocks.
    var video_idx_A = [0, 3, 6]
    var audio_idx_A = [1, 4]
    var text_idx_A = [2, 5, 7]
    var timestep_indices_all0 = List[Int]()
    for _ in range(S):
        timestep_indices_all0.append(0)

    print("")
    print("[frontend] running minimax_h3_frontend_embed (assignment A) ...")
    var frontend_A = minimax_h3_frontend_embed[TR_S, TR_H, TR_DH](
        video_rows, audio_rows, text_rows, timesteps,
        video_idx_A, audio_idx_A, text_idx_A, S, w, config, ctx,
    )
    var magH = _mag(frontend_A.hidden, ctx)
    var magT = _mag(frontend_A.temb, ctx)
    print("  hidden shape", frontend_A.hidden.shape()[0], frontend_A.hidden.shape()[1], " mean_abs", magH[0], " max_abs", magH[1], " nan", magH[2])
    print("  temb   shape", frontend_A.temb.shape()[0], frontend_A.temb.shape()[1], " mean_abs", magT[0], " max_abs", magT[1], " nan", magT[2])
    if len(frontend_A.hidden.shape()) != 2 or frontend_A.hidden.shape()[0] != S or frontend_A.hidden.shape()[1] != hidden_size:
        raise Error("E2E: frontend_A.hidden shape mismatch at the frontend->stack seam")
    if magH[2] != 0:
        print("  FAIL NaN in frontend hidden")
        n_fail += 1

    # ─────────────────────────────────────────────────────────────────────
    # [1] SCATTER BIT-EXACT ROUND TRIP
    # ─────────────────────────────────────────────────────────────────────
    print("")
    print("[1] scatter bit-exact round trip (independent recompute vs frontend_embed's internal scatter)")
    var my_video_f32 = minimax_h3_video_patch_embed(video_rows, w, ctx)
    var my_video = cast_tensor(my_video_f32, STDtype.BF16, ctx)
    var my_audio_f32 = minimax_h3_audio_patch_embed(audio_rows, w, ctx)
    var my_audio = cast_tensor(my_audio_f32, STDtype.BF16, ctx)
    var my_text0 = minimax_h3_condition_embed(text_rows, w, ctx)
    var my_text = minimax_h3_token_refiner[TR_S, TR_H, TR_DH](my_text0, w, config, ctx)

    var gathered_video = gather_rows(frontend_A.hidden, video_idx_A, ctx)
    var gathered_audio = gather_rows(frontend_A.hidden, audio_idx_A, ctx)
    var gathered_text = gather_rows(frontend_A.hidden, text_idx_A, ctx)
    var diff_video = _max_abs_diff(gathered_video, my_video, ctx)
    var diff_audio = _max_abs_diff(gathered_audio, my_audio, ctx)
    var diff_text = _max_abs_diff(gathered_text, my_text, ctx)
    print("  video max_abs_diff", diff_video, " audio max_abs_diff", diff_audio, " text max_abs_diff", diff_text)
    if diff_video != Float32(0.0) or diff_audio != Float32(0.0) or diff_text != Float32(0.0):
        print("  FAIL scatter did not place a modality's rows at its own index list")
        n_fail += 1
    else:
        print("  ok   every modality's packed rows exactly equal its own independently-recomputed embeds")

    # ─────────────────────────────────────────────────────────────────────
    # [2] PERMUTATION INVARIANCE (transposition/off-by-one detector)
    # ─────────────────────────────────────────────────────────────────────
    print("")
    print("[2] permutation invariance: same content, indices reversed within each modality, num_layers=0")
    var video_idx_B = [6, 3, 0]
    var audio_idx_B = [4, 1]
    var text_idx_B = [7, 5, 2]

    var block_mod_dummy = List[ArcPointer[Tensor]]()
    for i in range(2):
        var shp: List[Int] = [1, 6 * hidden_size]
        block_mod_dummy.append(ArcPointer(randn(shp^, UInt64(600 + i), STDtype.BF16, ctx)))
    var fmod_dummy_shape: List[Int] = [1, 2 * hidden_size]
    var final_mod_dummy = ArcPointer(randn(fmod_dummy_shape^, 700, STDtype.BF16, ctx))
    var modcache_dummy = MiniMaxH3ModCache(block_mod_dummy^, final_mod_dummy^, 1)
    var adaln_dummy = List[Int]()
    for _ in range(S):
        adaln_dummy.append(0)

    var final_mod_uniform_shape: List[Int] = [1, 2 * hidden_size]
    var final_mod_uniform = randn(final_mod_uniform_shape^, 800, STDtype.F32, ctx)

    var hiddenA0 = minimax_h3_run_stack[S](
        frontend_A.hidden.clone(ctx), st, modcache_dummy, adaln_dummy, cos_t, sin_t, rope.rotary_dim,
        config, ctx, 0,
    )
    var outA = minimax_h3_final_layer(
        hiddenA0, final_mod_uniform, timestep_indices_all0, video_idx_A, audio_idx_A, w, config, ctx,
    )

    var frontend_B = minimax_h3_frontend_embed[TR_S, TR_H, TR_DH](
        video_rows, audio_rows, text_rows, timesteps,
        video_idx_B, audio_idx_B, text_idx_B, S, w, config, ctx,
    )
    var hiddenB0 = minimax_h3_run_stack[S](
        frontend_B.hidden.clone(ctx), st, modcache_dummy, adaln_dummy, cos_t, sin_t, rope.rotary_dim,
        config, ctx, 0,
    )
    var outB = minimax_h3_final_layer(
        hiddenB0, final_mod_uniform, timestep_indices_all0, video_idx_B, audio_idx_B, w, config, ctx,
    )

    var diff_video_out = _max_abs_diff(outA.video_out, outB.video_out, ctx)
    var diff_audio_out = _max_abs_diff(outA.audio_out, outB.audio_out, ctx)
    print("  video_out max_abs_diff (A vs B)", diff_video_out, " audio_out max_abs_diff (A vs B)", diff_audio_out)
    if diff_video_out != Float32(0.0) or diff_audio_out != Float32(0.0):
        print("  FAIL output changed when the SAME logical rows were routed through DIFFERENT physical positions")
        n_fail += 1
    else:
        print("  ok   output is identical regardless of which physical packed position each logical row used")

    # ─────────────────────────────────────────────────────────────────────
    # [3]/[4] REAL 2-LAYER FULL PATH: adaLN differentiation + shapes/NaN/magnitude
    # ─────────────────────────────────────────────────────────────────────
    print("")
    print("[3]/[4] real 2-layer stack: adaLN differentiation through composition + shape/NaN/magnitude")
    # token_tags for assignment A's physical layout: pos0,3,6=video(0); 1,4=audio(2); 2,5,7=text(1)
    var token_tags = [0, 2, 1, 0, 2, 1, 0, 1]
    var adaln_indices_real = minimax_h3_adaln_rows(timestep_indices_all0, token_tags)

    var block_mod_real = List[ArcPointer[Tensor]]()
    for i in range(2):
        var shp2: List[Int] = [3, 6 * hidden_size]  # 1 timestep * 3 tags
        block_mod_real.append(ArcPointer(randn(shp2^, UInt64(900 + i), STDtype.BF16, ctx)))
    var fmod_real_shape: List[Int] = [1, 2 * hidden_size]
    var final_mod_real = ArcPointer(randn(fmod_real_shape^, 950, STDtype.BF16, ctx))
    var modcache_real = MiniMaxH3ModCache(block_mod_real^, final_mod_real^, 1)

    var stacked_real = minimax_h3_run_stack[S](
        frontend_A.hidden.clone(ctx), st, modcache_real, adaln_indices_real, cos_t, sin_t, rope.rotary_dim,
        config, ctx, 2,
    )
    if len(stacked_real.shape()) != 2 or stacked_real.shape()[0] != S or stacked_real.shape()[1] != hidden_size:
        raise Error("E2E: run_stack output shape mismatch at the stack->final_layer seam")
    var magStack = _mag(stacked_real, ctx)
    print("  stack(real adaLN) shape", stacked_real.shape()[0], stacked_real.shape()[1], " mean_abs", magStack[0], " max_abs", magStack[1], " nan", magStack[2])
    if magStack[2] != 0:
        print("  FAIL NaN in real 2-layer stack output")
        n_fail += 1

    var adaln_indices_flat = List[Int]()
    for _ in range(S):
        adaln_indices_flat.append(0)
    var stacked_flat = minimax_h3_run_stack[S](
        frontend_A.hidden.clone(ctx), st, modcache_real, adaln_indices_flat, cos_t, sin_t, rope.rotary_dim,
        config, ctx, 2,
    )
    var diff_stack = _max_abs_diff(stacked_real, stacked_flat, ctx)
    print("  real-adaLN vs flat-adaLN stack output max_abs_diff", diff_stack)
    if diff_stack == Float32(0.0):
        print("  FAIL adaLN differentiation had NO effect through the full frontend->2-block composition")
        n_fail += 1
    else:
        print("  ok   per-(timestep,tag) adaLN differentiation propagates through frontend + 2 real blocks")

    var final_out = minimax_h3_final_layer(
        stacked_real, final_mod_uniform, timestep_indices_all0, video_idx_A, audio_idx_A, w, config, ctx,
    )
    if len(final_out.video_out.shape()) != 2 or final_out.video_out.shape()[0] != NV or final_out.video_out.shape()[1] != config.video_patch_dim():
        raise Error("E2E: final_layer video_out shape mismatch")
    if len(final_out.audio_out.shape()) != 2 or final_out.audio_out.shape()[0] != NA or final_out.audio_out.shape()[1] != config.audio_latents_dim:
        raise Error("E2E: final_layer audio_out shape mismatch")
    var magVO = _mag(final_out.video_out, ctx)
    var magAO = _mag(final_out.audio_out, ctx)
    print("  video_out shape", final_out.video_out.shape()[0], final_out.video_out.shape()[1], " mean_abs", magVO[0], " max_abs", magVO[1], " nan", magVO[2])
    print("  audio_out shape", final_out.audio_out.shape()[0], final_out.audio_out.shape()[1], " mean_abs", magAO[0], " max_abs", magAO[1], " nan", magAO[2])
    if magVO[2] != 0 or magAO[2] != 0:
        print("  FAIL NaN in final_layer output")
        n_fail += 1

    print("")
    if n_fail != 0:
        raise Error(String("minimax_h3_forward_e2e_probe: ") + String(n_fail) + " check(s) FAILED")
    print("ALL CHECKS PASS — wiring only, NOT numeric parity")
