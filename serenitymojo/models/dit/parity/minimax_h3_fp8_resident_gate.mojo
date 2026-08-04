# serenitymojo/models/dit/parity/minimax_h3_fp8_resident_gate.mojo
#
# FP8-RESIDENT vs BF16-STREAMED parity gate for the MiniMax-H3 denoiser —
# the correctness gate for `models/dit/minimax_h3_fp8_resident.mojo` (the
# krea2-precedent fp8-resident base). REAL FL2VA weights, all 13 shards.
#
# SCHEME NOTE (2026-08-04): the store's default scheme is now INT8-WEIGHT-
# ONLY per-row (see its header — E4M3 measured 0.998347/0.997255/0.988741
# here vs the 0.999 bar and was swapped on Alex's decision). This gate's
# `fp8` MODE NAME is historical: it runs the store's DEFAULT scheme, and the
# store prints which at build. The `ref` arm and its saved references are
# scheme-independent (bf16-streamed) and remain valid across scheme swaps.
# The `sens` mode is still the E4M3 round-trip diagnostic (kept: it is the
# record of WHY the swap happened).
#
# TWO CHECKS, both A/B of the SAME code path with only the weight SOURCE
# changed (`minimax_h3_run_stack`'s `resident` arg — everything else,
# including `minimax_h3_block_forward`, is shared bit-for-bit):
#
#   [1] ONE BLOCK (layer 0): identical random [S, hidden] bf16 input through
#       `minimax_h3_run_stack[S](..., num_layers=1)` twice — bf16-streamed vs
#       fp8-resident-dequant. cos + max_abs reported. BAR: cos >= 0.999
#       (E4M3 per-row quantization noise; krea2 precedent ~0.99 documented in
#       krea2_dit.mojo's _blk_w8).
#   [2] FULL 50-BLOCK E2E, 2 REAL EULER STEPS: frontend_embed -> 50-block
#       stack -> final_layer -> MiniMaxH3DualSchedule step, twice per arm,
#       compounding any quantization drift through the ENTIRE composition
#       (states re-embedded each step, exactly the t2va denoise-loop shape at
#       a small S). Compares the resulting video/audio LATENT STATES.
#       BAR: cos >= 0.999 on both.
#
# PLUS a `sens` mode (no bar, diagnostic): per-GEMM-class sensitivity on
# block 0 — round-trips ONE of {qkv, out_proj, fc1, fc2} through
# encode->dequant at a time (others stay original bf16), reruns the block
# forward, and reports each variant's cos vs the all-bf16 forward. This is
# the data for the "keep the most sensitive class bf16" escalation rung: it
# says WHERE the quantization error actually comes from, measured, before
# anyone spends residency bytes on a guess.
#
# WHAT IS SYNTHETIC AND WHY THAT IS SOUND: the per-block AdaLN mod tables are
# random (deterministic per (step, layer) seed, IDENTICAL across arms) — same
# choice as minimax_h3_forward_e2e_probe.mojo, because modcache is an INPUT to
# the stack, is byte-identical between the two arms, and is not part of the
# changed component (the resident store never touches adaln_proj — fp8_policy
# evicts it). Frontend/final-layer weights are REAL and shared by both arms.
#
# TWO PROCESSES, NOT ONE — MEASURED OOM, NOT A STYLE CHOICE (2026-08-03,
# first in-process run): streamed arms + 50-block quantize + resident arms in
# ONE process reached ~23 GiB before the resident e2e arm began (the device
# pool retains the streamed arms' 100 x ~0.77 GiB churn on top of the
# 17.96 GiB store) and layer 0's dequant OOM'd — the error surfaced as
# `minimax_h3_run_stack: layer 0 failed to load` because the stack's
# try/except relabels ANY weight-source failure with the layer number; the
# resident path itself was fine (check [1]'s dequant arm had already passed
# in the same process). The fix is a process boundary: `ref` mode runs the
# two bf16-streamed arms and SAVES the three reference tensors to a
# safetensors; `fp8` mode starts a FRESH process (fresh pool), loads the
# references, builds the store, runs the two resident arms, compares. Every
# generated input (randn seeds, rope, mod tables) is deterministic, so the
# two processes see byte-identical inputs. In `fp8` mode the e2e arm runs
# FIRST (it is the memory-heavy one — fail fast), then the block arm.
#
# Run (binary build — mojo run/JIT cannot resolve flame_cudnn_sdpa_bf16; same
# recipe as minimax_h3_stack_real_weight_gate.mojo):
#   cd /home/alex/mojodiffusion
#   pixi run scripts/mem_safe.sh mojo build --optimization-level 2 -j 1 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/dit/parity/minimax_h3_fp8_resident_gate.mojo \
#     -o output/checks/minimax_h3_fp8_resident_gate \
#   && output/checks/minimax_h3_fp8_resident_gate sens \
#   && output/checks/minimax_h3_fp8_resident_gate ref \
#   && output/checks/minimax_h3_fp8_resident_gate fp8

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.fp8_quant import fp8_e4m3_encode_perrow, fp8_e4m3_rowscale
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import full_device, reshape

from serenitymojo.models.minimax_h3.dit_frontend import (
    minimax_h3_rope_inv_freq,
    minimax_h3_rope_table,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    MiniMaxH3DiTConfig,
    minimax_h3_adaln_rows,
    minimax_h3_block_forward,
    minimax_h3_block_prefix,
    minimax_h3_block_tensor_names,
    minimax_h3_released_config,
)
from serenitymojo.models.dit.minimax_h3_fp8_resident import (
    MiniMaxH3ResidentFp8,
    minimax_h3_build_resident_fp8,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    _minimax_h3_marker_tensor,
    minimax_h3_load_block_device,
    minimax_h3_load_fc1_device,
    minimax_h3_load_qkv_device,
)
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache
from serenitymojo.models.dit.minimax_h3_sampling import MiniMaxH3DualSchedule
from serenitymojo.models.dit.minimax_h3_stack import minimax_h3_run_stack
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_final_layer,
    minimax_h3_frontend_embed,
)

comptime CHECKPOINT_DIR = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
)
comptime REF_PATH = (
    "/home/alex/mojodiffusion/output/minimax_h3_block/fp8_resident_ref.safetensors"
)

comptime S = 8            # packed sequence, same tiny geometry as the e2e probe
comptime NV = 3           # video rows
comptime NA = 2           # audio rows
comptime NT = 3           # text tokens (comptime: token_refiner's S param)
comptime TR_H = 56
comptime TR_DH = 128
comptime NUM_LAYERS = 50  # the full released stack
comptime NUM_STEPS = 2    # e2e denoise steps per arm
comptime SCHED_STEPS = 4  # schedule length (>= NUM_STEPS; set_timesteps needs >=2)
comptime COS_BAR = 0.999


# ─────────────────────────────────────────────────────────────────────────────
# Real frontend/final-layer weights — same builder as
# minimax_h3_forward_e2e_probe.mojo::_build_frontend_weights (kept standalone
# per parity-file convention; that probe's copy points at a scoped 11-shard
# scratch index, this one at the now-complete real directory).
# ─────────────────────────────────────────────────────────────────────────────
def _build_frontend_weights(
    st: ShardedSafeTensors, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
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
    direct.append("final_layer.norm.weight")
    direct.append("final_layer.video_out.weight")
    direct.append("final_layer.video_out.bias")
    direct.append("final_layer.audio_out.weight")
    direct.append("final_layer.audio_out.bias")
    direct.append("final_layer.adaln_proj.linear.weight")
    direct.append("final_layer.adaln_proj.linear.bias")
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
    return w^


def _real_final_modulation(
    temb: Tensor, w: Dict[String, ArcPointer[Tensor]], ctx: DeviceContext
) raises -> Tensor:
    """Same math (and F32-cast requirement) as the e2e probe's helper of the
    same name — see that file for the measured dtype finding."""
    var activated_f32 = silu(temb, ctx)
    var activated_bf16 = cast_tensor(activated_f32, STDtype.BF16, ctx)
    var mod_bf16 = linear_bias(
        activated_bf16,
        w["final_layer.adaln_proj.linear.weight"][],
        w["final_layer.adaln_proj.linear.bias"][],
        ctx,
    )
    return cast_tensor(mod_bf16, STDtype.F32, ctx)


def _step_modcache(
    config: MiniMaxH3DiTConfig, step: Int, ctx: DeviceContext
) raises -> MiniMaxH3ModCache:
    """Synthetic per-step AdaLN tables (1 timestep x 3 tags per layer),
    DETERMINISTIC in (step, layer) — rebuilt identically for both arms (and
    both PROCESSES), so the mod input is byte-identical and only the weight
    source differs."""
    var hidden = config.hidden_size
    var block_mod = List[ArcPointer[Tensor]]()
    for li in range(NUM_LAYERS):
        var shp: List[Int] = [3, 6 * hidden]
        block_mod.append(
            ArcPointer(randn(shp^, UInt64(5000 + step * 1000 + li), STDtype.BF16, ctx))
        )
    # final_mod placeholder — never read by minimax_h3_run_stack; the REAL
    # final modulation is computed per step from real adaln_proj bytes.
    var fshape: List[Int] = [1, 2 * hidden]
    var final_mod = ArcPointer(full_device(fshape^, Float32(0.0), STDtype.F32, ctx))
    return MiniMaxH3ModCache(block_mod^, final_mod^, 1)


struct _ArmLatents(Movable):
    var video: Tensor
    var audio: Tensor

    def __init__(out self, var video: Tensor, var audio: Tensor):
        self.video = video^
        self.audio = audio^


def _run_denoise_arm(
    resident: Optional[MiniMaxH3ResidentFp8],
    st: ShardedSafeTensors,
    w: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    video_init: Tensor,
    audio_init: Tensor,
    text_rows: Tensor,
    cos_t: Tensor,
    sin_t: Tensor,
    rotary_dim: Int,
    arm_name: String,
    ctx: DeviceContext,
) raises -> _ArmLatents:
    """NUM_STEPS real Euler steps of the t2va denoise-loop shape at S=8:
    frontend_embed -> 50-block stack (weight source = `resident` when present,
    else disk stream) -> final_layer -> dual-schedule step. Everything except
    the weight source is identical between arms by construction."""
    var vstate = video_init.clone(ctx)
    var astate = audio_init.clone(ctx)
    var sched = MiniMaxH3DualSchedule()
    sched.set_timesteps(SCHED_STEPS)

    # Assignment mirrors the e2e probe's interleaved layout A.
    var video_idx = [0, 3, 6]
    var audio_idx = [1, 4]
    var text_idx = [2, 5, 7]
    var token_tags = [0, 2, 1, 0, 2, 1, 0, 1]
    var ts_idx_all0 = List[Int]()
    for _ in range(S):
        ts_idx_all0.append(0)
    var adaln_idx = minimax_h3_adaln_rows(ts_idx_all0, token_tags)

    for i in range(NUM_STEPS):
        var video_ts = sched.video_timestep(i)
        var audio_ts = sched.audio_timestep(i)
        var ts_shape: List[Int] = [1]
        var ts_t = Tensor.from_host([video_ts], ts_shape^, STDtype.F32, ctx)
        var embed = minimax_h3_frontend_embed[NT, TR_H, TR_DH](
            vstate, astate, text_rows, ts_t,
            video_idx, audio_idx, text_idx, S, w, config, ctx,
        )
        var mc = _step_modcache(config, i, ctx)
        var hidden_out = minimax_h3_run_stack[S](
            embed.hidden.clone(ctx), st, mc, adaln_idx, cos_t, sin_t,
            rotary_dim, config, ctx, NUM_LAYERS, 0, resident,
        )
        var final_mod = _real_final_modulation(embed.temb, w, ctx)
        var out = minimax_h3_final_layer(
            hidden_out, final_mod, ts_idx_all0, video_idx, audio_idx,
            w, config, ctx,
        )
        vstate = sched.step_video_device(out.video_out, video_ts, vstate, ctx)
        astate = sched.step_audio_device(out.audio_out, audio_ts, astate, ctx)
        ctx.synchronize()
        print(
            "  [", arm_name, "] step", i + 1, "/", NUM_STEPS,
            " video_t=", video_ts, " audio_t=", audio_ts, " done",
        )
    return _ArmLatents(vstate^, astate^)


def _norm(host: List[Float32]) -> Float64:
    var n2 = Float64(0.0)
    for i in range(len(host)):
        n2 += Float64(host[i]) * Float64(host[i])
    return sqrt(n2)


def _mag_stats(host: List[Float32]) -> Tuple[Float32, Float32]:
    """(mean_abs, max_abs) — the reference arm's own magnitudes, printed so a
    reported max_abs DIFF has a denominator (escalation rung (a))."""
    var sum_abs = Float32(0.0)
    var max_abs = Float32(0.0)
    for i in range(len(host)):
        var v = host[i]
        var a = v if v >= Float32(0.0) else -v
        sum_abs += a
        if a > max_abs:
            max_abs = a
    return (sum_abs / Float32(max(len(host), 1)), max_abs)


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


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic shared inputs — regenerated identically by BOTH processes.
# ─────────────────────────────────────────────────────────────────────────────
def _block_input(hidden: Int, ctx: DeviceContext) raises -> Tensor:
    return randn([S, hidden], 21, STDtype.BF16, ctx)


def _shared_adaln_rows() raises -> List[Int]:
    var token_tags = [0, 2, 1, 0, 2, 1, 0, 1]
    var ts_idx_all0 = List[Int]()
    for _ in range(S):
        ts_idx_all0.append(0)
    return minimax_h3_adaln_rows(ts_idx_all0, token_tags)


# ─────────────────────────────────────────────────────────────────────────────
# `sens` mode: per-GEMM-class sensitivity on block 0 (diagnostic, no bar).
# ─────────────────────────────────────────────────────────────────────────────
def _roundtrip_fp8(w: Tensor, ctx: DeviceContext) raises -> Tensor:
    """encode -> dequant through the SAME two gated kernels the store uses:
    the exact bf16 weight the resident path hands the block forward."""
    var s = fp8_e4m3_rowscale(w, ctx)
    var q = fp8_e4m3_encode_perrow(w, s, ctx)
    return fp8_e4m3_dequant_perrow_to_bf16(q, s, ctx)


def _sens_variant_weights(
    base: Dict[String, ArcPointer[Tensor]],
    quantize: List[String],
    ctx: DeviceContext,
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Copy of block 0's weight Dict with ONLY the tensors named in
    `quantize` round-tripped through fp8; everything else (including the
    transform markers, legitimately — `base` came from
    minimax_h3_load_block_device) carried over by ArcPointer."""
    var out = Dict[String, ArcPointer[Tensor]]()
    var names = minimax_h3_block_tensor_names(0)
    for i in range(len(names)):
        ref name = names[i]
        var hit = False
        for j in range(len(quantize)):
            if quantize[j] == name:
                hit = True
        if hit:
            out[name] = ArcPointer(_roundtrip_fp8(base[name][], ctx))
        else:
            out[name] = base[name].copy()
    out[MINIMAX_H3_QKV_DEINTERLEAVED_MARKER] = ArcPointer(_minimax_h3_marker_tensor(ctx))
    out[MINIMAX_H3_FC1_SWAPPED_MARKER] = ArcPointer(_minimax_h3_marker_tensor(ctx))
    return out^


def _run_sens(
    st: ShardedSafeTensors, config: MiniMaxH3DiTConfig, ctx: DeviceContext,
    cos_t: Tensor, sin_t: Tensor, rotary_dim: Int,
) raises:
    var hidden = config.hidden_size
    var p = minimax_h3_block_prefix(0)
    var base = minimax_h3_load_block_device(st, 0, config, ctx)
    var adaln_idx = _shared_adaln_rows()
    # Same mod values as _step_modcache(step=0) layer 0 (seed 5000).
    var mod = randn([3, 6 * hidden], UInt64(5000), STDtype.BF16, ctx)
    var block_in = _block_input(hidden, ctx)
    var x3 = reshape(block_in, [1, S, hidden], ctx)

    var ref_out = minimax_h3_block_forward[S](
        x3, base, 0, config, mod, adaln_idx, cos_t, sin_t, rotary_dim, ctx,
    )
    var ref_host = cast_tensor(ref_out, STDtype.F32, ctx).to_host(ctx)
    var ref_mag = _mag_stats(ref_host)
    print("  bf16 reference block output: mean_abs=", ref_mag[0],
          " max_abs=", ref_mag[1], " n=", len(ref_host))

    var kinds = List[String]()
    kinds.append(p + "attn.qkv_proj.weight")
    kinds.append(p + "attn.out_proj.weight")
    kinds.append(p + "mlp.fc1.weight")
    kinds.append(p + "mlp.fc2.weight")

    var harness = ParityHarness(COS_BAR)
    for k in range(len(kinds)):
        var one = List[String]()
        one.append(kinds[k].copy())
        var wv = _sens_variant_weights(base, one, ctx)
        var out_k = minimax_h3_block_forward[S](
            x3, wv, 0, config, mod, adaln_idx, cos_t, sin_t, rotary_dim, ctx,
        )
        var r = harness.compare(cast_tensor(out_k, STDtype.F32, ctx), ref_host, ctx)
        print("  sens ONLY", kinds[k], " fp8: cos=", r.cos, " max_abs=", r.max_abs)

    var all4 = List[String]()
    for k in range(len(kinds)):
        all4.append(kinds[k].copy())
    var wall = _sens_variant_weights(base, all4, ctx)
    var out_all = minimax_h3_block_forward[S](
        x3, wall, 0, config, mod, adaln_idx, cos_t, sin_t, rotary_dim, ctx,
    )
    var r_all = harness.compare(cast_tensor(out_all, STDtype.F32, ctx), ref_host, ctx)
    print("  sens ALL FOUR fp8: cos=", r_all.cos, " max_abs=", r_all.max_abs,
          " (must reproduce the store-path blockCos — internal consistency)")


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: minimax_h3_fp8_resident_gate <sens|ref|fp8>")
    var mode = String(args[1])
    if mode != "sens" and mode != "ref" and mode != "fp8":
        raise Error("usage: minimax_h3_fp8_resident_gate <sens|ref|fp8>")

    print("MiniMax-H3 FP8-RESIDENT vs BF16-STREAMED parity gate — mode:", mode)
    print("  checkpoint:", String(CHECKPOINT_DIR))

    var ctx = DeviceContext()
    var config = minimax_h3_released_config()
    config.validate()
    var hidden = config.hidden_size

    var st = ShardedSafeTensors.open(String(CHECKPOINT_DIR))
    print("  shards:", st.num_shards(), " tensors:", st.num_tensors())

    # ── shared rope over S (same recipe as the e2e probe) ──
    var inv_freq = minimax_h3_rope_inv_freq(config.rope_inv_freq_len)
    var position_ids = List[Float64]()
    for r in range(S):
        position_ids.append(Float64(r))
        position_ids.append(Float64(0))
        position_ids.append(Float64(0))
    var rope = minimax_h3_rope_table(position_ids, S, inv_freq)
    var cos_t = Tensor.from_host(rope.cos, [S, rope.rotary_dim], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(rope.sin, [S, rope.rotary_dim], STDtype.F32, ctx)

    if mode == "sens":
        print("")
        print("[sens] per-GEMM-class fp8 sensitivity, block 0, real weights")
        _run_sens(st, config, ctx, cos_t, sin_t, rope.rotary_dim)
        print("")
        print("SENS DONE")
        return

    # Deterministic shared inputs (identical in both processes).
    var adaln_idx = _shared_adaln_rows()
    var mc0 = _step_modcache(config, 0, ctx)
    var block_in = _block_input(hidden, ctx)
    var video_init = randn([NV, config.video_patch_dim()], 11, STDtype.F32, ctx)
    var audio_init = randn([NA, config.audio_latents_dim], 12, STDtype.F32, ctx)
    var text_rows = randn([NT, config.text_dim], 13, STDtype.BF16, ctx)

    if mode == "ref":
        # ── bf16-STREAMED reference arms; save the three reference tensors ──
        var w = _build_frontend_weights(st, config, ctx)
        print("  frontend/final-layer weights built:", len(w), "tensors (real)")
        print("")
        print("[ref-1] block 0, bf16-STREAMED")
        var no_resident = Optional[MiniMaxH3ResidentFp8](None)
        var block_ref = minimax_h3_run_stack[S](
            block_in.clone(ctx), st, mc0, adaln_idx, cos_t, sin_t,
            rope.rotary_dim, config, ctx, 1, 0, no_resident,
        )
        var block_ref_f32 = cast_tensor(block_ref, STDtype.F32, ctx)
        var block_mag = _mag_stats(block_ref_f32.to_host(ctx))
        print("  block ref magnitudes: mean_abs=", block_mag[0], " max_abs=", block_mag[1])

        print("")
        print("[ref-2] e2e", NUM_STEPS, "steps x", NUM_LAYERS, "blocks, bf16-STREAMED")
        var ref_arm = _run_denoise_arm(
            no_resident, st, w, config, video_init, audio_init, text_rows,
            cos_t, sin_t, rope.rotary_dim, String("streamed"), ctx,
        )
        var vid_mag = _mag_stats(ref_arm.video.to_host(ctx))
        var aud_mag = _mag_stats(ref_arm.audio.to_host(ctx))
        print("  latent ref magnitudes: video mean_abs=", vid_mag[0], " max_abs=", vid_mag[1],
              " | audio mean_abs=", aud_mag[0], " max_abs=", aud_mag[1])

        var names = List[String]()
        names.append(String("block_ref"))
        names.append(String("video_ref"))
        names.append(String("audio_ref"))
        var tensors = List[ArcPointer[Tensor]]()
        tensors.append(ArcPointer(block_ref_f32^))
        tensors.append(ArcPointer(ref_arm.video.clone(ctx)))
        tensors.append(ArcPointer(ref_arm.audio.clone(ctx)))
        save_safetensors(names, tensors, String(REF_PATH), ctx)
        print("")
        print("REF DONE — saved ->", String(REF_PATH))
        return

    # ── mode == "fp8": fresh process — load refs, build store, run, gate ──
    var st_ref = SafeTensors.open(String(REF_PATH))
    var block_ref_host = _load_f32(st_ref, String("block_ref"))
    var ref_video_host = _load_f32(st_ref, String("video_ref"))
    var ref_audio_host = _load_f32(st_ref, String("audio_ref"))
    print("  references loaded from", String(REF_PATH))

    # Frontend weights BEFORE the store — run 3 measured why this order is
    # mandatory: the store build leaves the pool's reserved memory well above
    # the store's own bytes (size-bucketed retention of the build
    # transients), so any LARGE allocation after the build OOMs. After the
    # store exists, nothing bigger than small activations may allocate.
    var w = _build_frontend_weights(st, config, ctx)
    ctx.synchronize()
    print("  frontend/final-layer weights built:", len(w), "tensors (real)")

    print("")
    print("[store] quantizing all", NUM_LAYERS, "blocks to 1-byte resident",
          "(store default scheme; printed below) ...")
    var store = minimax_h3_build_resident_fp8(st, config, ctx, NUM_LAYERS)
    print(
        "  resident:",
        Float64(store.resident_bytes()) / (1024.0 * 1024.0 * 1024.0),
        "GiB on device (1B/param + per-row F32 scales + bf16 norms + scratch)",
    )
    var resident = Optional[MiniMaxH3ResidentFp8](store^)
    var harness = ParityHarness(COS_BAR)
    var n_fail = 0

    # E2E arm FIRST — the memory-heavy check fails fast if the fit is wrong.
    print("")
    print("[2b] e2e", NUM_STEPS, "steps x", NUM_LAYERS, "blocks, FP8-RESIDENT arm")
    var fp8_arm = _run_denoise_arm(
        resident, st, w, config, video_init, audio_init, text_rows,
        cos_t, sin_t, rope.rotary_dim, String("fp8-resident"), ctx,
    )
    var r_video = harness.compare(fp8_arm.video, ref_video_host, ctx)
    var r_audio = harness.compare(fp8_arm.audio, ref_audio_host, ctx)
    var video_ratio = Float64(-1.0)
    var ref_video_norm = _norm(ref_video_host)
    if ref_video_norm != 0.0:
        video_ratio = _norm(fp8_arm.video.to_host(ctx)) / ref_video_norm
    var audio_ratio = Float64(-1.0)
    var ref_audio_norm = _norm(ref_audio_host)
    if ref_audio_norm != 0.0:
        audio_ratio = _norm(fp8_arm.audio.to_host(ctx)) / ref_audio_norm
    var vref_mag = _mag_stats(ref_video_host)
    var aref_mag = _mag_stats(ref_audio_host)
    print("  e2eVideoCos=", r_video.cos, " e2eVideoMaxAbs=", r_video.max_abs,
          " (ref max_abs=", vref_mag[1], ") |fp8|/|bf16|=", video_ratio, " n=", r_video.n)
    print("  e2eAudioCos=", r_audio.cos, " e2eAudioMaxAbs=", r_audio.max_abs,
          " (ref max_abs=", aref_mag[1], ") |fp8|/|bf16|=", audio_ratio, " n=", r_audio.n)
    if r_video.passed and r_audio.passed:
        print("  ok   [2] e2e latents fp8-resident cos >=", COS_BAR, "(video AND audio)")
    else:
        print("  FAIL [2] e2e latents fp8-resident cos <", COS_BAR)
        n_fail += 1

    print("")
    print("[1b] block 0, FP8-RESIDENT dequant arm")
    var block_fp8 = minimax_h3_run_stack[S](
        block_in.clone(ctx), st, mc0, adaln_idx, cos_t, sin_t,
        rope.rotary_dim, config, ctx, 1, 0, resident,
    )
    var block_fp8_f32 = cast_tensor(block_fp8, STDtype.F32, ctx)
    var r_block = harness.compare(block_fp8_f32, block_ref_host, ctx)
    var block_ratio = Float64(-1.0)
    var block_ref_norm = _norm(block_ref_host)
    if block_ref_norm != 0.0:
        block_ratio = _norm(block_fp8_f32.to_host(ctx)) / block_ref_norm
    var bref_mag = _mag_stats(block_ref_host)
    print("  blockCos=", r_block.cos, " blockMaxAbs=", r_block.max_abs,
          " (ref max_abs=", bref_mag[1], ") |fp8|/|bf16|=", block_ratio, " n=", r_block.n)
    if r_block.passed:
        print("  ok   [1] single-block fp8-resident cos >=", COS_BAR)
    else:
        print("  FAIL [1] single-block fp8-resident cos <", COS_BAR)
        n_fail += 1

    print("")
    if n_fail == 0:
        print("GATE PASS blockCos=", r_block.cos, " e2eVideoCos=", r_video.cos,
              " e2eAudioCos=", r_audio.cos)
    else:
        print("GATE FAIL (", n_fail, "check(s) below the", COS_BAR, "bar — numbers above )")
        raise Error("MiniMax-H3 fp8-resident parity gate failed")
