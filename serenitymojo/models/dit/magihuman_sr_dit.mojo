# models/dit/magihuman_sr_dit.mojo — daVinci-MagiHuman SR (super-res) DiT,
# pure Mojo + MAX. Inference-only, GPU-only.
#
# Reference (read LINE BY LINE, READ-ONLY):
#   /home/alex/EriDiffusion/inference-flame/src/models/magihuman_sr_dit.rs
#   /home/alex/EriDiffusion/inference-flame/src/sampling/magihuman_unipc.rs
#   Weights: /home/alex/.serenity/models/dits/magi_human_sr1080_bf16.safetensors
#            (BF16, 563 tensors, 40 layers, 30.6 GB)
#
# ── SR-vs-base delta (the ONLY architectural difference) ─────────────────────
#   SAME 40-layer stack, SAME mm_layers=[0..3,36..39] / gelu7=[0..3], SAME
#   hidden=5120, head_dim=128, GQA x5, SwiGLU7/GELU7, attn gating, adapter,
#   Fourier RoPE, final video/audio heads, SAME norm shapes, SAME MLP shapes.
#   DIFFERENT weight LAYOUT only:
#     * base SHARED layer: ONE fused `attention.linear_qkv.weight` [7208,5120]
#       split into q[5120]/k[1024]/v[1024]/g[40].
#     * SR SHARED layer: FOUR separate weights
#         linear_q [5120,5120], linear_k [1024,5120],
#         linear_v [1024,5120], linear_g [40,5120], + linear_proj [5120,5120].
#     * base MM layer: ONE fused per-modality `linear_qkv` [7208*3,5120].
#     * SR MM layer: per-modality split — linear_{q,k,v,g}_{video,audio,text}
#       (3 modality variants of each of the 4), + linear_proj_{v,a,t}, and
#       per-modality mlp.{up_gate_proj,down_proj}_{video,audio,text}.
#
#   Q/K/V/G via 4 separate matmuls of `x @ W_i.T` is NUMERICALLY IDENTICAL to a
#   single matmul against the row-concatenated weight [q;k;v;g] (each output
#   column depends only on its own weight row). So the faithful, zero-new-math
#   port is: at LOAD, concat the split weights along out-dim 0 into the fused
#   `linear_qkv.weight` the base block library already consumes, then call the
#   EXISTING base block forwards UNCHANGED. Same for per-modality MM (concat the
#   3 modality variants the same way the base mm path slices them by group).
#
# REUSE (100% of math): magihuman_dit.{magihuman_shared_block_forward,
#   magihuman_mm_block_forward, magihuman_stack_forward, magihuman_final_heads,
#   magihuman_adapter_embed, magihuman_rope_from_coords, MagiHumanConfig}.
#   This file only adds the SR weight-layout ADAPTER (concat split -> fused).
#
# ── UniPC (magihuman_unipc.rs) ───────────────────────────────────────────────
#   The SR sampler is the distill `step_ddim` path (cfg_number=1), NOT the full
#   Cosmos UniPC predictor/corrector. Schedule:
#     sigma_max=(N-1)/N, sigma_min=0; sigmas=linspace(sigma_max,sigma_min,n+1)[:-1]
#     sigmas = shift*s/(1+(shift-1)*s)   (Wan flow shift, shift=5.0 default)
#     timesteps = int(sigma*N); sigmas.append(0).
#   The sigma-schedule builder already exists verbatim in sampling/unipc.mojo
#   (`build_unipc_sigma_schedule`). step_ddim itself is added below.
#     curr_t=sigmas[idx]; prev_t=sigmas[idx+1]
#     cur_clean = curr_state - curr_t * velocity
#     prev_state = prev_t * noise + (1-prev_t) * cur_clean
#
# DTYPE: bf16 weights+input, F32 accumulate. Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add_scalar, concat as _ta_concat, mul_scalar, add, sub
from serenitymojo.models.dit.magihuman_dit import (
    MagiHumanConfig,
    magihuman_shared_block_forward,
    magihuman_mm_block_forward,
)


# ── Concat split Q/K/V/G weights (row-major, out-dim 0) into fused linear_qkv ──
# Each W_i is [out_i, in]; result is [sum(out_i), in] = [7208, 5120] (shared) or
# the per-modality stack. Numerically identical to 4 separate matmuls.
def _fuse_qkvg(
    wq: Tensor, wk: Tensor, wv: Tensor, wg: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var qk = _ta_concat(0, ctx, wq, wk)
    var qkv = _ta_concat(0, ctx, qk, wv)
    return _ta_concat(0, ctx, qkv, wg)


# ── SR SHARED block forward (layers 4..35) ────────────────────────────────────
# Reads the SR split-linear weight dict (keys WITHOUT block prefix: linear_q/k/v/
# g/proj, mlp.up_gate_proj/down_proj, *.p1 norm gains) and dispatches to the base
# shared block by presenting a fused `attention.linear_qkv.weight`. The split
# weights are fused ONCE at load (see sr_fuse_shared_weights); the dict handed in
# here already carries `attention.linear_qkv.weight`.
def magihuman_sr_shared_block_forward[L: Int, H: Int, Hkv: Int, DH: Int](
    x_seq: Tensor,
    cos_e: Tensor,
    sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: MagiHumanConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    return magihuman_shared_block_forward[L, H, Hkv, DH](
        x_seq, cos_e, sin_e, w, cfg, ctx
    )


# ── SR MM block forward (layers 0..3 GELU7, 36..39 SwiGLU7) ───────────────────
# The base mm path slices the fused per-modality weight [out*3, in] into 3 chunks
# by group. We pre-build that fused per-modality layout at load by concatenating
# (video, audio, text) variants of each split linear into the modality-stacked
# order base expects, with q/k/v/g already fused per modality.
def magihuman_sr_mm_block_forward[L: Int, H: Int, Hkv: Int, DH: Int](
    x_seq: Tensor,
    cos_e: Tensor,
    sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: MagiHumanConfig,
    gs: List[Int],
    use_swiglu7: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    return magihuman_mm_block_forward[L, H, Hkv, DH](
        x_seq, cos_e, sin_e, w, cfg, gs, use_swiglu7, ctx
    )


# ── Load-time weight adapter: SR split -> base fused (SHARED layer) ────────────
# Given a raw SR weight dict for one shared layer (BF16 tensors, keys WITHOUT
# block prefix), build the base-compatible dict: fuse q/k/v/g -> linear_qkv,
# pass linear_proj/mlp.* through, and pre-add 1 to the norm gains (.p1).
def sr_fuse_shared_weights(
    raw: Dict[String, ArcPointer[Tensor]], ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var out = Dict[String, ArcPointer[Tensor]]()
    var fused = _fuse_qkvg(
        raw["attention.linear_q.weight"][],
        raw["attention.linear_k.weight"][],
        raw["attention.linear_v.weight"][],
        raw["attention.linear_g.weight"][],
        ctx,
    )
    out["attention.linear_qkv.weight"] = ArcPointer(fused^)
    out["attention.linear_proj.weight"] = ArcPointer(
        _passthrough(raw["attention.linear_proj.weight"][], ctx)
    )
    out["mlp.up_gate_proj.weight"] = ArcPointer(
        _passthrough(raw["mlp.up_gate_proj.weight"][], ctx)
    )
    out["mlp.down_proj.weight"] = ArcPointer(
        _passthrough(raw["mlp.down_proj.weight"][], ctx)
    )
    var norms = [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight", "mlp.pre_norm.weight",
    ]
    for sfx in norms:
        var s = String(sfx)
        out[s + ".p1"] = ArcPointer(add_scalar(raw[s][], 1.0, ctx))
    return out^


# ── Load-time weight adapter: SR per-modality split -> base fused (MM layer) ───
# Each linear has 3 modality variants (_video/_audio/_text), each itself a
# q/k/v/g split (for the qkv group). Base mm slices a [out*3,in] tensor into 3
# group chunks, so per modality we must present [out, in] = fused(q,k,v,g) and
# stack the 3 modalities along out-dim 0 in (video, audio, text) order.
def sr_fuse_mm_weights(
    raw: Dict[String, ArcPointer[Tensor]], ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var out = Dict[String, ArcPointer[Tensor]]()

    # qkv: per modality fuse(q,k,v,g) -> [7208,in]; stack v,a,t -> [7208*3,in].
    var qkv_v = _fuse_qkvg(
        raw["attention.linear_q_video.weight"][],
        raw["attention.linear_k_video.weight"][],
        raw["attention.linear_v_video.weight"][],
        raw["attention.linear_g_video.weight"][], ctx,
    )
    var qkv_a = _fuse_qkvg(
        raw["attention.linear_q_audio.weight"][],
        raw["attention.linear_k_audio.weight"][],
        raw["attention.linear_v_audio.weight"][],
        raw["attention.linear_g_audio.weight"][], ctx,
    )
    var qkv_t = _fuse_qkvg(
        raw["attention.linear_q_text.weight"][],
        raw["attention.linear_k_text.weight"][],
        raw["attention.linear_v_text.weight"][],
        raw["attention.linear_g_text.weight"][], ctx,
    )
    out["attention.linear_qkv.weight"] = ArcPointer(
        _stack3(qkv_v^, qkv_a^, qkv_t^, ctx)
    )

    out["attention.linear_proj.weight"] = ArcPointer(_stack3(
        _passthrough(raw["attention.linear_proj_video.weight"][], ctx),
        _passthrough(raw["attention.linear_proj_audio.weight"][], ctx),
        _passthrough(raw["attention.linear_proj_text.weight"][], ctx), ctx,
    ))
    out["mlp.up_gate_proj.weight"] = ArcPointer(_stack3(
        _passthrough(raw["mlp.up_gate_proj_video.weight"][], ctx),
        _passthrough(raw["mlp.up_gate_proj_audio.weight"][], ctx),
        _passthrough(raw["mlp.up_gate_proj_text.weight"][], ctx), ctx,
    ))
    out["mlp.down_proj.weight"] = ArcPointer(_stack3(
        _passthrough(raw["mlp.down_proj_video.weight"][], ctx),
        _passthrough(raw["mlp.down_proj_audio.weight"][], ctx),
        _passthrough(raw["mlp.down_proj_text.weight"][], ctx), ctx,
    ))

    # MM norms are stored [dim*3] already (one weight, 3 modality chunks) — same
    # as base. Just pre-add 1.
    var norms = [
        "attention.pre_norm.weight", "attention.q_norm.weight",
        "attention.k_norm.weight", "mlp.pre_norm.weight",
    ]
    for sfx in norms:
        var s = String(sfx)
        out[s + ".p1"] = ArcPointer(add_scalar(raw[s][], 1.0, ctx))
    return out^


def _stack3(var a: Tensor, var b: Tensor, var c: Tensor, ctx: DeviceContext) raises -> Tensor:
    var ab = _ta_concat(0, ctx, a, b)
    return _ta_concat(0, ctx, ab, c)


# Contiguous copy (slice the whole first dim) so the returned Tensor owns its
# bytes independent of the input dict entry.
from serenitymojo.ops.tensor_algebra import slice as _ta_slice
def _passthrough(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return _ta_slice(t, 0, 0, t.shape()[0], ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# UniPC distill step_ddim (magihuman_unipc.rs::step_ddim)
# ═══════════════════════════════════════════════════════════════════════════════
# The sigma schedule itself is built by sampling/unipc.build_unipc_sigma_schedule
# (already a line-for-line port of FlowUniPcDDim::new). This adds the one missing
# distill step:
#   curr_t = sigmas[idx]; prev_t = sigmas[idx+1]
#   cur_clean  = curr_state - curr_t * velocity
#   prev_state = prev_t * noise + (1 - prev_t) * cur_clean
# velocity/curr_state/noise are same shape & dtype; caller supplies noise (RNG
# controlled externally for replayable parity).
def magihuman_unipc_step_ddim(
    velocity: Tensor,
    idx: Int,
    curr_state: Tensor,
    noise: Tensor,
    sigmas: List[Float64],
    ctx: DeviceContext,
) raises -> Tensor:
    if idx < 0 or idx + 1 >= len(sigmas):
        raise Error("magihuman_unipc_step_ddim: idx out of range")
    var curr_t = Float32(sigmas[idx])
    var prev_t = Float32(sigmas[idx + 1])
    # cur_clean = curr_state - curr_t * velocity
    var v_scaled = mul_scalar(velocity, curr_t, ctx)
    var cur_clean = sub(curr_state, v_scaled, ctx)
    # prev_state = prev_t * noise + (1 - prev_t) * cur_clean
    var term_noise = mul_scalar(noise, prev_t, ctx)
    var term_clean = mul_scalar(cur_clean, Float32(1.0) - prev_t, ctx)
    return add(term_noise, term_clean, ctx)
