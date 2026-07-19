# serenitymojo/models/ltx2/parity/ltx2_av_mojo_parity.mojo
#
# REAL numerical parity for the LTX-2.3 22B AV transformer block (Mojo side).
#
# This runs the ACTUAL Mojo `ltx2_av_block_forward[1, 64, 16, 32]` with real
# block-0 weights from the checkpoint and compares its outputs against the
# oracle (block0_ref.safetensors) produced by musubi's
# BasicAVTransformerBlock._forward in Python.
#
# Run:
#   pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/ltx2/parity/ltx2_av_mojo_parity.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.ltx2.ltx2_av_block import (
    AVBlockWeights,
    ltx2_av_block_forward,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# block0_ref_halfd.safetensors: same inputs as block0_ref.safetensors BUT with
# PE tables in half-D format (cos/sin[B, S, D/2] from even-indexed elements of
# the original full-D random tables, then repeat_interleave(2) before passing to
# musubi, so the oracle PE matches Mojo's rope_interleaved half-D convention).
comptime REF_ST   = "/home/alex/mojodiffusion/output/ltx2_av/block0_ref_mojo_compat.safetensors"
comptime HALFD_PE = "/home/alex/mojodiffusion/output/ltx2_av/block0_halfd_pe.safetensors"
comptime INTER_ST = "/home/alex/mojodiffusion/output/ltx2_av/block0_intermediates.safetensors"
comptime CKPT     = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors"
comptime PFX      = "model.diffusion_model.transformer_blocks.0."

# ── Block-0 compile-time dims ─────────────────────────────────────────────────
comptime B     = 1
comptime Sv    = 64
comptime N_TXT = 16
comptime Sa    = 32
comptime EPS   = Float32(1e-6)


# ── Simple pair for cos + max_abs ─────────────────────────────────────────────
@fieldwise_init
struct CosResult(Copyable, Movable):
    var cos: Float64
    var max_abs: Float64


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cos_maxabs(a: List[Float32], b: List[Float32]) raises -> CosResult:
    """Cosine similarity and max-abs-err between two host F32 lists."""
    var n = len(a)
    if n != len(b):
        raise Error("cos_maxabs: length mismatch")
    var dot: Float64 = 0.0
    var na:  Float64 = 0.0
    var nb:  Float64 = 0.0
    var mx:  Float64 = 0.0
    for i in range(n):
        var ai = Float64(a[i])
        var bi = Float64(b[i])
        dot += ai * bi
        na  += ai * ai
        nb  += bi * bi
        var d = ai - bi
        if d < 0.0: d = -d
        if d > mx:  mx = d
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float64
    if denom == 0.0:
        cos = 1.0 if (na == 0.0 and nb == 0.0) else 0.0
    else:
        cos = dot / denom
    return CosResult(cos, mx)


def _load_ref_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    """Load a tensor from the ref safetensors as F32 host list."""
    var info  = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv    = from_parts(info.dtype, info.shape.copy(), bytes)
    var n     = tv.numel()
    var out   = List[Float32]()
    if tv.dtype == STDtype.F32:
        var p = tv.data.unsafe_ptr().bitcast[Float32]()
        for i in range(n):
            out.append(p[i])
    else:
        raise Error(String("_load_ref_f32: unexpected dtype: ") + tv.dtype.name())
    return out^


def _shape1(a: Int) -> List[Int]:
    return [a]

def _shape2(a: Int, b: Int) -> List[Int]:
    return [a, b]

def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    return [a, b, c]


def _load_ref_as_bf16(
    ref st: SafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Load a ref tensor (F32 on disk) as BF16 on device."""
    var info  = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv    = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view_as_bf16(tv, ctx)


def _ckpt_bf16(
    ref ckpt: SafeTensors, full_key: String, ctx: DeviceContext
) raises -> Tensor:
    """Load a checkpoint tensor (BF16 or F32 on disk) as BF16 on device."""
    var info  = ckpt.tensor_info(full_key)
    var bytes = ckpt.tensor_bytes(full_key)
    var tv    = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype == STDtype.BF16:
        return Tensor.from_view(tv, ctx)
    else:
        return Tensor.from_view_as_bf16(tv, ctx)


# ── Weight loader ─────────────────────────────────────────────────────────────

def _load_weights(ref ckpt: SafeTensors, ctx: DeviceContext) raises -> AVBlockWeights:
    """Load the 86 block-0 weights into AVBlockWeights (all bf16 on device)."""

    # Helper to build full key (avoids a nested closure)
    var pfx = String(PFX)

    # ── Video self-attn (attn1) ───────────────────────────────────────────────
    var v_wq = _ckpt_bf16(ckpt, pfx + "attn1.to_q.weight",        ctx)
    var v_bq = _ckpt_bf16(ckpt, pfx + "attn1.to_q.bias",          ctx)
    var v_wk = _ckpt_bf16(ckpt, pfx + "attn1.to_k.weight",        ctx)
    var v_bk = _ckpt_bf16(ckpt, pfx + "attn1.to_k.bias",          ctx)
    var v_wv = _ckpt_bf16(ckpt, pfx + "attn1.to_v.weight",        ctx)
    var v_bv = _ckpt_bf16(ckpt, pfx + "attn1.to_v.bias",          ctx)
    var v_qn = _ckpt_bf16(ckpt, pfx + "attn1.q_norm.weight",      ctx)
    var v_kn = _ckpt_bf16(ckpt, pfx + "attn1.k_norm.weight",      ctx)
    var v_gw = _ckpt_bf16(ckpt, pfx + "attn1.to_gate_logits.weight", ctx)
    var v_gb = _ckpt_bf16(ckpt, pfx + "attn1.to_gate_logits.bias",   ctx)
    var v_wo = _ckpt_bf16(ckpt, pfx + "attn1.to_out.0.weight",    ctx)
    var v_bo = _ckpt_bf16(ckpt, pfx + "attn1.to_out.0.bias",      ctx)

    # ── Video cross-attn (attn2) ──────────────────────────────────────────────
    var v2_wq = _ckpt_bf16(ckpt, pfx + "attn2.to_q.weight",       ctx)
    var v2_bq = _ckpt_bf16(ckpt, pfx + "attn2.to_q.bias",         ctx)
    var v2_wk = _ckpt_bf16(ckpt, pfx + "attn2.to_k.weight",       ctx)
    var v2_bk = _ckpt_bf16(ckpt, pfx + "attn2.to_k.bias",         ctx)
    var v2_wv = _ckpt_bf16(ckpt, pfx + "attn2.to_v.weight",       ctx)
    var v2_bv = _ckpt_bf16(ckpt, pfx + "attn2.to_v.bias",         ctx)
    var v2_qn = _ckpt_bf16(ckpt, pfx + "attn2.q_norm.weight",     ctx)
    var v2_kn = _ckpt_bf16(ckpt, pfx + "attn2.k_norm.weight",     ctx)
    var v2_gw = _ckpt_bf16(ckpt, pfx + "attn2.to_gate_logits.weight", ctx)
    var v2_gb = _ckpt_bf16(ckpt, pfx + "attn2.to_gate_logits.bias",   ctx)
    var v2_wo = _ckpt_bf16(ckpt, pfx + "attn2.to_out.0.weight",   ctx)
    var v2_bo = _ckpt_bf16(ckpt, pfx + "attn2.to_out.0.bias",     ctx)

    # ── Video FFN (ff) ────────────────────────────────────────────────────────
    var v_wff0 = _ckpt_bf16(ckpt, pfx + "ff.net.0.proj.weight",   ctx)
    var v_bff0 = _ckpt_bf16(ckpt, pfx + "ff.net.0.proj.bias",     ctx)
    var v_wff2 = _ckpt_bf16(ckpt, pfx + "ff.net.2.weight",        ctx)
    var v_bff2 = _ckpt_bf16(ckpt, pfx + "ff.net.2.bias",          ctx)

    # ── Video AdaLN tables (F32 on disk → bf16) ───────────────────────────────
    var v_table        = _ckpt_bf16(ckpt, pfx + "scale_shift_table",              ctx)
    var v_prompt_table = _ckpt_bf16(ckpt, pfx + "prompt_scale_shift_table",       ctx)
    var v_a2v_table    = _ckpt_bf16(ckpt, pfx + "scale_shift_table_a2v_ca_video", ctx)

    # ── Audio self-attn (audio_attn1) ─────────────────────────────────────────
    var a_wq = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_q.weight",      ctx)
    var a_bq = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_q.bias",        ctx)
    var a_wk = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_k.weight",      ctx)
    var a_bk = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_k.bias",        ctx)
    var a_wv = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_v.weight",      ctx)
    var a_bv = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_v.bias",        ctx)
    var a_qn = _ckpt_bf16(ckpt, pfx + "audio_attn1.q_norm.weight",    ctx)
    var a_kn = _ckpt_bf16(ckpt, pfx + "audio_attn1.k_norm.weight",    ctx)
    var a_gw = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_gate_logits.weight", ctx)
    var a_gb = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_gate_logits.bias",   ctx)
    var a_wo = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_out.0.weight",  ctx)
    var a_bo = _ckpt_bf16(ckpt, pfx + "audio_attn1.to_out.0.bias",    ctx)

    # ── Audio cross-attn (audio_attn2) ────────────────────────────────────────
    var a2_wq = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_q.weight",     ctx)
    var a2_bq = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_q.bias",       ctx)
    var a2_wk = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_k.weight",     ctx)
    var a2_bk = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_k.bias",       ctx)
    var a2_wv = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_v.weight",     ctx)
    var a2_bv = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_v.bias",       ctx)
    var a2_qn = _ckpt_bf16(ckpt, pfx + "audio_attn2.q_norm.weight",   ctx)
    var a2_kn = _ckpt_bf16(ckpt, pfx + "audio_attn2.k_norm.weight",   ctx)
    var a2_gw = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_gate_logits.weight", ctx)
    var a2_gb = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_gate_logits.bias",   ctx)
    var a2_wo = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_out.0.weight", ctx)
    var a2_bo = _ckpt_bf16(ckpt, pfx + "audio_attn2.to_out.0.bias",   ctx)

    # ── Audio FFN (audio_ff) ──────────────────────────────────────────────────
    var a_wff0 = _ckpt_bf16(ckpt, pfx + "audio_ff.net.0.proj.weight", ctx)
    var a_bff0 = _ckpt_bf16(ckpt, pfx + "audio_ff.net.0.proj.bias",   ctx)
    var a_wff2 = _ckpt_bf16(ckpt, pfx + "audio_ff.net.2.weight",      ctx)
    var a_bff2 = _ckpt_bf16(ckpt, pfx + "audio_ff.net.2.bias",        ctx)

    # ── Audio AdaLN tables ────────────────────────────────────────────────────
    var a_table        = _ckpt_bf16(ckpt, pfx + "audio_scale_shift_table",        ctx)
    var a_prompt_table = _ckpt_bf16(ckpt, pfx + "audio_prompt_scale_shift_table", ctx)
    var a_a2v_table    = _ckpt_bf16(ckpt, pfx + "scale_shift_table_a2v_ca_audio", ctx)

    # ── Cross-modal: audio_to_video_attn (Q=video [Dv=4096], KV=audio [Da=2048]) ─
    # to_q: [2048, 4096], to_k/v: [2048, 2048], to_out: [4096, 2048]
    var a2v_wq = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_q.weight",          ctx)
    var a2v_bq = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_q.bias",            ctx)
    var a2v_wk = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_k.weight",          ctx)
    var a2v_bk = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_k.bias",            ctx)
    var a2v_wv = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_v.weight",          ctx)
    var a2v_bv = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_v.bias",            ctx)
    var a2v_qn = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.q_norm.weight",        ctx)
    var a2v_kn = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.k_norm.weight",        ctx)
    var a2v_wo = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_out.0.weight",      ctx)
    var a2v_bo = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_out.0.bias",        ctx)
    # gate_in = modulated video Q [B, Sv, Dv=4096]
    var a2v_gw = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_gate_logits.weight", ctx)
    var a2v_gb = _ckpt_bf16(ckpt, pfx + "audio_to_video_attn.to_gate_logits.bias",   ctx)

    # ── Cross-modal: video_to_audio_attn (Q=audio [Da=2048], KV=video [Dv=4096]) ─
    # to_q: [2048, 2048], to_k/v: [2048, 4096], to_out: [2048, 2048]
    var v2a_wq = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_q.weight",          ctx)
    var v2a_bq = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_q.bias",            ctx)
    var v2a_wk = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_k.weight",          ctx)
    var v2a_bk = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_k.bias",            ctx)
    var v2a_wv = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_v.weight",          ctx)
    var v2a_bv = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_v.bias",            ctx)
    var v2a_qn = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.q_norm.weight",        ctx)
    var v2a_kn = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.k_norm.weight",        ctx)
    var v2a_wo = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_out.0.weight",      ctx)
    var v2a_bo = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_out.0.bias",        ctx)
    # gate_in = modulated audio Q [B, Sa, Da=2048]
    var v2a_gw = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_gate_logits.weight", ctx)
    var v2a_gb = _ckpt_bf16(ckpt, pfx + "video_to_audio_attn.to_gate_logits.bias",   ctx)

    return AVBlockWeights(
        # Video self-attn
        v_wq^, v_bq^, v_wk^, v_bk^, v_wv^, v_bv^, v_qn^, v_kn^,
        v_gw^, v_gb^, v_wo^, v_bo^,
        # Video cross-attn
        v2_wq^, v2_bq^, v2_wk^, v2_bk^, v2_wv^, v2_bv^, v2_qn^, v2_kn^,
        v2_gw^, v2_gb^, v2_wo^, v2_bo^,
        # Video FFN
        v_wff0^, v_bff0^, v_wff2^, v_bff2^,
        # Video AdaLN tables
        v_table^, v_prompt_table^, v_a2v_table^,
        # Audio self-attn
        a_wq^, a_bq^, a_wk^, a_bk^, a_wv^, a_bv^, a_qn^, a_kn^,
        a_gw^, a_gb^, a_wo^, a_bo^,
        # Audio cross-attn
        a2_wq^, a2_bq^, a2_wk^, a2_bk^, a2_wv^, a2_bv^, a2_qn^, a2_kn^,
        a2_gw^, a2_gb^, a2_wo^, a2_bo^,
        # Audio FFN
        a_wff0^, a_bff0^, a_wff2^, a_bff2^,
        # Audio AdaLN tables
        a_table^, a_prompt_table^, a_a2v_table^,
        # Cross-modal a2v
        a2v_wq^, a2v_bq^, a2v_wk^, a2v_bk^, a2v_wv^, a2v_bv^,
        a2v_qn^, a2v_kn^, a2v_wo^, a2v_bo^,
        a2v_gw^, a2v_gb^,
        # Cross-modal v2a
        v2a_wq^, v2a_bq^, v2a_wk^, v2a_bk^, v2a_wv^, v2a_bv^,
        v2a_qn^, v2a_kn^, v2a_wo^, v2a_bo^,
        v2a_gw^, v2a_gb^,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() raises:
    print("=== LTX-2.3 22B AV block-0 Mojo parity ===")

    var ctx = DeviceContext()

    # ── 1. Open ref safetensors (inputs + gold outputs) ───────────────────────
    print("[parity] opening ref safetensors:", REF_ST)
    var ref_st = SafeTensors.open(String(REF_ST))
    print("[parity] ref tensors:", ref_st.count())

    # ── 2. Load inputs (F32 on disk → bf16 on device) ────────────────────────
    print("[parity] loading inputs ...")

    var vx       = _load_ref_as_bf16(ref_st, "video_x",
                       _shape3(B, Sv, 4096), ctx)
    var ax       = _load_ref_as_bf16(ref_st, "audio_x",
                       _shape3(B, Sa, 2048), ctx)
    var ctx_v    = _load_ref_as_bf16(ref_st, "video_ctx",
                       _shape3(B, N_TXT, 4096), ctx)
    var ctx_a    = _load_ref_as_bf16(ref_st, "audio_ctx",
                       _shape3(B, N_TXT, 2048), ctx)

    # Timestep embeddings: oracle stores as [B, 1, 9*D] (T=1 broadcast dim).
    # The block's _ada_vec expects [B, 9*D] (2D flat). numel is identical;
    # we load then reshape to drop the T=1 middle dim.
    var vtemb_raw = _load_ref_as_bf16(ref_st, "video_timesteps",
                        _shape2(B, 9*4096), ctx)
    var vtemb    = reshape(vtemb_raw, _shape2(B, 9*4096), ctx)

    var atemb_raw = _load_ref_as_bf16(ref_st, "audio_timesteps",
                        _shape2(B, 9*2048), ctx)
    var atemb    = reshape(atemb_raw, _shape2(B, 9*2048), ctx)

    # prompt_ts: mojo_compat oracle stores [B, 1, 2*D] (N_TXT=1 mean-reduced).
    # Mojo _cross_attn_path expects prompt_temb [B, 2*D], so reshape to drop the T=1 dim.
    var vprompt_raw = _load_ref_as_bf16(ref_st, "video_prompt_ts",
                          _shape2(B, 2*4096), ctx)
    var vprompt  = reshape(vprompt_raw, _shape2(B, 2*4096), ctx)
    var aprompt_raw = _load_ref_as_bf16(ref_st, "audio_prompt_ts",
                          _shape2(B, 2*2048), ctx)
    var aprompt  = reshape(aprompt_raw, _shape2(B, 2*2048), ctx)

    # Cross-modal scale-shift & gate timesteps: oracle stores [B, 1, 4*D]
    # and [B, 1, D]; block expects [B, 4*D] and [B, D].
    var vcross_ss_raw = _load_ref_as_bf16(ref_st, "video_cross_ss_ts",
                            _shape2(B, 4*4096), ctx)
    var vcross_ss = reshape(vcross_ss_raw, _shape2(B, 4*4096), ctx)

    var vcross_g_raw = _load_ref_as_bf16(ref_st, "video_cross_gate_ts",
                           _shape2(B, 1*4096), ctx)
    var vcross_g  = reshape(vcross_g_raw, _shape2(B, 1*4096), ctx)

    var across_ss_raw = _load_ref_as_bf16(ref_st, "audio_cross_ss_ts",
                            _shape2(B, 4*2048), ctx)
    var across_ss = reshape(across_ss_raw, _shape2(B, 4*2048), ctx)

    var across_g_raw = _load_ref_as_bf16(ref_st, "audio_cross_gate_ts",
                           _shape2(B, 1*2048), ctx)
    var across_g  = reshape(across_g_raw, _shape2(B, 1*2048), ctx)

    # RoPE tables — loaded from the half-D file (even-indexed from original full-D).
    # Mojo rope_interleaved expects cos/sin[B*S*H, Dh/2]; we store them as
    # [B, S, Dh/2] and let the kernel flatten leading dims as [rows, Dh/2].
    # video self-attn: Dh=128 → half=64; store [B, Sv, H*Dh/2=2048]
    # (The kernel sees rows=B*Sv*H=2048, half=Dh/2=64 → numel=131072 ✓)
    var halfd_st   = SafeTensors.open(String(HALFD_PE))
    var vrope_cos  = _load_ref_as_bf16(halfd_st, "video_pe_cos_half",
                         _shape3(B, Sv, 2048), ctx)
    var vrope_sin  = _load_ref_as_bf16(halfd_st, "video_pe_sin_half",
                         _shape3(B, Sv, 2048), ctx)
    # audio self-attn: Dh=64 → half=32; store [B, Sa, H*Dh/2=1024]
    var arope_cos  = _load_ref_as_bf16(halfd_st, "audio_pe_cos_half",
                         _shape3(B, Sa, 1024), ctx)
    var arope_sin  = _load_ref_as_bf16(halfd_st, "audio_pe_sin_half",
                         _shape3(B, Sa, 1024), ctx)

    # Cross-modal PE: cross-modal attn uses audio head geometry (Dh_a=64 → half=32).
    # video side [B, Sv, H*Dh_a/2=1024], audio side [B, Sa, H*Dh_a/2=1024]
    var vcross_cos = _load_ref_as_bf16(halfd_st, "video_cross_pe_cos_half",
                         _shape3(B, Sv, 1024), ctx)
    var vcross_sin = _load_ref_as_bf16(halfd_st, "video_cross_pe_sin_half",
                         _shape3(B, Sv, 1024), ctx)
    var across_cos = _load_ref_as_bf16(halfd_st, "audio_cross_pe_cos_half",
                         _shape3(B, Sa, 1024), ctx)
    var across_sin = _load_ref_as_bf16(halfd_st, "audio_cross_pe_sin_half",
                         _shape3(B, Sa, 1024), ctx)

    # ── 3. Load gold outputs (F32) ────────────────────────────────────────────
    print("[parity] loading gold outputs ...")
    var gold_video = _load_ref_f32(ref_st, "video_out")
    var gold_audio = _load_ref_f32(ref_st, "audio_out")
    print("[parity] gold video_out numel:", len(gold_video),
          "  audio_out numel:", len(gold_audio))

    # ── 4. Open checkpoint and load block-0 weights ───────────────────────────
    print("[parity] opening checkpoint (mmap, header scan only) ...")
    var ckpt = SafeTensors.open(String(CKPT))
    print("[parity] checkpoint total tensors:", ckpt.count())
    print("[parity] loading block-0 weights ...")
    var weights = _load_weights(ckpt, ctx)
    print("[parity] weights loaded OK")

    # ── 5. Run Mojo forward ───────────────────────────────────────────────────
    print("[parity] running ltx2_av_block_forward[B=1, Sv=64, N_TXT=16, Sa=32] ...")
    var out = ltx2_av_block_forward[B, Sv, N_TXT, Sa](
        weights,
        vx,  ax,
        ctx_v, ctx_a,
        vtemb, atemb,
        vprompt, aprompt,
        vcross_ss, vcross_g,
        across_ss, across_g,
        vrope_cos, vrope_sin,
        arope_cos, arope_sin,
        vcross_cos, vcross_sin,
        across_cos, across_sin,
        EPS,
        ctx,
    )
    ctx.synchronize()
    print("[parity] forward done")

    # ── 6. Read back Mojo outputs ─────────────────────────────────────────────
    var mojo_video = out.vx.to_host(ctx)
    var mojo_audio = out.ax.to_host(ctx)

    print("[parity] mojo video_out numel:", len(mojo_video),
          "  audio_out numel:", len(mojo_audio))

    # Sanity: non-zero check (all-zero would indicate a dead forward)
    var vsum: Float64 = 0.0
    var asum: Float64 = 0.0
    for i in range(len(mojo_video)):
        var v = Float64(mojo_video[i])
        if v < 0.0: v = -v
        vsum += v
    for i in range(len(mojo_audio)):
        var v = Float64(mojo_audio[i])
        if v < 0.0: v = -v
        asum += v
    print("[parity] mojo video abs-sum:", vsum, "  audio abs-sum:", asum)
    if vsum == 0.0:
        print("[parity] WARNING: mojo video output is all-zero (dead forward?)")
    if asum == 0.0:
        print("[parity] WARNING: mojo audio output is all-zero (dead forward?)")

    # ── 7. Compare ────────────────────────────────────────────────────────────
    var vr = _cos_maxabs(mojo_video, gold_video)
    var ar = _cos_maxabs(mojo_audio, gold_audio)

    print("")
    print("=== PARITY RESULTS ===")
    print("video: cos =", vr.cos, "  max_abs_err =", vr.max_abs,
          "  [PASS]" if vr.cos >= 0.99 else "  [FAIL]")
    print("audio: cos =", ar.cos, "  max_abs_err =", ar.max_abs,
          "  [PASS]" if ar.cos >= 0.99 else "  [FAIL]")

    # ── 8. Oracle self-consistency check via intermediates ────────────────────
    print("")
    print("[parity] loading intermediates for oracle self-check ...")
    var inter_st   = SafeTensors.open(String(INTER_ST))
    var inter_vout = _load_ref_f32(inter_st, "video_out")
    var inter_aout = _load_ref_f32(inter_st, "audio_out")
    var vc = _cos_maxabs(gold_video, inter_vout)
    var ac = _cos_maxabs(gold_audio, inter_aout)
    print("oracle self-check: video cos =", vc.cos, "  audio cos =", ac.cos)
    if vc.cos < 0.9999 or ac.cos < 0.9999:
        print("WARNING: ref vs intermediates mismatch — oracle may be inconsistent")

    print("")
    print("=== DONE ===")
