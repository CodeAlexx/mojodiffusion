# serenitymojo/models/scail2/scail2_block_train.mojo
#
# One SCAIL-2 i2v transformer block: forward (saving activations) + hand-chained
# backward (training) + LoRA variants (12 targets). This is a THIN FORK of the
# gated Wan2.2 i2v training block
#   serenitymojo/models/wan22/wan22_block.mojo::wan22_i2v_block_lora_{forward,backward}
# and mirrors the inference forward
#   serenitymojo/models/scail2/scail2_block.mojo::scail2_block_forward.
#
# THE ONLY MATH CHANGE vs the Wan2.2 i2v block is the RESIDUAL DTYPE: SCAIL-2
# pins its residual stream in F32 (scail2_gated_residual_f32 / scail2_residual_f32,
# scail2_block.mojo:56-75). The three residual adds (self-attn gated, cross-attn
# ungated, ffn gated) accumulate/retain F32; the self-/ffn sub-layer inputs are
# re-cast to bf16 before their linears (F32 modulate), and norm3's input is
# re-cast to bf16 (norm3.weight dtype). The residual-branch backward operates on
# the F32 stream; the attention/FFN/norm interiors stay bf16 exactly as Wan2.2.
#
# Everything else MATCHES Wan2.2 i2v exactly and is REUSED unchanged (imported):
#   - self-attn: mod_pre -> q/k/v linear -> rms_norm(q,k) -> to_bshd -> rope
#     (interleaved, _expand_rope_per_head) -> sdpa scale=1/sqrt(DH) -> o linear
#     -> gated residual.
#   - cross-attn: layer_norm(norm3) -> caq linear + rms_norm(caq); text_k/v +
#     rms_norm(text_k); img k_img/v_img + rms_norm(img_k); TWO sdpa_rect (text
#     S×TXT, image S×IMG, shared query) -> ADD -> o linear -> UNGATED residual.
#     (v is NOT rms-normed.)
#   - FFN: ffn.0 -> gelu(TANH) -> ffn.2 -> gated residual.
#   - modulation (AdaLN): 6-way shift/scale/gate reused via the external
#     WanModVecs seam; SCAIL-2 modulates in F32 (F32 vectors), matching
#     scail2_block.mojo (dit wan22_mod_pre F32 modulate + gate_f32).
#   - 12 LoRA targets: self_attn.{q,k,v,o}, cross_attn.{q,k,v,o},
#     cross_attn.{k_img,v_img}, ffn.0, ffn.2.
#
# Backward arms reused verbatim: wan_gate_residual_backward + wan_modulate_backward
# (both pure elementwise; F32-safe), linear_backward, gelu_backward,
# layer_norm_backward_dx, rms_norm_backward_dx, sdpa_backward, sdpa_backward_rect,
# rope_backward, _lora_bwd_opt. No new kernels.
#
# Seam contract (identical to wan22 i2v): host List[Float32] weight boundary
# (from_host -> bf16); LoRA A/B stored bf16, grads flow F32; multi-consumer grad
# fan-in accumulated in a FIXED left-fold order (shared cross-attn query grad =
# d_q_text + d_q_image) for bit-stability.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# ── forward / backward ops ────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd, sdpa_flash_backward,
)
from serenitymojo.ops.tensor_algebra import reshape, add, mul, add_scalar
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward_dx, layer_norm_backward_dx,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_rect, SdpaGrads,
)
from serenitymojo.ops.rope_struct_backward import rope_backward

# ── reused wan22 i2v structs + helpers (the block is a thin fork) ─────────────
from serenitymojo.models.wan22.wan22_block import (
    # host helpers
    _add_lists, _ones, _zeros, _t16, _ta16, _clone_t,
    _to_bshd, _from_bshd, _expand_rope_per_head, _cross_attention,
    _add_lora_delta, _lora_bwd_opt,
    # AdaLN primitives (elementwise; F32-safe) + carriers
    ModPre, wan_modulate_backward, wan_gate_residual_backward,
    WanModVecs,
    # weight / lora / grad structs (reused unchanged)
    WanI2VBlockWeights, WanI2VBlockLora,
    WanBlockGrads, WanBlockLoraGrads, WanI2VBlockLoraGrads,
)
from serenitymojo.models.klein.lora_block import KleinLoraGrads


comptime TArc = ArcPointer[Tensor]


# ── SCAIL-2 F32-residual AdaLN forward primitives ─────────────────────────────
# mod_pre: o = layer_norm(x) [in x.dtype] -> cast F32 -> (1+scale_f32)*ln + shift_f32
# -> cast BF16 (the linear input). Returns (o_bf16, ln_f32). ln_f32 is saved for
# the F32 modulate backward. Matches scail2_block.mojo (dit wan22_mod_pre: normed
# cast to F32, F32 scale/shift) + the bf16 sub-layer input re-cast.
def scail2_mod_pre(
    x: Tensor, scale_f32: Tensor, shift_f32: Tensor, dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ModPre:
    var ones_t = Tensor.from_host(_ones(dim), [dim], x.dtype(), ctx)
    var zeros_t = Tensor.from_host(_zeros(dim), [dim], x.dtype(), ctx)
    var ln = layer_norm(x, ones_t, zeros_t, eps, ctx)          # x.dtype
    var ln_f32 = cast_tensor(ln, STDtype.F32, ctx)
    var sc1 = add_scalar(scale_f32, 1.0, ctx)                  # F32
    var prod = mul(ln_f32, sc1, ctx)                           # F32
    var o_f32 = add(prod, shift_f32, ctx)                      # F32
    var o_bf16 = cast_tensor(o_f32, STDtype.BF16, ctx)         # linear input
    return ModPre(TArc(o_bf16^), TArc(ln_f32^))


# F32 gated residual: o = f32(x) + f32(y) * gate_f32.
def scail2_gated_residual_f32(
    x: Tensor, y: Tensor, gate_f32: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var xf = cast_tensor(x, STDtype.F32, ctx)
    var yf = cast_tensor(y, STDtype.F32, ctx)
    return add(xf, mul(yf, gate_f32, ctx), ctx)


# F32 ungated residual: o = f32(x) + f32(y).
def scail2_residual_f32(x: Tensor, y: Tensor, ctx: DeviceContext) raises -> Tensor:
    return add(cast_tensor(x, STDtype.F32, ctx), cast_tensor(y, STDtype.F32, ctx), ctx)


# ── SELF-attention forward/backward: DENSE (chunk-1) vs FLASH (chunk-2c) ──────
# The self-attn is NON-CAUSAL full attention over the whole concatenated
# sequence, softmax scale = 1/sqrt(DH). The DENSE path materializes an [1,H,S,S]
# score matrix (sdpa_nomask / sdpa_backward) — ~254 GB at the real video S≈39,872
# → OOM. The FLASH path routes q/k/v [1,S,H,DH] bf16 through the cuDNN v9 flash
# shim (ops/attention_flash.mojo), which never materializes S×S; peak memory is
# O(S). Everything ELSE in the block is byte-identical between the two paths;
# only the attention core is swapped, gated on the comptime FLASH flag.
#
# q_rope/k_rope/v are bf16 [1,S,H,DH]; the flash forward pads S→128-multiple and
# masks the pad rows via the real-length arg, so the output over the real S rows
# equals full attention. The flash BACKWARD RECOMPUTES the flash forward from the
# saved q_rope/k_rope/v (already host-saved as sa_q_rope/sa_k_rope/sa_v), so the
# Scail2Saved struct is UNCHANGED between the dense and flash variants. Flash bwd
# dQ is nondeterministic run-to-run (~1e-4) — a VALUE-class, not a bit gate.
def _scail2_sa_fwd[FLASH: Bool, S: Int, H: Int, DH: Int](
    q_rope: Tensor, k_rope: Tensor, v4: Tensor, scale: Float32,
    dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    comptime if FLASH:
        var ff = sdpa_flash_train_fwd[1, S, H, DH](q_rope, k_rope, v4, scale, ctx)
        # ff.o is a (possibly padded-buffer) view; buf.copy() bumps the refcount
        # so the [1,S,H,DH] output outlives ff (the klein/krea2 flash pattern).
        var att4 = Tensor(ff.o.buf.copy(), ff.o.shape(), ff.o.dtype())
        return _from_bshd(att4^, S, dim, ctx)
    else:
        var att4 = sdpa_nomask[1, S, H, DH](q_rope, k_rope, v4, scale, ctx)
        return _from_bshd(att4^, S, dim, ctx)


def _scail2_sa_bwd[FLASH: Bool, S: Int, H: Int, DH: Int](
    q_rope4: Tensor, k_rope4: Tensor, v4: Tensor, d_att4: Tensor,
    scale: Float32, ctx: DeviceContext,
) raises -> SdpaGrads:
    comptime if FLASH:
        # recompute the flash forward (q/k/v/o/stats) then run the flash backward.
        var ff = sdpa_flash_train_fwd[1, S, H, DH](q_rope4, k_rope4, v4, scale, ctx)
        var fb = sdpa_flash_backward[1, S, H, DH](ff, d_att4, scale, ctx)
        # buf.copy() refcount-bumps so fb destroys cleanly (no partial move).
        return SdpaGrads(
            Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype()),
            Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype()),
            Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype()),
        )
    else:
        return sdpa_backward[1, S, H, DH](q_rope4, k_rope4, v4, d_att4, scale, ctx)


# ── saved activations (DEVICE-RESIDENT on-GPU tape; TArc = ArcPointer[Tensor]).
#    The tape used to be host List[BFloat16]/List[Float32] (device→host per op in
#    the forward, host→device per op in the backward — ~93 round-trips/block that
#    made the trainer CPU-bound at ~195 s/step). It is now the SAME activations,
#    SAME dtypes (bf16 for the attention/FFN interior, F32 for the residual /
#    modulate stream), just kept ON DEVICE: the forward stores the already-on-GPU
#    Tensor (ArcPointer refcount bump — no copy), and the backward reads it back
#    straight into the (already device-native) backward arms. The bf16 device
#    tensor holds the same bytes the old to_host_bf16→from_host_bf16 path produced,
#    so the dense path stays bit-identical. Per-block RECOMPUTE keeps only ONE
#    block's tape live at a time (streaming trainer), so device VRAM stays ~one
#    block. Shapes: the *_pre fields keep their [1,M,H,Dh] forward layout (same
#    bytes as [M,dim]); the backward reshapes them to [M,dim] at the rms-norm arm.
@fieldwise_init
struct Scail2Saved(Copyable, Movable):
    var x: TArc            # [S,dim]      block input (bf16)
    var sa_ln: TArc        # [S,dim]  F32 ln (modulate input, self-attn)
    var sa_in: TArc        # [S,dim]      mod_pre output (self-attn linear input)
    var sa_q_pre: TArc     # [1,S,H,Dh]   q_flat reshaped (rms-norm bwd input as [S,dim])
    var sa_k_pre: TArc      # [1,S,H,Dh]
    var sa_v: TArc          # [1,S,H,Dh]
    var sa_q_rope: TArc     # [1,S,H,Dh]
    var sa_k_rope: TArc     # [1,S,H,Dh]
    var sa_att: TArc       # [S,dim]      attn out (pre o-linear)
    var x_sa: TArc         # [S,dim]  F32 gated residual after self-attn
    var ca_n3: TArc        # [S,dim]      layer_norm(norm3) on bf16 x_sa
    var ca_q_pre: TArc     # [1,S,H,Dh]
    var ca_q_rms: TArc     # [1,S,H,Dh]
    var context_txt: TArc  # [TXT,dim]
    var ca_k_pre: TArc     # [1,TXT,H,Dh]
    var ca_k_rms: TArc     # [1,TXT,H,Dh]
    var ca_v: TArc         # [1,TXT,H,Dh]
    var img_context: TArc  # [IMG,dim]
    var img_k_pre: TArc    # [1,IMG,H,Dh]
    var img_k_rms: TArc    # [1,IMG,H,Dh]
    var img_v: TArc        # [1,IMG,H,Dh]
    var ca_att: TArc       # [S,dim]      x_txt + x_img (pre o-linear)
    var x_ca: TArc         # [S,dim]  F32 ungated residual after cross-attn
    var ffn_ln: TArc       # [S,dim]  F32 ln (modulate input, ffn) — LN(x_ca F32)
    var ffn_in: TArc       # [S,dim]      mod_pre output (ffn.0 input)
    var ffn_h: TArc        # [S,ffn]      ffn.0 out (pre-gelu)
    var ffn_act: TArc      # [S,ffn]      gelu(ffn_h)


struct Scail2BlockForward(Copyable, Movable):
    var x_out: List[Float32]        # [S,dim] F32
    var saved: Scail2Saved

    def __init__(out self, var x_out: List[Float32], var saved: Scail2Saved):
        self.x_out = x_out^
        self.saved = saved^


# Reuse the wan22 i2v weight / lora / grad structs verbatim.
comptime Scail2BlockLora = WanI2VBlockLora
comptime Scail2BlockLoraGrads = WanI2VBlockLoraGrads
comptime Scail2BlockWeights = WanI2VBlockWeights


# ── FORWARD (F32 residual, saving activations, 12 LoRA targets) ───────────────
def scail2_block_lora_forward[
    S: Int, TXT: Int, IMG: Int, H: Int, DH: Int, FLASH: Bool = False
](
    x_h: List[Float32], context_txt_h: List[Float32], context_img_h: List[Float32],
    mv: WanModVecs, w: WanI2VBlockWeights, lora: WanI2VBlockLora,
    cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> Scail2BlockForward:
    var scale = Float32(1.0) / sqrt(Float32(DH))

    var x = _ta16(x_h, [S, dim], ctx)
    var context_txt = _ta16(context_txt_h, [TXT, dim], ctx)
    var context_img = _ta16(context_img_h, [IMG, dim], ctx)
    var context_txt_host = context_txt_h.copy()
    var context_img_host = context_img_h.copy()

    # F32 AdaLN vectors [S,dim] (SCAIL-2 modulates in F32).
    var shift_sa = Tensor.from_host(mv.shift_sa.copy(), [S, dim], STDtype.F32, ctx)
    var scale_sa = Tensor.from_host(mv.scale_sa.copy(), [S, dim], STDtype.F32, ctx)
    var gate_sa = Tensor.from_host(mv.gate_sa.copy(), [S, dim], STDtype.F32, ctx)
    var shift_ffn = Tensor.from_host(mv.shift_ffn.copy(), [S, dim], STDtype.F32, ctx)
    var scale_ffn = Tensor.from_host(mv.scale_ffn.copy(), [S, dim], STDtype.F32, ctx)
    var gate_ffn = Tensor.from_host(mv.gate_ffn.copy(), [S, dim], STDtype.F32, ctx)

    var cos16 = cast_tensor(cos, STDtype.BF16, ctx)
    var sin16 = cast_tensor(sin, STDtype.BF16, ctx)

    # ── self-attention (LoRA on q/k/v/o) ──
    var sa_mp = scail2_mod_pre(x[], scale_sa, shift_sa, dim, eps, ctx)
    var sa_in_h = sa_mp.o[].to_host(ctx)
    var q_base = linear(sa_mp.o[], w.base.sa_wq[], Optional[Tensor](_clone_t(w.base.sa_bq[], ctx)), ctx)
    var k_base = linear(sa_mp.o[], w.base.sa_wk[], Optional[Tensor](_clone_t(w.base.sa_bk[], ctx)), ctx)
    var v_base = linear(sa_mp.o[], w.base.sa_wv[], Optional[Tensor](_clone_t(w.base.sa_bv[], ctx)), ctx)
    var q_flat = _add_lora_delta(q_base, sa_in_h, lora.base.sa_q, S, ctx)
    var k_flat = _add_lora_delta(k_base, sa_in_h, lora.base.sa_k, S, ctx)
    var v_flat = _add_lora_delta(v_base, sa_in_h, lora.base.sa_v, S, ctx)
    var q_rms_flat = rms_norm(q_flat, w.base.sa_qn[], eps, ctx)
    var k_rms_flat = rms_norm(k_flat, w.base.sa_kn[], eps, ctx)
    var q_pre = _to_bshd(q_flat^, S, H, DH, ctx)
    var k_pre = _to_bshd(k_flat^, S, H, DH, ctx)
    var v4 = _to_bshd(v_flat^, S, H, DH, ctx)
    var q_rms = _to_bshd(q_rms_flat^, S, H, DH, ctx)
    var k_rms = _to_bshd(k_rms_flat^, S, H, DH, ctx)
    var cos_e = _expand_rope_per_head(cos16, S, H, DH // 2, ctx)
    var sin_e = _expand_rope_per_head(sin16, S, H, DH // 2, ctx)
    var q_rope = rope_interleaved(q_rms, cos_e, sin_e, ctx)
    var k_rope = rope_interleaved(k_rms, cos_e, sin_e, ctx)
    var sa_att = _scail2_sa_fwd[FLASH, S, H, DH](q_rope, k_rope, v4, scale, dim, ctx)
    var sa_att_h = sa_att.to_host(ctx)
    var sa_out_base = linear(sa_att, w.base.sa_wo[], Optional[Tensor](_clone_t(w.base.sa_bo[], ctx)), ctx)
    var sa_out = _add_lora_delta(sa_out_base, sa_att_h, lora.base.sa_o, S, ctx)
    var x_sa = scail2_gated_residual_f32(x[], sa_out, gate_sa, ctx)         # F32

    # ── DUAL cross-attention (text + image; shared query). norm3 on bf16 x_sa ──
    var x_sa_bf16 = cast_tensor(x_sa, STDtype.BF16, ctx)
    var n3 = layer_norm(x_sa_bf16, w.base.n3_w[], w.base.n3_b[], eps, ctx)
    var n3_h = n3.to_host(ctx)
    var caq_base = linear(n3, w.base.ca_wq[], Optional[Tensor](_clone_t(w.base.ca_bq[], ctx)), ctx)
    var caq_flat = _add_lora_delta(caq_base, n3_h, lora.base.ca_q, S, ctx)
    var caq_rms_flat = rms_norm(caq_flat, w.base.ca_qn[], eps, ctx)
    var caq_pre = _to_bshd(caq_flat^, S, H, DH, ctx)
    var caq_rms = _to_bshd(caq_rms_flat^, S, H, DH, ctx)
    # TEXT branch
    var cak_base = linear(context_txt[], w.base.ca_wk[], Optional[Tensor](_clone_t(w.base.ca_bk[], ctx)), ctx)
    var cav_base = linear(context_txt[], w.base.ca_wv[], Optional[Tensor](_clone_t(w.base.ca_bv[], ctx)), ctx)
    var cak_flat = _add_lora_delta(cak_base, context_txt_host, lora.base.ca_k, TXT, ctx)
    var cav_flat = _add_lora_delta(cav_base, context_txt_host, lora.base.ca_v, TXT, ctx)
    var cak_rms_flat = rms_norm(cak_flat, w.base.ca_kn[], eps, ctx)
    var cak_pre = reshape(cak_flat^, [1, TXT, H, DH], ctx)
    var cav4 = reshape(cav_flat^, [1, TXT, H, DH], ctx)
    var cak_rms = reshape(cak_rms_flat^, [1, TXT, H, DH], ctx)
    var ca_att4_txt = _cross_attention[S, TXT, H, DH](caq_rms, cak_rms, cav4, scale, ctx)
    var ca_att_txt = _from_bshd(ca_att4_txt^, S, dim, ctx)
    # IMAGE branch (k_img -> norm_k_img, v_img no norm)
    var imgk_base = linear(context_img[], w.ca_wk_img[], Optional[Tensor](_clone_t(w.ca_bk_img[], ctx)), ctx)
    var imgv_base = linear(context_img[], w.ca_wv_img[], Optional[Tensor](_clone_t(w.ca_bv_img[], ctx)), ctx)
    var imgk_flat = _add_lora_delta(imgk_base, context_img_host, lora.img_k, IMG, ctx)
    var imgv_flat = _add_lora_delta(imgv_base, context_img_host, lora.img_v, IMG, ctx)
    var imgk_rms_flat = rms_norm(imgk_flat, w.ca_kn_img[], eps, ctx)
    var imgk_pre = reshape(imgk_flat^, [1, IMG, H, DH], ctx)
    var imgv4 = reshape(imgv_flat^, [1, IMG, H, DH], ctx)
    var imgk_rms = reshape(imgk_rms_flat^, [1, IMG, H, DH], ctx)
    var ca_att4_img = _cross_attention[S, IMG, H, DH](caq_rms, imgk_rms, imgv4, scale, ctx)
    var ca_att_img = _from_bshd(ca_att4_img^, S, dim, ctx)
    # ADD the two branch outputs BEFORE the shared output projection
    var ca_att = add(ca_att_txt, ca_att_img, ctx)
    var ca_att_h = ca_att.to_host(ctx)
    var ca_out_base = linear(ca_att, w.base.ca_wo[], Optional[Tensor](_clone_t(w.base.ca_bo[], ctx)), ctx)
    var ca_out = _add_lora_delta(ca_out_base, ca_att_h, lora.base.ca_o, S, ctx)
    var x_ca = scail2_residual_f32(x_sa, ca_out, ctx)                       # F32

    # ── FFN (LoRA on ffn.0 / ffn.2). norm2 sees full F32 x_ca ──
    var ffn_mp = scail2_mod_pre(x_ca, scale_ffn, shift_ffn, dim, eps, ctx)
    var ffn_in_h = ffn_mp.o[].to_host(ctx)
    var ffn_h_base = linear(ffn_mp.o[], w.base.ffn0_w[], Optional[Tensor](_clone_t(w.base.ffn0_b[], ctx)), ctx)
    var ffn_h = _add_lora_delta(ffn_h_base, ffn_in_h, lora.base.ffn0, S, ctx)
    var ffn_act = gelu(ffn_h, ctx)
    var ffn_act_h = ffn_act.to_host(ctx)
    var ffn_out_base = linear(ffn_act, w.base.ffn2_w[], Optional[Tensor](_clone_t(w.base.ffn2_b[], ctx)), ctx)
    var ffn_out = _add_lora_delta(ffn_out_base, ffn_act_h, lora.base.ffn2, S, ctx)
    var x_final = scail2_gated_residual_f32(x_ca, ffn_out, gate_ffn, ctx)   # F32

    var x_out = x_final.to_host(ctx)
    # ── DEVICE TAPE: store the already-on-GPU activations directly (ArcPointer
    #    refcount bump for the TArc-carried ones; move the plain Tensors into a
    #    fresh TArc). No to_host/to_host_bf16 — the tape never leaves the GPU. ──
    var saved = Scail2Saved(
        x.copy(),
        sa_mp.ln.copy(), sa_mp.o.copy(),
        TArc(q_pre^), TArc(k_pre^), TArc(v4^),
        TArc(q_rope^), TArc(k_rope^),
        TArc(sa_att^), TArc(x_sa^),
        TArc(n3^), TArc(caq_pre^), TArc(caq_rms^),
        context_txt.copy(),
        TArc(cak_pre^), TArc(cak_rms^), TArc(cav4^),
        context_img.copy(),
        TArc(imgk_pre^), TArc(imgk_rms^), TArc(imgv4^),
        TArc(ca_att^), TArc(x_ca^),
        ffn_mp.ln.copy(), ffn_mp.o.copy(),
        TArc(ffn_h^), TArc(ffn_act^),
    )
    return Scail2BlockForward(x_out^, saved^)


# ── BACKWARD (F32 residual, 12 LoRA targets) ─────────────────────────────────
def scail2_block_lora_backward[
    S: Int, TXT: Int, IMG: Int, H: Int, DH: Int, FLASH: Bool = False
](
    d_out_h: List[Float32], mv: WanModVecs, w: WanI2VBlockWeights,
    lora: WanI2VBlockLora, saved: Scail2Saved, cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> WanI2VBlockLoraGrads:
    var scale = Float32(1.0) / sqrt(Float32(DH))
    var cos_e = _expand_rope_per_head(cos, S, H, DH // 2, ctx)              # F32
    var sin_e = _expand_rope_per_head(sin, S, H, DH // 2, ctx)
    var ones_f32 = Tensor.from_host(_ones(dim), [dim], STDtype.F32, ctx)
    var ones_bf16 = Tensor.from_host(_ones(dim), [dim], STDtype.BF16, ctx)

    # ── DEVICE TAPE READ: the saved activations are already on the GPU
    #    (Scail2Saved now carries device TArc), so the backward arms read them
    #    straight from `saved.<field>[]` — the ~93 per-block from_host reloads are
    #    GONE. The ONLY to_host that remain here are the LoRA-backward host inputs
    #    (the LoRA math is host-side) + the block-OUTPUT boundary grads. ──
    var sa_in_h = saved.sa_in[].to_host(ctx)
    var sa_att_h = saved.sa_att[].to_host(ctx)
    var n3_h = saved.ca_n3[].to_host(ctx)
    var context_txt_h = saved.context_txt[].to_host(ctx)
    var context_img_h = saved.img_context[].to_host(ctx)
    var ca_att_h = saved.ca_att[].to_host(ctx)

    var d_out = Tensor.from_host(d_out_h.copy(), [S, dim], STDtype.F32, ctx)  # F32

    # ════════════════ FFN backward (LoRA on ffn.0 / ffn.2) ════════════════
    var gate_ffn_t = Tensor.from_host(mv.gate_ffn.copy(), [S, dim], STDtype.F32, ctx)
    var ffn_act_h = saved.ffn_act[].to_host(ctx)
    # recompute ffn_out (base only; y is used only for the frozen d_gate) in F32
    var ffn_out_rc = cast_tensor(
        linear(saved.ffn_act[], w.base.ffn2_w[], Optional[Tensor](_clone_t(w.base.ffn2_b[], ctx)), ctx),
        STDtype.F32, ctx,
    )
    var gb_ffn2 = wan_gate_residual_backward(d_out, ffn_out_rc, gate_ffn_t, ctx)
    var d_gate_ffn = gb_ffn2.d_gate.to_host(ctx)
    var d_x_ca_resid = TArc(gb_ffn2.d_x.clone(ctx))                          # F32
    var d_ffn_out_bf16 = cast_tensor(gb_ffn2.d_y, STDtype.BF16, ctx)
    var d_ffn_out_h = d_ffn_out_bf16.to_host(ctx)
    # FROZEN base ffn.2: dx-only (d_w/d_b discarded by the stack — skip them).
    var dx_ffn2 = linear_backward_dx(d_ffn_out_bf16, w.base.ffn2_w[], S, ffn, dim, ctx)
    var d_ffn2_w = List[Float32]()
    var d_ffn2_b = List[Float32]()
    var ffn2_g = _lora_bwd_opt(lora.base.ffn2, d_ffn_out_h, ffn_act_h, S, ffn, ctx)
    var d_ffn_act = add(dx_ffn2, _t16(ffn2_g.d_x.copy(), [S, ffn], ctx), ctx)
    var d_ffn_h = gelu_backward(d_ffn_act, saved.ffn_h[], ctx)
    var ffn_in_h = saved.ffn_in[].to_host(ctx)
    # FROZEN base ffn.0: dx-only.
    var dx_ffn0 = linear_backward_dx(d_ffn_h, w.base.ffn0_w[], S, dim, ffn, ctx)
    var d_ffn0_w = List[Float32]()
    var d_ffn0_b = List[Float32]()
    var d_ffn_h_h = d_ffn_h.to_host(ctx)
    var ffn0_g = _lora_bwd_opt(lora.base.ffn0, d_ffn_h_h, ffn_in_h, S, dim, ctx)
    var d_ffn_in = add(dx_ffn0, _t16(ffn0_g.d_x.copy(), [S, dim], ctx), ctx)  # bf16
    # F32 modulate backward (ln = LN(x_ca F32); layer_norm_backward_dx on F32 x_ca)
    var d_ffn_in_f32 = cast_tensor(d_ffn_in, STDtype.F32, ctx)
    var scale_ffn_t = Tensor.from_host(mv.scale_ffn.copy(), [S, dim], STDtype.F32, ctx)
    var mb_ffn = wan_modulate_backward(d_ffn_in_f32, saved.ffn_ln[], scale_ffn_t, ctx)
    var d_scale_ffn = mb_ffn.d_scale.to_host(ctx)
    var d_shift_ffn = mb_ffn.d_shift.to_host(ctx)
    var lnb_ffn = layer_norm_backward_dx(mb_ffn.d_ln, saved.x_ca[], ones_f32, eps, ctx)  # F32
    var d_x_ca = TArc(add(d_x_ca_resid[], lnb_ffn, ctx))                     # F32

    # ════════════════ DUAL cross-attention backward (LoRA q/k/v/o + k_img/v_img)
    var d_x_ca_bf16 = cast_tensor(d_x_ca[], STDtype.BF16, ctx)
    var d_ca_out_h = d_x_ca_bf16.to_host(ctx)
    # FROZEN base cross_attn.o: dx-only.
    var dx_cao = linear_backward_dx(d_x_ca_bf16, w.base.ca_wo[], S, dim, dim, ctx)
    var d_ca_wo = List[Float32]()
    var d_ca_bo = List[Float32]()
    var ca_o_g = _lora_bwd_opt(lora.base.ca_o, d_ca_out_h, ca_att_h, S, dim, ctx)
    # grad of the SUM (o-input); the add sends this SAME grad to BOTH branches.
    var d_ca_att = add(dx_cao, _t16(ca_o_g.d_x.copy(), [S, dim], ctx), ctx)
    var d_ca_att4_txt = reshape(d_ca_att, [1, S, H, DH], ctx)
    var d_ca_att4_img = reshape(d_ca_att, [1, S, H, DH], ctx)

    # TEXT branch SDPA backward (kv=TXT); shared saved query re-borrowed for IMG
    var csb_txt = sdpa_backward_rect[1, S, TXT, H, DH](
        saved.ca_q_rms[], saved.ca_k_rms[], saved.ca_v[], d_ca_att4_txt, scale, ctx,
    )
    # IMAGE branch SDPA backward (kv=IMG), SAME saved query
    var csb_img = sdpa_backward_rect[1, S, IMG, H, DH](
        saved.ca_q_rms[], saved.img_k_rms[], saved.img_v[], d_ca_att4_img, scale, ctx,
    )
    # shared query grad = TEXT + IMG (FIXED left-fold order)
    var d_caq_rms4 = add(csb_txt.d_q, csb_img.d_q, ctx)
    var d_caq_rms_flat = reshape(d_caq_rms4, [S, dim], ctx)
    # *_pre saved as [1,M,H,DH]; reshape to [M,dim] (same bytes) for the rms arm.
    var lsv_ca_q_pre = reshape(saved.ca_q_pre[], [S, dim], ctx)
    var rb_caq_dx = rms_norm_backward_dx(d_caq_rms_flat, lsv_ca_q_pre, w.base.ca_qn[], eps, ctx)
    # TEXT k/v grads
    var d_cak_rms_flat = reshape(csb_txt.d_k, [TXT, dim], ctx)
    var lsv_ca_k_pre = reshape(saved.ca_k_pre[], [TXT, dim], ctx)
    var rb_cak_dx = rms_norm_backward_dx(d_cak_rms_flat, lsv_ca_k_pre, w.base.ca_kn[], eps, ctx)
    var d_cav_flat = reshape(csb_txt.d_v, [TXT, dim], ctx)
    var d_caq_h = rb_caq_dx.to_host(ctx)
    var d_cak_h = rb_cak_dx.to_host(ctx)
    var d_cav_h = d_cav_flat.to_host(ctx)
    # IMAGE k/v grads (norm_k_img frozen -> d_x-only)
    var d_imgk_rms_flat = reshape(csb_img.d_k, [IMG, dim], ctx)
    var lsv_img_k_pre = reshape(saved.img_k_pre[], [IMG, dim], ctx)
    var rb_imgk_dx = rms_norm_backward_dx(d_imgk_rms_flat, lsv_img_k_pre, w.ca_kn_img[], eps, ctx)
    var d_imgv_flat = reshape(csb_img.d_v, [IMG, dim], ctx)
    var d_imgk_h = rb_imgk_dx.to_host(ctx)
    var d_imgv_h = d_imgv_flat.to_host(ctx)

    # projection linears — FROZEN base: dx-only (d_w/d_b discarded by the stack).
    var dx_caq = linear_backward_dx(rb_caq_dx, w.base.ca_wq[], S, dim, dim, ctx)
    var dx_cak = linear_backward_dx(rb_cak_dx, w.base.ca_wk[], TXT, dim, dim, ctx)
    var dx_cav = linear_backward_dx(d_cav_flat, w.base.ca_wv[], TXT, dim, dim, ctx)
    var dx_imgk = linear_backward_dx(rb_imgk_dx, w.ca_wk_img[], IMG, dim, dim, ctx)
    var dx_imgv = linear_backward_dx(d_imgv_flat, w.ca_wv_img[], IMG, dim, dim, ctx)
    var d_ca_wq = List[Float32]()
    var d_ca_bq = List[Float32]()
    var d_ca_wk = List[Float32]()
    var d_ca_bk = List[Float32]()
    var d_ca_wv = List[Float32]()
    var d_ca_bv = List[Float32]()
    var d_ca_wk_img = List[Float32]()
    var d_ca_bk_img = List[Float32]()
    var d_ca_wv_img = List[Float32]()
    var d_ca_bv_img = List[Float32]()
    var ca_q_g = _lora_bwd_opt(lora.base.ca_q, d_caq_h, n3_h, S, dim, ctx)
    var ca_k_g = _lora_bwd_opt(lora.base.ca_k, d_cak_h, context_txt_h, TXT, dim, ctx)
    var ca_v_g = _lora_bwd_opt(lora.base.ca_v, d_cav_h, context_txt_h, TXT, dim, ctx)
    var img_k_g = _lora_bwd_opt(lora.img_k, d_imgk_h, context_img_h, IMG, dim, ctx)
    var img_v_g = _lora_bwd_opt(lora.img_v, d_imgv_h, context_img_h, IMG, dim, ctx)
    # n3 grad = base d_x(caq) + LoRA d_x(ca_q)  (bf16)
    var d_n3_in = add(dx_caq, _t16(ca_q_g.d_x.copy(), [S, dim], ctx), ctx)
    # TEXT context grad = base(cak+cav) + LoRA(ca_k + ca_v)
    var d_ctx_base = add(dx_cak, dx_cav, ctx)
    var d_ctx_lora = add(
        _t16(ca_k_g.d_x.copy(), [TXT, dim], ctx), _t16(ca_v_g.d_x.copy(), [TXT, dim], ctx), ctx
    )
    var d_context_t = TArc(add(d_ctx_base, d_ctx_lora, ctx))
    # IMAGE context grad = base(imgk+imgv) + LoRA(img_k + img_v)
    var d_imgctx_base = add(dx_imgk, dx_imgv, ctx)
    var d_imgctx_lora = add(
        _t16(img_k_g.d_x.copy(), [IMG, dim], ctx), _t16(img_v_g.d_x.copy(), [IMG, dim], ctx), ctx
    )
    var d_context_img = add(d_imgctx_base, d_imgctx_lora, ctx).to_host(ctx)

    # norm3 backward: layer_norm ran on bf16 x_sa -> feed bf16 x_sa
    var lnb_n3_dx = layer_norm_backward_dx(d_n3_in, x_sa_bf16_from(saved.x_sa[], ctx), w.base.n3_w[], eps, ctx)
    var d_n3_w = _zeros(dim)
    var d_n3_b = _zeros(dim)
    # d_x_sa = residual branch (F32) + norm3 branch (bf16 -> F32)
    var d_x_sa = TArc(add(cast_tensor(lnb_n3_dx, STDtype.F32, ctx), d_x_ca[], ctx))  # F32

    # ════════════════ Self-attention backward (LoRA q/k/v/o) ════════════════
    var gate_sa_t = Tensor.from_host(mv.gate_sa.copy(), [S, dim], STDtype.F32, ctx)
    var sa_out_rc = cast_tensor(
        linear(saved.sa_att[], w.base.sa_wo[], Optional[Tensor](_clone_t(w.base.sa_bo[], ctx)), ctx),
        STDtype.F32, ctx,
    )
    var gb_sa = wan_gate_residual_backward(d_x_sa[], sa_out_rc, gate_sa_t, ctx)
    var d_gate_sa = gb_sa.d_gate.to_host(ctx)
    var d_x_resid = TArc(gb_sa.d_x.clone(ctx))                              # F32
    var d_sa_out_bf16 = cast_tensor(gb_sa.d_y, STDtype.BF16, ctx)
    var d_sa_out_h = d_sa_out_bf16.to_host(ctx)
    # FROZEN base self_attn.o: dx-only.
    var dx_sao = linear_backward_dx(d_sa_out_bf16, w.base.sa_wo[], S, dim, dim, ctx)
    var d_sa_wo = List[Float32]()
    var d_sa_bo = List[Float32]()
    var sa_o_g = _lora_bwd_opt(lora.base.sa_o, d_sa_out_h, sa_att_h, S, dim, ctx)
    var d_sa_att = add(dx_sao, _t16(sa_o_g.d_x.copy(), [S, dim], ctx), ctx)
    var d_sa_att4 = reshape(d_sa_att, [1, S, H, DH], ctx)

    var ssb = _scail2_sa_bwd[FLASH, S, H, DH](
        saved.sa_q_rope[], saved.sa_k_rope[], saved.sa_v[], d_sa_att4, scale, ctx,
    )
    var ssb_dq_f32 = cast_tensor(ssb.d_q, STDtype.F32, ctx)
    var ssb_dk_f32 = cast_tensor(ssb.d_k, STDtype.F32, ctx)
    var d_q_rms = rope_backward(ssb_dq_f32, cos_e, sin_e, True, ctx)
    var d_k_rms = rope_backward(ssb_dk_f32, cos_e, sin_e, True, ctx)
    var d_q_rms_b = cast_tensor(d_q_rms, STDtype.BF16, ctx)
    var d_q_rms_flat = reshape(d_q_rms_b, [S, dim], ctx)
    var lsv_sa_q_pre = reshape(saved.sa_q_pre[], [S, dim], ctx)
    var rb_saq_dx = rms_norm_backward_dx(d_q_rms_flat, lsv_sa_q_pre, w.base.sa_qn[], eps, ctx)
    var d_k_rms_b = cast_tensor(d_k_rms, STDtype.BF16, ctx)
    var d_k_rms_flat = reshape(d_k_rms_b, [S, dim], ctx)
    var lsv_sa_k_pre = reshape(saved.sa_k_pre[], [S, dim], ctx)
    var rb_sak_dx = rms_norm_backward_dx(d_k_rms_flat, lsv_sa_k_pre, w.base.sa_kn[], eps, ctx)
    var d_sav_flat = reshape(ssb.d_v, [S, dim], ctx)
    var d_saq_h = rb_saq_dx.to_host(ctx)
    var d_sak_h = rb_sak_dx.to_host(ctx)
    var d_sav_h = d_sav_flat.to_host(ctx)

    # FROZEN base self_attn.q/k/v: dx-only.
    var dx_saq = linear_backward_dx(rb_saq_dx, w.base.sa_wq[], S, dim, dim, ctx)
    var dx_sak = linear_backward_dx(rb_sak_dx, w.base.sa_wk[], S, dim, dim, ctx)
    var dx_sav = linear_backward_dx(d_sav_flat, w.base.sa_wv[], S, dim, dim, ctx)
    var d_sa_wq = List[Float32]()
    var d_sa_bq = List[Float32]()
    var d_sa_wk = List[Float32]()
    var d_sa_bk = List[Float32]()
    var d_sa_wv = List[Float32]()
    var d_sa_bv = List[Float32]()
    var sa_q_g = _lora_bwd_opt(lora.base.sa_q, d_saq_h, sa_in_h, S, dim, ctx)
    var sa_k_g = _lora_bwd_opt(lora.base.sa_k, d_sak_h, sa_in_h, S, dim, ctx)
    var sa_v_g = _lora_bwd_opt(lora.base.sa_v, d_sav_h, sa_in_h, S, dim, ctx)
    var d_sa_in_base = add(add(dx_saq, dx_sak, ctx), dx_sav, ctx)
    var d_sa_in_lora = add(
        add(_t16(sa_q_g.d_x.copy(), [S, dim], ctx), _t16(sa_k_g.d_x.copy(), [S, dim], ctx), ctx),
        _t16(sa_v_g.d_x.copy(), [S, dim], ctx), ctx,
    )
    var d_sa_in = add(d_sa_in_base, d_sa_in_lora, ctx)                       # bf16
    # F32 modulate backward; norm1 ran on bf16 block input -> bf16 layer_norm_backward
    var d_sa_in_f32 = cast_tensor(d_sa_in, STDtype.F32, ctx)
    var scale_sa_t = Tensor.from_host(mv.scale_sa.copy(), [S, dim], STDtype.F32, ctx)
    var mb_sa = wan_modulate_backward(d_sa_in_f32, saved.sa_ln[], scale_sa_t, ctx)
    var d_scale_sa = mb_sa.d_scale.to_host(ctx)
    var d_shift_sa = mb_sa.d_shift.to_host(ctx)
    var d_ln_sa_bf16 = cast_tensor(mb_sa.d_ln, STDtype.BF16, ctx)
    var lnb_sa = layer_norm_backward_dx(d_ln_sa_bf16, saved.x[], ones_bf16, eps, ctx)  # bf16
    var d_x = add(cast_tensor(lnb_sa, STDtype.F32, ctx), d_x_resid[], ctx)  # F32
    var d_x_h = d_x.to_host(ctx)
    var d_context_h = d_context_t[].to_host(ctx)

    var d_sa_qn = _zeros(dim)
    var d_sa_kn = _zeros(dim)
    var d_ca_qn = _zeros(dim)
    var d_ca_kn = _zeros(dim)
    var d_ca_kn_img = _zeros(dim)

    var base_grads = WanBlockGrads(
        d_x_h^, d_context_h^,
        d_sa_wq^, d_sa_wk^, d_sa_wv^, d_sa_wo^,
        d_sa_bq^, d_sa_bk^, d_sa_bv^, d_sa_bo^,
        d_sa_qn^, d_sa_kn^,
        d_ca_wq^, d_ca_wk^, d_ca_wv^, d_ca_wo^,
        d_ca_bq^, d_ca_bk^, d_ca_bv^, d_ca_bo^,
        d_ca_qn^, d_ca_kn^,
        d_n3_w^, d_n3_b^,
        d_ffn0_w^, d_ffn0_b^, d_ffn2_w^, d_ffn2_b^,
        d_shift_sa^, d_scale_sa^, d_gate_sa^,
        d_shift_ffn^, d_scale_ffn^, d_gate_ffn^,
    )
    var lora_grads = WanBlockLoraGrads(
        base_grads^,
        sa_q_g.d_a.copy(), sa_q_g.d_b.copy(),
        sa_k_g.d_a.copy(), sa_k_g.d_b.copy(),
        sa_v_g.d_a.copy(), sa_v_g.d_b.copy(),
        sa_o_g.d_a.copy(), sa_o_g.d_b.copy(),
        ca_q_g.d_a.copy(), ca_q_g.d_b.copy(),
        ca_k_g.d_a.copy(), ca_k_g.d_b.copy(),
        ca_v_g.d_a.copy(), ca_v_g.d_b.copy(),
        ca_o_g.d_a.copy(), ca_o_g.d_b.copy(),
        ffn0_g.d_a.copy(), ffn0_g.d_b.copy(),
        ffn2_g.d_a.copy(), ffn2_g.d_b.copy(),
    )
    return WanI2VBlockLoraGrads(
        lora_grads^,
        d_context_img^,
        d_ca_wk_img^, d_ca_bk_img^, d_ca_wv_img^, d_ca_bv_img^, d_ca_kn_img^,
        img_k_g.d_a.copy(), img_k_g.d_b.copy(),
        img_v_g.d_a.copy(), img_v_g.d_b.copy(),
    )


# norm3 ran on the bf16 re-cast of the F32 x_sa; produce that bf16 view.
def x_sa_bf16_from(x_sa_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(x_sa_f32, STDtype.BF16, ctx)
