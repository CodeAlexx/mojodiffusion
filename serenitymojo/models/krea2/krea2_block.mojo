# models/krea2/krea2_block.mojo — Krea-2-Raw SingleStreamBlock TRAINING unit.
#
# Forward (saving activations) + hand-chained backward for ONE krea2
# SingleStreamBlock, with LoRA on the 8 block nn.Linears. This MIRRORS the
# verified inference forward in models/dit/krea2_dit.mojo
# (`krea2_single_stream_block`, line 812) op-for-op; it does NOT re-derive the
# math. It is the Klein single_block.mojo save-acts + hand-chain template
# adapted to krea2's three differences from a FLUX single block:
#   (1) GQA  — k/v have KVHEADS<HEADS heads, repeat_kv'd before SDPA (and
#       sum-reduced in backward via repeat_kv_backward).
#   (2) a SIGMOID GATE on the attention output: a = wo(sdpa(...) * sigmoid(gate)),
#       where gate = gate_proj(xm). New backward path: mul + sigmoid_backward.
#   (3) RMSNorm weight = scale + 1.0 (the F32 reparam, mmdit.py:172-177). We pass
#       `scale + 1` as the rms weight; the prenorm/postnorm/qknorm scales are
#       FROZEN (not LoRA targets), so their grads aren't needed.
#
# AdaLN-Zero double branch (mmdit.py SingleStreamBlock.forward, 328-337):
#   prescale,preshift,pregate,postscale,postshift,postgate = mod(vec)   # raw chunks
#   x1 = x + pregate  * attn ((1+prescale )*prenorm (x)  + preshift)
#   x2 = x1 + postgate * mlp ((1+postscale)*postnorm(x1) + postshift)
#   attn(y) = wo( sdpa(QKNorm+RoPE+GQA on wq/wk/wv(y)) * sigmoid(gate(y)) )
#   mlp(y)  = down( silu(gate(y)) * up(y) )
#
# LoRA on all 8 Linears (krea2.py:148 target_lora_modules=["SingleStreamDiT"],
# resolved to the per-block nn.Linears by lora_special.py): wq wk wv gate wo
# (attention) + mlp_gate mlp_up mlp_down. LoRA math == the Klein lora_block
# helper: y' = linear(x,W) + scale*((x@Aᵀ)@Bᵀ), A=[rank,in], B=[out,rank].
# (We REUSE LoraAdapterDevice + the klein_lora unfused fwd/bwd — they are
# model-agnostic LoRA-on-one-Linear primitives.)
#
# Historical parity gates also exercise F32 oracle tensors, but product training
# preserves BF16/F16/FP8 storage boundaries. This block may use F32 internally for
# reductions, score math, and optimizer-bound gradients; it must return/store
# model activations and LoRA params in their boundary dtype.
#
# Mojo 1.0.0b1: `def` only; Tensor move-only (Movable structs, no Tensor in a
# collection); no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

comptime TArc = ArcPointer[Tensor]

# autograd_v2 slab-block attn sdpa switch (krea2_block_graph.mojo slab recorder):
# False = MATH sdpa_nomask (deterministic, the BIT GATE path — bit-exact vs the
# hand-chain). True = cuDNN FLASH (O(L), the PRODUCTION/TRAINER path; flash dQ is
# nondeterministic → value-tolerance grads, NOT bit). DEFAULT False so the block bit
# gate proves the slab block math-exact; the trainer flips it for the flash fit/speed
# path (the math O(L²) scores make the math attn 13.4GB — doesn't fit with the 12GB
# fp8 base on 24GB; flash O(L) ~2.2GB fits). Only the slab recorder reads it; the
# hand-chain is untouched.
comptime KREA2_SLAB_FLASH = False

# autograd_v2 slab-conductor path: True = the SEGMENTED (2-segment activation
# checkpoint) per-block backward (per-segment slab ~6.65GB, MEASURED to fit ~22GB/24GB
# — the safe path). False = the WHOLE-BLOCK slab recorder (no segmentation; MEASURED
# slab.peak_bytes = 12.23GB at L=4864 flash → 12GB fp8 + 12.2GB > 24GB, EXPECTED to
# OOM at setup — kept selectable so the trainer run can confirm/refute the fit
# directly, per the lead's "the trainer run IS the fit test"). DEFAULT True (the
# fitting path); flip to False to test the whole-block path.
comptime KREA2_SLAB_SEGMENTED = True

# ── forward ops ──────────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu, sigmoid
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_padmask_bf16, sdpa_flash_backward_padmask_bf16,
)
from serenitymojo.ops.gqa_backward import repeat_kv_f32, repeat_kv_backward
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, slice, concat, add, mul, mul_scalar, zeros_device,
)
from serenitymojo.ops.cast import cast_tensor

# ── backward arms (all pre-built + gated elsewhere) ──────────────────────────
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, LinearGrads,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
)
from serenitymojo.ops.activation_backward import sigmoid_backward_from_output
from serenitymojo.models.dit.krea2_dit import (
    krea2_rmsnorm,
    krea2_rmsnorm_backward_dx,
)

# ── LoRA primitive (model-agnostic LoRA-on-one-Linear) ───────────────────────
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident_unfused,
    klein_lora_bwd_device_resident_unfused,
    klein_lora_bwd_device_resident_tensors_unfused,
    KleinLoraDeviceGrads,
    KleinLoraDeviceGradTensors,
)
from serenitymojo.training.dora_substitution_device import (
    DoRAAdapterDevice, DoRADeviceGrads,
)
from serenitymojo.training.oft_onetrainer_device import OFTOTDeviceGrads
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    Krea2BlockDirectDoRA, Krea2BlockDirectOFT, Krea2DirectOFTDeviceSlot,
    krea2_direct_dora_projection_forward_resident,
    krea2_direct_dora_projection_backward_resident,
    krea2_direct_oft_projection_forward_resident,
    krea2_direct_oft_projection_backward_resident,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _no_bias() -> Optional[Tensor]:
    return Optional[Tensor](None)


def _add_scale_one(scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """RMSNorm weight reparam: weight = scale + 1.0 (mmdit.py:175). scale is the
    raw [D] parameter in the checkpoint/storage dtype; we materialize
    (scale+1) as the rms_norm weight without forcing an F32 storage boundary.
    The [1] one broadcasts against the [D]/[Dh] scale."""
    var o = List[Float32]()
    o.append(Float32(1.0))
    var one = Tensor.from_host(o^, [1], scale.dtype(), ctx)
    return add(scale, one, ctx)


def _lora_fwd(
    x: Tensor, lo: Optional[LoraAdapterDevice], M: Int, ctx: DeviceContext
) raises -> Optional[Tensor]:
    """LoRA delta scale*((x@Aᵀ)@Bᵀ) for one Linear, or None if no adapter."""
    if lo:
        return Optional[Tensor](
            klein_lora_fwd_device_resident_unfused(x, lo.value(), M, ctx)
        )
    return Optional[Tensor](None)


def _linear_lora(
    x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice], M: Int, ctx: DeviceContext
) raises -> Tensor:
    """y = linear(x,W) [no bias] + LoRA delta (if present)."""
    var nb = _no_bias()
    var y = linear(x, w, nb^, ctx)
    var d = _lora_fwd(x, lo, M, ctx)
    if d:
        y = add(y, d.value(), ctx)
    return y^


# ── trainable weights (FROZEN base + per-Linear LoRA adapters) ───────────────
# Base projection matrices are torch Linear weight layout [out, in]; rmsnorm/mod
# params are the RAW [D]/[Dh]/[6D] vectors (we add +1 to the rms scales inside).
struct Krea2BlockWeights(Copyable, Movable):
    var wq: TArc          # [HEADS*HEADDIM, features]
    var wk: TArc          # [KVHEADS*HEADDIM, features]
    var wv: TArc          # [KVHEADS*HEADDIM, features]
    var gate_w: TArc      # [features, features]
    var wo: TArc          # [features, features]
    var mlp_gate_w: TArc  # [mlpdim, features]
    var mlp_up_w: TArc    # [mlpdim, features]
    var mlp_down_w: TArc  # [features, mlpdim]
    var qnorm_scale: TArc # [HEADDIM] raw
    var knorm_scale: TArc # [HEADDIM] raw
    var prenorm_scale: TArc   # [features] raw
    var postnorm_scale: TArc  # [features] raw
    var mod_lin: TArc     # [6*features] (DoubleSharedModulation.lin)

    def __init__(
        out self,
        var wq: TArc, var wk: TArc, var wv: TArc, var gate_w: TArc, var wo: TArc,
        var mlp_gate_w: TArc, var mlp_up_w: TArc, var mlp_down_w: TArc,
        var qnorm_scale: TArc, var knorm_scale: TArc,
        var prenorm_scale: TArc, var postnorm_scale: TArc, var mod_lin: TArc,
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^


# The 8 LoRA adapters (each Optional so the block reduces to the frozen base
# when all are absent). Order matches the 8 target Linears.
struct Krea2BlockLora(Copyable, Movable):
    var wq: Optional[LoraAdapterDevice]
    var wk: Optional[LoraAdapterDevice]
    var wv: Optional[LoraAdapterDevice]
    var gate_w: Optional[LoraAdapterDevice]
    var wo: Optional[LoraAdapterDevice]
    var mlp_gate_w: Optional[LoraAdapterDevice]
    var mlp_up_w: Optional[LoraAdapterDevice]
    var mlp_down_w: Optional[LoraAdapterDevice]

    def __init__(
        out self,
        var wq: Optional[LoraAdapterDevice], var wk: Optional[LoraAdapterDevice],
        var wv: Optional[LoraAdapterDevice], var gate_w: Optional[LoraAdapterDevice],
        var wo: Optional[LoraAdapterDevice],
        var mlp_gate_w: Optional[LoraAdapterDevice],
        var mlp_up_w: Optional[LoraAdapterDevice],
        var mlp_down_w: Optional[LoraAdapterDevice],
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


# ── saved activations (device-resident) ──────────────────────────────────────
struct Krea2BlockSaved(Copyable, Movable):
    var x: TArc          # [1,L,features] block input
    var xm: TArc         # [1,L,features] modulate(prenorm(x))  — attn-proj input
    var q_pre: TArc      # [1,L,HEADS,HEADDIM]   wq(xm) reshaped (pre-QKNorm)
    var k_pre: TArc      # [1,L,KVHEADS,HEADDIM] wk(xm)
    var v: TArc          # [1,L,KVHEADS,HEADDIM] wv(xm) (untouched by QKNorm)
    var q_rope: TArc     # [1,L,HEADS,HEADDIM]   rope(qnorm(q_pre))
    var k_rope: TArc     # [1,L,KVHEADS,HEADDIM] rope(knorm(k_pre))
    var k_full: TArc     # [1,L,HEADS,HEADDIM]   repeat_kv(k_rope)
    var v_full: TArc     # [1,L,HEADS,HEADDIM]   repeat_kv(v)
    var attn_flat: TArc  # [1,L,features]        sdpa(...) merged
    var gate_pre: TArc   # [1,L,features]        gate_w(xm) (pre-sigmoid)
    var sg: TArc         # [1,L,features]        sigmoid(gate_pre)
    var gated: TArc      # [1,L,features]        attn_flat * sg  (wo input)
    var a: TArc          # [1,L,features]        wo(gated) — the attn branch output (gate_residual y)
    var x1: TArc         # [1,L,features]        x + pregate*attn
    var xm2: TArc        # [1,L,features]        modulate(postnorm(x1)) — mlp input
    var mlp_gate: TArc   # [1,L,mlpdim]          mlp_gate_w(xm2)
    var mlp_up: TArc     # [1,L,mlpdim]          mlp_up_w(xm2)
    var sw: TArc         # [1,L,mlpdim]          swiglu(mlp_gate, mlp_up) — down input
    var m: TArc          # [1,L,features]        mlp_down(sw) — mlp branch output (gate_residual y)
    # the rms-normed (pre-modulate) activations needed for modulate_backward
    var xn: TArc         # [1,L,features] prenorm(x)
    var xn2: TArc        # [1,L,features] postnorm(x1)
    # ── flash-padmask saved set (Phase: length-bucket flash training) ──────────
    # Present ONLY when the masked/padded SDPA ran the cuDNN flash-padmask path
    # (real_len < L). The flash backward consumes the bf16 q/k/v/o + F32 stats
    # WITHOUT recompute (= klein KLEIN_SDPA_FLASH tape pattern). On the no-pad
    # (full-attn) path these are None and the backward uses sdpa_backward
    # (BIT-IDENTICAL to the pre-flash block — the F32 parity gate guard).
    var flash_q: Optional[TArc]   # [1,L,HEADS,Dh] bf16
    var flash_k: Optional[TArc]   # [1,L,HEADS,Dh] bf16 (post-GQA k_full)
    var flash_v: Optional[TArc]   # [1,L,HEADS,Dh] bf16 (post-GQA v_full)
    var flash_o: Optional[TArc]   # [1,L,HEADS,Dh] bf16 padded SDPA output
    var flash_stats: Optional[TArc]  # [1,HEADS,L,1] F32 softmax LSE

    def __init__(
        out self,
        var x: TArc, var xm: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rope: TArc, var k_rope: TArc, var k_full: TArc, var v_full: TArc,
        var attn_flat: TArc, var gate_pre: TArc, var sg: TArc, var gated: TArc,
        var a: TArc, var x1: TArc, var xm2: TArc,
        var mlp_gate: TArc, var mlp_up: TArc, var sw: TArc, var m: TArc,
        var xn: TArc, var xn2: TArc,
        var flash_q: Optional[TArc] = Optional[TArc](None),
        var flash_k: Optional[TArc] = Optional[TArc](None),
        var flash_v: Optional[TArc] = Optional[TArc](None),
        var flash_o: Optional[TArc] = Optional[TArc](None),
        var flash_stats: Optional[TArc] = Optional[TArc](None),
    ):
        self.x = x^
        self.xm = xm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.k_full = k_full^
        self.v_full = v_full^
        self.attn_flat = attn_flat^
        self.gate_pre = gate_pre^
        self.sg = sg^
        self.gated = gated^
        self.a = a^
        self.x1 = x1^
        self.xm2 = xm2^
        self.mlp_gate = mlp_gate^
        self.mlp_up = mlp_up^
        self.sw = sw^
        self.m = m^
        self.xn = xn^
        self.xn2 = xn2^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^


struct Krea2BlockForward(Movable):
    var out: TArc            # [1,L,features] device-resident block output
    var saved: Krea2BlockSaved

    def __init__(out self, var out: TArc, var saved: Krea2BlockSaved):
        self.out = out^
        self.saved = saved^


# ── per-Linear LoRA grad pair (host F32; None when adapter absent) ────────────
struct Krea2LoraGrad(Copyable, Movable):
    var d_a: Optional[List[Float32]]
    var d_b: Optional[List[Float32]]

    def __init__(
        out self, var d_a: Optional[List[Float32]], var d_b: Optional[List[Float32]]
    ):
        self.d_a = d_a^
        self.d_b = d_b^


# ── backward result ──────────────────────────────────────────────────────────
struct Krea2BlockGrads(Movable):
    var d_x: TArc                 # input grad [1,L,features]
    # the 8 LoRA dA/dB (None when the adapter is absent)
    var wq: Krea2LoraGrad
    var wk: Krea2LoraGrad
    var wv: Krea2LoraGrad
    var gate_w: Krea2LoraGrad
    var wo: Krea2LoraGrad
    var mlp_gate_w: Krea2LoraGrad
    var mlp_up_w: Krea2LoraGrad
    var mlp_down_w: Krea2LoraGrad

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2LoraGrad, var wk: Krea2LoraGrad, var wv: Krea2LoraGrad,
        var gate_w: Krea2LoraGrad, var wo: Krea2LoraGrad,
        var mlp_gate_w: Krea2LoraGrad, var mlp_up_w: Krea2LoraGrad,
        var mlp_down_w: Krea2LoraGrad,
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


# ── DEVICE-resident per-Linear LoRA grad pair (TArc; None when adapter absent) ─
# Sibling of Krea2LoraGrad. The d_A/d_B stay on device (no per-adapter to_host),
# and the streamed stack either copies a block's 8 pairs to host under the
# per-block streaming fence or copies them D2D into shared AdamW state in the
# krea2devicegrad path. The HOST Krea2LoraGrad above is the bit-gate oracle and
# is left untouched.
struct Krea2LoraGradT(Copyable, Movable):
    var d_a: Optional[TArc]
    var d_b: Optional[TArc]

    def __init__(
        out self, var d_a: Optional[TArc], var d_b: Optional[TArc]
    ):
        self.d_a = d_a^
        self.d_b = d_b^


struct Krea2BlockGradsT(Movable):
    var d_x: TArc                 # input grad [1,L,features]
    var wq: Krea2LoraGradT
    var wk: Krea2LoraGradT
    var wv: Krea2LoraGradT
    var gate_w: Krea2LoraGradT
    var wo: Krea2LoraGradT
    var mlp_gate_w: Krea2LoraGradT
    var mlp_up_w: Krea2LoraGradT
    var mlp_down_w: Krea2LoraGradT

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2LoraGradT, var wk: Krea2LoraGradT, var wv: Krea2LoraGradT,
        var gate_w: Krea2LoraGradT, var wo: Krea2LoraGradT,
        var mlp_gate_w: Krea2LoraGradT, var mlp_up_w: Krea2LoraGradT,
        var mlp_down_w: Krea2LoraGradT,
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


# ── modulation: out = vec + lin; chunk 6 along last dim → 6 raw [features] ────
# (mmdit.py DoubleSharedModulation.forward). vec [1,6F], lin [6F]; b==1 so each
# chunk is [features]; reshape to a clean [features] param for modulate/gate.
def _mod6(
    vec: Tensor, mod_lin: Tensor, features: Int, ctx: DeviceContext
) raises -> List[TArc]:
    var s = add(vec, mod_lin, ctx)          # [1, 6F] (lin [6F] broadcasts)
    var out = List[TArc]()
    for i in range(6):
        var c = slice(s, 1, i * features, features, ctx)   # [1, features]
        out.append(TArc(reshape_owned(c^, [features])))    # [features]
    return out^


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD (saves activations) — mirrors krea2_dit.mojo:812-859 + LoRA on 8 Linears
# ══════════════════════════════════════════════════════════════════════════════
def krea2_single_stream_block_lora[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,            # [1, L, features] F32
    vec: Tensor,          # [1, 6*features] F32  (timestep modulation vec)
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor,   # [L, HEADDIM/2] per-token RoPE table
    cos_q: Tensor, sin_q: Tensor,   # [L*HEADS, HEADDIM/2]   tiled for BSHD q
    cos_k: Tensor, sin_k: Tensor,   # [L*KVHEADS, HEADDIM/2] tiled for BSHD k
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # length-bucket pad: the VALID
        # contiguous-prefix length of the [0:real_len] real tokens; [real_len:L] is
        # text-pad. None (or real_len == L) = full attention via sdpa_nomask
        # (BIT-IDENTICAL to the pre-mask block — the F32 parity gate guard). Present
        # & < L = cuDNN flash-padmask SDPA: cuDNN masks the [real_len:L] tail rows
        # internally (NO materialized [1,H,L,L] mask, NO materialized scores). The
        # token order MUST be [valid(0:real_len) | pad(real_len:L)] — see the
        # trainer's [TXT_real | IMG | TXT_pad] reorder. real_len threads to bwd too.
) raises -> Krea2BlockForward:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L                              # rows for LoRA (b==1, [L, features])
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    # mod(vec) → 6 raw chunks.
    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    # ── ATTENTION branch ─────────────────────────────────────────────────────
    # xm = (1+prescale)*prenorm(x) + preshift
    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)
    var xm = modulate(xn, prescale[], preshift[], ctx)          # [1,L,features]

    # projections (+ LoRA). xm is [1,L,features]; linear treats leading dims as rows.
    var q = _linear_lora(xm, w.wq[], lora.wq, M, ctx)           # [1,L,HEADS*HEADDIM]
    var k = _linear_lora(xm, w.wk[], lora.wk, M, ctx)           # [1,L,KVHEADS*HEADDIM]
    var v_lin = _linear_lora(xm, w.wv[], lora.wv, M, ctx)       # [1,L,KVHEADS*HEADDIM]
    var gate_pre = _linear_lora(xm, w.gate_w[], lora.gate_w, M, ctx)  # [1,L,features]

    # reshape BSHD.
    var q_pre = reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])

    # QKNorm over HEADDIM (weight = scale+1); v untouched.
    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)

    # RoPE on q,k (per-head tiled tables).
    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)

    # GQA: repeat_kv to HEADS.
    var k_full = repeat_kv_f32(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = repeat_kv_f32(v, L, KVHEADS, n_rep, HEADDIM, ctx)

    # SDPA. No pad (default, or real_len == L) = full attention (sdpa_nomask) — the
    # per-block gate path, BIT-IDENTICAL to the pre-flash block. Length-bucket pad
    # (real_len present & < L) = cuDNN flash-padmask SDPA: cuDNN masks the
    # [real_len:L] tail rows internally (the token order is [valid | pad]); NO
    # materialized [1,H,L,L] mask (the 4.5GB resident the old sdpa_chunked path
    # needed), NO materialized [L,L] scores. The flash bf16 q/k/v/o + F32 stats go
    # to the tape for the flash backward (no recompute, no re-cast).
    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = real_len and real_len.value() < L
    if use_flash:
        var rl = real_len.value()
        var ff = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, rl, scale, ctx
        )
        # ff.att is BF16 [1,L,HEADS,Dh] (pad-tail rows are masked-out garbage the
        # downstream gate zeroes via the pad d_out). Save the BF16 set for bwd.
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = reshape_owned(att^, [1, L, features])

    # sigmoid gate + product, then wo.
    var sg = sigmoid(gate_pre, ctx)                             # [1,L,features]
    var gated = mul(attn_flat, sg, ctx)                        # [1,L,features]
    var a = _linear_lora(gated, w.wo[], lora.wo, M, ctx)       # [1,L,features]

    # x1 = x + pregate * a
    var x1 = residual_gate(x_t[], pregate[], a, ctx)

    # ── MLP branch ───────────────────────────────────────────────────────────
    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2 = modulate(xn2, postscale[], postshift[], ctx)     # [1,L,features]

    var mg = _linear_lora(xm2, w.mlp_gate_w[], lora.mlp_gate_w, M, ctx)  # [1,L,mlpdim]
    var mu = _linear_lora(xm2, w.mlp_up_w[], lora.mlp_up_w, M, ctx)      # [1,L,mlpdim]
    var sw = swiglu(mg, mu, ctx)                                # silu(mg)*mu [1,L,mlpdim]
    var m = _linear_lora(sw, w.mlp_down_w[], lora.mlp_down_w, M, ctx)    # [1,L,features]

    var x2 = residual_gate(x1, postgate[], m, ctx)             # x1 + postgate*m

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


def krea2_single_stream_block_dora[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,
    vec: Tensor,
    w: Krea2BlockWeights, dora: Krea2BlockDirectDoRA,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockForward:
    """Krea2 SingleStreamBlock forward with direct DoRA W_eff projection hooks."""
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)
    var xm = modulate(xn, prescale[], preshift[], ctx)

    var q = krea2_block_direct_dora_projection_forward(xm, w.wq[], dora.wq, M, ctx)
    var k = krea2_block_direct_dora_projection_forward(xm, w.wk[], dora.wk, M, ctx)
    var v_lin = krea2_block_direct_dora_projection_forward(xm, w.wv[], dora.wv, M, ctx)
    var gate_pre = krea2_block_direct_dora_projection_forward(xm, w.gate_w[], dora.gate_w, M, ctx)

    var q_pre = reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])

    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)

    var k_full = repeat_kv_f32(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = repeat_kv_f32(v, L, KVHEADS, n_rep, HEADDIM, ctx)

    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = real_len and real_len.value() < L
    if use_flash:
        var rl = real_len.value()
        var ff = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, rl, scale, ctx
        )
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = reshape_owned(att^, [1, L, features])

    var sg = sigmoid(gate_pre, ctx)
    var gated = mul(attn_flat, sg, ctx)
    var a = krea2_block_direct_dora_projection_forward(gated, w.wo[], dora.wo, M, ctx)

    var x1 = residual_gate(x_t[], pregate[], a, ctx)

    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2 = modulate(xn2, postscale[], postshift[], ctx)

    var mg = krea2_block_direct_dora_projection_forward(xm2, w.mlp_gate_w[], dora.mlp_gate_w, M, ctx)
    var mu = krea2_block_direct_dora_projection_forward(xm2, w.mlp_up_w[], dora.mlp_up_w, M, ctx)
    var sw = swiglu(mg, mu, ctx)
    var m = krea2_block_direct_dora_projection_forward(sw, w.mlp_down_w[], dora.mlp_down_w, M, ctx)

    var x2 = residual_gate(x1, postgate[], m, ctx)

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


def krea2_single_stream_block_oft[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,
    vec: Tensor,
    w: Krea2BlockWeights, oft: Krea2BlockDirectOFT,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockForward:
    """Krea2 SingleStreamBlock forward with direct OFT projection hooks."""
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)
    var xm = modulate(xn, prescale[], preshift[], ctx)

    var q = krea2_block_direct_oft_projection_forward(xm, w.wq[], oft.wq, M, ctx)
    var k = krea2_block_direct_oft_projection_forward(xm, w.wk[], oft.wk, M, ctx)
    var v_lin = krea2_block_direct_oft_projection_forward(xm, w.wv[], oft.wv, M, ctx)
    var gate_pre = krea2_block_direct_oft_projection_forward(xm, w.gate_w[], oft.gate_w, M, ctx)

    var q_pre = reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])

    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)

    var k_full = repeat_kv_f32(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = repeat_kv_f32(v, L, KVHEADS, n_rep, HEADDIM, ctx)

    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = real_len and real_len.value() < L
    if use_flash:
        var rl = real_len.value()
        var ff = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, rl, scale, ctx
        )
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = reshape_owned(att^, [1, L, features])

    var sg = sigmoid(gate_pre, ctx)
    var gated = mul(attn_flat, sg, ctx)
    var a = krea2_block_direct_oft_projection_forward(gated, w.wo[], oft.wo, M, ctx)

    var x1 = residual_gate(x_t[], pregate[], a, ctx)

    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2 = modulate(xn2, postscale[], postshift[], ctx)

    var mg = krea2_block_direct_oft_projection_forward(xm2, w.mlp_gate_w[], oft.mlp_gate_w, M, ctx)
    var mu = krea2_block_direct_oft_projection_forward(xm2, w.mlp_up_w[], oft.mlp_up_w, M, ctx)
    var sw = swiglu(mg, mu, ctx)
    var m = krea2_block_direct_oft_projection_forward(sw, w.mlp_down_w[], oft.mlp_down_w, M, ctx)

    var x2 = residual_gate(x1, postgate[], m, ctx)

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD (hand-chained) — exact reverse of the forward graph above
# ══════════════════════════════════════════════════════════════════════════════
# d_x for a LoRA Linear + the adapter's dA/dB, in one Movable struct (Mojo Tuple
# element transfer-out is fragile, so we use an explicit struct like the rest of
# the codebase).
struct _LinBwd(Movable):
    var d_x: Tensor
    var lora: Krea2LoraGrad

    def __init__(out self, var d_x: Tensor, var lora: Krea2LoraGrad):
        self.d_x = d_x^
        self.lora = lora^


def _linear_bwd_dx(
    d_y: Tensor, x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _LinBwd:
    """d_x for a LoRA Linear = base linear_backward_dx + LoRA branch d_x.
    Base W is FROZEN (no d_w needed). LoRA dA/dB returned in the pair (None when
    the adapter is absent)."""
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    if lo:
        var g = klein_lora_bwd_device_resident_unfused(d_y, x, lo.value(), M, ctx)
        d_x = add(d_x, g.d_x, ctx)
        var pair = Krea2LoraGrad(
            Optional[List[Float32]](g.d_a.copy()),
            Optional[List[Float32]](g.d_b.copy()),
        )
        return _LinBwd(d_x^, pair^)
    return _LinBwd(d_x^, Krea2LoraGrad(None, None))


# ── DEVICE-grad sibling of _LinBwd / _linear_bwd_dx ──────────────────────────
# Identical GEMM math, but the LoRA dA/dB stay on DEVICE (TArc) — no internal
# to_host. klein_lora_bwd_device_resident_tensors_unfused is the SAME unfused
# chain as klein_lora_bwd_device_resident_unfused minus the _to_host_pair_f32
# fence, so the device-grad path is bit-identical to the host path (the trainer
# proves this by the bit-identical loss gate).
struct _LinBwdT(Movable):
    var d_x: Tensor
    var lora: Krea2LoraGradT

    def __init__(out self, var d_x: Tensor, var lora: Krea2LoraGradT):
        self.d_x = d_x^
        self.lora = lora^


def _linear_bwd_dx_dev(
    d_y: Tensor, x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _LinBwdT:
    """Device-grad d_x for a LoRA Linear. Base W FROZEN (no d_w). LoRA dA/dB stay
    on device (no per-adapter to_host) — the SAME GEMM math as _linear_bwd_dx."""
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    if lo:
        var g = klein_lora_bwd_device_resident_tensors_unfused(d_y, x, lo.value(), M, ctx)
        d_x = add(d_x, g.d_x[], ctx)
        var pair = Krea2LoraGradT(
            Optional[TArc](g.d_a.copy()),
            Optional[TArc](g.d_b.copy()),
        )
        return _LinBwdT(d_x^, pair^)
    return _LinBwdT(d_x^, Krea2LoraGradT(None, None))


struct Krea2DirectDoRAGradT(Copyable, Movable):
    var d_a: Optional[TArc]
    var d_b: Optional[TArc]
    var d_m: Optional[TArc]

    def __init__(
        out self, var d_a: Optional[TArc], var d_b: Optional[TArc],
        var d_m: Optional[TArc],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^


struct Krea2DirectOFTGradT(Copyable, Movable):
    var d_vec: Optional[TArc]

    def __init__(out self, var d_vec: Optional[TArc]):
        self.d_vec = d_vec^


struct Krea2BlockDirectDoRAGradsT(Movable):
    var d_x: TArc
    var wq: Krea2DirectDoRAGradT
    var wk: Krea2DirectDoRAGradT
    var wv: Krea2DirectDoRAGradT
    var gate_w: Krea2DirectDoRAGradT
    var wo: Krea2DirectDoRAGradT
    var mlp_gate_w: Krea2DirectDoRAGradT
    var mlp_up_w: Krea2DirectDoRAGradT
    var mlp_down_w: Krea2DirectDoRAGradT

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2DirectDoRAGradT, var wk: Krea2DirectDoRAGradT,
        var wv: Krea2DirectDoRAGradT, var gate_w: Krea2DirectDoRAGradT,
        var wo: Krea2DirectDoRAGradT,
        var mlp_gate_w: Krea2DirectDoRAGradT,
        var mlp_up_w: Krea2DirectDoRAGradT,
        var mlp_down_w: Krea2DirectDoRAGradT,
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


struct Krea2BlockDirectOFTGradsT(Movable):
    var d_x: TArc
    var wq: Krea2DirectOFTGradT
    var wk: Krea2DirectOFTGradT
    var wv: Krea2DirectOFTGradT
    var gate_w: Krea2DirectOFTGradT
    var wo: Krea2DirectOFTGradT
    var mlp_gate_w: Krea2DirectOFTGradT
    var mlp_up_w: Krea2DirectOFTGradT
    var mlp_down_w: Krea2DirectOFTGradT

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2DirectOFTGradT, var wk: Krea2DirectOFTGradT,
        var wv: Krea2DirectOFTGradT, var gate_w: Krea2DirectOFTGradT,
        var wo: Krea2DirectOFTGradT,
        var mlp_gate_w: Krea2DirectOFTGradT,
        var mlp_up_w: Krea2DirectOFTGradT,
        var mlp_down_w: Krea2DirectOFTGradT,
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


struct _DirectDoRALinBwdT(Movable):
    var d_x: Tensor
    var dora: Krea2DirectDoRAGradT

    def __init__(out self, var d_x: Tensor, var dora: Krea2DirectDoRAGradT):
        self.d_x = d_x^
        self.dora = dora^


struct _DirectOFTLinBwdT(Movable):
    var d_x: Tensor
    var oft: Krea2DirectOFTGradT

    def __init__(out self, var d_x: Tensor, var oft: Krea2DirectOFTGradT):
        self.d_x = d_x^
        self.oft = oft^


def krea2_block_direct_dora_projection_forward(
    x: Tensor, w: Tensor, ad: Optional[DoRAAdapterDevice],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Direct DoRA projection hook for Krea2 block lowering.

    Present adapter path returns x @ W_eff^T. It is not an additive LoRA delta.
    """
    if ad:
        return krea2_direct_dora_projection_forward_resident(ad.value(), x, w, M, ctx)
    var nb = _no_bias()
    return linear(x, w, nb^, ctx)


def krea2_block_direct_dora_projection_backward_dev(
    d_y: Tensor, x: Tensor, w: Tensor, ad: Optional[DoRAAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _DirectDoRALinBwdT:
    """Direct DoRA projection backward hook for Krea2 block lowering.

    Present adapter path returns the full W_eff d_x from the DoRA primitive. Do
    not add a separate frozen-W base d_x.
    """
    if ad:
        var g = krea2_direct_dora_projection_backward_resident(
            ad.value(), d_y, x, w, M, ctx,
        )
        return _DirectDoRALinBwdT(
            g.d_x.clone(ctx),
            Krea2DirectDoRAGradT(
                Optional[TArc](TArc(g.d_a.clone(ctx))),
                Optional[TArc](TArc(g.d_b.clone(ctx))),
                Optional[TArc](TArc(g.d_m.clone(ctx))),
            ),
        )
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return _DirectDoRALinBwdT(d_x^, Krea2DirectDoRAGradT(None, None, None))


def krea2_block_direct_oft_projection_forward(
    x: Tensor, w: Tensor, ad: Optional[Krea2DirectOFTDeviceSlot],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Direct OFT projection hook for Krea2 block lowering."""
    if ad:
        return krea2_direct_oft_projection_forward_resident(ad.value(), x, w, M, ctx)
    var nb = _no_bias()
    return linear(x, w, nb^, ctx)


def krea2_block_direct_oft_projection_backward_dev(
    d_y: Tensor, x: Tensor, w: Tensor, ad: Optional[Krea2DirectOFTDeviceSlot],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _DirectOFTLinBwdT:
    """Direct OFT projection backward hook for Krea2 block lowering."""
    if ad:
        var g = krea2_direct_oft_projection_backward_resident(
            ad.value(), d_y, x, w, M, ctx,
        )
        return _DirectOFTLinBwdT(
            g.d_x.clone(ctx),
            Krea2DirectOFTGradT(Optional[TArc](TArc(g.d_vec.clone(ctx)))),
        )
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return _DirectOFTLinBwdT(d_x^, Krea2DirectOFTGradT(None))


def krea2_single_stream_block_lora_backward[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, L, features] upstream grad of the block output
    vec: Tensor,          # [1, 6*features]  (for the raw mod chunks)
    w: Krea2BlockWeights, lora: Krea2BlockLora, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call:
        # None (or real_len == L) = sdpa_backward (full attn, BIT-IDENTICAL to the
        # pre-flash block). Present & < L = cuDNN flash-padmask backward, consuming
        # the saved flash bf16 q/k/v/o + F32 stats (set in the forward), passing the
        # SAME real_len so the bwd respects the same [real_len:L] pad masking.
        # FAIL-LOUD if real_len < L but the saved tape has no flash set (fwd/bwd
        # real_len mismatch).
) raises -> Krea2BlockGrads:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    # raw mod chunks (pregate/postgate/prescale/postscale needed in backward).
    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── MLP branch backward: x2 = residual_gate(x1, postgate, m) ──────────────
    # o = x1 + postgate*m  → d_x1(res)=d_out ; d_m = d_out*postgate (per-channel).
    # (`m` is saved; gate_residual_backward only needs y's value for d_g, which we
    # skip with compute_gate_grad=False — but it still shape-checks y.)
    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    # grg2.d_x = passthrough to x1 ; grg2.d_y = grad into m (= d_out*postgate)
    var d_m = grg2.d_y.clone(ctx)

    # m = mlp_down(sw)  → d_sw (base dx + LoRA dx) + mlp_down dA/dB
    var bw_down = _linear_bwd_dx(
        d_m, saved.sw[], w.mlp_down_w[], lora.mlp_down_w, M, mlpdim, features, ctx
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()

    # sw = swiglu(mlp_gate, mlp_up) → d_mlp_gate, d_mlp_up
    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    # mlp_gate = mlp_gate_w(xm2) ; mlp_up = mlp_up_w(xm2)  → both feed xm2
    var bw_mg = _linear_bwd_dx(
        sgb.d_gate, saved.xm2[], w.mlp_gate_w[], lora.mlp_gate_w, M, features, mlpdim, ctx
    )
    var bw_mu = _linear_bwd_dx(
        sgb.d_up, saved.xm2[], w.mlp_up_w[], lora.mlp_up_w, M, features, mlpdim, ctx
    )
    var g_mg = bw_mg.lora.copy()
    var g_mu = bw_mu.lora.copy()
    var d_xm2 = add(bw_mg.d_x, bw_mu.d_x, ctx)             # [1,L,features]

    # xm2 = modulate(xn2, postscale, postshift) → d_xn2 (drop d_scale/d_shift)
    # modulate_backward requires scale to match the (bf16) acts dtype — the F32-scale
    # production path needs the cast (forward modulate casts internally; backward raises).
    # Mixed precision: saved.xn2 is F32 (rms_norm w/ F32 scale) but d_xm2 is bf16
    # (matmul-backward grad). modulate operated in F32 in the fwd → cast both grad-in
    # and scale to the F32 acts dtype so modulate_backward is dtype-consistent.
    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    # xn2 = postnorm(x1) (weight=postnorm+1, FROZEN) → d_x1 via rms_norm_backward.
    # Mixed precision: the saved acts are bf16 (block input feeds bf16 through the
    # norm→modulate→matmul chain) but the postnorm scale is F32. The FORWARD
    # rms_norm casts the F32 scale DOWN to the act dtype and computes in bf16
    # (norm.mojo:173-174); mirror that here — cast (scale+1) to the act dtype so
    # rms_norm_backward runs the all-bf16 path (go/x/weight matched), not the
    # F32-acts-only mixed path. In the F32 gate this cast is F32→F32 (no-op).
    # FROZEN norm scale → d_x only (rms_norm_backward_dx skips the O(cols²) discarded
    # d_g kernel that was 89% of the step; see norm_backward.mojo:374).
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    # x1 feeds the residual (grg2.d_x) AND postnorm(x1) → SUM.
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    # ── ATTENTION branch backward: x1 = residual_gate(x, pregate, a) ──────────
    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)                          # grad into a (=d_x1*pregate)

    # a = wo(gated) → d_gated (base dx + LoRA dx) + wo dA/dB
    var bw_wo = _linear_bwd_dx(
        d_a, saved.gated[], w.wo[], lora.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()

    # gated = attn_flat * sg  → d_attn_flat = d_gated*sg ; d_sg = d_gated*attn_flat
    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    # Differentiate from the saved sigmoid output; PyTorch autograd saves y.
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    # attn_flat = reshape(sdpa(q_rope, k_full, v_full)) → sdpa backward. Length-bucket
    # pad (real_len present & < L): the cuDNN flash-padmask backward from the saved
    # bf16 q/k/v/o + F32 stats (no recompute), passing the SAME real_len so the bwd
    # respects the [real_len:L] pad masking. No pad: the math sdpa_backward
    # (BIT-IDENTICAL to the pre-flash block). FLASH dQ is NONDETERMINISTIC run-to-run
    # (cuDNN atomics on the dQ accumulation) → flash-path grads are value-tolerance,
    # NOT bit-exact (see krea2_mask_pad_gate's documented tolerance).
    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 block bwd: real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        # BF16 flash bwd consumes BF16 d_att and returns BF16 dQ/dK/dV. F32 stays
        # inside cuDNN stats/score math, not at the model/activation boundary.
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())
    # d_q_sb [1,L,HEADS,Dh] ; d_k_sb [1,L,HEADS,Dh] ; d_v_sb [1,L,HEADS,Dh]

    # GQA backward: repeat_kv sum-reduce HEADS → KVHEADS for k and v.
    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    # RoPE backward (cos/sin non-learnable → only d_x).
    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    # QKNorm backward (weight=qnorm/knorm+1, FROZEN) → d_q_pre, d_k_pre. Same
    # mixed-precision mirror as the rb2 call above: q_pre/k_pre are bf16 acts, the
    # qnorm/knorm scales are F32 → cast (scale+1) down to the act dtype so the
    # all-bf16 rms_norm_backward path runs (matches the bf16 forward QKNorm).
    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    # flatten BSHD grads back to [1,L,*] for the projection backward.
    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    # projections feed xm: q=wq(xm), k=wk(xm), v=wv(xm), gate=gate_w(xm).
    var bw_q = _linear_bwd_dx(d_q, saved.xm[], w.wq[], lora.wq, M, features, HEADS * HEADDIM, ctx)
    var bw_k = _linear_bwd_dx(d_k, saved.xm[], w.wk[], lora.wk, M, features, KVHEADS * HEADDIM, ctx)
    var bw_v = _linear_bwd_dx(d_v_flat, saved.xm[], w.wv[], lora.wv, M, features, KVHEADS * HEADDIM, ctx)
    var bw_g = _linear_bwd_dx(d_gate_pre, saved.xm[], w.gate_w[], lora.gate_w, M, features, features, ctx)
    var g_wq = bw_q.lora.copy()
    var g_wk = bw_k.lora.copy()
    var g_wv = bw_v.lora.copy()
    var g_gate = bw_g.lora.copy()

    # sum the four projection input-grads into d_xm.
    var d_xm = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)

    # xm = modulate(xn, prescale, preshift) → d_xn (drop param grads).
    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    # xn = prenorm(x) (weight=prenorm+1, FROZEN) → d_x via rms_norm_backward. Same
    # mixed-precision mirror: saved.x is the bf16 block input, prenorm scale is F32
    # → cast (scale+1) down to the act dtype for the all-bf16 path (matches fwd).
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    # x feeds: residual (grg1.d_x), prenorm(x) (rb1.d_x). SUM.
    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockGrads(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-GRAD BACKWARD — bit-identical math to krea2_single_stream_block_lora_backward
# above, but the 8 LoRA dA/dB stay on DEVICE (_linear_bwd_dx_dev, no per-adapter
# to_host). Returns Krea2BlockGradsT. The streamed stack consumes those transient
# grads block-by-block: either per-block D2H for host-list compatibility or D2D
# preload into AdamW state. The body is otherwise a verbatim clone of the host
# backward — keep the two in lockstep if the block math changes.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_single_stream_block_lora_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, L, features] upstream grad of the block output
    vec: Tensor,          # [1, 6*features]  (for the raw mod chunks)
    w: Krea2BlockWeights, lora: Krea2BlockLora, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call
        # (same contract as the host backward).
) raises -> Krea2BlockGradsT:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── MLP branch backward ──────────────────────────────────────────────────
    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)

    var bw_down = _linear_bwd_dx_dev(
        d_m, saved.sw[], w.mlp_down_w[], lora.mlp_down_w, M, mlpdim, features, ctx
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mg = _linear_bwd_dx_dev(
        sgb.d_gate, saved.xm2[], w.mlp_gate_w[], lora.mlp_gate_w, M, features, mlpdim, ctx
    )
    var bw_mu = _linear_bwd_dx_dev(
        sgb.d_up, saved.xm2[], w.mlp_up_w[], lora.mlp_up_w, M, features, mlpdim, ctx
    )
    var g_mg = bw_mg.lora.copy()
    var g_mu = bw_mu.lora.copy()
    var d_xm2 = add(bw_mg.d_x, bw_mu.d_x, ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    # ── ATTENTION branch backward ────────────────────────────────────────────
    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)

    var bw_wo = _linear_bwd_dx_dev(
        d_a, saved.gated[], w.wo[], lora.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 block bwd (dev): real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var bw_q = _linear_bwd_dx_dev(d_q, saved.xm[], w.wq[], lora.wq, M, features, HEADS * HEADDIM, ctx)
    var bw_k = _linear_bwd_dx_dev(d_k, saved.xm[], w.wk[], lora.wk, M, features, KVHEADS * HEADDIM, ctx)
    var bw_v = _linear_bwd_dx_dev(d_v_flat, saved.xm[], w.wv[], lora.wv, M, features, KVHEADS * HEADDIM, ctx)
    var bw_g = _linear_bwd_dx_dev(d_gate_pre, saved.xm[], w.gate_w[], lora.gate_w, M, features, features, ctx)
    var g_wq = bw_q.lora.copy()
    var g_wk = bw_k.lora.copy()
    var g_wv = bw_v.lora.copy()
    var g_gate = bw_g.lora.copy()

    var d_xm = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


# Direct DoRA device-grad backward. This is the same block chain as the LoRA
# device-grad backward above, but each projection backward is full W_eff
# substitution: when a direct adapter is present, the helper returns the full
# d_x and DoRA d_A/d_B/d_m. Base W is frozen.
def krea2_single_stream_block_dora_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,
    vec: Tensor,
    w: Krea2BlockWeights, dora: Krea2BlockDirectDoRA, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockDirectDoRAGradsT:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)

    var bw_down = krea2_block_direct_dora_projection_backward_dev(
        d_m, saved.sw[], w.mlp_down_w[], dora.mlp_down_w,
        M, mlpdim, features, ctx,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.dora.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mg = krea2_block_direct_dora_projection_backward_dev(
        sgb.d_gate, saved.xm2[], w.mlp_gate_w[], dora.mlp_gate_w,
        M, features, mlpdim, ctx,
    )
    var bw_mu = krea2_block_direct_dora_projection_backward_dev(
        sgb.d_up, saved.xm2[], w.mlp_up_w[], dora.mlp_up_w,
        M, features, mlpdim, ctx,
    )
    var g_mg = bw_mg.dora.copy()
    var g_mu = bw_mu.dora.copy()
    var d_xm2 = add(bw_mg.d_x, bw_mu.d_x, ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)

    var bw_wo = krea2_block_direct_dora_projection_backward_dev(
        d_a, saved.gated[], w.wo[], dora.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.dora.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 direct DoRA bwd: real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var bw_q = krea2_block_direct_dora_projection_backward_dev(
        d_q, saved.xm[], w.wq[], dora.wq, M, features, HEADS * HEADDIM, ctx,
    )
    var bw_k = krea2_block_direct_dora_projection_backward_dev(
        d_k, saved.xm[], w.wk[], dora.wk, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_v = krea2_block_direct_dora_projection_backward_dev(
        d_v_flat, saved.xm[], w.wv[], dora.wv, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_g = krea2_block_direct_dora_projection_backward_dev(
        d_gate_pre, saved.xm[], w.gate_w[], dora.gate_w, M, features, features, ctx,
    )
    var g_wq = bw_q.dora.copy()
    var g_wk = bw_k.dora.copy()
    var g_wv = bw_v.dora.copy()
    var g_gate = bw_g.dora.copy()

    var d_xm = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockDirectDoRAGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


# Direct OFT device-grad backward. Same chain as the LoRA device-grad backward,
# but each projection consumes the current frozen W_orig plus resident OFT vec
# and returns direct d_vec/d_x without a dense full-delta carrier.
def krea2_single_stream_block_oft_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,
    vec: Tensor,
    w: Krea2BlockWeights, oft: Krea2BlockDirectOFT, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockDirectOFTGradsT:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)

    var bw_down = krea2_block_direct_oft_projection_backward_dev(
        d_m, saved.sw[], w.mlp_down_w[], oft.mlp_down_w,
        M, mlpdim, features, ctx,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.oft.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mg = krea2_block_direct_oft_projection_backward_dev(
        sgb.d_gate, saved.xm2[], w.mlp_gate_w[], oft.mlp_gate_w,
        M, features, mlpdim, ctx,
    )
    var bw_mu = krea2_block_direct_oft_projection_backward_dev(
        sgb.d_up, saved.xm2[], w.mlp_up_w[], oft.mlp_up_w,
        M, features, mlpdim, ctx,
    )
    var g_mg = bw_mg.oft.copy()
    var g_mu = bw_mu.oft.copy()
    var d_xm2 = add(bw_mg.d_x, bw_mu.d_x, ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)

    var bw_wo = krea2_block_direct_oft_projection_backward_dev(
        d_a, saved.gated[], w.wo[], oft.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.oft.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 direct OFT bwd: real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var bw_q = krea2_block_direct_oft_projection_backward_dev(
        d_q, saved.xm[], w.wq[], oft.wq, M, features, HEADS * HEADDIM, ctx,
    )
    var bw_k = krea2_block_direct_oft_projection_backward_dev(
        d_k, saved.xm[], w.wk[], oft.wk, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_v = krea2_block_direct_oft_projection_backward_dev(
        d_v_flat, saved.xm[], w.wv[], oft.wv, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_g = krea2_block_direct_oft_projection_backward_dev(
        d_gate_pre, saved.xm[], w.gate_w[], oft.gate_w, M, features, features, ctx,
    )
    var g_wq = bw_q.oft.copy()
    var g_wk = bw_k.oft.copy()
    var g_wv = bw_v.oft.copy()
    var g_gate = bw_g.oft.copy()

    var d_xm = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockDirectOFTGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )
