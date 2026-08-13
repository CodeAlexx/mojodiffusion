# serenitymojo/models/ideogram4/parity/ideogram4_block_ft_parity.mojo
#
# PARITY GATE for the Ideogram-4 block FULL-FINETUNE backward
# (models/ideogram4/ideogram4_block_ft.mojo : ideogram4_block_ft_backward_dev):
# ALL 13 trainable param grads (v2 FULL SURFACE — 6 matmul dW + adaln bias d_b
# + 6 rms-scale d_g, F32 device) + the input grads d_x / d_adaln_input vs
# torch autograd at the REAL dims (S=260, Hidden=4608, H=18, Dh=256, FF=12288,
# Adaln=512) on the REAL block-0 checkpoint weights (fp8 -> bf16, the same
# loader the trainer uses). Bars: cos >= 0.999 for the v1 captures (unchanged)
# and cos >= 0.9999 for the 7 NEW 1D arms (FULL_SURFACE_PLAN Phase B bar).
#
# Oracle: per MJ-1041 the ideogram4 oracle is AI-TOOLKIT —
# models/dit/parity/ideogram4_aitoolkit_oracle.py in its BASE-ONLY mode
# (IDEOGRAM4_FT_ORACLE=1 -> ideogram4_aitoolkit_block0_ft.safetensors). The
# LoRA-mode fixture is NOT valid for FT (the adapters shift the dX chain and,
# through downstream activations, the dW values — krea2 trap; and its base
# weights are frozen, so it has no W.grad refs at all).
#
# The forward is ideogram4_block_lora_forward with DEFAULT (zero-B) adapters:
# up = down @ 0 = 0 and add(base, 0*scale) = base, so the produced tape
# (Ideogram4BlockActs) is the base block's bit-for-bit — no separate FT
# forward needed. v2 (FULL_SURFACE_PLAN Phase B row 1): the previously-frozen
# 1D params (adaln bias, attn/ffn rms scales, norm_q/norm_k) are now TRAINED
# and their grads gated here against the oracle's bwd.grad.<name> refs.
#
# Run (oracle FIRST, as SEPARATE commands — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   IDEOGRAM4_FT_ORACLE=1 /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/dit/parity/ideogram4_aitoolkit_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#       -I . -I /home/alex/MOJO-libs -Xlinker -lm \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker $PWD/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#       serenitymojo/models/ideogram4/parity/ideogram4_block_ft_parity.mojo \
#       -o /tmp/ideogram4_block_ft_parity
#   /tmp/ideogram4_block_ft_parity

from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.parity import ParityHarness

from serenitymojo.models.ideogram4.block import (
    I4_SLOT_QKV,
    I4_SLOT_O,
    I4_SLOT_W1,
    I4_SLOT_W2,
    I4_SLOT_W3,
    I4_SLOT_ADALN,
    Ideogram4BlockWeights,
    load_ideogram4_block_weights,
    build_ideogram4_lora_set,
    ideogram4_block_lora_forward,
    LArc,
)
from serenitymojo.models.ideogram4.ideogram4_block_ft import (
    I4_FT_SLOT_ADALN_B,
    I4_FT_SLOT_ATTN_NORM1,
    I4_FT_SLOT_ATTN_NORM2,
    I4_FT_SLOT_FFN_NORM1,
    I4_FT_SLOT_FFN_NORM2,
    I4_FT_SLOT_NORM_Q,
    I4_FT_SLOT_NORM_K,
    ideogram4_block_ft_backward_dev,
)
from serenitymojo.models.ideogram4.config import (
    IDEOGRAM4_ADALN_DIM,
    IDEOGRAM4_HEAD_DIM,
    IDEOGRAM4_HIDDEN,
    IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_NUM_HEADS,
)

# Real block-0 checkpoint (fp8) — the SAME file the oracle dequantized to bf16.
comptime CKPT_DIR = "/home/alex/.serenity/models/ideogram-4-fp8/transformer"
# ai-toolkit FT oracle fixture (block 0, base-only graph).
comptime FIX = (
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
    "ideogram4_aitoolkit_block0_ft.safetensors"
)

# Geometry — fixed by the oracle (GH=GW=16 -> NIMG=256, NTEXT=4 -> L=260).
comptime S = 260
comptime Hidden = IDEOGRAM4_HIDDEN              # 4608
comptime Heads = IDEOGRAM4_NUM_HEADS            # 18
comptime Dh = IDEOGRAM4_HEAD_DIM                # 256
comptime FF = IDEOGRAM4_INTERMEDIATE_SIZE       # 12288
comptime Adaln = IDEOGRAM4_ADALN_DIM            # 512
comptime BAR = 0.999
comptime BAR_V2 = 0.9999   # the 7 new 1D arms (Phase B campaign bar)


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


# Load a fixture tensor (F32 on disk) -> BF16 device Tensor with `shape` (the
# oracle is the bf16 production path; feeding bf16 matches block.mojo).
def _fix_bf16(
    st: ShardedSafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var t = Tensor.from_view_as_bf16(st.tensor_view(name), ctx)
    return reshape(t, shape^, ctx)


# Load the fixture's host F32 reference for a capture.
def _fix_ref_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> List[Float32]:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx).to_host(ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Ideogram-4 BLOCK FULL-FT ai-toolkit ORACLE parity (cos >= 0.999) ===")
    print("S=", S, " Hidden=", Hidden, " H=", Heads, " Dh=", Dh, " FF=", FF,
          " Adaln=", Adaln, " (base-only oracle: IDEOGRAM4_FT_ORACLE=1)")

    var fx = ShardedSafeTensors.open(String(FIX))
    var ck = ShardedSafeTensors.open(String(CKPT_DIR))

    # ── REAL block-0 weights (mojo fp8->bf16 loader; same as the oracle) ───────
    var w = load_ideogram4_block_weights(ck, 0, ctx)
    print("[gate] loaded REAL block-0 weights (fp8 -> bf16) from", CKPT_DIR)

    # ── ZERO-B adapters: the LoRA forward with B=0 IS the base forward ─────────
    var loras = build_ideogram4_lora_set[Hidden, FF, Adaln](4, Float32(4.0), ctx, 1)
    var bl = List[LArc]()
    for slot in range(6):
        bl.append(loras.ad[slot])
    print("[gate] zero-B adapters built (base forward bit-for-bit; no LoRA in bwd)")

    # ── oracle forward inputs ─────────────────────────────────────────────────
    var x = _fix_bf16(fx, "fwd.block0_in_x", _s2(S, Hidden), ctx)       # [S,H]
    var adaln = _fix_bf16(fx, "fwd.adaln_input", _s2(1, Adaln), ctx)    # [1,Adaln]
    var cosf = _fix_bf16(fx, "fwd.cos", _s3(1, S, Dh), ctx)             # [1,S,Dh]
    var sinf = _fix_bf16(fx, "fwd.sin", _s3(1, S, Dh), ctx)             # [1,S,Dh]

    # ── FORWARD (base tape) ───────────────────────────────────────────────────
    var fwd = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
        x, adaln, cosf, sinf, w, bl, ctx
    )

    var h = ParityHarness(BAR)
    var n_fail = 0

    var ref_out = _fix_ref_f32(fx, "fwd.block0_out", ctx)
    var r_out = h.compare(fwd.out, ref_out, ctx)
    print("  block0_out     ", r_out)
    if not r_out.passed:
        n_fail += 1

    # ── FT BACKWARD (6 dW F32 + d_x/d_adaln carries) ──────────────────────────
    var d_out = _fix_bf16(fx, "bwd.d_out", _s2(S, Hidden), ctx)         # [S,H]
    var bwd = ideogram4_block_ft_backward_dev[S, Hidden, Heads, Dh, FF, Adaln](
        d_out, fwd.acts^, cosf, sinf, w, ctx
    )

    var ref_dx = _fix_ref_f32(fx, "bwd.d_x", ctx)
    var r_dx = h.compare(bwd.d_x, ref_dx, ctx)
    print("  d_x            ", r_dx)
    if not r_dx.passed:
        n_fail += 1

    var ref_dadaln = _fix_ref_f32(fx, "bwd.d_adaln_input", ctx)
    var r_dadaln = h.compare(bwd.d_adaln_input, ref_dadaln, ctx)
    print("  d_adaln_input  ", r_dadaln)
    if not r_dadaln.passed:
        n_fail += 1

    # dW per slot vs the torch W.grad refs (bwd.grad.<aitk param name>).
    var slot_names = List[String]()
    slot_names.append("attention.qkv.weight")
    slot_names.append("attention.o.weight")
    slot_names.append("feed_forward.w1.weight")
    slot_names.append("feed_forward.w2.weight")
    slot_names.append("feed_forward.w3.weight")
    slot_names.append("adaln_modulation.weight")
    var slot_idx = List[Int]()
    slot_idx.append(I4_SLOT_QKV); slot_idx.append(I4_SLOT_O); slot_idx.append(I4_SLOT_W1)
    slot_idx.append(I4_SLOT_W2); slot_idx.append(I4_SLOT_W3); slot_idx.append(I4_SLOT_ADALN)

    for i in range(len(slot_names)):
        var nm = slot_names[i]
        var idx = slot_idx[i]
        var ref_dw = _fix_ref_f32(fx, String("bwd.grad.") + nm, ctx)
        var r_dw = h.compare(bwd.dw[idx][], ref_dw, ctx)
        print("  dW." + nm + "  ", r_dw)
        if not r_dw.passed:
            n_fail += 1

    # ── v2 FULL-SURFACE arms: adaln bias + 6 rms scales (1D, bar 0.9999) ──────
    var h2 = ParityHarness(BAR_V2)
    var v2_names = List[String]()
    v2_names.append("adaln_modulation.bias")
    v2_names.append("attention_norm1.weight")
    v2_names.append("attention_norm2.weight")
    v2_names.append("ffn_norm1.weight")
    v2_names.append("ffn_norm2.weight")
    v2_names.append("attention.norm_q.weight")
    v2_names.append("attention.norm_k.weight")
    var v2_idx = List[Int]()
    v2_idx.append(I4_FT_SLOT_ADALN_B)
    v2_idx.append(I4_FT_SLOT_ATTN_NORM1)
    v2_idx.append(I4_FT_SLOT_ATTN_NORM2)
    v2_idx.append(I4_FT_SLOT_FFN_NORM1)
    v2_idx.append(I4_FT_SLOT_FFN_NORM2)
    v2_idx.append(I4_FT_SLOT_NORM_Q)
    v2_idx.append(I4_FT_SLOT_NORM_K)

    for i in range(len(v2_names)):
        var nm = v2_names[i]
        var idx = v2_idx[i]
        var ref_dw = _fix_ref_f32(fx, String("bwd.grad.") + nm, ctx)
        var r_dw = h2.compare(bwd.dw[idx][], ref_dw, ctx)
        print("  dP." + nm + "  ", r_dw)
        if not r_dw.passed:
            n_fail += 1

    print("------------------------------------------------------------")
    if n_fail == 0:
        print("IDEOGRAM4 BLOCK FULL-FT PARITY PASS (9 v1 captures cos >= 0.999")
        print("  + 7 v2 full-surface arms cos >= 0.9999)")
    else:
        raise Error(
            String("IDEOGRAM4 BLOCK FULL-FT PARITY FAIL: ")
            + String(n_fail) + " capture(s) below bar"
        )
