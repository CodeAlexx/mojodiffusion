# autograd_v2/tests/ideogram4_block_aitoolkit_parity.mojo — ai-toolkit ORACLE gate.
#
# Proves the serenitymojo Ideogram-4 block forward+backward matches the
# ai-toolkit production oracle fixture (block 0), NOT a self-consistency compare.
#
# Sibling of autograd_v2/tests/ideogram4_block_parity.mojo (the engine==hand-chain
# bit gate). That gate feeds SYNTHETIC inputs and compares the engine against the
# hand-chain. THIS gate instead:
#   * loads the REAL block-0 weights (the mojo fp8->bf16 loader, same checkpoint
#     the oracle dequantized),
#   * loads the EXACT LoRA A/B the oracle used (fixture lora.{slot}.A/.B, mojo
#     convention A=[rank,in] B=[out,rank]),
#   * loads the oracle's forward inputs (fwd.block0_in_x / adaln_input / cos / sin),
#   * runs ideogram4_block_lora_forward, compares vs fwd.block0_out,
#   * loads the oracle's seeded upstream grad (bwd.d_out),
#   * runs ideogram4_block_lora_backward, compares d_x / d_adaln_input and each
#     LoRA dA/dB vs the oracle (bwd.d_x / bwd.d_adaln_input / bwd.lora.{slot}.{dA,dB}).
#
# The mojo block runs BF16 end-to-end and the oracle is the bf16 production path
# (fp8 weights folded to bf16) — so inputs are fed BF16 and the bar is COSINE.
# BAR cos >= 0.999 per capture; FAIL-LOUD (raise / nonzero exit) on any miss.
#
# IDEOGRAM4_SDPA_FLASH is OFF in block.mojo (Dh=256 unsupported by the cuDNN
# flash backward) so the math/native SDPA path runs — same backend the oracle used.
# The oracle's segment_ids are all 1 (one packed sample) -> its block-diagonal
# mask is all-True == the mojo block's sdpa_nomask, so unmasked attention is the
# correct comparison (segment_ids/attn_mask/position_ids are not fed here).
#
# Run (oracle FIRST if the fixture is missing):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/dit/parity/ideogram4_aitoolkit_oracle.py
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/autograd_v2/tests/ideogram4_block_aitoolkit_parity.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.parity import ParityHarness

from serenitymojo.models.ideogram4.lora_module import LoraAdapter
from serenitymojo.models.ideogram4.block import (
    I4_SLOTS_PER_BLOCK,
    I4_SLOT_QKV,
    I4_SLOT_O,
    I4_SLOT_W1,
    I4_SLOT_W2,
    I4_SLOT_W3,
    I4_SLOT_ADALN,
    Ideogram4BlockWeights,
    load_ideogram4_block_weights,
    ideogram4_block_lora_forward,
    ideogram4_block_lora_backward,
    LArc,
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
# ai-toolkit oracle fixture (block 0).
comptime FIX = (
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
    "ideogram4_aitoolkit_block0.safetensors"
)

# Geometry — fixed by the oracle (GH=GW=16 -> NIMG=256, NTEXT=4 -> L=260).
comptime S = 260
comptime Hidden = IDEOGRAM4_HIDDEN              # 4608
comptime Heads = IDEOGRAM4_NUM_HEADS            # 18
comptime Dh = IDEOGRAM4_HEAD_DIM                # 256
comptime FF = IDEOGRAM4_INTERMEDIATE_SIZE       # 12288
comptime Adaln = IDEOGRAM4_ADALN_DIM            # 512
comptime RANK = 16
comptime ALPHA = Float32(16.0)                  # alpha/rank = 1.0 (dyadic)
comptime BAR = 0.999


# ── shape helpers (no variadic List ctor) ────────────────────────────────────
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


# Load a fixture tensor (F32 on disk) -> BF16 device Tensor with `shape`.
# The oracle is the bf16 production path; feeding bf16 matches block.mojo.
def _fix_bf16(
    st: ShardedSafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var t = Tensor.from_view_as_bf16(st.tensor_view(name), ctx)
    # collapse to the mojo block's expected rank (e.g. [1,S,H] -> [S,H]); numel
    # is preserved so reshape is a pure metadata change.
    return reshape(t, shape^, ctx)


# Load the fixture's host F32 reference for a capture (compared in F64 by the
# ParityHarness against the mojo tensor read back as F32).
def _fix_ref_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> List[Float32]:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx).to_host(ctx)


# Build one LoRA adapter from the fixture's exact A/B (NOT random) — bf16, the
# mojo convention A=[rank,in] B=[out,rank], rank 16 / alpha 16.
def _fix_lora(
    st: ShardedSafeTensors, slot: String, in_f: Int, out_f: Int, ctx: DeviceContext
) raises -> LArc:
    var a = _fix_bf16(st, String("lora.") + slot + ".A", _s2(RANK, in_f), ctx)
    var b = _fix_bf16(st, String("lora.") + slot + ".B", _s2(out_f, RANK), ctx)
    return LArc(LoraAdapter(a^, b^, RANK, ALPHA))


def main() raises:
    var ctx = DeviceContext()
    print("=== Ideogram-4 BLOCK ai-toolkit ORACLE parity (cos >= 0.999) ===")

    var fx = ShardedSafeTensors.open(String(FIX))
    var ck = ShardedSafeTensors.open(String(CKPT_DIR))

    # ── REAL block-0 weights (mojo fp8->bf16 loader; same as the oracle) ───────
    var w = load_ideogram4_block_weights(ck, 0, ctx)
    print("[gate] loaded REAL block-0 weights (fp8 -> bf16) from", CKPT_DIR)

    # ── LoRA set: the EXACT adapters the oracle used (fixture lora.*) ──────────
    # Built in I4_SLOT_* order: qkv(0), o(1), w1(2), w2(3), w3(4), adaln(5). The
    # (in,out) per slot are the base-Linear dims the oracle wrapped.
    var bl = List[LArc]()
    bl.append(_fix_lora(fx, "qkv", Hidden, 3 * Hidden, ctx))   # I4_SLOT_QKV
    bl.append(_fix_lora(fx, "o", Hidden, Hidden, ctx))         # I4_SLOT_O
    bl.append(_fix_lora(fx, "w1", Hidden, FF, ctx))            # I4_SLOT_W1
    bl.append(_fix_lora(fx, "w2", FF, Hidden, ctx))            # I4_SLOT_W2
    bl.append(_fix_lora(fx, "w3", Hidden, FF, ctx))            # I4_SLOT_W3
    bl.append(_fix_lora(fx, "adaln", Adaln, 4 * Hidden, ctx))  # I4_SLOT_ADALN
    print("[gate] loaded 6 LoRA adapters from the fixture (mojo A=[r,in] B=[out,r])")

    # ── oracle forward inputs ─────────────────────────────────────────────────
    var x = _fix_bf16(fx, "fwd.block0_in_x", _s2(S, Hidden), ctx)       # [S,H]
    var adaln = _fix_bf16(fx, "fwd.adaln_input", _s2(1, Adaln), ctx)    # [1,Adaln]
    var cosf = _fix_bf16(fx, "fwd.cos", _s3(1, S, Dh), ctx)             # [1,S,Dh]
    var sinf = _fix_bf16(fx, "fwd.sin", _s3(1, S, Dh), ctx)             # [1,S,Dh]

    # ── FORWARD ───────────────────────────────────────────────────────────────
    var fwd = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
        x, adaln, cosf, sinf, w, bl, ctx
    )

    var h = ParityHarness(BAR)
    var n_fail = 0

    var ref_out = _fix_ref_f32(fx, "fwd.block0_out", ctx)
    var r_out = h.compare(fwd.out, ref_out, ctx)
    print("  block0_out         ", r_out)
    if not r_out.passed:
        n_fail += 1

    # ── BACKWARD ──────────────────────────────────────────────────────────────
    var d_out = _fix_bf16(fx, "bwd.d_out", _s2(S, Hidden), ctx)         # [S,H]
    var bwd = ideogram4_block_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
        d_out, fwd.acts^, cosf, sinf, w, bl, ctx
    )

    var ref_dx = _fix_ref_f32(fx, "bwd.d_x", ctx)
    var r_dx = h.compare(bwd.d_x, ref_dx, ctx)
    print("  d_x                ", r_dx)
    if not r_dx.passed:
        n_fail += 1

    var ref_dadaln = _fix_ref_f32(fx, "bwd.d_adaln_input", ctx)
    var r_dadaln = h.compare(bwd.d_adaln_input, ref_dadaln, ctx)
    print("  d_adaln_input      ", r_dadaln)
    if not r_dadaln.passed:
        n_fail += 1

    # LoRA dA/dB per slot, fixture key (bwd.lora.<slot>.<dA|dB>) -> mojo slot index.
    var slot_names = List[String]()
    slot_names.append("qkv"); slot_names.append("o"); slot_names.append("w1")
    slot_names.append("w2"); slot_names.append("w3"); slot_names.append("adaln")
    var slot_idx = List[Int]()
    slot_idx.append(I4_SLOT_QKV); slot_idx.append(I4_SLOT_O); slot_idx.append(I4_SLOT_W1)
    slot_idx.append(I4_SLOT_W2); slot_idx.append(I4_SLOT_W3); slot_idx.append(I4_SLOT_ADALN)

    for i in range(len(slot_names)):
        var nm = slot_names[i]
        var idx = slot_idx[i]
        var ref_da = _fix_ref_f32(fx, String("bwd.lora.") + nm + ".dA", ctx)
        var r_da = h.compare(bwd.lora_grads.d_a[idx][], ref_da, ctx)
        print("  dA." + nm + "  ", r_da)
        if not r_da.passed:
            n_fail += 1
        var ref_db = _fix_ref_f32(fx, String("bwd.lora.") + nm + ".dB", ctx)
        var r_db = h.compare(bwd.lora_grads.d_b[idx][], ref_db, ctx)
        print("  dB." + nm + "  ", r_db)
        if not r_db.passed:
            n_fail += 1

    print("------------------------------------------------------------")
    if n_fail == 0:
        print("IDEOGRAM4 BLOCK ai-toolkit PARITY PASS (all 15 captures cos >= 0.999)")
    else:
        raise Error(
            String("IDEOGRAM4 BLOCK ai-toolkit PARITY FAIL: ")
            + String(n_fail) + " capture(s) below cos 0.999"
        )
