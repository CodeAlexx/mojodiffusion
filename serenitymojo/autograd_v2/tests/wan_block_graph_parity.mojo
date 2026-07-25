# autograd_v2/tests/wan_block_graph_parity.mojo — Phase 2 WHOLE-BLOCK gate.
#
# Gates the assembled fine-grained whole-block backward (wan22_block_graph.mojo
# wan22_block_lora_graph_backward — recompute the ENTIRE block from its input
# through the record_* wrappers, then engine backward) against the DEVICE oracle
# wan22_block_lora_backward_devnative run on the same block input + same d_out.
#
# PARITY CLASSES (HANDOFF §6, C15 fan-in — decided BEFORE running, not after):
#   * 10 LoRA slots (sa q/k/v/o, ca q/k/v/o, ffn0, ffn2) → BIT-EXACT. Each d_A/d_B
#     is a function of that projection's output grad + its saved input, computed
#     UPSTREAM of any fan-in fold, so fold order cannot reach it.
#   * d_x / d_context → 4dp VALUE-CLASS. The oracle folds the 3-way sa_in fan-in
#     as BASE-TREE + LORA-TREE (wan22_block.mojo:1925-1927); OPK_WAN_PROJ_LORA
#     folds base+lora per-proj, so the engine's slot-ordered left fold groups the
#     same six bf16 addends differently. bf16 add is not associative → this is a
#     legitimate reassociation, NOT a numerical error.
# A BIT failure on any LoRA slot is a REAL failure (that is where the training
# signal lives). Degenerate (all-zero) grads must fail the teeth check.
#
# Wan is math-mode/deterministic (no flash in this path) → the LoRA class is a
# TRUE bit gate, not a variance class.
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/wan_block_graph_parity.mojo -o /tmp/wan_block_graph
# Run: env LD_LIBRARY_PATH=<cshim+cudnn+pixi> /tmp/wan_block_graph
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockLora, _ta16,
    wan22_block_lora_forward, wan22_block_lora_backward_devnative,
)
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.wan22_block_graph import wan22_block_lora_graph_backward

comptime H = 24
comptime Dh = 8
comptime DIM = H * Dh        # 192
comptime S = 5
comptime TXT = 4
comptime FFN = 40
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime LSCALE = Float32(1.0)
comptime VTOL = Float32(1e-03)   # 4dp value class for d_x / d_context

def _slot_names() -> List[String]:
    """Flat LoRA slot order W_SA_Q..W_FFN2 == 0..9 (wan22_block.mojo:1939)."""
    var n = List[String]()
    n.append(String("sa_q"))
    n.append(String("sa_k"))
    n.append(String("sa_v"))
    n.append(String("sa_o"))
    n.append(String("ca_q"))
    n.append(String("ca_k"))
    n.append(String("ca_v"))
    n.append(String("ca_o"))
    n.append(String("ffn0"))
    n.append(String("ffn2"))
    return n^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _make_adapter(
    var a: List[Float32], var b: List[Float32], in_f: Int, out_f: Int,
) -> LoraAdapter:
    return LoraAdapter(
        a^, b^, RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _weights(ctx: DeviceContext) raises -> WanBlockWeights:
    return WanBlockWeights(
        _randn(DIM * DIM, 1, 0.06), _randn(DIM * DIM, 2, 0.06),
        _randn(DIM * DIM, 3, 0.06), _randn(DIM * DIM, 4, 0.06),
        _randn(DIM, 5, 0.05), _randn(DIM, 6, 0.05), _randn(DIM, 7, 0.05), _randn(DIM, 8, 0.05),
        _randn(DIM, 9, 0.1), _randn(DIM, 10, 0.1),
        _randn(DIM * DIM, 11, 0.06), _randn(DIM * DIM, 12, 0.06),
        _randn(DIM * DIM, 13, 0.06), _randn(DIM * DIM, 14, 0.06),
        _randn(DIM, 15, 0.05), _randn(DIM, 16, 0.05), _randn(DIM, 17, 0.05), _randn(DIM, 18, 0.05),
        _randn(DIM, 19, 0.1), _randn(DIM, 20, 0.1),
        _randn(DIM, 21, 0.1), _randn(DIM, 22, 0.05),
        _randn(FFN * DIM, 23, 0.05), _randn(FFN, 24, 0.05),
        _randn(DIM * FFN, 25, 0.05), _randn(DIM, 26, 0.05),
        DIM, FFN, Dh, ctx,
    )


def _modvecs() raises -> WanModVecs:
    return WanModVecs(
        _randn(S * DIM, 31, 0.1), _randn(S * DIM, 32, 0.1), _randn(S * DIM, 33, 0.1),
        _randn(S * DIM, 34, 0.1), _randn(S * DIM, 35, 0.1), _randn(S * DIM, 36, 0.1),
    )


def _lora() -> WanBlockLora:
    # every adapter nonzero in BOTH A and B so no d_A is degenerate-by-zero.
    return WanBlockLora(
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 101, 0.07), _randn(DIM * RANK, 102, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 103, 0.07), _randn(DIM * RANK, 104, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 105, 0.07), _randn(DIM * RANK, 106, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 107, 0.07), _randn(DIM * RANK, 108, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 109, 0.07), _randn(DIM * RANK, 110, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 111, 0.07), _randn(DIM * RANK, 112, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 113, 0.07), _randn(DIM * RANK, 114, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 115, 0.07), _randn(DIM * RANK, 116, 0.05), DIM, DIM)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * DIM, 150, 0.06), _randn(FFN * RANK, 151, 0.04), DIM, FFN)),
        Optional[LoraAdapter](_make_adapter(_randn(RANK * FFN, 160, 0.06), _randn(DIM * RANK, 161, 0.04), FFN, DIM)),
    )


def _cmp_bit(
    name: String, got: Tensor, want: Tensor, ctx: DeviceContext, mut allok: Bool
) raises:
    var hg = got.to_host(ctx)
    var hw = want.to_host(ctx)
    var bad = 0
    if len(hg) != len(hw):
        bad = -1
    else:
        for i in range(len(hg)):
            if hg[i] != hw[i]:
                bad += 1
    var verdict = String("PASS") if bad == 0 else String("FAIL")
    if bad != 0:
        allok = False
    print("GATE(bit)", name, verdict, "n_mismatch=", bad, "/", len(hw))


def _cmp_value(
    name: String, got: Tensor, want: Tensor, ctx: DeviceContext, mut allok: Bool
) raises:
    """4dp value class (the C15 reassociation class for d_x/d_context)."""
    var hg = got.to_host(ctx)
    var hw = want.to_host(ctx)
    if len(hg) != len(hw):
        allok = False
        print("GATE(4dp)", name, "FAIL len", len(hg), "vs", len(hw))
        return
    var maxdiff = Float32(0.0)
    var nbit = 0
    for i in range(len(hg)):
        var d = hg[i] - hw[i]
        if d < Float32(0.0):
            d = -d
        if d > maxdiff:
            maxdiff = d
        if hg[i] != hw[i]:
            nbit += 1
    var ok = maxdiff <= VTOL
    if not ok:
        allok = False
    var verdict = String("PASS") if ok else String("FAIL")
    print(
        "GATE(4dp)", name, verdict, "max_abs_diff=", maxdiff,
        " tol=", VTOL, " (n_bit_diff=", nbit, "/", len(hw), ")",
    )


def _nonzero(t: Tensor, ctx: DeviceContext) raises -> Int:
    var h = t.to_host(ctx)
    var nz = 0
    for i in range(len(h)):
        if h[i] != 0.0:
            nz += 1
    return nz


def _gate(batched_cross: Bool, ctx: DeviceContext) raises -> Bool:
    print("")
    print("==== wan_block_graph_parity  batched_cross =", batched_cross, "====")
    print("dims: H=", H, " Dh=", Dh, " DIM=", DIM, " S=", S, " TXT=", TXT, " FFN=", FFN)
    var w = _weights(ctx)
    var mv = _modvecs()
    var lora = _lora()
    var cos = Tensor.from_host(_randn(S * (Dh // 2), 201, 1.0), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_randn(S * (Dh // 2), 202, 1.0), [S, Dh // 2], STDtype.F32, ctx)
    var x_h = _randn(S * DIM, 210, 1.0)
    var context_h = _randn(TXT * DIM, 211, 1.0)

    # hand-chain forward (produces the save-all acts the oracle backward reads)
    # forward and graph recompute MUST use the same cross-attn kernel.
    var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
        x_h.copy(), context_h.copy(), mv, w, lora, cos, sin, DIM, FFN, EPS, ctx,
        True, False, batched_cross,
    )
    var d_out = Tensor.from_host(_randn(S * DIM, 220, 1.0), [S, DIM], STDtype.BF16, ctx)

    # ORACLE: device-native whole-block backward (certified bit-equal to host).
    var gd = wan22_block_lora_backward_devnative[H, Dh, S, TXT](
        d_out, mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    # GRAPH: recompute the whole block from its INPUT (same bf16 rounding as the
    # forward's own _ta16 upload), engine backward seeded with the same d_out.
    var x_in = _ta16(x_h.copy(), [S, DIM], ctx)
    var context_in = _ta16(context_h.copy(), [TXT, DIM], ctx)
    var gg = wan22_block_lora_graph_backward[H, Dh, S, TXT](
        d_out, x_in, context_in, mv, w, lora, cos, sin, DIM, FFN, EPS, ctx,
        batched_cross,
    )

    var allok = True
    var names = _slot_names()
    # ── the 10 LoRA slots: BIT ──
    for i in range(10):
        _cmp_bit(names[i] + "_dA", gg.d_a[i].value()[], gd.d_a[i].value()[], ctx, allok)
        _cmp_bit(names[i] + "_dB", gg.d_b[i].value()[], gd.d_b[i].value()[], ctx, allok)
    # ── block d_x / d_context: 4dp value class (C15 fold reassociation) ──
    _cmp_value("d_x", gg.d_x[], gd.d_x, ctx, allok)
    _cmp_value("d_context", gg.d_context[], gd.d_context, ctx, allok)

    # ── teeth: a degenerate (all-zero) comparison must not read as PASS ──
    var t_saq = _nonzero(gg.d_a[0].value()[], ctx)
    var t_ffn2 = _nonzero(gg.d_b[9].value()[], ctx)
    var t_dx = _nonzero(gg.d_x[], ctx)
    var t_dctx = _nonzero(gg.d_context[], ctx)
    print("teeth: sa_q_dA nz=", t_saq, " ffn2_dB nz=", t_ffn2,
          " d_x nz=", t_dx, " d_context nz=", t_dctx, " (all must be > 0)")
    var teeth = t_saq > 0 and t_ffn2 > 0 and t_dx > 0 and t_dctx > 0

    if allok and teeth:
        print("VERDICT: PASS — whole-block graph == devnative (10 LoRA slots BIT,")
        print("                d_x/d_context 4dp value-class per C15)")
    else:
        print("VERDICT: FAIL — whole-block graph diverged")
    return allok and teeth


def main() raises:
    var ctx = DeviceContext()
    # Run BOTH cross-attn modes. batched_cross only changes WHICH kernel both sides
    # use for the cross-attn forward; because forward and recompute use the SAME one,
    # the LoRA slots must stay BIT-EQUAL in both modes. (The value-class difference
    # of batched-vs-looped is between BUILDS, not between graph and devnative.)
    var ok_loop = _gate(False, ctx)
    var ok_batched = _gate(True, ctx)
    print("")
    if ok_loop and ok_batched:
        print("OVERALL: PASS — both cross-attn modes gate clean")
    else:
        print("OVERALL: FAIL — loop_ok=", ok_loop, " batched_ok=", ok_batched)
