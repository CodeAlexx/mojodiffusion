# autograd_v2/tests/wan_ffn_section_parity.mojo — Phase 2 FFN-SECTION bit gate.
#
# Gates the assembled FFN section (wan22_block_graph.mojo
# wan22_ffn_section_graph_backward — recompute FFN from x_ca through the fine-
# grained record_* wrappers) against the REAL device oracle: the ffn0/ffn2 LoRA
# grads it produces must be BIT-EQUAL to slots 8 (ffn0) / 9 (ffn2) of
# wan22_block_lora_backward_devnative run on the same d_out + saved.x_ca. Those
# grads depend ONLY on d_out + recomputed acts (NOT on d_x_ca), so this needs ZERO
# inline oracle re-derivation. Wan is math-mode/deterministic → TRUE bit gate. This
# ALSO proves the record-side recompute forward (linear + _add_lora_delta) bit-
# matches the hand-chain's save-all activations (else ffn2's d_A would diverge).
#
# Determinism gate (graph vs devnative on the SAME weights) → random weights are
# fine (not a torch-value gate). All 10 attention+ffn adapters are built nonzero so
# the full-block devnative forward/backward is well-posed; only slots 8/9 compared.
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/wan_ffn_section_parity.mojo -o /tmp/wan_ffn_sec
# Run: env LD_LIBRARY_PATH=<cshim+cudnn+pixi> /tmp/wan_ffn_sec
#
# Mojo 1.0.0b1, NVIDIA.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockLora, _tbf16,
    wan22_block_lora_forward, wan22_block_lora_backward_devnative,
)
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.wan22_block_graph import wan22_ffn_section_graph_backward

comptime H = 24
comptime Dh = 8
comptime DIM = H * Dh        # 192
comptime S = 5
comptime TXT = 4
comptime FFN = 40
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime LSCALE = Float32(1.0)


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


def _cmp(name: String, got: Tensor, want: Tensor, ctx: DeviceContext, mut allok: Bool) raises:
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
    print("GATE", name, verdict, "n_mismatch=", bad, "/", len(hw))


def main() raises:
    var ctx = DeviceContext()
    print("==== wan_ffn_section_parity (graph FFN == devnative slots 8/9, BIT) ====")
    var w = _weights(ctx)
    var mv = _modvecs()
    var lora = _lora()
    var cos = Tensor.from_host(_randn(S * (Dh // 2), 201, 1.0), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_randn(S * (Dh // 2), 202, 1.0), [S, Dh // 2], STDtype.F32, ctx)
    var x_h = _randn(S * DIM, 210, 1.0)
    var context_h = _randn(TXT * DIM, 211, 1.0)

    var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
        x_h.copy(), context_h.copy(), mv, w, lora, cos, sin, DIM, FFN, EPS, ctx,
    )
    var d_out = Tensor.from_host(_randn(S * DIM, 220, 1.0), [S, DIM], STDtype.BF16, ctx)

    # device oracle: whole-block devnative backward → slots 8 (ffn0) / 9 (ffn2).
    var gd = wan22_block_lora_backward_devnative[H, Dh, S, TXT](
        d_out, mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    # graph FFN section: recompute from saved x_ca, seed d_out.
    var x_ca = TArc(_tbf16(fwd.saved.x_ca.copy(), [S, DIM], ctx))
    var fg = wan22_ffn_section_graph_backward[S, DIM, FFN](d_out, x_ca, mv, w, lora, EPS, ctx)

    var allok = True
    _cmp("ffn0_dA", fg.ffn0_da.value()[], gd.d_a[8].value()[], ctx, allok)
    _cmp("ffn0_dB", fg.ffn0_db.value()[], gd.d_b[8].value()[], ctx, allok)
    _cmp("ffn2_dA", fg.ffn2_da.value()[], gd.d_a[9].value()[], ctx, allok)
    _cmp("ffn2_dB", fg.ffn2_db.value()[], gd.d_b[9].value()[], ctx, allok)

    # teeth: grads must be nonzero (non-degenerate).
    var teeth = fg.ffn0_da.value()[].to_host(ctx)
    var tz = 0
    for i in range(len(teeth)):
        if teeth[i] != 0.0:
            tz += 1
    print("teeth: ffn0_dA nonzero count=", tz, " (must be > 0)")

    if allok and tz > 0:
        print("VERDICT: PASS — graph FFN section == devnative ffn0/ffn2 (BIT-EQUAL)")
    else:
        print("VERDICT: FAIL — FFN section diverged")
