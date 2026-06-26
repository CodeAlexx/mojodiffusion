# autograd_v2/tests/krea2_block_parity.mojo — Phase 4b SAME-PROCESS BIT GATE
# (AUTOGRAD_V2_MOJO_DESIGN.md C14): the engine's per-block COARSE graph backward
# (krea2_single_stream_block_graph_backward) must equal the hand-chain
# krea2_single_stream_block_lora_backward BIT-for-BIT on a real-shaped synthetic block
# with NONZERO LoRA B (so d_A is non-degenerate — a degenerate compare must FAIL).
#
# real_len = None → FULL attention (math sdpa_nomask, DETERMINISTIC) so bit equality is
# achievable (flash dQ is nondeterministic run-to-run → NOT in this gate's path, per the
# task: "flash is not in this gate's block path"). Mirrors
# autograd_v2/tests/ideogram4_block_parity.mojo + klein_block_parity.mojo.
#
# Real krea2 trainer dims (train_krea2.mojo): HEADS=48 KVHEADS=12 HEADDIM=128
# features=6144. The composite node feeds its d_x into the engine x LEAF (the engine
# genuinely routes the single inter-block dependency, so this gate checks the engine
# path); the 8 host-list LoRA pairs are captured out-of-band and compared too.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/krea2_block_parity.mojo -o /tmp/krea2_block_parity
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib /tmp/krea2_block_parity

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add_scalar, mul_scalar, zeros_device
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.dit.krea2_dit import _tile_rope_table
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, lora_adapter_to_device,
)
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
    krea2_single_stream_block_lora,
    krea2_single_stream_block_lora_backward,
)
from serenitymojo.autograd_v2.krea2_block_graph import (
    krea2_single_stream_block_graph_backward,
)

comptime TArc = ArcPointer[Tensor]


def _s1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _wt(out_f: Int, in_f: Int, seed: UInt64, ctx: DeviceContext) raises -> TArc:
    # frozen base weight [out_f, in_f]. F32 = the krea2 F32 parity-gate convention
    # (krea2_block.mojo: the no-pad path is "BIT-IDENTICAL to the pre-flash block",
    # and every mixed-precision cast is "a no-op in the F32 gate"). The gate proves
    # engine == oracle on byte-identical inputs; F32 sidesteps the production bf16
    # mixed-precision cast subtleties (gate_residual_backward raises on dtype skew).
    return TArc(mul_scalar(
        randn(_s2(out_f, in_f), seed, STDtype.F32, ctx), Float32(0.04), ctx))


def _ones1(n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(add_scalar(zeros_device(_s1(n), STDtype.F32, ctx), Float32(1.0), ctx))


def _zeros1(n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(zeros_device(_s1(n), STDtype.F32, ctx))


def _arc(t: Tensor) raises -> TArc:
    return TArc(Tensor(t.buf.copy(), t.shape(), t.dtype()))


def _nonzero_b(var lo: LoraAdapterDevice, seed: UInt64, ctx: DeviceContext) raises -> LoraAdapterDevice:
    # overwrite B (zero from make_lora_adapter) with randn so d_A is non-degenerate;
    # cast A and B to F32 for the uniform-F32 gate (sidesteps mixed-precision skew).
    var a_f32 = cast_tensor(lo.a[], STDtype.F32, ctx)
    var b = mul_scalar(randn(lo.b[].shape(), seed, STDtype.F32, ctx), Float32(0.04), ctx)
    return LoraAdapterDevice(TArc(a_f32^), TArc(b^), lo.rank, lo.in_f, lo.out_f, lo.scale)


def _mk_lora(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext
) raises -> Optional[LoraAdapterDevice]:
    var host = make_lora_adapter(rank, alpha, in_f, out_f, seed)
    var dev = lora_adapter_to_device(host, ctx)
    return Optional[LoraAdapterDevice](_nonzero_b(dev^, seed + UInt64(9000), ctx))


def _diff_t(name: String, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Int:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var nm = 0
    var nz = 0
    for i in range(len(ah)):
        if ah[i] != bh[i]:
            nm += 1
        if ah[i] != Float32(0.0):
            nz += 1
    print("  ", name, "n_mismatch", nm, "/", len(ah), " (nonzero_in_oracle=", nz, ")")
    if nz == 0:
        raise Error(String("DEGENERATE compare (all-zero oracle tensor): ") + name)
    return nm


def _diff_list(name: String, a: List[Float32], b: List[Float32]) raises -> Int:
    if len(a) != len(b):
        raise Error(String("length mismatch ") + name)
    var nm = 0
    var nz = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            nm += 1
        if a[i] != Float32(0.0):
            nz += 1
    print("  ", name, "n_mismatch", nm, "/", len(a), " (nonzero_in_oracle=", nz, ")")
    if nz == 0:
        raise Error(String("DEGENERATE compare (all-zero oracle list): ") + name)
    return nm


def _cmp_pair(
    name: String, h: Krea2LoraGrad, e: Krea2LoraGrad
) raises -> Int:
    if not h.d_a or not h.d_b or not e.d_a or not e.d_b:
        raise Error(String("missing LoRA grad pair: ") + name)
    var t = 0
    t += _diff_list(name + ".d_a", h.d_a.value(), e.d_a.value())
    t += _diff_list(name + ".d_b", h.d_b.value(), e.d_b.value())
    return t


def main() raises:
    var ctx = DeviceContext()
    # Real krea2 trainer dims (smaller L for a fast same-process gate; the block math is
    # L-agnostic, so L=512 exercises the SAME kernels/folds as L=4864 with NONZERO pad
    # only changing flash, which this no-pad gate excludes by design).
    comptime HEADS = 48
    comptime KVHEADS = 12
    comptime HEADDIM = 128
    comptime FEATURES = HEADS * HEADDIM   # 6144
    comptime MLPDIM = 16384
    comptime L = 512
    var eps = Float32(1.0e-5)
    var rank = 16
    var alpha = Float32(16.0)

    # frozen base weights (bf16) + raw norm scales (F32) + mod_lin (F32 [6F]).
    var w = Krea2BlockWeights(
        _wt(HEADS * HEADDIM, FEATURES, UInt64(1), ctx),       # wq
        _wt(KVHEADS * HEADDIM, FEATURES, UInt64(2), ctx),     # wk
        _wt(KVHEADS * HEADDIM, FEATURES, UInt64(3), ctx),     # wv
        _wt(FEATURES, FEATURES, UInt64(4), ctx),              # gate_w
        _wt(FEATURES, FEATURES, UInt64(5), ctx),              # wo
        _wt(MLPDIM, FEATURES, UInt64(6), ctx),                # mlp_gate_w
        _wt(MLPDIM, FEATURES, UInt64(7), ctx),                # mlp_up_w
        _wt(FEATURES, MLPDIM, UInt64(8), ctx),                # mlp_down_w
        _zeros1(HEADDIM, ctx), _zeros1(HEADDIM, ctx),         # qnorm/knorm raw (scale+1)
        _zeros1(FEATURES, ctx), _zeros1(FEATURES, ctx),       # prenorm/postnorm raw
        TArc(mul_scalar(randn(_s1(6 * FEATURES), UInt64(20), STDtype.F32, ctx), Float32(0.1), ctx)),  # mod_lin
    )

    # 8 LoRA adapters, NONZERO B, matching the block's 8 Linears (in/out per krea2 dims).
    var lora = Krea2BlockLora(
        _mk_lora(rank, alpha, FEATURES, HEADS * HEADDIM, UInt64(100), ctx),    # wq
        _mk_lora(rank, alpha, FEATURES, KVHEADS * HEADDIM, UInt64(101), ctx),  # wk
        _mk_lora(rank, alpha, FEATURES, KVHEADS * HEADDIM, UInt64(102), ctx),  # wv
        _mk_lora(rank, alpha, FEATURES, FEATURES, UInt64(103), ctx),           # gate_w
        _mk_lora(rank, alpha, FEATURES, FEATURES, UInt64(104), ctx),           # wo
        _mk_lora(rank, alpha, FEATURES, MLPDIM, UInt64(105), ctx),             # mlp_gate_w
        _mk_lora(rank, alpha, FEATURES, MLPDIM, UInt64(106), ctx),             # mlp_up_w
        _mk_lora(rank, alpha, MLPDIM, FEATURES, UInt64(107), ctx),             # mlp_down_w
    )

    # vec [1, 6F], block input x [1,L,features], d_out — all F32 (the F32 parity-gate
    # convention; production runs bf16 acts but the gate proves engine == oracle on
    # byte-identical F32 inputs, which is dtype-agnostic for the bit-equality claim).
    var vec = mul_scalar(randn(_s2(1, 6 * FEATURES), UInt64(30), STDtype.F32, ctx), Float32(0.1), ctx)
    var x = mul_scalar(randn([1, L, FEATURES], UInt64(40), STDtype.F32, ctx), Float32(0.5), ctx)
    var d_out = mul_scalar(randn([1, L, FEATURES], UInt64(50), STDtype.F32, ctx), Float32(0.5), ctx)

    # rope tables: identity (cos=1, sin=0) per-token [L, HEADDIM/2] + tiled for q/k.
    var cos = add_scalar(zeros_device(_s2(L, HEADDIM // 2), STDtype.F32, ctx), Float32(1.0), ctx)
    var sin = zeros_device(_s2(L, HEADDIM // 2), STDtype.F32, ctx)
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var rl = Optional[Int](None)  # FULL attention (math sdpa, deterministic)

    # ── HAND-CHAIN ORACLE (recompute fwd → saved → block backward) ──────────────
    var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        TArc(Tensor(x.buf.copy(), x.shape(), x.dtype())), vec, w, lora,
        cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, rl,
    )
    var hand = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_out, vec, w, lora, rb.saved,
        cos_q, sin_q, cos_k, sin_k, eps, ctx, rl,
    )

    # ── ENGINE (coarse graph) ───────────────────────────────────────────────────
    var scratch = ScratchRingAllocator(ctx, 64 * 1024 * 1024, 2)
    var eng = krea2_single_stream_block_graph_backward[L, HEADS, KVHEADS, HEADDIM](
        _arc(d_out), _arc(x), vec, w, lora,
        cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, scratch, rl,
    )

    # ── BIT compare (d_x + 8 LoRA dA/dB) ────────────────────────────────────────
    var total = 0
    total += _diff_t("d_x", hand.d_x[], eng.d_x[], ctx)
    total += _cmp_pair("wq", hand.wq, eng.wq)
    total += _cmp_pair("wk", hand.wk, eng.wk)
    total += _cmp_pair("wv", hand.wv, eng.wv)
    total += _cmp_pair("gate_w", hand.gate_w, eng.gate_w)
    total += _cmp_pair("wo", hand.wo, eng.wo)
    total += _cmp_pair("mlp_gate_w", hand.mlp_gate_w, eng.mlp_gate_w)
    total += _cmp_pair("mlp_up_w", hand.mlp_up_w, eng.mlp_up_w)
    total += _cmp_pair("mlp_down_w", hand.mlp_down_w, eng.mlp_down_w)

    if total == 0:
        print("KREA2 BLOCK PARITY PASS (engine == hand-chain, bit-equal)")
    else:
        raise Error(String("KREA2 BLOCK PARITY FAIL: total mismatches ") + String(total))
