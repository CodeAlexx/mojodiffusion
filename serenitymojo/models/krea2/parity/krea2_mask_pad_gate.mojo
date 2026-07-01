# serenitymojo/models/krea2/parity/krea2_mask_pad_gate.mojo
#
# MASKED-PAD PARITY GATE for the Krea-2 SingleStreamBlock length-bucket
# FLASH-padmask (models/krea2/krea2_block.mojo). SELF-CONSISTENT (no torch
# oracle): proves the cuDNN flash-padmask isolates the pad tokens so the REAL-token
# grads are pad-length INVARIANT (a shorter pad buffer and a longer one give the
# same real-token grads), i.e. the [real_len:L] tail mask works in fwd AND bwd.
#
# ── what it proves ────────────────────────────────────────────────────────────
# Token layout is the trainer's reordered [TXT_real | IMG | TXT_pad], so the valid
# tokens are a CONTIGUOUS PREFIX [0:real_len] (real_len = LT_text + IMG) and the
# pad is the tail. Two flash-padmask runs of the SAME block weights + LoRA on the
# SAME real prefix, differing ONLY in how much pad follows:
#   (A) buffer LA, real_len, pad tail [real_len:LA].  Reference grads.
#   (B) buffer LB > LA, same real_len, longer pad tail [real_len:LB].
# The upstream d_out on the PAD tail rows is ZERO in BOTH (only real text + image
# rows carry loss grad — exactly the trainer, whose velocity loss is on the image
# tokens and whose final-layer backward zeros the txt+pad rows).
# Then on the real rows [0:real_len]:
#   • the forward out matches (DETERMINISTIC fwd) tight (cos>=0.999).
#   • every LoRA dA/dB and d_x matches (cos>=0.999) — FLASH so VALUE-TOLERANCE,
#     not bit-exact: cuDNN's flash dQ accumulation is NONDETERMINISTIC run-to-run
#     (atomics), so dQ-derived grads (dA on wq, d_x) drift ~few e-3 in max_abs but
#     stay cos>=0.999. Deterministic grads (out, dB, dV-derived) are tighter.
#
# Buffers MUST be 128-aligned (the cuDNN flash-padmask contract; LA=256, LB=384).
# RoPE: text positions are identity (pos 0), image positions a fixed per-token
# rotation IDENTICAL across both buffers (replicating krea2_build_pos: txt pos
# zero, img grid independent of the pad length) — so the only difference between A
# and B is the pad tail length, which is exactly what we test.
#
# Run (NO oracle needed — just build + run; ORCHESTRATOR runs the GPU smoke):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_mask_pad_gate.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
    krea2_single_stream_block_lora,
    krea2_single_stream_block_lora_backward,
)

comptime TArc = ArcPointer[Tensor]

# small dims, but Dh=128 (cuDNN flash requires it) and 128-aligned buffer lengths.
comptime HEADS = 8
comptime KVHEADS = 2
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM          # 1024
comptime HALF = HEADDIM // 2                  # 64
comptime MLPDIM = 256
comptime EPS = Float32(1e-5)
comptime RANK = 8
comptime LSCALE = Float32(16.0) / Float32(8.0)   # alpha/rank = 2.0

# token geometry: TXT_real then IMG then TXT_pad. Valid prefix = LT_TEXT + IMG.
comptime LT_TEXT = 40                         # real caption length
comptime IMG = 160                            # image tokens
comptime REAL = LT_TEXT + IMG                  # 200 — valid contiguous prefix
comptime LA = 256                             # run-A buffer (128-aligned, pad [200:256])
comptime LB = 384                             # run-B buffer (128-aligned, pad [200:384])


# ── random device tensor ───────────────────────────────────────────────────────
def _rand(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _adapter(in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext) raises -> Optional[LoraAdapterDevice]:
    var a = TArc(_rand(_shape2(RANK, in_f), seed, ctx))
    var b = TArc(_rand(_shape2(out_f, RANK), seed + 1, ctx))
    return Optional[LoraAdapterDevice](LoraAdapterDevice(a^, b^, RANK, in_f, out_f, LSCALE))


# Build per-token cos/sin tiled tables [L*heads, HALF]. Real text rows
# (token < LT_TEXT) = identity (cos=1, sin=0, position 0); image rows (the IMG
# block at [LT_TEXT : REAL]) = a fixed rotation cos=cos(theta_k), sin=sin(theta_k)
# where theta_k depends ONLY on the image-token index k (NOT the sequence
# position) — identical across the LA and LB tilings. Pad-tail rows (token >= REAL)
# = identity too (they are masked out, value irrelevant).
from std.math import cos as mcos, sin as msin
def _tile_table(
    L: Int, heads: Int, is_cos: Bool, ctx: DeviceContext
) raises -> Tensor:
    var out = List[Float32]()
    for tok in range(L):
        var k = tok - LT_TEXT           # image-token index (>=0 in [LT_TEXT:REAL])
        for _h in range(heads):
            for c in range(HALF):
                if tok < LT_TEXT or tok >= REAL:
                    out.append(Float32(1.0) if is_cos else Float32(0.0))  # identity
                else:
                    # fixed per-(image-token,channel) angle, identical in both buffers.
                    var ang = Float32(0.013) * Float32(k + 1) * Float32(c + 1)
                    out.append(mcos(ang) if is_cos else msin(ang))
    return Tensor.from_host(out^, _shape2(L * heads, HALF), STDtype.BF16, ctx)


# weights + lora shared by BOTH runs (same seeds → identical params).
def _make_weights(ctx: DeviceContext) raises -> Krea2BlockWeights:
    return Krea2BlockWeights(
        TArc(_rand(_shape2(HEADS * HEADDIM, FEATURES), 1, ctx)),
        TArc(_rand(_shape2(KVHEADS * HEADDIM, FEATURES), 2, ctx)),
        TArc(_rand(_shape2(KVHEADS * HEADDIM, FEATURES), 3, ctx)),
        TArc(_rand(_shape2(FEATURES, FEATURES), 4, ctx)),
        TArc(_rand(_shape2(FEATURES, FEATURES), 5, ctx)),
        TArc(_rand(_shape2(MLPDIM, FEATURES), 6, ctx)),
        TArc(_rand(_shape2(MLPDIM, FEATURES), 7, ctx)),
        TArc(_rand(_shape2(FEATURES, MLPDIM), 8, ctx)),
        TArc(_rand(_shape1(HEADDIM), 9, ctx)),
        TArc(_rand(_shape1(HEADDIM), 10, ctx)),
        TArc(_rand(_shape1(FEATURES), 11, ctx)),
        TArc(_rand(_shape1(FEATURES), 12, ctx)),
        TArc(_rand(_shape1(6 * FEATURES), 13, ctx)),
    )


def _make_lora(ctx: DeviceContext) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _adapter(FEATURES, HEADS * HEADDIM, 100, ctx),
        _adapter(FEATURES, KVHEADS * HEADDIM, 102, ctx),
        _adapter(FEATURES, KVHEADS * HEADDIM, 104, ctx),
        _adapter(FEATURES, FEATURES, 106, ctx),
        _adapter(FEATURES, FEATURES, 108, ctx),
        _adapter(FEATURES, MLPDIM, 110, ctx),
        _adapter(FEATURES, MLPDIM, 112, ctx),
        _adapter(MLPDIM, FEATURES, 114, ctx),
    )


# Build x_buf [1,L,F] from the real-prefix host buffer real_h [REAL*F]: rows
# [0:REAL] copied, the pad tail [REAL:L] filled with garbage. For d_out the pad
# tail is ZEROED (only real text + image rows carry loss grad).
def _scatter(real_h: List[Float32], L: Int, zero_pad: Bool) -> List[Float32]:
    var out = List[Float32]()
    for r in range(REAL):
        for c in range(FEATURES):
            out.append(real_h[r * FEATURES + c])
    for r in range(L - REAL):
        for c in range(FEATURES):
            if zero_pad:
                out.append(Float32(0.0))
            else:
                out.append(Float32(0.37) * Float32((r * 7 + c) % 11) - Float32(1.5))
    return out^


# pull the REAL rows [0:REAL] out of a [1,L,F] host buffer.
def _gather_real(buf_h: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for r in range(REAL):
        for c in range(FEATURES):
            out.append(buf_h[r * FEATURES + c])
    return out^


def _check(
    mut h: ParityHarness, name: String, a: List[Float32], b: List[Float32], mut allok: Bool
) raises:
    var r = h.compare_host(a, b)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n,
          "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _check_lora(
    mut h: ParityHarness, nm: String, refg: Krea2LoraGrad, gotg: Krea2LoraGrad, mut allok: Bool
) raises:
    if not refg.d_a or not gotg.d_a or not refg.d_b or not gotg.d_b:
        print("  ", nm, " MISSING grads — FAIL"); allok = False; return
    _check(h, nm + " dA", gotg.d_a.value(), refg.d_a.value(), allok)
    _check(h, nm + " dB", gotg.d_b.value(), refg.d_b.value(), allok)


# Run the flash-padmask block fwd+bwd on a buffer of length L with the given
# real-prefix x + d_out. Returns (out_real, dx_real, grads). RoPE tables are tiled
# per-L (the angles depend only on the token index, identical across L).
struct _Run(Movable):
    var out_real: List[Float32]
    var dx_real: List[Float32]
    var grads: Krea2BlockGradsHolder

    def __init__(out self, var out_real: List[Float32], var dx_real: List[Float32], var grads: Krea2BlockGradsHolder):
        self.out_real = out_real^
        self.dx_real = dx_real^
        self.grads = grads^


# small movable holder for the 8 LoRA grads we compare (Krea2BlockGrads is Movable
# but holds a TArc d_x we don't keep; copy the 8 Krea2LoraGrad which are Copyable).
struct Krea2BlockGradsHolder(Movable):
    var wq: Krea2LoraGrad
    var wk: Krea2LoraGrad
    var wv: Krea2LoraGrad
    var gate_w: Krea2LoraGrad
    var wo: Krea2LoraGrad
    var mlp_gate_w: Krea2LoraGrad
    var mlp_up_w: Krea2LoraGrad
    var mlp_down_w: Krea2LoraGrad

    def __init__(
        out self,
        var wq: Krea2LoraGrad, var wk: Krea2LoraGrad, var wv: Krea2LoraGrad,
        var gate_w: Krea2LoraGrad, var wo: Krea2LoraGrad,
        var mlp_gate_w: Krea2LoraGrad, var mlp_up_w: Krea2LoraGrad, var mlp_down_w: Krea2LoraGrad,
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


def _run_flash[L: Int](
    real_x_h: List[Float32], real_dout_h: List[Float32],
    w: Krea2BlockWeights, lora: Krea2BlockLora, vec: Tensor,
    ctx: DeviceContext,
) raises -> _Run:
    var x_h = _scatter(real_x_h, L, False)                    # pad-tail = garbage
    var x = Tensor.from_host(x_h, _shape3(1, L, FEATURES), STDtype.BF16, ctx)
    var dout_h = _scatter(real_dout_h, L, True)               # pad-tail d_out = ZERO
    var dout = Tensor.from_host(dout_h, _shape3(1, L, FEATURES), STDtype.BF16, ctx)

    var cos_q = _tile_table(L, HEADS, True, ctx)
    var sin_q = _tile_table(L, HEADS, False, ctx)
    var cos_k = _tile_table(L, KVHEADS, True, ctx)
    var sin_k = _tile_table(L, KVHEADS, False, ctx)
    var cos0 = Tensor.from_host(_zeros(L * HALF), _shape2(L, HALF), STDtype.BF16, ctx)
    var sin0 = cos0.clone(ctx)

    var rl = Optional[Int](REAL)             # the valid contiguous prefix length
    var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        TArc(x.clone(ctx)), vec, w, lora, cos0, sin0,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, rl,
    )
    var g = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        dout.clone(ctx), vec, w, lora, fwd.saved,
        cos_q, sin_q, cos_k, sin_k, EPS, ctx, rl,
    )
    var out_real = _gather_real(fwd.out[].to_host(ctx))
    var dx_real = _gather_real(g.d_x[].to_host(ctx))
    var holder = Krea2BlockGradsHolder(
        g.wq.copy(), g.wk.copy(), g.wv.copy(), g.gate_w.copy(),
        g.wo.copy(), g.mlp_gate_w.copy(), g.mlp_up_w.copy(), g.mlp_down_w.copy(),
    )
    return _Run(out_real^, dx_real^, holder^)


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_mask_pad_gate (FLASH padmask: real-token grads pad-length invariant) ====")
    print("LT_TEXT=", LT_TEXT, " IMG=", IMG, " REAL=", REAL, " LA=", LA, " LB=", LB)

    var w = _make_weights(ctx)
    var lora = _make_lora(ctx)
    var vec = _rand(_shape2(1, 6 * FEATURES), 50, ctx)

    # ── REAL prefix inputs (text + image), shared by both runs ──────────────────
    var x_real = _rand(_shape3(1, REAL, FEATURES), 60, ctx)
    var d_out_real = _rand(_shape3(1, REAL, FEATURES), 61, ctx)
    var x_real_h = x_real.to_host(ctx)
    var d_out_real_h = d_out_real.to_host(ctx)

    # ── (A) flash-padmask at LA, (B) flash-padmask at LB (longer pad tail) ──────
    var run_a = _run_flash[LA](x_real_h, d_out_real_h, w, lora, vec, ctx)
    var run_b = _run_flash[LB](x_real_h, d_out_real_h, w, lora, vec, ctx)

    var harness = ParityHarness()
    var allok = True

    # ── forward out on the real rows (DETERMINISTIC fwd) ────────────────────────
    print("")
    print("---- forward out on REAL rows: LB-pad vs LA-pad (deterministic) ----")
    _check(harness, "out(real rows)", run_b.out_real, run_a.out_real, allok)

    # ── LoRA dA/dB (all 8) — pad rows must contribute NOTHING ───────────────────
    print("")
    print("---- LoRA dA/dB: LB-pad vs LA-pad (must match; dQ-derived value-tol) ----")
    _check_lora(harness, "wq", run_a.grads.wq, run_b.grads.wq, allok)
    _check_lora(harness, "wk", run_a.grads.wk, run_b.grads.wk, allok)
    _check_lora(harness, "wv", run_a.grads.wv, run_b.grads.wv, allok)
    _check_lora(harness, "gate", run_a.grads.gate_w, run_b.grads.gate_w, allok)
    _check_lora(harness, "wo", run_a.grads.wo, run_b.grads.wo, allok)
    _check_lora(harness, "mlp_gate", run_a.grads.mlp_gate_w, run_b.grads.mlp_gate_w, allok)
    _check_lora(harness, "mlp_up", run_a.grads.mlp_up_w, run_b.grads.mlp_up_w, allok)
    _check_lora(harness, "mlp_down", run_a.grads.mlp_down_w, run_b.grads.mlp_down_w, allok)

    # ── d_x on the REAL rows (dQ-derived → flash value-tolerance) ───────────────
    print("")
    print("---- d_x on REAL rows: LB-pad vs LA-pad (dQ-derived value-tol) ----")
    _check(harness, "d_x(real rows)", run_b.dx_real, run_a.dx_real, allok)

    print("")
    if allok:
        print("VERDICT: PASS — flash padmask isolates pad tokens; real-token grads are",
              "pad-length invariant (cos>=0.999)")
    else:
        print("VERDICT: FAIL — real-token grads diverged across pad lengths (see above)")


def _zeros(n: Int) -> List[Float32]:
    var s = List[Float32]()
    for _ in range(n):
        s.append(Float32(0.0))
    return s^
