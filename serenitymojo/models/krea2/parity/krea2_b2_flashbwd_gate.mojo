# serenitymojo/models/krea2/parity/krea2_b2_flashbwd_gate.mojo
#
# CONVICTION TEST (team-lead): flash BACKWARD on SLICES vs STANDALONE. My earlier op
# test proved the flash FORWARD bit-identical on slices; the BACKWARD was never tested.
# The b2 block backward calls sdpa_flash_backward_padmask_bf16 per half on slices of a
# concatenated [1,2L,H,Dh] tape, TWICE per block (half0 then half1). cuDNN flash-bwd
# workspace/graph is cache-keyed + shape-sensitive (MJ-1031). If sliced-bwd diverges
# from standalone-bwd BEYOND the standalone-vs-standalone noise floor → convicted.
# Real-ish dims (H=48, Dh=128, L=1408) and real_len<L (padmask arm).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_padmask_bf16, sdpa_flash_backward_padmask_bf16,
)

comptime TArc = ArcPointer[Tensor]
comptime H = 48
comptime DH = 128
comptime L = 1408            # 11*128
comptime RL0 = 1198
comptime RL1 = 1170


def _r(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)
def _s4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0)
    for i in range(n):
        dot += Float64(a[i]) * Float64(b[i]); na += Float64(a[i]) * Float64(a[i]); nb += Float64(b[i]) * Float64(b[i])
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_b2_flashbwd_gate (flash BACKWARD: sliced vs standalone, H=", H, " L=", L, ") ====")
    var scale = Float32(1.0) / sqrt(Float32(DH))

    # two samples' q/k/v/dA (BSHD [1,L,H,Dh] bf16)
    var q0 = _r(_s4(1, L, H, DH), 1, ctx); var k0 = _r(_s4(1, L, H, DH), 2, ctx); var v0 = _r(_s4(1, L, H, DH), 3, ctx)
    var q1 = _r(_s4(1, L, H, DH), 4, ctx); var k1 = _r(_s4(1, L, H, DH), 5, ctx); var v1 = _r(_s4(1, L, H, DH), 6, ctx)
    var dA0 = _r(_s4(1, L, H, DH), 7, ctx); var dA1 = _r(_s4(1, L, H, DH), 8, ctx)

    # per-half forward (the b2 block runs the flash fwd per half → concats the saved tape)
    var f0 = sdpa_flash_train_fwd_padmask_bf16[1, L, H, DH](q0, k0, v0, RL0, scale, ctx)
    var f1 = sdpa_flash_train_fwd_padmask_bf16[1, L, H, DH](q1, k1, v1, RL1, scale, ctx)

    # ── STANDALONE backward (per half, separately) ──────────────────────────────
    var b0a = sdpa_flash_backward_padmask_bf16[1, L, H, DH](
        f0.q_bf.copy(), f0.k_bf.copy(), f0.v_bf.copy(), f0.o_bf.copy(), f0.stats.copy(), dA0.clone(ctx), RL0, scale, ctx)
    var b0b = sdpa_flash_backward_padmask_bf16[1, L, H, DH](
        f0.q_bf.copy(), f0.k_bf.copy(), f0.v_bf.copy(), f0.o_bf.copy(), f0.stats.copy(), dA0.clone(ctx), RL0, scale, ctx)
    var b1a = sdpa_flash_backward_padmask_bf16[1, L, H, DH](
        f1.q_bf.copy(), f1.k_bf.copy(), f1.v_bf.copy(), f1.o_bf.copy(), f1.stats.copy(), dA1.clone(ctx), RL1, scale, ctx)

    # ── SLICED backward (EXACT b2 block pattern): concat the tape, slice per half, call
    # fb0 THEN fb1 in sequence (same shape twice — the cache-collision suspect) ──────
    var qbf = concat(1, ctx, f0.q_bf[], f1.q_bf[])
    var kbf = concat(1, ctx, f0.k_bf[], f1.k_bf[])
    var vbf = concat(1, ctx, f0.v_bf[], f1.v_bf[])
    var obf = concat(1, ctx, f0.o_bf[], f1.o_bf[])
    var stt = concat(2, ctx, f0.stats[], f1.stats[])       # [1,H,2L,1]
    var dAs = concat(1, ctx, dA0, dA1)
    var fb0 = sdpa_flash_backward_padmask_bf16[1, L, H, DH](
        TArc(slice(qbf, 1, 0, L, ctx)), TArc(slice(kbf, 1, 0, L, ctx)),
        TArc(slice(vbf, 1, 0, L, ctx)), TArc(slice(obf, 1, 0, L, ctx)),
        TArc(slice(stt, 2, 0, L, ctx)), slice(dAs, 1, 0, L, ctx), RL0, scale, ctx)
    var fb1 = sdpa_flash_backward_padmask_bf16[1, L, H, DH](
        TArc(slice(qbf, 1, L, L, ctx)), TArc(slice(kbf, 1, L, L, ctx)),
        TArc(slice(vbf, 1, L, L, ctx)), TArc(slice(obf, 1, L, L, ctx)),
        TArc(slice(stt, 2, L, L, ctx)), slice(dAs, 1, L, L, ctx), RL1, scale, ctx)

    # host
    var h0a_q = b0a.d_q.to_host(ctx); var h0a_k = b0a.d_k.to_host(ctx); var h0a_v = b0a.d_v.to_host(ctx)
    var h0b_q = b0b.d_q.to_host(ctx)
    var h1a_q = b1a.d_q.to_host(ctx); var h1a_k = b1a.d_k.to_host(ctx); var h1a_v = b1a.d_v.to_host(ctx)
    var f0_q = fb0.d_q.to_host(ctx); var f0_k = fb0.d_k.to_host(ctx); var f0_v = fb0.d_v.to_host(ctx)
    var f1_q = fb1.d_q.to_host(ctx); var f1_k = fb1.d_k.to_host(ctx); var f1_v = fb1.d_v.to_host(ctx)

    print("")
    print("  NOISE FLOOR  standalone dQ run-A vs run-B: cos =", _cos(h0a_q, h0b_q), "  (flash dQ is nondeterministic)")
    print("")
    print("  half0 sliced vs standalone:  dQ cos =", _cos(f0_q, h0a_q), "   dK cos =", _cos(f0_k, h0a_k), "   dV cos =", _cos(f0_v, h0a_v))
    print("  half1 sliced vs standalone:  dQ cos =", _cos(f1_q, h1a_q), "   dK cos =", _cos(f1_k, h1a_k), "   dV cos =", _cos(f1_v, h1a_v))
    print("")
    var floor = _cos(h0a_q, h0b_q)
    var worst = _cos(f1_q, h1a_q)
    if _cos(f0_q, h0a_q) < worst:
        worst = _cos(f0_q, h0a_q)
    print("  VERDICT:", "CONVICTED — sliced flash-bwd diverges below the noise floor" if worst < floor - Float64(0.005) else "not convicted — sliced matches standalone within the flash noise floor")
