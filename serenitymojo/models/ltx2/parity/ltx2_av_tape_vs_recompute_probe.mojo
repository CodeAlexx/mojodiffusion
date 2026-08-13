# serenitymojo/models/ltx2/parity/ltx2_av_tape_vs_recompute_probe.mojo
#
# P6.2 AV activation-tape GATE 0: measure, on THIS box, which of the two
# candidate fixes for the AV-arm OOM is cheaper per block —
#
#   (A) TAPE      = offload_to_host_fast (D2H) + restore_to_device_fast (H2D)
#   (B) RECOMPUTE = one more ltx2_block_forward_av_train at the real geometry
#
# at the TRAINER geometry S_V=576 / S_A=16 / N_TXT=1024 (train_ltx2_av.mojo).
# Reports (A) both for the contract's single 401.6MB F32 tensor AND for the
# realistic ~70-tensor-per-block acts set (per-call staging overhead is real),
# and (B) as the median of >=5 reps after a warmup.
#
# Measurement only — allocates ONE block's acts, never 48. No training run.
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_tape_vs_recompute_probe.mojo \
#   -o /tmp/ltx2_av_tape_probe && /tmp/ltx2_av_tape_probe

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import env_or, serenity_checkpoint
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.training.checkpoint import (
    HostOffload, offload_to_host_fast, restore_to_device_fast,
)
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_av_backward import (
    ltx2_block_forward_av_train, LTX2AVBlockActs, AVAttnActs,
)
from serenitymojo.models.ltx2.ltx2_av_stack import ltx2_av_build_head

comptime CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime S_V = 576
comptime S_A = 16
comptime N_TXT = 1024
comptime VD = 4096
comptime AD = 2048
comptime SIGMA = Float32(0.7)
comptime EPS = Float32(1e-6)
comptime REPS = 5


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


# deterministic non-degenerate filler (no denormals, no NaN) -------------------
def _filled(n: Int, seed: Int) -> List[Float32]:
    var v = List[Float32]()
    v.resize(n, Float32(0.0))
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 40) & UInt64(0xFFFF)) / Float32(65536.0)
        v[i] = (u - Float32(0.5)) * Float32(0.2)
    return v^


def _t(n: Int, var shape: List[Int], seed: Int, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_filled(n, seed), shape^, STDtype.F32, ctx)


def _median(var v: List[Float64]) -> Float64:
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    return v[len(v) // 2]


def _fmt(v: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0: s += ", "
        s += String(v[i])
    return s + "]"


# ── acts walkers: flatten every Tensor in the block acts, in field order ──────
def _attn_bytes(a: AVAttnActs) -> Int:
    return (a.q_src.nbytes() + a.kv_src.nbytes() + a.q_pre.nbytes()
            + a.k_pre.nbytes() + a.v4.nbytes() + a.q_sd.nbytes()
            + a.k_sd.nbytes() + a.att_flat.nbytes() + a.gl.nbytes()
            + a.gates.nbytes() + a.att_g.nbytes() + a.out.nbytes())


def _acts_bytes(a: LTX2AVBlockActs) -> Int:
    var t = (a.hidden.nbytes() + a.ahs.nbytes() + a.hs1.nbytes()
             + a.ahss1.nbytes() + a.hs2.nbytes() + a.ahss2.nbytes()
             + a.hs3.nbytes() + a.ahss3.nbytes() + a.h1_v.nbytes()
             + a.h1_a.nbytes())
    t += _attn_bytes(a.at1) + _attn_bytes(a.aat1) + _attn_bytes(a.at2)
    t += _attn_bytes(a.aat2) + _attn_bytes(a.a2v) + _attn_bytes(a.v2a)
    return t


def _acts_elems(a: LTX2AVBlockActs) -> Int:
    return _acts_bytes(a) // 4


def _off_attn(
    a: AVAttnActs, mut out: List[HostOffload], ctx: DeviceContext
) raises:
    out.append(offload_to_host_fast(a.q_src, ctx))
    out.append(offload_to_host_fast(a.kv_src, ctx))
    out.append(offload_to_host_fast(a.q_pre, ctx))
    out.append(offload_to_host_fast(a.k_pre, ctx))
    out.append(offload_to_host_fast(a.v4, ctx))
    out.append(offload_to_host_fast(a.q_sd, ctx))
    out.append(offload_to_host_fast(a.k_sd, ctx))
    out.append(offload_to_host_fast(a.att_flat, ctx))
    out.append(offload_to_host_fast(a.gl, ctx))
    out.append(offload_to_host_fast(a.gates, ctx))
    out.append(offload_to_host_fast(a.att_g, ctx))
    out.append(offload_to_host_fast(a.out, ctx))


def _off_acts(
    a: LTX2AVBlockActs, ctx: DeviceContext
) raises -> List[HostOffload]:
    var out = List[HostOffload]()
    out.append(offload_to_host_fast(a.hidden, ctx))
    out.append(offload_to_host_fast(a.ahs, ctx))
    _off_attn(a.at1, out, ctx)
    _off_attn(a.aat1, out, ctx)
    out.append(offload_to_host_fast(a.hs1, ctx))
    out.append(offload_to_host_fast(a.ahss1, ctx))
    _off_attn(a.at2, out, ctx)
    _off_attn(a.aat2, out, ctx)
    out.append(offload_to_host_fast(a.hs2, ctx))
    out.append(offload_to_host_fast(a.ahss2, ctx))
    _off_attn(a.a2v, out, ctx)
    _off_attn(a.v2a, out, ctx)
    out.append(offload_to_host_fast(a.hs3, ctx))
    out.append(offload_to_host_fast(a.ahss3, ctx))
    out.append(offload_to_host_fast(a.h1_v, ctx))
    out.append(offload_to_host_fast(a.h1_a, ctx))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt = env_or("LTX2_AV_CKPT", serenity_checkpoint(String(CKPT_NAME)))
    print("=== GATE 0: LTX2 AV tape (D2H+H2D) vs RECOMPUTE, per block ===")
    print("  geometry S_V=", S_V, " S_A=", S_A, " N_TXT=", N_TXT,
          " VD=", VD, " AD=", AD, " F32,  reps=", REPS)

    # ── inputs at the trainer geometry (values are filler; timing is data-free) ─
    var st = ShardedSafeTensors.open(ckpt)
    var head = ltx2_av_build_head[S_V, S_A, N_TXT](st, SIGMA, ctx)
    var hidden = _t(S_V * VD, _sh3(1, S_V, VD), 1, ctx)
    var ahs = _t(S_A * AD, _sh3(1, S_A, AD), 2, ctx)
    var enc = _t(N_TXT * VD, _sh3(1, N_TXT, VD), 3, ctx)
    var aenc = _t(N_TXT * AD, _sh3(1, N_TXT, AD), 4, ctx)

    # rope carriers: [S*H, hrd] (see ltx2_av_stack_smoke._load_rope)
    var v_cos = _t(S_V * 32 * 64, _sh2(S_V * 32, 64), 5, ctx)
    var v_sin = _t(S_V * 32 * 64, _sh2(S_V * 32, 64), 6, ctx)
    var a_cos = _t(S_A * 32 * 32, _sh2(S_A * 32, 32), 7, ctx)
    var a_sin = _t(S_A * 32 * 32, _sh2(S_A * 32, 32), 8, ctx)
    var ca_v_cos = _t(S_V * 32 * 32, _sh2(S_V * 32, 32), 9, ctx)
    var ca_v_sin = _t(S_V * 32 * 32, _sh2(S_V * 32, 32), 10, ctx)
    var ca_a_cos = _t(S_A * 32 * 32, _sh2(S_A * 32, 32), 11, ctx)
    var ca_a_sin = _t(S_A * 32 * 32, _sh2(S_A * 32, 32), 12, ctx)

    var w = LTX2AVBlockWeights.load(ckpt, 0, cfg, ctx).to_f32(ctx)
    print("  block-0 weights loaded (F32)")

    # ── (B) RECOMPUTE: one ltx2_block_forward_av_train, warmup + REPS ─────────
    var warm = ltx2_block_forward_av_train[S_V, S_A, N_TXT](
        w, hidden, ahs, enc, aenc,
        head.v_temb, head.a_temb, head.v_ca_ss, head.a_ca_ss,
        head.v_ca_gate, head.a_ca_gate, head.v_prompt_ts, head.a_prompt_ts,
        v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
        EPS, ctx,
    )
    ctx.synchronize()
    var acts_bytes = _acts_bytes(warm.acts)
    var acts_elems = _acts_elems(warm.acts)
    print("  ONE block acts: elems=", acts_elems, " bytes=", acts_bytes,
          " (=", Float64(acts_bytes) / 1048576.0, "MiB F32)")

    var fwd_ms = List[Float64]()
    for r in range(REPS):
        var t0 = perf_counter_ns()
        var f = ltx2_block_forward_av_train[S_V, S_A, N_TXT](
            w, hidden, ahs, enc, aenc,
            head.v_temb, head.a_temb, head.v_ca_ss, head.a_ca_ss,
            head.v_ca_gate, head.a_ca_gate, head.v_prompt_ts, head.a_prompt_ts,
            v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
            EPS, ctx,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        fwd_ms.append(Float64(t1 - t0) / 1.0e6)
        _ = f^
        _ = r
    var recompute_ms = _median(fwd_ms.copy())
    print("  (B) RECOMPUTE per block: reps_ms=", _fmt(fwd_ms),
          " MEDIAN=", recompute_ms, "ms")

    # ── (A1) contract form: ONE F32 tensor of the same total bytes ────────────
    var big = zeros_device(_sh2(acts_elems, 1), STDtype.F32, ctx)
    ctx.synchronize()
    print("  (A1) single tensor nbytes=", big.nbytes())
    var w_off = offload_to_host_fast(big, ctx)          # warmup D2H
    var w_res = restore_to_device_fast(w_off, ctx)      # warmup H2D
    _ = w_res^
    _ = w_off^

    var d2h1 = List[Float64]()
    var h2d1 = List[Float64]()
    for r in range(REPS):
        var t0 = perf_counter_ns()
        var o = offload_to_host_fast(big, ctx)
        var t1 = perf_counter_ns()
        var d = restore_to_device_fast(o, ctx)
        var t2 = perf_counter_ns()
        d2h1.append(Float64(t1 - t0) / 1.0e6)
        h2d1.append(Float64(t2 - t1) / 1.0e6)
        _ = d^
        _ = o^
        _ = r
    var d2h1_ms = _median(d2h1.copy())
    var h2d1_ms = _median(h2d1.copy())
    print("  (A1) SINGLE-TENSOR D2H reps_ms=", _fmt(d2h1), " MEDIAN=", d2h1_ms, "ms")
    print("  (A1) SINGLE-TENSOR H2D reps_ms=", _fmt(h2d1), " MEDIAN=", h2d1_ms, "ms")
    _ = big^

    # ── (A2) realistic form: the actual ~70-tensor acts set ──────────────────
    var w2 = _off_acts(warm.acts, ctx)                  # warmup
    var n_tensors = len(w2)
    _ = w2^
    var d2h2 = List[Float64]()
    var h2d2 = List[Float64]()
    for r in range(REPS):
        var t0 = perf_counter_ns()
        var offs = _off_acts(warm.acts, ctx)
        var t1 = perf_counter_ns()
        var back = List[ArcPointer[Tensor]]()
        for i in range(len(offs)):
            back.append(ArcPointer[Tensor](restore_to_device_fast(offs[i], ctx)))
        var t2 = perf_counter_ns()
        d2h2.append(Float64(t1 - t0) / 1.0e6)
        h2d2.append(Float64(t2 - t1) / 1.0e6)
        _ = back^
        _ = offs^
        _ = r
    var d2h2_ms = _median(d2h2.copy())
    var h2d2_ms = _median(h2d2.copy())
    print("  (A2) ACTS-SET (", n_tensors, "tensors) D2H reps_ms=", _fmt(d2h2),
          " MEDIAN=", d2h2_ms, "ms")
    print("  (A2) ACTS-SET (", n_tensors, "tensors) H2D reps_ms=", _fmt(h2d2),
          " MEDIAN=", h2d2_ms, "ms")

    # ── decision rule (applied to the contract's A1 number) ──────────────────
    var tape1 = d2h1_ms + h2d1_ms
    var tape2 = d2h2_ms + h2d2_ms
    print("")
    print("  tape_A1(D2H+H2D) =", tape1, "ms   recompute =", recompute_ms, "ms")
    print("  tape_A2(D2H+H2D) =", tape2, "ms   (realistic per-block tape cost)")
    print("  rule: TAPE if tape < recompute*0.77 =", recompute_ms * 0.77,
          "; RECOMPUTE if recompute < tape*0.77 =", tape1 * 0.77)
    if tape1 < recompute_ms * 0.77:
        print("  DECISION(A1): BUILD THE TAPE (tape strictly cheaper)")
    elif recompute_ms < tape1 * 0.77:
        print("  DECISION(A1): RECOMPUTE WINS -> STOP AND REPORT")
    else:
        print("  DECISION(A1): WITHIN 30% -> BUILD THE TAPE (tie-break)")
    print("")
    print("  projected device-resident acts, 48 blocks:")
    print("    today  =", Float64(acts_bytes) * 48.0 / 1.073741824e9, "GiB (all blocks retained)")
    print("    tape   =", Float64(acts_bytes) / 1.073741824e9, "GiB (one block at a time)")
    _ = warm^
    print("ltx2_av_tape_vs_recompute_probe DONE")
