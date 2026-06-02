# models/pid/pid_ops_smoke.mojo — GPU smoke + parity for models/pid/pid_ops.mojo.
#
# Unit-gates the four PiD scalar primitives against a PyTorch reference (random
# seeded weights, tiny inputs) dumped via system python3 into
# parity/pid_basics_ref_data.mojo. F32 throughout.
#
# Gates (HARD RULE: real numeric on GPU, not compile-only):
#   1. patchify(x)              == F.unfold(x).transpose(1,2)   (bit-close)
#   2. unpatchify(tokens)       == F.fold(...)                  (bit-close)
#   2b. unpatchify(patchify(x)) == x                            (round-trip)
#   3. ntk_rope_tables_2d       cos/sin == real/imag(freqs_cis) (cos>=0.999)
#   4. timestep_embedding(t,..) == TimestepConditioner.timestep_embedding (mp=10)
#   5. timestep_conditioner     == full TimestepConditioner.forward
#   6. sigma_aware_gate         == SigmaAwareGatePerTokenPerDim.forward
#
# Gate: cos >= 0.999 AND each prints max_abs. Patch ops gate at cos==1 / tiny max.
#
# Run: pixi run mojo run -I . serenitymojo/models/pid/pid_ops_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.pid.pid_ops import (
    patchify, unpatchify, ntk_rope_tables_2d, timestep_embedding,
    timestep_conditioner, sigma_aware_gate,
)
from serenitymojo.models.pid.parity.pid_basics_ref_data import (
    UF_B, UF_C, UF_H, UF_W, UF_PS, UF_L, UF_TOK,
    uf_x, uf_tokens_ref, uf_fold_ref,
    ROPE_DIM, ROPE_REF, ROPE_RH, ROPE_RW, ROPE_L, ROPE_HALF,
    rope_cos_ref, rope_sin_ref,
    TS_N, TS_FREQ, TS_HIDDEN,
    ts_t, ts_freq_ref, ts_mlp0_w, ts_mlp0_b, ts_mlp2_w, ts_mlp2_b, ts_out_ref,
    GATE_D, GATE_B, GATE_N, GATE_LOG_ALPHA,
    gate_x, gate_lq, gate_sigma, gate_w, gate_b, gate_out_ref,
)


comptime F32 = STDtype.F32


def _report(label: String, r: ParityResult) -> Bool:
    var tag = "PASS" if r.passed else "FAIL"
    print(label, "  cos=", r.cos, "  max_abs=", r.max_abs, "  n=", r.n, "  [", tag, "]")
    return r.passed


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    print("=== PiD basics smoke (pid_ops.mojo) — F32 vs PyTorch ref (seed 7777) ===")

    # ── Gate 1: patchify == F.unfold ─────────────────────────────────────────
    var xt = Tensor.from_host(uf_x(), [UF_B, UF_C, UF_H, UF_W], F32, ctx)
    var tokens = patchify(xt, UF_PS, ctx)
    var r1 = h.compare(tokens, uf_tokens_ref(), ctx)
    all_pass = all_pass and _report("1 patchify==F.unfold          ", r1)

    # ── Gate 2: unpatchify == F.fold (feed the torch reference tokens) ───────
    var ref_tokens = Tensor.from_host(uf_tokens_ref(), [UF_B, UF_L, UF_TOK], F32, ctx)
    var pix = unpatchify(ref_tokens, UF_C, UF_H, UF_W, UF_PS, ctx)
    var r2 = h.compare(pix, uf_fold_ref(), ctx)
    all_pass = all_pass and _report("2 unpatchify==F.fold          ", r2)

    # ── Gate 2b: round-trip unpatchify(patchify(x)) == x ─────────────────────
    var rt = unpatchify(tokens, UF_C, UF_H, UF_W, UF_PS, ctx)
    var r2b = h.compare(rt, uf_x(), ctx)
    all_pass = all_pass and _report("2b unpatchify(patchify(x))==x ", r2b)

    # ── Gate 3: NTK 2D RoPE tables ───────────────────────────────────────────
    var tables = ntk_rope_tables_2d(
        ROPE_DIM, ROPE_RH, ROPE_RW, ROPE_REF, ROPE_REF, ctx
    )
    var r3c = h.compare(tables.cos, rope_cos_ref(), ctx)
    all_pass = all_pass and _report("3a ntk_rope cos==real(cis)    ", r3c)
    var r3s = h.compare(tables.sin, rope_sin_ref(), ctx)
    all_pass = all_pass and _report("3b ntk_rope sin==imag(cis)    ", r3s)

    # ── Gate 4: timestep_embedding (max_period=10) ───────────────────────────
    var tt = Tensor.from_host(ts_t(), [TS_N], F32, ctx)
    var ts_emb = timestep_embedding(tt, TS_FREQ, ctx, Float32(10.0))
    var r4 = h.compare(ts_emb, ts_freq_ref(), ctx)
    all_pass = all_pass and _report("4 timestep_embedding(mp=10)   ", r4)

    # ── Gate 5: full TimestepConditioner.forward ─────────────────────────────
    var w0 = Tensor.from_host(ts_mlp0_w(), [TS_HIDDEN, TS_FREQ], F32, ctx)
    var b0 = Tensor.from_host(ts_mlp0_b(), [TS_HIDDEN], F32, ctx)
    var w2 = Tensor.from_host(ts_mlp2_w(), [TS_HIDDEN, TS_HIDDEN], F32, ctx)
    var b2 = Tensor.from_host(ts_mlp2_b(), [TS_HIDDEN], F32, ctx)
    var tc_out = timestep_conditioner(tt, w0, b0, w2, b2, TS_FREQ, ctx, Float32(10.0))
    var r5 = h.compare(tc_out, ts_out_ref(), ctx)
    all_pass = all_pass and _report("5 TimestepConditioner.forward ", r5)

    # ── Gate 6: sigma-aware gate ─────────────────────────────────────────────
    var gx = Tensor.from_host(gate_x(), [GATE_B, GATE_N, GATE_D], F32, ctx)
    var gl = Tensor.from_host(gate_lq(), [GATE_B, GATE_N, GATE_D], F32, ctx)
    var gs = Tensor.from_host(gate_sigma(), [GATE_B], F32, ctx)
    var gw = Tensor.from_host(gate_w(), [GATE_D, 2 * GATE_D], F32, ctx)
    var gb = Tensor.from_host(gate_b(), [GATE_D], F32, ctx)
    var gate_out = sigma_aware_gate(gx, gl, gs, gw, gb, GATE_LOG_ALPHA, ctx)
    var r6 = h.compare(gate_out, gate_out_ref(), ctx)
    all_pass = all_pass and _report("6 SigmaAwareGate.forward      ", r6)

    print("============================================================")
    if all_pass:
        print("ALL GATES PASS")
    else:
        print("SOME GATES FAILED")
        raise Error("pid_ops smoke: gate failure")
