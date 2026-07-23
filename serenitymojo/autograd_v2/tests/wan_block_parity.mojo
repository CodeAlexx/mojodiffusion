# autograd_v2/tests/wan_block_parity.mojo - Phase 1 SAME-PROCESS BIT GATE for the
# Wan2.2-A14B T2V per-block LoRA backward driven by the graph engine
# (AUTOGRAD_V2_MOJO_DESIGN.md P7 rollout; C14 bit-level oracle). Wan is MATH-MODE
# sdpa (sdpa_backward / sdpa_backward_rect — deterministic), so unlike Klein this
# admits a TRUE bit gate: the graph apply arm calls the SAME wan22_block_lora_
# backward oracle on the SAME saved WanSaved, so every returned grad must be
# BYTE-IDENTICAL. ANY mismatch here is a WIRING BUG in the graph path.
#
# Loads NOTHING from disk: synthetic-but-real-shaped inputs at real head geometry
# (Dh=128, the actual Wan2.2 head dim; interleaved RoPE, square + rect sdpa, gelu,
# rms/layer norm — every block code path exercised) with a lean H/S/ffn to bound
# memory/time. Deterministic LCG host patterns; LoRA A AND B are NONZERO so every
# d_A is non-degenerate (B=0 would gate d_A vacuously). A degenerate (all-zero)
# compared tensor FAILS the gate; a deliberately-corrupted tensor FAILS the gate
# (the teeth proof at the end).
#
# Per block, runs on IDENTICAL tensors in the SAME process:
#   oracle = wan22_block_lora_forward + wan22_block_lora_backward
#            (models/wan22/wan22_block.mojo — the trainer's per-block pair)
#   graph  = wan22_block_lora_graph_backward (autograd_v2 Phase 1)
# and compares base d_x + d_context + all 20 LoRA d_A/d_B host lists BIT-EQUAL.
# Prints one GATE line per tensor with n_mismatch; bar is n_mismatch=0 on every
# line.
#
# Build: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/wan_block_parity.mojo -o /tmp/wan_block_parity
# Run:   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#   /tmp/wan_block_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockLora, WanBlockLoraGrads,
    wan22_block_lora_forward, wan22_block_lora_backward,
)
from serenitymojo.autograd_v2.wan_block_graph import wan22_block_lora_graph_backward


# Real head geometry (Dh=128 = the actual Wan2.2 head dim); lean H/S/ffn to bound
# memory/time — the wiring bit-equality is shape-independent.
comptime H = 16
comptime Dh = 128
comptime DIM = H * Dh          # 2048
comptime S = 32
comptime TXT = 16
comptime FFN = 8192
comptime RANK = 16
comptime EPS = Float32(1.0e-6)
comptime LSCALE = Float32(1.0)   # alpha/rank = 16/16


def _pattern(n: Int, seed: UInt64, amp: Float32) -> List[Float32]:
    """Deterministic host pattern (the klein_block_parity.mojo LCG shape):
    uniform in (-amp, amp), fully reproducible from the seed."""
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * amp)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _make_adapter(in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    # A AND B nonzero (B=0 would make every d_A identically zero — a vacuous gate).
    return LoraAdapter(
        _pattern(RANK * in_f, seed, Float32(0.02)),
        _pattern(out_f * RANK, seed ^ 0x9E3779B97F4A7C15, Float32(0.02)),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _adapter(in_f: Int, out_f: Int, seed: UInt64) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](_make_adapter(in_f, out_f, seed))


def _load_weights(ctx: DeviceContext) raises -> WanBlockWeights:
    return WanBlockWeights(
        _pattern(DIM * DIM, 100, Float32(0.02)), _pattern(DIM * DIM, 101, Float32(0.02)),
        _pattern(DIM * DIM, 102, Float32(0.02)), _pattern(DIM * DIM, 103, Float32(0.02)),
        _pattern(DIM, 104, Float32(0.05)), _pattern(DIM, 105, Float32(0.05)),
        _pattern(DIM, 106, Float32(0.05)), _pattern(DIM, 107, Float32(0.05)),
        _pattern(DIM, 108, Float32(0.5)), _pattern(DIM, 109, Float32(0.5)),
        _pattern(DIM * DIM, 110, Float32(0.02)), _pattern(DIM * DIM, 111, Float32(0.02)),
        _pattern(DIM * DIM, 112, Float32(0.02)), _pattern(DIM * DIM, 113, Float32(0.02)),
        _pattern(DIM, 114, Float32(0.05)), _pattern(DIM, 115, Float32(0.05)),
        _pattern(DIM, 116, Float32(0.05)), _pattern(DIM, 117, Float32(0.05)),
        _pattern(DIM, 118, Float32(0.5)), _pattern(DIM, 119, Float32(0.5)),
        _pattern(DIM, 120, Float32(0.5)), _pattern(DIM, 121, Float32(0.05)),
        _pattern(FFN * DIM, 122, Float32(0.02)), _pattern(FFN, 123, Float32(0.05)),
        _pattern(DIM * FFN, 124, Float32(0.02)), _pattern(DIM, 125, Float32(0.05)),
        DIM, FFN, Dh, ctx,
    )


def _load_mod() raises -> WanModVecs:
    # per-token AdaLN vectors [S,dim].
    return WanModVecs(
        _pattern(S * DIM, 200, Float32(0.1)), _pattern(S * DIM, 201, Float32(0.1)),
        _pattern(S * DIM, 202, Float32(0.5)), _pattern(S * DIM, 203, Float32(0.1)),
        _pattern(S * DIM, 204, Float32(0.1)), _pattern(S * DIM, 205, Float32(0.5)),
    )


def _load_lora() raises -> WanBlockLora:
    # all 10 adapters present, nonzero A + B (the 20 d_A/d_B compared below).
    return WanBlockLora(
        _adapter(DIM, DIM, 300), _adapter(DIM, DIM, 301),
        _adapter(DIM, DIM, 302), _adapter(DIM, DIM, 303),
        _adapter(DIM, DIM, 304), _adapter(DIM, DIM, 305),
        _adapter(DIM, DIM, 306), _adapter(DIM, DIM, 307),
        _adapter(DIM, FFN, 308), _adapter(FFN, DIM, 309),
    )


def _gate(name: String, a: List[Float32], b: List[Float32], mut allok: Bool):
    """Raw bit-equality over two host F32 lists + a non-degeneracy guard (an
    all-zero compared tensor would gate vacuously -> FAIL)."""
    if len(a) != len(b):
        print(
            "GATE wan_parity " + name + " FAIL n_mismatch=-1 (len "
            + String(len(a)) + " vs " + String(len(b)) + ")"
        )
        allok = False
        return
    var n_mismatch = 0
    var first = -1
    var nonzero = False
    for i in range(len(a)):
        if a[i] != b[i]:
            n_mismatch += 1
            if first < 0:
                first = i
        if a[i] != 0.0:
            nonzero = True
    if not nonzero:
        print(
            "GATE wan_parity " + name + " FAIL n_mismatch=" + String(n_mismatch)
            + " (DEGENERATE: oracle tensor is all-zero, gate vacuous)"
        )
        allok = False
        return
    var verdict = String("PASS") if n_mismatch == 0 else String("FAIL")
    var line = (
        "GATE wan_parity " + name + " " + verdict
        + " n_mismatch=" + String(n_mismatch) + " n=" + String(len(a))
    )
    if n_mismatch > 0:
        line += " first_off=" + String(first)
    print(line)
    if n_mismatch != 0:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("=== wan_block_parity: Phase 1 same-process BIT gate (T2V) ===")
    print(
        "dims: H=", H, " Dh=", Dh, " DIM=", DIM, " S=", S, " TXT=", TXT,
        " FFN=", FFN, " rank=", RANK,
    )

    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()
    var cos = Tensor.from_host(_pattern(S * (Dh // 2), 901, Float32(0.7)), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_pattern(S * (Dh // 2), 902, Float32(0.7)), [S, Dh // 2], STDtype.F32, ctx)
    var x = _pattern(S * DIM, 701, Float32(1.0))
    var context = _pattern(TXT * DIM, 702, Float32(1.0))
    var d_out = _pattern(S * DIM, 703, Float32(0.01))

    print("[wan] forward once -> saved activations ...")
    var fwd = wan22_block_lora_forward[H, Dh, S, TXT](
        x.copy(), context.copy(), mv, w, lora, cos, sin, DIM, FFN, EPS, ctx,
    )

    print("[wan] oracle: hand-chain wan22_block_lora_backward ...")
    var g_or = wan22_block_lora_backward[H, Dh, S, TXT](
        d_out.copy(), mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    print("[wan] graph: wan22_block_lora_graph_backward ...")
    var g_gr = wan22_block_lora_graph_backward[H, Dh, S, TXT](
        d_out.copy(), mv, w, lora, fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )

    var allok = True
    _gate(String("d_x"), g_or.base.d_x, g_gr.base.d_x, allok)
    _gate(String("d_context"), g_or.base.d_context, g_gr.base.d_context, allok)
    _gate(String("sa_q dA"), g_or.sa_q_da, g_gr.sa_q_da, allok)
    _gate(String("sa_q dB"), g_or.sa_q_db, g_gr.sa_q_db, allok)
    _gate(String("sa_k dA"), g_or.sa_k_da, g_gr.sa_k_da, allok)
    _gate(String("sa_k dB"), g_or.sa_k_db, g_gr.sa_k_db, allok)
    _gate(String("sa_v dA"), g_or.sa_v_da, g_gr.sa_v_da, allok)
    _gate(String("sa_v dB"), g_or.sa_v_db, g_gr.sa_v_db, allok)
    _gate(String("sa_o dA"), g_or.sa_o_da, g_gr.sa_o_da, allok)
    _gate(String("sa_o dB"), g_or.sa_o_db, g_gr.sa_o_db, allok)
    _gate(String("ca_q dA"), g_or.ca_q_da, g_gr.ca_q_da, allok)
    _gate(String("ca_q dB"), g_or.ca_q_db, g_gr.ca_q_db, allok)
    _gate(String("ca_k dA"), g_or.ca_k_da, g_gr.ca_k_da, allok)
    _gate(String("ca_k dB"), g_or.ca_k_db, g_gr.ca_k_db, allok)
    _gate(String("ca_v dA"), g_or.ca_v_da, g_gr.ca_v_da, allok)
    _gate(String("ca_v dB"), g_or.ca_v_db, g_gr.ca_v_db, allok)
    _gate(String("ca_o dA"), g_or.ca_o_da, g_gr.ca_o_da, allok)
    _gate(String("ca_o dB"), g_or.ca_o_db, g_gr.ca_o_db, allok)
    _gate(String("ffn0 dA"), g_or.ffn0_da, g_gr.ffn0_da, allok)
    _gate(String("ffn0 dB"), g_or.ffn0_db, g_gr.ffn0_db, allok)
    _gate(String("ffn2 dA"), g_or.ffn2_da, g_gr.ffn2_da, allok)
    _gate(String("ffn2 dB"), g_or.ffn2_db, g_gr.ffn2_db, allok)

    # ── TEETH: the gate must FAIL a degenerate all-zero pair and a corrupted
    # pair (proves it is not vacuously passing).
    print("--- teeth (these MUST report FAIL) ---")
    var teeth_ok = True

    var z = _zeros(len(g_or.base.d_x))
    var t1 = True
    _gate(String("TEETH degenerate(zeros)"), z.copy(), z.copy(), t1)
    if t1:
        print("  !! teeth FAIL: degenerate pair did NOT fail the gate")
        teeth_ok = False

    var corrupt = g_gr.base.d_x.copy()
    corrupt[0] = corrupt[0] + Float32(1.0)
    var t2 = True
    _gate(String("TEETH corrupt(d_x)"), g_or.base.d_x, corrupt, t2)
    if t2:
        print("  !! teeth FAIL: corrupted d_x did NOT fail the gate")
        teeth_ok = False

    print("")
    if allok and teeth_ok:
        print("=== wan_block_parity: ALL GATES PASS (every grad bit-equal; teeth bite) ===")
    else:
        print("=== wan_block_parity: FAIL — see GATE lines above ===")
        raise Error("wan_block_parity: bit mismatch or teeth failure (wiring bug)")
