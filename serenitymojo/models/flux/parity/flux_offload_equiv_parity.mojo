# serenitymojo/models/flux/parity/flux_offload_equiv_parity.mojo
#
# RESIDENT-vs-OFFLOAD EQUIVALENCE GATE for the Flux (flux1-dev) FULL DiT STACK
# *WITH LoRA*. This is the CRITICAL gate that proves the block-swap offload path
# (flux_stack_lora_forward_offload / _backward_offload) is bit-faithful to the
# already-parity-verified RESIDENT path (flux_stack_lora_forward / _backward).
#
# DESIGN (fully self-contained Mojo — NO Python oracle needed):
#   1. Generate deterministic random weights (host F32) for NUM_DOUBLE double +
#      NUM_SINGLE single blocks at REAL per-block dims (D=3072, H=24, Dh=128;
#      small FMLP/N/depth so it fits comfortably and runs fast).
#   2. Build the RESIDENT dbw/sbw + FluxLoraSet from those lists, run
#      flux_stack_lora_forward + flux_stack_lora_backward → REFERENCE out + grads.
#   3. Write the SAME block weights to a temp safetensors with the BFL block keys
#      (double_blocks.{bi}.{img,txt}_attn/mlp.* ; single_blocks.{bi}.linear1/2,
#      .norm.*) — exactly the keys _double_weights_from_block /
#      _single_weights_from_block read. Open a TurboPlannedLoader over it with
#      build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE), run
#      flux_stack_lora_forward_offload + flux_stack_lora_backward_offload with the
#      SAME base/LoRA/inputs/cos/sin → OFFLOAD out + grads.
#   4. Assert out + EVERY adapter d_A/d_B match cos>=0.9999 (both are F32 Mojo
#      paths streaming the SAME F32 weights → identity to fp determinism).
#      Also asserts 0 nonfinite LoRA grads on the offload path.
#
# This proves the offload path's correctness. A separate REAL-WEIGHT memory smoke
# (flux_offload_mem_smoke.mojo) streams real flux1-dev blocks and reports peak GB.
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/flux/parity/flux_offload_equiv_parity.mojo \
#       -o /tmp/flux_offload_equiv
#   /tmp/flux_offload_equiv

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.io.safetensors_writer import save_safetensors

from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import FluxStackBase, EmbedMlp, ModLin, DoubleModLin
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet,
    flux_stack_lora_forward, flux_stack_lora_backward,
    flux_stack_lora_forward_offload, flux_stack_lora_backward_offload,
    total_adapters, DBL_SLOTS_PER_BLOCK,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.offload.plan import build_flux_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
# Flat /tmp path (always exists) — TurboPlannedLoader.open accepts a single
# .safetensors file directly (ShardedSafeTensors single-file fast path).
comptime CKPT_PATH = "/tmp/flux_offload_equiv_model.safetensors"

# ── dims (REAL per-block H/Dh/D; small depth/Fmlp/N for fit + speed) ──
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime N_IMG = 4
comptime N_TXT = 3
comptime S = N_TXT + N_IMG
comptime FMLP = 32
comptime IN_CH = 64
comptime TXT_CH = 40
comptime OUT_CH = 64
comptime T_DIM = 16
comptime VEC_DIM = 20
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)
comptime RANK = 4
comptime ALPHA = Float32(2.0)        # alpha != rank so scale != 1 (exercises scaling)
comptime LSCALE = ALPHA / Float32(RANK)


# ── deterministic host-list generators ──────────────────────────────────────
def _rand(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# LoRA adapter with NONZERO B (so every grad arm is exercised), shared scale.
def _adapter(in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    return LoraAdapter(
        _rand(RANK * in_f, seed, 0.05), _rand(out_f * RANK, seed + 7777, 0.05),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


# ── per-block weight host-list bundles (one source of truth for resident+offload) ─
struct _StreamLists(Copyable, Movable):
    var wqkv: List[Float32]
    var bqkv: List[Float32]
    var wproj: List[Float32]
    var bproj: List[Float32]
    var wmlp0: List[Float32]
    var bmlp0: List[Float32]
    var wmlp2: List[Float32]
    var bmlp2: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]

    def __init__(
        out self,
        var wqkv: List[Float32], var bqkv: List[Float32],
        var wproj: List[Float32], var bproj: List[Float32],
        var wmlp0: List[Float32], var bmlp0: List[Float32],
        var wmlp2: List[Float32], var bmlp2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
    ):
        self.wqkv = wqkv^; self.bqkv = bqkv^
        self.wproj = wproj^; self.bproj = bproj^
        self.wmlp0 = wmlp0^; self.bmlp0 = bmlp0^
        self.wmlp2 = wmlp2^; self.bmlp2 = bmlp2^
        self.q_norm = q_norm^; self.k_norm = k_norm^


def _gen_stream(seed: UInt64) -> _StreamLists:
    return _StreamLists(
        _rand(3 * D * D, seed + 1, 0.02), _rand(3 * D, seed + 2, 0.02),
        _rand(D * D, seed + 3, 0.02), _rand(D, seed + 4, 0.02),
        _rand(FMLP * D, seed + 5, 0.02), _rand(FMLP, seed + 6, 0.02),
        _rand(D * FMLP, seed + 7, 0.02), _rand(D, seed + 8, 0.02),
        _rand(Dh, seed + 9, 0.1), _rand(Dh, seed + 10, 0.1),
    )


def _stream_weights(sl: _StreamLists, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        sl.wqkv.copy(), sl.bqkv.copy(), sl.wproj.copy(), sl.bproj.copy(),
        sl.wmlp0.copy(), sl.bmlp0.copy(), sl.wmlp2.copy(), sl.bmlp2.copy(),
        sl.q_norm.copy(), sl.k_norm.copy(), D, FMLP, Dh, ctx,
    )


struct _SingleLists(Copyable, Movable):
    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]

    def __init__(
        out self,
        var w1: List[Float32], var b1: List[Float32],
        var w2: List[Float32], var b2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
    ):
        self.w1 = w1^; self.b1 = b1^; self.w2 = w2^; self.b2 = b2^
        self.q_norm = q_norm^; self.k_norm = k_norm^


def _gen_single(seed: UInt64) -> _SingleLists:
    return _SingleLists(
        _rand((3 * D + FMLP) * D, seed + 1, 0.02), _rand(3 * D + FMLP, seed + 2, 0.02),
        _rand(D * (D + FMLP), seed + 3, 0.02), _rand(D, seed + 4, 0.02),
        _rand(Dh, seed + 5, 0.1), _rand(Dh, seed + 6, 0.1),
    )


def _single_weights(sl: _SingleLists, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        sl.w1.copy(), sl.b1.copy(), sl.w2.copy(), sl.b2.copy(),
        sl.q_norm.copy(), sl.k_norm.copy(), D, FMLP, Dh, ctx,
    )


# Append one named F32 tensor to the safetensors write buffers.
def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    names.append(name)
    tensors.append(TArc(Tensor.from_host(vals.copy(), shape^, STDtype.F32, ctx)))


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32],
    mut allok: Bool, mut npass: Int, mut nfail: Int,
) raises:
    var r = harness.compare_host(actual, expected)
    if not r.passed:
        print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n, "   FAIL")
        allok = False
        nfail += 1
    else:
        npass += 1


def main() raises:
    var ctx = DeviceContext()
    print("==== flux_offload_equiv_parity (RESIDENT vs BLOCK-SWAP OFFLOAD) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " RANK=", RANK, " ALPHA=", ALPHA,
          " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE)

    # ── stack-level base (embeds + per-block mod.lin + final layer) ──
    var time_in = EmbedMlp(_rand(D * T_DIM, 100, 0.02), _rand(D, 101, 0.02),
                           _rand(D * D, 102, 0.02), _rand(D, 103, 0.02), T_DIM, D, ctx)
    var guid_in = EmbedMlp(_rand(D * T_DIM, 110, 0.02), _rand(D, 111, 0.02),
                           _rand(D * D, 112, 0.02), _rand(D, 113, 0.02), T_DIM, D, ctx)
    var vec_in = EmbedMlp(_rand(D * VEC_DIM, 120, 0.02), _rand(D, 121, 0.02),
                          _rand(D * D, 122, 0.02), _rand(D, 123, 0.02), VEC_DIM, D, ctx)

    var dbl_mod = List[DoubleModLin]()
    for bi in range(NUM_DOUBLE):
        var sd = UInt64(200 + bi * 10)
        var im = ModLin(_rand(6 * D * D, sd + 1, 0.02), _rand(6 * D, sd + 2, 0.02), 6 * D, D, ctx)
        var tm = ModLin(_rand(6 * D * D, sd + 3, 0.02), _rand(6 * D, sd + 4, 0.02), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    for bi in range(NUM_SINGLE):
        var sd = UInt64(300 + bi * 10)
        sgl_mod.append(ModLin(_rand(3 * D * D, sd + 1, 0.02), _rand(3 * D, sd + 2, 0.02), 3 * D, D, ctx))

    var base = FluxStackBase(
        _rand(D * IN_CH, 400, 0.02), _rand(D, 401, 0.02),
        _rand(D * TXT_CH, 402, 0.02), _rand(D, 403, 0.02),
        time_in^, True, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        _rand(2 * D * D, 404, 0.02), _rand(2 * D, 405, 0.02),
        _rand(OUT_CH * D, 406, 0.02), _rand(OUT_CH, 407, 0.02),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )

    # ── per-block weight lists (single source of truth) ──
    var dbl_lists = List[_StreamLists]()   # 2 per double block: img then txt
    for bi in range(NUM_DOUBLE):
        dbl_lists.append(_gen_stream(UInt64(1000 + bi * 100)))       # img
        dbl_lists.append(_gen_stream(UInt64(1000 + bi * 100 + 50)))  # txt
    var sgl_lists = List[_SingleLists]()
    for bi in range(NUM_SINGLE):
        sgl_lists.append(_gen_single(UInt64(5000 + bi * 100)))

    # ── resident block weight structs ──
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        dbw.append(DoubleBlockWeights(
            _stream_weights(dbl_lists[bi * 2 + 0], ctx),
            _stream_weights(dbl_lists[bi * 2 + 1], ctx),
        ))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_single_weights(sgl_lists[bi], ctx))

    # ── LoRA set (NONZERO B; flat order build_flux_lora_set produces) ──
    var ad = List[LoraAdapter]()
    var seed = UInt64(70000)
    for _ in range(NUM_DOUBLE):
        for _stream in range(2):
            ad.append(_adapter(D, D, seed)); seed += 1     # to_q
            ad.append(_adapter(D, D, seed)); seed += 1     # to_k
            ad.append(_adapter(D, D, seed)); seed += 1     # to_v
            ad.append(_adapter(D, D, seed)); seed += 1     # proj
            ad.append(_adapter(D, FMLP, seed)); seed += 1  # mlp0
            ad.append(_adapter(FMLP, D, seed)); seed += 1  # mlp2
    for _ in range(NUM_SINGLE):
        ad.append(_adapter(D, D, seed)); seed += 1         # to_q
        ad.append(_adapter(D, D, seed)); seed += 1         # to_k
        ad.append(_adapter(D, D, seed)); seed += 1         # to_v
        ad.append(_adapter(D, FMLP, seed)); seed += 1      # proj_mlp
        ad.append(_adapter(D + FMLP, D, seed)); seed += 1  # linear2
    var lora = FluxLoraSet(ad^, NUM_DOUBLE, NUM_SINGLE, RANK)

    # ── stack inputs (shared by both paths) ──
    var img_tokens = _rand(N_IMG * IN_CH, 800, 1.0)
    var txt_tokens = _rand(N_TXT * TXT_CH, 801, 1.0)
    var timestep = _rand(1, 802, 1.0)
    var guidance = Optional[List[Float32]](_rand(1, 803, 1.0))
    var vector = _rand(VEC_DIM, 804, 1.0)
    var cos = _rand(S * H * (Dh // 2), 805, 1.0)
    var sin = _rand(S * H * (Dh // 2), 806, 1.0)
    var d_out = _rand(N_IMG * OUT_CH, 807, 1.0)

    # ════════════════════════════════════════════════════════════════════════
    # REFERENCE: resident forward + backward
    # ════════════════════════════════════════════════════════════════════════
    print("")
    print("---- running RESIDENT path (reference) ----")
    var fwd_r = flux_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var g_r = flux_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, dbw, sbw, lora,
        cos.copy(), sin.copy(), fwd_r,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Write the SAME block weights to a temp safetensors with BFL block keys.
    # ════════════════════════════════════════════════════════════════════════
    print("---- writing reduced-depth checkpoint to ", CKPT_PATH, " ----")
    var names = List[String]()
    var tensors = List[TArc]()
    for bi in range(NUM_DOUBLE):
        var dp = String("double_blocks.") + String(bi)
        var streams: List[String] = ["img", "txt"]
        for si in range(2):
            ref sl = dbl_lists[bi * 2 + si]
            var ap = dp + "." + streams[si] + "_attn"
            var mp = dp + "." + streams[si] + "_mlp"
            _add(names, tensors, ap + ".qkv.weight", sl.wqkv, [3 * D, D], ctx)
            _add(names, tensors, ap + ".qkv.bias", sl.bqkv, [3 * D], ctx)
            _add(names, tensors, ap + ".proj.weight", sl.wproj, [D, D], ctx)
            _add(names, tensors, ap + ".proj.bias", sl.bproj, [D], ctx)
            _add(names, tensors, mp + ".0.weight", sl.wmlp0, [FMLP, D], ctx)
            _add(names, tensors, mp + ".0.bias", sl.bmlp0, [FMLP], ctx)
            _add(names, tensors, mp + ".2.weight", sl.wmlp2, [D, FMLP], ctx)
            _add(names, tensors, mp + ".2.bias", sl.bmlp2, [D], ctx)
            _add(names, tensors, ap + ".norm.query_norm.scale", sl.q_norm, [Dh], ctx)
            _add(names, tensors, ap + ".norm.key_norm.scale", sl.k_norm, [Dh], ctx)
    for bi in range(NUM_SINGLE):
        var sp = String("single_blocks.") + String(bi)
        ref sl = sgl_lists[bi]
        _add(names, tensors, sp + ".linear1.weight", sl.w1, [3 * D + FMLP, D], ctx)
        _add(names, tensors, sp + ".linear1.bias", sl.b1, [3 * D + FMLP], ctx)
        _add(names, tensors, sp + ".linear2.weight", sl.w2, [D, D + FMLP], ctx)
        _add(names, tensors, sp + ".linear2.bias", sl.b2, [D], ctx)
        _add(names, tensors, sp + ".norm.query_norm.scale", sl.q_norm, [Dh], ctx)
        _add(names, tensors, sp + ".norm.key_norm.scale", sl.k_norm, [Dh], ctx)
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    # ════════════════════════════════════════════════════════════════════════
    # OFFLOAD: stream the same weights, run forward + backward.
    # ════════════════════════════════════════════════════════════════════════
    print("---- running BLOCK-SWAP OFFLOAD path ----")
    var plan = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT_PATH), plan^, cfg, ctx)

    var fwd_o = flux_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var g_o = flux_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader, lora,
        cos.copy(), sin.copy(), fwd_o,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    # ════════════════════════════════════════════════════════════════════════
    # COMPARE: offload vs resident (cos>=0.9999 everywhere).
    # ════════════════════════════════════════════════════════════════════════
    var harness = ParityHarness(0.9999)
    var allok = True
    var npass = 0
    var nfail = 0

    print("")
    print("---- forward output equivalence ----")
    _check(harness, "out", fwd_o.out, fwd_r.out, allok, npass, nfail)

    print("")
    print("---- load-bearing input/embed grad equivalence ----")
    _check(harness, "d_img_tokens", g_o.d_img_tokens, g_r.d_img_tokens, allok, npass, nfail)
    _check(harness, "d_txt_tokens", g_o.d_txt_tokens, g_r.d_txt_tokens, allok, npass, nfail)
    _check(harness, "d_vec", g_o.d_vec, g_r.d_vec, allok, npass, nfail)
    _check(harness, "d_timestep", g_o.d_timestep, g_r.d_timestep, allok, npass, nfail)
    _check(harness, "d_guidance", g_o.d_guidance, g_r.d_guidance, allok, npass, nfail)
    _check(harness, "d_vector", g_o.d_vector, g_r.d_vector, allok, npass, nfail)

    print("")
    print("---- ALL adapter d_A / d_B equivalence (only FAILs printed) ----")
    var n = total_adapters(lora)
    for i in range(n):
        _check(harness, String("d_a[") + String(i) + "]", g_o.d_a[i], g_r.d_a[i], allok, npass, nfail)
        _check(harness, String("d_b[") + String(i) + "]", g_o.d_b[i], g_r.d_b[i], allok, npass, nfail)

    print("")
    print("checks: PASS =", npass, " FAIL =", nfail,
          " offload nonfinite_lora_grads =", g_o.nonfinite_lora_grads,
          " (n_adapters =", n, ")")
    if allok and g_o.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — offload path bit-faithful to resident (out + all d_A/d_B cos>=0.9999, 0 nonfinite)")
    else:
        print("VERDICT: FAIL — offload diverged from resident or grads nonfinite (see FAIL lines)")
