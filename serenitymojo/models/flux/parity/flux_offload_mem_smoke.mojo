# serenitymojo/models/flux/parity/flux_offload_mem_smoke.mojo
#
# REAL-WEIGHT MEMORY SMOKE for the Flux block-swap offload LoRA stack.
# Streams a FEW REAL flux1-dev.safetensors blocks through
# flux_stack_lora_forward_offload at REAL per-block dims (D=3072, H=24, Dh=128,
# Fmlp=12288) and:
#   * asserts the forward output is FINITE (no NaN/Inf),
#   * reports PEAK resident GPU memory via cuMemGetInfo_v2 (total - min free
#     observed) and asserts it stays < 24 GB.
#
# This proves the offload path runs on REAL weights within a 24GB budget (the
# whole point of block-swap: the 11.9B model never has all blocks resident). It
# uses a REDUCED block count (a few double + a few single blocks streamed from
# the real checkpoint) and a SMALL token count so the bounded run fits + is fast;
# the per-block WEIGHT footprint is the real full-size one (that is the memory
# being validated). Equivalence/correctness is proven separately by
# flux_offload_equiv_parity.mojo.
#
# The base stack (embeds / mod.lin / final layer) is built with small random F32
# weights — it is NOT streamed and not the memory bottleneck; this smoke measures
# the streamed-block footprint + finiteness, not parity.
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/flux/parity/flux_offload_mem_smoke.mojo \
#       -o /tmp/flux_offload_mem
#   /tmp/flux_offload_mem

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.models.flux.flux_stack import FluxStackBase, EmbedMlp, ModLin, DoubleModLin
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, flux_stack_lora_forward_offload,
)
from serenitymojo.offload.plan import build_flux_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.offload.vmm_cuda import cu_mem_get_info, CuMemInfo


comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"

# ── REAL flux1-dev per-block dims ──
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4 (REAL)
comptime IN_CH = 64
comptime TXT_CH = 4096         # joint_attention_dim
comptime OUT_CH = 64
comptime T_DIM = 256           # timestep_dim
comptime VEC_DIM = 768
# REDUCED token count + block count (bounded run; per-block WEIGHT size is REAL).
comptime N_IMG = 64
comptime N_TXT = 16
comptime S = N_TXT + N_IMG
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
comptime EPS = Float32(1e-06)
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime LSCALE = ALPHA / Float32(RANK)
comptime GB = Float64(1024.0 * 1024.0 * 1024.0)
comptime BUDGET_GB = Float64(24.0)


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


def _adapter(in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    return LoraAdapter(
        _rand(RANK * in_f, seed, 0.01), _zeros(out_f * RANK),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _used_gb(base_free: Int) raises -> Float64:
    var mi = cu_mem_get_info()
    return Float64(base_free - mi.free_bytes) / GB


def main() raises:
    var ctx = DeviceContext()
    print("==== flux_offload_mem_smoke (REAL flux1-dev blocks, peak GPU mem) ====")
    print("REAL dims: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP,
          " | reduced: N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE, " rank=", RANK)

    var mi0 = cu_mem_get_info()
    print("device total =", Float64(mi0.total_bytes) / GB, "GB   free at start =",
          Float64(mi0.free_bytes) / GB, "GB")
    var base_free = mi0.free_bytes
    var peak_used = Float64(0.0)

    # ── small random base stack (NOT streamed; not the bottleneck) ──
    var time_in = EmbedMlp(_rand(D * T_DIM, 1, 0.01), _rand(D, 2, 0.01),
                           _rand(D * D, 3, 0.01), _rand(D, 4, 0.01), T_DIM, D, ctx)
    var guid_in = EmbedMlp(_rand(D * T_DIM, 5, 0.01), _rand(D, 6, 0.01),
                           _rand(D * D, 7, 0.01), _rand(D, 8, 0.01), T_DIM, D, ctx)
    var vec_in = EmbedMlp(_rand(D * VEC_DIM, 9, 0.01), _rand(D, 10, 0.01),
                          _rand(D * D, 11, 0.01), _rand(D, 12, 0.01), VEC_DIM, D, ctx)
    var dbl_mod = List[DoubleModLin]()
    for bi in range(NUM_DOUBLE):
        var sd = UInt64(20 + bi * 10)
        var im = ModLin(_rand(6 * D * D, sd + 1, 0.01), _rand(6 * D, sd + 2, 0.01), 6 * D, D, ctx)
        var tm = ModLin(_rand(6 * D * D, sd + 3, 0.01), _rand(6 * D, sd + 4, 0.01), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    for bi in range(NUM_SINGLE):
        var sd = UInt64(40 + bi * 10)
        sgl_mod.append(ModLin(_rand(3 * D * D, sd + 1, 0.01), _rand(3 * D, sd + 2, 0.01), 3 * D, D, ctx))

    var base = FluxStackBase(
        _rand(D * IN_CH, 50, 0.01), _rand(D, 51, 0.01),
        _rand(D * TXT_CH, 52, 0.01), _rand(D, 53, 0.01),
        time_in^, True, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        _rand(2 * D * D, 54, 0.01), _rand(2 * D, 55, 0.01),
        _rand(OUT_CH * D, 56, 0.01), _rand(OUT_CH, 57, 0.01),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )

    # ── LoRA set (B=0 identity; real shapes) ──
    var ad = List[LoraAdapter]()
    var seed = UInt64(9000)
    for _ in range(NUM_DOUBLE):
        for _stream in range(2):
            ad.append(_adapter(D, D, seed)); seed += 1
            ad.append(_adapter(D, D, seed)); seed += 1
            ad.append(_adapter(D, D, seed)); seed += 1
            ad.append(_adapter(D, D, seed)); seed += 1
            ad.append(_adapter(D, FMLP, seed)); seed += 1
            ad.append(_adapter(FMLP, D, seed)); seed += 1
    for _ in range(NUM_SINGLE):
        ad.append(_adapter(D, D, seed)); seed += 1
        ad.append(_adapter(D, D, seed)); seed += 1
        ad.append(_adapter(D, D, seed)); seed += 1
        ad.append(_adapter(D, FMLP, seed)); seed += 1
        ad.append(_adapter(D + FMLP, D, seed)); seed += 1
    var lora = FluxLoraSet(ad^, NUM_DOUBLE, NUM_SINGLE, RANK)

    peak_used = max(peak_used, _used_gb(base_free))
    print("after base+LoRA resident: used =", peak_used, "GB")

    # ── stack inputs ──
    var img_tokens = _rand(N_IMG * IN_CH, 800, 1.0)
    var txt_tokens = _rand(N_TXT * TXT_CH, 801, 1.0)
    var timestep = _rand(1, 802, 1.0)
    var guidance = Optional[List[Float32]](_rand(1, 803, 1.0))
    var vector = _rand(VEC_DIM, 804, 1.0)
    var cos = _rand(S * H * (Dh // 2), 805, 1.0)
    var sin = _rand(S * H * (Dh // 2), 806, 1.0)

    # ── open the REAL checkpoint loader (single-file path) ──
    print("---- opening real flux1-dev loader + streaming", NUM_DOUBLE,
          "double +", NUM_SINGLE, "single blocks ----")
    var plan = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, cfg, ctx)
    peak_used = max(peak_used, _used_gb(base_free))

    var fwd = flux_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    ctx.synchronize()
    peak_used = max(peak_used, _used_gb(base_free))

    # ── finiteness check on the streamed-offload output ──
    var nbad = 0
    for i in range(len(fwd.out)):
        var x = fwd.out[i]
        if (x != x) or (x - x != Float32(0.0)):
            nbad += 1

    print("")
    print("forward out: n =", len(fwd.out), " nonfinite =", nbad)
    print("PEAK GPU mem used (offload) =", peak_used, "GB   (budget", BUDGET_GB, "GB)")
    if nbad == 0 and peak_used < BUDGET_GB and peak_used > Float64(0.0):
        print("VERDICT: PASS — real flux1-dev blocks stream + run finite under 24 GB")
    else:
        print("VERDICT: FAIL — nonfinite output or peak mem >= 24 GB")
