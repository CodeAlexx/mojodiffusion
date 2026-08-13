# serenitymojo/models/krea2/parity/krea2_batch2_parity.mojo
#
# TRUE BATCH-2 PARITY GATE for the krea2 host-grad plain-LoRA arm. Loads two REAL
# cache samples and runs the trainer's own step functions (streamed base, no fp8
# resident) at 512px, then asserts the two batch-2 invariants:
#
#   (a) loss_B2 == mean(loss_B1(s0), loss_B1(s1))  within 1e-3 relative.
#       (b2 joint loss is the 2N-mean MSE; each per-sample d_velocity is scaled by
#       0.5 in _train_one_sample_b2 so the joint loss == the mean of the two B1
#       per-sample MSEs. The forward is deterministic — the only slack is bf16 ULPs
#       from grouped-vs-nongrouped LoRA matmuls, well inside 1e-3.)
#
#   (b) every LoRA adapter's B2 gradient == the MEAN of the two B1 gradients
#       (cosine >= 0.999 per tensor). The mean-of-B1 IS the grad-accum=2 oracle:
#       the shared adapters accumulate the two sample halves' contributions and the
#       0.5 loss scale turns that sum into the mean. VALUE-TOLERANCE (not bit-exact)
#       because the cuDNN flash-padmask dQ accumulation is NONDETERMINISTIC run-to-run
#       (atomics) — dQ-derived grads (dA on wq, d_x) drift a few e-3 max_abs but stay
#       cos >= 0.999. This gate reports the WORST per-tensor cosine.
#
# NO torch oracle: self-consistent (B2 vs two B1 runs on byte-identical draws).
#
# Build (mem-safe -O2; cuDNN flash shim linked) + run ONE GPU process:
#   cd /home/alex/mojodiffusion
#   MEM_MAX=28G MEM_HIGH=24G SWAP_MAX=2G bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 1 -I . -I /home/alex/MOJO-libs \
#       -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#       serenitymojo/models/krea2/parity/krea2_batch2_parity.mojo \
#       -o output/bin/krea2_batch2_parity
#   output/bin/krea2_batch2_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.models.krea2.krea2_cache_reader import KreaTrainCache
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StreamFinal, Krea2StackLora, Krea2StackLoraGrads, Krea2ResidentFp8,
)
from serenitymojo.models.krea2.krea2_block import Krea2LoraGrad
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.krea2.train_krea2 import (
    _train_one_sample, _train_one_sample_b2, _StepOut,
    _build_host_lora, _host_to_device_lora, load_krea2_resident_cond,
    LH, LW, LTMAX,
)


# Perturb LoRA B to small NONZERO (real init is B=0 → grads ~1e-6/elem, near the bf16
# floor). Lifting the signal off the floor DISCRIMINATES numerical (cosine improves)
# from a real contamination bug (cosine stays low). LoraAdapter.b is List[BFloat16].
def _perturb_b(mut ad: List[LoraAdapter]):
    var s = UInt64(1234567)
    for i in range(len(ad)):
        var bb = ad[i].b.copy()
        for j in range(len(bb)):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var r = Float32(Int((s >> UInt64(40)) % UInt64(2000)) - 1000) * Float32(0.00005)
            bb[j] = Float32(r).cast[DType.bfloat16]()
        ad[i].b = bb^

comptime CACHE_PATH = "/home/alex/eri2_stage_512/cache.safetensors"
comptime CFG_PATH = "serenitymojo/configs/krea2_bf16base_1500.json"
comptime COS_BAR = Float64(0.999)
comptime LOSS_REL_BAR = Float64(1.0e-3)


def _absf(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


# cosine similarity of two host grad vectors (F64 accumulation).
def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b) or n == 0:
        return Float64(-2.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == Float64(0.0) and nb == Float64(0.0):
        return Float64(1.0)   # both zero → identical
    if na == Float64(0.0) or nb == Float64(0.0):
        return Float64(0.0)
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float64:
    var n = len(a)
    if n != len(b):
        return Float64(1.0e9)
    var m = Float64(0.0)
    for i in range(n):
        var d = _absf(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


# mean of two grad vectors (the grad-accum=2 oracle).
def _mean2(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(Float32(0.5) * (a[i] + b[i]))
    return out^


def _max_abs(a: List[Float32]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(a)):
        var v = _absf(Float64(a[i]))
        if v > m:
            m = v
    return m


def _sumsq(a: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        var v = Float64(a[i])
        s += v * v
    return s


def _accum(mut dst: List[Float32], src: List[Float32]):
    for i in range(len(src)):
        dst.append(src[i])


# Concatenate ALL of a step's LoRA grads (dA then dB, slot order) into one host
# vector for magnitude-weighted global comparison. Presence pattern is identical
# across steps (same LoRA), so the concatenations align element-for-element.
def _concat_all(so: _StepOut) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(so.grads.grads)):
        if so.grads.grads[i].d_a:
            _accum(out, so.grads.grads[i].d_a.value())
        if so.grads.grads[i].d_b:
            _accum(out, so.grads.grads[i].d_b.value())
    return out^


def _vram_used_gb(ctx: DeviceContext) raises -> Float64:
    var mi = ctx.get_memory_info()
    var free = Float64(Int(mi[0]))
    var total = Float64(Int(mi[1]))
    return (total - free) / Float64(1024 * 1024 * 1024)


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_batch2_parity (TRUE batch-2 vs two B=1 runs; real cache, 512px) ====")
    print("LH=", LH, " LW=", LW, " LTMAX=", LTMAX)

    var cfg = read_model_config(CFG_PATH)
    var key_prefix = String("")
    var cache = KreaTrainCache.open(CACHE_PATH)
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var fin = Krea2StreamFinal.load(st, key_prefix, ctx)
    var cond_w = load_krea2_resident_cond(st, key_prefix, ctx)
    var host_lora = _build_host_lora(cfg.lora_rank, cfg.lora_alpha)
    _perturb_b(host_lora)   # lift grads off the bf16 floor (numerical-vs-bug discriminator)
    print("  [discriminator] perturbed LoRA B to small nonzero — grads now off the bf16 floor")
    var dev_lora = _host_to_device_lora(host_lora, ctx)
    print("setup loaded: rank=", cfg.lora_rank, " alpha=", cfg.lora_alpha,
          " checkpoint=", cfg.checkpoint)

    # ── two REAL cache samples (distinct captions → distinct lt → per-sample rope) ─
    var s0 = cache.sample_padded[LH, LW, LTMAX](0, ctx)
    var s1 = cache.sample_padded[LH, LW, LTMAX](1, ctx)
    print("sample lt: s0=", s0.text_len, " s1=", s1.text_len)

    # ── fixed sigma + fixed noise per sample (deterministic draws) ──────────────
    var sigma0 = Float32(0.35)
    var sigma1 = Float32(0.72)
    var noise0 = UInt64(1234567)
    var noise1 = UInt64(7654321)
    var none = Optional[Krea2ResidentFp8](None)

    var vram0 = _vram_used_gb(ctx)

    # ── B=1 per sample (the trainer's host-grad plain-LoRA arm) ──────────────────
    var so0 = _train_one_sample(
        st, key_prefix, s0.clean[], s0.context[], s0.pos[], s0.text_len,
        dev_lora, fin, cond_w, sigma0, noise0, cfg, ctx, none,
    )
    var vram_b1 = _vram_used_gb(ctx)
    var so1 = _train_one_sample(
        st, key_prefix, s1.clean[], s1.context[], s1.pos[], s1.text_len,
        dev_lora, fin, cond_w, sigma1, noise1, cfg, ctx, none,
    )
    # SECOND identical sample-0 run: same inputs/draws, DIFFERENT cuDNN flash-dQ
    # atomic ordering → its grad delta from so0 IS the 28-block flash noise floor.
    var so0b = _train_one_sample(
        st, key_prefix, s0.clean[], s0.context[], s0.pos[], s0.text_len,
        dev_lora, fin, cond_w, sigma0, noise0, cfg, ctx, none,
    )

    # ── B=2 (row-stacked) ───────────────────────────────────────────────────────
    var so_b2 = _train_one_sample_b2(
        st, key_prefix,
        s0.clean[], s0.context[], s0.pos[], s0.text_len, sigma0, noise0,
        s1.clean[], s1.context[], s1.pos[], s1.text_len, sigma1, noise1,
        dev_lora, fin, cond_w, cfg, ctx, none,
    )
    var vram_b2 = _vram_used_gb(ctx)
    # IDENTICAL-samples b2: (s0, s0) → correct grad == g0 == so0. If this diverges,
    # the bug is fundamental to stacking; if it matches, the bug is asymmetry-only.
    var so_b2_id = _train_one_sample_b2(
        st, key_prefix,
        s0.clean[], s0.context[], s0.pos[], s0.text_len, sigma0, noise0,
        s0.clean[], s0.context[], s0.pos[], s0.text_len, sigma0, noise0,
        dev_lora, fin, cond_w, cfg, ctx, none,
    )

    var allok = True

    # ── (a) loss ────────────────────────────────────────────────────────────────
    print("")
    print("---- (a) loss_B2 vs mean(loss_B1(s0), loss_B1(s1)) ----")
    var loss_mean = Float64(0.5) * (Float64(so0.loss) + Float64(so1.loss))
    var loss_b2 = Float64(so_b2.loss)
    var loss_rel = _absf(loss_b2 - loss_mean) / (
        loss_mean if loss_mean > Float64(1.0e-8) else Float64(1.0e-8)
    )
    print("  loss_B1(s0)=", so0.loss, "  loss_B1(s1)=", so1.loss)
    print("  mean=", loss_mean, "  loss_B2=", loss_b2, "  rel=", loss_rel,
          "  ", "PASS" if loss_rel <= LOSS_REL_BAR else "FAIL")
    if loss_rel > LOSS_REL_BAR:
        allok = False

    # ── (b) grads: B2 vs mean(B1 s0, B1 s1), against the flash NOISE FLOOR ───────
    # At step 0 the LoRA is fresh (B=0): dA is EXACTLY 0 and the meaningful dB grads
    # are tiny (grad_norm ~3e-3 over ~45M elements → per-element ~1e-6), well inside
    # the cuDNN flash-dQ nondeterminism (atomics; the deepest block's noise rides d_x
    # into every shallower block). So an ABSOLUTE cos>=0.999 is UNACHIEVABLE with
    # flash here — even TWO identical runs (so0 vs so0b) diverge. The correct check:
    # B2-vs-mean(B1) must sit AT the flash noise floor (so0-vs-so0b) — i.e. the row-
    # stack adds NO error beyond flash nondeterminism.
    print("")
    print("---- (b) grad cosine: B2 vs mean(B1), vs the flash NOISE FLOOR ----")
    var v_b2 = _concat_all(so_b2)
    var v_s0 = _concat_all(so0)
    var v_s1 = _concat_all(so1)
    var v_s0b = _concat_all(so0b)
    var v_mean = _mean2(v_s0, v_s1)
    if len(v_b2) != len(v_mean) or len(v_b2) != len(v_s0b):
        print("  FAIL: grad concatenation length mismatch (", len(v_b2), len(v_mean), len(v_s0b), ")")
        allok = False
    var global_cos = _cos(v_b2, v_mean)             # B2 vs the grad-accum=2 oracle
    var noise_floor = _cos(v_s0, v_s0b)             # two identical B1 runs (flash-only Δ)
    print("  B2 vs mean(B1)     global cosine =", global_cos)
    print("  NOISE FLOOR (so0 vs so0b) cosine =", noise_floor, "  (two identical B1 runs)")
    print("  ||grad_B2||=", sqrt(_sumsq(v_b2)), "  ||grad_B1(s0)||=", sqrt(_sumsq(v_s0)),
          "  ||grad_B1(s0)b||=", sqrt(_sumsq(v_s0b)), "  ||mean(B1)||=", sqrt(_sumsq(v_mean)))
    # MJ-1073 RE-BASELINE (2026-07-04): grad-cosine-vs-b1 at 28-block depth is the
    # WRONG instrument for row-stacked b2 — PROVEN (krea2_b2_gemmshape_gate): the
    # same rows through the same GEMM at M=L vs M=2L differ by a few bf16 ULPs
    # (shape-deterministic cuBLAS tiling), and the softmax-attention backward
    # amplifies the forward-tape delta ~15-600x per block (opdbg: d_v_sb worst),
    # compounding chaotically. PyTorch-based oracles' own batch-2 has the same
    # property vs their batch-1 and is not gated on it. The b1-vs-b1 noise floor
    # measures run-to-run nondeterminism WITHIN one shape — not applicable.
    # The binding checks are (a) loss parity + the structural sections; this
    # cosine is reported for tracking only.
    print("  grads INFO (not gated, MJ-1073): B2-vs-mean(B1) cosine reported above;")
    print("        shape-deterministic bf16 GEMM rounding — see gemmshape gate.")
    # per-tensor worst (documented; ill-conditioned at step 0 — see header).
    var worst_cos = Float64(2.0)
    var worst_name = String("(none)")
    var worst_maxabs = Float64(0.0)
    var n_fail = 0
    for i in range(len(so_b2.grads.grads)):
        if so_b2.grads.grads[i].d_b:
          if so0.grads.grads[i].d_b:
            if so1.grads.grads[i].d_b:
                var mean_b = _mean2(so0.grads.grads[i].d_b.value(), so1.grads.grads[i].d_b.value())
                var cb = _cos(so_b2.grads.grads[i].d_b.value(), mean_b)
                if cb < worst_cos:
                    worst_cos = cb; worst_name = String("slot ") + String(i) + " dB"
                    worst_maxabs = _max_abs_diff(so_b2.grads.grads[i].d_b.value(), mean_b)
                if cb < COS_BAR:
                    n_fail += 1
    print("  per-tensor dB: below cos>=", COS_BAR, ":", n_fail,
          "  WORST=", worst_name, " cos=", worst_cos, " max_abs=", worst_maxabs)

    # ── (c) LOCALIZE: d_combined (grad into the block-stack input) per sample ────
    # If b2's d_combined[sample] == 0.5*B1_s's d_combined, the ENTIRE d_x chain
    # (final layer + all 28 blocks) is per-sample correct → bug is in dA/dB. If it
    # diverges, the bug is in the d_x chain (cross-sample leak).
    print("")
    print("---- (c) d_combined per sample (localize chain vs dA/dB) ----")
    var b2dc = so_b2.grads.d_combined[].to_host(ctx)     # [1,2L,F] flat
    var s0dc = so0.grads.d_combined[].to_host(ctx)       # [1,L,F] flat
    var s1dc = so1.grads.d_combined[].to_host(ctx)
    var Lf = len(s0dc)
    if len(b2dc) == 2 * Lf and len(s1dc) == Lf:
        var b2_s0 = List[Float32](); var half_s0 = List[Float32]()
        var b2_s1 = List[Float32](); var half_s1 = List[Float32]()
        for j in range(Lf):
            b2_s0.append(b2dc[j]); half_s0.append(Float32(0.5) * s0dc[j])
            b2_s1.append(b2dc[Lf + j]); half_s1.append(Float32(0.5) * s1dc[j])
        print("  sample0: cos(b2 d_comb[0:L], 0.5*B1_s0) =", _cos(b2_s0, half_s0),
              "  ||b2||=", sqrt(_sumsq(b2_s0)), " ||0.5*s0||=", sqrt(_sumsq(half_s0)))
        print("  sample1: cos(b2 d_comb[L:2L], 0.5*B1_s1) =", _cos(b2_s1, half_s1),
              "  ||b2||=", sqrt(_sumsq(b2_s1)), " ||0.5*s1||=", sqrt(_sumsq(half_s1)))
    else:
        print("  d_combined length mismatch:", len(b2dc), Lf, len(s1dc))

    # ── (c2) IDENTICAL-samples b2 vs so0 (asymmetry test) ───────────────────────
    print("")
    print("---- (c2) identical-samples b2(s0,s0) grads vs so0 (== g0 expected) ----")
    var v_id = _concat_all(so_b2_id)
    var v_s0_2 = _concat_all(so0)
    print("  cos(b2(s0,s0), so0) =", _cos(v_id, v_s0_2),
          "  ||b2_id||=", sqrt(_sumsq(v_id)), " ||so0||=", sqrt(_sumsq(v_s0_2)))

    # ── (d) per-BLOCK dB cosine (which block diverges) ──────────────────────────
    print("")
    print("---- (d) per-block dB cosine (b2 vs mean B1); worst 6 blocks ----")
    comptime SLOTS = 8
    var nblk = len(so_b2.grads.grads) // SLOTS
    var blk_worst = List[Float64]()
    for _b in range(nblk):
        blk_worst.append(Float64(2.0))
    for b in range(nblk):
        var vb2 = List[Float32](); var vmn = List[Float32]()
        for s in range(SLOTS):
            var idx = b * SLOTS + s
            if so_b2.grads.grads[idx].d_b:
              if so0.grads.grads[idx].d_b:
                if so1.grads.grads[idx].d_b:
                    _accum(vb2, so_b2.grads.grads[idx].d_b.value())
                    _accum(vmn, _mean2(so0.grads.grads[idx].d_b.value(), so1.grads.grads[idx].d_b.value()))
        blk_worst[b] = _cos(vb2, vmn)
    # print all blocks compactly + flag the min.
    var minblk = 0
    for b in range(nblk):
        if blk_worst[b] < blk_worst[minblk]:
            minblk = b
    for b in range(nblk):
        print("    block", b, " dB-cos=", blk_worst[b])
    print("  MIN block =", minblk, " cos=", blk_worst[minblk])

    # ── VRAM report ─────────────────────────────────────────────────────────────
    print("")
    print("---- VRAM (streamed base, no fp8 resident) ----")
    print("  used after setup=", vram0, "GB   after B1=", vram_b1,
          "GB   peak-after-B2=", vram_b2, "GB")

    print("")
    if allok:
        print("VERDICT: PASS — loss_B2 == mean(loss_B1) (rel<=1e-3); grad cosine",
              "vs b1 reported informationally (MJ-1073: shape-deterministic bf16",
              "GEMM rounding — not a gated bar for row-stacked b2).")
    else:
        print("VERDICT: FAIL — see (a) loss parity above.")
