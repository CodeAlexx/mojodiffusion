# models/wan22/parity/wan22_train_vs_infer_block_parity.mojo
#
# THE GATE NOBODY WROTE: is the TRAINING block the same function as the INFERENCE
# block?
#
#   training : models/wan22/wan22_block.mojo::wan22_block_lora_forward
#   inference: models/dit/wan22_dit.mojo::wan22_block_forward
#
# These are two independent implementations of the same Wan2.2 block. Each is
# separately gated (training vs Torchref at cos>=0.999; inference via the Bernini/Wan
# product gates), and each is self-consistent — which is exactly why the gap hid.
# Nothing ever checked them AGAINST EACH OTHER, and a LoRA is trained in the first
# and rendered by the second. If they differ, an adapter learned in one is not valid
# in the other.
#
# WHY THIS EXISTS (2026-07-25): a 3500-step eri2 LoRA reached MSE 0.142 in the
# TRAINING forward (verified by resuming the checkpoint and re-running one step) but
# renders PURE NOISE through the inference forward, under both the Bernini APG recipe
# and the plain Wan UniPC+CFG recipe, at both 256px and 1024px. The base model renders
# perfectly through inference. That isolates the defect to the block forwards.
#
# METHOD: same dequantized block-0 weights, same x / context / e0 / rope, LoRA ABSENT
# on both sides. Anything but a match is the bug.
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build --optimization-level 2 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -ldl -Xlinker -lsqlite3 \
#     -Xlinker -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs -Xlinker -lcuda \
#     -Xlinker -L$CONDA_PREFIX/lib -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/wan22/parity/wan22_train_vs_infer_block_parity.mojo -o /tmp/wan_tvi
#
# Mojo 1.0.0b1, NVIDIA.

from std.collections import Dict, List, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.wan22_dit import Wan22Config, wan22_block_forward
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanBlockLora, WanModVecs, wan22_block_lora_forward,
)
from serenitymojo.models.wan22.wan22_fp8_stream import Wan22A14BFP8Stream
from serenitymojo.lora import LoraSet
from serenitymojo.ops.tensor_algebra import add as t_add
from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.tensor import Tensor

comptime TArc = ArcPointer[Tensor]

# Real Bernini/Wan A14B block dims, one 256-token frame (the training geometry).
comptime S = 256
comptime TXT = 512
comptime H = 40
comptime DH = 128
comptime DIM = 5120
comptime FFN = 13824
comptime CACHE = "/home/alex/.serenity/models/checkpoints/Bernini-R-Diffusers/serenity_fp8_e4m3_de8c462/high"


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = (Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0))
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var v = (Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0))
        out.append((u + v - Float32(1.0)) * scale * Float32(1.7320508))
    return out^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _g(blk: Dict[String, TArc], name: String) raises -> TArc:
    if name not in blk:
        raise Error(String("block missing: ") + name)
    return blk[name].copy()


def _cos_stats(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var mx = Float32(0.0)
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    var den = (na ** 0.5) * (nb ** 0.5)
    var cos = Float32(dot / den) if den > 0.0 else Float32(0.0)
    var out = List[Float32]()
    out.append(cos)
    out.append(mx)
    out.append(Float32(na ** 0.5))
    out.append(Float32(nb ** 0.5))
    return out^


def _adapter_for(
    ls: LoraSet, base_key: String, ctx: DeviceContext,
) raises -> Optional[LoraAdapter]:
    """One LoraAdapter (training representation) from the LoraSet's A/B for base_key."""
    if not ls.has_base(base_key):
        return Optional[LoraAdapter](None)
    var ab = ls.load_ab_for_base(base_key, ctx)
    var a = cast_tensor(ab[0], STDtype.F32, ctx)   # [rank, in]
    var b = cast_tensor(ab[1], STDtype.F32, ctx)   # [out, rank]
    var rank = a.shape()[0]
    var in_f = a.shape()[1]
    var out_f = b.shape()[0]
    var sc = ls.scale_for_base(base_key, Float32(1.0), ctx)
    var z_a = List[Float32]()
    z_a.resize(rank * in_f, Float32(0.0))
    var z_b = List[Float32]()
    z_b.resize(out_f * rank, Float32(0.0))
    return Optional[LoraAdapter](LoraAdapter(
        a.to_host(ctx), b.to_host(ctx), rank, in_f, out_f, sc,
        z_a.copy(), z_a.copy(), z_b.copy(), z_b.copy(),
    ))


def _block_lora_from_set(
    ls: LoraSet, bi: Int, ctx: DeviceContext,
) raises -> WanBlockLora:
    """The 10 wan adapters for block `bi`, in the trainer's slot order."""
    var p = String("blocks.") + String(bi) + String(".")
    return WanBlockLora(
        _adapter_for(ls, p + String("self_attn.q.weight"), ctx),
        _adapter_for(ls, p + String("self_attn.k.weight"), ctx),
        _adapter_for(ls, p + String("self_attn.v.weight"), ctx),
        _adapter_for(ls, p + String("self_attn.o.weight"), ctx),
        _adapter_for(ls, p + String("cross_attn.q.weight"), ctx),
        _adapter_for(ls, p + String("cross_attn.k.weight"), ctx),
        _adapter_for(ls, p + String("cross_attn.v.weight"), ctx),
        _adapter_for(ls, p + String("cross_attn.o.weight"), ctx),
        _adapter_for(ls, p + String("ffn.0.weight"), ctx),
        _adapter_for(ls, p + String("ffn.2.weight"), ctx),
    )


def main() raises:
    var ctx = DeviceContext()
    print("==== Wan2.2 block: TRAINING forward vs INFERENCE forward ====")
    print("  S=", S, " TXT=", TXT, " H=", H, " DH=", DH, " DIM=", DIM, " FFN=", FFN)
    print("  weights: block 0 of", CACHE)
    print("  LoRA ABSENT on both sides — this compares the BASE block only.")

    # ── one source of weights for both sides ──
    var stream = Wan22A14BFP8Stream.open(String(CACHE))
    var blk = stream.load_block_bf16(0, ctx)     # block-local keys, BF16, dequantized
    print("  loaded", len(blk), "block tensors")
    # unmerged snapshot: the training side must see the ORIGINAL weights, since it
    # applies the LoRA on the activation path rather than merging it.
    var blk0 = Dict[String, TArc]()
    for ref kv in blk.items():
        blk0[kv.key] = kv.value.copy()

    # ── shared inputs ──
    var x_h = _randn(S * DIM, 11, 1.0)
    var ctx_h = _randn(TXT * DIM, 22, 1.0)
    # e0: ONE row (numel == 6*dim) in F32 — exactly what batch-1 inference passes
    # (wan22_a14b_streamed_dit._time returns [1,1,6,dim]; _chunk6 slices axis 2), and
    # the block casts modulation to F32 before adding, so e0 must be F32 too.
    var e0_h = _randn(6 * DIM, 33, 1.0)
    var cos_h = _randn(S * (DH // 2), 44, 1.0)
    var sin_h = _randn(S * (DH // 2), 55, 1.0)

    var x_t = _t(x_h.copy(), [S, DIM], ctx)
    var ctx_t = _t(ctx_h.copy(), [TXT, DIM], ctx)
    var e0_t = Tensor.from_host(e0_h.copy(), [1, 1, 6, DIM], STDtype.F32, ctx)
    var cos_t = _t(cos_h.copy(), [S, DH // 2], ctx)
    var sin_t = _t(sin_h.copy(), [S, DH // 2], ctx)

    # ── INFERENCE forward ──
    var cfg = Wan22Config.t2v_a14b()
    var out_inf = wan22_block_forward[S, TXT, H, DH](
        x_t, e0_t, ctx_t, cos_t, sin_t, blk, cfg, ctx, TXT
    )
    var inf_h = cast_tensor(out_inf, STDtype.F32, ctx).to_host(ctx)
    print("  inference out:", out_inf.shape()[0], "x", out_inf.shape()[1])

    # ── TRAINING forward, same weights ──
    # modvecs: the inference block computes e = (modulation + e0).chunk(6) internally;
    # build the identical six vectors for the training block, which takes them as args.
    var mod_h = cast_tensor(_g(blk, String("modulation"))[], STDtype.F32, ctx).to_host(ctx)
    var sh = List[Float32]()
    var sc = List[Float32]()
    var ga = List[Float32]()
    var shf = List[Float32]()
    var scf = List[Float32]()
    var gaf = List[Float32]()
    for s in range(S):
        for d in range(DIM):
            sh.append(e0_h[0 * DIM + d] + mod_h[0 * DIM + d])
            sc.append(e0_h[1 * DIM + d] + mod_h[1 * DIM + d])
            ga.append(e0_h[2 * DIM + d] + mod_h[2 * DIM + d])
            shf.append(e0_h[3 * DIM + d] + mod_h[3 * DIM + d])
            scf.append(e0_h[4 * DIM + d] + mod_h[4 * DIM + d])
            gaf.append(e0_h[5 * DIM + d] + mod_h[5 * DIM + d])
    var mv = WanModVecs(sh^, sc^, ga^, shf^, scf^, gaf^)

    var w = WanBlockWeights.from_device(
        _g(blk, String("self_attn.q.weight")), _g(blk, String("self_attn.k.weight")),
        _g(blk, String("self_attn.v.weight")), _g(blk, String("self_attn.o.weight")),
        _g(blk, String("self_attn.q.bias")), _g(blk, String("self_attn.k.bias")),
        _g(blk, String("self_attn.v.bias")), _g(blk, String("self_attn.o.bias")),
        _g(blk, String("self_attn.norm_q.weight")), _g(blk, String("self_attn.norm_k.weight")),
        _g(blk, String("cross_attn.q.weight")), _g(blk, String("cross_attn.k.weight")),
        _g(blk, String("cross_attn.v.weight")), _g(blk, String("cross_attn.o.weight")),
        _g(blk, String("cross_attn.q.bias")), _g(blk, String("cross_attn.k.bias")),
        _g(blk, String("cross_attn.v.bias")), _g(blk, String("cross_attn.o.bias")),
        _g(blk, String("cross_attn.norm_q.weight")), _g(blk, String("cross_attn.norm_k.weight")),
        _g(blk, String("norm3.weight")), _g(blk, String("norm3.bias")),
        _g(blk, String("ffn.0.weight")), _g(blk, String("ffn.0.bias")),
        _g(blk, String("ffn.2.weight")), _g(blk, String("ffn.2.bias")),
    )
    var empty = Optional[LoraAdapter](None)
    var lora = WanBlockLora(
        empty.copy(), empty.copy(), empty.copy(), empty.copy(),
        empty.copy(), empty.copy(), empty.copy(), empty.copy(),
        empty.copy(), empty.copy(),
    )
    var fwd = wan22_block_lora_forward[H, DH, S, TXT](
        x_h, ctx_h, mv, w, lora, cos_t, sin_t, DIM, FFN, cfg.eps, ctx, False,
    )
    print("  training  out:", len(fwd.x_out), "elements")

    # ── compare ──
    var st = _cos_stats(fwd.x_out, inf_h)
    print("")
    print("  ||train|| =", st[2], "   ||infer|| =", st[3])
    print("  cosine    =", st[0])
    print("  max|diff| =", st[1])
    print("")
    var base_ok = st[0] > Float32(0.999)
    if base_ok:
        print("BASE: MATCH — the two block forwards agree (cos > 0.999).")
    else:
        print("BASE: MISMATCH — the blocks differ even without a LoRA. THIS is the bug.")
        return

    # ══ PHASE 2: same comparison WITH the trained LoRA on both sides ══
    # training applies it on the ACTIVATION path (y = Wx + s*B(Ax));
    # the inference overlay MERGES it into the weight (W' = W + s*B@A) in BF16.
    # Algebraically identical — this measures whether they are numerically so.
    print("")
    print("==== PHASE 2: with the trained LoRA on both sides ====")
    var lp = String(
        "/home/alex/mojodiffusion/output/wan22_lora/eri2_3500/eri2_wan22.safetensors"
    )
    var ls = LoraSet.load(lp)
    print("  lora:", lp, " format", ls.format_name(), " modules", len(ls.mappings))

    # (a) INFERENCE side: merge delta into the block-0 weights, exactly as
    #     Wan22A14BStreamedDiT._apply_block_lora does.
    var pre = String("blocks.0.")
    var merged = 0
    for ref m in ls.mappings:
        if not m.base_key.startswith(pre):
            continue
        var local = String(m.base_key.removeprefix(pre))
        if local not in blk:
            continue
        ref wt = blk[local][]
        var sc2 = ls._module_scale(m, Float32(1.0), ctx)
        var dl = ls._compute_delta(m, sc2, wt.dtype(), ctx)
        blk[local] = TArc(t_add(wt, dl, ctx))
        merged += 1
    print("  merged", merged, "deltas into the inference weights")

    var out_inf2 = wan22_block_forward[S, TXT, H, DH](
        x_t, e0_t, ctx_t, cos_t, sin_t, blk, cfg, ctx, TXT
    )
    var inf2_h = cast_tensor(out_inf2, STDtype.F32, ctx).to_host(ctx)

    # (b) TRAINING side: native activation-path LoRA, weights UNMERGED.
    var w2 = WanBlockWeights.from_device(
        _g(blk0, String("self_attn.q.weight")), _g(blk0, String("self_attn.k.weight")),
        _g(blk0, String("self_attn.v.weight")), _g(blk0, String("self_attn.o.weight")),
        _g(blk0, String("self_attn.q.bias")), _g(blk0, String("self_attn.k.bias")),
        _g(blk0, String("self_attn.v.bias")), _g(blk0, String("self_attn.o.bias")),
        _g(blk0, String("self_attn.norm_q.weight")), _g(blk0, String("self_attn.norm_k.weight")),
        _g(blk0, String("cross_attn.q.weight")), _g(blk0, String("cross_attn.k.weight")),
        _g(blk0, String("cross_attn.v.weight")), _g(blk0, String("cross_attn.o.weight")),
        _g(blk0, String("cross_attn.q.bias")), _g(blk0, String("cross_attn.k.bias")),
        _g(blk0, String("cross_attn.v.bias")), _g(blk0, String("cross_attn.o.bias")),
        _g(blk0, String("cross_attn.norm_q.weight")), _g(blk0, String("cross_attn.norm_k.weight")),
        _g(blk0, String("norm3.weight")), _g(blk0, String("norm3.bias")),
        _g(blk0, String("ffn.0.weight")), _g(blk0, String("ffn.0.bias")),
        _g(blk0, String("ffn.2.weight")), _g(blk0, String("ffn.2.bias")),
    )
    var la = _block_lora_from_set(ls, 0, ctx)
    var fwd2 = wan22_block_lora_forward[H, DH, S, TXT](
        x_h, ctx_h, mv, w2, la, cos_t, sin_t, DIM, FFN, cfg.eps, ctx, False,
    )

    var st2 = _cos_stats(fwd2.x_out, inf2_h)
    print("")
    print("  ||train+lora|| =", st2[2], "   ||infer+lora|| =", st2[3])
    print("  cosine         =", st2[0])
    print("  max|diff|      =", st2[1])
    print("")
    if st2[0] > Float32(0.999):
        print("VERDICT: LoRA application AGREES between train and infer.")
        print("  => the merge is fine; the failure is in the sampling trajectory.")
    else:
        print("VERDICT: LoRA application DIVERGES.")
        print("  activation-path LoRA (training) != weight-merge LoRA (inference).")
        print("  THIS is the bug.")
