# models/wan22/parity/wan22_stack_train_vs_infer_parity.mojo
#
# STACK-LEVEL parity: our TRAINING stack vs our INFERENCE stack, same weights,
# same inputs, NO LoRA. This is the gap every existing gate leaves open.
#
#   training : models/wan22/wan22_stack_lora.mojo::wan22_stack_lora_forward_offload
#   inference: models/wan22/wan22_a14b_streamed_dit.mojo::forward_cfg_pair
#
# WHY (2026-07-25): the block-level gates all PASS —
#   * our train block vs our infer block: cos 0.9999771 (wan22_train_vs_infer_block_parity)
#   * our train block vs MUSUBI's WanAttentionBlock: cos >= 0.9998 on all 20 LoRA grads
#     (wan22_block_lora_parity_musubi)
# yet the TRAINING stack scores MSE ~2.1 at t=0.30 on real cached data, which is WORSE
# than predicting zero (var(noise - x0) = 1 + var(x0) ~ 1.2), while the INFERENCE stack
# renders clean 1024px images from the same weights. Something between the block and the
# stack is wrong, and only these two things differ there:
#   patch embed | time embedding + e0 broadcast | text embedding/padding | head
#
# The block gates cannot see any of it: the block takes e0 and the embedded context as
# INPUTS.
#
# Runs on the REAL Wan2.2-T2V-A14B fp8 cache (NOT Bernini).
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build --optimization-level 2 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -ldl -Xlinker -lsqlite3 \
#     -Xlinker -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs -Xlinker -lcuda \
#     -Xlinker -L$CONDA_PREFIX/lib -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/wan22/parity/wan22_stack_train_vs_infer_parity.mojo -o /tmp/wan_stack_tvi
#
# Mojo 1.0.0b1, NVIDIA.

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.dit.wan22_dit import wan22_build_rope
from serenitymojo.models.wan22.wan22_a14b_streamed_dit import Wan22A14BStreamedDiT
from serenitymojo.models.wan22.weights import load_wan22_stack_base
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22StackBase, build_wan22_lora_set, wan22_stack_lora_forward_offload,
    _time_features,
)
from serenitymojo.offload.wan22_plan import build_wan22_block_plan
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.tensor import Tensor

comptime TArc = ArcPointer[Tensor]

# Training geometry: one 256px frame -> latent [16,1,32,32] -> S=256.
comptime FG = 1
comptime HG = 16
comptime WG = 16
comptime S = FG * HG * WG
comptime TXT = 512
comptime CTXL = 512
comptime TEXT_DIM = 4096
comptime H = 40
comptime Dh = 128
comptime DIM = 5120
comptime FFN = 13824
comptime IN_CH = 64          # 16 latent channels packed 2x2
comptime OUT_CH = 64
comptime FREQ_DIM = 256
comptime EPS = Float32(1e-6)
comptime ROPE_THETA = Float32(10000.0)
comptime LAT_C = 16
comptime LAT_H = 32
comptime LAT_W = 32
comptime CKPT = "/home/alex/.serenity/models/checkpoints/wan2.2_t2v_a14b_fp8_e4m3/low"


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var v = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)
        out.append((u + v - Float32(1.0)) * scale * Float32(1.7320508))
    return out^


def _cos_max(a: List[Float32], b: List[Float32]) -> List[Float32]:
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
    var out = List[Float32]()
    out.append(Float32(dot / den) if den > 0.0 else Float32(0.0))
    out.append(mx)
    out.append(Float32(na ** 0.5))
    out.append(Float32(nb ** 0.5))
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== Wan2.2-A14B STACK parity: TRAINING vs INFERENCE ====")
    print("  ckpt:", CKPT)
    print("  S=", S, " latent [", LAT_C, 1, LAT_H, LAT_W, "]  NO LoRA (B=0)")

    # ── shared inputs ──
    # x0 in PATCH space [S, IN_CH] is the trainer's native representation; the same
    # numbers reshaped to [16,1,32,32] are what the inference stack takes, because
    # patchify3d(1,2,2) is a pure permutation of the latent elements.
    # ⚠ patchify3d and unpatchify3d are NOT inverses (measured: 15360/16384 elements
    # differ on a round trip). The LATENT is therefore the single source of truth:
    # inference takes it directly, training takes patchify3d() of it. Deriving the
    # latent from a patch-space vector instead would feed the two stacks different data.
    var x_lat_h = _randn(LAT_C * FG * LAT_H * LAT_W, 101, 1.0)
    var x_lat_src = Tensor.from_host(
        x_lat_h.copy(), [LAT_C, FG, LAT_H, LAT_W], STDtype.F32, ctx
    )
    var x_patch = patchify3d(x_lat_src, 1, 2, 2, ctx).to_host(ctx)
    var txt = _randn(TXT * TEXT_DIM, 202, 0.085)   # umt5 scale, matches the cache
    var t_raw = Float32(0.30)
    # SAME timestep value to BOTH sides, so this measures the stacks and not the
    # t*1000+1 vs sigma*1000 convention difference (reported separately below).
    var t_model = t_raw * Float32(1000.0) + Float32(1.0)
    print("  t_raw=", t_raw, " t_model fed to BOTH stacks =", t_model)

    # ── TRAINING stack ──
    var base_st = SafeTensors.open(String(CKPT) + String("/shared.safetensors"))
    var base = load_wan22_stack_base(base_st, ctx)
    var plan = build_wan22_block_plan(40)
    var off = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(String(CKPT), plan^, off, ctx, False)
    var lora = build_wan22_lora_set(40, DIM, FFN, 16, Float32(16.0))  # B=0 -> inert

    var rope = wan22_build_rope(FG, HG, WG, Dh, ROPE_THETA, STDtype.F32, ctx)
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)

    # ── BISECT 1: the time-embedding chain (e0) ──
    # training _time_features -> e0_flat [S*6*dim] (per-token; all rows equal for one t)
    # inference _time         -> e0      [1,1,6,dim] (single row, broadcast in-block)
    var tf = _time_features(t_model, S, DIM, FREQ_DIM, base, ctx)
    var tr_e0 = tf[0].copy()
    var model_pre = Wan22A14BStreamedDiT.open(String(CKPT), ctx)
    var it = model_pre._time[S](t_model, STDtype.BF16, ctx)
    var if_e0 = cast_tensor(it[0], STDtype.F32, ctx).to_host(ctx)
    var tr_row = List[Float32]()
    for i in range(6 * DIM):
        tr_row.append(tr_e0[i])          # row 0 of the per-token e0
    var e0st = _cos_max(tr_row, if_e0)
    print("")
    print("  [BISECT 1] e0 time embedding: cos =", e0st[0], " max|diff| =", e0st[1])
    print("             ||train_e0|| =", e0st[2], "  ||infer_e0|| =", e0st[3])
    if e0st[0] < Float32(0.999):
        print("             ^^ TIME EMBEDDING DIVERGES — this is upstream of every block gate")
    else:
        print("             ^^ time embedding matches; divergence is elsewhere")

    var fwd = wan22_stack_lora_forward_offload[H, Dh, S, TXT](
        x_patch.copy(), txt.copy(), t_model, base, loader, lora,
        cos_h.copy(), sin_h.copy(),
        DIM, FFN, IN_CH, TEXT_DIM, OUT_CH, FREQ_DIM, EPS, ctx,
        False, True, False, True, True,
    )
    # [S, OUT_CH] patch space -> [16,1,32,32] latent space, matching inference's head.
    var tr_t = Tensor.from_host(fwd.out.copy(), [S, OUT_CH], STDtype.F32, ctx)
    var tr_lat = unpatchify3d(tr_t, LAT_C, FG, LAT_H, LAT_W, 1, 2, 2, ctx)
    var train_h = tr_lat.to_host(ctx)   # LATENT space — the inference head's own space
    print("  training stack out:", len(train_h), "elements")

    # ── INFERENCE stack (same weights, fresh) ──
    var model = Wan22A14BStreamedDiT.open(String(CKPT), ctx)
    var x_lat4 = Tensor.from_host(
        x_lat_h.copy(), [LAT_C, FG, LAT_H, LAT_W], STDtype.F32, ctx
    )
    var ctx_t = Tensor.from_host(txt.copy(), [TXT, TEXT_DIM], STDtype.BF16, ctx)
    var pair = model.forward_cfg_pair[FG, HG, WG, S, TXT, CTXL, H, Dh](
        cast_tensor(x_lat4, STDtype.BF16, ctx), t_model, ctx_t, ctx_t, TXT, TXT, ctx,
    )
    # already latent space (the head unpatchified it with the same op) — compare as-is
    var inf_h = cast_tensor(pair.cond, STDtype.F32, ctx).to_host(ctx)
    print("  inference stack out:", len(inf_h), "elements")

    # ── compare ──
    var st = _cos_max(train_h, inf_h)
    print("")
    print("  ||train|| =", st[2], "   ||infer|| =", st[3])
    print("  cosine    =", st[0])
    print("  max|diff| =", st[1])
    print("")
    if st[0] > Float32(0.999):
        print("VERDICT: STACKS MATCH (cos > 0.999).")
        print("  => the anti-correlated training loss is NOT a stack divergence;")
        print("     look at the DATA (cached latents / text) or the target definition.")
    else:
        print("VERDICT: STACKS DIVERGE — the training stack and the inference stack")
        print("  compute DIFFERENT functions from identical weights and inputs.")
        print("  Everything between the block and the stack is suspect:")
        print("    patch embed | time embedding + e0 broadcast | text embed/pad | head")
    print("")
    print("NOTE (separate from the above): in production the trainer feeds")
    print("  t_model = t*1000 + 1 while the Wan sampler feeds sigma*1000 (no +1).")
    print("  This harness fed the SAME value to both to isolate the stacks.")
