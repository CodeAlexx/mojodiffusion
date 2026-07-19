# smoke/ideogram4_batch2_parity.mojo
#
# TRUE BATCH-2 PARITY GATE for the ideogram4 DEVICE-GRAD plain-LoRA arm. Runs the
# trainer's own resident compute functions on TWO distinct samples derived from the
# real predict fixture (same latents, DIFFERENT text_len + t_flow — which stresses
# the per-sample MRoPE-table concat and cross-sample independence, the load-bearing
# parts of the row-stacked b2 path), then asserts:
#
#   (a) BINDING — per-sample forward parity + joint 2N-mean loss:
#         loss0_B2 == loss0_B1  and  loss1_B2 == loss1_B1   (rel < 1e-3)
#         loss_B2  == mean(loss0_B1, loss1_B1)              (rel < 1e-3)
#       If the batched stack leaked one sample's attention into the other, or the
#       per-sample rope table were mis-stacked, these per-sample losses would move.
#
#   (b) INFORMATIONAL — per-adapter LoRA-B grad cosine of the b2 grad vs the MEAN of
#       the two b1 grads (the grad-accum=2 oracle: each per-sample d_velocity is
#       scaled by 0.5 in compute_resident_b2 so the SUMMED grads == the mean). Value
#       tolerance, NOT bit — reported (worst per-tensor cosine), NOT asserted (SDPA
#       math-mode + bf16 GEMM ULP class). d_a is 0 at B=0 (down·b, b=0) so only d_b
#       carries signal here.
#
# NO torch oracle: self-consistent (b2 vs two b1 runs on byte-identical inputs).
# The two arms feed the SAME device optimizer (apply_ideogram4_lora_grads); this
# gate checks the grads the b2 path HANDS that optimizer.
#
# Build (mem-safe -O2; cuDNN forward shim linked — ideogram4 flash bwd is OFF, Dh=256):
#   cd trainer
#   MEM_MAX=45G MEM_HIGH=38G SWAP_MAX=4G bash /home/alex/mojodiffusion/scripts/mem_safe.sh \
#     pixi run bash -c 'mojo build --optimization-level 2 --num-threads 2 -I . -I src \
#       -I /home/alex/mojodiffusion -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#       -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/.pixi/envs/default/lib \
#       smoke/ideogram4_batch2_parity.mojo -o target/ideogram4_batch2_parity'
#   target/ideogram4_batch2_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.ideogram4_resident import Ideogram4Weights
from serenitymojo.training.train_config import TrainConfig as LeversConfig

from serenity_trainer.model.Ideogram4LoRABlock import (
    Ideogram4LoraSet,
    Ideogram4StackLoraGrads,
    build_ideogram4_native_lora_set,
)
from serenity_trainer.trainer.Ideogram4LoRATrainStep import (
    ideogram4_lora_train_compute_resident,
    ideogram4_lora_train_compute_resident_b2,
)


comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"

comptime NT = 651
comptime GH = 16
comptime GW = 16
comptime TFLOW0 = Float32(0.7)
comptime TFLOW1 = Float32(0.55)
comptime LOSS_REL_BAR = Float64(1.0e-3)


def _absf(x: Float64) -> Float64:
    return x if x >= Float64(0.0) else -x


def _rel(a: Float32, b: Float32) -> Float64:
    var af = Float64(a)
    var bf = Float64(b)
    var denom = _absf(af)
    if _absf(bf) > denom:
        denom = _absf(bf)
    if denom < Float64(1.0e-12):
        return _absf(af - bf)
    return _absf(af - bf) / denom


def _cos_db2_vs_mean(
    gb2: Tensor, g0: Tensor, g1: Tensor, ctx: DeviceContext
) raises -> Float64:
    var hb2 = gb2.to_host(ctx)
    var h0 = g0.to_host(ctx)
    var h1 = g1.to_host(ctx)
    var n = len(hb2)
    if n == 0 or len(h0) != n or len(h1) != n:
        return Float64(-2.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(n):
        var av = Float64(hb2[i])
        var bv = Float64(0.5) * (Float64(h0[i]) + Float64(h1[i]))
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na < Float64(1.0e-20) or nb < Float64(1.0e-20):
        return Float64(1.0)   # both ~zero (adapter untouched this step)
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)
    var rw = Ideogram4Weights.load(ShardedSafeTensors.open(COND), ctx)

    var clean = cast_tensor(
        Tensor.from_view(fx.tensor_view("clean_latent"), ctx), STDtype.F32, ctx
    )
    var noise = cast_tensor(
        Tensor.from_view(fx.tensor_view("noise"), ctx), STDtype.F32, ctx
    )
    var noisy = cast_tensor(
        Tensor.from_view(fx.tensor_view("noisy"), ctx), STDtype.F32, ctx
    )
    var llm = cast_tensor(
        Tensor.from_view(fx.tensor_view("llm_features"), ctx), STDtype.BF16, ctx
    )

    # Two DISTINCT samples: same latents, different text_len + t_flow. Sample 1's
    # half caption forces a DIFFERENT MRoPE table + pad mask than sample 0 — the
    # exact per-sample treatment the row-stacked b2 must keep independent.
    var text_len0 = NT
    var text_len1 = NT // 2

    var loras = build_ideogram4_native_lora_set(16, Float32(16.0), ctx)

    # ── b2 (row-stacked, device-grad) ──────────────────────────────────────────
    var loss_j = Float32(0.0)
    var l0_b2 = Float32(0.0)
    var l1_b2 = Float32(0.0)
    var grads_b2 = ideogram4_lora_train_compute_resident_b2[NT, GH, GW](
        rw,
        noisy, clean, noise, TFLOW0, llm, text_len0,
        noisy, clean, noise, TFLOW1, llm, text_len1,
        loras, LeversConfig.default(),
        loss_j, l0_b2, l1_b2, ctx,
    )

    # ── two b1 runs (the oracle) ────────────────────────────────────────────────
    var l0_b1 = Float32(0.0)
    var grads0 = ideogram4_lora_train_compute_resident[NT, GH, GW](
        rw, noisy, clean, noise, TFLOW0, llm, loras,
        LeversConfig.default(), l0_b1, ctx, text_len0,
    )
    var l1_b1 = Float32(0.0)
    var grads1 = ideogram4_lora_train_compute_resident[NT, GH, GW](
        rw, noisy, clean, noise, TFLOW1, llm, loras,
        LeversConfig.default(), l1_b1, ctx, text_len1,
    )

    print("[i4-b2] loss0  b2=", l0_b2, " b1=", l0_b1)
    print("[i4-b2] loss1  b2=", l1_b2, " b1=", l1_b1)
    print("[i4-b2] joint  b2=", loss_j, " mean(b1)=", Float32(0.5) * (l0_b1 + l1_b1))

    var r0 = _rel(l0_b2, l0_b1)
    var r1 = _rel(l1_b2, l1_b1)
    var rj = _rel(loss_j, Float32(0.5) * (l0_b1 + l1_b1))
    print("[i4-b2] rel: loss0=", r0, " loss1=", r1, " joint=", rj,
          " (bar ", LOSS_REL_BAR, ")")

    # ── INFORMATIONAL: worst per-adapter d_b cosine (b2 vs mean-of-b1) ──────────
    var worst = Float64(2.0)
    var n_ad = len(grads_b2.d_b)
    for i in range(n_ad):
        var c = _cos_db2_vs_mean(
            grads_b2.d_b[i][], grads0.d_b[i][], grads1.d_b[i][], ctx
        )
        if c < worst:
            worst = c
    print("[i4-b2] INFORMATIONAL worst d_b cosine (b2 vs mean-of-b1) over ",
          n_ad, " adapters = ", worst)

    var ok = (r0 < LOSS_REL_BAR) and (r1 < LOSS_REL_BAR) and (rj < LOSS_REL_BAR)
    if ok:
        print("[i4-b2] BINDING PASS: per-sample forward + joint 2N-mean loss parity")
    else:
        raise Error("[i4-b2] BINDING FAIL: per-sample loss parity out of tolerance")
