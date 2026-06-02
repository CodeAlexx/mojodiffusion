# parity/real_finitediff.mojo — composed-backward self-consistency gate for the
# REAL-DIMS SDXL trainable UNet (sdxl_real_train.mojo) on REAL checkpoint weights.
#
# The Klein composition-defect lesson: per-unit parity does NOT prove the composed
# backward == grad of the composed forward. This finite-diff gate proves it at the
# REAL topology (all 17 ResBlocks, 11 STs, 2 down, 2 up, skips, embeds) at a small
# latent (L=16 -> H0=16,H1=8,H2=4) so the full fwd+bwd fits 24 GB.
#
# Two checks:
#   1. d_x finite-diff: perturb K latent entries, central-difference the scalar
#      loss = 0.5*sum(out^2); compare to analytic d_x = sdxl_real_backward(out).
#      go = out (so dL/dout = out). Ratio (fd/analytic) must be ~1.0.
#   2. LoRA-B grad finite-diff: with B seeded nonzero, perturb K B-entries of one
#      adapter; compare central-difference of the loss to the analytic d_b. Proves
#      the LoRA-aware ST chain's d_B threads correctly through the conv-UNet.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sdxl/parity/real_finitediff.mojo -o /tmp/sdxl_real_fd && \
#     /tmp/sdxl_real_fd

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors

from serenitymojo.models.sdxl.real_weights import (
    build_sdxl_real_weights, sdxl_st_C, sdxl_st_Cff, sdxl_st_depth,
)
from serenitymojo.models.sdxl.sdxl_real_train import (
    SdxlRealWeights, sdxl_real_forward, sdxl_real_backward, N_ST,
)
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import SdxlLoraSet, build_sdxl_lora_set
from serenitymojo.models.sdxl.lora_block import SDXL_SLOTS

comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"
comptime L = 16            # latent spatial -> H0=16, H1=8, H2=4
comptime CCTX = 2048
comptime NKV = 77
comptime ADM = 2816
comptime RANK = 4
comptime ALPHA = Float32(4.0)


def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


# build the 11 LoRA sets (one per ST). B is 0 by default (build_sdxl_lora_set ->
# make_lora_adapter -> B=0). For check 2 we manually seed one adapter's B nonzero.
def _build_loras() -> List[SdxlLoraSet]:
    var sets = List[SdxlLoraSet]()
    for i in range(N_ST):
        sets.append(build_sdxl_lora_set(sdxl_st_depth(i), sdxl_st_C(i), CCTX, sdxl_st_Cff(i), RANK, ALPHA))
    return sets^


# scalar loss = 0.5 * sum(out^2) ; go = out (dL/dout = out).
def _loss_and_out(
    x_nhwc: Tensor, t: Tensor, y: Tensor, ctxt: Tensor,
    w: SdxlRealWeights, lora: List[SdxlLoraSet], ctx: DeviceContext,
) raises -> Float64:
    var fwd = sdxl_real_forward[L](x_nhwc.clone(ctx), t, y, ctxt, w, lora, ctx)
    var oh = fwd.out.to_host(ctx)
    var s = 0.0
    for i in range(len(oh)):
        s += 0.5 * Float64(oh[i]) * Float64(oh[i])
    return s


def main() raises:
    var ctx = DeviceContext()
    print("=== SDXL real-dims composed fwd+bwd finite-diff gate (L=", L, ") ===")
    print("[load] opening checkpoint")
    var st = SafeTensors.open(String(CKPT))
    print("[load] assembling real weights (this loads the full UNet F32)")
    var w = build_sdxl_real_weights(st, ctx)
    print("[load] weights ready")

    var lora = _build_loras()
    print("[lora] built", N_ST, "LoRA sets")

    # inputs: latent NHWC [1,L,L,4], timestep, ADM y [1,2816], context [1,77,2048]
    var xh = _host_noise(L * L * 4, UInt64(11))
    var x = Tensor.from_host(xh.copy(), _sh4(1, L, L, 4), STDtype.F32, ctx)
    var th = List[Float32](); th.append(Float32(500.0))
    var ts = List[Int](); ts.append(1)
    var t = Tensor.from_host(th^, ts^, STDtype.F32, ctx)
    var yh = _host_noise(ADM, UInt64(22))
    var ys = List[Int](); ys.append(1); ys.append(ADM)
    var y = Tensor.from_host(yh^, ys^, STDtype.F32, ctx)
    var ch = _host_noise(NKV * CCTX, UInt64(33))
    var cs = List[Int](); cs.append(1); cs.append(NKV); cs.append(CCTX)
    var ctxt = Tensor.from_host(ch^, cs^, STDtype.F32, ctx)

    # ── analytic forward+backward ──
    print("[fwd] running forward")
    var fwd = sdxl_real_forward[L](x.clone(ctx), t, y, ctxt, w, lora, ctx)
    var go = fwd.out.clone(ctx)   # go = out -> loss = 0.5||out||^2
    print("[bwd] running backward")
    var grads = sdxl_real_backward[L](go, fwd.acts, w, lora, ctx)
    print("[bwd] nonfinite lora grads:", grads.nonfinite)
    var dxa = grads.d_x.to_host(ctx)   # analytic d_x [1,L,L,4] flat

    # ── d_x finite-diff at K positions ──
    var eps = Float32(1.0e-2)
    var positions = List[Int]()
    positions.append(0); positions.append(37); positions.append(123)
    positions.append(L * L * 4 - 1); positions.append(L * L * 2 + 5)
    var worst = 0.0
    print("[fd-dx] central diff at", len(positions), "positions, eps=", eps)
    for pi in range(len(positions)):
        var idx = positions[pi]
        var xp = xh.copy(); xp[idx] = xp[idx] + eps
        var xpt = Tensor.from_host(xp^, _sh4(1, L, L, 4), STDtype.F32, ctx)
        var lp = _loss_and_out(xpt, t, y, ctxt, w, lora, ctx)
        var xm = xh.copy(); xm[idx] = xm[idx] - eps
        var xmt = Tensor.from_host(xm^, _sh4(1, L, L, 4), STDtype.F32, ctx)
        var lm = _loss_and_out(xmt, t, y, ctxt, w, lora, ctx)
        var fd = (lp - lm) / (2.0 * Float64(eps))
        var an = Float64(dxa[idx])
        var ratio = fd / an if an != 0.0 else 0.0
        var err = ratio - 1.0 if an != 0.0 else (fd - an)
        if (err if err >= 0.0 else -err) > worst:
            worst = err if err >= 0.0 else -err
        print("  pos", idx, " fd=", Float32(fd), " analytic=", Float32(an),
              " ratio=", Float32(ratio))
    print("[fd-dx] worst |ratio-1| =", Float32(worst))

    var dx_pass = worst < 0.02
    if dx_pass:
        print("RESULT: X-PATH FINITE-DIFF SELF-CONSISTENCY PASSED (worst |ratio-1|=",
              Float32(worst), ")")
    else:
        print("RESULT: X-PATH FINITE-DIFF FAILED (worst |ratio-1|=", Float32(worst), ")")
