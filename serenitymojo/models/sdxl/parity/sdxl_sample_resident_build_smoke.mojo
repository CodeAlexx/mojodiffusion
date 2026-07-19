# serenitymojo/models/sdxl/parity/sdxl_sample_resident_build_smoke.mojo
#
# BUILD-ONLY type-check gate for training/sdxl_sample_resident.mojo (MJ-1088 fix).
# sdxl_sample_resident[H,W] is a PARAMETRIC function, so its body — the corrected
# CFG/uncond logic — is only elaborated at an instantiation. This smoke instantiates
# it at a tiny L both ways:
#   1. DEFAULT (no uncond embeds) -> the sampler runs cfg=1.0 (no guidance), the
#      MJ-1088 fix path: it must NOT fabricate a zeros-uncond.
#   2. REAL-CFG (context_uncond/y_uncond supplied) -> the drop-in real-guidance path.
# plus sdxl_decode_latent_to_png[H,W]. Compiling this file proves both paths and the
# new Optional[Tensor] trailing params type-check.
#
# This is BUILD-ONLY: it is guarded by has_accelerator() and, if ever run, would
# need the real SDXL checkpoint + VAE on disk (build_sdxl_real_weights / decode). The
# render itself is driven by the trainer's sample cadence, not by this smoke.
#
# Build (run from /home/alex/mojodiffusion; NEVER -O3):
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath \
#     -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/sdxl/parity/sdxl_sample_resident_build_smoke.mojo \
#     -o output/bin/sdxl_sample_resident_build_smoke

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.collections import List, Optional

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors

from serenitymojo.models.sdxl.real_weights import (
    build_sdxl_real_weights, sdxl_st_C, sdxl_st_Cff, sdxl_st_depth,
)
from serenitymojo.models.sdxl.sdxl_real_train import SdxlRealWeights, N_ST
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, build_sdxl_lora_set,
)
from serenitymojo.training.sdxl_sample_resident import (
    sdxl_sample_resident, sdxl_decode_latent_to_png,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"
comptime VAE = "/home/alex/.serenity/models/vae/sdxl_vae.safetensors"
comptime L = 16
comptime CCTX = 2048
comptime NKV = 77
comptime ADM = 2816
comptime RANK = 4
comptime ALPHA = Float32(4.0)


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _sh(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _build_loras() -> List[SdxlLoraSet]:
    var sets = List[SdxlLoraSet]()
    for i in range(N_ST):
        sets.append(build_sdxl_lora_set(
            sdxl_st_depth(i), sdxl_st_C(i), CCTX, sdxl_st_Cff(i), RANK, ALPHA
        ))
    return sets^


def main() raises:
    comptime if not has_accelerator():
        print("sdxl_sample_resident_build_smoke: GPU required (build-only type gate)")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        var st = SafeTensors.open(String(CKPT))
        var w = build_sdxl_real_weights(st, ctx)
        var lora = _build_loras()

        var context = Tensor.from_host(_zeros(NKV * CCTX), _sh(1, NKV, CCTX), STDtype.F32, ctx)
        var y_shape = List[Int](); y_shape.append(1); y_shape.append(ADM)
        var y = Tensor.from_host(_zeros(ADM), y_shape.copy(), STDtype.F32, ctx)
        var init_noise = _zeros(4 * L * L)

        # (1) DEFAULT no-uncond path -> cfg=1.0 (MJ-1088 fix). The requested cfg=7.5
        # is intentionally ignored (no real uncond) and downgraded to no-guidance.
        var latent = sdxl_sample_resident[L, L](
            w, lora, context.clone(ctx), y.clone(ctx), init_noise.copy(),
            4, Float32(7.5), ctx,
        )

        # (2) REAL-CFG path -> supply real uncond embeds (here zeros only to satisfy
        # the type gate; a driver passes the empty/negative CLIP encode).
        var ctx_uncond = Tensor.from_host(_zeros(NKV * CCTX), _sh(1, NKV, CCTX), STDtype.F32, ctx)
        var y_uncond = Tensor.from_host(_zeros(ADM), y_shape.copy(), STDtype.F32, ctx)
        var latent2 = sdxl_sample_resident[L, L](
            w, lora, context.clone(ctx), y.clone(ctx), init_noise.copy(),
            4, Float32(7.5), ctx,
            Optional[Tensor](ctx_uncond^), Optional[Tensor](y_uncond^),
        )

        sdxl_decode_latent_to_png[L, L](latent^, String(VAE), String("/tmp/_sdxl_build_smoke.png"), ctx)
        print("sdxl_sample_resident_build_smoke: type gate OK (both CFG paths + decode)")
        _ = latent2
