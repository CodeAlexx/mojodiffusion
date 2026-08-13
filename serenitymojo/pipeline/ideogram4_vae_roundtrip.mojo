# pipeline/ideogram4_vae_roundtrip.mojo — GATE 1 for ideogram4 FlowEdit (task #25).
#
# Proves the full latent carrier chain the FlowEdit driver will ride, with NO DiT:
#   staged image [1,3,1024,1024] F32 [-1,1]
#     -> encode_ideogram4_latents            (gated cos 0.99996 vs torch oracle)
#     -> grid [1,128,GH,GW] -> token [1,NIMG,128]   (permute(0,2,3,1)+reshape —
#        the trainer's Ideogram4SampleResident decode precedent, inverse of the
#        ideogram4_generate.mojo:133-135 unpatch)
#     -> token -> grid inverse (exactness check, must be bit-identical)
#     -> denorm (z*latent_scale + latent_shift, [1,1,128] broadcast)
#     -> unpatch [1,GH,GW,2,2,32] -> [1,32,2GH,2GW]
#     -> Flux2-family VAE decode -> PNG + PSNR vs the staged source.
#
# BUILD (same idiom as ideogram4_prepare.mojo — cshim not needed here but harmless):
#   cd /home/alex/mojodiffusion && \
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/ideogram4_vae_roundtrip.mojo -o /tmp/i4_vae_roundtrip
# RUN:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/i4_vae_roundtrip <staged_1024.safetensors> <out.png>
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.math import log, sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, permute, mul, add
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.models.vae.ideogram4_tiled_decode import ideogram4_tiled_decode
from serenitymojo.image.png import save_png
from serenitymojo.models.vae.ldm_encoder import (
    load_ideogram4_vae_encoder,
    encode_ideogram4_latents,
)
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder

comptime I4_VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime I4_LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime HEIGHT = 1024
comptime WIDTH = 1024
comptime LH = HEIGHT // 8      # 128 (VAE latent edge)
comptime LW = WIDTH // 8       # 128
comptime GH = LH // 2          # 64 (packed token grid)
comptime GW = LW // 2          # 64
comptime NIMG = GH * GW        # 4096


def _stats(h: List[Float32]) -> Tuple[Float64, Float64]:
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    return (m, sqrt(vv))


# Encode -> grid -> token; the encoder frees on return (before the decoder
# loads). Also asserts the token<->grid permute pair is an exact inverse.
def _encode_tokens(
    img: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var venc = load_ideogram4_vae_encoder[LH, LW](String(I4_VAE), ctx)
    var z_grid = encode_ideogram4_latents[LH, LW](venc, img, shift, scale, ctx)
    var zst = _stats(z_grid.to_host(ctx))
    print("[gate1] normalized latent mean =", Float32(zst[0]),
          " std =", Float32(zst[1]), " (expect std ~1.0)")

    # ── grid -> token [1,NIMG,128] (the missing ~10-line permute) ──────────────
    var z_hwc = permute(z_grid, [0, 2, 3, 1], ctx)              # [1,GH,GW,128]
    var z_tok = reshape(z_hwc, [1, NIMG, 128], ctx)             # [1,NIMG,128]

    # ── token -> grid inverse: must be BIT-IDENTICAL to z_grid ─────────────────
    var z_back = permute(reshape(z_tok, [1, GH, GW, 128], ctx), [0, 3, 1, 2], ctx)
    var gh_h = z_grid.to_host(ctx)
    var gb_h = z_back.to_host(ctx)
    var max_ad = Float32(0.0)
    for i in range(len(gh_h)):
        var d = gh_h[i] - gb_h[i]
        if d < 0.0:
            d = -d
        if d > max_ad:
            max_ad = d
    print("[gate1] grid->token->grid max |diff| =", max_ad, " (must be 0.0)")
    if max_ad != 0.0:
        raise Error("i4_vae_roundtrip: token/grid permute is NOT an inverse")
    return z_tok^


def main() raises:
    var a = argv()
    if len(a) < 3:
        raise Error("usage: i4_vae_roundtrip <staged_1024.safetensors> <out.png>")
    var src_path = String(a[1])
    var out_png = String(a[2])

    var ctx = DeviceContext()

    # ── staged source [1,3,1024,1024] F32 [-1,1] -> BF16 (probe convention) ────
    var imgs = ShardedSafeTensors.open(src_path)
    var img_f32 = Tensor.from_view(imgs.tensor_view("image"), ctx)
    var sh = img_f32.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 3 or sh[2] != HEIGHT or sh[3] != WIDTH:
        raise Error("i4_vae_roundtrip: staged image must be [1,3,1024,1024] F32")
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)

    # ── encode -> packed normalized grid [1,128,GH,GW] F32 ─────────────────────
    var ln = ShardedSafeTensors.open(String(I4_LATENTNORM))
    var shift = Tensor.from_view(ln.tensor_view("latent_shift"), ctx)   # [128] F32
    var scale = Tensor.from_view(ln.tensor_view("latent_scale"), ctx)   # [128] F32
    var z_tok = _encode_tokens(img, shift, scale, ctx)          # [1,NIMG,128] F32
    ctx.synchronize()
    cu_mempool_trim_current(0)

    # ── denorm + unpatch (exact ideogram4_generate.mojo:129-135 tail) ──────────
    var scale_r = reshape(scale, [1, 1, 128], ctx)
    var shift_r = reshape(shift, [1, 1, 128], ctx)
    var zd = add(mul(z_tok, scale_r, ctx), shift_r, ctx)        # [1,NIMG,128] F32
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)               # [1,32,GH,2,GW,2]
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)      # [1,32,128,128]

    # ── decode + PSNR vs source (whole if VRAM allows, else 3x3 tiled — the
    #    ideogram4_backend policy; whole 1024 needs ~14 GiB free after the DiT
    #    on 24GB; on this 16GB card the tiled path is the expected route) ───────
    var latent_bf = cast_tensor(latent, STDtype.BF16, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)
    var mem = cu_mem_get_info()
    var free_gib = Float64(mem.free_bytes) / 1073741824.0
    var out: Tensor
    if mem.free_bytes > 14 * 1024 * 1024 * 1024:
        print("[gate1] WHOLE-image decode (free =", Float32(free_gib), "GiB)")
        var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](String(I4_VAE), ctx)
        out = dec.decode(latent_bf, ctx)                        # [1,3,H,W] [-1,1]
    else:
        print("[gate1] TILED 3x3 decode (free =", Float32(free_gib),
              "GiB < 14 GiB whole-decode bar)")
        out = ideogram4_tiled_decode[2 * GH, 2 * GW](latent_bf, String(I4_VAE), ctx)
    save_png(out, out_png, ctx)

    var src_h = img_f32.to_host(ctx)
    var out_h = out.to_host(ctx)
    var n = len(src_h)
    var se = 0.0
    var ad = 0.0
    for i in range(n):
        # map [-1,1] -> [0,1] so PSNR uses the standard unit dynamic range
        var d = Float64(src_h[i] - out_h[i]) * 0.5
        se += d * d
        ad += d if d >= 0.0 else -d
    var mse = se / Float64(n)
    var mad = ad / Float64(n)
    var psnr = 10.0 * log(1.0 / mse) / log(10.0)
    print("[gate1] VAE roundtrip 1024x1024: PSNR =", Float32(psnr),
          "dB  MAD([0,1]) =", Float32(mad))
    print("[gate1] wrote", out_png)
