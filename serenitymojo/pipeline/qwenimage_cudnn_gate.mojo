# pipeline/qwenimage_cudnn_gate.mojo — A/B gate for the qwenimage VAE cuDNN
# conv fast path (task #16).
#
# Loads the qwenimage VAE encoder + decoder TWICE each — naive-conv path
# (QRSCF + SDK naive kernel, the pre-#16 behavior) and cuDNN path
# (conv3d_fcqrs_cudnn, the wan22/LTX2/LingBot-proven mechanism) — on a REAL
# 512x512 staged image, and reports:
#   * encode parity: cosine + max|diff| between the two [1,16,64,64] latents
#   * decode parity: cosine + max|diff| + PSNR between the two [1,3,512,512]
#     outputs (decoding the SAME normalized latent)
#   * speed: per-op wall time, 2 iterations each (iter0 includes cuDNN algo
#     FindEx; iter1 = steady state)
#
# Run:
#   <bin> [staged_src.safetensors]   (default: the krea2 FlowEdit demo source)
#
# Mojo 1.0.0b1, NVIDIA GPU. Foreground gate — no artifacts written.

from max.gpu.host import DeviceContext
from std.math import sqrt, log
from std.sys import argv
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.models.krea2.krea2_prepare_cache import (
    _mean_ch,
    _std_ch,
    _normalize_latent,
    KREA2_VAE_ENC_FILE,
)
from serenitymojo.pipeline.krea2_paths import KREA2_VAE_DIR


comptime HEIGHT = 512
comptime WIDTH = 512
comptime LH = HEIGHT // 8   # 64
comptime LW = WIDTH // 8    # 64

comptime DEFAULT_SRC = (
    "/home/alex/trainings/krea2_edit_qwen/demo/flowedit_src_512.safetensors"
)


def _stats(a: List[Float32], b: List[Float32]) raises -> Tuple[Float64, Float64, Float64]:
    """(cosine, max|a-b|, mse) in F64."""
    if len(a) != len(b):
        raise Error("stats: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var mad = Float64(0.0)
    var mse = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        if d < 0:
            d = -d
        if d > mad:
            mad = d
        mse += d * d
    mse /= Float64(len(a))
    var cos = dot / (sqrt(na) * sqrt(nb) + 1e-30)
    return (cos, mad, mse)


def _psnr_signed(mse: Float64) -> Float64:
    """PSNR for [-1,1]-range signals (peak-to-peak 2.0 -> peak^2 = 4)."""
    if mse <= 0:
        return Float64(999.0)
    return 10.0 * log(4.0 / mse) / log(Float64(10.0))


def main() raises:
    var src_path = String(DEFAULT_SRC)
    var args = argv()
    if len(args) > 1:
        src_path = String(args[1])

    var ctx = DeviceContext()
    print("[gate] source:", src_path)

    # staged source [1,3,512,512] F32 -> BF16 (encoder convention).
    var imgs = ShardedSafeTensors.open(src_path)
    var img_f32 = Tensor.from_view(imgs.tensor_view(String("image")), ctx)
    var sh = img_f32.shape()
    if len(sh) != 4 or sh[1] != 3 or sh[2] != HEIGHT or sh[3] != WIDTH:
        raise Error("[gate] staged source must be [1,3,512,512]")
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)

    # ── ENCODE: naive vs cuDNN ────────────────────────────────────────────────
    print("[gate] loading encoder (naive) ...")
    var enc_naive = QwenImageVaeEncoder[HEIGHT, WIDTH].load(
        KREA2_VAE_ENC_FILE, ctx, Optional[Bool](False)
    )
    ctx.synchronize()
    var lat_naive: Optional[Tensor] = None
    for it in range(2):
        var t0 = Int(perf_counter_ns())
        var lat = enc_naive.encode_mean(img, ctx)   # [1,16,64,64] BF16
        ctx.synchronize()
        var dt = Float64(Int(perf_counter_ns()) - t0) / 1e9
        print("[gate] encode NAIVE  iter", it, "=", dt, "s")
        lat_naive = Optional[Tensor](lat^)

    print("[gate] loading encoder (cuDNN) ...")
    var enc_cudnn = QwenImageVaeEncoder[HEIGHT, WIDTH].load(
        KREA2_VAE_ENC_FILE, ctx, Optional[Bool](True)
    )
    ctx.synchronize()
    var lat_cudnn: Optional[Tensor] = None
    for it in range(2):
        var t0 = Int(perf_counter_ns())
        var lat = enc_cudnn.encode_mean(img, ctx)
        ctx.synchronize()
        var dt = Float64(Int(perf_counter_ns()) - t0) / 1e9
        print("[gate] encode CUDNN  iter", it, "=", dt, "s")
        lat_cudnn = Optional[Tensor](lat^)

    var ln_f32 = cast_tensor(lat_naive.value(), STDtype.F32, ctx)
    var lc_f32 = cast_tensor(lat_cudnn.value(), STDtype.F32, ctx)
    var enc_stats = _stats(ln_f32.to_host(ctx), lc_f32.to_host(ctx))
    print("[gate] ENCODE parity: cos =", enc_stats[0],
          " max|diff| =", enc_stats[1], " mse =", enc_stats[2])

    # ── DECODE: same normalized latent through both paths ────────────────────
    var mean_ch = _mean_ch(ctx)
    var std_ch = _std_ch(ctx)
    var z_norm = _normalize_latent(lc_f32, mean_ch, std_ch, ctx)  # [1,16,64,64] F32
    var z_bf16 = cast_tensor(z_norm, STDtype.BF16, ctx)

    print("[gate] loading decoder (naive) ...")
    var dec_naive = QwenImageVaeDecoder[LH, LW].load(
        String(KREA2_VAE_DIR), ctx, Optional[Bool](False)
    )
    ctx.synchronize()
    var out_naive: Optional[Tensor] = None
    for it in range(2):
        var t0 = Int(perf_counter_ns())
        var out = dec_naive.decode(z_bf16, ctx)     # [1,3,512,512]
        ctx.synchronize()
        var dt = Float64(Int(perf_counter_ns()) - t0) / 1e9
        print("[gate] decode NAIVE  iter", it, "=", dt, "s")
        out_naive = Optional[Tensor](out^)

    print("[gate] loading decoder (cuDNN) ...")
    var dec_cudnn = QwenImageVaeDecoder[LH, LW].load(
        String(KREA2_VAE_DIR), ctx, Optional[Bool](True)
    )
    ctx.synchronize()
    var out_cudnn: Optional[Tensor] = None
    for it in range(2):
        var t0 = Int(perf_counter_ns())
        var out = dec_cudnn.decode(z_bf16, ctx)
        ctx.synchronize()
        var dt = Float64(Int(perf_counter_ns()) - t0) / 1e9
        print("[gate] decode CUDNN  iter", it, "=", dt, "s")
        out_cudnn = Optional[Tensor](out^)

    var on_f32 = cast_tensor(out_naive.value(), STDtype.F32, ctx)
    var oc_f32 = cast_tensor(out_cudnn.value(), STDtype.F32, ctx)
    var dec_stats = _stats(on_f32.to_host(ctx), oc_f32.to_host(ctx))
    print("[gate] DECODE parity: cos =", dec_stats[0],
          " max|diff| =", dec_stats[1],
          " PSNR =", _psnr_signed(dec_stats[2]), "dB")

    # ── DECODE via wan21-keys loader (anima path: exercises the wan21 residual
    #    shortcut + resample dispatch too) ──────────────────────────────────────
    print("[gate] loading decoder wan21-keys (naive) ...")
    var dw_naive = QwenImageVaeDecoder[LH, LW].load_wan21_keys(
        KREA2_VAE_ENC_FILE, ctx, Optional[Bool](False)
    )
    var ow_naive = dw_naive.decode_wan21_keys(z_bf16, ctx)
    ctx.synchronize()
    print("[gate] loading decoder wan21-keys (cuDNN) ...")
    var dw_cudnn = QwenImageVaeDecoder[LH, LW].load_wan21_keys(
        KREA2_VAE_ENC_FILE, ctx, Optional[Bool](True)
    )
    var t0w = Int(perf_counter_ns())
    var ow_cudnn = dw_cudnn.decode_wan21_keys(z_bf16, ctx)
    ctx.synchronize()
    print("[gate] decode wan21 CUDNN =",
          Float64(Int(perf_counter_ns()) - t0w) / 1e9, "s")
    var own_f32 = cast_tensor(ow_naive, STDtype.F32, ctx)
    var owc_f32 = cast_tensor(ow_cudnn, STDtype.F32, ctx)
    var w21_stats = _stats(own_f32.to_host(ctx), owc_f32.to_host(ctx))
    print("[gate] DECODE-wan21 parity: cos =", w21_stats[0],
          " max|diff| =", w21_stats[1],
          " PSNR =", _psnr_signed(w21_stats[2]), "dB")
    print("[gate] DONE")
