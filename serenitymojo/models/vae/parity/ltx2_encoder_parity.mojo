# ltx2_encoder_parity.mojo — full LTX-2.3 video VAE ENCODER parity vs the oracle.
#
# Loads the LTX-2.3 encoder (vae.encoder.* + vae.per_channel_statistics.*),
# encodes the FIXED video the oracle dumped (ltx2enc_video_9x64x64.bin,
# NCDHW [1,3,9,64,64] in [-1,1]), and compares the NORMALIZED mean latent to the
# oracle (ltx2enc_moments_9x64x64.bin, [1,128,2,2,2]) AND the raw mean latent to
# (ltx2enc_raw_9x64x64.bin) with the foundation ParityHarness (cos + max_abs).
#
# Oracle: parity/ltx2_encoder_oracle.py — a faithful torch transcription of
# inference-flame/src/vae/ltx2_encoder.rs run in BF16 on GPU using the SAME 2.3
# checkpoint (the HF diffusers LTX-2 VAE is a different architecture, so the Rust
# forward is the spec).
#
# DTYPE: weights BF16, forward BF16 (conv3d F32-accumulate, pixel_norm/normalize
# F32), input fed BF16 — matches Rust/oracle.
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity/ltx2_encoder_parity.mojo
# DEV-ONLY: Python never runs here. The .bin files are static host references.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.vae.ltx2_vae_encoder import (
    LTX2VaeEncoderWeights,
    encode,
    encode_raw,
)


comptime CKPT = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
)
comptime PARITY_DIR = (
    "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
)
comptime T = 9
comptime H = 64
comptime W = 64


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("bad bin size for ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def _std(v: List[Float32]) -> Float32:
    var n = len(v)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += v[i]
    mean /= Float32(n)
    var var_ = Float32(0.0)
    for i in range(n):
        var d = v[i] - mean
        var_ += d * d
    var_ /= Float32(n)
    return sqrt(var_)


def _l2(v: List[Float32]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(v)):
        s += Float64(v[i]) * Float64(v[i])
    return sqrt(s)


def main() raises:
    var ctx = DeviceContext()
    var tag = String(T) + "x" + String(H) + "x" + String(W)

    print("[parity] loading LTX-2.3 VAE encoder from", CKPT)
    var weights = LTX2VaeEncoderWeights.load(String(CKPT), ctx)
    print("[parity] encoder weights loaded")

    # Fixed oracle video NCDHW [1,3,T,H,W] in [-1,1]. Fed as BF16 (Rust dtype).
    var vidvals = _read_f32_bin(
        String(PARITY_DIR) + "/ltx2enc_video_" + tag + ".bin"
    )
    var vsh = List[Int]()
    vsh.append(1); vsh.append(3); vsh.append(T); vsh.append(H); vsh.append(W)
    var video = Tensor.from_host(vidvals, vsh^, STDtype.BF16, ctx)
    print("[parity] video std:", _std(vidvals))

    # ── raw mean latent parity ────────────────────────────────────────────────
    print("[parity] encoding (raw) ...")
    var raw = encode_raw(weights, video, ctx)
    var rs = raw.shape()
    print("[parity] raw latent shape:",
          rs[0], rs[1], rs[2], rs[3], rs[4])
    var raw_host = raw.to_host(ctx)
    print("[parity] raw latent std:", _std(raw_host))
    var raw_ref = _read_f32_bin(
        String(PARITY_DIR) + "/ltx2enc_raw_" + tag + ".bin"
    )
    print("[parity] raw ref std:", _std(raw_ref))
    var harness = ParityHarness(0.99)
    var res_raw = harness.compare(raw, raw_ref, ctx)
    var mag_raw = _l2(raw_host) / _l2(raw_ref)
    print("[parity] RAW MEAN latent:", res_raw)
    print("[parity] RAW magRatio (|out|/|ref|):", mag_raw)

    # ── normalized moments parity ─────────────────────────────────────────────
    print("[parity] encoding (normalized) ...")
    var norm = encode(weights, video, ctx)
    var norm_host = norm.to_host(ctx)
    print("[parity] norm latent std:", _std(norm_host))
    var norm_ref = _read_f32_bin(
        String(PARITY_DIR) + "/ltx2enc_moments_" + tag + ".bin"
    )
    print("[parity] norm ref std:", _std(norm_ref))
    var res_norm = harness.compare(norm, norm_ref, ctx)
    var mag_norm = _l2(norm_host) / _l2(norm_ref)
    print("[parity] NORMALIZED moments:", res_norm)
    print("[parity] NORM magRatio (|out|/|ref|):", mag_norm)
