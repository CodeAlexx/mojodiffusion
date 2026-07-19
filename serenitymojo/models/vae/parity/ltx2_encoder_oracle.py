# ltx2_encoder_oracle.py — torch reference for the LTX-2.3 Video VAE ENCODER.
#
# This is a FAITHFUL transcription of inference-flame/src/vae/ltx2_encoder.rs
# (the Rust ground truth) run in BF16 on GPU, using the SAME 2.3 checkpoint the
# Mojo encoder loads (vae.encoder.* + vae.per_channel_statistics.*).
#
# Why not diffusers AutoencoderKLLTXVideo? The HF LTX-2 diffusers VAE is a
# DIFFERENT architecture (conv_out [129,2048], `downsamplers`/`resnets`/`mid_block`
# key naming) than the production 2.3 checkpoint (conv_out [129,1024], down_blocks
# with res_blocks + SpaceToDepth). The Rust encoder IS the spec for the 2.3 file,
# so the faithful oracle replicates the Rust forward exactly on the 2.3 weights.
#
# DTYPE: weights cast to BF16, forward in BF16 (pixel_norm + stats in F32 then
# cast back, exactly like Rust). Input video fed as BF16. Matches the Rust path
# (cudnn_conv2d_bf16, F32-accumulate inside the conv) as closely as torch allows.
#
# Dumps (parity dir):
#   ltx2enc_video_<T>x<H>x<W>.bin    [1,3,T,H,W] fixed input video in [-1,1] (F32)
#   ltx2enc_moments_<...>.bin        [1,128,T',H',W'] normalized mean latent (F32)
#   ltx2enc_raw_<...>.bin            [1,128,T',H',W'] raw (un-normalized) mean (F32)
#   ltx2enc_meta_<...>.json          shapes + stats
#
# DEV-ONLY parity oracle. Never shipped. Run with system python3 (torch cu128).

import json
import struct
import numpy as np
import torch
import torch.nn.functional as F

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
OUTDIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"

PATCH_SIZE = 4
LATENT_CH = 128
PIXEL_NORM_EPS = 1e-6
ENCODER_NORM_OUT_EPS = 1e-8

DEV = "cuda"
WDT = torch.bfloat16

# Block schedule — exact mirror of ENCODER_BLOCKS in ltx2_encoder.rs.
# ("mid", channels, n_res) | ("s2d", in_ch, out_ch, (st,sh,sw))
ENCODER_BLOCKS = [
    ("mid", 128, 4),
    ("s2d", 128, 256, (1, 2, 2)),
    ("mid", 256, 6),
    ("s2d", 256, 512, (2, 1, 1)),
    ("mid", 512, 4),
    ("s2d", 512, 1024, (2, 2, 2)),
    ("mid", 1024, 2),
    ("s2d", 1024, 1024, (2, 2, 2)),
    ("mid", 1024, 2),
]


def read_safetensors_filtered(path, pred):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = f.tell()
        out = {}
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            if not pred(k):
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            dt = v["dtype"]
            shp = v["shape"]
            if dt == "BF16":
                arr = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shp)
            elif dt == "F32":
                arr = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shp)
            elif dt == "F16":
                arr = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shp)
            else:
                raise ValueError(f"unhandled dtype {dt} for {k}")
            out[k] = arr.clone()
    return out


def w(weights, key):
    if key not in weights:
        raise KeyError(f"missing weight: {key}")
    return weights[key].to(DEV, WDT)


# ── CausalConv3d: left-only temporal replicate pad (causal=True), symmetric
#    spatial pad. ltx2_encoder.rs:127-199. ───────────────────────────────────
def causal_conv3d(x, weight, bias, k=3):
    # x: [B,Cin,D,H,W]
    d = x.shape[2]
    if d == 0:
        return x
    time_pad = k - 1  # left-only
    if time_pad > 0:
        first = x[:, :, 0:1]
        first_rep = first.repeat(1, 1, time_pad, 1, 1)
        x = torch.cat([first_rep, x], dim=2)
    h_pad = k // 2
    w_pad = k // 2
    # conv3d with pad_d=0 (already padded left), symmetric spatial pad.
    out = F.conv3d(x, weight, bias=bias, stride=1, padding=(0, h_pad, w_pad))
    return out


def pixel_norm(x, eps):
    xf = x.float()
    msq = (xf * xf).mean(dim=1, keepdim=True)
    denom = torch.rsqrt(msq + eps)
    return (xf * denom).to(x.dtype)


def resnet_block(weights, prefix, x):
    h = pixel_norm(x, PIXEL_NORM_EPS)
    h = F.silu(h)
    h = causal_conv3d(h, w(weights, f"{prefix}.conv1.conv.weight"),
                      w(weights, f"{prefix}.conv1.conv.bias"))
    h = pixel_norm(h, PIXEL_NORM_EPS)
    h = F.silu(h)
    h = causal_conv3d(h, w(weights, f"{prefix}.conv2.conv.weight"),
                      w(weights, f"{prefix}.conv2.conv.bias"))
    return x + h


def space_to_depth(x, stride):
    # [B,C,F,H,W] -> [B,C*p1*p2*p3, F/p1,H/p2,W/p3]; ltx2_encoder.rs:350-363
    b, c, f, h, ww = x.shape
    p1, p2, p3 = stride
    f2, h2, w2 = f // p1, h // p2, ww // p3
    y = x.reshape(b, c, f2, p1, h2, p2, w2, p3)
    y = y.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return y.reshape(b, c * p1 * p2 * p3, f2, h2, w2)


def s2d_downsample(weights, prefix, x, in_ch, out_ch, stride):
    prod = stride[0] * stride[1] * stride[2]
    group_size = (in_ch * prod) // out_ch
    st = stride[0]
    # 1. causal temporal pad (prepend st-1 first frames)
    if st > 1:
        first = x[:, :, 0:1]
        first_rep = first.repeat(1, 1, st - 1, 1, 1)
        xp = torch.cat([first_rep, x], dim=2)
    else:
        xp = x
    # 2. residual = space_to_depth(xp) -> group average
    residual = space_to_depth(xp, stride)
    if group_size > 1:
        rb, rc, rt, rh, rw = residual.shape
        n_groups = rc // group_size
        residual = residual.reshape(rb, n_groups, group_size, rt, rh, rw).mean(dim=2)
    # 3. main = causal_conv3d(xp) -> space_to_depth
    conv_out = causal_conv3d(xp, w(weights, f"{prefix}.conv.conv.weight"),
                             w(weights, f"{prefix}.conv.conv.bias"))
    main = space_to_depth(conv_out, stride)
    return main + residual


def patchify(x):
    # [B,3,T,H,W] -> [B,48,T,H/4,W/4]; ltx2_encoder.rs:379-393
    b, c, t, h, ww = x.shape
    p = PATCH_SIZE
    hp, wp = h // p, ww // p
    y = x.reshape(b, c, t, 1, hp, p, wp, p)
    y = y.permute(0, 1, 3, 7, 5, 2, 4, 6).contiguous()
    return y.reshape(b, c * p * p, t, hp, wp)


def encode(weights, video):
    h = patchify(video)
    h = causal_conv3d(h, w(weights, "vae.encoder.conv_in.conv.weight"),
                      w(weights, "vae.encoder.conv_in.conv.bias"))
    for i, spec in enumerate(ENCODER_BLOCKS):
        prefix = f"vae.encoder.down_blocks.{i}"
        if spec[0] == "mid":
            _, channels, n_res = spec
            for r in range(n_res):
                h = resnet_block(weights, f"{prefix}.res_blocks.{r}", h)
        else:
            _, in_ch, out_ch, stride = spec
            h = s2d_downsample(weights, prefix, h, in_ch, out_ch, stride)
    # norm_out (eps 1e-8) + SiLU
    h = pixel_norm(h, ENCODER_NORM_OUT_EPS)
    h = F.silu(h)
    # conv_out -> [B,129,...]
    h = causal_conv3d(h, w(weights, "vae.encoder.conv_out.conv.weight"),
                      w(weights, "vae.encoder.conv_out.conv.bias"))
    # expand last channel: repeat 127x, concat -> 256ch
    n_ch = h.shape[1]  # 129
    last = h[:, n_ch - 1:n_ch]
    repeated = last.repeat(1, n_ch - 2, 1, 1, 1)  # 127
    h = torch.cat([h, repeated], dim=1)  # 256
    # take mean = first 128 channels
    mu = h[:, 0:LATENT_CH]
    return mu


def normalize(weights, mu):
    std = w(weights, "vae.per_channel_statistics.std-of-means").reshape(1, LATENT_CH, 1, 1, 1).float()
    mean = w(weights, "vae.per_channel_statistics.mean-of-means").reshape(1, LATENT_CH, 1, 1, 1).float()
    xf = mu.float()
    return ((xf - mean) / std).to(mu.dtype)


def dump_bin(path, t):
    a = t.detach().to("cpu", torch.float32).contiguous().numpy().astype("<f4")
    with open(path, "wb") as f:
        f.write(a.tobytes())


def main():
    torch.manual_seed(1234)
    # LTX uses the "1 + 8k" frame convention (causal). T=9 keeps every temporal
    # space_to_depth divisible after the (st-1)-frame causal pad. -> T'=2.
    T, H, W = 9, 64, 64
    print(f"[oracle] loading LTX-2.3 encoder weights from {CKPT}")
    weights = read_safetensors_filtered(
        CKPT,
        lambda k: k.startswith("vae.encoder.") or k.startswith("vae.per_channel_statistics."),
    )
    print(f"[oracle] loaded {len(weights)} tensors")

    # Fixed non-degenerate video in [-1,1], NCDHW [1,3,T,H,W].
    g = torch.Generator().manual_seed(1234)
    video = (torch.rand(1, 3, T, H, W, generator=g) * 2.0 - 1.0)
    video = video.to(DEV, WDT)

    with torch.no_grad():
        mu = encode(weights, video)            # [1,128,T',H',W'] bf16
        norm = normalize(weights, mu)          # normalized

    tag = f"{T}x{H}x{W}"
    print(f"[oracle] mu shape {tuple(mu.shape)}  norm shape {tuple(norm.shape)}")

    dump_bin(f"{OUTDIR}/ltx2enc_video_{tag}.bin", video)
    dump_bin(f"{OUTDIR}/ltx2enc_raw_{tag}.bin", mu)
    dump_bin(f"{OUTDIR}/ltx2enc_moments_{tag}.bin", norm)

    mu_f = mu.float()
    norm_f = norm.float()
    meta = {
        "T": T, "H": H, "W": W,
        "lat_shape": list(mu.shape),
        "raw_mean": float(mu_f.mean()), "raw_std": float(mu_f.std()),
        "norm_mean": float(norm_f.mean()), "norm_std": float(norm_f.std()),
        "video_mean": float(video.float().mean()), "video_std": float(video.float().std()),
    }
    with open(f"{OUTDIR}/ltx2enc_meta_{tag}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("[oracle] meta:", json.dumps(meta, indent=1))
    print("[oracle] done.")


if __name__ == "__main__":
    main()
