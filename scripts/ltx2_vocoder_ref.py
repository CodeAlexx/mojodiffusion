#!/usr/bin/env python3
"""LTX-2.3 BigVGAN vocoder + BWE parity oracle (Plan P4 / P1).

Faithful Python port of the Rust `LTX2VocoderWithBWE::forward`
(inference-flame/src/vae/ltx2_vocoder.rs), loading the REAL vocoder weights from
the distilled checkpoint. Reproduces every op of the Rust REFERENCE tensor path
(activation1d kaiser anti-alias, snake-beta, ConvTranspose1d upsample, AMPBlock1
averaging, BWE compute_mel + bwe_generator + sinc skip + clamp), in BF16 compute
to match the Rust `to_dtype(BF16)` path.

Dumps, all as F32, to output/ltx2_vocoder/vocoder_ref.safetensors:
  mel_in      [1, 2, T, 64]   deterministic test mel-spectrogram input
  wav_ref     [1, 2, L48]     reference 48 kHz stereo waveform (BWE)
  wav16_ref   [1, 2, L16]     reference 16 kHz base-vocoder waveform (pre-BWE)

The Mojo smoke `serenitymojo/pipeline/ltx2_vocoder_smoke.mojo` loads the SAME mel
input, runs the Mojo vocoder, and gates max_abs < 0.01 / cos >= 0.999 on wav_ref.

Run:
  python3 scripts/ltx2_vocoder_ref.py
"""
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_vocoder"
OUT = os.path.join(OUT_DIR, "vocoder_ref.safetensors")

# F32 compute (op-identical to the Rust BF16 path, strictly more accurate) so
# the gate reflects STRUCTURAL correctness, not BF16 conv-accumulation jitter
# — the same doctrine as scripts/ltx2_av_block0_parity.py. The Mojo conv1d
# accumulates in F32; matching the oracle in F32 makes the gate meaningful.
DTYPE = torch.float32
NUM_KERNELS_PER_STAGE = 3
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
DILATIONS = [1, 3, 5]
BASE_UPSAMPLE_RATES = [5, 2, 2, 2, 2, 2]
BWE_UPSAMPLE_RATES = [6, 5, 2, 2, 2]

ACT_UP_REPLICATE_PAD = 5
ACT_UP_SLICE_LEFT = 15
ACT_UP_SLICE_RIGHT = 15
ACT_DOWN_PAD_LEFT = 5
ACT_DOWN_PAD_RIGHT = 6
ACT_RATIO = 2

T_MEL = 8  # small mel input -> base L = 8*320 = 2560 @16k, 7680 @48k; GPU-light


def get_padding(k, d):
    return (k * d - d) // 2


def replicate_pad1d(x, left, right):
    # x: [B, C, L]; edge-replicate on last axis.
    return F.pad(x, (left, right), mode="replicate")


def zero_insert1d(x, stride):
    if stride <= 1:
        return x
    b, c, l = x.shape
    x4 = x.reshape(b, c, l, 1)
    zeros = torch.zeros(b, c, l, stride - 1, dtype=x.dtype, device=x.device)
    cat = torch.cat([x4, zeros], dim=3).reshape(b, c, l * stride)
    return cat[:, :, : (l - 1) * stride + 1]


def snake_beta_fast(x, alpha_exp, inv_beta_eps):
    # x + inv_beta_eps * sin^2(alpha_exp * x); params broadcast [C,1,1].
    ax = x * alpha_exp
    s = torch.sin(ax)
    return x + (s * s) * inv_beta_eps


def precompute_conv_transpose_weight(weight, groups=1):
    # [Cin, Cout/g, K] -> Conv1d [Cout, Cin/g, K]: flip last + swap Cin/Cout/grp.
    c_in, c_out_per_group, k = weight.shape
    c_in_per_group = c_in // groups
    c_out = c_out_per_group * groups
    flipped = torch.flip(weight, dims=[-1])
    grouped = flipped.reshape(groups, c_in_per_group, c_out_per_group, k)
    permuted = grouped.permute(0, 2, 1, 3)
    return permuted.reshape(c_out, c_in_per_group, k).contiguous()


def activation1d(x, alpha_exp, inv_beta_eps, up_filter, down_filter):
    # Reference tensor-op path (ltx2_vocoder.rs:247-285). x: [B,C,L], B==1.
    b, c, l = x.shape
    assert b == 1
    x_bc = x.reshape(b * c, 1, l)
    # upsample 2x
    x_pad = replicate_pad1d(x_bc, ACT_UP_REPLICATE_PAD, ACT_UP_REPLICATE_PAD)
    x_zi = zero_insert1d(x_pad, ACT_RATIO)
    x_padded = F.pad(x_zi, (11, 11))  # zero pad
    y = F.conv1d(x_padded, up_filter, None, stride=1, padding=0)
    y = y * float(ACT_RATIO)
    y_len = y.shape[2]
    y = y[:, :, ACT_UP_SLICE_LEFT : y_len - ACT_UP_SLICE_RIGHT]
    # snake (params [C,1,1] broadcast over [C,1,L'])
    y = snake_beta_fast(y, alpha_exp, inv_beta_eps)
    # downsample 2x
    y_pad = replicate_pad1d(y, ACT_DOWN_PAD_LEFT, ACT_DOWN_PAD_RIGHT)
    out = F.conv1d(y_pad, down_filter, None, stride=ACT_RATIO, padding=0)
    l_out = out.shape[2]
    return out.reshape(b, c, l_out)


class ActParams:
    def __init__(self, w, prefix, dev):
        alpha = w[f"{prefix}.act.alpha"].to(dev, DTYPE)
        beta = w[f"{prefix}.act.beta"].to(dev, DTYPE)
        c = alpha.shape[0]
        self.alpha_exp = alpha.reshape(c, 1, 1).exp()
        beta_exp = beta.reshape(c, 1, 1).exp()
        self.inv_beta_eps = 1.0 / (beta_exp + 1e-9)
        # up/down filters [1,1,12], used as conv weight on folded [B*C,1,L].
        self.up_filter = w[f"{prefix}.upsample.filter"].to(dev, DTYPE)
        self.down_filter = w[f"{prefix}.downsample.lowpass.filter"].to(dev, DTYPE)

    def apply(self, x):
        return activation1d(
            x, self.alpha_exp, self.inv_beta_eps, self.up_filter, self.down_filter
        )


class AmpBlock1:
    def __init__(self, w, prefix, kernel_size, dev):
        self.k = kernel_size
        self.convs1_w = [w[f"{prefix}.convs1.{i}.weight"].to(dev, DTYPE) for i in range(3)]
        self.convs1_b = [w[f"{prefix}.convs1.{i}.bias"].to(dev, DTYPE) for i in range(3)]
        self.convs2_w = [w[f"{prefix}.convs2.{i}.weight"].to(dev, DTYPE) for i in range(3)]
        self.convs2_b = [w[f"{prefix}.convs2.{i}.bias"].to(dev, DTYPE) for i in range(3)]
        self.acts1 = [ActParams(w, f"{prefix}.acts1.{i}", dev) for i in range(3)]
        self.acts2 = [ActParams(w, f"{prefix}.acts2.{i}", dev) for i in range(3)]

    def forward(self, x):
        for i in range(3):
            d = DILATIONS[i]
            pad1 = get_padding(self.k, d)
            xt = self.acts1[i].apply(x)
            xt = F.conv1d(xt, self.convs1_w[i], self.convs1_b[i], stride=1, padding=pad1, dilation=d)
            xt = self.acts2[i].apply(xt)
            pad2 = get_padding(self.k, 1)
            xt = F.conv1d(xt, self.convs2_w[i], self.convs2_b[i], stride=1, padding=pad2, dilation=1)
            x = x + xt
        return x


class Vocoder:
    def __init__(self, w, prefix, apply_final_tanh, dev):
        self.prefix = prefix
        self.apply_final_tanh = apply_final_tanh
        self.conv_pre_w = w[f"{prefix}.conv_pre.weight"].to(dev, DTYPE)
        self.conv_pre_b = w[f"{prefix}.conv_pre.bias"].to(dev, DTYPE)
        num_ups = 0
        while f"{prefix}.ups.{num_ups}.weight" in w.keys():
            num_ups += 1
        self.num_ups = num_ups
        if num_ups == 6:
            self.rates = BASE_UPSAMPLE_RATES
        elif num_ups == 5:
            self.rates = BWE_UPSAMPLE_RATES
        else:
            raise ValueError(f"bad ups count {num_ups}")
        self.ups_preconv = []
        self.ups_bias = []
        self.ups_k = []
        for i in range(num_ups):
            w_raw = w[f"{prefix}.ups.{i}.weight"].to(dev, DTYPE)
            self.ups_k.append(w_raw.shape[2])
            self.ups_preconv.append(precompute_conv_transpose_weight(w_raw, 1))
            self.ups_bias.append(w[f"{prefix}.ups.{i}.bias"].to(dev, DTYPE))
        self.resblocks = []
        for stage_i in range(num_ups):
            for kernel_i in range(NUM_KERNELS_PER_STAGE):
                bidx = stage_i * NUM_KERNELS_PER_STAGE + kernel_i
                k = RESBLOCK_KERNEL_SIZES[kernel_i]
                self.resblocks.append(AmpBlock1(w, f"{prefix}.resblocks.{bidx}", k, dev))
        self.act_post = ActParams(w, f"{prefix}.act_post", dev)
        self.conv_post_w = w[f"{prefix}.conv_post.weight"].to(dev, DTYPE)
        self.conv_post_b = (
            w[f"{prefix}.conv_post.bias"].to(dev, DTYPE)
            if f"{prefix}.conv_post.bias" in w.keys()
            else None
        )

    def forward(self, mel):
        # mel [B,2,T,F] -> [B,2,F,T] -> [B,2F,T]
        b, s, t, f = mel.shape
        assert s == 2
        x = mel.permute(0, 1, 3, 2).reshape(b, 2 * f, t)
        x = F.conv1d(x, self.conv_pre_w, self.conv_pre_b, stride=1, padding=7 // 2)
        for i in range(self.num_ups):
            stride = self.rates[i]
            k = self.ups_k[i]
            padding = (k - stride) // 2
            side = (k - 1) - padding
            x_zi = zero_insert1d(x, stride)
            x_padded = F.pad(x_zi, (side, side))
            x = F.conv1d(x_padded, self.ups_preconv[i], self.ups_bias[i], stride=1, padding=0)
            start = i * NUM_KERNELS_PER_STAGE
            acc = None
            for j in range(NUM_KERNELS_PER_STAGE):
                out = self.resblocks[start + j].forward(x)
                acc = out if acc is None else acc + out
            x = acc * (1.0 / NUM_KERNELS_PER_STAGE)
        x = self.act_post.apply(x)
        x = F.conv1d(x, self.conv_post_w, self.conv_post_b, stride=1, padding=7 // 2)
        if self.apply_final_tanh:
            x = torch.tanh(x)
        return x


def hann_sinc_resample_filter(ratio, dev):
    rolloff = 0.99
    lowpass_filter_width = 6.0
    width = math.ceil(lowpass_filter_width / rolloff)
    kernel_size = 2 * width * ratio + 1
    pad = width
    pad_left = 2 * width * ratio
    pad_right = kernel_size - ratio
    data = []
    for i in range(kernel_size):
        time_axis = (i / ratio - width) * rolloff
        time_clamped = max(-lowpass_filter_width, min(lowpass_filter_width, time_axis))
        window = math.cos(time_clamped * math.pi / lowpass_filter_width / 2.0) ** 2
        if abs(time_axis) < 1e-8:
            sinc = 1.0
        else:
            xx = math.pi * time_axis
            sinc = math.sin(xx) / xx
        data.append(sinc * window * rolloff / ratio)
    filt = torch.tensor(data, dtype=DTYPE, device=dev).reshape(1, 1, kernel_size)
    return filt, pad, pad_left, pad_right


class VocoderWithBWE:
    def __init__(self, w, dev):
        self.dev = dev
        self.vocoder = Vocoder(w, "vocoder.vocoder", True, dev)
        self.bwe = Vocoder(w, "vocoder.bwe_generator", False, dev)
        self.mel_basis = w["vocoder.mel_stft.mel_basis"].to(dev, DTYPE)
        self.forward_basis = w["vocoder.mel_stft.stft_fn.forward_basis"].to(dev, DTYPE)
        self.input_sr = 16000
        self.output_sr = 48000
        self.hop_length = 80
        self.ratio = self.output_sr // self.input_sr
        (
            rf,
            self.resample_pad,
            self.resample_pad_left,
            self.resample_pad_right,
        ) = hann_sinc_resample_filter(self.ratio, dev)
        self.resample_preconv = precompute_conv_transpose_weight(rf, 1)

    def compute_mel(self, audio):
        b, n_ch, t = audio.shape
        flat = audio.reshape(b * n_ch, t)
        win_length = self.forward_basis.shape[2]
        left_pad = win_length - self.hop_length
        flat_padded = F.pad(flat.unsqueeze(1), (left_pad, 0))
        spec = F.conv1d(flat_padded, self.forward_basis, None, stride=self.hop_length)
        n_freqs = spec.shape[1] // 2
        real = spec[:, :n_freqs]
        imag = spec[:, n_freqs:]
        mag = torch.sqrt(real * real + imag * imag)
        n_mels = self.mel_basis.shape[0]
        t_frames = mag.shape[2]
        mag_t = mag.permute(0, 2, 1)
        mel = mag_t.matmul(self.mel_basis.permute(1, 0)).permute(0, 2, 1)
        mel = mel.clamp(1e-5, 1e10).log()
        return mel.reshape(b, n_ch, n_mels, t_frames)

    def sinc_upsample(self, x):
        b, c, l = x.shape
        x_bc = x.reshape(b * c, 1, l)
        x_pad = replicate_pad1d(x_bc, self.resample_pad, self.resample_pad)
        k = self.resample_preconv.shape[2]
        x_zi = zero_insert1d(x_pad, self.ratio)
        x_padded = F.pad(x_zi, (k - 1, k - 1))
        y = F.conv1d(x_padded, self.resample_preconv, None, stride=1, padding=0) * float(self.ratio)
        y_len = y.shape[2]
        crop_len = y_len - self.resample_pad_left - self.resample_pad_right
        y = y[:, :, self.resample_pad_left : self.resample_pad_left + crop_len]
        return y.reshape(b, c, crop_len)

    def forward(self, mel):
        x = self.vocoder.forward(mel)
        wav16 = x.clone()
        length_low_rate = x.shape[2]
        output_length = length_low_rate * self.output_sr // self.input_sr
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))
        mel_bwe = self.compute_mel(x)
        mel_for_bwe = mel_bwe.permute(0, 1, 3, 2)
        residual = self.bwe.forward(mel_for_bwe)
        skip = self.sinc_upsample(x)
        rl = residual.shape[2]
        sl = skip.shape[2]
        mix = min(rl, sl)
        residual = residual[:, :, :mix]
        skip = skip[:, :, :mix]
        out = (residual + skip).clamp(-1.0, 1.0)
        if out.shape[2] > output_length:
            out = out[:, :, :output_length]
        return out, wav16


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith("vocoder."):
                w[k] = f.get_tensor(k)

    class WDict:
        def __init__(self, d):
            self.d = d

        def __getitem__(self, k):
            return self.d[k]

        def keys(self):
            return self.d.keys()

    wd = WDict(w)
    model = VocoderWithBWE(wd, dev)

    # Deterministic mel input [1, 2, T, 64], modest magnitude (log-mel range).
    g = torch.Generator(device="cpu").manual_seed(1234)
    n_mels = model.mel_basis.shape[0]  # 64
    mel_in = (torch.rand(1, 2, T_MEL, n_mels, generator=g) * 8.0 - 6.0)  # ~[-6, 2]
    mel_in_d = mel_in.to(dev, DTYPE)

    with torch.no_grad():
        wav, wav16 = model.forward(mel_in_d)

    print(f"mel_in {tuple(mel_in.shape)}")
    print(f"wav16  {tuple(wav16.shape)} range [{wav16.float().min():.4f}, {wav16.float().max():.4f}]")
    print(f"wav48  {tuple(wav.shape)} range [{wav.float().min():.4f}, {wav.float().max():.4f}]")
    rms = wav.float().pow(2).mean().sqrt().item()
    print(f"wav48 rms {rms:.5f}")

    out = {
        "mel_in": mel_in.float().contiguous(),
        "wav_ref": wav.float().cpu().contiguous(),
        "wav16_ref": wav16.float().cpu().contiguous(),
    }
    save_file(out, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
