"""Krea-2 TEACHER-FORCED per-block REAL-weight parity oracle (chunk 7b, RESIDENT).

For each SingleStreamBlock i in 0..19 (the tapped blocks) it dumps, from the REAL
raw.safetensors (read SELECTIVELY, one block at a time): the block's 13 weights,
the teacher-forced x_in[i] (the reference's own per-block input, sliced to the 23
real tokens), and TWO references for x_out[i]:
  - xout_math.i : the block run in BF16 with the MATH SDPA backend (== the cuDNN
                  production tap at the block level, cos 0.99999). THE ARBITER —
                  matches torch's actual bf16 dtype flow.
  - xout_f32.i  : the block run in FULL F32 (for characterizing bf16 fragility on
                  the outlier channels ch2569/3389; NOT the gate).
Plus the shared tvec and cos/sin [256,64] (from the reference freqs tap).

The Mojo probe runs krea2_single_stream_block on the BYTE-IDENTICAL x_in and
compares x_out per-channel (cos is magnitude-blind and hid the Rust outlier-
channel bug on ch 2569/3389).

NO full forward, NO offload — selective weight reads + a single-block forward per
block (resident, ~0.9 GB + activations). SAFE for a remote session.

Run:
    /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_block_tf_real.py
"""

import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
import mmdit  # noqa: E402
from mmdit import SingleStreamBlock, SingleMMDiTConfig, _mask  # noqa: E402

RAW = "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
TAPS = "/home/alex/EriTrainer/trainer/parity/krea2_forward/block_taps.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_block_tf_real_oracle.safetensors"
DEV = "cuda"

BLOCKS = list(range(20))  # all 20 tapped blocks

WEIGHT_SUFFIXES = [
    "mod.lin", "prenorm.scale", "postnorm.scale",
    "attn.wq.weight", "attn.wk.weight", "attn.wv.weight",
    "attn.gate.weight", "attn.wo.weight",
    "attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale",
    "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight",
]

CFG = SingleMMDiTConfig(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
    layers=1, patch=2, channels=16, txtheads=20, txtkvheads=20, txtlayers=12,
)


def _attention_math(q, k, v, mask=None, scale=None, gqa=False):
    with sdpa_kernel(SDPBackend.MATH):
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)
    return rearrange(x, "B H L D -> B L (H D)")


def _build_block(bi, dtype, sd):
    blk = SingleStreamBlock(CFG.features, CFG.heads, CFG.multiplier, CFG.bias, CFG.kvheads)
    blk.load_state_dict(sd, strict=True)
    return blk.to(DEV, dtype).eval()


def main():
    from safetensors.torch import load_file
    taps = load_file(TAPS)
    dump = {"tvec": taps["tvec"].float().contiguous()}
    freqs = taps["freqs"]
    dump["cos"] = freqs[0, :, :, 0, 0].contiguous().float()   # [256,64]
    dump["sin"] = freqs[0, :, :, 1, 0].contiguous().float()
    tvec_dev = taps["tvec"].to(DEV)
    freqs_dev = taps["freqs"].to(DEV)
    keep = torch.zeros(1, 256, device=DEV, dtype=torch.bool)
    keep[0, :23] = True
    m = _mask(keep)

    # the reference forces cuDNN; force MATH so the bf16 reference is the math the
    # Mojo masked sdpa faithfully implements (== cuDNN at the block level).
    mmdit.attention = _attention_math

    with safe_open(RAW, "pt", device="cpu") as rf:
        for i in BLOCKS:
            sd = {s: rf.get_tensor(f"blocks.{i}.{s}") for s in WEIGHT_SUFFIXES}
            for s in WEIGHT_SUFFIXES:
                dump[f"w.blocks.{i}.{s}"] = sd[s].contiguous()
            dump[f"x_in.{i}"] = taps[f"x_in.{i}"].float().contiguous()  # [1,23,6144]
            x_in23 = taps[f"x_in.{i}"].to(DEV)

            # bf16 reference (ARBITER): block in bf16, MATH sdpa
            blk_bf = _build_block(i, torch.bfloat16, sd)
            x_pad = torch.zeros(1, 256, 6144, device=DEV, dtype=torch.bfloat16)
            x_pad[:, :23, :] = x_in23.bfloat16()
            with torch.no_grad():
                y_bf = blk_bf(x_pad, tvec_dev.bfloat16(), freqs_dev.bfloat16(), m)[:, :23, :].float()
            dump[f"xout_math.{i}"] = y_bf.cpu().contiguous()

            # F32 reference (characterization)
            blk_f = _build_block(i, torch.float32, sd)
            x_pad32 = torch.zeros(1, 256, 6144, device=DEV, dtype=torch.float32)
            x_pad32[:, :23, :] = x_in23.float()
            with torch.no_grad():
                y_f = blk_f(x_pad32, tvec_dev.float(), freqs_dev.float(), m)[:, :23, :].float()
            dump[f"xout_f32.{i}"] = y_f.cpu().contiguous()
            print(f"  block {i}: weights + xout_math + xout_f32 done", flush=True)

    dump["meta_blocks"] = torch.tensor(BLOCKS, dtype=torch.int32)
    save_file(dump, OUT)
    import os
    print(f"OK dumped {len(dump)} tensors (blocks {BLOCKS}) -> {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
