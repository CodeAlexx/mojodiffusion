"""Krea-2 TEACHER-FORCED per-block REAL-weight parity oracle (chunk 7b, RESIDENT).

Builds a COMPACT derived oracle for a few chosen SingleStreamBlocks from the REAL
raw.safetensors, reusing the existing per-block taps. For each chosen block i it
dumps: the block's 13 REAL weights (read SELECTIVELY from raw.safetensors — one
block at a time, ~0.9 GB), the teacher-forced x_in[i] and x_out[i] (the reference's
own per-block input/output, sliced to the 23 real tokens), the shared tvec, and
cos/sin [256,64] EXTRACTED from the reference's freqs tap (cos=freqs[...,0,0],
sin=freqs[...,1,0]). The Mojo probe runs krea2_single_stream_block on the
BYTE-IDENTICAL x_in and compares x_out per-channel (NOT just cos — cos is
magnitude-blind and hid the Rust outlier-channel bug on ch 2569/3389).

NO full forward, NO offload — only selective weight reads + tensor slicing. SAFE
for a remote session (no GPU model load at all; pure safetensors I/O).

Run:
    /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_block_tf_real.py
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RAW = "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
TAPS = "/home/alex/EriTrainer/trainer/parity/krea2_forward/block_taps.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_block_tf_real_oracle.safetensors"

# blocks tapped are 0..19; sample early/mid/late to localize any divergence.
BLOCKS = [0, 13, 19]

# the 13 weights per SingleStreamBlock (mmdit.py).
WEIGHT_SUFFIXES = [
    "mod.lin",
    "prenorm.scale",
    "postnorm.scale",
    "attn.wq.weight",
    "attn.wk.weight",
    "attn.wv.weight",
    "attn.gate.weight",
    "attn.wo.weight",
    "attn.qknorm.qnorm.scale",
    "attn.qknorm.knorm.scale",
    "mlp.gate.weight",
    "mlp.up.weight",
    "mlp.down.weight",
]


def main():
    dump = {}

    # taps: x_in/x_out (sliced to 23 real tokens), tvec, freqs.
    with safe_open(TAPS, "pt", device="cpu") as tf:
        tvec = tf.get_tensor("tvec")          # [1,1,36864] f32
        freqs = tf.get_tensor("freqs")        # [1,256,64,2,2] f32
        dump["tvec"] = tvec.float().contiguous()
        # cos = freqs[...,0,0], sin = freqs[...,1,0]  -> [256, 64]
        cos = freqs[0, :, :, 0, 0].contiguous().float()   # [256, 64]
        sin = freqs[0, :, :, 1, 0].contiguous().float()
        dump["cos"] = cos
        dump["sin"] = sin
        for i in BLOCKS:
            dump[f"x_in.{i}"] = tf.get_tensor(f"x_in.{i}").float().contiguous()    # [1,23,6144]
            dump[f"x_out.{i}"] = tf.get_tensor(f"x_out.{i}").float().contiguous()  # [1,23,6144]
            dump[f"attn_out.{i}"] = tf.get_tensor(f"attn_out.{i}").float().contiguous()  # [1,23,6144]
            dump[f"mlp_out.{i}"] = tf.get_tensor(f"mlp_out.{i}").float().contiguous()    # [1,23,6144]
        # record the real-token length
        l_real = dump[f"x_in.{BLOCKS[0]}"].shape[1]
    print(f"taps: tvec={tuple(tvec.shape)} freqs={tuple(freqs.shape)} L_REAL={l_real}", flush=True)

    # weights: read ONLY the chosen blocks' tensors from raw.safetensors (selective).
    with safe_open(RAW, "pt", device="cpu") as rf:
        for i in BLOCKS:
            for suf in WEIGHT_SUFFIXES:
                key = f"blocks.{i}.{suf}"
                t = rf.get_tensor(key)
                # keep dtype as stored (bf16 weights, f32 scales/mod.lin)
                dump[f"w.{key}"] = t.contiguous()
            print(f"  loaded block {i} weights ({len(WEIGHT_SUFFIXES)})", flush=True)

    dump["meta_blocks"] = torch.tensor(BLOCKS, dtype=torch.int32)
    dump["meta_lreal"] = torch.tensor([l_real], dtype=torch.int32)
    save_file(dump, OUT)
    import os
    print(
        f"OK dumped {len(dump)} tensors (blocks {BLOCKS}) -> {OUT} "
        f"({os.path.getsize(OUT)/1e6:.1f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
