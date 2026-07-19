#!/usr/bin/env python3
"""Pre-dequantize the LTX-2.3 distilled FP8 DiT to BF16 (streaming writer).

Why: the LTX-2 reference's streaming loader cannot consume this checkpoint's
scaled-mm-style export (`.input_scale` keys -> KeyError), and its fp8-cast
policy re-downcasts the dequantized weights to UNSCALED fp8 (61% of weights
fall below the e4m3 normal range -> 3.5% relL2 weight error). The faithful
oracle weighting is exactly what the Mojo side computes per block:
    w_bf16 = (w_fp8.float() * weight_scale).to(bfloat16)     (input_scale unused)
so we export that once and run the reference with quantization=None
(plain bf16 streaming, which the reference supports).

Only `model.diffusion_model.*` keys are kept (the DiffusionStage transformer);
`*_scale` companions are dropped (consumed by the fold). VAE / decoders keep
using the original checkpoint.

Run:
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_dequant_fp8_to_bf16.py \
      [src] [dst]
"""
import json
import struct
import sys

import torch
import safetensors

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
DST = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"

_DT = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16",
       torch.int64: "I64", torch.int32: "I32", torch.uint8: "U8"}


def main() -> None:
    with open(SRC, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        src_hdr = json.loads(f.read(n))
    meta = src_hdr.get("__metadata__", {})

    keep = []
    scales = set()
    for k in src_hdr:
        if k == "__metadata__":
            continue
        if not k.startswith("model.diffusion_model."):
            continue
        if k.endswith("_scale"):
            scales.add(k)
            continue
        keep.append(k)
    keep.sort()
    print(f"kept {len(keep)} tensors, dropped {len(scales)} scale keys")

    with safetensors.safe_open(SRC, framework="pt", device="cpu") as h:
        # Pass 1: header (shapes + output dtypes).
        entries = {}
        offset = 0
        out_dtypes = {}
        for k in keep:
            info = src_hdr[k]
            shape = info["shape"]
            if info["dtype"] == "F8_E4M3":
                odt, esz = "BF16", 2
                out_dtypes[k] = "deq"
            else:
                odt = info["dtype"]
                esz = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "U8": 1}[odt]
                out_dtypes[k] = "copy"
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * esz
            entries[k] = {"dtype": odt, "shape": shape,
                          "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        hdr = {"__metadata__": {**meta, "dequant": "fp8*weight_scale->bf16 (mojo contract)"}}
        hdr.update(entries)
        hdr_bytes = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        pad = (8 - len(hdr_bytes) % 8) % 8
        hdr_bytes += b" " * pad
        for k in entries:
            entries[k]  # noqa

        n_deq = 0
        with open(DST, "wb") as out:
            out.write(struct.pack("<Q", len(hdr_bytes)))
            out.write(hdr_bytes)
            for i, k in enumerate(keep):
                t = h.get_tensor(k)
                if out_dtypes[k] == "deq":
                    sk = k + "_scale"  # ...weight_scale / ...bias_scale
                    if sk not in scales:
                        raise KeyError(f"fp8 tensor {k} has no sibling {sk}")
                    scale = h.get_tensor(sk)
                    if scale.ndim != 0:
                        raise ValueError(f"non-scalar scale for {k}: {tuple(scale.shape)}")
                    t = (t.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
                    n_deq += 1
                t = t.contiguous()
                out.write(t.view(torch.uint8).numpy().tobytes()
                          if t.dtype == torch.bfloat16 or t.dtype == torch.float16
                          else t.numpy().tobytes())
                if i % 500 == 0:
                    print(f"  {i}/{len(keep)}", flush=True)
        print(f"dequantized {n_deq} fp8 tensors -> {DST}")


if __name__ == "__main__":
    main()
