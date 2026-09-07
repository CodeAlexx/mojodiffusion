#!/usr/bin/env python3
"""Export a BF16 checkpoint whose quantized layers are the SquareQ slab's
reconstructed weights W_hat (core.reconstruct_weight), passthrough tensors
copied verbatim. Lets the ordinary admitted plan (e.g. FLUX.2 Klein W8A8) run
on exactly what the W4 slab encodes, isolating the storage-fidelity cost of
W4 from any compute-route change. Streams one tensor at a time (17 GB output
for Klein 9B without holding it in RAM).

Usage: squareq_export_reconstructed_bf16.py --slab DIR --out FILE.safetensors
"""
import argparse, json, os, struct, sys
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squareq import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slab", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
plan = json.load(open(os.path.join(a.slab, "squareq-plan.json")))
weight_map = json.load(open(os.path.join(a.slab, "model.safetensors.index.json")))["weight_map"]
DT = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16", torch.int32: "I32", torch.int64: "I64"}

# Pass 1: enumerate output tensors (name, dtype, shape) in a stable order.
entries = []  # (name, dtype_str, shape, source)   source = ("recon", base) | ("pass", shard)
quantized = set(plan["layers"].keys())
for key in sorted(plan["layers"], key=lambda k: k):
    L = plan["layers"][key]
    entries.append((key, "BF16", [L["out"], L["in"]], ("recon", key[:-len(".weight")])))
for key in plan["passthrough"]:
    shard = weight_map[key]
    with safe_open(os.path.join(a.slab, shard), "pt") as f:
        t = f.get_slice(key)
        entries.append((key, DT[t.get_dtype() if hasattr(t, "get_dtype") else torch.bfloat16] if False else None, list(t.get_shape()), ("pass", shard)))
# fill passthrough dtypes properly
fixed = []
for name, dt, shape, src in entries:
    if dt is None:
        with safe_open(os.path.join(a.slab, src[1]), "pt") as f:
            tensor = f.get_tensor(name)
        dt = DT[tensor.dtype]
    fixed.append((name, dt, shape, src))
entries = fixed
sizes = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}
header = {}
offset = 0
for name, dt, shape, _ in entries:
    n = 1
    for d in shape: n *= d
    nbytes = n * sizes[dt]
    header[name] = {"dtype": dt, "shape": shape, "data_offsets": [offset, offset + nbytes]}
    offset += nbytes
header["__metadata__"] = {"source": "squareq_w4_v1 reconstructed", "slab": os.path.abspath(a.slab),
                          "quantized_layers": str(len(quantized))}
hjson = json.dumps(header, separators=(",", ":")).encode()
pad = (8 - len(hjson) % 8) % 8
hjson += b" " * pad
with open(a.out, "wb") as out:
    out.write(struct.pack("<Q", len(hjson))); out.write(hjson)
    done = 0
    for name, dt, shape, src in entries:
        if src[0] == "recon":
            base = src[1]; shard = weight_map[base + ".qweight"]
            with safe_open(os.path.join(a.slab, shard), "pt") as f:
                w_hat = core.reconstruct_weight(f.get_tensor(base + ".qweight"), f.get_tensor(base + ".wscales"),
                                                f.get_tensor(base + ".lora_down"), f.get_tensor(base + ".lora_up"))
            tensor = w_hat.to(torch.bfloat16).contiguous()
        else:
            with safe_open(os.path.join(a.slab, src[1]), "pt") as f:
                tensor = f.get_tensor(name).contiguous()
        assert list(tensor.shape) == shape and DT[tensor.dtype] == dt, name
        out.write(tensor.view(torch.uint8).numpy().tobytes() if tensor.dtype != torch.bfloat16 else tensor.view(torch.int16).numpy().tobytes())
        done += 1
        if done % 20 == 0 or src[0] == "recon" and done <= 3:
            print(f"[recon-export] {done}/{len(entries)} {name}", flush=True)
print("wrote", a.out, os.path.getsize(a.out) / 1e9, "GB")
