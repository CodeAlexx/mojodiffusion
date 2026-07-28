#!/usr/bin/env python3
"""SquareQ v3 streaming slab builder — bounded RAM, restartable.

Reads a checkpoint shard-by-shard / tensor-by-tensor (never accumulates the
model), quantizes eligible linears to squareq_w4_v1 (H256-rotated rank-R
residual, int4-g64), and writes SHARDED output with a weight-map index +
squareq-plan.json + build-state.json. A killed build resumes at the first
incomplete output shard and produces byte-identical output (per-layer seeds
derive from the layer name, not visit order).

Usage:
  python scripts/squareq_build_slab.py --model klein4b \
      --src models/klein4b/transformer.safetensors \
      --out-dir models/klein4b/squareq_w4_r32 [--rank 32] [--blocks-per-shard 4]

Gate (chunk 1): peak RSS < 8 GB on Klein-4B; kill -9 mid-build -> resume ->
byte-identical outputs; plan cos_w floor reported; sidecar ~0.28x bf16.
"""

import argparse
import hashlib
import os
import re
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squareq import core  # noqa: E402
from squareq.plan import (  # noqa: E402
    atomic_write_json,
    finalize_plan,
    load_state,
    new_plan,
    new_state,
)
from squareq.stwriter import StreamingSafetensorsWriter  # noqa: E402

_ST2TORCH = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
    "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64,
}

MODELS = {
    # include: 2D .weight keys eligible for quantization; group_key: shard
    # affinity (same block -> same shard) for locality + resume granularity.
    "klein4b": {
        "include": re.compile(r"^(double_blocks|single_blocks)\.\d+\..*\.weight$"),
        "group": re.compile(r"^((?:double_blocks|single_blocks)\.\d+)\."),
    },
    "krea2": {
        "include": re.compile(r"^blocks\.\d+\..*\.weight$"),
        "group": re.compile(r"^(blocks\.\d+)\."),
    },
}


def layer_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def eligible(key: str, shape, include) -> bool:
    return (
        bool(include.match(key))
        and len(shape) == 2
        and shape[0] >= 1024
        and shape[1] >= 1024
        and shape[1] % core.HBLOCK == 0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group", type=int, default=64, help="scale group along in (64 or 32)")
    ap.add_argument("--format", default="int4", choices=("int4", "nvfp4"),
                    help="residual codec: int4 (dequant-first, any GPU) or nvfp4 (native FP4 fwd, sm_120)")
    ap.add_argument("--blocks-per-shard", type=int, default=4)
    a = ap.parse_args()
    preset = MODELS[a.model]
    os.makedirs(a.out_dir, exist_ok=True)

    st = os.stat(a.src)
    state_path = os.path.join(a.out_dir, "build-state.json")
    state = load_state(state_path, a.src, st.st_size, int(st.st_mtime))
    if state is None:
        state = new_state(a.src, st.st_size, int(st.st_mtime))
        atomic_write_json(state_path, state)
        print("[squareq-build] fresh build")
    else:
        print(f"[squareq-build] resuming; {len(state['completed_shards'])} shards done")

    # ── pass 1: classify every tensor from metadata only ─────────────────────
    with safe_open(a.src, "pt") as f:
        keys = sorted(f.keys())
        meta = {}
        for k in keys:
            sl = f.get_slice(k)
            meta[k] = (sl.get_shape(), sl.get_dtype())

    plan = new_plan(a.model, a.src, "squareq_nvfp4_v1" if a.format == "nvfp4" else core.FORMAT_TAG, core.HBLOCK, a.group)
    plan["rank_default"] = a.rank
    # Seed from the previous run's partial plan BEFORE any shard work — the
    # per-shard partial write below overwrites the file, so completed shards'
    # stats must be recovered here, not in an end-of-run merge.
    partial_path = os.path.join(a.out_dir, "squareq-plan.partial.json")
    if state["completed_shards"] and os.path.exists(partial_path):
        import json

        with open(partial_path) as f:
            prev = json.load(f)
        plan["layers"].update(prev.get("layers", {}))
        for k in prev.get("passthrough", []):
            if k not in plan["passthrough"]:
                plan["passthrough"].append(k)
    groups: dict = {}  # group name ("" = misc/passthrough) -> [keys]
    for k in keys:
        m = preset["group"].match(k)
        g = m.group(1) if (m and eligible(k, meta[k][0], preset["include"])) else ""
        groups.setdefault(g, []).append(k)

    block_names = sorted(g for g in groups if g)
    shards: list = []  # (shard_filename_stub, [group names])
    for i in range(0, len(block_names), a.blocks_per_shard):
        shards.append(block_names[i : i + a.blocks_per_shard])
    shards.append([""])  # misc/passthrough shard last
    nsh = len(shards)

    weight_map = {}
    shard_files = [f"model-{i + 1:05d}-of-{nsh:05d}.safetensors" for i in range(nsh)]

    # ── pass 2: quantize/copy one tensor at a time, one shard at a time ──────
    for si, groups_in_shard in enumerate(shards):
        fname = shard_files[si]
        fpath = os.path.join(a.out_dir, fname)
        done = fname in state["completed_shards"] and os.path.exists(fpath)
        w = None if done else StreamingSafetensorsWriter(
            fpath, metadata={"format": ("squareq_nvfp4_v1" if a.format == "nvfp4" else core.FORMAT_TAG), "rank": a.rank, "group": a.group}
        )
        # declare (and always account plan/weight_map, even when skipping)
        decls: list = []  # (out_name, src_key, kind)
        for g in groups_in_shard:
            for k in groups[g]:
                shape, dt = meta[k]
                if g and eligible(k, shape, preset["include"]):
                    base = k[: -len(".weight")]
                    o, i_f = shape
                    r = a.rank
                    if a.format == "nvfp4":
                        parts = [
                            (base + ".nvq", torch.uint8, [o, i_f // 2]),
                            (base + ".nvs", torch.uint8, [core.scale_tiles_bytes(o, i_f // 16)]),
                            (base + ".nvg", torch.float32, [1]),
                            (base + ".lora_down", torch.bfloat16, [i_f, r]),
                            (base + ".lora_up", torch.bfloat16, [o, r]),
                        ]
                    else:
                        parts = [
                            (base + ".qweight", torch.uint8, [o, i_f // 2]),
                            (base + ".wscales", torch.bfloat16, [i_f // a.group, o]),
                            (base + ".lora_down", torch.bfloat16, [i_f, r]),
                            (base + ".lora_up", torch.bfloat16, [o, r]),
                        ]
                    for nm, dtp, shp in parts:
                        if w:
                            w.declare(nm, dtp, shp)
                        weight_map[nm] = fname
                    decls.append((base, k, "quant"))
                else:
                    if w:
                        w.declare(k, _ST2TORCH[dt], shape)
                    weight_map[k] = fname
                    decls.append((k, k, "pass"))
        if done:
            # plan entries must still exist: recompute stats? NO — plan is
            # rewritten only from live quantization; on resume we reload the
            # previous plan file to keep completed layers' stats.
            continue
        w.write_header()
        try:
            with safe_open(a.src, "pt") as f:
                for base, k, kind in decls:
                    if kind == "pass":
                        w.write_tensor(k, f.get_tensor(k))
                        if k not in plan["passthrough"]:
                            plan["passthrough"].append(k)
                        continue
                    wt = f.get_tensor(k).float()
                    if a.format == "nvfp4":
                        ld_, lu_ = core.svd_lowrank_init(wt, a.rank, seed=layer_seed(k))
                        resid = (wt.double() - lu_.double() @ ld_.double().t())
                        rrot = core.rht_grouped(resid).float()
                        nvq, nvs, nvg = core.nvfp4_encode_weight_tiled(rrot)
                        back = core.rht_grouped(
                            core.nvfp4_decode_weight_tiled(nvq, nvs, nvg, *wt.shape).double()
                        ).float() + lu_ @ ld_.t()
                        aa, bb = back.flatten().double(), wt.double().flatten()
                        cos_w = float(aa @ bb / (aa.norm() * bb.norm() + 1e-30))
                        bytes_q = nvq.numel() + nvs.numel() + 4 + (ld_.numel() + lu_.numel()) * 2
                        tensors = {
                            "nvq": nvq, "nvs": nvs, "nvg": nvg,
                            "lora_down": ld_.to(torch.bfloat16).contiguous(),
                            "lora_up": lu_.to(torch.bfloat16).contiguous(),
                        }
                        stats = {"out": wt.shape[0], "in": wt.shape[1], "rank": a.rank,
                                 "cos_w": cos_w, "rel_l2": 0.0,
                                 "bytes_q": bytes_q, "bytes_bf16": wt.numel() * 2}
                        suffixes = ("nvq", "nvs", "nvg", "lora_down", "lora_up")
                    else:
                        tensors, stats = core.quantize_layer(
                            wt, rank=a.rank, group=a.group, seed=layer_seed(k)
                        )
                        suffixes = ("qweight", "wscales", "lora_down", "lora_up")
                    for suffix in suffixes:
                        w.write_tensor(base + "." + suffix, tensors[suffix])
                    plan["layers"][k] = {**stats, "format": ("squareq_nvfp4_v1" if a.format == "nvfp4" else core.FORMAT_TAG)}
                    print(
                        f"[squareq-build] shard {si + 1}/{nsh}  {k}"
                        f"  [{stats['out']}x{stats['in']}]  cos_w={stats['cos_w']:.5f}"
                    )
                    del wt, tensors
            w.close()
        except BaseException:
            w.abort()
            raise
        state["completed_shards"].append(fname)
        atomic_write_json(state_path, state)
        # persist plan incrementally so resume keeps completed layers' stats
        atomic_write_json(os.path.join(a.out_dir, "squareq-plan.partial.json"), plan)

    index = {
        "metadata": {"format": core.FORMAT_TAG},
        "weight_map": weight_map,
    }
    atomic_write_json(os.path.join(a.out_dir, "model.safetensors.index.json"), index)
    atomic_write_json(
        os.path.join(a.out_dir, "squareq-plan.json"), finalize_plan(plan)
    )
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    t = plan["totals"]
    print(
        f"[squareq-build] DONE: {t['quantized_layers']} quantized"
        f" (cos_w min {t['cos_w_min']:.5f} mean {t['cos_w_mean']:.5f}),"
        f" {t['passthrough_tensors']} passthrough,"
        f" compression {t['compression']:.3f}x of bf16"
    )


if __name__ == "__main__":
    main()
