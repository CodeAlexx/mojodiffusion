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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squareq import core  # noqa: E402
from squareq.plan import (  # noqa: E402
    atomic_write_json,
    finalize_plan,
    load_state,
    new_plan,
    new_state,
)
from squareq.source import open_source, source_identity  # noqa: E402
from squareq.stwriter import StreamingSafetensorsWriter  # noqa: E402

_FORMAT_TAG = {
    "int4": core.FORMAT_TAG,
    "nvfp4": "squareq_nvfp4_v1",
    "int8": core.FORMAT_TAG_W8,
    "convrot_w8": core.FORMAT_TAG_CONVROT,
}

_ST2TORCH = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
    "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64,
}

MODELS = {
    # include: 2D .weight keys eligible for quantization; group_key: shard
    # affinity (same block -> same shard) for locality + resume granularity.
    # exclude: EXPLICIT sensitivity policy (not just filter luck) — AdaLN /
    # modulation / norm projections stay bf16 even if they match include.
    # The builder REPORTS what the policy excluded so it is visible, not silent.
    "klein4b": {
        "include": re.compile(r"^(double_blocks|single_blocks)\.\d+\..*\.weight$"),
        "group": re.compile(r"^((?:double_blocks|single_blocks)\.\d+)\."),
        "exclude": re.compile(r"(_mod\.|modulation|adaln|norm)", re.IGNORECASE),
    },
    "krea2": {
        "include": re.compile(r"^blocks\.\d+\..*\.weight$"),
        "group": re.compile(r"^(blocks\.\d+)\."),
        "exclude": re.compile(r"(\.mod\.|modulation|adaln|norm)", re.IGNORECASE),
    },
    "ltx2": {
        "include": re.compile(r"^model\.diffusion_model\.transformer_blocks\.\d+\..*\.weight$"),
        "group": re.compile(r"^(model\.diffusion_model\.transformer_blocks\.\d+)\."),
        "exclude": re.compile(r"(scale_shift|adaln|norm|gate_logits|modulation)", re.IGNORECASE),
    },
    # MiniMax-H3 FL2VA transformer (50 blocks, hidden 5376). Quantizable set is
    # exactly the four block linears — attn.qkv_proj [21504,5376],
    # attn.out_proj [5376,7168], mlp.fc1 [28672,5376], mlp.fc2 [5376,14336] —
    # all with in % 256 == 0. Everything else is named EXPLICITLY in `exclude`
    # so the policy report shows it: adaln_proj (AdaLN modulation, [96768,2688]),
    # norm1/norm2 + attn.q_norm/k_norm (RMSNorm gains), video/audio_patch_proj
    # (img_in), condition_proj + token_refiner (txt_in), time_embedder
    # (time_text_embed), final_layer (norm_out/proj_out), rope.inv_freq.
    "minimax_h3": {
        # start-anchored: the builder matches includes with re.match
        "include": re.compile(r".*\.weight$"),
        "group": re.compile(r"^(blocks\.\d+)\."),
        "exclude": re.compile(
            r"(adaln|modulation|scale_shift|norm"
            r"|patch_proj|condition_proj|token_refiner|time_embedder"
            r"|final_layer|rope)",
            re.IGNORECASE,
        ),
    },
}


def layer_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def natkey(s: str):
    """Natural order, so `--max-groups 2` picks blocks.0/blocks.1 rather than
    whatever two names happen to sort first lexicographically."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s))


def eligible(key: str, shape, include, exclude=None) -> bool:
    return (
        bool(include.match(key))
        and not (exclude and exclude.search(key))
        and len(shape) == 2
        and shape[0] >= 1024
        and shape[1] >= 1024
        and shape[1] % core.HBLOCK == 0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--src", required=True,
                    help="single .safetensors OR a sharded directory (index.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group", type=int, default=64, help="scale group along in (64 or 32)")
    ap.add_argument("--format", default="int4",
                    choices=("int4", "nvfp4", "int8", "convrot_w8"),
                    help="residual codec: int4 (dequant-first, any GPU), nvfp4 (native FP4 fwd, "
                         "sm_120), int8 (low-rank + per-row int8), convrot_w8 (rotated per-channel "
                         "int8, no low-rank branch)")
    ap.add_argument("--budget-x", type=float, default=0.0,
                    help="adaptive per-layer ranks: target TOTAL quantized bytes as a fraction "
                         "of bf16 (e.g. 0.32). 0 = fixed --rank everywhere. int4 format only.")
    ap.add_argument("--blocks-per-shard", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=0,
                    help="SMOKE ONLY: keep the first N block groups and DROP every other tensor "
                         "(including passthrough). Output is a truncated slab, not a loadable model.")
    a = ap.parse_args()
    preset = MODELS[a.model]
    os.makedirs(a.out_dir, exist_ok=True)
    if a.format == "convrot_w8":
        # convrot has no scale groups along `in`: one f32 scale per output row,
        # and the only grouping is the Hadamard block.
        a.group = core.HBLOCK
        a.rank = 0

    src_bytes, src_mtime = source_identity(a.src)
    state_path = os.path.join(a.out_dir, "build-state.json")
    state = load_state(state_path, a.src, src_bytes, src_mtime)
    if state is None:
        state = new_state(a.src, src_bytes, src_mtime)
        atomic_write_json(state_path, state)
        print("[squareq-build] fresh build")
    else:
        print(f"[squareq-build] resuming; {len(state['completed_shards'])} shards done")

    # ── pass 1: classify every tensor from metadata only ─────────────────────
    with open_source(a.src) as f:
        keys = sorted(f.keys())
        meta = {}
        for k in keys:
            sl = f.get_slice(k)
            meta[k] = (sl.get_shape(), sl.get_dtype())

    plan = new_plan(a.model, a.src, _FORMAT_TAG[a.format], core.HBLOCK, a.group)
    plan["rank_default"] = a.rank
    if a.format == "convrot_w8":
        plan["lora_rank"] = 0

    # ── adaptive rank planner (chunk 8): one truncated SVD per layer, evaluate
    # candidate ranks by ACTUAL quantized weight error, then greedy-allocate
    # rank upgrades by marginal error-reduction per byte under --budget-x. ──
    rank_by_layer = {}
    if a.budget_x > 0:
        if a.format != "int4":
            raise SystemExit("--budget-x supports --format int4 only")
        CAND = [16, 32, 64, 128]  # rank>=16: zero-width lora tensors would break loaders
        elig = [k for k in keys if eligible(k, meta[k][0], preset["include"], preset.get("exclude"))
                and preset["group"].match(k)]
        tables = {}
        total_bf16 = 0
        with open_source(a.src) as f:
            for n, k in enumerate(elig):
                wt = f.get_tensor(k).float()
                o, i_f = wt.shape
                total_bf16 += o * i_f * 2
                gen = torch.random.get_rng_state()
                torch.manual_seed(layer_seed(k))
                U, S, V = torch.svd_lowrank(wt, q=CAND[-1] + 16, niter=6)
                torch.random.set_rng_state(gen)
                rows = []
                for r in CAND:
                    if r == 0:
                        resid = wt.double()
                        lr = None
                    else:
                        lu = (U[:, :r] * S[:r]).double()
                        ld = V[:, :r].double()
                        resid = wt.double() - lu @ ld.t()
                        lr = (lu @ ld.t()).float()
                    rrot = core.rht_grouped(resid).float()
                    q, sc = core.quant_int4_g64(rrot, group=a.group)
                    back = core.rht_grouped(
                        core.dequant_int4_g64(q, sc, group=a.group).double()
                    ).float()
                    if lr is not None:
                        back = back + lr
                    err2 = float(((back - wt) ** 2).sum())
                    nbytes = o * i_f // 2 + (i_f // a.group) * o * 2 + r * (o + i_f) * 2
                    rows.append((r, nbytes, err2))
                tables[k] = rows
                del wt, U, S, V
                if (n + 1) % 20 == 0 or n + 1 == len(elig):
                    print(f"[squareq-plan] candidates {n + 1}/{len(elig)}")
        budget = a.budget_x * total_bf16
        cur = {k: 0 for k in elig}                    # index into CAND rows
        used = sum(tables[k][0][1] for k in elig)
        import heapq
        heap = []
        for k in elig:
            r0, b0, e0 = tables[k][0]
            r1, b1, e1 = tables[k][1]
            heapq.heappush(heap, (-(e0 - e1) / (b1 - b0), k))
        while heap:
            neg_gain, k = heapq.heappop(heap)
            ci = cur[k]
            if ci + 1 >= len(tables[k]):
                continue
            db = tables[k][ci + 1][1] - tables[k][ci][1]
            if used + db > budget:
                continue
            cur[k] = ci + 1
            used += db
            if ci + 2 < len(tables[k]):
                _, b1, e1 = tables[k][ci + 1]
                _, b2, e2 = tables[k][ci + 2]
                heapq.heappush(heap, (-(e1 - e2) / (b2 - b1), k))
        rank_by_layer = {k: tables[k][cur[k]][0] for k in elig}
        hist = {}
        for r in rank_by_layer.values():
            hist[r] = hist.get(r, 0) + 1
        print(f"[squareq-plan] adaptive ranks (budget {a.budget_x}x): {sorted(hist.items())}"
              f"  used={used / total_bf16:.4f}x")
        plan["rank_by_layer"] = rank_by_layer
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
        g = m.group(1) if (m and eligible(k, meta[k][0], preset["include"], preset.get("exclude"))) else ""
        groups.setdefault(g, []).append(k)

    excl = preset.get("exclude")
    if excl:
        policy_excluded = [
            k for k in keys
            if preset["include"].match(k) and excl.search(k)
            and len(meta[k][0]) == 2
        ]
        print(f"[squareq-build] sensitivity policy: {len(policy_excluded)} matching "
              f"2D tensors kept bf16 (mod/adaln/norm): {policy_excluded[:3]}")
        plan["policy_excluded"] = policy_excluded

    # Why every OTHER 2D .weight stayed bf16, so the policy is auditable rather
    # than inferred from what is missing. Counted by reason, listed by family.
    reasons: dict = {}
    for k in keys:
        shape = meta[k][0]
        if len(shape) != 2 or not k.endswith(".weight"):
            continue
        if eligible(k, shape, preset["include"], preset.get("exclude")) and preset["group"].match(k):
            continue
        if not preset["include"].match(k):
            why = "not_matched_by_include"
        elif excl and excl.search(k):
            why = "policy_exclude"
        elif shape[0] < 1024 or shape[1] < 1024:
            why = "too_small"
        elif shape[1] % core.HBLOCK != 0:
            why = f"in%{core.HBLOCK}!=0"
        else:
            why = "outside_block_groups"
        fam = re.sub(r"\d+", "N", k)
        reasons.setdefault(why, {}).setdefault(fam, 0)
        reasons[why][fam] += 1
    plan["exclusion_report"] = {
        w: {"tensors": sum(f.values()), "families": dict(sorted(f.items()))}
        for w, f in sorted(reasons.items())
    }
    for why, rep in plan["exclusion_report"].items():
        print(f"[squareq-build] excluded ({why}): {rep['tensors']} 2D weights -> "
              f"{list(rep['families'])[:4]}")

    block_names = sorted(g for g in groups if g)
    if a.max_groups > 0:
        keep = set(sorted(block_names, key=natkey)[: a.max_groups])
        dropped = sum(len(v) for g, v in groups.items() if g not in keep)
        groups = {g: v for g, v in groups.items() if g in keep}
        block_names = sorted(keep)
        plan["truncated_max_groups"] = a.max_groups
        print(f"[squareq-build] !! TRUNCATED BUILD: keeping {sorted(keep, key=natkey)}, "
              f"dropping {dropped} tensors. Output is NOT a loadable model.")

    shards: list = []  # (shard_filename_stub, [group names])
    for i in range(0, len(block_names), a.blocks_per_shard):
        shards.append(block_names[i : i + a.blocks_per_shard])
    if groups.get(""):
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
            fpath, metadata={"format": _FORMAT_TAG[a.format], "rank": a.rank, "group": a.group}
        )
        # declare (and always account plan/weight_map, even when skipping)
        decls: list = []  # (out_name, src_key, kind)
        for g in groups_in_shard:
            for k in groups[g]:
                shape, dt = meta[k]
                if g and eligible(k, shape, preset["include"], preset.get("exclude")):
                    base = k[: -len(".weight")]
                    o, i_f = shape
                    r = rank_by_layer.get(k, a.rank)
                    if a.format == "convrot_w8":
                        parts = [
                            (base + ".qweight", torch.int8, [o, i_f]),
                            (base + ".wscale", torch.float32, [o]),
                        ]
                    elif a.format == "int8":
                        parts = [
                            (base + ".q8weight", torch.int8, [o, i_f]),
                            (base + ".w8scale", torch.bfloat16, [o]),
                            (base + ".lora_down", torch.bfloat16, [i_f, r]),
                            (base + ".lora_up", torch.bfloat16, [o, r]),
                        ]
                    elif a.format == "nvfp4":
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
            with open_source(a.src) as f:
                for base, k, kind in decls:
                    if kind == "pass":
                        w.write_tensor(k, f.get_tensor(k))
                        if k not in plan["passthrough"]:
                            plan["passthrough"].append(k)
                        continue
                    wt = f.get_tensor(k).float()
                    if a.format == "convrot_w8":
                        tensors, stats = core.quantize_layer_convrot(wt)
                        suffixes = ("qweight", "wscale")
                    elif a.format == "int8":
                        tensors, stats = core.quantize_layer_w8(
                            wt, rank=rank_by_layer.get(k, a.rank), seed=layer_seed(k)
                        )
                        suffixes = ("q8weight", "w8scale", "lora_down", "lora_up")
                    elif a.format == "nvfp4":
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
                            wt, rank=rank_by_layer.get(k, a.rank), group=a.group,
                            seed=layer_seed(k)
                        )
                        suffixes = ("qweight", "wscales", "lora_down", "lora_up")
                    for suffix in suffixes:
                        w.write_tensor(base + "." + suffix, tensors[suffix])
                    plan["layers"][k] = {**stats, "format": _FORMAT_TAG[a.format]}
                    print(
                        f"[squareq-build] shard {si + 1}/{nsh}  {k}"
                        f"  [{stats['out']}x{stats['in']}]  cos_w={stats['cos_w']:.5f}"
                    )
                    del wt, tensors
            w.close()
        except BaseException:
            w.abort()
            raise
        # Partial plan BEFORE the state: a kill between the two re-does the
        # shard (idempotent), whereas the other order would mark it complete
        # with its layer stats missing from the plan forever.
        atomic_write_json(os.path.join(a.out_dir, "squareq-plan.partial.json"), plan)
        state["completed_shards"].append(fname)
        atomic_write_json(state_path, state)

    index = {
        "metadata": {"format": _FORMAT_TAG[a.format]},
        "weight_map": weight_map,
    }
    atomic_write_json(os.path.join(a.out_dir, "model.safetensors.index.json"), index)
    atomic_write_json(
        os.path.join(a.out_dir, "squareq-plan.json"), finalize_plan(plan)
    )
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    t = plan["totals"]
    if not t["quantized_layers"]:
        raise SystemExit(
            f"[squareq-build] NOTHING QUANTIZED for --model {a.model}: the include/exclude "
            f"policy matched no eligible 2D weight. See exclusion_report in squareq-plan.json."
        )
    print(
        f"[squareq-build] DONE: {t['quantized_layers']} quantized"
        f" (cos_w min {t['cos_w_min']:.5f} mean {t['cos_w_mean']:.5f}),"
        f" {t['passthrough_tensors']} passthrough,"
        f" compression {t['compression']:.3f}x of bf16"
    )


if __name__ == "__main__":
    main()
