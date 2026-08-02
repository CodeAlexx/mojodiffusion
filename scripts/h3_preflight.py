#!/usr/bin/env python3
"""First thing to run the moment MiniMax-H3 weights land. HEADER ONLY.

Reads each shard's safetensors header — the 8-byte length prefix plus the JSON
directory — and never touches tensor data. On a 61.7 GiB checkpoint that is a
few hundred KB of reads and takes about a second, so it can run WHILE the rest
is still downloading.

WHAT IT ANSWERS, in the order that matters:

  1. Does the released checkpoint have the keys our loader targets? The whole
     port is built against `transformer_key_plan.txt`, which came from running
     the diffusers converter's `--dry_run` with NO WEIGHTS. That plan is a
     third party's reading of a model nobody outside MiniMax had seen. This is
     the first moment it can be checked against reality.

  2. Are the shapes what we assumed? In particular the two fused tensors our
     loader rewrites: `attn.qkv_proj.weight` must be [3*H*D, in] with per-head
     interleaving, and `mlp.fc1.weight` must be [2*ff, in] storing [gate;value].
     A different fusion order is silent — it produces a model that runs and
     generates noise.

  3. Is the dtype rule intact? Exactly 12 fp32 tensors under five prefixes,
     everything else bf16 INCLUDING adaLN. The trap is that
     `final_layer.adaln_proj` sits beside the fp32 output heads and is bf16.

  4. What does it actually cost on disk, and does unit 14's fp8 arithmetic
     still hold against the real shapes?

EXIT CODE is 0 only if the checkpoint matches the plan exactly. Anything else
means STOP AND READ, not "probably fine" — a mismatch here invalidates gated
units 12 and 14 and has to be understood before any weight is loaded.

Usage:
    python3 scripts/h3_preflight.py /path/to/MiniMax-H3/transformer
    python3 scripts/h3_preflight.py --self-test
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from collections import defaultdict

PLAN = "/home/alex/minimax_h3_ref/transformer_key_plan.txt"

F32_PREFIXES_ORIGINAL = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# `rope.inv_freq` is recomputed by the port and dropped by the converter. Its
# presence in the checkpoint is expected and is NOT a mismatch.
EXPECTED_EXTRA = {"rope.inv_freq"}

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def read_header(path):
    """The safetensors directory: 8-byte little-endian length, then JSON."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: shorter than the 8-byte length prefix")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > 512 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header length {length}")
        blob = handle.read(length)
        if len(blob) != length:
            raise ValueError(f"{path}: header truncated ({len(blob)} of {length})")
    return json.loads(blob)


def parse_plan():
    """Expected ORIGINAL-namespace keys and shapes.

    The plan lists shapes against DIFFUSERS keys, so the original shape has to
    be reconstructed: a fused `qkv_proj` maps to three same-shaped diffusers
    tensors and is stored as their concatenation along dim 0. Everything else
    is one to one."""
    groups = []           # (original_key, [(diffusers_key, shape, dtype), ...])
    current = None
    for line in open(PLAN):
        m = re.search(r"^(\S*)\s+->\?\s+(\S+)\s+\[([\d, ]+)\]\s+(F32|BF16)", line)
        if not m:
            continue
        original, diffusers = m.group(1), m.group(2)
        shape = [int(x) for x in m.group(3).split(",")]
        dtype = m.group(4)
        if original:
            current = (original, [])
            groups.append(current)
        if current is None:
            raise ValueError(f"continuation line with no original key: {line!r}")
        current[1].append((diffusers, shape, dtype))

    expected = {}
    for original, targets in groups:
        shapes = [t[1] for t in targets]
        dtypes = {t[2] for t in targets}
        if len(dtypes) != 1:
            raise ValueError(f"{original}: mixed dtypes {dtypes}")
        if len(targets) == 1:
            shape = shapes[0]
        else:
            first = shapes[0]
            if any(s != first for s in shapes):
                raise ValueError(f"{original}: fused targets differ in shape {shapes}")
            shape = [first[0] * len(shapes)] + first[1:]
        expected[original] = (shape, dtypes.pop(), len(targets))
    return expected


def is_f32_original(key):
    return key.startswith(F32_PREFIXES_ORIGINAL)


def inventory(directory):
    shards = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".safetensors")
    )
    if not shards:
        raise SystemExit(f"no .safetensors under {directory}")
    found = {}
    per_shard = []
    for path in shards:
        header = read_header(path)
        n = 0
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            found[key] = (list(entry["shape"]), entry["dtype"])
            n += 1
        per_shard.append((os.path.basename(path), n))
    return found, per_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--self-test", action="store_true",
                    help="build a header-only checkpoint from the plan and check "
                         "this script against it")
    args = ap.parse_args()

    expected = parse_plan()
    print(f"plan: {len(expected)} original keys expected "
          f"({sum(v[2] for v in expected.values())} diffusers keys after splitting)")

    if args.self_test:
        directory = self_test_fixture(expected)
    elif args.directory:
        directory = args.directory
    else:
        ap.error("give a directory or --self-test")

    found, per_shard = inventory(directory)
    print(f"checkpoint: {len(found)} tensors across {len(per_shard)} shard(s)")
    for name, n in per_shard[:8]:
        print(f"    {name}  {n} tensors")
    if len(per_shard) > 8:
        print(f"    ... and {len(per_shard) - 8} more")

    problems = 0

    # ── 1. key set ───────────────────────────────────────────────────────────
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected) - EXPECTED_EXTRA)
    print()
    print("[1] key set")
    if missing:
        problems += len(missing)
        print(f"    MISSING {len(missing)} keys the loader needs:")
        for key in missing[:20]:
            print(f"      {key}")
        if len(missing) > 20:
            print(f"      ... and {len(missing) - 20} more")
    else:
        print(f"    ok   all {len(expected)} planned keys present")
    if extra:
        # Not fatal on its own, but it means the released model has surface the
        # diffusers PR did not describe — which is exactly the kind of thing
        # that needs reading, not skipping.
        problems += len(extra)
        print(f"    UNEXPECTED {len(extra)} keys not in the plan:")
        for key in extra[:20]:
            print(f"      {key}  {found[key][0]}  {found[key][1]}")
        if len(extra) > 20:
            print(f"      ... and {len(extra) - 20} more")
    else:
        print("    ok   no unplanned keys")
    for key in sorted(EXPECTED_EXTRA & set(found)):
        print(f"    note {key} present as expected (recomputed by the port)")

    # ── 2. shapes ────────────────────────────────────────────────────────────
    print()
    print("[2] shapes")
    shape_bad = []
    for key, (want_shape, _, _) in expected.items():
        if key not in found:
            continue
        got_shape = found[key][0]
        if got_shape != want_shape:
            shape_bad.append((key, got_shape, want_shape))
    if shape_bad:
        problems += len(shape_bad)
        print(f"    MISMATCH {len(shape_bad)}:")
        for key, got, want in shape_bad[:20]:
            print(f"      {key}: got {got}, plan says {want}")
    else:
        print(f"    ok   every present key matches the planned shape")

    # the two fused tensors the loader rewrites — called out by name because a
    # wrong fusion order is silent
    print()
    print("[3] the fused tensors the loader rewrites")
    for probe, label in (
        ("blocks.0.attn.qkv_proj.weight", "qkv (per-head interleaved -> q|k|v)"),
        ("blocks.0.mlp.fc1.weight", "fc1 ([gate;value] -> [value;gate])"),
    ):
        if probe in found:
            got = found[probe][0]
            want = expected.get(probe, (None,))[0]
            status = "ok  " if got == want else "FAIL"
            if got != want:
                problems += 1
            print(f"    {status} {probe} {got}  ({label})")
        else:
            problems += 1
            print(f"    FAIL {probe} absent — the loader has nothing to rewrite")

    # ── 4. dtypes ────────────────────────────────────────────────────────────
    print()
    print("[4] dtype rule: exactly 12 fp32 under five prefixes, rest bf16")
    hist = defaultdict(int)
    dtype_bad = []
    f32_keys = []
    for key, (_, dtype) in found.items():
        hist[dtype] += 1
        if key in EXPECTED_EXTRA:
            continue
        want_f32 = is_f32_original(key)
        got_f32 = dtype in ("F32", "F64")
        if want_f32 != got_f32:
            dtype_bad.append((key, dtype, "F32" if want_f32 else "BF16"))
        if got_f32:
            f32_keys.append(key)
    print(f"    dtypes present: {dict(sorted(hist.items()))}")
    if dtype_bad:
        problems += len(dtype_bad)
        print(f"    MISMATCH {len(dtype_bad)}:")
        for key, got, want in dtype_bad[:20]:
            print(f"      {key}: {got}, expected {want}")
    else:
        print("    ok   every key's dtype follows the five-prefix rule")
    if len(f32_keys) == 12:
        print("    ok   exactly 12 fp32 tensors")
    else:
        problems += 1
        print(f"    FAIL {len(f32_keys)} fp32 tensors, expected 12:")
        for key in sorted(f32_keys):
            print(f"      {key}")
    adaln = [k for k in found if "adaln" in k]
    adaln_f32 = [k for k in adaln if found[k][1] in ("F32", "F64")]
    if adaln and not adaln_f32:
        print(f"    ok   all {len(adaln)} adaLN tensors are bf16 (the trap: they "
              "sit beside the fp32 heads)")
    elif adaln_f32:
        problems += 1
        print(f"    FAIL {len(adaln_f32)} adaLN tensors are fp32")

    # ── 5. footprint, against unit 14 ────────────────────────────────────────
    print()
    print("[5] footprint")
    params = sum(math.prod(s) for s, _ in found.values())
    disk = sum(math.prod(s) * DTYPE_BYTES.get(d, 2) for s, d in found.values())
    gib = 1024**3
    print(f"    {params/1e9:.4f} B parameters, {disk/gib:.2f} GiB on disk")
    ada_params = sum(math.prod(found[k][0]) for k in adaln)
    if ada_params:
        print(f"    adaLN is {ada_params/1e9:.4f} B ({100*ada_params/params:.1f}% "
              "of the model) — unit 14 evicts this to a per-step cache")
    free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
    print(f"    free on /: {free/gib:.1f} GiB")

    print()
    if problems == 0:
        print("PREFLIGHT PASS — the checkpoint matches the plan the port was "
              "built against.")
        print("Next: fp8 conversion per unit 14 (fp8_policy.mojo), then load.")
        return 0
    print(f"PREFLIGHT FAIL — {problems} problem(s). STOP AND READ.")
    print("A mismatch here invalidates gated units 12 (loader) and 14 (fp8")
    print("policy); both were built against the plan, not against weights.")
    return 1


def self_test_fixture(expected):
    """A real safetensors header describing the real checkpoint, with NO data.

    The point is to exercise this script end to end before the weights exist:
    the header is byte-accurate and the offsets describe 61.7 GiB that is never
    written, so the file is a few hundred KB. read_header only ever reads the
    prefix and the JSON, so it neither notices nor cares.
    """
    directory = "/tmp/claude-1000/-home-alex-mojodiffusion/h3_preflight_selftest"
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "model-00001-of-00001.safetensors")

    entries = {}
    offset = 0
    for key, (shape, dtype, _) in sorted(expected.items()):
        size = math.prod(shape) * DTYPE_BYTES[dtype]
        entries[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    # rope.inv_freq is in the checkpoint but dropped by the converter, so the
    # fixture includes it: the script must report it as expected, not extra.
    entries["rope.inv_freq"] = {
        "dtype": "F32", "shape": [64], "data_offsets": [offset, offset + 256]
    }

    blob = json.dumps(entries).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
    print(f"self-test: wrote a header-only fixture describing "
          f"{offset/1024**3:.2f} GiB in {os.path.getsize(path)/1024:.0f} KB")
    return directory


if __name__ == "__main__":
    sys.exit(main())
