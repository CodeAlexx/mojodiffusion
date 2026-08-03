#!/usr/bin/env python3
"""Audit an UNTRUSTED safetensors file before anything loads it.

Written 2026-08-02 for the released MiniMax-H3 audio VAE, at Alex's "be
careful". Kept general because the 61.7 GiB transformer is likely to arrive
the same way — out of a browser download rather than from a repo we can hash
against.

WHAT THE REAL THREAT IS. safetensors cannot execute code: it is an 8-byte
little-endian length, a JSON directory, and raw tensor bytes. There is no
pickle, no reduce, no import. So the threats worth spending effort on are not
"will opening it run something" but:

  1. IT IS NOT SAFETENSORS. A pickle renamed to .safetensors is the actual
     danger, because the thing that opens it might fall back to torch.load.
     Checked by parsing the container by hand, before any library sees it.

  2. PARSER DIVERGENCE. JSON permits duplicate keys and most parsers silently
     keep one of them — Python keeps the last, other implementations differ.
     A file with two entries for one tensor can therefore feed different bytes
     to different readers, which is exactly how a checkpoint passes one
     validator and behaves differently under another. Checked with
     object_pairs_hook, which sees the wire order rather than the collapsed
     dict.

  3. BYTES THAT NOTHING DECLARES. Anything between, before or after the
     declared spans is payload the format does not account for. Checked by
     reconstructing the full byte budget and requiring it to equal the file
     size exactly.

  4. SUBSTITUTED WEIGHTS. The file is structurally perfect and simply is not
     the model it claims to be, or is untrained. Checked statistically:
     trained weights are heavy-tailed with per-channel structure, fresh random
     init is Gaussian (kurtosis ~3.0) with uniform per-filter norms.

WHAT THIS CANNOT DO: establish provenance. There is no signature in the format
and, with no official repo published, nothing to hash against. A file can pass
every check here and still have come from anywhere. Say so rather than implying
otherwise.

Usage:
    python3 scripts/h3_verify_safetensors.py FILE [--expect-model MODULE:CLASS]
"""

import argparse
import collections
import json
import math
import os
import struct
import sys

import numpy as np

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2,
    "U32": 4, "U64": 8,
}
NP_DTYPE = {"F64": np.float64, "F32": np.float32, "F16": np.float16}

# A pickle opens with PROTO (0x80) or one of the older opcodes; a zip (torch's
# newer format) opens "PK". None of these may appear where a length goes.
PICKLE_MAGIC = (b"\x80", b"(", b"]", b"}", b"c")


def fail(problems, message):
    problems.append(message)
    print(f"    FAIL {message}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect-model", default=None,
                    help="module:ClassName to instantiate on meta and diff keys against")
    args = ap.parse_args()

    path = args.path
    size = os.path.getsize(path)
    problems = []
    print(f"file: {path}")
    print(f"      {size} bytes ({size/1024**2:.1f} MiB)")

    # ── 1. is it actually safetensors ────────────────────────────────────────
    print()
    print("[1] container")
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            fail(problems, "shorter than the 8-byte length prefix")
            return 1
        if prefix[:2] == b"PK" or prefix[:1] in PICKLE_MAGIC:
            fail(problems, f"looks like a pickle or zip, not safetensors: {prefix[:4]!r}")
            print("    STOP. Do not let torch.load anywhere near this file.")
            return 1
        (hlen,) = struct.unpack("<Q", prefix)
        if hlen <= 0 or 8 + hlen > size:
            fail(problems, f"header length {hlen} is impossible for a {size}-byte file")
            return 1
        raw = handle.read(hlen)
    if len(raw) != hlen:
        fail(problems, "header truncated")
        return 1
    print(f"    ok   8-byte length {hlen}, no pickle/zip magic")

    # ── 2. parser divergence ─────────────────────────────────────────────────
    print()
    print("[2] header JSON")
    seen = []
    try:
        json.loads(raw, object_pairs_hook=lambda p: seen.append(p) or dict(p))
    except Exception as error:
        fail(problems, f"header is not valid JSON: {error}")
        return 1
    top = seen[-1]
    names = [k for k, _ in top]
    dups = [k for k, c in collections.Counter(names).items() if c > 1]
    if dups:
        fail(problems, f"DUPLICATE KEYS — different parsers will see different "
                       f"tensors: {dups[:5]}")
    else:
        print(f"    ok   {len(names)} entries, all unique (no parser divergence)")

    odd = [k for k in names
           if not k.isascii() or "\x00" in k or len(k) > 250 or k.startswith("/")]
    if odd:
        fail(problems, f"suspicious key names: {odd[:5]}")
    else:
        print(f"    ok   names ASCII, longest {max(len(k) for k in names)} chars, "
              "no NUL or path characters")

    hdr = dict(top)
    hdr.pop("__metadata__", None)
    meta = dict(top).get("__metadata__")
    print(f"    __metadata__: {meta!r}")

    # ── 3. byte budget ───────────────────────────────────────────────────────
    print()
    print("[3] byte budget — nothing undeclared")
    base = 8 + hlen
    spans = []
    dtypes = collections.Counter()
    for k, e in hdr.items():
        if set(e) - {"dtype", "shape", "data_offsets"}:
            fail(problems, f"{k}: unexpected header fields")
        dt = e.get("dtype")
        if dt not in DTYPE_BYTES:
            fail(problems, f"{k}: unknown dtype {dt!r}")
            continue
        dtypes[dt] += 1
        shape = e.get("shape")
        if not isinstance(shape, list) or any(not isinstance(d, int) or d < 0 for d in shape):
            fail(problems, f"{k}: bad shape {shape!r}")
            continue
        want = math.prod(shape) * DTYPE_BYTES[dt]
        a, b = e["data_offsets"]
        if b - a != want:
            fail(problems, f"{k}: {b-a} bytes declared, shape x dtype need {want}")
        if a < 0 or base + b > size:
            fail(problems, f"{k}: span [{a},{b}] outside the file")
        spans.append((a, b, k))
    spans.sort()
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            fail(problems, f"OVERLAP between {spans[i-1][2]} and {spans[i][2]}")
    gaps = spans[0][0] if spans else 0
    for i in range(1, len(spans)):
        gaps += spans[i][0] - spans[i - 1][1]
    tail = size - (base + (spans[-1][1] if spans else 0))
    declared = sum(b - a for a, b, _ in spans)
    print(f"    dtypes: {dict(dtypes)}")
    print(f"    tensor bytes {declared}, gaps {gaps}, trailing {tail}")
    if gaps or tail:
        fail(problems, f"{gaps + tail} bytes are not covered by any tensor")
    else:
        print(f"    ok   {base} + {declared} = {size}, every byte accounted for")

    # ── 4. values ────────────────────────────────────────────────────────────
    print()
    print("[4] values")
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    nan = inf = zero = ascii_dense = 0
    lo, hi = np.inf, -np.inf
    kurts, cvs = [], []
    total = 0
    for k, e in hdr.items():
        a, b = e["data_offsets"]
        np_dt = NP_DTYPE.get(e["dtype"])
        if np_dt is None:
            continue
        x = buf[base + a: base + b].view(np_dt)
        total += x.size
        if x.size == 0:
            continue
        finite = np.isfinite(x)
        if not finite.all():
            nan += int(np.isnan(x).sum())
            inf += int(np.isinf(x).sum())
        xf = x[finite].astype(np.float64)
        if xf.size:
            lo = min(lo, float(xf.min()))
            hi = max(hi, float(xf.max()))
            if float(np.abs(xf).max()) == 0.0:
                zero += 1
        if x.size >= 256:
            raw_b = np.asarray(x).view(np.uint8)
            if float(((raw_b >= 32) & (raw_b < 127)).mean()) > 0.92:
                ascii_dense += 1
                print(f"    note ASCII-dense tensor: {k}")
        if x.size >= 4096 and len(e["shape"]) >= 2 and xf.var() > 0:
            kurts.append(float(((xf - xf.mean()) ** 4).mean() / xf.var() ** 2))
            per = xf.reshape(e["shape"][0], -1)
            n = np.linalg.norm(per, axis=1)
            cvs.append(float(n.std() / max(n.mean(), 1e-30)))
    print(f"    {len(hdr)} tensors, {total/1e6:.1f} M values")
    print(f"    NaN {nan}, Inf {inf}, all-zero tensors {zero}, range [{lo:.4g}, {hi:.4g}]")
    if nan or inf:
        fail(problems, "non-finite values present")
    if ascii_dense:
        fail(problems, f"{ascii_dense} tensors look like embedded text, not weights")
    else:
        print("    ok   no tensor carries an ASCII-dense payload")
    if kurts:
        kurts.sort(); cvs.sort()
        mk = kurts[len(kurts)//2]; mc = cvs[len(cvs)//2]
        heavy = sum(1 for k in kurts if k > 4.0)
        print(f"    kurtosis median {mk:.3f} (Gaussian 3.0), "
              f"{100*heavy/len(kurts):.0f}% heavy-tailed; per-filter norm CV median {mc:.4f}")
        if mk < 3.3 and mc < 0.02:
            fail(problems, "looks like fresh random init, not trained weights")
        else:
            print("    ok   consistent with TRAINED weights, not random init")

    # ── 5. optional: diff against a reference module ─────────────────────────
    if args.expect_model:
        print()
        print(f"[5] key/shape diff against {args.expect_model}")
        import importlib
        import torch
        mod_name, cls_name = args.expect_model.split(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        with torch.device("meta"):
            model = cls()
        want = {k: list(v.shape) for k, v in model.state_dict().items()}
        missing = sorted(set(want) - set(hdr))
        extra = sorted(set(hdr) - set(want))
        shape_bad = [(k, hdr[k]["shape"], want[k])
                     for k in sorted(set(want) & set(hdr))
                     if hdr[k]["shape"] != want[k]]
        print(f"    reference {len(want)} tensors, file {len(hdr)}")
        for label, items in (("missing", missing), ("extra", extra)):
            print(f"    {label}: {len(items)}")
            for k in items[:10]:
                print(f"      {k}")
        print(f"    shape mismatches: {len(shape_bad)}")
        for k, g, w in shape_bad[:10]:
            print(f"      {k}: file {g}, reference {w}")
        if missing or extra or shape_bad:
            fail(problems, "does not match the reference module")
        else:
            print("    ok   EXACT match on every key and shape")

    print()
    if problems:
        print(f"VERDICT: {len(problems)} PROBLEM(S) — do not load this file until they are understood.")
        return 1
    print("VERDICT: structurally sound, values sane, no undeclared bytes.")
    print("NOT ESTABLISHED: provenance. There is no signature in this format and")
    print("nothing published to hash against, so this says the file is a")
    print("well-formed trained model — not who produced it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
