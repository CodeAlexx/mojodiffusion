#!/usr/bin/env python3
"""Full-model weight-only INT4 quantizer (SVDQuant W4A16, clean row-major layout).

Walks a DiT checkpoint, SVD-quantizes every CLASS-A linear (2D `.weight`,
in%group==0, not a norm/embedder/gate) to rank-R low-rank + group-G int4
residual (scripts/svdquant_selfquant.quantize_svdquant_w4a16 math), and writes
ONE slab: per quantized layer the tensors `<key>.{qweight,wscales,lora_down,
lora_up,smooth}` (the layer's `<key>.bias`, if any, passes through verbatim and
is the layer bias — the "6th" tensor). Every non-class-A tensor passes through
VERBATIM (dtype preserved) so the slab is a COMPLETE loadable model. Emits a JSON
manifest naming the quantized layers + geometry. CPU-only (no GPU).

Weight-only: our Mojo inference is dequant-first (weights→bf16, GEMM in bf16;
activations STAY bf16), so smooth is ONES (no activation smoothing). If activation
smoothing is added later, match nunchaku's DIVIDE convention (main=(x/smooth)@W^T)
per the Phase-2 finding (MJ-1095), NOT multiply.

Reconstruct (Mojo ops/svdquant.mojo, twos_complement/lo_even, smooth=1):
    W_rec[o,k] = int4(nibble)*wscales[k//G,o] + (lora_up @ lora_down^T)[o,k]

Usage:
  # fast report only (classify + projected size + sample fidelity, NO write):
  .venv/bin/python svdquant_quantize_model.py <src.safetensors>
  # full build:
  .venv/bin/python svdquant_quantize_model.py <src.safetensors> --build --out <slab.safetensors> \
      [--key-prefix ...] [--rank 32] [--group 64] [--sample 10] [--full-svd]
"""
import argparse, json, os, re, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, "/home/alex/mojodiffusion/scripts")
from svdquant_selfquant import quantize_svdquant_w4a16, NBITS_MAX

EXCLUDE_SUBSTR = ("norm", "embed", "gate_logits", "scale_shift", "adaln",
                  "patchify", "proj_out", "time_", "caption", "connector")


def name_is_class_a(key, shape):
    """Class-A by NAME/shape, EXCLUDING the in%group test (checked separately so
    a divisibility miss can fail loud instead of silently passing through)."""
    if not key.endswith(".weight") or len(shape) != 2:
        return False
    out, inh = shape
    if inh < 256 or out < 256:
        return False
    if any(s in key for s in EXCLUDE_SUBSTR):
        return False
    return True


def quantize_layer(W, rank, group, full_svd):
    """Returns (qbyte, wscales, lora_down, lora_up, smooth). full_svd delegates to
    the Mojo-gate-trusted svdquant_selfquant.quantize_svdquant_w4a16 (byte-for-
    byte); the default uses a truncated SVD (equal rank-R subspace, ~30x faster
    on wide layers) with the IDENTICAL residual pack."""
    if full_svd:
        return quantize_svdquant_w4a16(W, rank, group)
    out, inh = W.shape
    assert inh % group == 0
    Wf = W.float()
    q = min(rank + 8, min(out, inh))
    Ul, Sl, Vl = torch.svd_lowrank(Wf, q=q, niter=4)
    L1 = Ul[:, :rank] * Sl[:rank]              # [out, R]
    L2 = Vl[:, :rank].t()                       # [R, in]
    residual = Wf - L1 @ L2
    ngrp = inh // group
    r = residual.view(out, ngrp, group)
    scale = r.abs().amax(dim=2) / NBITS_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(r / scale[:, :, None]), -8, 7).to(torch.int16).view(out, inh)
    nib = (q & 0xF).to(torch.int16)
    lo = nib[:, 0::2]; hi = nib[:, 1::2]
    qbyte = (lo | (hi << 4)).to(torch.uint8).view(torch.int8)
    wscales = scale.t().contiguous().to(torch.bfloat16)
    lora_down = L2.t().contiguous().to(torch.bfloat16)
    lora_up = L1.contiguous().to(torch.bfloat16)
    smooth = torch.ones(inh, dtype=torch.bfloat16)
    return qbyte, wscales, lora_down, lora_up, smooth


def recon_cos(W, qb, ws, ld, lu, group):
    out, inh = W.shape
    v = torch.zeros(out, inh, dtype=torch.int16)
    nb = (qb.view(torch.uint8).to(torch.int16) & 0xFF)
    lo = nb & 0xF; hi = (nb >> 4) & 0xF
    v[:, 0::2] = torch.where(lo >= 8, lo - 16, lo)
    v[:, 1::2] = torch.where(hi >= 8, hi - 16, hi)
    sc = ws.float().t()[:, torch.arange(inh) // group]
    Wr = v.float() * sc + lu.float() @ ld.float().t()
    # float64 reductions: float32 cosine over ~67M-element wide-ff vectors
    # accumulates enough norm error to return cos > 1.0 (a metric artifact, not
    # a real match) — double is stable and correct.
    a = Wr.flatten().double(); b = W.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _dtbytes(dt):
    dt = dt.upper()
    if dt in ("F64", "I64", "U64"): return 8
    if dt in ("F32", "I32", "U32"): return 4
    if dt in ("F16", "BF16", "I16", "U16"): return 2
    return 1


def quant_bytes(out, inh, rank, group):
    return out*(inh//2) + (inh//group)*out*2 + inh*rank*2 + out*rank*2 + inh*2


def _spread_sample(keys, n):
    if len(keys) <= n:
        return list(keys)
    buckets = {}
    for k in keys:
        buckets.setdefault(re.sub(r"\.\d+\.", ".N.", k), []).append(k)
    sigs = sorted(buckets); out = []; i = 0
    while len(out) < n and any(buckets[s] for s in sigs):
        s = sigs[i % len(sigs)]
        if buckets[s]:
            out.append(buckets[s].pop(len(buckets[s]) // 2))
        i += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--key-prefix", default="")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--full-svd", action="store_true")
    ap.add_argument("--build", action="store_true",
                    help="quantize + write the slab (long); default is report-only")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    f = safe_open(a.src, "pt")
    keys = list(f.keys())

    # ── classify ─────────────────────────────────────────────────────────────
    quant_keys, pass_keys, bad_in = [], [], []
    q_bytes = p_bytes = src_bytes = 0
    for k in keys:
        sl = f.get_slice(k); shape = list(sl.get_shape()); dt = sl.get_dtype()
        n = 1
        for s in shape: n *= s
        src_bytes += n * _dtbytes(dt)
        in_scope = k.startswith(a.key_prefix) if a.key_prefix else True
        if in_scope and name_is_class_a(k, shape):
            out, inh = shape
            if inh % a.group != 0:
                bad_in.append((k, shape)); pass_keys.append(k); p_bytes += n * _dtbytes(dt)
                continue
            quant_keys.append(k); q_bytes += quant_bytes(out, inh, a.rank, a.group)
        else:
            pass_keys.append(k); p_bytes += n * _dtbytes(dt)

    slab = q_bytes + p_bytes
    print(f"=== {os.path.basename(a.src)} ===")
    print(f"tensors: {len(keys)}  |  class-A quantized: {len(quant_keys)}  |  passthrough: {len(pass_keys)}")
    print(f"source:          {src_bytes/1e9:8.2f} GB")
    print(f"projected slab:  {slab/1e9:8.2f} GB   ({slab/src_bytes:.3f}x)   "
          f"[quant {q_bytes/1e9:.2f} + passthrough {p_bytes/1e9:.2f}]")
    if bad_in:
        print(f"\n*** FAIL-LOUD: {len(bad_in)} name-class-A layer(s) with in %% {a.group} != 0 "
              f"(kept bf16 passthrough — handle separately):")
        for k, sh in bad_in: print(f"      {k}  {sh}")
    else:
        print(f"  (all quantized layers have in %% {a.group} == 0)")

    # ── sample fidelity ──────────────────────────────────────────────────────
    sample = _spread_sample(quant_keys, a.sample)
    print(f"\n=== sample W_recon cos ({len(sample)} layers, "
          f"{'FULL' if a.full_svd else 'lowrank'} SVD, rank {a.rank} group {a.group}) ===")
    coss = []
    for k in sample:
        W = f.get_tensor(k); out, inh = W.shape; t0 = time.time()
        qb, ws, ld, lu, sm = quantize_layer(W, a.rank, a.group, a.full_svd)
        c = recon_cos(W, qb, ws, ld, lu, a.group); coss.append(c)
        print(f"  {k:72s} [{out},{inh}] cos={c:.5f} ({time.time()-t0:.2f}s)")
    if coss:
        print(f"\n  mean cos = {sum(coss)/len(coss):.5f}   min cos = {min(coss):.5f}")

    if not a.build:
        print("\n(report only — pass --build --out <path> to write the slab)")
        return

    # ── build ────────────────────────────────────────────────────────────────
    assert a.out, "--build requires --out"
    print(f"\n=== building slab → {a.out} ===")
    manifest = {"format": "svdquant_w4a16_clean", "rank": a.rank, "group": a.group,
                "sign": "twos_complement", "nibble": "lo_even", "smooth": "ones",
                "svd": "full" if a.full_svd else "lowrank", "src": a.src, "layers": {}}
    out_tensors = {}
    t0 = time.time(); done = 0
    for k in quant_keys:
        base = k[:-len(".weight")]
        W = f.get_tensor(k); out, inh = W.shape
        qb, ws, ld, lu, sm = quantize_layer(W, a.rank, a.group, a.full_svd)
        out_tensors[base + ".qweight"] = qb.contiguous()
        out_tensors[base + ".wscales"] = ws.contiguous()
        out_tensors[base + ".lora_down"] = ld.contiguous()
        out_tensors[base + ".lora_up"] = lu.contiguous()
        out_tensors[base + ".smooth"] = sm.contiguous()
        bkey = base + ".bias"
        manifest["layers"][base] = {
            "in": inh, "out": out, "rank": a.rank, "group": a.group,
            "tensors": ["qweight", "wscales", "lora_down", "lora_up", "smooth"],
            "bias": bkey if bkey in keys else None,
            "shapes": {"qweight": list(qb.shape), "wscales": list(ws.shape),
                       "lora_down": list(ld.shape), "lora_up": list(lu.shape),
                       "smooth": list(sm.shape)},
            "dtypes": {"qweight": "I8", "wscales": "BF16", "lora_down": "BF16",
                       "lora_up": "BF16", "smooth": "BF16"},
        }
        done += 1
        if done % 100 == 0:
            print(f"  quantized {done}/{len(quant_keys)} ({time.time()-t0:.0f}s)", flush=True)
    for k in pass_keys:                       # VERBATIM (dtype preserved)
        out_tensors[k] = f.get_tensor(k).contiguous()
    print(f"  quantized {done}, passthrough {len(pass_keys)}; writing {len(out_tensors)} tensors...")
    save_file(out_tensors, a.out, metadata={"svdquant_manifest": "see sidecar .manifest.json"})
    man = os.path.splitext(a.out)[0] + ".manifest.json"
    with open(man, "w") as mf: json.dump(manifest, mf, indent=1)
    sz = os.path.getsize(a.out)
    print(f"  wrote {a.out}  ({sz/1e9:.2f} GB = {sz/src_bytes:.3f}x of source)")
    print(f"  wrote {man}  ({len(manifest['layers'])} quantized layers)")
    print(f"  build time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
