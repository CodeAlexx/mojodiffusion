"""Legacy SquareQ API (drop-in for the old /home/alex/SquareQ/src package).

Serves the four pre-existing scripts (svdquant_w4a4_gate.py, _make_fixture.py,
_build_slab.py, svdquant_selfquant.py) with the exact tuple contracts they
expect, implemented on the v3 core. The slab builder is REWRITTEN as a
streaming two-pass writer (bounded RAM) — the old dict-accumulate builder is
what earned the SIGKILL-9 scar.

Contracts (verbatim from the callers):
    hadamard(n) -> normalized Sylvester H [n,n] f64 (orthonormal, symmetric)
    quantize_svdquant_w4a4(W, rank=128)
        -> (qweight u8 [out,in/2], wscale bf16 [out], lora_down bf16 [in,R],
            lora_up bf16 [out,R])           # per-OUT scale, full-row rotation
    w4a4_forward(X, qb, ws, ld, lu) -> f64 [M,out]
    _unpack_int4_perout(qb) -> int16 [out,in]
    quantize_svdquant_w4a16(W, rank, group, full_svd=False)
        -> (qbyte u8 [out,in/2], wscales bf16 [in/group,out],
            lora_down bf16 [in,R], lora_up bf16 [out,R], smooth bf16 [in]=1)
    build_svdquant_w4a4_slab(src, out) -> metrics dict
"""

import os

import torch
from safetensors import safe_open

from .core import hadamard_matrix, pack_int4, unpack_int4
from .stwriter import StreamingSafetensorsWriter

NBITS_MAX = 7  # symmetric int4: absmax maps to +/-7 (range [-8, 7])


def hadamard(n: int) -> torch.Tensor:
    return hadamard_matrix(n, dtype=torch.float64, normalized=True)


def _unpack_int4_perout(qb: torch.Tensor) -> torch.Tensor:
    return unpack_int4(qb)


def _svd_factors(w: torch.Tensor, rank: int, full_svd: bool):
    wd = w.float()
    if full_svd:
        u, s, vh = torch.linalg.svd(wd.double(), full_matrices=False)
        l_up = (u[:, :rank] * s[:rank]).float()
        l_down = vh[:rank, :].t().float()
    else:
        q = min(rank + 16, min(wd.shape))
        u, s, v = torch.svd_lowrank(wd, q=q, niter=6)
        l_up = (u[:, :rank] * s[:rank]).contiguous()
        l_down = v[:, :rank].contiguous()
    return l_down, l_up  # [in,R], [out,R]


def quantize_svdquant_w4a4(w: torch.Tensor, rank: int = 128):
    """QuaRot-style W4A4: full-row normalized Hadamard rotation of the rank-R
    residual, per-OUT int4 scale. Requires in to be a power of two."""
    wf = w.float()
    out_f, in_f = wf.shape
    if in_f & (in_f - 1) != 0:
        raise ValueError(f"w4a4: in={in_f} not a power of two")
    l_down, l_up = _svd_factors(wf, rank, full_svd=False)
    resid = wf.double() - l_up.double() @ l_down.double().t()
    rrot = resid @ hadamard(in_f)
    ws = rrot.abs().amax(1, keepdim=True) / NBITS_MAX
    ws = torch.where(ws == 0, torch.ones_like(ws), ws)
    ws = ws.float().to(torch.bfloat16).double()  # store-dtype rounded scale
    q = torch.clamp(torch.round(rrot / ws), -8, 7)
    return (
        pack_int4(q),
        ws.squeeze(1).to(torch.bfloat16).contiguous(),
        l_down.to(torch.bfloat16).contiguous(),
        l_up.to(torch.bfloat16).contiguous(),
    )


def w4a4_forward(x, qb, ws, ld, lu):
    """Reference forward: rotate+per-row-int4 the activation, int4 GEMM in
    doubles, add the low-rank branch (un-rotated x)."""
    in_f = qb.shape[1] * 2
    h = hadamard(in_f)
    rdeq = unpack_int4(qb).double() * ws.double().unsqueeze(1)
    xr = x.double() @ h
    xs = xr.abs().amax(1, keepdim=True) / NBITS_MAX
    xs = torch.where(xs == 0, torch.ones_like(xs), xs)
    xq = torch.clamp(torch.round(xr / xs), -8, 7) * xs
    return xq @ rdeq.t() + (x.double() @ ld.double()) @ lu.double().t()


def quantize_svdquant_w4a16(w: torch.Tensor, rank: int, group: int, full_svd: bool = False):
    """Weight-only W4A16 (no rotation): group-`group` x per-out int4 of the
    rank-R residual. The exact Class-A format ops/svdquant.mojo reads."""
    wf = w.float()
    out_f, in_f = wf.shape
    if in_f % group != 0:
        raise ValueError(f"w4a16: in={in_f} % group={group} != 0")
    l_down, l_up = _svd_factors(wf, rank, full_svd)
    resid = (wf.double() - l_up.double() @ l_down.double().t()).float()
    r = resid.view(out_f, in_f // group, group)
    gs = r.abs().amax(2, keepdim=True) / NBITS_MAX
    gs = torch.where(gs == 0, torch.ones_like(gs), gs)
    gs = gs.to(torch.bfloat16).float()
    q = torch.clamp(torch.round(r / gs), -8, 7).view(out_f, in_f)
    return (
        pack_int4(q),
        gs.view(out_f, in_f // group).t().contiguous().to(torch.bfloat16),
        l_down.to(torch.bfloat16).contiguous(),
        l_up.to(torch.bfloat16).contiguous(),
        torch.ones(in_f, dtype=torch.bfloat16),
    )


# ── streaming hybrid slab builder (legacy LTX-2 layout) ───────────────────────

_W4A4_IN = (2048, 4096, 8192)
_W4A16_IN = (16384,)
_CLASS_A_MARKS = (".attn1.", ".attn2.", ".ff.", ".audio_ff.", ".ff_context.")


def _classify(name: str, shape) -> str:
    if len(shape) != 2 or not name.endswith(".weight"):
        return "pass"
    if not any(m in name for m in _CLASS_A_MARKS):
        return "pass"
    in_f = shape[1]
    if in_f in _W4A4_IN:
        return "w4a4"
    if in_f in _W4A16_IN:
        return "w4a16"
    return "pass"


def build_svdquant_w4a4_slab(src: str, out: str, rank: int = 128, w4a16_rank: int = 32):
    """Two-pass streaming build: pass 1 classifies every tensor from metadata
    and declares the full output layout; pass 2 quantizes/copies one tensor at
    a time. Peak RAM = one source tensor + its encoding."""
    metrics = {"w4a4_count": 0, "w4a16_count": 0, "passthrough_count": 0,
               "sample_w4a4_cos": None, "slab_bytes": 0}
    with safe_open(src, "pt") as f:
        keys = list(f.keys())
        shapes = {}
        dtypes = {}
        for k in keys:
            sl = f.get_slice(k)
            shapes[k] = sl.get_shape()
            dtypes[k] = sl.get_dtype()

    w = StreamingSafetensorsWriter(out, metadata={"format": "svdquant_w4a4_hybrid",
                                                  "rank_w4a4": rank,
                                                  "rank_w4a16": w4a16_rank})
    plan = {}
    _ST2TORCH = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
                 "U8": torch.uint8, "I8": torch.int8, "I32": torch.int32,
                 "I64": torch.int64}
    for k in keys:
        cls = _classify(k, shapes[k])
        plan[k] = cls
        base = k[: -len(".weight")] if k.endswith(".weight") else k
        if cls == "w4a4":
            o, i = shapes[k]
            w.declare(base + ".qweight", torch.uint8, [o, i // 2])
            w.declare(base + ".wscale", torch.bfloat16, [o])
            w.declare(base + ".lora_down", torch.bfloat16, [i, rank])
            w.declare(base + ".lora_up", torch.bfloat16, [o, rank])
        elif cls == "w4a16":
            o, i = shapes[k]
            w.declare(base + ".qweight", torch.uint8, [o, i // 2])
            w.declare(base + ".wscales", torch.bfloat16, [i // 64, o])
            w.declare(base + ".lora_down", torch.bfloat16, [i, w4a16_rank])
            w.declare(base + ".lora_up", torch.bfloat16, [o, w4a16_rank])
            w.declare(base + ".smooth", torch.bfloat16, [i])
        else:
            w.declare(k, _ST2TORCH[dtypes[k]], shapes[k])
    w.write_header()

    try:
        with safe_open(src, "pt") as f:
            for n, k in enumerate(keys):
                cls = plan[k]
                base = k[: -len(".weight")] if k.endswith(".weight") else k
                if cls == "pass":
                    w.write_tensor(k, f.get_tensor(k))
                    metrics["passthrough_count"] += 1
                    continue
                wt = f.get_tensor(k).float()
                if cls == "w4a4":
                    qb, ws, ld, lu = quantize_svdquant_w4a4(wt, rank=rank)
                    w.write_tensor(base + ".qweight", qb)
                    w.write_tensor(base + ".wscale", ws)
                    w.write_tensor(base + ".lora_down", ld)
                    w.write_tensor(base + ".lora_up", lu)
                    if metrics["sample_w4a4_cos"] is None:
                        rec = (unpack_int4(qb).double() * ws.double().unsqueeze(1)
                               ) @ hadamard(wt.shape[1]) + lu.double() @ ld.double().t()
                        a = rec.flatten()
                        b = wt.double().flatten()
                        metrics["sample_w4a4_cos"] = (a @ b / (a.norm() * b.norm())).item()
                    metrics["w4a4_count"] += 1
                else:
                    qb, wsc, ld, lu, sm = quantize_svdquant_w4a16(wt, w4a16_rank, 64)
                    w.write_tensor(base + ".qweight", qb)
                    w.write_tensor(base + ".wscales", wsc)
                    w.write_tensor(base + ".lora_down", ld)
                    w.write_tensor(base + ".lora_up", lu)
                    w.write_tensor(base + ".smooth", sm)
                    metrics["w4a16_count"] += 1
                del wt
        w.close()
    except BaseException:
        w.abort()
        raise
    metrics["slab_bytes"] = os.path.getsize(out)
    return metrics
