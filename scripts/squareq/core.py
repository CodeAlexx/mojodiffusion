"""SquareQ v3 core: grouped Hadamard, SVD low-rank init, int4-g64 codec,
NVFP4 codec oracle. Torch-only, CPU-safe, deterministic.

Everything here is the byte-level source of truth for the Mojo runtime gates:
the Mojo kernels must reproduce `dequant_int4_g64` bit-exactly on the ints and
`reconstruct_weight` to cos ~ 1.0 in bf16.
"""

import torch

FORMAT_TAG = "squareq_w4_v1"
HBLOCK = 256  # Hadamard block along the input dim; every K must be % 256 == 0
GROUP = 64  # scale group along the input dim (Class-A layout reuse)
QMAX = 7.0  # int4 twos-complement range [-8, 7]; absmax maps to +/-7

_H_CACHE: dict = {}


def hadamard_matrix(n: int, dtype=torch.float64, normalized: bool = True) -> torch.Tensor:
    """Sylvester Hadamard H_n (n a power of two). Normalized => orthonormal,
    symmetric, self-inverse: H @ H == I."""
    if n & (n - 1) != 0 or n < 1:
        raise ValueError(f"hadamard_matrix: n={n} not a power of two")
    key = (n, dtype, normalized)
    if key in _H_CACHE:
        return _H_CACHE[key]
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    if normalized:
        h = h / (float(n) ** 0.5)
    _H_CACHE[key] = h
    return h


def rht_grouped(t: torch.Tensor, block: int = HBLOCK) -> torch.Tensor:
    """Block-diagonal Hadamard along the LAST dim: [..., K] -> [..., K] with
    each contiguous `block`-segment rotated by normalized H_block.
    Self-inverse: rht_grouped(rht_grouped(t)) == t (up to fp)."""
    k = t.shape[-1]
    if k % block != 0:
        raise ValueError(f"rht_grouped: K={k} not divisible by block={block}")
    h = hadamard_matrix(block, dtype=t.dtype if t.dtype.is_floating_point else torch.float64)
    shape = t.shape
    return (t.reshape(*shape[:-1], k // block, block) @ h).reshape(shape)


def svd_lowrank_init(
    w: torch.Tensor, rank: int, niter: int = 6, seed: int = 0
) -> tuple:
    """Deterministic truncated-SVD init of the low-rank branch.
    Returns (lora_down [in, r] f32, lora_up [out, r] f32) with
    correction = lora_up @ lora_down^T ~ best rank-r approx of w [out, in]."""
    wd = w.float()
    out_f, in_f = wd.shape
    q = min(rank + 16, min(out_f, in_f))
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(wd, q=q, niter=niter)
    finally:
        torch.random.set_rng_state(gen_state)
    lora_up = (u[:, :rank] * s[:rank]).contiguous()  # [out, r]
    lora_down = v[:, :rank].contiguous()  # [in, r]
    return lora_down, lora_up


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """[out, in] int in [-8,7] -> u8 [out, in/2]. Twos-complement nibbles,
    LOW nibble = EVEN input index (svdquant.mojo SIGN_MODE/NIBBLE_ORDER)."""
    if q.shape[-1] % 2 != 0:
        raise ValueError("pack_int4: in dim must be even")
    qi = q.to(torch.int16) & 0xF  # twos-complement nibble
    lo = qi[..., 0::2]
    hi = qi[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8).contiguous()


def unpack_int4(qb: torch.Tensor) -> torch.Tensor:
    """u8 [out, in/2] -> int16 [out, in] in [-8, 7]."""
    b = qb.to(torch.int16)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    out = torch.empty(*qb.shape[:-1], qb.shape[-1] * 2, dtype=torch.int16)
    out[..., 0::2] = torch.where(lo >= 8, lo - 16, lo)
    out[..., 1::2] = torch.where(hi >= 8, hi - 16, hi)
    return out


# MSE scale-sweep fractions (ai-toolkit ConvRot precedent: absmax is not the
# MSE-optimal int4 scale; sweeping fractions of absmax per group and keeping
# the min-MSE one cut worst-layer output error ~34% combined with g32 —
# measured on Klein-4B single_blocks.0.linear2, 2026-07-28). Deterministic.
SCALE_SWEEP = (0.85, 0.88, 0.91, 0.94, 0.97, 1.0)


def quant_int4_g64(
    rrot: torch.Tensor, group: int = GROUP, mse_scales: bool = True
) -> tuple:
    """Quantize the rotated residual [out, in] to int4 with group-`group`
    (along in) x per-out scales. `mse_scales` sweeps SCALE_SWEEP fractions of
    absmax per group and keeps the min-MSE candidate (stored-bf16-rounded
    scales throughout). Returns (qweight u8 [out, in/2],
    wscales bf16 [in/group, out])."""
    out_f, in_f = rrot.shape
    if in_f % group != 0:
        raise ValueError(f"quant_int4_g64: in={in_f} not divisible by {group}")
    r = rrot.float().view(out_f, in_f // group, group)
    amax = r.abs().amax(2, keepdim=True)
    fracs = SCALE_SWEEP if mse_scales else (1.0,)
    best_q = best_gs = best_err = None
    for frac in fracs:
        gs = amax * frac / QMAX
        gs = torch.where(gs == 0, torch.ones_like(gs), gs)
        # bf16 round-trip the scale FIRST so quantization targets the stored scale.
        gs = gs.to(torch.bfloat16).float()
        q = torch.clamp(torch.round(r / gs), -8, 7)
        err = ((q * gs - r) ** 2).sum(2, keepdim=True)
        if best_err is None:
            best_q, best_gs, best_err = q, gs, err
        else:
            m = err < best_err
            best_q = torch.where(m, q, best_q)
            best_gs = torch.where(m, gs, best_gs)
            best_err = torch.where(m, err, best_err)
    qweight = pack_int4(best_q.view(out_f, in_f))
    wscales = best_gs.view(out_f, in_f // group).t().contiguous().to(torch.bfloat16)
    return qweight, wscales


def dequant_int4_g64(
    qweight: torch.Tensor, wscales: torch.Tensor, group: int = GROUP
) -> torch.Tensor:
    """Inverse of quant_int4_g64 -> f32 [out, in] (still rotated)."""
    v = unpack_int4(qweight).float()  # [out, in]
    out_f, in_f = v.shape
    sc = wscales.float().t()  # [out, in/group]
    return v * sc.repeat_interleave(group, dim=1)


def reconstruct_weight(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    hblock: int = HBLOCK,
    group: int = GROUP,
) -> torch.Tensor:
    """W_hat [out, in] f32 = dequant @ H_bd + lora_up @ lora_down^T.
    This is EXACTLY what the Mojo squareq_reconstruct_weight computes."""
    rrot = dequant_int4_g64(qweight, wscales, group=group)
    r = rht_grouped(rrot.double(), block=hblock).float()
    return r + lora_up.float() @ lora_down.float().t()


def quantize_layer(
    w: torch.Tensor,
    rank: int = 32,
    hblock: int = HBLOCK,
    group: int = GROUP,
    seed: int = 0,
) -> tuple:
    """Full SquareQ v3 encode of one linear weight w [out, in].
    Returns (tensors dict {qweight, wscales, lora_down, lora_up}, stats dict)."""
    out_f, in_f = w.shape
    if in_f % hblock != 0:
        raise ValueError(f"quantize_layer: in={in_f} not divisible by {hblock}")
    wf = w.float()
    lora_down, lora_up = svd_lowrank_init(wf, rank, seed=seed)
    resid = wf.double() - (lora_up.double() @ lora_down.double().t())
    rrot = rht_grouped(resid, block=hblock).float()
    qweight, wscales = quant_int4_g64(rrot, group=group)
    w_hat = reconstruct_weight(qweight, wscales, lora_down, lora_up, hblock, group)
    a = w_hat.flatten().double()
    b = wf.flatten().double()
    cos_w = (a @ b / (a.norm() * b.norm() + 1e-30)).item()
    rel = ((w_hat - wf).norm() / (wf.norm() + 1e-30)).item()
    bytes_q = (
        qweight.numel()
        + wscales.numel() * 2
        + (lora_down.numel() + lora_up.numel()) * 2
    )
    tensors = {
        "qweight": qweight,
        "wscales": wscales,
        "lora_down": lora_down.to(torch.bfloat16).contiguous(),
        "lora_up": lora_up.to(torch.bfloat16).contiguous(),
    }
    stats = {
        "out": out_f,
        "in": in_f,
        "rank": rank,
        "cos_w": cos_w,
        "rel_l2": rel,
        "bytes_q": bytes_q,
        "bytes_bf16": w.numel() * 2,
    }
    return tensors, stats


# ── NVFP4 codec oracle (chunk 7: native Blackwell FP4 GEMM slab variant) ──────
# Layout: values e2m1 packed 2/byte (low nibble = even index) in blocks of 16
# along the last dim; each block scaled by an UNSIGNED e4m3 byte scale; one
# global f32 scale per tensor. Matches cuBLASLt VEC16_UE4M3 expectations.

_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _ue4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """f32 (>=0) -> unsigned-e4m3 byte (4-bit exp bias 7, 3-bit mantissa,
    saturating, no sign bit)."""
    x = x.float().clamp(min=0.0)
    byte = torch.zeros_like(x, dtype=torch.uint8)
    pos = x > 0
    xe = x[pos]
    e = torch.floor(torch.log2(xe)).clamp(-6, 8)
    m = torch.round((xe / torch.pow(2.0, e) - 1.0) * 8.0).clamp(0, 7)
    # mantissa overflow rounds up the exponent
    ovf = m > 7
    e = torch.where(ovf, e + 1, e)
    m = torch.where(ovf, torch.zeros_like(m), m)
    e = e.clamp(-6, 8)
    sub = e < -6  # subnormal range -> flush toward zero handling below
    biased = (e + 7).clamp(0, 15)
    b = (biased.to(torch.uint8) << 3) | m.to(torch.uint8)
    b = torch.where(sub, torch.zeros_like(b), b)
    byte[pos] = b
    return byte


def _ue4m3_decode(b: torch.Tensor) -> torch.Tensor:
    e = ((b.to(torch.int16) >> 3) & 0xF).float() - 7.0
    m = (b.to(torch.int16) & 0x7).float() / 8.0
    v = torch.pow(2.0, e) * (1.0 + m)
    return torch.where(b == 0, torch.zeros_like(v), v)


def nvfp4_encode(t: torch.Tensor, block: int = 16) -> tuple:
    """f32 [..., K] -> (packed u8 [..., K/2], block_scales u8 [..., K/block],
    global_scale f32 tensor []). Decode with nvfp4_decode."""
    k = t.shape[-1]
    if k % block != 0:
        raise ValueError(f"nvfp4_encode: K={k} not divisible by {block}")
    x = t.float()
    gs = x.abs().amax()
    gs = gs / (6.0 * 448.0)  # e2m1 max * e4m3 max
    if gs == 0:
        gs = torch.tensor(1.0)
    xb = (x / gs).reshape(*x.shape[:-1], k // block, block)
    bs = xb.abs().amax(-1, keepdim=True) / 6.0
    bs_b = _ue4m3_encode(bs.squeeze(-1))
    bs_d = _ue4m3_decode(bs_b).unsqueeze(-1)
    bs_d = torch.where(bs_d == 0, torch.ones_like(bs_d), bs_d)
    v = xb / bs_d
    sign = (v < 0).to(torch.uint8) << 3
    mag = v.abs().unsqueeze(-1)
    idx = (mag - _E2M1_MAG.view(1, 1, 1, -1).to(mag.dtype)).abs().argmin(-1)
    code = (sign | idx.to(torch.uint8)).reshape(*x.shape[:-1], k)
    lo = code[..., 0::2]
    hi = code[..., 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed, bs_b.contiguous(), gs.float().reshape(())


def nvfp4_decode(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
    block: int = 16,
) -> torch.Tensor:
    b = packed.to(torch.int16)
    codes = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int16)
    codes[..., 0::2] = b & 0xF
    codes[..., 1::2] = (b >> 4) & 0xF
    mag = _E2M1_MAG[(codes & 0x7).long()]
    val = torch.where((codes & 0x8) != 0, -mag, mag)
    k = val.shape[-1]
    bs = _ue4m3_decode(block_scales).unsqueeze(-1)
    out = (val.reshape(*val.shape[:-1], k // block, block) * bs).reshape(val.shape)
    return out * global_scale.float()


# ── NVFP4 tiled weight payload (chunk 7: native FP4 forward) ─────────────────
# cuBLASLt VEC16_UE4M3 wants block scales in 512-byte tiles of 128 rows x 4
# scale-cols with a 32-row interleave (verified bit-exact by
# serenitymojo/ops/parity/fp4_layout_probe.mojo).

def _phys_scale_offset(r: int, c: int, kb_pad4: int) -> int:
    tile = (r // 128) * (kb_pad4 // 4) + (c // 4)
    return tile * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + (c % 4)


def scale_tiles_bytes(rows: int, kblocks: int) -> int:
    rows_pad = ((rows + 127) // 128) * 128
    kb_pad = ((kblocks + 3) // 4) * 4
    return (rows_pad // 128) * (kb_pad // 4) * 512


def nvfp4_encode_weight_tiled(rrot: torch.Tensor) -> tuple:
    """Rotated residual [out, in] -> (nvq u8 [out,in/2] e2m1 lo=even,
    nvs u8 TILED ue4m3 block-16 scales, nvg f32 per-tensor global scale).
    Decode contract: value = e2m1(code) * ue4m3(scale) * nvg."""
    out_f, in_f = rrot.shape
    if in_f % 16 != 0:
        raise ValueError("nvfp4_encode_weight_tiled: in % 16 != 0")
    x = rrot.float()
    g = (x.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30)
    xb = (x / g).reshape(out_f, in_f // 16, 16)
    bs = xb.abs().amax(-1) / 6.0
    bs_b = _ue4m3_encode(bs)
    bs_d = _ue4m3_decode(bs_b)
    safe = torch.where(bs_d == 0, torch.ones_like(bs_d), bs_d)
    v = xb / safe.unsqueeze(-1)
    v = torch.where((bs_d == 0).unsqueeze(-1), torch.zeros_like(v), v)
    sign = (v < 0).to(torch.uint8) << 3
    idx = (v.abs().unsqueeze(-1) - _E2M1_MAG.view(1, 1, 1, -1)).abs().argmin(-1)
    code = (sign | idx.to(torch.uint8)).reshape(out_f, in_f)
    nvq = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    kb = in_f // 16
    kb_pad = ((kb + 3) // 4) * 4
    nvs = torch.zeros(scale_tiles_bytes(out_f, kb), dtype=torch.uint8)
    rr = torch.arange(out_f).view(-1, 1).expand(out_f, kb)
    cc = torch.arange(kb).view(1, -1).expand(out_f, kb)
    tile = (rr // 128) * (kb_pad // 4) + (cc // 4)
    off = tile * 512 + (rr % 32) * 16 + ((rr // 32) % 4) * 4 + (cc % 4)
    nvs[off.reshape(-1)] = bs_b.reshape(-1)
    return nvq, nvs, g.float().reshape(1)


def nvfp4_decode_weight_tiled(nvq, nvs, nvg, out_f: int, in_f: int) -> torch.Tensor:
    """Inverse of nvfp4_encode_weight_tiled -> f32 [out, in] (still rotated)."""
    b = nvq.to(torch.int16)
    codes = torch.empty(out_f, in_f, dtype=torch.int16)
    codes[:, 0::2] = b & 0xF
    codes[:, 1::2] = (b >> 4) & 0xF
    mag = _E2M1_MAG[(codes & 0x7).long()]
    val = torch.where((codes & 0x8) != 0, -mag, mag)
    kb = in_f // 16
    kb_pad = ((kb + 3) // 4) * 4
    rr = torch.arange(out_f).view(-1, 1).expand(out_f, kb)
    cc = torch.arange(kb).view(1, -1).expand(out_f, kb)
    tile = (rr // 128) * (kb_pad // 4) + (cc // 4)
    off = tile * 512 + (rr % 32) * 16 + ((rr // 32) % 4) * 4 + (cc % 4)
    bs = _ue4m3_decode(nvs[off.reshape(-1)].reshape(out_f, kb))
    return (val.reshape(out_f, kb, 16) * bs.unsqueeze(-1)).reshape(out_f, in_f) * nvg.float()
