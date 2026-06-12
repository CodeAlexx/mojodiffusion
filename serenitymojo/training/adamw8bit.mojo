# training/adamw8bit.mojo — host-math 8-bit AdamW, bitsandbytes 0.49.2
# block-wise parity (T2.A).
#
# Reference (the proven Rust-stack oracle chain):
#   * algorithm: /home/alex/EriDiffusion/flame-core/src/adam8bit_kernel.rs —
#     pure port of the on-device math of bnb `optim.AdamW8bit`
#     (block_wise=True), byte-parity-gated against bnb 0.49.2 by
#     crates/eridiffusion-cli/src/bin/parity_adam8bit_bnb{,_wd,_tail,
#     _bf16grad,_multistep}.rs.
#   * oracle dumps: /home/alex/EriDiffusion/EriDiffusion-v2/tests/parity/
#     adam8bit_data/ — bnb 0.49.2 F.optimizer_update_8bit_blockwise("adam",
#     ...) before/after snapshots produced by tests/parity/
#     adam8bit_bnb_python_ref*.py (bnb 0.49.2 + torch 2.10.0+cu128).
#
# State layout per parameter (n elements):
#   m_codes  [n]            u8 codes into the SIGNED dynamic LUT (qmap1)
#   v_codes  [n]            u8 codes into the UNSIGNED dynamic LUT (qmap2)
#   m_absmax [ceil(n/256)]  per-256-block F32 scale for m
#   v_absmax [ceil(n/256)]  per-256-block F32 scale for v
# Dequant of element i: qmap[code[i]] * absmax[i / 256]. Block size is
# HARDCODED to 256 in bnb (optim/optimizer.py:478) — do not change.
#
# Per-step math (bnb "adam" with weight_decay — bnb's C++ blockwise kernel
# applies wd DECOUPLED, proven by the Rust _wd parity bin):
#   m_old = qmap_s[m_code] * m_absmax[blk]
#   v_old = qmap_u[v_code] * v_absmax[blk]
#   m_new = beta1*m_old + (1-beta1)*g
#   v_new = beta2*v_old + (1-beta2)*g*g
#   p    -= lr * (m_new/bc1) / (sqrt(v_new/bc2) + eps)     bc = 1 - beta^t
#   if wd != 0:  p -= lr*wd*p                              (decoupled, AFTER)
#   absmax' = max_block |new|  (fallback 1e-12 if a block is all-zero)
#   code'   = argmin_c |qmap[c] - new/absmax'|             (first-wins ties)
# ALL element math is Float32 (the bnb kernel is f32) — measured on the bnb
# dumps (adamw8bit_parity.mojo, 2026-06-11): codes EXACT-EQUAL on every f32
# scenario incl. all 10 multistep steps; bf16grad has 2 signed-code
# mismatches (LUT-tiebreak boundary cases, inside the Rust ref bin's own
# <=5 allowance); param max|Δ| <= 2.4e-7 (~1.5 f32 ulp of the updated
# params, FAR below one 8-bit quantum); absmax max|Δ| = 0.0 (bit-exact).
#
# LUT (adam8bit_create_dynamic_map): port of bnb functional.py
# create_dynamic_map(signed, 7, 8). BIT-EXACT vs bnb's dump requires
# reproducing torch.linspace(0.1, 1, k, f32)'s CPU kernel exactly:
#   step = (1.0f - 0.1f) / f32(k-1)
#   j <  k//2 : fmaf(step,  f32(j),       0.1f)   (single-rounding FMA)
#   j >= k//2 : fmaf(-step, f32(k-1-j),   1.0f)   (filled BACK from `end`)
# then means = (b[j]+b[j+1])/2 in f32, scaled by the f32 power of ten.
# (Empirically verified bit-exact against before.qmap1/qmap2 — plain
# `0.1+j*step` in either f32 or f64 is NOT bit-exact at k>=17.)
#
# Host-math first (List[Float32] params/grads, the adafactor.mojo T1.C
# pattern); a fused GPU kernel can land later behind the same levers
# dispatch. bf16 grads: upcast bf16->f32 BEFORE calling (bnb's host dispatch
# does g.float(); the upcast is exact).
#
# Parity gate: training/tests/adamw8bit_parity.mojo vs the bnb dumps
# (5 scenarios: basic, weight-decay, tail block, bf16 grads, 10-step).
#
# Mojo 1.0.0b1.

from std.math import fma, sqrt

comptime ADAMW8BIT_BLOCK_SIZE = 256
"""bnb block size for the blockwise 8-bit optimizer state
(optim/optimizer.py:478). Hardcoded in bnb's kernels; do not change."""


def _linspace_f32_torch(k: Int) -> List[Float32]:
    """torch.linspace(0.1, 1, k, dtype=f32) CPU-kernel bit-exact:
    f32 step, FMA fill, second half filled backwards from `end`."""
    var out = List[Float32](capacity=k)
    var step = (Float32(1.0) - Float32(0.1)) / Float32(k - 1)
    var half = k // 2
    for j in range(k):
        if j < half:
            out.append(fma(step, Float32(j), Float32(0.1)))
        else:
            out.append(fma(-step, Float32(k - 1 - j), Float32(1.0)))
    return out^


def adam8bit_create_dynamic_map(signed: Bool) raises -> List[Float32]:
    """256-entry dynamic-exponent qmap, bnb 0.49.2 create_dynamic_map(signed,
    max_exponent_bits=7, total_bits=8). signed=True -> the m LUT ("dynamic"),
    signed=False -> the v LUT ("udynamic"). Sorted ascending; bit-exact vs
    bnb's torch output (gated in adamw8bit_parity.mojo at 0.0 max|Δ|)."""
    # f32 powers of ten 1e-6..1e0 (literal rounding == torch's f32 cast of
    # the python float 10**(-6+i)).
    var scales = List[Float32]()
    scales.append(Float32(1.0e-6))
    scales.append(Float32(1.0e-5))
    scales.append(Float32(1.0e-4))
    scales.append(Float32(1.0e-3))
    scales.append(Float32(1.0e-2))
    scales.append(Float32(1.0e-1))
    scales.append(Float32(1.0))

    var data = List[Float32]()
    for i in range(7):
        var fraction_items: Int
        if signed:
            fraction_items = (1 << i) + 1
        else:
            fraction_items = (1 << (i + 1)) + 1
        var bd = _linspace_f32_torch(fraction_items)
        var scale = scales[i]
        for j in range(fraction_items - 1):
            var mean = (bd[j] + bd[j + 1]) / Float32(2.0)
            data.append(scale * mean)
        if signed:
            for j in range(fraction_items - 1):
                var mean = (bd[j] + bd[j + 1]) / Float32(2.0)
                data.append(-(scale * mean))
    data.append(Float32(0.0))
    data.append(Float32(1.0))
    if len(data) != 256:
        # bnb asserts len == 2**total_bits with these constants (254 means
        # + 0 + 1 for both signed and unsigned).
        raise Error(
            String("adam8bit_create_dynamic_map: ")
            + String(len(data))
            + String(" entries != 256")
        )
    # Ascending insertion sort (256 entries; avoids stdlib sort API churn).
    for i in range(1, 256):
        var v = data[i]
        var j = i - 1
        while j >= 0 and data[j] > v:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = v
    return data^


struct Adam8bitState(Copyable, Movable):
    """Per-parameter block-wise 8-bit AdamW moment state. Zero-init matches
    bnb Optimizer8bit.init_state (optim/optimizer.py:497-519): all codes 0,
    all absmax 0.0 -> initial dequant is exactly 0 regardless of LUT."""

    var m_codes: List[UInt8]
    var v_codes: List[UInt8]
    var m_absmax: List[Float32]
    var v_absmax: List[Float32]
    var n: Int
    var step: Int  # completed steps (0 before the first step; bnb is 1-based)

    def __init__(out self, n: Int):
        self.m_codes = List[UInt8](capacity=n)
        self.v_codes = List[UInt8](capacity=n)
        for _ in range(n):
            self.m_codes.append(UInt8(0))
            self.v_codes.append(UInt8(0))
        var blocks = (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
        self.m_absmax = List[Float32](capacity=blocks)
        self.v_absmax = List[Float32](capacity=blocks)
        for _ in range(blocks):
            self.m_absmax.append(Float32(0.0))
            self.v_absmax.append(Float32(0.0))
        self.n = n
        self.step = 0


def _pow_f32(base: Float32, e: Int) -> Float32:
    # f32 repeated multiply (bias correction beta^t; t small, 1-ulp class
    # differences vs powi are ~1e-11 on the param — far inside the gate).
    var out = Float32(1.0)
    for _ in range(e):
        out = out * base
    return out


def adam8bit_step_bnb(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Adam8bitState,
    qmap_signed: List[Float32],
    qmap_unsigned: List[Float32],
    step: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises:
    """One bnb-parity blockwise 8-bit AdamW step at 1-based `step` (the bias
    correction exponent). Mutates p and state in place; does NOT touch
    state.step (the gate drives arbitrary before-states; trainers use
    adamw8bit_step below). g must be F32 (upcast bf16 grads BEFORE calling
    — exact, mirrors bnb's host g.float())."""
    var n = state.n
    if len(p) != n or len(g) != n:
        raise Error("adam8bit_step_bnb: p/g length != state.n")
    if len(qmap_signed) != 256 or len(qmap_unsigned) != 256:
        raise Error("adam8bit_step_bnb: qmap length != 256")
    if step < 1:
        raise Error("adam8bit_step_bnb: step must be >= 1")
    var n_blocks = (n + ADAMW8BIT_BLOCK_SIZE - 1) // ADAMW8BIT_BLOCK_SIZE
    if len(state.m_absmax) < n_blocks or len(state.v_absmax) < n_blocks:
        raise Error("adam8bit_step_bnb: absmax buffers too small")

    var bc1 = Float32(1.0) - _pow_f32(beta1, step)
    var bc2 = Float32(1.0) - _pow_f32(beta2, step)
    var one_m_b1 = Float32(1.0) - beta1
    var one_m_b2 = Float32(1.0) - beta2

    var m_new = List[Float32](capacity=ADAMW8BIT_BLOCK_SIZE)
    var v_new = List[Float32](capacity=ADAMW8BIT_BLOCK_SIZE)
    for _ in range(ADAMW8BIT_BLOCK_SIZE):
        m_new.append(Float32(0.0))
        v_new.append(Float32(0.0))

    for blk in range(n_blocks):
        var base = blk * ADAMW8BIT_BLOCK_SIZE
        var cnt = min(ADAMW8BIT_BLOCK_SIZE, n - base)
        var am_prev = state.m_absmax[blk]
        var av_prev = state.v_absmax[blk]

        # Pass 1: dequant, AdamW math, param update, stash new moments.
        for t in range(cnt):
            var i = base + t
            var gv = g[i]
            var m_old = qmap_signed[Int(state.m_codes[i])] * am_prev
            var v_old = qmap_unsigned[Int(state.v_codes[i])] * av_prev
            var mn = beta1 * m_old + one_m_b1 * gv
            var vn = beta2 * v_old + one_m_b2 * gv * gv
            m_new[t] = mn
            v_new[t] = vn
            var m_hat = mn / bc1
            var v_hat = vn / bc2
            var upd = lr * m_hat / (sqrt(v_hat) + eps)
            var pv = p[i] - upd
            if weight_decay != Float32(0.0):
                pv = pv - lr * weight_decay * pv  # decoupled, AFTER the update
            p[i] = pv

        # Block max-abs reduction (max is order-independent — exact).
        var amm = Float32(0.0)
        var amv = Float32(0.0)
        for t in range(cnt):
            var am = m_new[t] if m_new[t] >= Float32(0.0) else -m_new[t]
            var av = v_new[t] if v_new[t] >= Float32(0.0) else -v_new[t]
            if am > amm:
                amm = am
            if av > amv:
                amv = av
        if amm == Float32(0.0):
            amm = Float32(1.0e-12)  # all-zero-block guard (kernel parity)
        if amv == Float32(0.0):
            amv = Float32(1.0e-12)
        state.m_absmax[blk] = amm
        state.v_absmax[blk] = amv

        # Pass 2: requant — linear argmin scan, strict < (first/lowest code
        # wins ties — the kernel's tiebreak).
        for t in range(cnt):
            var i = base + t
            var m_norm = m_new[t] / amm
            var v_norm = v_new[t] / amv
            var best_m = 0
            var d0m = qmap_signed[0] - m_norm
            var best_m_d = d0m if d0m >= Float32(0.0) else -d0m
            var best_v = 0
            var d0v = qmap_unsigned[0] - v_norm
            var best_v_d = d0v if d0v >= Float32(0.0) else -d0v
            for c in range(1, 256):
                var dm = qmap_signed[c] - m_norm
                if dm < Float32(0.0):
                    dm = -dm
                if dm < best_m_d:
                    best_m_d = dm
                    best_m = c
                var dv = qmap_unsigned[c] - v_norm
                if dv < Float32(0.0):
                    dv = -dv
                if dv < best_v_d:
                    best_v_d = dv
                    best_v = c
            state.m_codes[i] = UInt8(best_m)
            state.v_codes[i] = UInt8(best_v)


def adamw8bit_step(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Adam8bitState,
    qmap_signed: List[Float32],
    qmap_unsigned: List[Float32],
    k: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
) raises:
    """Trainer entry: one step at trainer step k (1-based, the same t the
    AdamW path passes). Fails loud on a step-count desync (no save/resume
    sidecar yet — the levers contract)."""
    if state.step != k - 1:
        raise Error(
            String("adamw8bit_step: step desync (state.step=")
            + String(state.step)
            + String(", trainer step=")
            + String(k)
            + String(")")
        )
    adam8bit_step_bnb(
        p, g, state, qmap_signed, qmap_unsigned, k,
        lr, beta1, beta2, eps, weight_decay,
    )
    state.step = k
