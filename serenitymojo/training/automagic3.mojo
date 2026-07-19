# training/automagic3.mojo — pure-Mojo port of ai-toolkit Automagic3
# (/home/alex/ai-toolkit/toolkit/optimizers/automagic3.py), wired through
# training/levers.mojo (the ONE shared optimizer dispatch — Tier-1 modularity).
#
# Adaptive optimizer: an HF-Adafactor-FACTORED second moment (dim>=2; full
# per-element v for 1D) produces a magnitude-normalised update; a per-element
# UPDATE-SIGN history then drives ONE adaptive learning rate for the whole
# param GROUP (= the trained adapter slice, A and B pooled together):
#     lr *= exp(clamp(pooled_vote, -1, 1))                        once per step
# where, per element, over its last H update signs:
#     all H signs agree                 -> +|update|  ("step too small")
#     the H-1 transitions all flip      -> -|update|  ("overshoot")
#     else                              ->  0          (noise)
# pooled (|update|-weighted) over EVERY element of EVERY param in the group.
# The controller ABSTAINS for the first H steps (window warmup); the param
# update ALWAYS applies.
#
# Per-param update (mirrors automagic3.py:_update_param, fused=False path):
#   grad treated as f32 (LoRA grads from our backward are finite; the Python's
#     nan_to_num is omitted — documented).
#   sq = grad*grad.
#   dim>=2 (HF-factored, eps folded INSIDE the EMA add):
#     row = beta2*row + (1-beta2)*(mean(sq, dim=-1) + eps)        # shape rows
#     col = beta2*col + (1-beta2)*(mean(sq, dim=-2) + eps)        # shape cols
#     update = ( rsqrt(row / mean(row)) (x) rsqrt(col) ) * grad   # rank-1 recon
#   dim<2 (1D full second moment, eps added AFTER):
#     v = beta2*v + (1-beta2)*sq
#     update = rsqrt(v + eps) * grad
#   TWO-STAGE trust region (both required):
#     update /= max(RMS(update)/clip, 1.0)         # RMS clip (aggregate)
#     update  = clamp(update, -clip, +clip)        # element clamp (max-norm)
#   sign vote -> ctl accumulators (group-pooled, see above).
#   decoupled WD: if wd!=0: update += wd*p ; p -= lr*update.    (lr = ctl.lr)
#
# CRITICAL (rev-2 spec §2.1, landmine 1): automagic3's factored math is the
# HF/transformers reconstruction (beta2-EMA, eps-folded, rsqrt(row/mean(row))
# (x) rsqrt(col)) — NOT training/adafactor.mojo::adafactor_step_2d, which is
# torch.optim's DIFFERENT Adafactor (rho_t=min(lr,1/sqrt(t)),
# alpha=max(eps2,RMS(p))*rho_t, lerp EMA, var_est=outer/mean(row)). adafactor.mojo
# is reused ONLY as a reduction-loop STYLE template (f64 accumulation) and for
# the row_var[rows]/col_var[cols] state SHAPE — never the update formula. The
# Rust EriDiffusion-v2 port (optimizers.rs:557) reached the same conclusion.
#
# bf16 writeback uses STOCHASTIC ROUNDING (NOT plain RNE) — REQUIRED, not a
# nicety. The LoRA adapters a/b are bf16 storage (train_step.mojo:134-135), and
# at the controller's small self-adapted lr the per-step update lr*|update| is
# routinely BELOW a bf16 ULP for |w| >= ~0.1: plain round-to-nearest then maps
# most steps to NO CHANGE and the weight STALLS at its init (measured: a 0.1
# weight under lr=1e-4 constant-sign grads never moves under RNE, vs the f32
# trajectory descending to ~0.067). automagic3.py's own docstring (improvement
# #4) calls this out and uses _sr_truncate so fp16/bf16 "keeps learning instead
# of stalling". We reproduce _sr_truncate exactly (sr_truncate_f32_to_bits):
# view f32 bits as int, add a uniform dither in the dropped 16 mantissa bits,
# mask them off, then the bf16 cast is an exact truncation that rounds up with
# probability == the truncated fractional part (unbiased). The dither comes from
# a deterministic counter-based RNG (Automagic3Rng, splitmix64) so the gate is
# reproducible-given-seed. ONLY automagic3 does this — the other levers
# optimizers (adafactor/sf/a8) keep RNE because their references mandate it;
# automagic3 is the only one whose reference mandates SR.
#
# Port notes (intentional, matching the Rust port + the existing levers
# optimizers):
#   * Host math: List[Float32] f32-master from the bf16 a/b
#     (_levers_bf16_to_f32), the f32 update, then the SR bf16 writeback above.
#   * The sign-history ring is host-side List[List[Bool]] planes (NOT bit-packed):
#     the Python's 1-bit packing is a pure STORAGE optimization; the per-element
#     s1/flips/up/down VOTE is bit-identical either way, which is what the gate
#     checks. (For LoRA the adapter slices are small.)
#   * ONE group -> one shared adaptive lr in Automagic3Ctl, accumulating num/den
#     across the WHOLE [start,end) A+B loop and nudging the lr ONCE after the
#     loop (levers.mojo). The scheduler's step_lr is IGNORED (the controller
#     self-adapts).
#   * grad finite-guard: non-finite grads are zeroed before the EMA
#     (automagic3.py:416 nan_to_num) so one inf/NaN can't poison the factored
#     second moment permanently.
#   * Reductions accumulate in Float64 (reduction order is otherwise
#     unreproducible); element math is Float32. F32 parity bar: lr-traj rel <=
#     2%, params cos >= 0.9999 (training/parity/automagic3_parity_probe.mojo).
#     bf16 "moves-not-stalled" bar: levers_optimizer_dispatch am-bf16-* cases.
#
# Mojo 1.0.0b1.

from std.math import sqrt, exp


comptime AUTOMAGIC3_DEFAULT_LR = Float64(1.0e-6)
comptime AUTOMAGIC3_DEFAULT_BETA2 = Float64(0.999)
comptime AUTOMAGIC3_DEFAULT_EPS = Float64(1.0e-30)
comptime AUTOMAGIC3_DEFAULT_CLIP = Float64(1.0)
comptime AUTOMAGIC3_DEFAULT_WD = Float64(0.0)
comptime AUTOMAGIC3_DEFAULT_H = 8
comptime AUTOMAGIC3_LR_MIN = Float64(1.0e-30)  # overflow guard, NOT a control rail
comptime AUTOMAGIC3_LR_MAX = Float64(1.0e3)


def automagic3_clamp_h(polarity_history: Int) -> Int:
    """Clamp the sign-history window to [2, 64] (automagic3.py:195)."""
    var hh = polarity_history
    if hh < 2:
        hh = 2
    if hh > 64:
        hh = 64
    return hh


comptime AUTOMAGIC3_SR_DEFAULT_SEED = UInt64(0x9E3779B97F4A7C15)
"""Default SR-dither seed. Deterministic-given-seed so the bf16-moves gate is
reproducible; a real run can pass cfg's seed for run-to-run variation."""


struct Automagic3Rng(Copyable, Movable):
    """Counter-based deterministic RNG for the bf16 stochastic-rounding dither.
    A splitmix64 stream over an internal counter: each next_u32() advances the
    counter and hashes it, so the sequence is fully determined by the seed (the
    gate replays it exactly) yet decorrelated element-to-element. This is a
    TRAINER RNG (seeded/counter-based), NOT a workflow-script RNG."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_u32(mut self) -> UInt32:
        # splitmix64 (Steele et al.): advance, then avalanche-mix.
        self.state = self.state + UInt64(0x9E3779B97F4A7C15)
        var z = self.state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        # take the high 32 bits (better mixed than the low ones)
        return UInt32(z >> 32)


def sr_truncate_f32_to_bits(x: Float32, drop_bits: Int, rnd: UInt32) -> Float32:
    """Stochastic-rounding bit-trick for a low-precision float that is a
    mantissa TRUNCATION of f32 (bf16 drops 16 bits, fp16 drops 13).
    Mirrors automagic3.py::_sr_truncate: view f32 bits as uint32, add a uniform
    integer in [0, 1<<drop_bits) into the dropped low mantissa bits, then mask
    them off. The result's high bits ARE the low-precision value: a subsequent
    narrowing cast (e.g. BFloat16(...)) is then an EXACT truncation that rounds
    up with probability equal to the truncated fractional part (unbiased).
    `rnd` is the caller-supplied uniform draw (only its low drop_bits are used)."""
    var as_int = x.to_bits[DType.uint32]()
    var span = UInt32(1) << UInt32(drop_bits)
    var add = rnd & (span - UInt32(1))
    as_int = as_int + add
    var mask = ~(span - UInt32(1))
    as_int = as_int & mask
    return Float32(from_bits=as_int)


def automagic3_writeback_bf16_sr(
    mut p: List[BFloat16], src: List[Float32], mut rng: Automagic3Rng
):
    """Stochastic-rounding bf16 writeback: for each f32 master value, dither the
    dropped 16 mantissa bits with a fresh RNG draw, then truncate to bf16. This
    is the automagic3-ONLY writeback (the reference mandates SR); the f32->bf16
    cast after sr_truncate is exact because the low 16 bits are zeroed. Drops 16
    bits because bf16 is the high half of f32 (23-7 mantissa)."""
    for i in range(len(p)):
        var dithered = sr_truncate_f32_to_bits(src[i], 16, rng.next_u32())
        p[i] = BFloat16(dithered)


struct Automagic3State(Copyable, Movable):
    """Per-parameter Automagic3 state for ONE param (one LoRA matrix). A param
    is either `factored` (dim>=2, [rows, cols] row-major) keeping row_var[rows]
    / col_var[cols], or 1D keeping the full v[numel]. In both cases it owns the
    sign-history ring (h planes of `numel` bits, oldest at hist_idx) and the
    fill count. Mirrors automagic3.py state: exp_avg_sq_{row,col} or
    exp_avg_sq, sign_history (H, numel) bool, hist_idx, hist_fill, step.

    Construct via the two factory ctors:
      Automagic3State(rows, cols, h)  with cols>=1 -> factored 2D [rows,cols]
      Automagic3State(numel, h)       -> 1D full-v of length numel
    matching how AdafactorState(rows, cols) is built per A/B in levers.mojo."""

    var factored: Bool
    var rows: Int
    var cols: Int
    var numel: Int
    var h: Int
    var row_var: List[Float32]   # [rows] (factored) — empty for 1D
    var col_var: List[Float32]   # [cols] (factored) — empty for 1D
    var v: List[Float32]         # [numel] (1D) — empty for factored
    var sign_history: List[List[Bool]]  # h planes, each List[Bool] len numel
    var hist_idx: Int            # index of the OLDEST plane (overwritten next)
    var hist_fill: Int           # planes stored so far (controller gated < h)
    var step: Int

    def __init__(out self, rows: Int, cols: Int, h: Int):
        """Factored 2D [rows, cols]."""
        self.factored = True
        self.rows = rows
        self.cols = cols
        self.numel = rows * cols
        self.h = h
        self.row_var = List[Float32]()
        for _ in range(rows):
            self.row_var.append(Float32(0.0))
        self.col_var = List[Float32]()
        for _ in range(cols):
            self.col_var.append(Float32(0.0))
        self.v = List[Float32]()
        self.sign_history = List[List[Bool]]()
        self.hist_idx = 0
        self.hist_fill = 0
        self.step = 0

    def __init__(out self, numel: Int, h: Int):
        """1D full second moment of length numel."""
        self.factored = False
        self.rows = numel
        self.cols = 0
        self.numel = numel
        self.h = h
        self.row_var = List[Float32]()
        self.col_var = List[Float32]()
        self.v = List[Float32]()
        for _ in range(numel):
            self.v.append(Float32(0.0))
        self.sign_history = List[List[Bool]]()
        self.hist_idx = 0
        self.hist_fill = 0
        self.step = 0


struct Automagic3Ctl(Movable):
    """OPTIMIZER-level Automagic3 bookkeeping (ONE per optimizer, NOT per
    param) — the structural difference from every other (per-tensor) Mojo
    optimizer. Holds the single shared adaptive lr plus the transient per-step
    vote accumulators pooled across the WHOLE adapter group. Modeled on
    AdamWScheduleFreeCtl (which holds the optimizer-level k).

    Per step: reset_accum() once, run automagic3_step_2d/_1d for every A and B
    in [start,end) (each accumulates w*up - w*down into num and w into den),
    then apply_vote() ONCE to nudge lr."""

    var lr: Float64
    var group_num: Float64
    var group_den: Float64
    var initialized: Bool
    var rng: Automagic3Rng  # SR-dither stream (advances across the whole run)

    def __init__(out self):
        self.lr = AUTOMAGIC3_DEFAULT_LR
        self.group_num = 0.0
        self.group_den = 0.0
        self.initialized = False
        self.rng = Automagic3Rng(AUTOMAGIC3_SR_DEFAULT_SEED)

    def init_lr(mut self, start_lr: Float64):
        """Seed the START lr on the first step (the controller adapts away
        from it). Idempotent: only seeds once."""
        if not self.initialized:
            self.lr = start_lr
            self.initialized = True

    def seed_rng(mut self, seed: UInt64):
        """Seed the SR-dither RNG (call once at init, before the first step,
        for a reproducible-given-seed run). Idempotent under init_lr's guard
        is NOT applied here — callers seed exactly once at lazy-init."""
        self.rng = Automagic3Rng(seed)

    def reset_accum(mut self):
        """Clear the per-step pooled-vote accumulators (call ONCE before the
        adapter loop)."""
        self.group_num = 0.0
        self.group_den = 0.0

    def apply_vote(mut self):
        """ONE lr nudge from the pooled vote (call ONCE after the adapter
        loop). automagic3.py:587-597: signal = clamp(num/max(den,1e-30), ±1);
        lr = clamp(lr*exp(signal), 1e-30, 1e3). The clamp is an overflow guard,
        NOT a control rail. Abstains (no nudge) when den==0 (warmup: no param's
        window is full yet)."""
        if self.group_den > 0.0:
            var den = self.group_den
            if den < 1.0e-30:
                den = 1.0e-30
            var signal = self.group_num / den
            if signal > 1.0:
                signal = 1.0
            elif signal < -1.0:
                signal = -1.0
            var new_lr = self.lr * exp(signal)
            if new_lr < AUTOMAGIC3_LR_MIN:
                new_lr = AUTOMAGIC3_LR_MIN
            elif new_lr > AUTOMAGIC3_LR_MAX:
                new_lr = AUTOMAGIC3_LR_MAX
            self.lr = new_lr


def _automagic3_update_and_vote(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Automagic3State,
    lr: Float64,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    mut ctl: Automagic3Ctl,
) raises:
    """Core: compute the magnitude-normalised update for ONE param, two-stage
    clip it, slide the sign-history ring and (once the window is full)
    accumulate its |update|-weighted vote into `ctl`, then apply the param
    update in place with the SHARED lr. `lr` MUST be ctl.lr (the self-adapted
    value); passed explicitly so the caller controls timing. Handles both the
    factored (state.factored) and 1D paths."""
    var n = state.numel
    if len(p) != n or len(g) != n:
        raise Error("automagic3: param/grad length != numel")

    var beta2_f = Float32(beta2)
    var one_minus_b2 = Float32(1.0 - beta2)
    var eps_f = Float32(eps)
    var clip_f = Float32(clip)
    var h = state.h

    # --- grad finite-guard (F3 / automagic3.py:416 nan_to_num) ---
    # Neutralise non-finite grads to 0 BEFORE they enter the factored EMA: a
    # single NaN/inf would otherwise poison row_var/col_var permanently (NaN
    # stays NaN forever) and corrupt the weights. NaN fails self-equality;
    # +/-inf is caught by the finite-magnitude bound. Large-but-finite grads
    # are left alone (the second-moment normalisation already bounds them).
    var gsan = List[Float32](capacity=n)
    for i in range(n):
        var gv = g[i]
        if gv != gv:                                   # NaN
            gv = Float32(0.0)
        elif gv > Float32(3.0e38) or gv < Float32(-3.0e38):  # +/-inf
            gv = Float32(0.0)
        gsan.append(gv)

    # --- second moment + magnitude-normalised update ---
    var update = List[Float32](capacity=n)
    if state.factored:
        var rows = state.rows
        var cols = state.cols
        # row = beta2*row + (1-beta2)*(mean(sq, over cols) + eps), per row.
        for r in range(rows):
            var s = Float64(0.0)
            for c in range(cols):
                var gv = Float64(gsan[r * cols + c])
                s += gv * gv
            var row_mean = Float32(s / Float64(cols))
            state.row_var[r] = (
                beta2_f * state.row_var[r] + one_minus_b2 * (row_mean + eps_f)
            )
        # col = beta2*col + (1-beta2)*(mean(sq, over rows) + eps), per col.
        for c in range(cols):
            var s = Float64(0.0)
            for r in range(rows):
                var gv = Float64(gsan[r * cols + c])
                s += gv * gv
            var col_mean = Float32(s / Float64(rows))
            state.col_var[c] = (
                beta2_f * state.col_var[c] + one_minus_b2 * (col_mean + eps_f)
            )
        # update = rsqrt(row / mean(row)).unsqueeze(-1)
        #          * rsqrt(col).unsqueeze(-2) * grad
        # mean(row) is the scalar mean over the [rows] row vector (the 2D
        # row_state's last dim — automagic3.py:240).
        var rsum = Float64(0.0)
        for r in range(rows):
            rsum += Float64(state.row_var[r])
        var rmean = Float32(rsum / Float64(rows))
        var rfac = List[Float32](capacity=rows)
        for r in range(rows):
            rfac.append(Float32(1.0) / sqrt(state.row_var[r] / rmean))
        var cfac = List[Float32](capacity=cols)
        for c in range(cols):
            cfac.append(Float32(1.0) / sqrt(state.col_var[c]))
        for r in range(rows):
            for c in range(cols):
                update.append(rfac[r] * cfac[c] * gsan[r * cols + c])
    else:
        # 1D: v = beta2*v + (1-beta2)*sq ; update = rsqrt(v + eps) * grad.
        for i in range(n):
            var gv = gsan[i]
            state.v[i] = beta2_f * state.v[i] + one_minus_b2 * (gv * gv)
        for i in range(n):
            update.append((Float32(1.0) / sqrt(state.v[i] + eps_f)) * gsan[i])

    # --- two-stage trust-region clip ---
    # 1) RMS clip: update /= max(RMS(update)/clip, 1.0); RMS = ||u||_2/sqrt(n).
    var u_sq = Float64(0.0)
    for i in range(n):
        u_sq += Float64(update[i]) * Float64(update[i])
    var u_rms = sqrt(u_sq / Float64(n))
    var scale_div = u_rms / clip
    if scale_div < 1.0:
        scale_div = 1.0
    var inv_scale = Float32(1.0 / scale_div)
    for i in range(n):
        update[i] = update[i] * inv_scale
    # 2) element clamp to +/- clip (max-norm trust region).
    for i in range(n):
        if update[i] > clip_f:
            update[i] = clip_f
        elif update[i] < -clip_f:
            update[i] = -clip_f

    # --- sign vote: slide history ring, vote once window is full ---
    # cur_bit = (update > 0); an exact-zero update records the NEGATIVE bit
    # (its |update| weight is 0, so harmless).
    var cur_sign = List[Bool](capacity=n)
    for i in range(n):
        cur_sign.append(update[i] > Float32(0.0))

    var idx = state.hist_idx
    if len(state.sign_history) < h:
        state.sign_history.append(cur_sign^)
    else:
        state.sign_history[idx] = cur_sign^
    state.hist_idx = (idx + 1) % h
    var fill = state.hist_fill + 1
    if fill > h:
        fill = h
    state.hist_fill = fill

    if fill == h:
        # chronological start = OLDEST plane = new hist_idx (roll(-hist_idx)).
        # Per element: s1 = sum of H bits; flips = sum over H-1 adjacent
        # transitions; up = all agree (s1==H or s1==0); down = perfect
        # alternation (flips==H-1); weight = |update|. Accumulate into the
        # GROUP ctl (NOT per-tensor).
        var start_plane = state.hist_idx
        for e in range(n):
            var s1 = 0
            var flips = 0
            var prev = False
            for k in range(h):
                var b = state.sign_history[(start_plane + k) % h][e]
                if b:
                    s1 += 1
                if k > 0 and (b != prev):
                    flips += 1
                prev = b
            var w = Float64(update[e])
            if w < 0.0:
                w = -w
            if s1 == h or s1 == 0:
                ctl.group_num += w     # all agree -> step too small
            elif flips == (h - 1):
                ctl.group_num -= w     # perfect alternation -> overshoot
            ctl.group_den += w

    state.step += 1

    # --- decoupled weight decay + writeback (f32 master) ---
    # automagic3.py:530-535: if wd!=0: update += wd*p; p -= lr*update.
    var lr_f = Float32(lr)
    if weight_decay != 0.0:
        var wd_f = Float32(weight_decay)
        for i in range(n):
            update[i] = update[i] + wd_f * p[i]
    for i in range(n):
        p[i] = p[i] - lr_f * update[i]


def automagic3_step_2d(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Automagic3State,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    mut ctl: Automagic3Ctl,
) raises:
    """One Automagic3 step on a 2D [state.rows, state.cols] param (row-major).
    Uses ctl.lr (the shared, self-adapted lr) and accumulates this param's
    |update|-weighted vote into ctl. Caller resets ctl accumulators ONCE before
    the group loop and calls ctl.apply_vote() ONCE after."""
    if not state.factored:
        raise Error("automagic3_step_2d called on a 1D state")
    _automagic3_update_and_vote(
        p, g, state, ctl.lr, beta2, eps, clip, weight_decay, ctl
    )


def automagic3_step_1d(
    mut p: List[Float32],
    g: List[Float32],
    mut state: Automagic3State,
    beta2: Float64,
    eps: Float64,
    clip: Float64,
    weight_decay: Float64,
    mut ctl: Automagic3Ctl,
) raises:
    """One Automagic3 step on a 1D [state.numel] param (full second moment).
    Same ctl semantics as automagic3_step_2d."""
    if state.factored:
        raise Error("automagic3_step_1d called on a factored state")
    _automagic3_update_and_vote(
        p, g, state, ctl.lr, beta2, eps, clip, weight_decay, ctl
    )
