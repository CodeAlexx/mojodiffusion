# training/adamw_schedulefree.mojo — host-math "schedule-free" AdamW (T1.C).
#
# Reference (the implementation SimpleTuner actually invokes): SimpleTuner key
# "adamw_schedulefree" -> AdamWScheduleFreeKahan — SimpleTuner's IN-HOUSE
# class, NOT facebookresearch/schedule_free:
#   * class: /home/alex/SimpleTuner/simpletuner/helpers/training/optimizers/
#     adamw_schedulefree/__init__.py:10-145 (imported at optimizer_param.py:20)
#   * registered: optimizer_param.py:249-259 with default_settings
#     betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8, and
#     override_lr_scheduler=True / is_schedulefree=True / can_warmup=True;
#     warmup_steps := args.lr_warmup_steps (optimizer_param.py:1114-1116).
#
# step() math (__init__.py:79-142), k = optimizer-level 0-based step counter:
#   sched        = (k+1)/warmup_steps if k < warmup_steps else 1     # :96-99
#   bc2          = 1 - beta2^(k+1)                                   # :101
#   adjusted_lr  = lr * sched * sqrt(bc2)                            # :102
#   per element:
#     g    += kahan_comp                  (kahan_sum)                # :115-117
#     m     = beta1*m + (1-beta1)*g                                  # :120
#     v     = beta2*v + (1-beta2)*g*g                                # :121
#     denom = sqrt(v) + eps                                          # :123
#     step_size = adjusted_lr / sqrt(bc2)   (= lr*sched)             # :125
#     if wd != 0: p += -wd * p   (decoupled, NOT lr-scaled)          # :127-128
#     p    += -step_size * (m/denom)                                 # :131-132
#     kahan update                                                   # :134-136
#   k += 1                                                           # :138
# There is NO bias correction on m (only the bc2 factor, which cancels in
# step_size) — faithful to the reference, do not "fix" it.
#
# ── REFERENCE QUIRKS (verified against the oracle run, 2026-06-11) ──────────
# 1. The Kahan compensation is mathematically DEAD CODE: kahan_comp is set to
#    fl(p-buf) + fl(buf-p) (__init__.py:134-136), and IEEE-754 subtraction is
#    correctly rounded + symmetric under negation, so the two terms are exact
#    negatives -> kahan_comp == 0.0 ALWAYS (oracle: max|kahan_comp| = 0.0
#    after 20 steps). We replicate the ops verbatim anyway.
# 2. train()/eval() (__init__.py:55-77) lerp p toward state["z"] — but
#    state["z"] is NEVER created (_initialize_state :47-53 has no "z"), so
#    both are no-ops; train weights == eval weights in this reference
#    (oracle asserts torch.equal after eval()).
#
# ── THE SCHEDULEFREE SAVE GOTCHA (design contract, read this) ────────────────
# In TRUE schedule-free (facebookresearch), the model trains on the y iterate
# but the weights you evaluate/SAVE are the x iterate: you MUST call
# optimizer.eval() before saving/sampling and optimizer.train() before
# resuming, or you ship the wrong weights. SimpleTuner's class keeps that API
# (and SimpleTuner calls eval() around save/validation) even though its
# implementation makes them no-ops. Our port mirrors the same contract:
# trainers MUST call adamw_schedulefree_eval() before save / validation
# sampling and adamw_schedulefree_train() before resuming the loop, so that
# if the dead z-iterate is ever fixed upstream (or we swap in the real
# facebookresearch math) no trainer changes are needed.
#
# Host-math first (List[Float32] params/grads, per-param state struct +
# optimizer-level ctl struct); fused GPU path later behind the same dispatch.
#
# Parity gate: training/tests/optimizer_parity.mojo vs
# /tmp/optimizer_oracle.safetensors (gen_optimizer_oracle.py, 20 steps).
#
# Mojo 1.0.0b1.

from std.math import sqrt


struct AdamWScheduleFreeState(Copyable, Movable):
    """Per-parameter state: exp_avg / exp_avg_sq / kahan_comp, all F32 zeros
    at init (reference _initialize_state, __init__.py:47-53)."""

    var exp_avg: List[Float32]
    var exp_avg_sq: List[Float32]
    var kahan_comp: List[Float32]

    def __init__(out self, n: Int):
        self.exp_avg = List[Float32]()
        self.exp_avg_sq = List[Float32]()
        self.kahan_comp = List[Float32]()
        for _ in range(n):
            self.exp_avg.append(Float32(0.0))
            self.exp_avg_sq.append(Float32(0.0))
            self.kahan_comp.append(Float32(0.0))


struct AdamWScheduleFreeCtl(Movable):
    """OPTIMIZER-level bookkeeping (one per optimizer, NOT per param):
    the 0-based step counter k (reference self.k), lr_max/last_lr, and
    train_mode. Step all params with the SAME k, then call end_step()
    ONCE — mirrors the reference where self.k += 1 happens after the
    param loop (__init__.py:93, :138)."""

    var k: Int
    var lr_max: Float64
    var last_lr: Float64
    var train_mode: Bool

    def __init__(out self):
        self.k = 0
        self.lr_max = -1.0
        self.last_lr = -1.0
        self.train_mode = True

    def end_step(mut self, adjusted_lr: Float64):
        """Call once after stepping every param for this k (:103, :138-139)."""
        if adjusted_lr > self.lr_max:
            self.lr_max = adjusted_lr
        self.last_lr = adjusted_lr
        self.k += 1


def adamw_schedulefree_adjusted_lr(
    k: Int, lr: Float64, beta2: Float64, warmup_steps: Int
) -> Float64:
    """adjusted_lr for step k (0-based): lr * warmup_sched * sqrt(1-beta2^(k+1))
    (__init__.py:96-102). Pass the result to end_step()."""
    var sched = 1.0
    if k < warmup_steps:
        sched = Float64(k + 1) / Float64(warmup_steps)
    var bc2 = 1.0 - beta2 ** Float64(k + 1)
    return lr * sched * sqrt(bc2)


def adamw_schedulefree_step_param(
    mut p: List[Float32],
    g: List[Float32],
    mut state: AdamWScheduleFreeState,
    k: Int,
    lr: Float64,
    beta1: Float64,
    beta2: Float64,
    eps: Float64,
    weight_decay: Float64,
    warmup_steps: Int,
    kahan_sum: Bool,
) raises:
    """One AdamWScheduleFreeKahan step for ONE param at optimizer step k
    (0-based, from AdamWScheduleFreeCtl.k). Mutates p and state in place;
    g is read-only (the reference mutates p.grad in place at :117 — we add
    kahan_comp into a local instead, same value). SimpleTuner
    "adamw_schedulefree" settings: betas=(0.9,0.999), eps=1e-8,
    weight_decay=1e-2, warmup_steps=args.lr_warmup_steps, kahan_sum=True."""
    var n = len(p)
    if len(g) != n or len(state.exp_avg) != n or len(state.exp_avg_sq) != n:
        raise Error("adamw_schedulefree_step_param: length mismatch")
    if kahan_sum and len(state.kahan_comp) != n:
        raise Error("adamw_schedulefree_step_param: kahan_comp length mismatch")

    var sched = 1.0
    if k < warmup_steps:
        sched = Float64(k + 1) / Float64(warmup_steps)  # :96-97
    var bc2 = 1.0 - beta2 ** Float64(k + 1)  # :101
    var adjusted_lr = lr * sched * sqrt(bc2)  # :102
    var step_size = Float32(adjusted_lr / sqrt(bc2))  # :125 (= lr*sched)

    var b1 = Float32(beta1)
    var b2 = Float32(beta2)
    var one_m_b1 = Float32(1.0) - b1
    var one_m_b2 = Float32(1.0) - b2
    var epsf = Float32(eps)
    var wdf = Float32(weight_decay)

    for i in range(n):
        var gv = g[i]
        if kahan_sum:
            gv = gv + state.kahan_comp[i]  # :115-117 (always +0.0, see header)
        var m = state.exp_avg[i] * b1 + one_m_b1 * gv  # :120
        var v = state.exp_avg_sq[i] * b2 + one_m_b2 * gv * gv  # :121
        state.exp_avg[i] = m
        state.exp_avg_sq[i] = v
        var denom = sqrt(v) + epsf  # :123
        var pv = p[i]
        if weight_decay != 0.0:
            pv = pv - wdf * pv  # :127-128 (p += -wd*p)
        var upd = (-step_size) * (m / denom)  # :131-132
        pv = pv + upd
        if kahan_sum:
            # :134-136 verbatim: buffer = p.add(-step_size*step);
            # kahan_comp = (p - buffer) + (buffer - p)  [== 0.0, kept faithful]
            var buffer = pv + upd
            var t1 = pv - buffer
            var t2 = buffer - pv
            state.kahan_comp[i] = t1 + t2
        p[i] = pv


def adamw_schedulefree_eval(
    mut ctl: AdamWScheduleFreeCtl,
    mut params: List[List[Float32]],
):
    """Switch to EVAL weights — MUST be called before save / validation
    sampling (the schedulefree save gotcha, see header). Reference
    (__init__.py:55-65): lerp p toward z with weight 1 - 1/beta1 — but z is
    never created in the reference, so this only flips train_mode. The
    params arg is kept in the signature so a future real-z implementation
    is a drop-in."""
    _ = params  # no z state in the reference -> weights unchanged
    ctl.train_mode = False


def adamw_schedulefree_train(
    mut ctl: AdamWScheduleFreeCtl,
    mut params: List[List[Float32]],
):
    """Switch back to TRAIN weights — call before resuming the train loop
    after a save/validation. Reference (__init__.py:67-77): lerp p toward z
    with weight 1 - beta1; no-op in the reference (no z), flips train_mode."""
    _ = params
    ctl.train_mode = True
