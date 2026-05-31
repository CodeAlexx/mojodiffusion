# SKEPTIC — Training Autograd Port, Round 3 (composed-chain math review)

**Reviewer stance:** assume the port is WRONG until the math forces otherwise.
**Method:** re-derive each composed backward by hand, compare to the Mojo gate
*and* the torch oracle, line by line, citing `file:line`. No verdict without
having read the actual code.

**Date:** 2026-05-30
**Files reviewed (all READ in full this session):**
- `serenitymojo/training/parity/composed_chain_parity.mojo` (223 L)
- `serenitymojo/training/parity/composed_chain_torch_oracle.py` (77 L)
- `serenitymojo/training/parity/checkpoint_parity.mojo` (230 L)
- `serenitymojo/training/parity/checkpoint_oracle.py` (37 L)
- `serenitymojo/training/parity/optim_parity.mojo` (202 L)
- `serenitymojo/training/parity/optim_oracle.py` (120 L)
- `serenitymojo/training/parity/optim_converge_oracle.py` (84 L)
- `serenitymojo/training/parity/schedule_parity.mojo` (203 L)
- `serenitymojo/training/optim.mojo`, `training/checkpoint.mojo`
- `serenitymojo/ops/norm_backward.mojo`, `ops/linalg_backward.mojo`
- flame-core `src/autograd.rs` RMSNorm-backward arm (≈L2122–2185)

**Files the task named but that DID NOT land** (verified by
`ls -t serenitymojo/training/parity/`): `block_composed_parity.mojo`,
`train_skeleton.mojo`, `optim_converge_parity.mojo` (only the *oracle*
`optim_converge_oracle.py` is present — the Mojo driver is missing),
`mixed_precision_parity.mojo`, and the `T5_ZIMAGE_TRAINING_MAP.md` map.
What landed instead: a 3-op `composed_chain_parity`, `checkpoint_parity`,
`optim_parity`, `schedule_parity`. **The lead should treat the requested set
as partially delivered** (see §6).

---

## §1 — composed_chain_parity (linear → rms_norm → mse)

### 1.1 What the gate actually chains
From the Mojo file (L139–170):

```
h      = linear(X, W1)                       # [S,Dm] = X[S,K] @ W1[Dm,K]^T
hn     = rms_norm(h, G)                      # RMSNorm over last dim Dm
dhn    = 2*(hn - target)/numel              # MSE leaf, numel = S*Dm  (L80-85)
(d_h,dg) = rms_norm_backward(dhn, h, G)      # ops/norm_backward.mojo
(dx,dW1) = linear_backward(d_h, X, W1, S,K,Dm)  # ops/linalg_backward.mojo
```

This is **NOT a transformer block.** It is a 3-op linear→norm→mse chain. The
task asked for the full block (`rms_norm→sdpa→residual→rms_norm→swiglu→
residual→mse`) **including the residual grad accumulation** — which is the #1
composition-bug source. **That residual structure is entirely absent here.**
The file header (L36–52) is honest about why: `sdpa_backward` and the
swiglu/mse symbols in `loss_swiglu_backward.mojo` are reported unimportable
under Mojo 1.0.0b1, so the chain was cut to the two ops that import.

**Skeptic verdict on scope:** this gate does **not** exercise the risk the task
is about. There is no fork/join, so no grad-summation path. A green result here
is necessary but *nowhere near sufficient* to retire the klein-class composition
risk. See §1.4.

### 1.2 Hand-derivation of the 3-op chain (my independent math)

Let `L = (1/N) Σ (hn - t)²`, `N = S·Dm`.
- `dL/dhn = 2(hn - t)/N`. ✅ matches `mse_grad` (L80–85): `2.0*(pred-tgt)/n`,
  `n = len(pred_h) = S·Dm`. Correct — divisor is full numel, matching torch
  `.mean()` over all elements (oracle L59).
- RMSNorm: `hn = h·inv·g`, `inv = 1/sqrt(mean(h²)+eps)` over last dim Dm.
  `dL/dh_c = g_c·go_c·inv − h_c·inv³·(1/Dm)·Σ_j(go_j·g_j·h_j)`.
  This is the textbook RMSNorm input grad. ✅
- linear `h = X @ W1^T`: `dX = d_h @ W1`, `dW1 = d_h^T @ X`. ✅

### 1.3 Mojo-kernel vs my derivation vs flame-core

**rms_norm_backward dx** (`ops/norm_backward.mojo` L82, L102, L111):
```
inv      = 1/sqrt(sh[0]/cols + eps)          # mean(x^2)+eps   ✅
sum_gwx  = Σ_j go_j·g_j·x_j                   # ✅
out      = g·go·inv − x·inv³·(sum_gwx/cols)   # L111
```
Compare flame-core `autograd.rs` (≈L2133):
```
dx = (gout*weight)*inv_rms − x*inv_rms^3*(1/N)*sum(gout*weight*x)
```
**MATCH** — identical term-for-term, including the `1/N` (=`1/cols`=`1/Dm`)
and the cubing of `inv`. No sign error, no missing factor.

**rms_norm_backward dg** (L116–138): `dg_c = Σ_rows go·x·inv_rms[row]`,
recomputing `inv` per row. ✅ matches flame-core `dW = Σ_batch (gout·x·inv_rms)`
(autograd.rs L2158 comment). The per-row recompute is the same discipline the
flame-core comment cites (`flame_norm_bf16.cu`). ✅

**linear_backward** (`ops/linalg_backward.mojo` L270–273):
```
dx[M,in] = grad_y[M,out] @ W[out,in]          (no transpose)   L271
dW[out,in] = grad_y[M,out]^T @ x[M,in]        (transpose_a)    L273
```
My derivation for `y = x @ W^T`: `dx = go @ W`, `dW = go^T @ x`. **MATCH.**
Note the forward convention is PyTorch `nn.Linear` (`y = x @ W^T`,
W:[out,in]), so `dx = go @ W` (contraction over `out`, no transpose) is
correct — a transpose here would be the classic bug, and it is **not** present.

**The cross-op handoff** (the one thing this gate *does* test): `d_h` =
`nb.d_x.to_host()` (L161) is fed as `grad_y` into `linear_backward` (L163–168)
with `M=S, in=K, out=Dm`. Dimensions line up (`d_h` is [S,Dm], out_features=Dm).
✅ Handoff order correct, shapes correct.

### 1.4 Is the oracle the IDENTICAL block? (false-pass check)

torch oracle (`composed_chain_torch_oracle.py`):
- `linear: a @ w.t()` (L49–50) ✅ same as Mojo `y = x @ W^T`.
- `rms_norm: ss=mean(h*h,-1); inv=rsqrt(ss+eps); h*inv*gain` (L52–55) ✅
  same mean-of-squares + eps + gain.
- `L = ((h_norm-target)**2).mean()` (L59) ✅ full-numel mean = Mojo `/Float32(n)`.
- f64 throughout (L34). The Mojo side runs F32. Cross-precision is fine for a
  cos≥0.999 gate. ✅

**Oracle is faithful.** No false-pass risk *for the 3 ops it covers.*

**BUT:** the embedded `REF_DX`/`REF_DW1` (L128–129) are static constants. There
is no in-test guarantee they were regenerated from the *current* oracle. The
header (L54) says "GENERATED … regenerate via the generator". **Risk:** if the
forward kernels change and the constants aren't re-emitted, the gate silently
checks against stale truth. The finite-diff oracle (oracle (b), L172–191)
**mitigates** this — it is computed live from the Mojo forward, so a forward
change is caught by the `cos(chained, fd)≥0.99` arm even if REF_DX is stale.
Good defensive design. The `cos(fd, torch)` cross-check (L200–201) further
flags stale constants (fd-vs-torch would drop). ✅ This is the strongest part
of the gate.

### 1.5 composed_chain VERDICT

| Item | Verdict |
|---|---|
| 3-op chain math (mse/rms/linear) | **MATCH** (derived = Mojo = flame-core) |
| Cross-op handoff (norm→linear) | **MATCH** (shapes + order correct) |
| Oracle fidelity (3 ops) | **MATCH** (torch identical; fd live cross-check) |
| **Residual grad accumulation** | **NOT TESTED** — no fork/join in chain |
| **SDPA in the chain** | **NOT TESTED** — `sdpa_backward` excluded |
| **swiglu/silu MLP in chain** | **NOT TESTED** — module export bug excluded it |

**Bottom line:** what is here is correct, but it does **not** discharge the
task. The klein-class bug (memory `project_klein_runaway_composition_backward`)
was specifically a *composed* backward going wrong across many blocks WITH
residual streams. A chain with zero residual joins cannot reproduce it. The
"COMPOSITION SOUND" verdict string (L217) **overclaims** — it proves only the
linear↔rms_norm handoff. **Recommend: rename the pass string, and treat the
real composed-block gate as still OPEN.**

---

## §2 — The residual-accumulation derivation the task asked for (not yet in code)

Since the file omits it, here is the derivation the future block gate MUST
satisfy, so the lead can check the next drop against it. For
`rms→sdpa→(+res1)→rms→swiglu→(+res2)→mse`, with `x` the block input:

```
n1   = rmsnorm(x)
a    = sdpa(n1)
r1   = x + a                       # FORK: x feeds both this add AND nothing else here
n2   = rmsnorm(r1)
m    = swiglu_mlp(n2)
r2   = r1 + m                      # FORK: r1 feeds both the add AND n2
L    = mse(r2, target)
```

Reverse mode (the grad of a tensor that fans out is the **SUM** over consumers):

```
dr2  = dL/dr2 = 2(r2-t)/N
dm   = dr2                          # r2 = r1 + m  ⇒ ∂r2/∂m = 1
dr1  = dr2                          # ... AND the residual edge r2=r1+m  ⇒ +dr2
dn2  = swiglu_bwd(dm)
dr1 += rmsnorm_bwd_x(dn2)           # **ACCUMULATE**: r1 feeds n2 AND the r2 add
da   = dr1
dx   = dr1                          # r1 = x + a  ⇒ residual edge gives dx += dr1
dn1  = sdpa_bwd(da)
dx  += rmsnorm_bwd_x(dn1)           # **ACCUMULATE**: x feeds n1 AND the r1 add
```

The two `+=` lines are exactly where a hand-threaded (tapeless) port drops a
term and produces the "every block clean, composition wrong" signature. **The
landed gate has none of these joins, so it cannot have caught such a bug.** Any
future `block_composed_parity.mojo` MUST show these two accumulations or it is
not testing the stated risk.

---

## §3 — checkpoint_parity (y = silu(x @ W^T))

### 3.1 Chain & three-way gate
`checkpoint.mojo`: forward `pre = x@W^T; y = silu(pre)` (L142–144). Backward:
`d_pre = silu_backward(grad_out, pre); (dx,dW) = linear_backward(d_pre,x,w)`
(saveall L163–165; recompute L196–198). Gate compares A=save-all,
B=checkpoint-recompute, C=torch (L182–209).

### 3.2 Hand-derivation
`loss = Σ(y·gout)` (oracle L26). `dy = gout`. `d_pre = gout·silu'(pre)`.
`dx = d_pre @ W`, `dW = d_pre^T @ x`. The Mojo recompute path restores `x` from
host bytes (L195), **recomputes `pre`** (L196 — does not save it), then runs the
identical `silu_backward`+`linear_backward`. ✅ Mathematically the recompute and
save-all paths are the *same ops on the same inputs*, so cos≥0.9999 is the
right (tight) self-gate (L221).

### 3.3 Oracle fidelity & a real concern
torch (`checkpoint_oracle.py` L25–27): `silu(x@W.t())`, `loss=(y*gout).sum()`.
✅ Identical block, identical loss. `grad_out` fed to the Mojo backward is
`gout` itself (L175 builds `gout` = `_gout_host`), and the torch loss
`Σ(y·gout)` makes `dL/dy = gout`. ✅ **The upstream grad matches the oracle's
implied dL/dy** — this is the subtle thing that is easy to get wrong (if the
Mojo side passed `1` as grad_out while torch used `Σ(y·gout)`, the gate would
be comparing different quantities). Here they agree. **MATCH.**

**Self-consistency caveat:** A==B (cos≥0.9999) only proves the offload
round-trips bytes and the recompute is deterministic — it does NOT independently
prove the backward math (both paths run the *same* `silu_backward`/
`linear_backward`). The math is anchored by the torch arm C (cos≥0.999) and the
byte-exact round-trip (`max_abs==0`, L223). The three arms together are
sufficient. ✅

### 3.4 checkpoint VERDICT: **MATCH.** Math correct, oracle faithful, grad_out
convention consistent with the oracle loss. The honest header (L19–35) flags
that the *general* closure-based flame-core checkpoint cannot be ported (Mojo
has no storable `dyn Fn`) and this is a CONCRETE single-block substitute — true,
and a real limitation the lead must know: **arbitrary checkpointed blocks are
not yet covered**, only `linear→silu`.

---

## §4 — optim_parity + optim.mojo (AdamW / SGD / clip)

### 4.1 AdamW — hand-check vs torch.optim.AdamW
Mojo `_adamw_kernel` (`optim.mojo` L54–70):
```
m = b1*m + (1-b1)*g
v = b2*v + (1-b2)*g*g
mhat = m/(1-b1^t)          # bias_c1 = 1-beta1**t  (L84)
vhat = v/(1-b2^t)          # bias_c2 = 1-beta2**t  (L85)
p = p - lr*wd*p            # decoupled WD, applied BEFORE adam term  (L67)
p = p - lr*mhat/(sqrt(vhat)+eps)                                    (L68)
```
torch.optim.AdamW decoupled update:
`p ← p − lr·wd·p ;  p ← p − lr·m̂/(√v̂ + eps)`. **MATCH**, including:
- **decoupled** WD (not Adam+L2) — the load-bearing distinction the oracle
  deliberately tests with `WD=0.01>0` (oracle L40). ✅
- `eps` **outside** the sqrt: `sqrt(vhat)+eps` (L68) = torch
  `√v̂ + eps`. ✅ (A common bug is `sqrt(vhat+eps)`; not present.)
- bias correction uses **per-step `t`** (1-based, L123 loop `range(1,steps+1)`
  → `bias_c1=1-b1^t`). ✅ The 3-step gate (`adamw_p3`) specifically exercises
  `t=1,2,3`, so a constant-`t` bug would fail it.

**Subtle ordering check:** torch applies decoupled WD as `p *= (1 - lr·wd)`,
i.e. `p ← p − lr·wd·p`, then the adam step. Mojo does exactly this order
(L67 then L68). ✅ MATCH.

### 4.2 SGD
Mojo (L120–127): `buf = momentum*buf + g; p = p - lr*wd*p - lr*buf`. torch SGD
(`momentum`, `dampening=0`): `buf = momentum*buf + g; p -= lr*buf`. With the
oracle's `SGD_WD=0` (oracle L43–44) the `lr*wd*p` term vanishes, so **MATCH at
wd=0**. The code comment (optim.mojo L17) and oracle comment both flag that
torch SGD WD is *coupled* L2 and only agrees with this decoupled form at wd=0.
**Honest and correct** — but note: **the wd≠0 SGD path is therefore UNTESTED**
and would NOT match torch. If any trainer uses SGD with wd>0, this is a latent
divergence. Flag for the lead.

### 4.3 Grad clip
Mojo `clip_grad_global_norm` (L162–219): `total=sqrt(Σg²)` over both tensors
(host reduction L167–177); `if total>max: scale=max/(total+1e-6)`; scale both
in place; **return `total` (the UNclipped norm).** torch
`clip_grad_norm_` returns the **total norm before clipping** and scales by
`max/(total+1e-6)`. **MATCH** — both the returned scalar (pre-clip norm) and
the `+1e-6` denominator epsilon match torch exactly (oracle L99). ✅
Dead variable `clipscale` (L181) is cosmetic, not a bug.

### 4.4 optim VERDICT: **MATCH** for AdamW (1 & 3 step), SGD (wd=0), clip.
**Gap:** SGD wd>0 is not torch-equivalent by construction and untested.

### 4.5 Multi-step convergence (the requested `optim_converge_parity`)
Only `optim_converge_oracle.py` landed; **no Mojo driver.** The oracle inlines
manual AdamW on `f(p)=Σ(p−target)²`, `g=2(p−target)` (L63–70), 500 steps,
lr=1e-2, wd=0. Its update (L65–70) is identical to the kernel's formula in §4.1.
**The oracle math is self-consistent and correct.** But with **no Mojo gate
present, there is nothing under test** — the convergence claim is unverified.
This is the multi-step counterpart that the klein bug actually needed (the
1-step parity passed historically while multi-step diverged). **Its absence is
the most important gap in this drop.** (See memory
`project_all_trainers_broken_2026-05-24`: "bug is MULTI-STEP".)

---

## §5 — schedule_parity (T6)

Re-read of `schedule.mojo` internals was blocked by intermittent tool-output
failures this session (the file READ cleanly once via the parity driver's
references but I could not re-confirm every kernel line). **What I CAN verify
from the gate driver** (`schedule_parity.mojo`, fully read):

- `flow_match_noise_target`: gate's host oracle (L83–85) is
  `x_t=(1−σ)·latent+σ·noise`, `target=noise−latent`. This is the standard
  rectified-flow / flow-matching forward with `σ`=t-as-noise-weight. ✅ This
  matches the project's flow-match convention (memory
  `reference_sensenova_u1_training`: `z=t·x+(1−t)·noise`, target = velocity).
  **One thing to verify in `schedule.mojo` once tooling recovers:** that the
  kernel computes `target = noise − latent` and NOT `latent − noise` (sign).
  The gate would catch a sign flip (cos would go to −1), so it is gated. ✅
- `ema_update`: oracle `decay·shadow+(1−decay)·live` (L110–112) + a hand-check
  (decay .999, 1→2 ⇒ 1.001, L119–127). ✅ standard EMA.
- `grad_accumulate`: `acc += new` (L139–142). ✅ trivial, gated.
- `sample_timestep_logit_normal`: **statistical** gate (L148–201): mean≈0.5,
  std∈[0.18,0.24], clamp [1/1000,1], and shift=3 pushes mean up. This is a
  distribution-shape gate, not a cos gate — appropriate for an RNG primitive,
  and the symmetry argument (mean=0.5 by `sigmoid(−z)=1−sigmoid(z)`, L155) is
  mathematically sound. The shift-monotonicity check (L192–201) is a good
  second constraint. ✅

**schedule VERDICT:** gate logic is mathematically sound and the oracles are
the host F64 computation itself (legitimate for closed-form numeric policy).
**Caveat (honesty):** I could not re-read every line of `schedule.mojo`'s
kernels this session due to tool flakiness, so I am gating my verdict to "gate
DESIGN is correct; kernel-internal sign/factor of `flow_match`/`ema` should be
spot-confirmed against `schedule.mojo` on the next pass." The cos gates would
catch a sign or factor error, so residual risk is low. **PROBABLE MATCH,
one re-read pending.**

---

## §6 — Z-Image training MAP

`T5_ZIMAGE_TRAINING_MAP.md` **did not land** (verified: not in repo root, not
under `serenitymojo/`, not in `training/parity/`). The "HAVE backward / MISSING"
audit the task asked me to spot-check **cannot be performed** — there is no map.

What I *can* report from reading the actual kernels, as raw inputs for whoever
writes that map:
- **HAVE (real, gated):** `rms_norm_backward`, `layer_norm_backward`,
  `group_norm_backward` (`ops/norm_backward.mojo`); `mm_backward`,
  `bmm_backward`, `linear_backward`, `addbias_backward`
  (`ops/linalg_backward.mojo`); `silu_backward` (used by checkpoint.mojo L48,
  L164); AdamW/SGD/clip (`training/optim.mojo`); flow-match/ema/grad-accum/
  timestep (`training/schedule.mojo`).
- **MISSING / BROKEN (per the composed_chain header L36–50, which I take as a
  finding to verify, not gospel):**
  - `sdpa_backward` exists but `attention_backward` (the name) does not; SDPA
    backward is **NOT exercised in any composed gate**. Single-op SDPA bwd
    parity should be confirmed separately (a `ops/parity/sdpa_*` file exists in
    the tree — not reviewed here).
  - `loss_swiglu_backward.mojo`: `mse_backward`, `mse_loss_backward`, `silu`,
    `mse_loss`, `swiglu` reported **unimportable** (parse-recovery drops every
    symbol defined before `swiglu_backward`). **This is a claimed PRE-EXISTING
    source bug.** If true, the swiglu MLP backward is effectively unusable in
    composed tests, which is why no real block gate exists. **The lead should
    verify this import bug directly** — it is the linchpin blocking the actual
    block composition gate.

**Z-Image map VERDICT:** cannot audit (artifact absent). The named drivers
(`block_composed_parity`, `train_skeleton`, `optim_converge_parity`,
`mixed_precision_parity`) are also absent. **This round delivered the
*foundational* per-op-handoff and optimizer gates, not the composed-block /
train-skeleton / mixed-precision gates the task targeted.**

---

## §7 — Overall verdict

| Chain | derived = Mojo? | oracle faithful? | residual-accum? | net |
|---|---|---|---|---|
| composed_chain (lin→rms→mse) | YES | YES (+live fd) | N/A (no join) | **MATCH but under-scoped** |
| checkpoint (lin→silu) | YES | YES | N/A | **MATCH** |
| AdamW 1/3-step | YES | YES | — | **MATCH** |
| SGD (wd=0) | YES | YES | — | **MATCH**; wd>0 untested |
| grad clip | YES | YES | — | **MATCH** |
| schedule (flow/ema/accum/t) | gate-sound | host-F64 oracle | — | **PROBABLE MATCH** (1 re-read pending) |
| **composed transformer BLOCK** | — | — | — | **NOT DELIVERED** |
| **train_skeleton (mse→lin→silu→lin)** | — | — | — | **NOT DELIVERED** |
| **optim multi-step convergence** | oracle-only | — | — | **NOT GATED** (no Mojo driver) |
| **mixed_precision** | — | — | — | **NOT DELIVERED** |
| **Z-Image training MAP** | — | — | — | **ABSENT** |

**No mathematical defect found in what landed.** Every backward formula I
re-derived matches both the Mojo kernel and the flame-core reference (RMSNorm dx
term-for-term incl. `inv³` and `1/N`; linear `dx=go@W`, `dW=go^T@x` with no
spurious transpose; AdamW decoupled-WD with `eps` outside the sqrt and per-step
bias correction; clip returns pre-clip norm with `+1e-6`). The oracles are the
identical blocks (no false-pass detected), and the `grad_out`/loss conventions
are consistent between Mojo and torch in every gate.

**The real finding is a coverage gap, not a sign error.** The single highest
risk — composed backward with **residual grad accumulation** across a real
transformer block, and **multi-step** optimizer convergence — is precisely
what the klein-class bug was, and **neither has a runnable Mojo gate in this
drop.** The `composed_chain` gate has *no fork/join*, so its "COMPOSITION
SOUND" string overstates; and `optim_converge` shipped an oracle with no driver.

### Action items for the lead
1. **Do not retire the klein-class composition risk** on the strength of
   `composed_chain_parity`. It tests one cross-op handoff, not residual joins.
2. **Verify the `loss_swiglu_backward.mojo` import bug** (header L45–50). It is
   the blocker for a real block gate. If real, fix the parse-recovery /
   symbol-ordering issue so swiglu+mse backward become importable.
3. **Land `optim_converge_parity.mojo`** to actually exercise the multi-step
   trajectory the oracle already describes (the historically failing axis).
4. **Add a residual-join gate** that satisfies the two `+=` accumulations in §2.
5. Rename `composed_chain`'s pass string from "COMPOSITION SOUND" to something
   scoped, e.g. "linear↔rms_norm handoff sound".
6. SGD wd>0 is not torch-equivalent by construction — document or gate it.
7. Produce `T5_ZIMAGE_TRAINING_MAP.md`; the HAVE/MISSING raw inventory in §6 is
   a starting point.

**Honesty note:** `schedule.mojo` kernel internals and the byte-level
`adam.rs` reference were not fully re-readable this session due to intermittent
empty tool output (the same files read cleanly earlier). The AdamW math was
confirmed from `optim.mojo` (read in full) + the oracle; the schedule verdict is
gated to "probable, one re-read pending." Nothing in this report rests on an
unread file — where I could not read, I said so.
