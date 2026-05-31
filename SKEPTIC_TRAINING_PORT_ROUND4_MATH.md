# SKEPTIC — Training Autograd Port, Round 4 (Multi-Block Stack + Checkpoint Recompute)

Reviewer stance: assume WRONG until the actual code proves otherwise. Every
verdict is backed by reading the real file bytes. Where the Read tool produced
garbled output (jumbled line numbers, spurious `...`/`o.element_type`), I
re-read via `python3` byte-level dump and used that as authoritative.

Date: 2026-05-30.

---

## 0. Reference-path corrections (VERIFIED) — the brief's cites are stale

| Brief said | Reality |
|---|---|
| `flame-core/src/training/training_offload.rs` | **Does not exist.** No `flame-core/src/training/` dir. |
| `autograd.rs ~3208` = `checkpoint_offload_boundary` / `backward_checkpoint_offload_boundary` (the Mojo port source) | `checkpoint_offload_boundary` DOES exist at `autograd.rs:3208`, and there's `Op::Checkpoint` (3082), `Op::CheckpointOffloadBoundary` (3344), `Op::CheckpointOffload` (3520), with backward arms at 4115 / 4390 / 4568. The cleanest contract statement is the **v2** impl: `autograd_v2/checkpoint.rs` (`checkpoint_v2`, `CheckpointGradFn::apply`). I used both as oracle. |
| `compute_gradients` composes block grads | **Confirmed**: `autograd.rs:3561`, reverse-walk callers at 2184/2764. |

The Mojo `checkpoint.mojo` doc header cites `training_offload.rs` (nonexistent)
and points at the v1 boundary fn. The **actual contract it implements** matches
both v1 (`Op::Checkpoint` recompute-from-saved-input) and v2
(`CheckpointGradFn`: re-run forward closure on saved inputs, nested backward).
Stale doc cite, not a math bug. (Fix the comment.)

## 1. Mojo file inventory (VERIFIED) — brief names vs reality

| Brief name | Actual |
|---|---|
| `stack_train_parity.mojo` | absent → `training/parity/block_composed_parity.mojo` (full DiT block) + `training/parity/composed_chain_parity.mojo` (3-op) |
| `dit_block.mojo` | absent; block fwd/bwd **inlined, hand-chained** in `block_composed_parity.mojo` (NO tape) |
| `checkpoint_block.mojo` / `_parity` | `training/checkpoint.mojo` + `training/parity/checkpoint_parity.mojo` |
| `loop.mojo` | absent → `training/parity/train_skeleton.mojo`, `training/zimage_train_step.mojo` |
| `autograd.mojo` new tape ops | **`serenitymojo/autograd.mojo`** (repo ROOT, not `training/`) |

---

## 2. CRITICAL STRUCTURAL FINDING (the brief's central assumption is FALSE)

The brief assumes the multi-block stack backward AND the new ops (rms/silu/
swiglu/mse) are wired **through the tape engine** (`autograd.mojo`). They are NOT.

- **Tape engine** `serenitymojo/autograd.mojo` has arms for ONLY
  `Add / Sub / Mul / MatMul / Linear` (OP_ADD..OP_LINEAR, lines 42–46;
  record_* 277–337; backward arms 369–404). **No rms_norm, silu, swiglu, sdpa,
  or mse tape arm exists.** A real DiT block therefore CANNOT be backpropagated
  by the tape today.
- The "stack/block" backward that exists is **hand-chained** in
  `block_composed_parity.mojo` — every grad handoff and residual sum written by
  hand with `add_lists`. No tape.

⇒ I derived against the hand-chained form (what exists). The tape-composed
N-block stack the brief describes is **unbuilt**.

---

## 3. ITEM 1 — MULTI-BLOCK (hand-chained block) BACKWARD

### 3a. Independent hand-derivation vs Mojo

Forward (block_composed_parity.mojo:163–184; identical to torch oracle):
`h1=rms(x,g1); q,k,v=lin(h1,·); attn=sdpa(q,k,v); ao=lin(attn,Wo); r1=x+ao;
h2=rms(r1,g2); gate=lin(h2,Wg); up=lin(h2,Wu); act=silu(gate)·up;
mlp=lin(act,Wd); y=r1+mlp; L=mean((y-t)²)`.

The 3 multi-path sums + 2 residual accumulations (the make-or-break steps):

| Node | Required reverse op | Mojo line | Verdict |
|---|---|---|---|
| y=r1+mlp split | d_r1←dy ; d_mlp←dy | 243–244 | **MATCH** |
| h2 → gate & up | d_h2 = d_h2_g + d_h2_u | 282 | **MATCH** |
| residual#1 r1 → y & rms(h2) | d_r1 += d_r1_norm | 294 | **MATCH** |
| h1 → q,k,v | d_h1 = d_h1_q+d_h1_k+d_h1_v | 343 | **MATCH** |
| residual#1 x → y(via r1) & rms(h1) | d_x = d_x_norm + d_r1 | 355 | **MATCH** |

Inter-block grad handoff (d_x of a sub-block becomes d_y of the next): the grad
out of the q/k/v linears (`d_h1`) is fed straight into `rms_norm_backward`
(346) — no swap, no drop. **No dropped/swapped handoff.**

Ordering check: `d_r1` is `+= d_r1_norm` at 294 BEFORE being consumed by
`linear_backward(d_r1, attn, Wo)` at 297. At 297 it correctly already holds
(residual-from-y)+(norm-path) = full grad of r1. ✓ Not a use-before-accumulate.

### 3b. vs flame-core composition

flame-core `compute_gradients` + GradientMap sum multiple `(id,grad)`
contributions per shared leaf (Add arm 3636; broadcast-reduce). Mojo `_accum`
(autograd.mojo:341–351) does the same (`if contains: old+g`); the hand-chain
replicates it with `add_lists`. **Composition model MATCHES flame-core.**

### 3c. Oracle correctness (no false pass)

`block_composed_torch_oracle.py` (read in full, 552 lines incl. the generated
template) builds the **identical** block: same rms `t/sqrt(mean(t²)+eps)·g`
(76–78), same `t@Wᵀ` linear (81–82), per-head `softmax((q@kᵀ)·scale)@v` with
`scale=1/√Dh` (85–87, 96–99), `silu(gate)·up` (105), two residuals (101,107),
`mean((y-t)²)` (108). torch grads via `autograd.grad` (110).

- **Head-layout (the klein txt_ids trap):** oracle header 27–30 proves
  row-major `[M,H·Dh]` col `(h·Dh+d)` == BSHD `[1,M,H,Dh]` slot `(s,h,d)`; torch
  slices `q[:, h*Dh:(h+1)*Dh]` (98) match the Mojo single `sdpa_nomask[1,M,H,Dh]`
  view. **Identical layout — the klein parity-oracle bug class is absent.**
- **Second oracle:** the gate also runs in-Mojo central finite-diff on its OWN
  forward (357–376) + cross-checks fd-vs-torch (389). Two independent oracles.

**VERDICT ITEM 1 (hand-chained block): MATCH** — derivation, Mojo, flame-core
agree; oracle correct and double-checked.

**CAVEATS (carry to handoff):**
1. This is ONE hand-chained block at M=4,D=8 — NOT a tape-composed N-block
   stack. The klein composition bug
   (`project_klein_runaway_composition_backward`) is retired only for this case.
2. A true multi-block tape stack does not exist (no rms/silu/swiglu/sdpa tape
   arms). The lead must not read this gate as "stack backward proven."

---

## 4. ITEM 2 — CHECKPOINT RECOMPUTE (`training/checkpoint.mojo`, full read)

### 4a. Recompute == save-all forward, op-for-op
- `block_forward` (135–144): `pre=linear(x,W); y=silu(pre)`.
- save-all bwd (147–171): `pre=linear(x,W); d_pre=silu_backward(go,pre); lg=linear_backward(d_pre,x,W)`.
- checkpoint bwd (174–202): `x=restore(...); pre=linear(x,W); d_pre=silu_backward(go,pre); lg=linear_backward(d_pre,x,W)`.

Recompute (196–198) = same 3 ops, same order, as save-all bwd (163–165).
`linear` is deterministic + `x` byte-restored ⇒ recomputed `pre` bit-identical.
**MATCH — recompute forward = original forward.**

### 4b. Only the input is saved; restore is bit-exact
- `offload_to_host` (79–96): copies **raw storage uint8 bytes** (NOT `to_host`,
  which widens bf16→f32), keeps shape+dtype. Only the input is saved.
- `restore_to_device` (99–113): H2D byte copy → byte-identical Tensor. **Bit-
  exact both legs.** Weight W is not checkpointed (resident).

Matches flame-core: v2 `CheckpointGradFn` holds only `saved_inputs`
(checkpoint.rs:81), re-runs forward closure on them; v1 `Op::Checkpoint` saves
`input` + recompute_fn. **MATCH for the concrete block.**

**VERDICT ITEM 2: MATCH.**
**CAVEAT:** only ONE hard-coded block kind (`linear→silu`) is checkpointable.
flame-core checkpoints ARBITRARY closures (v1 recompute_fn / v2 forward_fn).
Mojo 1.0.0b1 can't store closures (documented checkpoint.mojo:18–35), so each
new checkpointed block kind needs a hand-written recompute (or an Op-tag enum).
Coverage limit, not a correctness bug.

---

## 5. ITEM 3 — TAPE OPS + backward kernels

### 5a. Tape arms in `autograd.mojo` — operand save + grad-id routing
| Arm | save | backward | routing | Verdict |
|---|---|---|---|---|
| ADD (369–371) | — | d_lhs=g, d_rhs=g | elhs, erhs | **MATCH** (cf flame Add 3636) |
| SUB (372–374) | — | d_lhs=g, d_rhs=−g | elhs, erhs | **MATCH** (flame Sub 3658) |
| MUL (375–379) | saved0=a,saved1=b | d_lhs=g·b, d_rhs=g·a | elhs←g·saved1, erhs←g·saved0 | **MATCH** (flame Mul 3664) |
| MATMUL (380–392) | a,b + M,N,K | mm_backward(g,a,b,M,N,K) → d_a=g@Bᵀ, d_b=Aᵀ@g | d_a→elhs, d_b→erhs | **MATCH** |
| LINEAR (393–404) | x,W,b + M,out,in | see below | x→elhs, W→erhs, b→third_id | **MATCH** |

**LINEAR dim-routing scrutiny (the likeliest mis-route):**
- record (318–337): dim_m=M, **dim_n=out_f, dim_k=in_f**; third_id=b.id.
- backward (398–401): `linear_backward(g, x, W, dim_m, dim_k, dim_n)`.
- `linear_backward` sig (linalg_backward.mojo:249): `(grad_y, x, weight, M,
  in_features, out_features)`. So it receives in_features=dim_k=in_f,
  out_features=dim_n=out_f. **Dims un-swapped correctly. ✓**
- grad ids: d_x→elhs(x.id), d_w→erhs(W.id), d_b→third_id(b.id). **All correct. ✓**
- `linear_backward` math (282–286): d_x=g@W (no transpose, contract over out),
  d_W=gᵀ@x, d_b=colsum(g) — matches its PyTorch-conv header (W:[out,in]). ✓

**No mis-routed tape grad id found.**

### 5b. rms_norm / silu / swiglu / mse as TAPE ops → **N/A (not tape-wired)**
Verified against autograd.mojo:42–46. The brief's item-3 targets don't exist in
the tape. They exist only as standalone kernels (next).

### 5c. Standalone backward kernels (used by the hand-chained gates) — math
- **silu_backward** (activation_backward.mojo:62–65, 102–105, 270): d_x = g·s·(1+x(1−s)),
  s=σ(x); takes pre-activation x. **MATCH** (= go·silu'(x)).
- **swiglu_backward** (loss_swiglu_backward.mojo:199–206, 209): sig=σ(gate),
  silu_x=gate·sig, dsilu=sig+gate·sig(1−sig); **d_up=g·silu_x**, **d_gate=g·dsilu·up**.
  Correct product rule for silu(gate)·up. Routing in struct `SwigluGrads{d_gate,d_up}`
  consumed correctly by the gate (d_gate→h2 via Wg, d_up→h2 via Wu). **MATCH.**
- **rms_norm_backward** (norm_backward.mojo:105–112, 138): d_x_i = g_i·go_i·inv −
  x_i·inv³·(Σ go·g·x)/D ; d_g_i = Σ_rows go·x·inv. Standard RMSNorm grad,
  matches differentiated oracle rms. **MATCH.**
- **mse_backward** (loss_swiglu_backward.mojo:66, 90): d = (p−t)·2/N. **MATCH**
  `mean((p−t)²)`. (The gates inline the SAME `2(p−t)/N` leaf; the importable
  `mse_backward` kernel agrees. The "mse_backward unimportable" symptom in
  `composed_chain_parity` header is the known transient; inline leaf is
  identical math, gate not weakened.)
- **sdpa_backward** (attention_backward.mojo:406–424, recompute-based): scores=
  scale·q@kᵀ → softmax → d_v=probsᵀ@dout, d_probs=dout@vᵀ,
  d_scores=softmax_bwd(probs,d_probs) then scale folded, d_q=d_scores@k,
  d_k=d_scoresᵀ@q. Standard; scale-fold-after-softmax_bwd is correct
  (scores=scale·q@kᵀ ⇒ d(q@kᵀ)=scale·d_scores). Operand order in gate call
  `sdpa_backward[1,M,H,Dh](q,kk,vv,d_attn,SCALE)` matches sig `(q,k,v,d_out,scale)`.
  **MATCH.**

---

## 6. ITEM 4 — ORACLES (both gates)
Both gates carry torch f64 autograd AND an in-Mojo central finite-diff on the
same forward, plus a fd-vs-torch cross-check (composed_chain:200;
block_composed:389). The finite-diff perturbs the EXACT kernels the chained
backward differentiates ⇒ self-consistent. **No false-pass risk found.**

---

## 7. SUMMARY

| Item | Verdict | Cite |
|---|---|---|
| 1. Hand-chained block backward (residual sums + branch sums + handoffs) | **MATCH** | block_composed_parity.mojo:243,282,294,343,355 |
| 1b. True N-block TAPE stack | **DOES NOT EXIST** (no rms/silu/swiglu/sdpa tape arms) | autograd.mojo:42–46 |
| 2. Checkpoint: recompute==forward, only-input saved, bit-exact restore | **MATCH** | checkpoint.mojo:135–202; autograd_v2/checkpoint.rs:79–322 |
| 3a. Tape arms Add/Sub/Mul/MatMul/Linear: save + id routing + dims | **MATCH** | autograd.mojo:277–404; linalg_backward.mojo:153,249 |
| 3b. rms/silu/swiglu/mse as TAPE ops | **N/A — not tape-wired** | autograd.mojo:42–46 |
| 3c. standalone bwd kernels (silu/swiglu/rms/mse/sdpa) math + routing | **MATCH** | activation_backward:62; loss_swiglu_backward:90,199; norm_backward:105; attention_backward:406 |
| 4. Oracles compute identical thing (both gates) | **MATCH (no false pass)** | composed_chain:200; block_composed:389 |

### No math defect found in what is actually implemented.
No dropped/swapped inter-block handoff. No checkpoint recompute mismatch. No
mis-routed tape grad id. No wrong oracle.

### Things the lead must NOT over-read (must go in the handoff)
1. **The "stack" gate is ONE hand-chained block at M=4,D=8 — not a
   tape-composed N-block stack.** klein composition bug retired only for this
   case. A real multi-block tape stack is unbuilt.
2. **checkpoint.mojo = ONE hard-coded block kind (linear→silu).** flame-core
   checkpoints arbitrary closures; Mojo can't (1.0.0b1 closure-storage limit).
3. **Stale doc cites** in checkpoint.mojo (`training_offload.rs`,
   boundary-fn names) — the real contract is matched anyway. Fix comments.
4. **Read/cat tool corrupted** `loss_swiglu_backward.mojo`,
   `linalg_backward.mojo`, `attention_backward.mojo` displays this session
   (jumbled line numbers, spurious `...`). On-disk content is correct (verified
   via `python3` byte dump). Re-check those three via python, not Read, if
   re-auditing.

### Compilation
I did not run `mojo` to compile-check — that is the lead's tiebreaker. All read
files are syntactically complete on disk. `mse_backward` unimportable is the
documented known transient, not a defect.
