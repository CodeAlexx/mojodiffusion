# Skeptic Findings — Ops Gap Sweep (2026-06-03)

Adversarial review of TARGET A (`ops/rope_tables.mojo::build_multiaxis_rope_tables`
+ probe + oracle) and TARGET B (the load-bearing CLAIMS in
`ops/parity/OPS_GAP_AUDIT_2026-06-03.md` that every Phase 2-4 builder will trust).

Method: re-ran the probe (numerically verified, exit 0), read the new op + the
two existing apply kernels (`ops/rope.mojo`), and cross-checked the audit's
"reduces to existing kernel" / "already present" claims against the ACTUAL Rust
refs (`wan22_dit.rs`, `cosmos_predict25_dit.rs`, `magihuman_dit.rs`) and the
actual Mojo op sources.

---

## TARGET A — build_multiaxis_rope_tables

### Numerical verification (re-run myself)
```
pixi run mojo run -I . serenitymojo/ops/rope_tables_probe.mojo
→ rope_tables_probe max_err: 1.2665987e-07
→ rope_tables_probe PASS   EXIT=0
```
The op is **correct as written**. F32 compute, F32 store (correct call — parity
isolates op math; bf16 enters only at the apply site, matching cosmos's
documented "don't cast at construction" rule). It produces `[rows, half]` where
`half = sum(axes_dims[a]//2)`, axis blocks concatenated at half-dim offsets, with
per-axis `inv_freq = theta^(-i/ha)`. Treats `axes_dims` as FULL per-axis dims
(`ha = da//2`) — internally self-consistent and matches the probe.

### Contract cross-check vs the apply kernels + Rust refs — HOLDS (with caveats)

- **wan22 (interleaved):** Rust `apply_rope` (`wan22_dit.rs:364-426`) uses
  `axes=[44,42,42]` (full dims, head_dim=128), increments `dim_offset += axes[a]`
  (FULL dim), pairs `(re=base+dim_offset+2i, im=+1)` = interleaved within each
  axis block. The builder's global column `off_a+i` (half-dim offsets 0,22,43)
  maps through `rope_interleaved` to data pair `(x[2(off_a+i)], x[2(off_a+i)+1])`.
  Since each axis full-dim = `2*ha`, cumulative full offset `== 2*` cumulative
  half offset, so the kernel's `2*col` lands exactly on Rust's `dim_offset+2i`.
  **Contract matches.** Freq `1/theta^(2i/axis_dim) == theta^(-i/half)` — identical.
- **cosmos (halfsplit):** docstring `cosmos_predict25_dit.rs:26-42` + builder
  `:721-860`: table is `cat([cos(t),cos(h),cos(w)])` width `head_dim/2=64`, fed
  to `rope_halfsplit` which pairs `(d, d+half)` with the SAME angle in both
  halves. Builder's concat `[a0:22,a1:21,a2:21]=64` matches exactly. Freq
  `1/theta^(2k/dim_axis)` matches the builder. **Layout/kernel contract matches.**
- **magihuman:** BOTH rope paths reduce to `rope_halfsplit` fed `[rows, ro_dim/2]`:
  `rope_partial_halfsplit` (`:153`) = slice→halfsplit→cat (audit rank-4 compose,
  ✓), and `apply_rope_to_heads` (`:402`, HF `rotate_half` + `cat([cos,cos])`)
  algebraically equals the halfsplit kernel (`o[i]=x0*c-x1*s`, `o[i+half]=x1*c+x0*s`).
  **Both compose onto the present kernel.** ✓

So the apply-kernel/table-shape contract the audit asserts IS real for the 3
concrete models checked. No interleave-vs-halfsplit or axis-order mismatch in the
*kernel contract*.

---

## BLOCKERS

### BLOCKER 1 — `rope_tables.mojo` docstring gives axes_dims 2× too large for wan22 AND cosmos
- **Where:** `ops/rope_tables.mojo:104` (and the `theta`/usage note `:104-105`):
  `"For wan22 use [88,84,84] (the doubled [44,42,42] half-dims); for cosmos [88,84,84]."`
- **What's wrong:** wan22's `rope_axes = [44,42,42]` are the **FULL per-axis dims**
  (`wan22_dit.rs:189` `axes=[d-4*d6,2*d6,2*d6]`, head_dim=128, sum=128), NOT
  "half-dims". cosmos is the same `(dim_t=44,dim_h=42,dim_w=42)`
  (`cosmos_predict25_dit.rs:20`). The op treats its `axes_dims` arg as FULL dims
  (`ha = da//2`). So the CORRECT call is `axes_dims=[44,42,42]` (→ half=64=Dh/2).
  Following the docstring's `[88,84,84]` yields `half=128`, a `[rows,128]` table.
- **Why it matters:** `rope_interleaved`/`rope_halfsplit` `_rope_common_validate`
  requires `cos.numel() == rows*(D/2) == rows*64`. A `[rows,128]` table → hard
  `"cos numel must equal rows*(D/2)"` raise, OR (if a builder reshapes to dodge
  the check) a silently wrong rotation across the whole DiT. This is the FIRST
  thing a wan22/cosmos porter will copy. The op is right; the guidance is wrong.
- **Minimal fix:** change the docstring example to
  `For wan22/cosmos (head_dim=128) use axes_dims=[44,42,42]` and drop the
  "doubled / half-dims" phrasing. (No code change.)

---

## FRAGILE

### FRAGILE 1 — single scalar `theta` cannot express cosmos per-axis NTK theta
- **Where:** `ops/rope_tables.mojo:96` (`theta: Float32`) vs
  `cosmos_predict25_dit.rs:786-792`.
- **What:** cosmos computes a DIFFERENT theta per axis:
  `axis_theta = 10000 * ratio^(dim/(dim-2))` with per-axis
  `{h,w,t}_extrapolation_ratio`. For V2_2B all ratios=1.0 → uniform theta=10000,
  so the single-scalar builder is exact. But other cosmos configs (e.g. the
  `_2b_image` path, `:232-233`) set ratios to 3.0 (and t up to 24×) → genuinely
  different theta per axis, which this op CANNOT produce.
- **Why it matters:** the audit (§2) only documents zimage-vs-wan22 theta
  equivalence; it does NOT flag cosmos per-axis NTK. A cosmos-extrapolation
  porter using this builder with one theta gets silently wrong h/w/t frequencies
  (cos parity will fail, but only if they test the extrapolation config). Not a
  break for the V2_2B default; a real limitation for the documented variants.
- **Minimal fix:** either accept `theta` as a per-axis `List[Float32]` (len ==
  num_axes), OR add one sentence to the op docstring + audit §2: "single theta
  ⇒ all axes share rope_theta; cosmos NTK extrapolation (per-axis theta) must
  pre-scale positions or call once per axis." Cheapest: document the limitation.

### FRAGILE 2 — audit §0 / builder header undersell that the kernel-contract was only spot-checked, not the *position decomposition*
- **Where:** audit `:25-39`, builder `:26-29`.
- **What:** the builder takes positions as flat `[rows*num_axes]` token-major and
  trusts the CALLER to decompose `si -> (frame,height,width)` (wan22 does
  `fi=si/(h*w); hi=(si%(h*w))/w; wi=si%w`, `wan22_dit.rs:400-403`). That decode
  is NOT in the op and NOT in any present helper. Correct (it's caller policy),
  but a porter could feed the wrong axis order (the op concatenates axes in the
  order given; wan22/cosmos = [t/f, h, w]). Low risk, but undocumented as a
  caller responsibility.
- **Minimal fix:** one line in the op docstring: "axis order in `axes_dims` /
  `positions` must match the ref's `cat` order (wan22/cosmos = [frame,height,width])."

---

## STYLE

- `ops/rope_tables.mojo:23` claims wan22/cosmos use `theta^(-i/axis_half)` — true,
  but pairs with the wrong `[88,84,84]` example below it (BLOCKER 1). Fix together.
- Rust-side note (not our code): `wan22_dit.rs:123` comment calls `[44,42,42]`
  "half-dims per axis" — that Rust comment is itself wrong (they're full dims);
  likely the source of the Mojo docstring error. Worth a one-line correction
  upstream but out of scope here.

---

## TARGET B — audit "already present / reduces to" claims, verified

| Audit claim | Verdict | Evidence |
|---|---|---|
| every 3D/multi-axis RoPE → existing `rope_interleaved`/`rope_halfsplit` + `[rows,Dh/2]` table | **TRUE** (kernel contract) | wan22 interleaved, cosmos halfsplit, magihuman ×2 all map cleanly (see TARGET A). Caveat: per-axis theta (FRAGILE 1). |
| partial RoPE (magihuman, nava) = slice→rope→concat, no kernel | **TRUE** | `magihuman_dit.rs:153,402` both reduce to `rope_halfsplit`. |
| `layer_norm_no_affine` present & usable | **TRUE** | `ops/norm.mojo:603` `def layer_norm_no_affine(x:Tensor, eps, ctx) -> Tensor`. Probe f32 cos 0.99999999, bf16 0.9999973 PASS (re-ran phase24_gap_ops_probe, exit 0). LayoutTensor path, not TileTensor. |
| `gelu_exact` present & usable | **TRUE** | `ops/activations.mojo:322` `def gelu_exact(x:Tensor, ctx)`. Probe f32 cos 1.0, bf16 0.9999981 PASS. |
| `rms_norm` (== rms_norm_per_head over last dim) present | **TRUE** | `ops/norm.mojo:147` `def rms_norm(x:Tensor, weight:Tensor, eps, ctx)`. Per-head QK norm == rms_norm on `[...,Dh]` — correct, weight required (caller passes per-head γ). |
| `conv1d` present | **TRUE** | `ops/conv1d.mojo:188` full stride/pad/dilation/groups, `def conv1d(x,weight,bias,...,ctx)`. |
| sorted-MoE (top_k + grouped SwiGLU FFN + scatter) present | **TRUE** | `ops/moe.mojo:68 top_k_router`, `:193 grouped_expert_ffn` (SwiGLU hardcoded, matches hidream), `:363 gated_scatter_add`. Shared expert = linear+swiglu (compose). |
| `patchify3d` missing:true | **TRUE (confirmed absent)** | No `patchify3d`/3D patch-embed op anywhere in `ops/` or `models/`; only a 2D patchify in `ops/layout.mojo`. Next builder won't duplicate a hidden one, and correctly has nothing to reuse for 3D. |

All claimed-present ops are Tensor-API `def ...(... ctx: DeviceContext)` — callable
from a model builder, NOT behind the TileTensor+closure `gamma.origin.mut` wall.

No Rust/Python/autograd leak in the shipped op (oracle py is dev-only, header-
documented as non-runtime). Scope clean: pure Mojo+MAX, GPU-only, inference-only.

---

## Result

```
{
  component: "ops_gap_sweep",
  ropeTableOpOk: true,   // probe re-run: max_err 1.27e-7, exit 0; F32 compute/store;
                         // contract verified vs wan22(interleaved)+cosmos(halfsplit)+magihuman(partial)
  wrongAuditClaims: [
    "rope_tables.mojo:104 docstring: 'for wan22/cosmos use axes_dims=[88,84,84]' is 2x too large — actual full dims are [44,42,42] (head_dim 128); would break rope apply numel validation or silently mis-rotate"
  ],
  confirmedGaps: [
    "patchify3d genuinely absent (missing:true correct)",
    "single-theta builder cannot express cosmos per-axis NTK extrapolation (variants beyond V2_2B default)"
  ],
  blockers: [
    {
      where: "ops/rope_tables.mojo:104 (docstring example)",
      what: "tells wan22/cosmos porters to pass axes_dims=[88,84,84]; correct is [44,42,42] (full dims). [88,84,84] -> half=128 != head_dim/2=64 -> rope apply raises or silently wrong",
      fix: "change docstring to 'wan22/cosmos (head_dim=128): axes_dims=[44,42,42]'; remove 'doubled/half-dims' wording. No code change; op logic is correct."
    }
  ],
  verdict: "BLOCKERS (1)"
}
```
