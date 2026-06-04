# SKEPTIC FINDINGS — patchify3d / unpatchify3d (2026-06-03)

Adversarial review of `serenitymojo/ops/patchify3d.mojo` (+probe, +parity, +oracle)
vs `wan22_dit.rs:667-745` and `cosmos_predict25_dit.rs:1544-1616`. Assumed the
port lies; re-derived every layout independently; re-ran both gates.

## §0 receipts
- `serenitymojo/MAP.md` — "where does X live" wayfinding for the pure-Mojo+MAX inference-only GPU diffusion lib; Tensor is the core movable-not-copyable device-bytes type.
- `serenitymojo/docs/SERENITYMOJO_MODULES.md` — per-module public API; all ops take trailing `ctx: DeviceContext`, compute F32 over BF16/F16/F32 storage, return a fresh Tensor; parity gate cos≥0.999.
- wan22/cosmos patch_embed refs — wan22 `patchify` (c-slowest `(c,pf,ph,pw)`, F-major tokens) and `unpatchify` (einsum `fhwpqrc→cfphqwr`, c-fastest `(pf,ph,pw,c)`); cosmos doc-comment (1582-1587) explicitly warns the patchify `(c,r,m,n)` vs unpatchify `(p1,p2,t',c)` asymmetry is in the Python source and the FinalLayer linear is trained to the unpatchify layout.
- `/home/alex/.claude/skills/mojo-syntax/SKILL.md` — current Mojo: `comptime` not `alias`, `def` doesn't auto-raise (add `raises`), `fn`/`let`/`inout`/`owned` removed.

## Independent layout re-derivation (the #1 bug surface)

| Quantity | wan22 ref | Mojo port | Verdict |
|---|---|---|---|
| patchify within-patch | `dst_ch = ci·pf·ph·pw + pfi·ph·pw + phi·pw + pwi` (wan22:692) → (c,pf,ph,pw), c SLOWEST | line 18/61/265-comment: `ci·(pf·ph·pw)+pfi·(ph·pw)+phi·pw+pwi` | MATCH |
| unpatchify within-patch | `src_ch = pfi·ph·pw·c + phi·pw·c + pwi·c + ci` (wan22:729) → (pf,ph,pw,c), c FASTEST | line 265: `((pfi·ph+phi)·pw+pwi)·C+ci` = expands identically | MATCH |
| token order | `patch_idx = fi·ho·wo + hi·wo + wi` (wan22:683/724), F-major | patchify `fi·HO·WO+hi·WO+wi`; unpatchify `(fi·HO+hi)·WO+wi` (= same) | MATCH |
| asymmetry direction | patchify c-slowest, unpatchify c-fastest (DIFFERENT, by design) | port implements exactly this split | MATCH — builder did NOT swap/conflate |

The builder did NOT swap the two within-patch orders. patchify is genuinely
c-slowest; unpatchify is genuinely c-fastest; the asymmetry mirrors the ref
(confirmed verbatim against both wan22 loop bodies and the cosmos permute axes
`[0,2,4,6,1,3,5,7]` patchify / `[0,7,1,6,2,4,3,5]` unpatchify).

## No-transpose claim — VERIFIED
Independently built `w=[OUT,C,pf,ph,pw]`, `w.reshape(OUT, C·pf·ph·pw)` with NO
transpose, and confirmed element `[o, ci·pf·ph·pw+pfi·ph·pw+phi·pw+pwi]` ==
`w[o,ci,pfi,phi,pwi]` bit-exact for all positions. The conv-weight row-major
memory order IS the c-slowest unfold order this op emits. Not just shapes lining
up — actual memory order matches.

## Test is genuinely NON-symmetric (swap cannot hide) — VERIFIED
Concern: a within-patch swap passes a symmetric test (pt=ph=pw, C=patch_elems,
non-distinct values). Checked both geometries:
- parity/oracle: C=16, F=4,H=8,W=8, patch=(1,2,2), OUT=64 → C(16)≠patch_elems(4); grid FO=HO=WO=4 (>1 in all of F,H,W).
- probe: C=6, F=4,H=6,W=8, patch=(1,2,2) → C(6)≠patch_elems(4); FO=4,HO=3,WO=4 distinct, all >1.

I built an adversarial host model on the probe geometry: deliberately swapping
unpatchify to c-slowest gives **max-abs 99.0** vs correct; swapping ph↔pw gives
**max-abs 75.0** — both astronomically above the 1e-6 gate. (The ph=pw=2 equality
does NOT create a blind spot, because the c-fastest read interleaves c between
spatial slots, so a ph↔pw swap still reads wrong elements.) The probe's distinct
per-position values `(i·37+11)%101-50` expose any layout error. Test is honest.

## Round-trip claim — CLARIFIED (correct interpretation)
Builder's "max-abs 0.0 each direction" is NOT a `unpatchify(patchify(x))==x`
identity (which SHOULD differ, by design — they feed different trained linears).
It is each direction vs its OWN host recompute of the wan22 layout:
- probe line 72 / parity line 141 feed a c-slowest tensor into `unpatchify3d`, then check vs a host recompute of the c-fastest einsum read.
- patchify checked vs host c-slowest recompute.
Neither asserts a patchify→unpatchify identity. Claim is about the right thing.

## Reuse / hygiene — VERIFIED
- `ops/linear.linear` is the real op (imported line 26, called line 124 in parity), NOT a reimplemented matmul. F32 accumulation lives in `linear`, not in this gather op (correct — pure index relocation needs no accumulation).
- Positional embedding kept OUT: only in docstring/comments (lines 168,170), zero pos-emb code.
- Mojo hygiene: `comptime` (not `alias`); both public fns are `def … raises`; kernels non-raising; no `var ref`; no `fn`/`let`/`inout`/`owned`/autograd/backward leak; returns `Tensor(out_buf^, out_shape^, …)` with proper `^` moves; parity I/O routes through `io/ffi` (`sys_open/pread/close`).

## COMPILE/PARITY HONESTY — re-ran myself
- `pixi run mojo run -I . serenitymojo/ops/patchify3d_probe.mojo` → exit 0; patchify max-abs **0.0**, unpatchify max-abs **0.0** vs independent host recompute.
- oracle (`patchify3d_oracle.py`, serenityflow venv, CUDA) → exit 0; EQUIVALENCE conv3d(stride=k) vs unfold+linear f32 **cos=1.0000000000 max_abs=0.000e+00 → PROVEN**; bf16 conv vs bf16 unfold cos=1.0.
- `pixi run mojo run -I . serenitymojo/ops/parity/patchify3d_parity.mojo` → exit 0; **cos=0.9999962529629999** (n=4096, PASS), max_abs=0.125 (bf16-typical), magRatio=1.00005, unpatchify-layout max-abs **0.0**. Matches builder's claimed 0.9999963.

## Findings

### FRAGILE-1 — patchify output buffer sized by `x.nbytes()` (line 190)
`out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())`. Correct ONLY because
patchify is a pure permutation (input numel C·F·H·W == output numel L·PD). True
invariant here, but a non-obvious coupling. unpatchify (line 363) self-documents
with `total * seq.dtype().byte_size()`. Suggest patchify do the same:
`total * x.dtype().byte_size()` (== `x.nbytes()`, but explicit). Not a bug.

### STYLE-1 — three near-identical dtype kernels ×2 directions
Six kernels differ only in the dtype tag (f32/bf16/f16). Mirrors `ops/layout.mojo`
convention (documented in header), so acceptable; a `comptime`-parametric kernel
would cut ~200 lines. Cosmetic.

No BLOCKER or FRAGILE that affects correctness. Layout asymmetry is correct and
the parity tests genuinely exercise it.

---
{component:"patchify3d", reRanParity:true, cos:0.9999962529629999, layoutAsymmetryCorrect:true, testIsNonSymmetric:true, blockers:[], verdict:"clean"}
