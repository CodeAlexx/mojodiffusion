# SKEPTIC findings — PiD `lq_proj` (LQProjection2D, latent branch)

Date: 2026-05-29
Auditor: skeptic pass over the builder's `lq_proj` port.
Verdict: **PASS — port is correct and the gate fails closed.** One harness caveat noted (non-blocking).

## What was audited

- Source: `/tmp/PiD_repo/pid/_src/networks/lq_projection_2d.py` (`LQProjection2D`, `ResBlock`).
- Port: `serenitymojo/models/pid/lq_projection.mojo`.
- Reference dumper: `serenitymojo/models/pid/parity/gen_lq_projection_reference.py`.
- Gate smoke: `serenitymojo/models/pid/lq_projection_smoke.mojo`.

## Structural audit vs PiD source (line-by-line)

All confirmed matching the real repo class for the latent-only config
(`in_channels=0, latent_channels=16, hidden_dim=512, out_dim=1536, num_res_blocks=4, num_outputs=1, zero_init=False, pit_output=False`):

1. **ResBlock = pre-activation, residual on input.** Source `ResBlock.block =
   Sequential(GroupNorm, SiLU, Conv2d k3p1, GroupNorm, SiLU, Conv2d k3p1)`,
   `forward = x + block(x)`. Mojo `ResBlock.forward`: `gn0 → silu → conv0 →
   gn1 → silu → conv1`, then `_add(x, h)`. Norm/act/conv order and the residual
   target (original `x`, not the post-conv tensor) match exactly. ✅
2. **GroupNorm num_groups = 4.** Source `ResBlock(channels, num_groups=4)` default;
   Mojo `_NUM_GROUPS = 4`. ✅
3. **GroupNorm eps = 1e-5.** PyTorch `nn.GroupNorm` default eps=1e-5; Mojo
   `_GN_EPS = 1e-5`. ✅
4. **latent_proj stack.** Source `Sequential(Conv2d(in→hidden k3p1), SiLU,
   Conv2d(hidden→hidden k3p1), ResBlock×4)`. Mojo: `conv0 → silu → conv1 →
   res0..res3`. Channel flow `16 → 512 → 512 (→512 per ResBlock)`. ✅
5. **Conv geometry.** All convs k=3, stride=1, pad=1 (`k3 p1`). Mojo conv2d
   instantiated `[…,3,3,…,1,1,1,1]` (Kh=Kw=3, stride 1, pad 1). ✅
6. **Token flatten.** Source `merged.flatten(2).transpose(1,2)` over NCHW
   `[B,hidden,pH,pW]` → `[B,N,hidden]`. Mojo runs the stack in NHWC and
   reshapes `[B,pH,pW,hidden] → [B,N,hidden]`; NHWC spatial row-major order
   equals NCHW `flatten(2).transpose(1,2)`. Verified numerically by the gate. ✅
7. **Output head.** Source single `output_heads[0] = Linear(hidden→out_dim)`
   (num_outputs=1). Mojo `linear(tokens, head_w, head_b)`. ✅
8. **Sigma-conditioning.** The sigma-aware gate (`SigmaAwareGatePerTokenPerDim`,
   `gate()`) is a SEPARATE injection-time path applied to the transformer hidden
   state, NOT part of `LQProjection2D.forward()`. `forward()` returns the
   projected tokens only; sigma never enters it. The port correctly omits the
   gate from this module. Sigma-gating must be ported/gated when wiring the
   PixelDiT injection (`gate_modules`), and is OUT OF SCOPE for `lq_proj`. ✅

## Spatial-alignment caveat (already documented by builder)

`z_to_patch_ratio = (sr_scale*lsdf)/patch_size = (4*8)/16 = 2.0 > 1` → the real
forward does `F.interpolate(lq_latent, size=(pH,pW), mode="nearest")`. The gate
feeds the latent ALREADY at the patch grid (`zH=pH=4, zW=pW=4`), making that
interpolate an identity, so the conv stack + head are isolated and gated, but
the **nearest-upsample alignment path itself is NOT exercised** by this gate.
This is a correct scoping decision for a unit gate; the upsample path must be
covered when wiring full PixelDiT inference. Documented in the port header and
the generator — confirmed accurate.

## Re-run of the gate (clean)

Regenerated the reference (`gen_lq_projection_reference.py`, seed 1234) and ran
the smoke:

```
ParityResult(cos=0.9999999999992462, max_abs=5.1021575927734375e-05, n=24576, PASS)
LQ_PROJ GATE PASSED (cos >= 0.999)
```

cos = 0.99999999999 (≥0.999), max_abs = 5.1e-05 over 24576 elements — bit-close.

## Fail-closed verification (mutate → confirm fail)

**Strong structural mutation (decisive):** negated the loaded `head.weight`
(`*= -1`) while leaving `ref_output` at the original value. The Mojo forward
recomputes anti-correlated tokens:

```
ParityResult(cos=-0.9996910765056278, max_abs=74.5771484375, n=24576, FAIL)
LQ_PROJ GATE FAILED  → raises Error, process exits non-zero
```

The gate FAILS on a genuine port error and propagates the failure (non-zero
exit). **Fail-closed confirmed.** Reference was then restored and the clean
gate re-confirmed PASS (cos=0.99999999999, max_abs=5.1e-05).

## Harness caveat (NON-BLOCKING, project-wide)

`ParityHarness` (`serenitymojo/parity.mojo`) gates on **cosine only**:
`passed = (cos >= 0.999)`. `max_abs` is computed and printed but is NOT part of
the pass criterion. A probe that corrupted a SINGLE output element by +50
(1 of 24576) yielded `cos=0.99921 (PASS), max_abs=50.0` — cosine is robust to a
lone outlier, so the harness would not fail on a localized one-element error
despite a huge max_abs. This is the shared harness used by every op/module gate
in the repo, not a defect of this port, and the actual `lq_proj` output is
bit-close (max_abs=5.1e-05), so there is no localized error here. Flagging for
awareness: for ops where a localized error is plausible, consider also asserting
a `max_abs` bound. No change required for the `lq_proj` sign-off.

## Conclusion

The PiD `lq_proj` (LQProjection2D latent branch) port is structurally faithful
to the repo source (ResBlock pre-act order, GN groups/eps, conv geometry,
channel flow, token flatten, single output head, sigma correctly excluded),
the numeric gate passes bit-close (cos=0.99999999999, max_abs=5.1e-05), and the
gate is verified fail-closed via a structural weight mutation. Sign-off: PASS.
No git commit (per instructions).
