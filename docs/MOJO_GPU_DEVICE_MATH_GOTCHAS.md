# Mojo GPU device-math gotchas — unresolved-extern transcendentals

> Mojo 1.0.0b1, NVIDIA, JIT via `pixi run mojo run`. Findings here are tagged
> **MEASURED** (a tool result this session proved it) or **HYPOTHESIS** (inferred
> cause, not directly proven) per TENET 4.

## TL;DR
`from std.math import atan2; atan2(y, x)` **inside a GPU kernel fails to link**:
```
ptxas fatal : Unresolved extern function 'atan2f'
```
Fix: don't call `std.math.atan2` on the device — compute atan2 with a
**pure-arithmetic** helper (atan minimax poly + octant folding, no extern). Code
below. Same pattern applies to any transcendental whose `…f` symbol ptxas reports
as an unresolved extern.

## What was measured (2026-06-15, framepfx/mediaeditor GLSL→Mojo-GPU shader port)
- **MEASURED:** GLSL transition kernels (`tr_fade`/`tr_wipe`/`tr_fancy`) using
  `from std.math import atan2` and `atan2(y,x)` all failed at `ptxas` with
  `Unresolved extern function 'atan2f'`. Other math in the same kernels
  (`sin`, `cos`, `sqrt`, `exp`, `log`) compiled+ran fine.
- **MEASURED:** replacing the calls with the pure-arithmetic `_atan2` below made
  all three kernels compile + run on the GPU. PSNR vs a numpy `arctan2` reference
  of the same shader: `tr_fade` 41.6 dB, `tr_wipe` 71.8 dB (PASS ≥40 dB).
  (`tr_fancy` 25.4 dB is a separate procedural-shader reference-approximation
  issue, NOT the atan2 fix.)
- **HYPOTHESIS (cause):** `std.math.sin/cos/exp/log/sqrt` have device-intrinsic
  lowerings (consistent with serenitymojo's working GPU usage — see the vocoder
  "libdevice sin/exp" note in `serenitymojo/pipeline/ltx2_vocoder_smoke.mojo` and
  `MOJO_KERNELS.md` §11 sinusoidal-fill parity), but `std.math.atan2` does not, so
  it falls back to a host libm `atan2f` the device image can't resolve.
- **NOT cleanly measured:** whether single-arg `std.math.atan` lowers on the
  device (a probe this session failed to instantiate for an unrelated width-1
  reason). The pure-arithmetic helper below sidesteps the question entirely.

## The fix — drop-in device-safe atan2 (pure arithmetic, no extern)
~1e-6 accuracy (atan minimax poly on [0,1]) + octant folding. Uses only
`*`, `/`, `+`, `-` and comparisons — guaranteed to lower. No `min`/`max`/`abs`
calls (folded into conditionals to avoid any free-function/intrinsic surprises).

```mojo
fn _atan2(y: Float32, x: Float32) -> Float32:
    var ax = x if x >= 0.0 else -x
    var ay = y if y >= 0.0 else -y
    if ax == 0.0 and ay == 0.0:
        return Float32(0.0)
    var swap = ay > ax
    var a = (ax / ay) if swap else (ay / ax)
    var a2 = a * a
    var r = a * (0.9998660 + a2 * (-0.3302995 + a2 * (0.1801410 + a2 * (-0.0851330 + a2 * 0.0208351))))
    if swap:
        r = Float32(1.5707963267948966) - r   # pi/2 - r
    if x < 0.0:
        r = Float32(3.141592653589793) - r     # pi - r
    if y < 0.0:
        r = -r
    return r
```

## General rule
When `ptxas fatal : Unresolved extern function '<name>f'` appears for a math
function inside a GPU kernel, that `std.math` function has no device lowering on
this build. Either (a) replace it with a pure-arithmetic polynomial/CORDIC
implementation, or (b) verify the function is one of the known-lowering set
(`sin/cos/exp/log/sqrt`) before relying on it. Bit-exactness vs the host/numpy
reference is NOT guaranteed by a polynomial replacement — gate it by PSNR/ULP.

---

# Finding 2 — the GLSL `fract(sin(x)·43758.5453)` hash is NOT portable (2026-06-15)

The classic one-liner shader hash `fract(sin(dot(co,(12.9898,78.233)))*43758.5453)`
**cannot be gated per-pixel against a numpy reference.** Measured while gating the
`tr_geom` transition pack (gl-transitions → Mojo GPU):

- **MEASURED:** 7/10 transitions in `tr_geom` are bit-exact (99 dB) vs numpy —
  including tiling/mod/trig ones (polkadots, kaleidoscope, hexagonalize, pixelize).
  So coordinate math, `glsl_mod`, `floor`, bilinear `sample`, and `smoothstep` all
  port faithfully.
- **MEASURED:** the 3 that diverge (`randomsquares` 12 dB, `mosaic` 11 dB,
  `gridflip` 10 dB) are exactly the 3 that call the `sin`-hash, and their Mojo
  helper bodies (`glsl_rand`, `fract`, `smoothstep_f`) are **byte-identical** to the
  numpy reference. 101/121 hash tiles differ.
- **MEASURED:** the hash value for the same integer tile coords swings wildly by
  the sin backend. For `d = rx·12.9898 + ry·78.233` (up to ~912):
  - numpy `sin(f32)` vs numpy `sin(f64)` → maxdiff only 0.004 (agree),
  - numpy `sin(d)` vs `sin(d − 2π·floor(d/2π))` **in float32** → maxdiff **0.93**,
    105/121 tiles flip. The float32 argument-reduction is itself lossy, and the
    GPU's `sin` does its own (different) reduction → a different hash again.
- **MEASURED (structural correctness):** despite the different hash, the Mojo
  `randomsquares` output is a *correct* transition — **100% of pixels lie on the
  segment [S1(ptx,pty), S2(ptx,pty)]** (every pixel is a valid blend of the two
  sources at the right coords; only the random reveal order differs). The kernel is
  right; the per-pixel oracle is ill-posed.

**Rule:** `fract(sin(·)·k)` with a large argument is implementation-defined — every
`sin` (numpy-f32, numpy-f64, device SFU, f32-range-reduced) gives a different draw.
Do **not** gate hash-driven shaders by per-pixel PSNR vs an independent `sin`.
Gate them **structurally** instead: (a) output pixels lie on the [src0,src1]
blend manifold, (b) reveal fraction tracks `progress`, (c) run-to-run determinism.
If you need cross-impl reproducibility, replace the sin-hash with a device-stable
**integer** hash (PCG/Wang) in BOTH kernel and reference — but that deviates from
the gl-transitions source and changes the (cosmetic) reveal pattern.
