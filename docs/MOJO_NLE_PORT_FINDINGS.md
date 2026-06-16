# Mojo NLE port — reusable findings (MojoMedia)

> The Mojo 1.0.0b1 idioms and gotchas that cost real time porting a non-linear
> video editor (FramePFX C# + MediaEditor/MediaCore C++ → pure Mojo) onto the same
> MAX/GPU stack this repo uses. Sibling to `MOJO_CONVENTIONS.md` and
> `MOJO_GPU_DEVICE_MATH_GOTCHAS.md`. Project lives at github.com/CodeAlexx/MojoMedia.
>
> Tagged **MEASURED** (a tool result/test proved it) or **HYPOTHESIS** per TENET 4.
> This is the "how not to waste a day on the engine+GPU+UI boundary" doc.

---

## 0. One Mojo build → one binary across "separate" pixi envs

**MEASURED.** Two `pixi` projects (an engine env and this GPU env) were both
`Mojo 1.0.0b1 (a9591de6)`. A MediaCore engine unit built **and passed its gate inside
this GPU env** unchanged. So there is no toolchain barrier: engine structs + GPU
kernels + FFI shims all live in **one binary, one env** (the GPU env is a superset).
Don't architect an FFI/process bridge between Mojo "projects" that share a toolchain —
just add `-I <other_src>` and link the shims. Verified by `integration/ui_vertical.mojo`
(MediaCore compositor → GPU filter → MojoGUI `draw_image` → glReadPixels, bit-exact).

Build/run dance (note the per-run pkg clean, same as §0 of conventions):
```bash
cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
DISPLAY=:1 pixi run mojo run -I . -I /path/to/mediacore \
  -Xlinker -L<ffi> -Xlinker -l<shim> ... -Xlinker -lm -Xlinker -rpath -Xlinker <ffi> file.mojo
```

## 1. Owning-struct field re-deref → premature destruction (CODEGEN BITE)

**MEASURED.** A struct owning a heap buffer (`UnsafePointer` + `owns_data` + a custom
`__del__`) read correctly when its `.data[i]` was dereferenced once, but reading the
SAME field again across several *later* statements returned freed/garbage memory — two
byte-identical `print(mat.data[0..3])` back-to-back gave correct then garbage, with the
struct's metadata (`w/h/c/owns_data`) still intact. The owning value was destroyed early
by 1.0.0b1 codegen between uses.

**Rule:** for an owning struct, **read the heap field once into locals at produce-time**
(or hand the buffer straight to FFI/GPU upload), then use the locals. Don't scatter late
`mat.data[i]` reads across many statements. This also silently corrupted a *self-test's
displayed value while the actual gate (an earlier loop over the same buffer) passed* —
i.e. it manufactured a confusing false-looking failure. Capture-to-locals fixes both.

## 2. MediaCore unit fix taxonomy (apply PER UNIT — never a blind global sweep)

A broad regex sweep (e.g. adding `raises` to every `):`) BREAKS working code (matched
`struct`/`if`/`for`/destructors). **MEASURED**: the sweep regressed already-passing
units; reverting + per-unit surgical fixes was the only reliable path. Each unit is
fixed and gated individually, like a kernel. The recurring fixes:

| Symptom | Fix |
|---|---|
| `cannot call function that may raise in a context that cannot raise` | add `raises` to THAT fn only — NOT to `__del__`/copy/move ctors, NOT to `struct`/`for`/`while` lines |
| `value cannot be implicitly copied` (List / owned struct) | `.copy()` to read, `^` to move/return; List shift `a[i]=a[i+1]` → `.copy()`; read-modify-write-back → `.copy()` then `^` |
| `ambiguous call to 'copy'` | a custom `def copy` clashes with synthesized `Copyable.copy` → rename custom to `clone` |
| `__lt__` UInt↔Int at `global_idx` | `global_idx.x` is **Int** in this build → `if i < n` (drop `UInt(...)`); also literal-LHS `0.0 < x` → `x > 0.0` |
| bare tuple return `-> (A,B)` | `-> Tuple[A,B]` + `return Tuple[A,B](a,b)` |
| `alloc[T, Origin](n)` 2 params | `alloc[T](n)` (1 type param) |
| reserved-name param `out`/`ref` | rename (`dst`/`refv`) |
| `ld returned 1` | needs `-Xlinker -lm` (sin/cos/sinf) and/or the FFI shim `.so` |

(Device-math: `atan2`/`fract(sin·k)` — see `MOJO_GPU_DEVICE_MATH_GOTCHAS.md` Findings 1-2.)

## 3. Verification oracles that work for an NLE in Mojo

The main loop owns every gate; agents write but never self-gate. The independent oracles:
- **video compositing** → numpy straight-alpha-over; the discriminating fixture is a
  NON-opaque overlay (`a_eff=op·oa`) — a flat per-channel lerp passes the opaque case
  but fails this one. Also a **structural** check for procedural/hash shaders: every
  output pixel must lie on the `[src0,src1]` blend segment at its own coords (proves a
  transition is correct even when its random reveal order is a non-portable hash).
- **audio mixer** → `ffmpeg` CLI (amix + the libavfilter chain), sample-exact via a C shim.
- **video decode** → `ffmpeg` CLI, pixel-exact (frame-accurate seek = keyframe-back + decode-fwd).
- **UI rendering** → `glReadPixels` back-buffer readback to a PPM (no WM/screen-grab needed),
  then assert pixels in the drawn rect; inject **synthetic mouse state as function args**
  (not live `get_mouse_*`) so widgets/editors verify headless.
- **GPU filters** → numpy reference of the same algorithm, PSNR ≥ 40 dB (10/19 bit-exact).

## 4. Pointers
- Repo: `github.com/CodeAlexx/MojoMedia` — `docs/VERIFICATION.md` (measured numbers),
  `docs/HANDOFF.md` (pick-it-up state + build incantations), `integration/` (one-binary proofs).
- Device math: `docs/MOJO_GPU_DEVICE_MATH_GOTCHAS.md`. General idioms: `docs/MOJO_CONVENTIONS.md`.
