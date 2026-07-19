# MIXER — int16 soft-add audio mixer (parity to the documented model)

Parity target: reproduce the **yangfan/mixer** audio model exactly. The Mojo
mixer must equal the python oracle (`tests/mixer_oracle.py`) **bit-for-bit** on
every `(tracks, vols)` case.

This is **parity-to-the-documented-model**, not a blind byte-copy:
- the pairwise soft-add is **verbatim** from `mixerEngine/engine_tracker.cpp`;
- the per-track volume placement is **inferred** from `template.json`
  (`audio.trackers.{name}.volume`, an integer `0..100`) — the C++ soft-add
  itself shows no volume, so volume is a template.json upstream gain applied
  before mixing. We pin it as integer + trunc-toward-zero so Mojo == oracle.

## Sample model

- int16 PCM, interleaved stereo (2 channels).
- A buffer is `N` int16 "lanes", `N = sampleFrames * 2`.
- **All math is INTEGER. No float anywhere in the mix path.**
- Intermediates are computed in `int32`-wide range (python int in the oracle);
  only the final stored sample is cast/wrapped to int16.

## Exact formulas

### (1) Per-track volume gain

`template.json` field `audio.trackers.{name}.volume` is an integer `0..100`,
applied as integer gain **before** mixing:

```
g = (int32(sample) * volume) / 100      # C '/' : TRUNCATE TOWARD ZERO
g = clamp(g, -32768, 32767)
```

C `/` truncates toward zero. **Mojo `Int` `//` is FLOOR** (toward −inf), and
**numpy `//` is ALSO floor** — both differ from C for negative samples.
Implement trunc-toward-zero explicitly. Since `volume >= 0` and `100 > 0`, the
product sign equals the sample sign, so:

```
trunc_div(a, 100) = a // 100              if a >= 0
                  = -((-a) // 100)         if a < 0
```

Worked: `gain(20000, 50) = 10000`; `gain(-7, 50) = trunc(-350/100) = -3` (**not**
the floor result `-4`).

### (2) Pairwise saturating soft-add — VERBATIM `engine_tracker.cpp`

Per lane `i`, with `int32 d = dest[i]`, `int32 s = sour[i]`:

```c
if (d < 0 && s < 0)  r = d + s + (d * s / 32767);
else                 r = d + s - (d * s / 32767);
dest[i] = (int16) r;
```

- `d * s` is an `int32` product.
- `/32767` is integer division **truncated toward zero** (in the `else` branch
  `d*s` may be negative when signs differ).
- Store the result as int16 (**wrap** to `[-32768, 32767]`) to match the C
  `(int16)` cast exactly. The formula keeps `r` in range, but we wrap anyway.

Worked:
- `soft_add(0, 12345) = 0 + 12345 - 0 = 12345`
- `soft_add(20000, 20000)`: both `> 0` → `40000 - (20000*20000/32767) = 40000 - 12207 = 27793`
- `soft_add(-20000, -20000)`: both `< 0` → `-40000 + (400000000/32767) = -40000 + 12207 = -27793`

### (3) Mix K tracks → out

```
out[i] = 0 for all i
for k in 0..K:
    for i in 0..N:
        g = gain(tracks[k][i], vols[k])    # step (1)
        out[i] = soft_add(out[i], g)       # step (2), accumulator-based
```

**Fold order = track order.** This is load-bearing: the soft-add is
non-associative, so reordering tracks changes the result. With `out` starting at
0, `soft_add(0, g) = 0 + g - 0 = g`, so the first track folds in cleanly.

## template.json volume mapping

```
audio.trackers.{name}.volume : integer 0..100
```

Each tracker `name` maps to one track buffer in the mix. Its `volume` becomes the
`vols[k]` passed to `mix_tracks` for that track. `volume = 100` is identity
(modulo trunc), `volume = 0` drops the track (`gain -> 0`, and
`soft_add(acc, 0) = acc`). Track order in the mix follows the tracker order.

## Source citation

- Soft-add (step 2): `mixerEngine/engine_tracker.cpp` — copied verbatim above.
- Volume (step 1): inferred from `template.json` `audio.trackers.{name}.volume`;
  not present in `engine_tracker.cpp`. This is the documented-model parity point.

## Trunc-toward-zero caveat (read this)

The single biggest divergence risk between C/oracle and Mojo:

| input            | C `/` (trunc) | Mojo `//` / numpy `//` (floor) |
|------------------|---------------|--------------------------------|
| `-350 / 100`     | `-3`          | `-4`                           |
| `-7 * 50 / 100`  | `-3`          | `-4`                           |
| `d*s/32767` (neg)| toward 0      | toward −inf                    |

Both the volume division (`/100`) and the soft-add division (`/32767`) must use
trunc-toward-zero. Do **not** use Mojo `//` or numpy `//` in the mix path. The
oracle implements `trunc_div(a, b)` as `sign(a/b) * (|a| // |b|)` and the Mojo
side must do the equivalent (e.g. divide magnitudes, reapply sign).

## How to run

### Oracle (python, numpy optional)

```
python3 /home/alex/MOJO-libs/audio/tests/mixer_oracle.py
```

It prints `PASS`/`FAIL` for each worked vector, writes cross-check cases to
`/tmp/mixer_oracle.json`, and prints `MIXER_ORACLE: pass` (or `fail`).

`mix_tracks(tracks, vols)` is exposed as the reference entry point and returns an
int16 python list.

### Mojo self-test (cross-check against the oracle)

1. Run the oracle once to (re)generate `/tmp/mixer_oracle.json`.
2. The Mojo test loads that JSON, runs the Mojo mixer on each case's
   `(tracks, vols)`, and asserts the Mojo `out` equals the oracle `out`
   element-for-element. Any mismatch fails the self-test.

The JSON dump carries the int16 bounds and a `div_semantics` note
(`trunc-toward-zero (C '/'); NOT floor`) so the Mojo side can assert it is
matching the right division contract.
