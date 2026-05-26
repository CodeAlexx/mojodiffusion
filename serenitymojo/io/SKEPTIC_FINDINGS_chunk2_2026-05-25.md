# SKEPTIC FINDINGS — chunk2 (sharded + tensor_view) — 2026-05-25
Scope: sharded.mojo, tensor_view.mojo (read-only). Callers exercised: smoke_sharded.mojo + fresh parity/ scripts.

## Required-reading receipts (specific facts)
1. `sharded.mojo:444` — `tensor_bytes(self,name) -> Span[UInt8, origin_of(self.shards)]`; the indexed path opens each UNIQUE shard once via `file_to_idx` (`sharded.mojo:385-395`), and `open()` falls through to single-file (`_single_names()`) when no index file is found (`sharded.mojo:398-415`).
   `tensor_view.mojo:82-93` — `from_parts[mut,//,origin](...)` INFERS `origin` from the `data` span (not a named `origin_of(st)`), which is why the one-call `view_of` idiom does not compile (header NOTE lines 96-111).
2. `PORT_STATE.md:4` — chunk-2 claim "escape probe FAILS with correct origin error; parity 10/10 sampled byte-match." UNTESTED list: "all-521 + text_encoder(398, model.safetensors.index.json) full byte-parity through the loader; single-file fallback; _parse_weight_map on text_encoder index." (These are exactly what I closed below.)
3. Index files (globs resolved with `ls`; one snapshot `04cc4abb...`):
   - transformer `diffusion_pytorch_model.safetensors.index.json`: top keys `["metadata","weight_map"]`, `metadata={"total_size":12309817472}`, **521** weight_map entries (423 → shard 00001, 98 → shard 00002).
   - text_encoder `model.safetensors.index.json`: top keys `["metadata","weight_map"]`, `metadata={"total_size":8044936192}`, **398** entries (174 → 00001, 219 → 00002, 5 → 00003). In BOTH files `metadata` precedes `weight_map`, so the dedicated parser's `_skip_value` over the leading `metadata` object is exercised on real data.

---

## Findings

### F1: `ShardedSafeTensors.open` FAILS on the 2nd open of the same dir in one process — BLOCKER
- **Where**: `sharded.mojo` `open()` detection loops (`_path_exists` on `_join`-built paths, lines 370-372 / 400-402) → root cause in `ffi.mojo:80` `sys_open(path.unsafe_ptr(), ...)`.
- **What**: Open a model dir once and keep the result alive; the SECOND `ShardedSafeTensors.open(<same dir>)` raises `"no index and no known single-file safetensors in <dir>"` even though the files plainly exist. Hits BOTH the single-file fallback (vae) AND the indexed path (transformer): the index-file `_path_exists` check itself starts returning False.
- **Expected**: A second open of the same dir returns the same 244 / 521 tensors.
- **Why (root cause, PROVEN)**: `ffi.mojo`'s `sys_open` passes `String.unsafe_ptr()` to libc `open(2)` assuming NUL-termination. For **dynamically-built** Strings (those produced by `_join`, as opposed to comptime/argv constants) that assumption does not hold once a `SafeTensors` mmap from a dynamic path is held alive — the heap layout shifts and `unsafe_ptr()` no longer points at a NUL-terminated C string, so `open(2)` returns `-1`/ENOENT for a path that exists. Decisive isolation (`parity/probe_nulfix.mojo`): on the exact same path String, same instant —
  - `sys_open(p.unsafe_ptr())` → **-1** (3×)
  - `open()` on an explicit NUL-terminated byte copy of the same bytes → **18** (3×)
  Comptime/static path Strings are unaffected (that is why smoke_sharded + single-pass full parity pass). This is a **chunk-1 ffi latent bug that chunk-2 is the first to trigger**, because chunk-2 is the first code to feed `sys_open`/`SafeTensors.open` `_join`-constructed paths and keep mappings alive.
- **Severity**: BLOCKER. Single-pass model loads work (masking it), but any caller that opens the same dir twice, or loops, fails non-deterministically by heap layout.
- **Evidence**:
  - `parity/probe_same_twice.mojo` → `[vae #1] OK shards 1 tensors 244` then `[vae #2 SAME dir] RAISED: no index...`; `[tf #1] OK ... 521` then `[tf #2 SAME dir] RAISED: no index...`.
  - `parity/probe_nulfix.mojo` → the -1 vs 18 contrast above (the fix is to NUL-terminate in `sys_open`, or copy the path into an owned NUL-terminated buffer). Fix belongs in `ffi.mojo` (chunk-1), out of chunk-2 edit scope.
  - Cross-checks proving it is the dynamic-vs-comptime axis: opening with a comptime full path then opening `_join` paths in a loop → all succeed; `SafeTensors.open(_join(...))` held alive then `_path_exists(_join(...))` → False every iteration; `String(REAL)` (comptime) in the identical loop → True every iteration.

### F2: Reassigning a `ShardedSafeTensors` while a view borrows it is a use-after-free, NOT a compile error — CORRECT-BUT-FRAGILE / documentation overstatement
- **Where**: `sharded.mojo:26-39` + `:444-457` module/`tensor_bytes` doc ("REJECTS any attempt to let the view outlive `self`. compile error, verified"); same for `tensor_view` (`:459-470`). Inherited from `safetensors.mojo:172-196`.
- **What**: `var s = sh.tensor_bytes(name)` then `sh = ShardedSafeTensors.open(other)` COMPILES and runs; reading `s` afterward returns **stale, reused memory** (old mmap munmap'd). Same for `tensor_view`.
- **Expected (per the doc claim)**: a compile error.
- **Why**: Mojo 1.0.0b1's origin/borrow checker ties the span to `self` for *escape-return* and *explicit `__del__`* (both correctly rejected — see Clean checks), but does NOT reject *reassignment* of the borrowed binding while a borrow is live. The protection the header advertises as absolute is real for the common footguns (returning a view, dropping the source) but has this one hole.
- **Severity**: CORRECT-BUT-FRAGILE. The doc overstates ("REJECTS any attempt"). Not a chunk-2 regression — chunk-1 `SafeTensors.tensor_bytes` has the identical hole (verified). Recommend softening the claim, not a code change.
- **Evidence**: `parity/probe_arc_uaf.mojo` → `BEFORE reassign: first4 = 105 189 75 61` (matches Python truth 105) → `AFTER reassign: first4 = 72 62 80 62` (wrong = UAF). Confirmed identical on `tensor_view` and on chunk-1 `SafeTensors` directly.

### F3: empty `weight_map` opens with 0 shards / 0 tensors instead of raising — CORRECT-BUT-FRAGILE
- **Where**: `sharded.mojo:386-396` (the `for ref e in wmap.items()` loop simply does nothing for an empty map).
- **What**: An index whose `weight_map` is `{}` yields a `ShardedSafeTensors` with `num_shards()==0, num_tensors()==0` rather than an error.
- **Why**: Benign — any later `tensor_bytes`/`shard_index` raises `"... not found"`. But a malformed/empty index silently "succeeds," which can mask a bad download.
- **Severity**: CORRECT-BUT-FRAGILE (consider raising on empty weight_map).
- **Evidence**: `parity/probe_edges.mojo` case 3 → `OPENED OK: num_shards = 0  num_tensors = 0`.

### F4: single-file fallback error message is misleading when multiple `.safetensors` exist with no index — STYLE
- **Where**: `sharded.mojo:405-409`.
- **What**: A dir with two `.safetensors` files but no index raises `"no index and no known single-file safetensors in <dir>"` — but there ARE safetensors; the loader just can't disambiguate without an index. (It actually picks the first of `_single_names()` if one matches those exact names; only raises if neither known name is present.)
- **Severity**: STYLE. The behavior (require an index for multi-file, else use a known single name) is defensible; the message wording is just inaccurate for the "files present but unrecognized" case.
- **Evidence**: `parity/probe_edges.mojo` case 5/6.

---

## Full parity reproduction
All hashes are FNV-1a 64-bit. Two independent oracles per sharded dir:
(A) **loader vs direct chunk-1**: `ShardedSafeTensors.tensor_bytes(name)` FULL-length FNV vs opening the resolved shard directly with chunk-1 `SafeTensors.open` and hashing the same tensor (chunk-1 is the trusted byte oracle). Also compares dtype + shape + length.
(B) **loader vs independent Python**: Mojo loader windowed-FNV (first 64K + last 64K) vs a Python raw-byte oracle that parses each shard header and reads the exact data-segment slice straight from the file (NOT via the safetensors lib) — a fully independent ground truth.

- **transformer**: **521/521** byte-match.
  - (A) `sharded_full_parity.mojo` → `TOTAL: 521  MATCHED: 521  MISMATCH: 0` (full FNV + dtype + shape).
  - (B) `diff` of Mojo windowed dump vs `sharded_oracle.py` → IDENTICAL (name+len+windowed-FNV) 521/521.
- **text_encoder** (`model.safetensors.index.json`, 3 shards): **398/398** byte-match.
  - (A) `TOTAL: 398  MATCHED: 398  MISMATCH: 0`.
  - (B) `diff` IDENTICAL 398/398.
  - `_parse_weight_map` on the text_encoder index: Mojo parse = **398** name→shard pairs, `diff` vs Python `json.load(...)['weight_map']` → IDENTICAL (task item 2 closed; the leading `metadata` object is correctly skipped by `_skip_value`).
- **single-file VAE via `ShardedSafeTensors`** (no index → single-file fallback): **244/244** full-FNV byte-match (`sharded_vae_parity.mojo` → `VAE TOTAL: 244  MATCHED: 244  MISMATCH: 0`); `num_shards==1`, `num_tensors==244` (Python confirms 244 tensors excl `__metadata__`).

Summary line: **transformer 521/521 ; text_encoder 398/398 ; single-file VAE via ShardedSafeTensors 244/244.**

Kept scripts (NOT deleted): `parity/sharded_full_parity.mojo`, `parity/sharded_oracle.py`, `parity/sharded_vae_parity.mojo`, plus probes `parity/probe_same_twice.mojo` (F1 repro), `parity/probe_nulfix.mojo` (F1 root cause + fix proof), `parity/probe_arc_uaf.mojo` (F2), `parity/probe_sharded_escape.mojo` (escape reject), `parity/probe_edges.mojo` (F3/F4). The previous (crashed) skeptic's chunk-2 scripts were overwritten/removed as instructed.

## Clean checks (verified genuinely correct — not manufactured)
- **Byte parity through the loader is exact** for all three real paths (521 + 398 + 244), proven two independent ways (loader-vs-chunk1 full FNV, and loader-vs-Python-raw-byte). No mismatch in name, length, dtype, shape, or bytes.
- **`_parse_weight_map` is correct** on both indices: 521 and 398 pairs, identical to Python; the dedicated string→string parser handles the leading `metadata` object via balanced-brace `_skip_value`, and the `total_size` number value inside it.
- **Escape protections that ARE real** (build-only):
  - Returning/holding a view past explicit `sh^.__del__()` → COMPILE ERROR "use of uninitialized value 'sh'" (both `tensor_bytes` and `tensor_view`).
  - `from_parts` does NOT widen the origin: feeding a `tensor_bytes` span then `st^.__del__()` and using the view → COMPILE ERROR (narrow origin preserved; item 4c clean).
- **Edge cases that behave correctly**: missing shard file → clean raise `"failed to open: <path>"` (`probe_edges` case 1); index with no `weight_map` key → clean raise `"index JSON: no weight_map key"` (case 2); duplicate tensor name in weight_map → last wins, opens with 1 tensor (`decoder.conv_in.bias`) when run in isolation (collapses like JSON, matches Python).
- **Hygiene (item 6)**: no `python`/`PythonObject` anywhere in `sharded.mojo`/`tensor_view.mojo`; index read bounded by `MAX_INDEX_LEN = 256MB` (`sharded.mojo:47,101-103`) using the same chunk-1 FFI (`sys_open`/`sys_pread`/`file_size`); no new external deps (imports are only `std.memory.ArcPointer` + `serenitymojo.io.*`).
- **chunk-1 `SafeTensors.open` does NOT leak/corrupt on repeat** (2000× clean) — F1 is specifically the dynamic-path-String interaction introduced/exposed by chunk-2's loader, not a chunk-1 regression in isolation.

## Couldn't verify
- Whether the F1 dynamic-String `sys_open` failure is 100% deterministic vs heap-layout-dependent: it reproduces every run on "same dir twice" and on the tight repeat loop, but multi-DIFFERENT-dir sequences (transformer→text_encoder→vae→transformer) all succeed in one process — so the trigger depends on address reuse. Treated as BLOCKER regardless because "open same dir twice" is a plausible real pattern and it fails 100% of the time observed.
- F2 reassignment-UAF on a TRULY concurrent/threaded reader (Mojo 1.0.0b1 has no Send/Sync equivalent here; single-threaded only tested).
- Full-length (non-windowed) Python FNV cross-check: skipped because pure-Python byte-loop FNV over the 12GB+8GB shards is prohibitively slow; the FULL-length match is instead proven Mojo-side (loader == direct chunk-1, full FNV, 521+398), and the Python independent oracle confirms via windowed (first/last 64K) FNV + exact length. sha256/full-length on the multi-GB tensors not run.
- `>256MB` index file (the `MAX_INDEX_LEN` cap path) — real indices are ~30-65KB; the cap branch is untaken on real data. Note: `_read_file_bytes` copies the index byte-by-byte into a `List[UInt8]` (`sharded.mojo:116-119`), which would be slow near the 256MB cap (fine for real ~65KB indices).
