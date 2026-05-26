# SKEPTIC FINDINGS — serenity-safetensors→Mojo chunk1 — 2026-05-25
Scope: dtype, ffi, mmap, json_header, safetensors (read path). Reviewer ran adversarial probes + a real byte-parity oracle; did NOT edit scope modules.

Reference: /home/alex/serenity-safetensors/src/mmap.rs (full) + lib.rs:52-99 (dtype tables).
Parity tools (kept): serenitymojo/io/parity/{oracle.py,mojo_dump.mojo,compare.py,probe_*.mojo,README.md}

## Verdict
The reader is **byte-correct on every real Z-Image shard** (1163/1163 tensors, incl. a 9.3 GB segment with >2^33 offsets). The FFI widths, dtype tables, and JSON parser are all correct for real inputs. The one BLOCKER is a **memory-safety footgun** the builder already flagged: returned pointers have no lifetime tie to the region and SIGSEGV on use-after-destruction, with zero compile-time protection. Everything else is divergence-vs-Rust that does not affect real safetensors files.

## Findings

### F1: mmap pointer use-after-munmap — SIGSEGV, no compile-time protection
- Where: safetensors.mojo:153-159 (`tensor_ptr` returns bare `BytePtr`), mmap.mojo:139-142 (`__del__`→munmap), safetensors.mojo:49 (`SafeTensors(Movable)`).
- What: `tensor_ptr()` returns `region.as_ptr() + offset` as a raw `UnsafePointer[UInt8, MutExternalOrigin]`. `MutExternalOrigin` deliberately severs origin tracking, so the compiler does NOT keep the `SafeTensors`/`MmapRegion` alive for the pointer's users. Mojo destroys a value at its last textual use; when `SafeTensors` dies, `MmapRegion.__del__` munmaps, and any later deref reads unmapped memory.
- Expected: in Rust the equivalent `*const u8` is unsafe too, but the Rust API is gated behind `&self` borrows and a PyO3 `usize` return; the Mojo version offers a function that can return the pointer past the owner's lifetime with no diagnostic.
- Why it matters: the realistic loader pattern ("open shard, hand weight pointers to the model, drop the handle") silently corrupts. It is not a crash-every-time bug — if the freed pages happen to stay mapped you read stale-but-plausible bytes. That is the worst kind: intermittent data corruption in training/inference weights.
- Severity: **BLOCKER** (memory safety / silent corruption). Not a parity bug — the bytes are right while the region is alive.
- Evidence: `probe_lifetime.mojo` (returns ptr from a fn that drops `SafeTensors`, then derefs) **compiles** and exits **139 (SIGSEGV)**, faulting in the user frame at the munmap'd address. Control `probe_lifetime_ok.mojo` (region kept alive) derefs cleanly (byte=105, exit 0). The smoke only works because `st` is textually live for the whole function.

### F2: header `pread` is single-shot, no short-read retry loop
- Where: safetensors.mojo:73-77 (8-byte len) and :93-97 (header bytes).
- What: `sys_pread(fd, hbuf, header_len, 8)` is called once; if `hread != header_len` the open is failed. `pread(2)` may return fewer bytes than requested (EINTR mid-read, or partial reads on some FS), which a retry loop would absorb.
- Expected: mmap.rs:177/186 uses `file.read_exact(...)`, which **loops** until the buffer is full or a hard EOF/error.
- Why it matters: on local regular files within EOF, Linux pread almost always fills the buffer, so this won't bite the Z-Image files (it didn't). But under EINTR/NFS/odd FS it would spuriously fail an open that Rust would complete. Divergence from the reference's guarantee.
- Severity: **CORRECT-BUT-FRAGILE**.
- Evidence: code read; all 6 real files pread'd fully (parity PASS). No loop present; failure path is `raise`, not retry.

### F3: unknown dtype string FAILS the whole open (Rust stores it verbatim)
- Where: safetensors.mojo:143 `STDtype.from_name(e.dtype)` (raises on unknown); dtype.mojo:118-152.
- What: the Mojo index build calls `from_name`, which `raise`s "Unknown dtype" for any string outside the 15 canonical names. mmap.rs:210-213 stores `dtype` as a raw `String` (`unwrap_or("F32")`) and never validates — an unknown dtype would be carried, not rejected.
- Expected (Rust behavior): open succeeds, dtype string preserved.
- Why it matters: stricter than the reference. Real safetensors only emit the 15 names (Z-Image is all BF16), so no real file trips it; but it is a behavioral divergence — a file Rust opens, Mojo rejects. Arguably an improvement (fail-fast), but it is not parity.
- Severity: **CORRECT-BUT-FRAGILE** (divergence; stricter-than-ref).
- Evidence: dtype.mojo:152 `raise Error("Unknown dtype: ")`; mmap.rs:210-213 has no validation.

### F4: negative / float `data_offsets` RAISE (Rust silently defaults to 0)
- Where: json_header.mojo:164-182 (`_parse_int` digits-only), :185-207 (`_parse_int_array`).
- What: `_parse_int` reads only `0-9`; a `-` or `.` makes it raise ("expected integer" / "expected ',' or ']'"). serde's `as_u64()` returns `None` for negative/float, which mmap.rs:221-224 filters → `unwrap_or((0,0))` (silent default, size 0).
- Expected (Rust): no error; tensor gets offset/size (0,0).
- Why it matters: divergence on malformed input. Mojo's loud raise is arguably safer than Rust's silent zero-size tensor, but it is not parity. No real safetensors emits negative/float offsets.
- Severity: **CORRECT-BUT-FRAGILE** (divergence; arguably-better).
- Evidence: `probe_json.mojo` cases `negative-offset` → "JSON parse: expected integer at byte 48"; `float-offset` → "JSON parse: expected ',' or ']' in array at byte 49".

### F5: `\uXXXX` surrogate pairs produce invalid UTF-8 → raise (BMP-only decoder)
- Where: json_header.mojo:121-129 (`\u` path), :150-161 (`_emit_utf8`, 3-byte max).
- What: each `\uXXXX` is decoded and UTF-8-encoded independently; a surrogate pair (`😀`) emits two separate 3-byte sequences for the lone surrogates → invalid UTF-8 → `String(from_utf8=...)` raises. serde combines surrogate pairs into the real code point.
- Expected (Rust): correct non-BMP decoding.
- Why it matters: safetensors tensor names are effectively ASCII; non-BMP names don't occur. Pure robustness gap, not a real-file issue.
- Severity: **CORRECT-BUT-FRAGILE** (divergence; documented BMP-only).
- Evidence: `probe_json2.mojo` `surrogate-pair` → "Cannot construct a String from invalid UTF-8 data".

### F6: `__metadata__` skip uses single-delimiter brace counting (fragile on mixed nesting)
- Where: json_header.mojo:210-236 (`_skip_value`).
- What: when skipping a container it fixes `open_ch`/`close_ch` to the OUTER delimiter only (`{`+`}` OR `[`+`]`) and ignores the other pair. Strings are consumed correctly (so braces inside strings are safe). For valid JSON this is fine because a closing delimiter of the *other* type never appears unescaped where it would unbalance the count. But it is not a real JSON value skipper — pathological-but-valid structures could in principle mis-balance.
- Expected: serde parses arbitrary nested JSON.
- Why it matters: real `__metadata__` is a flat `{string:string}` map; the probes (nested object+array+escaped quote, braces-in-strings, string-valued `__metadata__`) all passed. Theoretical fragility only.
- Severity: **STYLE/CORRECT-BUT-FRAGILE**.
- Evidence: code read; `probe_json.mojo` nested-metadata + `probe_json2.mojo` metadata-brace-in-string / metadata-string-value all OK.

### F7: `header_len == 0` rejected (Rust would proceed)
- Where: safetensors.mojo:87-89 `if header_len <= 0: raise "Empty or invalid header length"`.
- What: mmap.rs:178-188 accepts header_len 0 (reads 0 header bytes, parses empty `{}`, data_offset 8); only `data_len == 0` is the gate. Mojo rejects header_len 0 up front. (Also serves as the catch for a corrupt huge u64 that wraps negative when accumulated into signed `Int` at :79-80, since `> MAX_HEADER_LEN` won't fire on a negative.)
- Severity: **STYLE** (divergence on a degenerate input; no real file has a 0-byte header).
- Evidence: code read.

## Parity reproduction
Independent oracle = raw-header parse (8-byte LE len + hand-parsed JSON), cross-checked against the official `safetensors` lib (0.7.0): on tf1, 423/423 keys match, 0 shape mismatches, dtype "BF16" — oracle is trustworthy. Hash = 64-bit FNV-1a over first 64 KiB + last 64 KiB of each tensor's mmap'd bytes (boundary-sensitive → catches any offset/size error); identical algo in Mojo and Python.

Match = (name, dtype.name(), shape, offset, size, windowed-fnv1a64) all equal.

| File | tensors | data-seg max offset | result |
|---|---|---|---|
| vae/diffusion_pytorch_model | 244 | 167.6 MB | **244/244 byte-identical** |
| transformer-00001-of-00002 | 423 | 9.97 GB (>2^33) | **423/423 byte-identical** |
| transformer-00002-of-00002 | 98 | 2.34 GB (>2^31) | **98/98 byte-identical** |
| text_encoder-00001-of-00003 | 174 | 3.96 GB (>2^31) | **174/174 byte-identical** |
| text_encoder-00002-of-00003 | 219 | 3.99 GB (>2^31) | **219/219 byte-identical** |
| text_encoder-00003-of-00003 | 5 | 99.6 MB | **5/5 byte-identical** |
| **TOTAL** | **1163** | — | **1163/1163** |

The >2 GB / >4 GB / >8 GB offsets are the exact "untested" territory the builder flagged (PORT_STATE skeptic-bait #2/#3). They pass — the FFI is 64-bit clean end to end and the JSON parser handles the large ints.

## Clean checks (genuinely could not break these)
- **FFI constants** (ffi.mojo:29-37): PROT_READ=1, MAP_PRIVATE=2, MAP_NORESERVE=0x4000, MADV_WILLNEED=3, MADV_DONTNEED=4, _SC_PAGESIZE=30, O_RDONLY=0, SEEK_END=2, SEEK_SET=0 — all verified against /usr/include + a gcc probe (`_SC_PAGESIZE`=30, sysconf→4096).
- **FFI widths**: mmap addr/length/offset=Int(64), prot/flags/fd=Int32; pread count/offset=Int(64). Matches the x86-64 C ABI (off_t/size_t/long all 8 bytes per gcc). `probe_ffi.mojo`: MAP_FAILED round-trips to -1; `BytePtr + 9_973_681_280` and page-align math produce no truncation.
- **MAP_FAILED detection** (mmap.mojo:97): `Int(base) == Int(map_failed())` == compare to -1 — correct for the all-ones (void*)-1 sentinel; verified.
- **dtype tables** (dtype.mojo): all 15 dtypes — byte_size groups match lib.rs:72-75 (8/4/2/1), name() matches lib.rs:82-96 exactly, from_name() round-trips. `probe_dtype.mojo` dumps all 15 correct.
- **mmap.rs:171-238 walk**: open→8-byte LE len→100 MB cap (MAX_HEADER_LEN)→read header→data_offset=8+len→data_len=file_len-data_offset→reject 0→mmap data seg→index with offset=do[0], size=do[1]-do[0] (saturating to 0), `__metadata__` skipped, missing-dtype→"F32". All mirrored; no reorder, no missing cap, offset/size math correct. Confirmed by 1163/1163 parity.
- **page-align math** (mmap.mojo:81-83, prefetch :125-128): identical to mmap.rs:76-78/124-126.
- **fd-close-after-mmap** (safetensors.mojo:132): safe on Linux (mapping holds its own ref); documented; Rust keeps `_file` but the result is equivalent.
- **madvise functional** (`probe_madvise.mojo`): prefetch (incl. missing-name guard), release_to_os, re-access-after-DONTNEED, data_size all work; bytes identical before/after release.
- **No Python in runtime path**: grep of all 5 scope modules + smoke shows only `std.*` and local relative imports. Hand-rolled JSON; no new deps.

## Couldn't verify
- **TOCTOU / true SIGBUS on truncation mid-access**: not exercised (would require truncating a file under an active mapping). Both Rust and Mojo accept this as inherent; the file-size pre-check is mirrored. Not a divergence.
- **header_len > 100 MB rejection on a real file**: synthetic only — no real file has such a header. The code path (`> MAX_HEADER_LEN` raise) is read but not run against a >100 MB header.
- **Concurrent reads / thread-safety**: Rust marks `MmapRegion: Send+Sync`. Mojo has no equivalent annotation and this was not tested; out of scope for the read-correctness gate but worth noting if the loader is shared across threads.
- **Exotic dtype byte-parity on a real file**: every Z-Image tensor is BF16. F8/U16/U32/U64/BOOL byte_size/name were unit-checked (`probe_dtype.mojo`) but never round-tripped through a real file (none exists locally).
- **`to_mojo_dtype` for non-BF16/F16/F32**: raises by design; not exercised against a real non-float file.
