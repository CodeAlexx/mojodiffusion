# BUILD_PLAN: serenity-safetensors → Mojo (read path) — PILOT

Adapted from port-plan (HF-model→Rust). This is Rust→Mojo; flame-core/PORT_SPEC receipts N/A.
Reference (read line-by-line, do NOT infer): /home/alex/serenity-safetensors/src/mmap.rs (full),
dtype byte-size table at src/lib.rs:72-96, dtype names src/lib.rs:82-96, shard weight_map in src/diffusers.rs.
Target: /home/alex/mojodiffusion/serenitymojo/io/ (Mojo 1.0.0b1). Scope = read path only (see PLAN.md decision).
Purpose: also the PILOT that proves the Rust→Mojo port pipeline (small, byte-verifiable).

## Component map (Rust ref → Mojo)
| Rust ref | Mojo target | risk | notes |
|---|---|---|---|
| dtype table lib.rs:72-96 | `dtype.mojo`: `STDtype` + `byte_size()` + `from_name()`/`name()` | low | F64/I64/U64=8, F32/I32/U32=4, F16/BF16/I16/U16=2, F8_E4M3/E5M2/I8/U8/BOOL=1. Map BF16/F16/F32→Mojo DType; carry label+size for rest. |
| libc mmap/madvise/munmap/sysconf | `ffi.mojo` via `external_call` | med | mmap(PROT_READ, MAP_PRIVATE\|MAP_NORESERVE), madvise(WILLNEED/DONTNEED), munmap, sysconf(_SC_PAGESIZE), open/fstat/close. Linux-only. USE mojo:ffi skill. |
| MmapRegion mmap.rs:46-154 | `mmap.mojo`: `MmapRegion` | med | page-align offset; SIGBUS size-check; prefetch_range; release_to_os; `__del__`→munmap. |
| header parse mmap.rs:172-188 | `json_header.mojo` (hand-rolled) | HIGH | 8-byte LE len (cap 100MB) → minimal JSON object parser for `{name:{dtype,shape,data_offsets}}` + skip `__metadata__`. No stdlib JSON, no Python. |
| MmapFile mmap.rs:165-252 | `safetensors.mojo`: `SafeTensors` | med | open→header→data_offset=8+len→mmap data seg→Dict[String,TensorRef{offset,size,dtype,shape}]. tensor_ptr/tensor_info/names/prefetch/release. |
| shard weight_map (diffusers.rs) | `sharded.mojo`: `ShardedSafeTensors` | med | parse `*.safetensors.index.json` weight_map → open N SafeTensors → unified name→(shard,TensorRef). Single-file path when no index. |
| (new) host view + H2D | `tensor_view.mojo` | low | view{ptr,dtype,shape,nbytes}; minimal `to_device` (DeviceContext.enqueue_copy) stub. Full Tensor = PLAN Decision #3, deferred. |

## Build order
1. `dtype.mojo` (no deps)
2. `ffi.mojo` (libc externs)
3. `mmap.mojo` (uses ffi)
4. `json_header.mojo` (no deps; pure parse)
5. `safetensors.mojo` (uses mmap + json_header + dtype)
6. `sharded.mojo` (uses safetensors + json_header)
7. `tensor_view.mojo` (uses dtype; optional device upload)
8. `tests/test_safetensors_parity.mojo` (+ Python byte-oracle script)

## Risk register
- **HIGH — hand-rolled JSON**: header names contain `.`/`/`/digits; values are int arrays + short strings. Edge: escaped chars (rare in tensor names), large ints (offsets > 2^31 → use Int/UInt64). Mitigate: test against real Z-Image transformer header (521 tensors) + text_encoder header.
- **MED — libc FFI**: off_t/size_t widths, MAP_NORESERVE/MADV_* constant values (Linux x86-64). Mitigate: hardcode Linux x86-64 constants; verify by mmap'ing a real shard and reading known bytes.
- **MED — Mojo collection conformance**: `TensorRef` needs Copyable/Movable; `Dict[String, TensorRef]`. Follow mojo-syntax.
- **LOW — "GPU-only" caveat**: weight loading is inherently host-side I/O (mmap→host pages→H2D). Loader yields host views; device upload is a separate small step. Not a contradiction with GPU-only *compute*.
- **LOW — exotic dtypes**: F8_E4M3/E5M2/U16/U32/U64/BOOL carry label+size only for now; compute mapping needed only for BF16/F16/F32 (Z-Image).

## Parity plan (the gate — byte-identical)
- Oracle: Python `safetensors.safe_open(file, framework="np")` (reference impl) AND/OR the Rust serenity-safetensors reader.
- Capture per tensor: (dtype, shape, nbytes, sha256(raw bytes)). Mojo reader must match oracle for EVERY tensor.
- Files: Z-Image transformer shards (`diffusion_pytorch_model-0000{1,2}-of-00002.safetensors` + index) and text_encoder (3 shards + index) at ~/.cache/huggingface/.../Tongyi-MAI--Z-Image.
- Method: load via Mojo reader, dump (name→sha256) ; compare to oracle dump. GATE = 100% match.

## Workflow (proven-pipeline pilot)
port-build (this plan) → port-skeptic (find where it lies) → port-bugfix → parity gate above. NOT model port-parity/smoke.
