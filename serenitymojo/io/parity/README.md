# Byte-parity oracle for the Mojo safetensors reader

Built during port-skeptic 2026-05-25. KEEP — this is the parity gate for chunk 1.

## Tools
- `oracle.py <file>` — independent Python oracle. Parses the raw 8-byte LE
  header-len + JSON header itself (NOT via the safetensors lib for the
  canonical answer; cross-checked against the lib in the findings doc). Emits
  one JSON line per tensor: `{name,dtype,shape,offset,size,fnv1a64}`. The
  `fnv1a64` is a 64-bit FNV-1a over the first 64 KiB + last 64 KiB of the
  tensor's mmap'd bytes (windowed for speed; the boundary bytes differ per
  tensor so any offset/size misalignment is caught).
- `mojo_dump.mojo <file>` — dumps the SAME fields via the pure-Mojo
  `SafeTensors` reader, using a bit-for-bit matching windowed FNV-1a.
- `compare.py <mojo_dump.txt> <oracle.jsonl>` — diffs them; PASS iff every
  tensor matches on all fields incl the hash.

## Reproduce
```
SNAP=~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021
F=$SNAP/transformer/diffusion_pytorch_model-00001-of-00002.safetensors
pixi run python serenitymojo/io/parity/oracle.py "$F" > /tmp/o.jsonl
pixi run mojo run -I . serenitymojo/io/parity/mojo_dump.mojo "$F" > /tmp/m.txt
pixi run python serenitymojo/io/parity/compare.py /tmp/m.txt /tmp/o.jsonl
```

## Result (2026-05-25)
1163/1163 tensors byte-identical across all 6 Z-Image shards (vae 244, tf1 423,
tf2 98, te1 174, te2 219, te3 5). See SKEPTIC_FINDINGS_2026-05-25.md.

## probe_*.mojo — targeted adversarial probes
- `probe_ffi.mojo` — MAP_FAILED round-trip, sysconf, >4 GB pointer/align math.
- `probe_lifetime.mojo` — use-after-munmap (SIGSEGVs: confirms the footgun).
- `probe_lifetime_ok.mojo` — control: deref while alive (no crash).
- `probe_json.mojo` / `probe_json2.mojo` — header parser edge cases.
- `probe_dtype.mojo` — all 15 dtype byte_size/name/from_name.
- `probe_madvise.mojo` — prefetch/release functional check.
