# MOJO-libs

A collection of **100% Mojo** systems libraries — networking, HTTP/1.1 + HTTP/2 +
TLS + WebSockets, JSON (with a high-performance tape parser), an async executor,
2D graphics (canvas, primitives, text, charts, PNG), image codecs + manipulation
(PNG/JPEG/WebP decode+encode, resize/filters, 16-bit/ICC/EXIF/CMYK), fast memory
allocators (arena/pool/slab/ring), a SQLite-format database engine (read + SELECT +
write, no FFI), a full PDF 1.7 writer + reader
(embedded/subset fonts, encryption, digital signatures), Linux desktop
clipboard integration, and an **immediate-mode GUI toolkit** (`ui/` —
widgets + layout/ID/style stacks, tables, movable/resizable windows,
popups & modals, a color picker, a retained draw-list path API, and
keyframe-curve / node-graph editors) — built from the ground up on
libc/OpenSSL/nghttp2/zlib/brotli via FFI.

Mojo's standard library has no sockets, TLS, HTTP, JSON, PDF, or desktop
clipboard API. These libraries fill that gap. The guiding principle is **"Mojo
for everything; C only for the gaps Mojo genuinely can't reach"** — and those
gaps are small (C-ABI callbacks for nghttp2 and OpenSSL ALPN; zlib's `z_stream`;
desktop clipboard ownership delegated to the user's Wayland/X11 provider). Everything
else — the event loop, sockets, the HTTP/1.1 and WebSocket protocols, the JSON
parser and validator, the DEFLATE codec (both directions), and the PDF crypto
stack (**MD5, SHA-1/256/384/512, RC4, AES, RSA, ECDSA, big-integer math** — all
pure Mojo) — is implemented here.

Toolchain: **Mojo 1.0.0b1 / MAX 26.3** (via [pixi](https://pixi.sh)).

---

## The libraries

### [`json/`](json/) — JSON parser, serializer, validator
- **Tree DOM** (`value`, `parser`, `serialize`): a full RFC 8259 parser/serializer
  with an ergonomic `JSONValue` tree (objects, arrays, all escapes, surrogate
  pairs, int/float). Easy to build and mutate.
- **Tape parser** (`tape`): a flat, allocation-light parser (simdjson/yyjson-style)
  with a path-based accessor — **~43× faster than the tree** and ~0.6× the speed of
  the C reference `yyjson`, in pure Mojo. Floats in the common range are parsed
  **correctly-rounded** (exact mantissa × exact power-of-ten — bit-identical to
  strtod, verified against Python); integers beyond Int64 promote to float instead
  of silently wrapping.
- **Codec** (`codec`): **reflection-driven** validation — derive required fields,
  reject unknown keys, and build schemas straight from a Mojo struct's fields at
  compile time, plus typed extractors that produce FastAPI-style `422` errors.
- **Document tooling**: **JSON Pointer** (RFC 6901), **JSON Patch** + **Merge Patch**
  (RFC 6902/7386), a **streaming SAX** pull-parser + **NDJSON**, **JSON Schema** (Draft-07
  subset — 60/60 verdicts match Python `jsonschema`), and **canonical/sorted** output
  (byte-identical to Python `sort_keys`). All RFC-vector / Python cross-checked.

| Parser | Throughput (8 MB doc) |
|---|---|
| `json.tape` (this lib) | **~800–860 MB/s** (~0.6× yyjson, ~43× the tree) |
| tree DOM (`json.parser`) | ~19 MB/s |
| yyjson 0.12 (C/SIMD, reference only) | ~1361 MB/s |

Both JSON parsers also enforce a recursion **depth limit (512)** — deeply nested
input raises instead of overflowing the stack.

### [`net/`](net/) — sockets, epoll, TLS, HTTP/2 transport
- libc-FFI TCP: `Socket` (RAII), `TCPListener`/`TCPStream`, `tcp_connect`.
- `Epoll`: an edge/level-triggered `epoll(7)` wrapper for single-thread concurrency.
- `tls`: server-side TLS 1.3 via OpenSSL FFI (+ ALPN `h2` selection).
- `h2`: HTTP/2 bindings to nghttp2 (the protocol session lives in a C shim because
  nghttp2 is callback-driven; Mojo shuttles bytes). Captures `:method`, `:path`
  **and request bodies** (DATA frames, including bodies larger than the flow-control
  window).
- `signals`: graceful shutdown via `signalfd` (SIGINT/SIGTERM as a pollable fd).

### [`http/`](http/) — HTTP/1.1 protocol + helpers
- `request`/`response`: parser + serializer (keep-alive, Content-Length, headers,
  `serialize_head()` for HEAD). Header scanning is byte-level so **binary request
  bodies are safe**. `Response` can also **stream** a body from a file —
  Content-Length (`stream_file`) or `Transfer-Encoding: chunked`
  (`stream_file_chunked`) — never materializing it whole.
- `router`: data-driven routing with `{path}` params; HEAD falls back to GET.
- `websocket`: RFC 6455 — handshake (**pure-Mojo SHA-1** + base64) + frame codec +
  **fragmentation reassembly** (`WsReassembler`: multi-frame messages, control frames
  interjected mid-message).
- `compress`: gzip (zlib via a C shim) + brotli (direct FFI, no shim), content-negotiated.
- `staticfiles`: serve files from disk (libc `open`/`read`) + streaming primitives
  (`open_ro`/`file_size_fd`/`read_fd`).
- `client`: a real blocking HTTP/1.1 client — `getaddrinfo` **DNS**, **`https://` client-side
  TLS** (handshake + SNI + cert/hostname verification via `net.tls`), **chunked decode**,
  **gzip/deflate decompression**, **redirect following**, recv timeout, keep-alive reuse.
  Verified 15/15 cleartext + 5/5 over TLS vs live local servers (link `-lssl -lcrypto`).
- `url`: URL parse + percent-encode/decode + query strings (vs Python `urllib`).
- `cookies` / `multipart`: `Cookie`/`Set-Cookie` + `CookieJar` + form-urlencoded; binary-safe
  multipart/form-data (round-trips with Python's `email` parser).
- `headers`: content negotiation (`Accept` q-values), conditional requests (ETag,
  `If-None-Match`/`If-Modified-Since`), `Range`, and HTTP-date (byte-identical to `email.utils`).

### [`async/`](async/) — coroutine event-loop executor
Mojo has **native `async`/`await`**; this is an executor that runs one coroutine
per connection on `net`'s epoll loop. A task `await`s a reusable `await_readable`
awaitable (which parks the coroutine and stashes its handle); the loop resumes it
on I/O readiness. Tasks are **multi-suspend** — one coroutine handles keep-alive
across many requests. `serve_conn` drives the **real `http/request` parser**
(buffered, Content-Length/chunked framed, partial-read reassembly, pipelining,
large bodies) — a genuine async HTTP/1.1 server, the asyncio/uvloop model in pure
Mojo. (Verified 7/7 incl. a request split across recvs and a 200 KB body.)

### [`mem/`](mem/) — fast memory allocators (arena, pool, slab, ring)
Custom allocators over `alloc`/`UnsafePointer` for heavy allocation churn:
**`Arena`** (bump/region, O(1) `reset`), **`GrowableArena`** (chunked, grows without
a cap), **`Pool`** (O(1) fixed-size object pool), **`SlabAllocator`** (size-class
general allocator), **`ByteRing`** (circular FIFO), **`AlignedBuffer`** (SIMD/cache-line
alignment), and **`MemStats`/`TrackingAllocator`** (accounting + leak detection). Hot
paths are `@always_inline` freelist pops / pointer bumps. **170 assertions across 6
test files, all passing**, with microbenchmarks: pool **2.9×**, slab **2.5×**, arena
bump+reset **~5.3×** vs raw `alloc`/`free`; ring push+pop **854 Mops/s** — all measured,
not asserted. (Package dir is `mem/`, not `memory/` — avoids the `std.memory` clash.)

### [`graphics/`](graphics/) — pro-class 2D graphics (canvas, vector paths, charts, PNG)
A from-scratch 2D library: an RGBA `Canvas` (alpha blend, `blit`, SSAA
`downsampled`), integer + **anti-aliased** primitives, **scanline polygon fill**,
**vector paths** (`move/line/quad/cubic`, fill with holes + width stroke) with
**affine transforms** (translate/scale/rotate), **gradients** (linear/radial),
rounded rects, a built-in **5x7 font** (A-Z, a-z), a full **chart suite**
(bar/line/area/scatter/pie/donut), and a **real PNG encoder** with real **DEFLATE**
compression (LZ77 + Huffman) — output opens anywhere. No image/compression library
linked. **163 pixel-level assertions** across 20 test files; PNGs validated with PIL
and DEFLATE round-tripped through `zlib`. Bridges to MojoUI as offscreen textures.
[See the gallery](graphics/#gallery) (all images rendered + PNG-encoded by the lib):

![graphics dashboard](graphics/images/dashboard.png)

### [`image/`](image/) — image codecs + manipulation (PNG/JPEG/WebP)
Decode **and** encode **PNG** (all color types, Adam7, 16-bit), **JPEG** (baseline
decode + encode), and **WebP** (VP8L lossless), plus a full **manipulation** suite
(resize nearest/bilinear/bicubic/**lanczos**, rotate/flip/crop, brightness/contrast/
gamma/saturation/levels, blur/sharpen/sobel/median/**unsharp**) and **studio**
features (**16-bit/F32 HDR**, **ICC** profiles, **EXIF** read/write, **CMYK**). Pure
Mojo — reuses `graphics`'s DEFLATE/INFLATE; no codec C library linked. Every codec
+ op is verified against **Pillow** (and littleCMS for ICC): lossless **pixel-exact**
(PNG/WebP/CMYK; bilinear/bicubic/lanczos-upscale resize bit-exact), lossy by **PSNR**
(JPEG 53–65 dB decode). **11 test files**, reproducible fixture generators committed.
Bridges to/from `graphics.Canvas`. **GPU-accelerated** color ops + convolution (`gpu.mojo`,
MAX) are bit-exact vs CPU and **~18× faster** on a 1024² gaussian (RTX 3090 Ti, measured).

### [`audio/`](audio/) — pure-Mojo WAV I/O (read + write)
Read/write **WAV** as interleaved `Float32` samples — decode PCM **u8/s16/s24/s32**
+ **IEEE float32** (any rate/channels, `WAVE_FORMAT_EXTENSIBLE`), encode **s16**
(round-to-nearest) or **float32**. Pure Mojo, no FFI. For model audio I/O (write
generated audio, read conditioning/reference clips) and to feed playback. Verified
**9/9** + an independent **scipy/soundfile** oracle: decode is **bit-exact** vs
`scipy.io.wavfile` (0.00e+00 over 55,726 samples), Mojo-written files valid for
other tools. Compressed payloads (mp3-in-wav) are rejected — `ffmpeg` to PCM first.

### [`ffmpeg/`](ffmpeg/) — CLI wrappers (decode → WAV, mux A/V)
Thin wrappers over the system `ffmpeg`/`ffprobe` (libc `system()`, no `libav*`
FFI): `read_audio`/`decode_to_wav` transcode any audio (mp3/m4a/…) → PCM WAV for
the pure-Mojo [`audio`](audio/) reader, and `mux_av`/`encode_frames` combine a
frame sequence (+ audio) into an mp4 — e.g. generated frames + generated audio
(LTX2/NAVA). Verified: an mp3 decodes to samples and a frames+wav mux produces a
valid **h264+aac** mp4 (independent `ffprobe` oracle). Requires `ffmpeg` on PATH.

### [`clipboard/`](clipboard/) — desktop clipboard for Mojo apps
Linux desktop clipboard helpers for app workflows like copying generated image
paths, prompts, logs, and user-selected text. Runtime provider detection covers
Wayland `wl-copy`/`wl-paste`, X11 `xclip`, and X11 `xsel`; OSC52 terminal
clipboard writes are available only when explicitly requested. Payload bytes
travel over provider stdin/stdout, not through shell-interpolated command strings.
The public API is intentionally small (`write_text`, `read_text`, `clear`,
`detect_backend`, `availability_report`) so apps can import one stable module
while backend support grows. Compile-safe tests always run; real clipboard
round-trip is opt-in because it mutates the user's clipboard.

### [`svg/`](svg/) — pure-Mojo SVG icon loader (subset → raster)
Parse an **SVG icon** and rasterize it to a `graphics.Canvas` (RGBA, transparent
bg) → PNG or GPU texture. Pure Mojo on top of the `graphics` vector engine: full
path-`d` mini-language (incl. elliptical-arc→cubic), basic shapes, `<g>`/nested
`transform`, `viewBox` fit, fill/stroke/opacity + `currentColor` from attributes
and inline `style=""`, even-odd fills (holes work). **Icon subset, not full SVG**
(no gradients/`<use>`/clip/mask/filter/text). Verified: parser **26/26** + shapes
**15/15** + xml **7/7** exact, and rendered icons cross-checked by an independent
**PIL oracle** (rect pixel-exact, circle within **0.3%** of a true filled circle,
`currentColor`/transform/viewBox exact).

### [`sqlite/`](sqlite/) — pure-Mojo SQLite-format engine (read + SELECT + write)
Reads **and** writes real **SQLite 3 database files** with **no FFI / no `libsqlite3`
linked** — the on-disk format (varints, record/serial-type codec, table B-trees,
`sqlite_master`), a small **SQL `SELECT`** engine (WHERE/AND/OR/parens/ORDER BY/LIMIT),
and a **writer** (`CREATE TABLE`/`INSERT`) that emits DBs the real `sqlite3` opens with
**`integrity_check: ok`**. A format-compatible *interop subset*, not a SQLite
reimplementation. Verified against system **libsqlite3 3.45** (Python `sqlite3`): reads
a 2000-row table (interior page + 11 leaves) + an overflow row byte-exact; `SELECT`
results match Python cell-for-cell; Mojo-written 500-row DBs pass `integrity_check` and
round-trip. (For a production DB, FFI to the real engine is the right call — this is
pure-Mojo format interop.)

### [`pdf/`](pdf/) — pure-Mojo PDF 1.7 writer (+ reader)
Generate real PDFs: vector graphics, standard-14 text (UTF-8→WinAnsi), **embedded
Unicode TrueType fonts** (Type0/CIDFontType2 Identity-H + ToUnicode, so text is
copyable) with **font subsetting** (DejaVuSans 760 KB → ~198 KB), **AFM-driven text
measurement, word wrapping, and inter-word justification** (`TJ`), document
metadata, **FlateDecode-compressed page streams**, gray/RGB/CMYK color,
**transparency** (ExtGState), **link annotations**, **outline bookmarks**, and
FlateDecode image XObjects (reusing `graphics`'s DEFLATE) — all with a **byte-exact
xref table** + trailer. Enterprise: **encryption** — RC4 (V2/R3) *and* **AES**
(AESV2 128-bit, AESV3 256-bit) via pure-Mojo MD5/RC4/SHA-256/384/512/AES, with
reader-side **decryption** of all three; **digital signatures** — **RSA** *and*
**ECDSA (P-256)** CMS/PKCS#7, validated by `pdfsig` + `openssl`, with in-library
**RSA key + X.509 certificate generation** (pure-Mojo BigInt/Miller-Rabin);
**object-stream + cross-reference-stream compression**; **XMP metadata + `/ID`** and
**PDF/A-2b** structure (embedded sRGB ICC OutputIntent); a font **subsetter** with
astral (format-12 cmap) support; and a **reader** (pure-Mojo DEFLATE *de*compressor)
parsing classic tables, xref streams, object streams, `/Prev` chains and encrypted
input. Verified with poppler (`pdfinfo`/`pdffonts`/`pdftotext`/`pdfsig`), pypdf
(strict + decrypt), fontTools, `exiftool`, pikepdf, and `openssl` (RSA/ECDSA/CMS/
keygen) — **32 tests**, plus a byte-level xref/`/Length` audit across a 32-page
stress file (0 critical / 0 major).
[See the rendered showcase.](pdf/#showcase-rendered-from-a-generated-pdf-via-pdftoppm)

### [`examples/`](examples/) — runnable servers + a FastAPI-style API
Five servers (HTTP+WebSockets, a **hardened** HTTP/1.1 server, HTTPS+wss, cleartext
HTTP/2, HTTP/2-or-HTTP/1.1 over TLS via ALPN), an HTTP client CLI, and a
typed/validated `POST /users` endpoint with auto **OpenAPI** + Swagger `/docs` —
verified against FastAPI as an oracle (matching status codes, bodies, and `422`
error shape). The hardened server adds request-size limits, a connection cap, idle
timeout, write backpressure, **streamed responses**, **prefork multi-core**, and
**graceful shutdown**.

---

## Building

Everything builds from the repo root with `-I .` so top-level packages (`json`,
`net`, `http`, `async`, `clipboard`, etc.) resolve. C shims compile to `.o` and
link via `-Xlinker`.

```bash
# pure-Mojo libs need nothing extra:
pixi run mojo run -I . json/tests/tape_test.mojo

# compile the C shims once:
cc -c -fPIC -O2 http/cshim/http_shim.c -o http/cshim/http_shim.o          # gzip (zlib)
cc -c -fPIC -O2 -I<nghttp2>/include net/cshim/h2_shim.c -o net/cshim/h2_shim.o
cc -c -fPIC -O2 -I/usr/include -I<nghttp2>/include net/cshim/alpn_shim.c -o net/cshim/alpn_shim.o

# an HTTP server (gzip+brotli):
pixi run mojo build -I . examples/main.mojo -o mojohttp \
    -Xlinker http/cshim/http_shim.o -Xlinker -lz \
    -Xlinker <brotli>/libbrotlienc.so -Xlinker <brotli>/libbrotlicommon.so
```

See each library's README and `examples/README.md` for full link lines.

## Status & honesty

Everything here is exercised against **independent oracles**, not self-assertion:
curl, raw-socket WebSocket/HTTP-2 clients, FastAPI, poppler/`pdfsig`, pypdf,
fontTools, `openssl`, Python `urllib`/`email`/`json`/`jsonschema`, pikepdf, and a
live local TLS server. A sampling of what is *measured*:

- **net/http**: HTTP/2 captures request bodies byte-identical to 500 KB; the
  hardened HTTP/1.1 server streams large files + binary bodies; the **client** does
  DNS, chunked, gzip/deflate, redirects, keep-alive, and **`https://` client-side
  TLS** (handshake + SNI + cert/hostname verification) — 15/15 cleartext + 5/5 over
  TLS vs live servers.
- **json**: tape parser ~800 MB/s; **JSON Schema** 60/60 verdicts match Python
  `jsonschema`; **canonical** output byte-identical to Python `sort_keys`;
  Pointer/Patch/Merge against the RFC vectors.
- **pdf**: writer + reader, embedded/subset Unicode fonts, **RC4 + AES** encryption
  *and* decryption, **RSA + ECDSA** signatures (`pdfsig`/`openssl`-validated), object
  streams, XMP, PDF/A structure — verified across 32 tests.
- **image**: PNG/JPEG/WebP decode+encode + manipulation + 16-bit/ICC/EXIF/CMYK —
  11 test files vs Pillow: PNG/WebP/CMYK pixel-exact, resize bit-exact, JPEG 53–65 dB.
- **sqlite**: reads real SQLite files (2000-row interior-page B-tree + overflow row,
  byte-exact), `SELECT` matches Python `sqlite3` cell-for-cell, and Mojo-written DBs
  pass real-`sqlite3` `integrity_check` — pure Mojo, no `libsqlite3` linked.
- **cli / config**: arg parser cross-checked byte-for-byte vs Python `argparse` (64/64);
  INI vs `configparser`, TOML vs `tomllib` (112 tests), with adversarial skeptic probes
  confirming TOML-invalid input + non-ASCII strings are handled correctly.
- **mem**: arena/pool/slab/ring allocators, 170 assertions, with microbenchmarks
  (pool 2.9×, slab 2.5×, arena bump+reset ~5.3× vs raw `alloc`/`free`).
- **clipboard**: compile-safe helper tests cover provider detection/reporting,
  OSC52 sequence generation, and Base64 vectors; real Wayland/X11 round-trip is
  opt-in with `CLIPBOARD_TEST_REAL=1` because it changes the user's clipboard.

Known limits are documented per-lib and are version/scope choices, not design
walls — e.g. the JSON codec's per-field *value* binding is one line per field
(Mojo 1.0.0b1 keeps field types symbolic in a `comptime for`); the HTTP/2 server is
cleartext prior-knowledge `h2c`; the JSON Schema `pattern` engine is a regex subset;
PDF/A is structurally built but not veraPDF-certified here; PDF signatures have no
RFC-3161 timestamp/LTV; and the bundled PDF reader doesn't decrypt encrypted input.
Anything needing OpenSSL (`net.tls`, and therefore `http.client`) must be built with
`-Xlinker -lssl -Xlinker -lcrypto`. External-network egress is unavailable in the
build sandbox, so the https client's real-CA *accept* path is implemented but only
its self-signed *rejection* is exercised here.

### [`cli/`](cli/) — command-line argument parsing
A clap/argparse-class parser: bool **flags** (+ combined shorts `-abc`, **count** `-vvv`),
typed **options** (str/int/float; `--o v`/`--o=v`/`-o v`/`-ov`; defaults, **required**,
**choices**, **multi-value**/repeatable), named **positionals** + variadic rest, `--`
stop, one-level **subcommands**, **env fallback**, **mutually-exclusive groups**, and
auto **`--help`/usage** + `--version`. Clear errors (unknown/missing/bad-coerce/invalid-
choice). **64/64 tests**, cross-checked byte-for-byte against Python `argparse`.

### [`config/`](config/) — config files + env layering (INI/TOML/.env)
Parse **INI** (vs `configparser`) and **TOML** (vs `tomllib`) into a shared typed tree,
layer them with `.env` + process env (**defaults → file → .env → env → overrides**), and
read back typed (`get_int`/`get_bool`/`get_list`, dotted paths, `*_or` defaults). The TOML
parser is UTF-8-correct and **rejects** TOML-invalid input (leading zeros, bad `_`,
duplicate keys, Int64 overflow) + out-of-scope constructs rather than mis-parsing.
**112 tests** across value/ini/toml/config + adversarial skeptic probes.

## Roadmap

- **Native chat client** — a pure-Mojo desktop chat app (a `chat-ui` analog) on
  **MojoUI** + `sqlite`/`http`/`json`. Plan in [`CHAT_UI_TODO.md`](CHAT_UI_TODO.md).

## License

Apache License 2.0. `yyjson` (referenced only for benchmarking, never linked) is
© its authors under MIT.
