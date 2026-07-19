# http — HTTP/1.1 protocol, routing, WebSockets, compression for Mojo

The HTTP layer on top of [`net`](../net/). Knows nothing about sockets; `net`
knows nothing about HTTP.

## Modules

| Module | What it is |
|---|---|
| `request.mojo` | `Request` + `parse_request` (request line, case-insensitive headers, body, query); `is_request_complete` / `request_consumed_len` for keep-alive framing; chunked-request decode; `keep_alive_for(raw)` (non-raising byte-level keep-alive decision, usable off the raw buffer — e.g. in an async coroutine body). Header scanning is **byte-level** (`Content-Length`/`Transfer-Encoding`), so a **binary body is safe** (no whole-buffer `lower()` that would hit Mojo's UTF-8 codepoint assert). |
| `response.mojo` | `Response` + `serialize` (status line, headers, computed Content-Length, `Connection: keep-alive`/`close`). Also **streamed** bodies: `stream_file(path)` (Content-Length) / `stream_file_chunked(path)` (`Transfer-Encoding: chunked`) — the server pumps the file, never materializing it whole. |
| `router.mojo` | `Router` — data-driven table of `(method, pattern, handler_id)` with `{name}` path params; `allow_for` (405/OPTIONS). (Handler *values* can't be stored in Mojo, so dispatch is by id.) |
| `websocket.mojo` | RFC 6455: handshake (**pure-Mojo SHA-1** + base64, verified against the spec vectors) + frame encode/decode (FIN/opcodes, 7/16/64-bit lengths, client unmasking) + **fragmentation reassembly** (`WsReassembler`: CONT frames, control frames interjected mid-message). |
| `compress.mojo` | `gzip_bytes` (zlib via `cshim/http_shim.c`) + `brotli_bytes` (direct FFI, no shim). Content-negotiated, brotli preferred. |
| `staticfiles.mojo` | `read_file` + content-type by extension, plus streaming primitives (`open_ro`/`file_size_fd` via `lseek`/`read_fd`/`close_fd`) used by `Response` streaming. |
| `client.mojo` | A real blocking HTTP/1.1 **client**: `getaddrinfo` **DNS**, `get`/`request`, **`https://` (client-side TLS** via `net.tls` — handshake + SNI + optional CA/hostname verification), **chunked-response decode**, **gzip/deflate decompression** (via pure-Mojo `graphics.inflate`), **redirect following** (capped + loop-guarded), recv **timeout**, and a keep-alive **connection-reuse** `Client` (http or https). http + https share one unified `Conn`. Verified 15/15 cleartext + 5/5 over TLS against live local servers. **Build/link with `-Xlinker -lssl -Xlinker -lcrypto`.** |
| `url.mojo` | **URL parsing** (`parse_url` → scheme/userinfo/host/port/path/query/fragment, default ports) + **percent-encode/decode** + **query string** parse/build. Cross-checked vs Python `urllib`. |
| `cookies.mojo` | Request `Cookie` + response `Set-Cookie` parse/serialize (path/domain/max-age/secure/httponly/samesite), a `CookieJar`, and **form-urlencoded** parse/build. |
| `multipart.mojo` | **multipart/form-data** parser — binary-safe (file bytes incl. `0x00` and boundary-like runs), `Content-Disposition` name/filename. Round-trips with Python's `email` parser byte-for-byte. |
| `headers.mojo` | **Content negotiation** (`Accept`/`Accept-Encoding` q-parsing + best-match), **conditional requests** (ETag, `If-None-Match`, `If-Modified-Since`), **Range** requests (`bytes=`, suffix, multi), and **HTTP-date** format/parse — byte-identical to Python `email.utils`. |

## C shim (`cshim/`)

Only gzip needs C: zlib's gzip framing uses a `z_stream` struct that's awkward to
drive from Mojo, so `http_shim.c` does the deflate. Brotli's one-shot
`BrotliEncoderCompress` is called **directly via FFI** — no shim, no header.

```bash
cc -c -fPIC -O2 http/cshim/http_shim.c -o http/cshim/http_shim.o
# link: -Xlinker http/cshim/http_shim.o -Xlinker -lz \
#       -Xlinker <brotli>/libbrotlienc.so -Xlinker <brotli>/libbrotlicommon.so
```

## Highlights

- **Keep-alive + pipelining**: HTTP/1.1 connections persist unless `Connection:
  close`; multiple requests per connection are sliced off the buffer.
- **WebSockets**: handshake's `Sec-WebSocket-Accept` runs on a from-scratch Mojo
  SHA-1 (no crypto lib linked). **Fragmented messages reassemble** (`WsReassembler`),
  state persists across `recv()` boundaries, control frames interject cleanly —
  verified with a raw masked-frame client (12 unit + 6 live assertions). Works over
  TLS too (`wss`, see `examples/tls_server.mojo`).
- **Streamed responses**: `stream_file`/`stream_file_chunked` send large files at
  O(one-chunk) memory; verified byte-identical (Content-Length and chunked framing).
- **Binary-safe bodies**: request bodies of arbitrary bytes (not just UTF-8) — a
  100 KB binary `POST` round-trips byte-identical.
- **Compression**: `/big` 4493 B → 77 B (brotli) / 111 B (gzip); curl round-trips.

```mojo
from http.request import parse_request
from http.response import Response
var req = parse_request(raw)
var r = Response(200)
r.set_header("Content-Type", "text/plain")
r.set_body("hello")
# send r.serialize() over a net.TCPStream / TLS conn
```

See [`examples/`](../examples/) for full servers wiring routing + middleware +
WebSockets + compression together on `net`'s epoll loop.
