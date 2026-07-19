# examples — runnable servers + a FastAPI-style API

These tie [`net`](../net/) + [`http`](../http/) + [`json`](../json/) together.
`app.mojo` holds the shared route table, handlers, and middleware used by every
server. Build from the **repo root** with `-I .`.

| File | What it runs | Port |
|---|---|---|
| `main.mojo` | HTTP/1.1: epoll + keep-alive + router + WebSockets (**fragmentation reassembly**) + gzip/brotli | 8080 |
| `server.mojo` | **Hardened** HTTP/1.1: limits (431/413/414), conn cap (503), idle timeout, write backpressure (EPOLLOUT), chunked-request decode, **streamed responses** (large files / chunked, O(chunk) memory), **prefork multi-core** (SO_REUSEPORT), **graceful shutdown** (SIGTERM via signalfd) | 8080 |
| `tls_server.mojo` | HTTPS + `wss` (TLS 1.3, OpenSSL) | 8443 |
| `h2_server.mojo` | Cleartext HTTP/2 (`h2c`, nghttp2) — captures **request bodies** | 8090 |
| `h2tls_server.mojo` | HTTP/2 **or** HTTP/1.1 over TLS, negotiated via ALPN | 8444 |
| `client_main.mojo` | HTTP client CLI | — |

## Endpoints (the FastAPI-style part)

`app.mojo` defines a typed `User` model and:

- `POST /users` — validates the JSON body into `User` (codec) → `201` + JSON, or
  **`422 {"detail":[{"type","loc","msg"}]}`** on a validation error.
- `GET /openapi.json` — an OpenAPI 3.0 doc; the `User` schema's `required` list is
  generated from reflection.
- `GET /docs` — Swagger UI.
- `POST /echo` — returns the request body verbatim (exercises the request-body
  path on both HTTP/1.1 and HTTP/2; binary-safe, verified byte-identical).
- plus `/`, `/health`, `/api/status`, `/user/{id}`, `/echo?msg=`, `/big`,
  `/static/{file}`, `/stream/{file}`, `/chunked/{file}`.

### Streamed responses (server.mojo)

`server.mojo` can send a body without ever holding the whole thing in memory —
it pumps the file one `READ_CHUNK` (64 KiB) at a time and only reads the next
chunk once the previous one has flushed, so peak memory is one chunk regardless
of file size. Two routes demonstrate it (files from `examples/www/`):

- `GET /stream/{file}` — streamed with a known **Content-Length** (the server
  stats the file with `lseek`). `HEAD` reports the size with no body.
- `GET /chunked/{file}` — streamed with **`Transfer-Encoding: chunked`** (the
  length is never pre-declared); the wire shows `<hexlen>\r\n<data>\r\n…` and a
  terminating `0\r\n\r\n`.

Both respect write backpressure (EPOLLOUT) and keep-alive — the connection is
reusable after the stream finishes. A `Response` opts in via `stream_file(path)`
or `stream_file_chunked(path)`; the buffered `body` path is unchanged.

**Verified against FastAPI as an oracle:** identical status codes and bodies on
the happy path, identical `422` error *shape*, matching extra-field handling.

### HTTP/2 request bodies (h2_server.mojo)

The `h2c` server captures DATA frames into the request body, so `POST` works over
HTTP/2:

```bash
curl --http2-prior-knowledge -X POST http://localhost:8090/users \
     -H 'content-type: application/json' \
     -d '{"id":42,"name":"alice","active":true}'        # -> 201
curl --http2-prior-knowledge -X POST http://localhost:8090/echo \
     --data-binary @somefile                            # echoed byte-identical
```

Bodies larger than the initial flow-control window arrive across reads (nghttp2's
default auto `WINDOW_UPDATE`); verified byte-identical up to 500 KB.

## Build

```bash
BR=<brotli libdir>; NGH=<libnghttp2 pkg dir>
cc -c -fPIC -O2 http/cshim/http_shim.c -o http/cshim/http_shim.o
cc -c -fPIC -O2 -I$NGH/include net/cshim/h2_shim.c -o net/cshim/h2_shim.o
cc -c -fPIC -O2 -I/usr/include -I$NGH/include net/cshim/alpn_shim.c -o net/cshim/alpn_shim.o

# HTTP server
pixi run mojo build -I . examples/main.mojo -o mojohttp \
    -Xlinker http/cshim/http_shim.o -Xlinker -lz \
    -Xlinker $BR/libbrotlienc.so -Xlinker $BR/libbrotlicommon.so -Xlinker -rpath -Xlinker $BR

# HTTP client
pixi run mojo build -I . examples/client_main.mojo -o mojoget
./mojoget /health
```
The HTTPS and HTTP/2 servers add `-lssl -lcrypto` and the nghttp2 shim/lib — see
each server file's header comment for the exact line. Static files are served from
`examples/www/` (run from the repo root).
