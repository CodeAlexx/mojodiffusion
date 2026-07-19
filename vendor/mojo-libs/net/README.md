# net — sockets, epoll, TLS, HTTP/2 transport for Mojo

Mojo has no networking. This is a `std::net`-style library built directly on libc
(and OpenSSL / nghttp2) via FFI. Linux x86-64.

## Modules

| Module | What it is |
|---|---|
| `syscalls.mojo` | libc externs + Linux x86-64 constants: `socket/bind/listen/accept/connect/recv/send/close/setsockopt/fcntl`, `errno`→`strerror`. |
| `socket.mojo` | `Socket` — owns an fd, closes it on drop (RAII), non-Copyable so the fd can't be double-closed. |
| `tcp.mojo` | `TCPListener` (bind+listen+accept), `TCPStream` (read/write), `tcp_connect(ip,port)`. Hand-packed `sockaddr_in`. |
| `poll.mojo` | `Epoll` — `epoll_create1/ctl/wait` wrapper. **The 12-byte packed `epoll_event` layout is verified empirically.** Drives single-thread concurrency. |
| `tls.mojo` | TLS 1.3 via OpenSSL FFI — **server**: context + cert/key, handshake, ALPN `h2`; **client**: `make_client_ctx`/`client_connect` (handshake + SNI + optional CA-chain & hostname verification via the system CA store); shared encrypted read/write (`WANT_READ/WANT_WRITE` aware). Used by `http.client` for `https://`. |
| `h2.mojo` | HTTP/2 bindings to nghttp2 — Mojo shuttles bytes to/from a C session; pops completed requests (`:method`, `:path`, **and the request body**). |
| `signals.mojo` | `install_signal_fd()` — block SIGINT/SIGTERM and deliver them as a pollable `signalfd` (no C-ABI handler), for graceful shutdown on the epoll loop. |

## C shims (`cshim/`)

nghttp2 and OpenSSL ALPN are **callback-driven**, and Mojo can't hand a C library
a C-ABI callback. So those callbacks live in tiny C shims; Mojo calls in:

- `h2_shim.c` — owns the nghttp2 server session (preface, HPACK, framing, streams)
  and exposes a byte-shuttle API (`h2_new/h2_recv/h2_next_request/h2_request_body/
  h2_respond/h2_send`). Accumulates request-body DATA frames per stream (nghttp2's
  auto `WINDOW_UPDATE` handles bodies past the initial flow-control window).
- `alpn_shim.c` — the `SSL_CTX_set_alpn_select_cb` callback (selects `h2` via
  nghttp2's helper) + `tls_is_h2`.

```bash
cc -c -fPIC -O2 -I<nghttp2>/include          net/cshim/h2_shim.c   -o net/cshim/h2_shim.o
cc -c -fPIC -O2 -I/usr/include -I<nghttp2>/include net/cshim/alpn_shim.c -o net/cshim/alpn_shim.o
```
Link: `-Xlinker net/cshim/h2_shim.o -Xlinker <nghttp2>/lib/libnghttp2.so -Xlinker -rpath -Xlinker <nghttp2>/lib`,
and for TLS `-Xlinker -lssl -Xlinker -lcrypto`.

## Example

```mojo
from net.tcp import TCPListener
var listener = TCPListener(8080)
var stream = listener.accept()      # blocking
var req = stream.read()
stream.write_all(String("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi"))
```

For concurrency use `poll.Epoll` (see `examples/main.mojo`): one non-blocking
listen socket, edge-triggered, with per-connection read buffers.

## Notes / gotchas (measured)

- `struct epoll_event` is **12 bytes packed** on x86-64 (not 16) — events@0, data@4.
- Mojo destroys values at their **last use**, not scope end — keep a `TCPListener`/
  `Epoll`/`Socket` referenced inside the loop or its fd gets closed early.
- TLS handshake runs blocking right after `accept()` (brief); reads/writes are
  non-blocking on the epoll loop afterward.
