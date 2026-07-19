# async — coroutine event-loop executor

Mojo has **native `async`/`await`** and a built-in `Coroutine` type (the
compiler-generated suspension machinery — the genuinely hard part — is there).
What it lacks is an executor that drives coroutines off real I/O. This is that
executor — and `serve_conn` now drives the **real `http/request` parser**, so it
is a genuine async HTTP/1.1 server, not a single-recv demo.

`executor.mojo` runs **one coroutine per connection** on [`net`](../net/)'s epoll
loop:

1. A task `await`s the reusable `await_readable(parked, fd)` awaitable, which calls
   `_suspend_async` to **park** — handing its coroutine handle to the scheduler
   (the `parked` array, keyed by fd) — then returns control to the loop.
2. The loop registers the fd with `epoll`; when the socket is readable it
   **resumes** the parked coroutine (`_coro_resume_fn`), which cascades back
   through the awaitable into the task.
3. Tasks are **multi-suspend**: `serve_conn` loops, awaiting between requests, so a
   single coroutine handles **keep-alive** over the whole connection. When it
   returns it sets a `done` flag and the loop frees the coroutine + closes the fd.

This is the asyncio/uvloop model in pure Mojo.

## Real HTTP/1.1, not a single-recv demo

`serve_conn` accumulates bytes across recvs/suspends into a per-connection buffer
and only replies once [`http/request`](../http/) reports a **complete** message
(`Content-Length` / chunked framed), then slices that request off and keeps the
rest. Verified against a raw-socket client (7/7):

- keep-alive: two requests on one connection (same fd, suspending between);
- `POST` body echoed **byte-identical** (Content-Length framing);
- a request **split across two recvs** (headers, then body 0.4 s later) is
  buffered until complete, then served — the old single-recv code would have
  replied to the partial;
- **pipelined** requests (two in one send) both served;
- a **200 KB** body (> `READ_CHUNK`) accumulated across recvs and echoed
  byte-identical;
- `Connection: close` honored.

```bash
pixi run mojo build -I . async/executor.mojo -o /tmp/asyncsrv
/tmp/asyncsrv      # real async HTTP/1.1 on :8095  (GET/, /health, POST echo)
```

## Coroutine-frame constraints (measured on this toolchain)

The coroutine lowering in 1.0.0b1 is narrow about what may live in / cross a
coroutine frame. Measured, with minimal repros:

- A `String` (non-trivial type) **survives** an `await` suspend — fine.
- An `UnsafePointer` held **across** an `await` mis-lowers the frame — so the
  recv scratch buffer is allocated *inside* each wake and freed before the next
  `await`; only the `String` buffer + ints cross the suspend.
- A `try`/`except` (or a *raising* call) inside the `async def` body mis-lowers
  it — so all parsing/serialization (which can raise) lives in a plain `def`
  (`process_one`) and the async body only makes **return-by-value** calls.

These are toolchain limits, not design walls; they shape how `serve_conn` is
written and are why keep-alive state stays trivial.

## The primitives (from `std.builtin.coroutine`)

These are **unstable, underscore-prefixed stdlib internals** — pinned to
**Mojo 1.0.0b1 / MAX 26.3** and the most likely thing here to break on a bump.
In `executor.mojo` they're confined to one documented block (with `resume`/
`destroy` wrappers) so a future break is a one-file fix. (A separate shim *module*
isn't possible: `async` is a reserved keyword, so this directory can't be imported
as a package, and a top-level file can't use relative imports.)

- `_suspend_async[body]()` — suspend the current coroutine; `body(handle)` stashes
  the handle for the scheduler.
- `_coro_resume_fn(handle)` — resume a stashed coroutine from outside.
- `_coro_destroy_fn(handle)` — free a finished coroutine frame.
- `Coroutine._set_noop_callback()` / `_take_handle()` — required before driving
  a coroutine externally (used only at the spawn site).

## Status

Working executor with a reusable `await_readable` awaitable, multi-suspend tasks
(keep-alive), and lifecycle reaping. Still single-thread and read-only awaitables;
natural extensions (an `await_writable` for backpressure, timeouts, integrating
the `http` request parser into `serve_conn` so it speaks real HTTP — see Scope
above) are ordinary work on these proven primitives — no language gap.
