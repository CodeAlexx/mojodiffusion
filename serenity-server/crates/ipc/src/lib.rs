//! serenity-ipc — parent-side driver that spawns and talks to a Mojo inference
//! worker over an AF_UNIX socketpair with newline-framed JSON.
//!
//! This is the Rust re-host of `serenitymojo/serve/proc_ipc.mojo` (socketpair /
//! fork+exec / fd inheritance / LineReader) and the parent half of
//! `process_isolated_backend.mojo` (_spawn_child / step / cancel / _kill_child).
//!
//! SPAWN CONTRACT (must match proc_ipc.mojo exactly):
//!   1. socketpair(AF_UNIX, SOCK_STREAM, 0) -> (parent_end, child_end).
//!   2. The CHILD inherits `child_end` ACROSS exec — so child_end must NOT be
//!      CLOEXEC. The child is told its fd number as the LAST argv (decimal string);
//!      the Mojo worker reads `Int32(Int(argv[last]))` and uses that exact number.
//!   3. fork+exec `worker_bin <pre_args...> <child_end_fd_number>`.
//!      Between fork and exec: async-signal-safe only (Command/pre_exec handles this).
//!   4. PARENT: close child_end, set parent_end O_NONBLOCK, keep it.
//!   5. The worker emits {"ev":"ready"} immediately; then one job per start line.
//!
//! The public API below is FROZEN for Phase A — the server crate codes against it.
//! Implement the bodies; do not change the signatures (document needed changes
//! instead). Run `cargo build -p serenity-ipc`; for an end-to-end smoke, drive the
//! real `output/bin/serenity_worker_stub` (it exists, CPU-only).
//! NEVER run `mojo build` / `pixi run build-*` — that OOM-kills the desktop.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command};

use anyhow::{anyhow, Context, Result};
use nix::fcntl::{fcntl, FcntlArg, OFlag};
use nix::sys::socket::{socketpair, AddressFamily, SockFlag, SockType};
use serenity_wire::{JobParams, WorkerEvent, CANCEL_LINE};

const MOJO_SYNC_MODE_ENV: &str = "MODULAR_DEVICE_CONTEXT_SYNC_MODE";
const CUDA_RAM_CACHE_DIR: &str = "/dev/shm/serenity-image-cuda-cache";
const CUDA_CACHE_MAXSIZE_BYTES: &str = "1073741824";

fn worker_requires_sync_mode(worker_bin: &Path) -> bool {
    worker_bin
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "serenity_worker_zimage")
}

/// Outcome of a single non-blocking poll for a worker event.
///
/// This is the TYPED form of [`WorkerHandle::next_event`]: it makes the
/// **clean peer-close** (the worker process exited — `recv() == 0`) a first-class
/// variant (`PeerClosed`) that the driver can react to deterministically, instead
/// of overloading the generic `Err` channel for both "worker is gone" and "a real
/// IPC fault occurred".
///
/// Why this matters (mirrors `process_isolated_backend.step()`): when the worker
/// exits — crash, OOM-kill, or its own `exit()` — the parent must NOT tear the
/// server down. It must mark the in-flight job FAILED and respawn the worker
/// lazily on the next job. A `PeerClosed` is the expected, recoverable signal for
/// that; a `Err` is reserved for genuine faults (recv error other than EAGAIN, a
/// malformed event line, etc.). Both are non-fatal to the server, but only
/// `PeerClosed` should trigger a lazy respawn — an `Err` from a malformed line
/// leaves the (still-running) child in place.
#[derive(Debug)]
pub enum EventPoll {
    /// A complete worker event was decoded this tick.
    Event(WorkerEvent),
    /// No data was ready this tick (EAGAIN / EINTR). Caller should sleep & retry.
    Idle,
    /// The worker closed its end of the socket (`recv() == 0`): the child has
    /// exited. The in-flight job (if any) must be failed and the worker respawned
    /// lazily. This is NOT a server-fatal condition.
    PeerClosed,
}

/// A live worker child + our end of the socket. Non-blocking line reader.
///
/// Mirrors `process_isolated_backend.mojo`'s state for one resident child:
/// the spawned `Child`, the parent-end `OwnedFd` (set O_NONBLOCK), and a byte
/// buffer that frames newline-delimited JSON lines like `proc_ipc.LineReader`.
pub struct WorkerHandle {
    child: Child,
    /// Our (parent) end of the AF_UNIX socketpair, marked O_NONBLOCK. Owned so it
    /// is closed on drop (the Mojo `_kill_child` closes parent_fd after reaping).
    parent_fd: OwnedFd,
    /// Pending bytes received but not yet consumed as a complete '\n' line —
    /// the equivalent of `LineReader.buf`.
    buf: Vec<u8>,
    /// True until a recv() returned 0 (peer closed). Once false, the child is gone.
    peer_open: bool,
}

/// Spawn `worker_bin <pre_args...> <child_fd>`, returning a driveable handle.
/// For the Phase-A stub: `spawn_worker("output/bin/serenity_worker_stub", &[])`.
/// (A future combined worker would use `pre_args=&["zimage"]` etc.)
///
/// Mirrors `proc_ipc.make_socketpair` + `process_isolated_backend._spawn_child`:
/// the child inherits `child_end` ACROSS exec at the SAME fd number we pass as the
/// last argv (so FD_CLOEXEC must be cleared on it), and the parent closes its copy
/// of `child_end` and marks `parent_end` O_NONBLOCK.
pub fn spawn_worker(
    worker_bin: &Path,
    pre_args: &[&str],
    environment: &[(std::ffi::OsString, std::ffi::OsString)],
) -> Result<WorkerHandle> {
    // 1. socketpair(AF_UNIX, SOCK_STREAM, 0) -> (parent_end, child_end).
    // No SOCK_CLOEXEC: the default (no CLOEXEC) is what we want on child_end so it
    // survives exec; we belt-and-braces clear FD_CLOEXEC in pre_exec below.
    let (parent_fd, child_fd) = socketpair(
        AddressFamily::Unix,
        SockType::Stream,
        None,
        SockFlag::empty(),
    )
    .context("socketpair(AF_UNIX, SOCK_STREAM) failed")?;

    // The decimal fd number the child must use — identical to the number we hand it
    // as the last argv (the Mojo worker does `Int32(Int(argv[last]))`).
    let child_fd_raw: RawFd = child_fd.as_raw_fd();
    let child_fd_arg = child_fd_raw.to_string();

    // 2. PARENT: set parent_end O_NONBLOCK (so next_event never blocks).
    set_nonblock(parent_fd.as_raw_fd()).context("set parent_end O_NONBLOCK")?;

    // 3. Build the Command BEFORE spawn: pre_args..., then the child fd number last.
    //    e.g. `serenity_worker_stub <child_fd>`.
    let mut cmd = Command::new(worker_bin);
    cmd.envs(environment.iter().cloned());
    // Product workers must not fault CUDA kernel modules from disk after their
    // exact step-0 denoise boundary.  Eager module loading moves that one-time
    // driver work into worker startup, before Ready/model activation, instead
    // of letting the first GEMM trigger physical reads during sampling.
    cmd.env("CUDA_MODULE_LOADING", "EAGER");
    // Resolve dynamic symbols before Ready as well.  Without this, the first
    // model GEMM can fault a few remaining shared-library pages after step 0.
    cmd.env("LD_BIND_NOW", "1");
    // The CUDA driver defaults its JIT cache to ~/.nv/ComputeCache. A measured
    // Krea job revisited that disk-backed index at step 7/8 (20,480 physical
    // bytes), so the fail-closed denoise gate correctly terminated the worker.
    // Keep JIT reuse, but put the cache on tmpfs so cache lookups can never hit
    // the SSD. If tmpfs setup fails, disable the cache rather than falling back
    // to the driver's disk default. Force the JIT support libraries to load at
    // worker initialization as an additional pre-step boundary guarantee.
    if std::fs::create_dir_all(CUDA_RAM_CACHE_DIR).is_ok() {
        cmd.env("CUDA_CACHE_PATH", CUDA_RAM_CACHE_DIR);
        cmd.env("CUDA_CACHE_MAXSIZE", CUDA_CACHE_MAXSIZE_BYTES);
    } else {
        cmd.env("CUDA_CACHE_DISABLE", "1");
    }
    cmd.env("CUDA_FORCE_PRELOAD_LIBRARIES", "1");
    // Measured on the 16 GiB product card: ZImage completes denoise in either
    // mode, but the async DeviceContext allocator can OOM while constructing
    // the already-shipped 64-grid VAE decoder (jobs 0105/0106). The identical
    // request completes in sync mode (job 0107, 14,482 MiB peak). Apply this at
    // the worker boundary so every server launch and model swap inherits the
    // proven setting; do not depend on a shell-specific environment variable.
    if worker_requires_sync_mode(worker_bin) {
        cmd.env(MOJO_SYNC_MODE_ENV, "true");
    }
    for a in pre_args {
        cmd.arg(a);
    }
    cmd.arg(&child_fd_arg);

    // pre_exec runs in the forked child between fork() and exec() — async-signal-safe
    // ONLY. We clear FD_CLOEXEC on child_fd so it survives exec at the same number.
    // std::process::Command would otherwise leave an unknown inherited fd as-is, but
    // the OwnedFd we created may carry CLOEXEC depending on platform defaults, so we
    // force it clear. fcntl(F_GETFD/F_SETFD) is async-signal-safe.
    //
    // NOTE: we intentionally do NOT close parent_fd in the child here. parent_fd is
    // marked O_NONBLOCK but not CLOEXEC; to faithfully match the Mojo child (which
    // does `sys_close(parent_end)` before execv) we close it in pre_exec. Closing an
    // fd is async-signal-safe.
    let parent_fd_raw: RawFd = parent_fd.as_raw_fd();
    unsafe {
        cmd.pre_exec(move || {
            // Close the parent end in the child (matches `_ = sys_close(parent_end)`).
            // EBADF is fine if it was never inherited; ignore the result.
            libc::close(parent_fd_raw);
            // Clear FD_CLOEXEC on the child socket fd so it survives exec.
            let flags = libc::fcntl(child_fd_raw, libc::F_GETFD);
            if flags < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let cleared = flags & !libc::FD_CLOEXEC;
            if libc::fcntl(child_fd_raw, libc::F_SETFD, cleared) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .with_context(|| format!("spawn worker binary {}", worker_bin.display()))?;

    // 4. PARENT: close child_end (our copy). Dropping the OwnedFd closes it; this is
    //    the parent's `_ = sys_close(child_end)` after fork.
    drop(child_fd);

    Ok(WorkerHandle {
        child,
        parent_fd,
        buf: Vec::with_capacity(8192),
        peer_open: true,
    })
}

impl WorkerHandle {
    /// OS process id of the live worker. The server uses this for read-only
    /// `/proc/<pid>/io` hardware-safety accounting during denoise.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Send `{"cmd":"start", ...}\n` (JobParams::to_start_line + newline frame).
    /// Mirrors `proc_ipc.write_msg` (append '\n', send with MSG_NOSIGNAL, loop
    /// until all bytes written).
    pub fn send_start(&mut self, p: &JobParams) -> Result<()> {
        let line = p.to_start_line();
        self.write_line(&line)
    }

    /// Send `{"cmd":"cancel"}\n` (serenity_wire::CANCEL_LINE).
    pub fn send_cancel(&mut self) -> Result<()> {
        self.write_line(CANCEL_LINE)
    }

    /// Release a worker that fenced itself at the exact final denoise boundary.
    /// The server sends this only after the physical-read gate samples `/proc`.
    pub fn send_sampling_ack(&mut self) -> Result<()> {
        self.write_line(r#"{"cmd":"sampling_ack"}"#)
    }

    /// Non-blocking poll for one worker event, with a TYPED outcome.
    ///
    /// This is the preferred entry point for the driver loop. It distinguishes:
    ///   * [`EventPoll::Event`]  — a complete event line was decoded.
    ///   * [`EventPoll::Idle`]   — no data this tick (EAGAIN / EINTR); sleep & retry.
    ///   * [`EventPoll::PeerClosed`] — the worker exited (`recv() == 0`). The driver
    ///     must fail the in-flight job and respawn the worker LAZILY; this is NOT
    ///     server-fatal. Mirrors `process_isolated_backend.step()` `still_open==False`.
    ///
    /// `Err` is reserved for REAL faults: a recv() error other than EAGAIN/EINTR, or
    /// a malformed event line that fails to parse. A parse error leaves the
    /// (still-running) child in place — only `PeerClosed` should trigger a respawn.
    ///
    /// Once a `PeerClosed` has been observed, every subsequent call also returns
    /// `PeerClosed` until the handle is replaced (the child is gone for good).
    pub fn next_event_poll(&mut self) -> Result<EventPoll> {
        // Drain a complete line already sitting in the buffer first (LineReader._drain_one).
        if let Some(line) = self.drain_one() {
            return Ok(EventPoll::Event(self.parse_line(&line)?));
        }
        if !self.peer_open {
            return Ok(EventPoll::PeerClosed);
        }
        // No complete line buffered — pull more bytes (LineReader.next_line recv path).
        let mut chunk = [0u8; 4096];
        let n = unsafe {
            libc::recv(
                self.parent_fd.as_raw_fd(),
                chunk.as_mut_ptr() as *mut libc::c_void,
                chunk.len(),
                0,
            )
        };
        if n == 0 {
            // recv == 0 -> peer closed (worker exited). Recoverable: caller respawns.
            self.peer_open = false;
            return Ok(EventPoll::PeerClosed);
        }
        if n < 0 {
            let err = std::io::Error::last_os_error();
            match err.raw_os_error() {
                // nonblocking: nothing ready right now (EAGAIN == EWOULDBLOCK on Linux).
                Some(libc::EAGAIN) => return Ok(EventPoll::Idle),
                // a signal interrupted recv — treat as "nothing ready this tick".
                Some(libc::EINTR) => return Ok(EventPoll::Idle),
                _ => return Err(anyhow!("recv from worker failed: {err}")),
            }
        }
        self.buf.extend_from_slice(&chunk[..n as usize]);
        // Try once more to drain a now-complete line.
        match self.drain_one() {
            Some(line) => Ok(EventPoll::Event(self.parse_line(&line)?)),
            None => Ok(EventPoll::Idle),
        }
    }

    /// Non-blocking. Ok(Some(ev)) if a complete line was ready; Ok(None) if no data
    /// yet (EAGAIN); Err on peer-close (recv==0) or a real error. Mirrors
    /// proc_ipc.mojo::LineReader.next_line + decode_ev.
    ///
    /// NOTE: this collapses the recoverable **clean peer-close** (the worker exited)
    /// and a genuine IPC fault into the same `Err` channel, so the caller cannot tell
    /// "worker is gone, respawn it" from "real fault". Prefer [`Self::next_event_poll`]
    /// (typed [`EventPoll::PeerClosed`]) in the driver loop; this shim is kept for the
    /// handshake path and back-compat.
    pub fn next_event(&mut self) -> Result<Option<WorkerEvent>> {
        match self.next_event_poll()? {
            EventPoll::Event(ev) => Ok(Some(ev)),
            EventPoll::Idle => Ok(None),
            EventPoll::PeerClosed => Err(anyhow!("worker process exited (peer closed)")),
        }
    }

    /// SIGKILL + reap (proc_ipc.mojo::proc_kill_wait) — the OS reclaims the child's
    /// VRAM. Best-effort, never panics. Mirrors `_kill_child` (SIGKILL, blocking
    /// reap via waitpid, then close parent_fd).
    pub fn kill(&mut self) {
        // SIGKILL (not SIGTERM) + blocking reap: the OS reclaims the child's memory
        // on exit regardless of graceful cleanup. std::process::Child::kill sends
        // SIGKILL; wait() reaps it so no zombie remains.
        let _ = self.child.kill();
        let _ = self.child.wait();
        // The parent_fd OwnedFd is closed when the handle is dropped; nothing else
        // to do here (mirrors `_ = sys_close(parent_fd)` on _kill_child).
    }

    // ── internals (mirror proc_ipc helpers) ──────────────────────────────────

    /// write_msg: append '\n', send-all with MSG_NOSIGNAL (a dead peer yields an
    /// error return, not a SIGPIPE).
    fn write_line(&mut self, line: &str) -> Result<()> {
        let mut framed = String::with_capacity(line.len() + 1);
        framed.push_str(line);
        framed.push('\n');
        let bytes = framed.as_bytes();
        let mut sent: usize = 0;
        let fd = self.parent_fd.as_raw_fd();
        while sent < bytes.len() {
            let w = unsafe {
                libc::send(
                    fd,
                    bytes[sent..].as_ptr() as *const libc::c_void,
                    bytes.len() - sent,
                    libc::MSG_NOSIGNAL,
                )
            };
            if w < 0 {
                let err = std::io::Error::last_os_error();
                match err.raw_os_error() {
                    Some(libc::EINTR) => continue,
                    // O_NONBLOCK socket buffer full — spin until it drains. The
                    // control lines are tiny so this is effectively immediate.
                    Some(libc::EAGAIN) => continue,
                    _ => return Err(anyhow!("write_line: send failed/peer-closed: {err}")),
                }
            }
            if w == 0 {
                return Err(anyhow!("write_line: send returned 0 (peer closed)"));
            }
            sent += w as usize;
        }
        Ok(())
    }

    /// LineReader._drain_one: pop one complete '\n'-terminated line (without the
    /// '\n') from the buffer, or None if there isn't one yet.
    fn drain_one(&mut self) -> Option<String> {
        let nl = self.buf.iter().position(|&b| b == b'\n')?;
        let line: Vec<u8> = self.buf.drain(..=nl).collect();
        // line includes the trailing '\n'; strip it for the JSON body.
        let body = &line[..line.len() - 1];
        Some(String::from_utf8_lossy(body).into_owned())
    }

    /// Parse a received JSON line into a WorkerEvent (decode_ev equivalent).
    fn parse_line(&self, line: &str) -> Result<WorkerEvent> {
        WorkerEvent::parse(line).with_context(|| format!("parse worker event: {line:?}"))
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Best-effort reap so we never leak a zombie if the caller forgot to kill().
        // try_wait avoids blocking if the child is already gone.
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// set_nonblock: mark `fd` O_NONBLOCK (proc_ipc.set_nonblock) so the parent's
/// next_event recv never blocks.
fn set_nonblock(fd: RawFd) -> Result<()> {
    let flags = fcntl(fd, FcntlArg::F_GETFL).context("fcntl(F_GETFL)")?;
    let mut oflag = OFlag::from_bits_truncate(flags);
    oflag.insert(OFlag::O_NONBLOCK);
    fcntl(fd, FcntlArg::F_SETFL(oflag)).context("fcntl(F_SETFL O_NONBLOCK)")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::worker_requires_sync_mode;
    use std::path::Path;

    #[test]
    fn zimage_worker_uses_measured_safe_device_context_mode() {
        assert!(worker_requires_sync_mode(Path::new(
            "/tmp/serenity_worker_zimage"
        )));
        assert!(!worker_requires_sync_mode(Path::new(
            "/tmp/serenity_worker_krea2"
        )));
    }
}
