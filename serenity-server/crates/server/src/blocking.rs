//! blocking — run a request handler's body on tokio's blocking pool.
//!
//! Several handlers (`/v1/video`, `/v1/caption`, `/v1/h3/director`,
//! `/v1/magic_prompt`, `/enhance_prompt`) are pure blocking work: they spawn a
//! GPU subprocess and hold it for seconds to minutes. Declared `async fn` and
//! run inline, each one parked a runtime worker thread for its whole duration,
//! and a couple of them together could starve every other request -- the UI
//! "hung" while a caption ran. On the blocking pool they cost nothing the
//! runtime notices.
//!
//! The second job of this wrapper is panic containment: a panic inside the
//! body surfaces here as a `JoinError`, is logged with its message, and is
//! answered as a 500 with that message. Inline, it reset the connection and
//! left nothing in the log.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

pub(crate) async fn offload<F>(work: F) -> Response
where
    F: FnOnce() -> Response + Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(response) => response,
        Err(error) => {
            let detail = if error.is_panic() {
                let payload = error.into_panic();
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                tracing::error!(panic = %message, "request handler panicked on the blocking pool");
                format!("handler panicked: {message}")
            } else {
                tracing::error!("request handler task was cancelled before it finished");
                "handler task cancelled".to_string()
            };
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({ "detail": detail })),
            )
                .into_response()
        }
    }
}

// ---- Subprocesses with a deadline.
//
// `Command::output()` waits forever. A Mojo captioner or LLM that wedges --
// a stuck CUDA context, a model load that never returns -- therefore held its
// GPU lease forever, and every later request answered 409 until the server
// was restarted by hand. This runs the child the same way `output()` does
// (both pipes drained concurrently so a chatty child cannot block on a full
// pipe) and kills it at the deadline, which releases the lease when the
// handler returns.

use std::io::Read;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Wall-clock limit for one GPU subprocess. Generous on purpose: a first-ever
/// magic-prompt run builds a prefix cache for minutes, and a legitimate run
/// must never be killed. `SERENITY_SUBPROCESS_TIMEOUT_SECS` overrides it.
pub(crate) fn subprocess_deadline() -> Duration {
    std::env::var("SERENITY_SUBPROCESS_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(900))
}

pub(crate) trait CommandDeadline {
    fn output_with_deadline(&mut self, deadline: Duration) -> std::io::Result<Output>;
}

impl CommandDeadline for Command {
    fn output_with_deadline(&mut self, deadline: Duration) -> std::io::Result<Output> {
        // Its own process group, so the deadline can kill everything the child
        // started and not just the child. The magic-prompt path spawns a shell
        // script that forks the real GPU process; killing the shell alone left
        // that process alive, holding both the GPU and the pipe -- and the
        // pipe drain below then waited for it anyway. The unit test for this
        // is the forking shape, on purpose.
        use std::os::unix::process::CommandExt;
        self.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        let mut child = self.spawn()?;
        let drain = |pipe: Option<Box<dyn Read + Send>>| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                if let Some(mut pipe) = pipe {
                    let _ = pipe.read_to_end(&mut bytes);
                }
                bytes
            })
        };
        let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
        let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= deadline {
                // The group id is the child's pid (process_group(0) above).
                kill_process_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout.join();
                let _ = stderr.join();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "subprocess exceeded the {}s deadline and was killed",
                        deadline.as_secs()
                    ),
                ));
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        Ok(Output {
            status,
            stdout: stdout.join().unwrap_or_default(),
            stderr: stderr.join().unwrap_or_default(),
        })
    }
}

/// SIGKILL a whole process group by its leader's pid, through kill(2)
/// directly.
///
/// This used to shell out as `/usr/bin/kill -KILL -<pid>`. procps kill
/// (4.0.4 here) reads "-1696300" as an option cluster and keeps its first
/// digit, so the syscall it issued was kill(-1, SIGKILL): every process of
/// the user, the session manager included. Any child whose pid began with 1
/// took the desktop down with it. That happened three times (2026-09-04
/// 20:20 and 20:23, 2026-09-05 07:22), each from this crate's own unit test,
/// and was misread as an out-of-memory kill each time. A shell proof of the
/// same command line passed because bash's builtin kill parses it correctly;
/// the binary does not. Never route a signal through argv again.
fn kill_process_group(leader: u32) {
    // 0 would be the caller's own group and 1 is init; negating either is
    // the catastrophe above, so refuse both rather than trust the caller.
    if leader <= 1 {
        return;
    }
    let Ok(pid) = libc::pid_t::try_from(leader) else {
        return;
    };
    // SAFETY: kill(2) with a negative pid signals exactly that process
    // group; the arguments are plain integers and the call has no other
    // memory effects.
    unsafe {
        libc::kill(-pid, libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_finished_child_returns_its_output() {
        let out = Command::new("sh")
            .arg("-c")
            .arg("printf hello; printf oops >&2")
            .output_with_deadline(Duration::from_secs(5))
            .expect("runs");
        assert!(out.status.success());
        assert_eq!(out.stdout, b"hello");
        assert_eq!(out.stderr, b"oops");
    }

    #[test]
    fn a_wedged_child_is_killed_at_the_deadline() {
        let started = Instant::now();
        let error = Command::new("sh")
            .arg("-c")
            .arg("sleep 30; true")
            .output_with_deadline(Duration::from_secs(1))
            .expect_err("must not wait thirty seconds");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(error.to_string().contains("1s deadline"), "{error}");
        // Killed promptly, not after the child's own thirty seconds.
        assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
    }

    #[test]
    fn a_chatty_child_cannot_block_on_a_full_pipe() {
        // 4 MiB on each pipe is well past the kernel buffer; without concurrent
        // draining this deadlocks and then times out.
        let out = Command::new("sh")
            .arg("-c")
            .arg("head -c 4194304 /dev/zero; head -c 4194304 /dev/zero >&2")
            .output_with_deadline(Duration::from_secs(20))
            .expect("drains both pipes");
        assert_eq!(out.stdout.len(), 4 * 1024 * 1024);
        assert_eq!(out.stderr.len(), 4 * 1024 * 1024);
    }

    #[test]
    fn the_deadline_is_configurable_and_fails_safe() {
        std::env::set_var("SERENITY_SUBPROCESS_TIMEOUT_SECS", "42");
        assert_eq!(subprocess_deadline(), Duration::from_secs(42));
        std::env::set_var("SERENITY_SUBPROCESS_TIMEOUT_SECS", "nonsense");
        assert_eq!(subprocess_deadline(), Duration::from_secs(900));
        std::env::remove_var("SERENITY_SUBPROCESS_TIMEOUT_SECS");
    }
}
