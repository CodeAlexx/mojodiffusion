//! Phase-A seam proof: spawn the REAL Mojo CPU stub worker over the AF_UNIX
//! socketpair IPC, drive one job to completion, and assert the output PNG exists.
//!
//! This exercises the full `spawn_worker` → `next_event` (Ready) → `send_start`
//! → `next_event*` (Progress…Done) loop against
//! `/home/alex/mojodiffusion/output/bin/serenity_worker_stub`, the same binary the
//! Rust control plane drives in production. No Mojo build is required here — the
//! stub binary already exists (CPU-only).
//!
//! Run:  cargo test -p serenity-ipc -- --nocapture

use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use serenity_ipc::{spawn_worker, WorkerHandle};
use serenity_wire::{JobParams, WorkerEvent};

const STUB_BIN: &str = "/home/alex/mojodiffusion/output/bin/serenity_worker_stub";

/// Poll `next_event` with a short sleep until it yields an event or `deadline`
/// passes. `Ok(None)` (EAGAIN, nothing ready) just means "keep polling".
fn poll_event(h: &mut WorkerHandle, deadline: Instant) -> WorkerEvent {
    loop {
        match h.next_event() {
            Ok(Some(ev)) => return ev,
            Ok(None) => {
                if Instant::now() >= deadline {
                    panic!("timed out waiting for a worker event");
                }
                sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("worker IPC error while polling: {e:#}"),
        }
    }
}

#[test]
fn stub_seam_runs_one_job_to_done() {
    let bin = Path::new(STUB_BIN);
    assert!(
        bin.exists(),
        "stub worker binary missing at {STUB_BIN} — Phase-A precondition"
    );

    // Unique output dir under the system temp dir.
    let out_dir: PathBuf = std::env::temp_dir().join(format!(
        "serenity_stub_seam_{}_{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    std::fs::create_dir_all(&out_dir).expect("create out_dir");

    // Spawn the stub worker: `serenity_worker_stub <child_fd>` (no kind arg).
    let mut h = spawn_worker(bin, &[]).expect("spawn_worker(serenity_worker_stub)");

    // 1. The worker emits {"ev":"ready"} immediately on startup.
    let ready_deadline = Instant::now() + Duration::from_secs(10);
    let first = poll_event(&mut h, ready_deadline);
    assert_eq!(
        first,
        WorkerEvent::Ready,
        "first event from worker must be Ready, got {first:?}"
    );
    println!("[seam] got Ready");

    // 2. Build a JobParams: small CPU stub job (6 steps @ 64x64).
    let job_id = "stub_seam_job".to_string();
    let mut p = JobParams::default();
    p.job_id = job_id.clone();
    p.model = "stub".to_string();
    p.prompt = "a phase-A seam smoke test".to_string();
    p.steps = 6;
    p.width = 64;
    p.height = 64;
    p.out_dir = out_dir.to_string_lossy().into_owned();

    h.send_start(&p).expect("send_start");
    println!("[seam] sent start: {}", p.to_start_line());

    // 3. Loop next_event until a terminal event. The stub sleeps ~100ms/step, so
    //    6 steps ≈ 0.6s; give generous headroom.
    let job_deadline = Instant::now() + Duration::from_secs(30);
    let mut last_step = 0i64;
    let final_ev = loop {
        let ev = poll_event(&mut h, job_deadline);
        match &ev {
            WorkerEvent::Ready => {
                // A stray ready after start would be unexpected, but harmless.
                println!("[seam] (stray) Ready");
            }
            WorkerEvent::Progress { step, total, .. } => {
                assert!(*step >= last_step, "progress steps must be monotonic");
                last_step = *step;
                println!("[seam] progress {step}/{total}");
            }
            WorkerEvent::Done { .. } | WorkerEvent::Failed { .. } | WorkerEvent::Cancelled => {
                break ev
            }
        }
    };

    // 4. Must end in Done, carrying the output path <out_dir>/<job_id>.png.
    let output_path = match &final_ev {
        WorkerEvent::Done { output_path } => output_path.clone(),
        other => panic!("job did not end in Done: {other:?}"),
    };
    println!("[seam] done: output_path = {output_path}");

    let expected = out_dir.join(format!("{job_id}.png"));
    assert_eq!(
        Path::new(&output_path),
        expected.as_path(),
        "done output_path must be <out_dir>/<job_id>.png"
    );

    // 5. The PNG must exist and be non-empty.
    let meta = std::fs::metadata(&output_path)
        .unwrap_or_else(|e| panic!("output PNG {output_path} not found: {e}"));
    assert!(meta.is_file(), "output_path must be a regular file");
    assert!(meta.len() > 0, "output PNG must be non-empty");
    println!("[seam] PNG exists, {} bytes — seam proven", meta.len());

    // Confirm the PNG signature for good measure.
    let head = std::fs::read(&output_path).expect("read PNG");
    assert_eq!(
        &head[..8],
        &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
        "output file must have a PNG magic header"
    );

    // 6. Clean shutdown: SIGKILL + reap (the OS reclaims the child).
    h.kill();

    // Best-effort cleanup of the temp dir.
    let _ = std::fs::remove_dir_all(&out_dir);
}
