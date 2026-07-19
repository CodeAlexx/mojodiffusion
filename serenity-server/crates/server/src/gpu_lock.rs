//! gpu_lock — the CROSS-PATH single-GPU lease (audit item L3, Phase B2).
//!
//! The one-in-flight serialization in main.rs only covers the IPC /v1/generate
//! driver thread. The subprocess GPU paths (/v1/video ltx2+wan22, /v1/caption,
//! /v1/magic_prompt) spawn their own GPU processes and used to bypass it, so a
//! 9-minute video render could co-run with a t2i denoise and OOM the 16GB card.
//!
//! Design (admission-reject, deadlock-free — mirrors the in_flight pattern; no
//! std-thread ever blocks on an async mutex):
//!   * `GpuOwner` = Arc<Mutex<Option<GpuLease>>> stored in AppState.
//!   * `try_acquire(owner, kind, id)` -> Ok(RAII `GpuGuard`) or Err(current
//!     lease snapshot). Guard Drop clears the lease, so a panicking handler
//!     still releases.
//!   * Subprocess handlers (video/caption/magic) acquire at the top and REJECT
//!     with 409 `gpu_busy_conflict_report` when held.
//!   * The generate driver acquires at promote time and simply does not promote
//!     while a subprocess holds the GPU — queued jobs stay queued (generate is
//!     already async via the JobBook), never rejected.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;

/// Monotonic tag counter so every subprocess acquisition gets a unique lease
/// id (guards only release the lease they created — see GpuGuard::drop).
static TAG: AtomicU64 = AtomicU64::new(1);

/// "caption-7"-style unique lease id for a subprocess request.
pub(crate) fn next_tag(kind: &str) -> String {
    format!("{kind}-{}", TAG.fetch_add(1, Ordering::Relaxed))
}

/// Who owns the GPU right now.
#[derive(Clone, Debug)]
pub(crate) struct GpuLease {
    /// The path kind: "generate" | "video" | "caption" | "magic".
    pub(crate) kind: &'static str,
    /// Job id / request tag for the 409 body and logs.
    pub(crate) id: String,
    pub(crate) since: Instant,
}

pub(crate) type GpuOwner = Arc<Mutex<Option<GpuLease>>>;

pub(crate) fn new_owner() -> GpuOwner {
    Arc::new(Mutex::new(None))
}

/// RAII lease guard: dropping it releases the GPU. Only the holder that set
/// the lease drops it (each guard remembers its own id), so a poisoned or
/// stale drop can never release someone else's lease.
#[derive(Debug)]
pub(crate) struct GpuGuard {
    owner: GpuOwner,
    id: String,
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        if let Ok(mut o) = self.owner.lock() {
            let ours = o.as_ref().map(|l| l.id == self.id).unwrap_or(false);
            if ours {
                *o = None;
            }
        }
    }
}

/// Try to take the GPU for (kind, id). Err returns a snapshot of the current
/// lease for the 409 body.
pub(crate) fn try_acquire(
    owner: &GpuOwner,
    kind: &'static str,
    id: &str,
) -> Result<GpuGuard, GpuLease> {
    let mut o = owner.lock().expect("gpu_owner mutex poisoned");
    if let Some(cur) = o.as_ref() {
        return Err(cur.clone());
    }
    *o = Some(GpuLease {
        kind,
        id: id.to_string(),
        since: Instant::now(),
    });
    Ok(GpuGuard {
        owner: owner.clone(),
        id: id.to_string(),
    })
}

/// The 409 body for a busy GPU — one schema for every subprocess path.
pub(crate) fn gpu_busy_conflict_report(requested_kind: &str, cur: &GpuLease) -> serde_json::Value {
    json!({
        "schema": "serenity.gpu_busy.v1",
        "error": "gpu busy: another job owns the GPU (single-GPU serialization)",
        "requested": requested_kind,
        "owner": {
            "kind": cur.kind,
            "id": cur.id,
            "held_seconds": cur.since.elapsed().as_secs(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_cycle() {
        let owner = new_owner();
        let g = try_acquire(&owner, "generate", "job-0001").expect("free GPU acquires");
        // Second acquire of any kind is rejected with the current lease.
        let e = try_acquire(&owner, "caption", "req-1").unwrap_err();
        assert_eq!(e.kind, "generate");
        assert_eq!(e.id, "job-0001");
        drop(g);
        // Released on drop: next acquire succeeds.
        let g2 = try_acquire(&owner, "video", "vid-1").expect("released after drop");
        drop(g2);
    }

    #[test]
    fn guard_only_releases_its_own_lease() {
        let owner = new_owner();
        let g1 = try_acquire(&owner, "caption", "a").unwrap();
        // Simulate a stale guard from a previous lease with a different id.
        let stale = GpuGuard {
            owner: owner.clone(),
            id: "not-a".to_string(),
        };
        drop(stale);
        assert!(
            try_acquire(&owner, "video", "b").is_err(),
            "stale guard must not release the live lease"
        );
        drop(g1);
        assert!(try_acquire(&owner, "video", "b").is_ok());
    }

    #[test]
    fn conflict_report_shape() {
        let owner = new_owner();
        let _g = try_acquire(&owner, "video", "vid-9").unwrap();
        let cur = owner.lock().unwrap().clone().unwrap();
        let body = gpu_busy_conflict_report("caption", &cur);
        assert_eq!(body["schema"], "serenity.gpu_busy.v1");
        assert_eq!(body["requested"], "caption");
        assert_eq!(body["owner"]["kind"], "video");
        assert_eq!(body["owner"]["id"], "vid-9");
    }
}
