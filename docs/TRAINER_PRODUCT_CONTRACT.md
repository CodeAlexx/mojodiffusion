# Trainer Product Contract

Status: binding for all trainer, control-plane, and UI work in this repository.

## User contract

A user provides product inputs: model, dataset, output name, training method,
recipe values, and optional resume selection. The product performs every
derived stage. It must not require a user or external assistant to manufacture
a configuration, cache, command line, directory link, or intermediate dataset.

Submitting a run returns a run identifier immediately after the workspace and
initial log are created. Validation and cache work happen inside that recorded
run so failures are visible on the Logs screen.

## Run workspace

The default output root is `<repository>/output`. An explicit product setting
may choose a different root, but all paths are resolved from that setting and
recorded in the run manifest. Source code contains no developer-home paths.

Every run owns exactly one directory:

```text
output/<run_id>/
  run.json
  status.json
  logs/
    train.log
  cache/
    manifest.json
    data...
  checkpoints/
    model...
    trainer-state...
  samples/
    ...
  commands.jsonl
```

No generated run input or output may live in a global temporary directory,
source tree `target` directory, dataset directory, or manually linked checkout.

`run.json` is immutable after admission except for explicitly versioned schema
migration. Runtime changes belong in `status.json`, append-only logs, checkpoint
metadata, and command records.

## Lifecycle

The shared lifecycle is:

```text
created
  -> validating
  -> caching | cache-ready
  -> loading
  -> training
  -> completed
```

Terminal alternatives are `failed`, `cancelled`, and `interrupted`. Every state
transition is persisted atomically before it is emitted to the UI. Restarting
the product reconstructs state from disk rather than memory.

A validation or caching failure still has a run identifier, status record, and
log. Preflight must never reject a request solely because a trainer-generated
configuration or cache does not exist yet.

## Configuration ownership

The trainer converts the admitted high-level request into the canonical
configuration stored in `run.json`. The exact effective values are persisted:

- model identity and immutable revision;
- dataset identity and manifest hash;
- training method and trainable parameter inventory;
- optimizer, learning rate, schedule, rank, alpha, and precision;
- batch, accumulation, dimensions, buckets, and step count;
- save, sample, progress, and cancellation cadence;
- random seeds and resume parent;
- backend, binary, native-library, and schema versions.

Defaults are versioned product data. UI presets may suggest values but cannot
hide or override the effective trainer configuration.

## Cache ownership

The trainer creates, validates, reuses, and invalidates its cache under the run
workspace. Model backends provide encoder hooks; they do not replace the shared
cache lifecycle.

`cache/manifest.json` includes at least:

- cache schema version;
- dataset manifest hash and ordered sample count;
- model and encoder identities with immutable revisions;
- encoder weight hashes;
- caption and conditioning policy;
- resolution, bucket, crop, augmentation, and dropout policy;
- stored tensor names, shapes, and dtypes;
- producer binary and native-library hashes;
- completion state and content hashes.

Cache creation uses a workspace lock and atomic publication. An incomplete
cache is never treated as reusable. A warm run may reuse a compatible cache by
content identity, but the selected cache is recorded in the new run manifest.
Changing any identity-bearing field invalidates reuse automatically.

`only_cache` is a normal trainer mode using the same lifecycle. It is not a
separate operator procedure.

## Logs and progress

`logs/train.log` exists before validation begins. Human-readable progress and
structured progress events are derived from the same trainer state. The UI can
attach at any time, including after process restart, and display existing logs
without a live child process.

Failures include the lifecycle stage, stable error code, actionable message,
and underlying cause. Missing product inputs fail loudly. Missing generated
artifacts cause the trainer to generate them or explain why generation failed.

## Save and resume

Checkpoint cadence writes both the inference artifact and a complete trainer-
state bundle required for exact resume. The bundle is the Mojo-written state
safetensors plus its adjacent `*.resume.json` identity manifest. The tensor
payload contains adapter weights and optimizer moments. The manifest binds that
payload by content hash and records step and epoch, scheduler state, data
ordering, random state, cache identity, effective configuration hash, trainer
binary identity, and parent checkpoint identity.

Resume rehashes the selected state and validates every identity-bearing field
before GPU allocation. It never silently resets optimizer or data-order state.

## UI and control-plane boundary

The UI submits model, dataset, recipe, output name, and optional resume intent.
It does not submit paths for generated configuration or cache artifacts.

Rust owns admission, HTTP/WebSocket transport, durable run indexing, queues,
GPU leases, process supervision, cancellation delivery, and file serving. Mojo
owns dataset encoding, model execution, loss, backward, optimization, sampling,
checkpoint serialization, and cache production.

The run list and Logs screen are projections of durable workspaces, not an
in-memory process list.

## Required acceptance gates

The first Krea vertical and every later backend must prove:

1. Cold run from an ordinary image-and-caption dataset creates its workspace,
   configuration, cache, logs, samples, checkpoints, and trainer state.
2. Warm run reuses a compatible cache without operator input.
3. Dataset, encoder, revision, or cache-policy changes invalidate reuse.
4. Validation and cache failures appear as durable runs with readable logs.
5. Cancellation is durable and leaves a resumable or explicitly terminal
   workspace.
6. Product restart discovers all runs and restores their honest status.
7. Resume continues optimizer, scheduler, random, and data-order state.
8. No production stage requires Python, a shell operator, or an AI assistant.
9. All generated paths remain beneath the selected output root.
10. Samples and checkpoints are bound to exact configuration and backend hashes.
