#!/usr/bin/env python3
"""Import a TensorBoard run (events.out.tfevents.*) into SerenityBoard's board.db.

Purpose: put SerenityTrainer (or any TB-logging) runs side-by-side with Mojo runs in
/board — same tags, same charts, same compare views. This is also the parity
oracle path: values land byte-faithfully (f32 events -> REAL column), steps and
wall_times verbatim from the event file.

Usage:
  <python-with-tensorboard> scripts/board_import_tfevents.py <tb_run_dir> [run_name]

  <tb_run_dir>  directory containing events.out.tfevents.* (one TB "run")
  [run_name]    board run name (default: parent_dir__leaf_dir)

Idempotent: scalars PK (run,tag,step) dedupes on re-import; the run row is
INSERT OR IGNORE'd. The webui reads board.db WAL-mode, so importing while the
server runs is safe (server holds no exclusive lock).
"""
import sys, os, sqlite3, json, glob

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webui", "board.db")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run_dir = os.path.abspath(sys.argv[1])
    if not glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
        sys.exit(f"no events.out.tfevents.* in {run_dir}")
    if len(sys.argv) > 2:
        run_name = sys.argv[2]
    else:
        parts = run_dir.rstrip("/").split("/")
        run_name = f"{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else parts[-1]

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if not tags:
        sys.exit(f"no scalar tags in {run_dir}")

    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    first_wt, last_wt, last_step, n_pts = None, 0.0, 0, 0
    for tag in tags:
        for ev in ea.Scalars(tag):
            v = float(ev.value)
            if v != v or v in (float("inf"), float("-inf")):
                continue  # board's insert_point skips non-finite too
            conn.execute(
                "INSERT OR IGNORE INTO scalars (run, tag, step, wall_time, value) VALUES (?,?,?,?,?)",
                (run_name, tag, int(ev.step), float(ev.wall_time), v),
            )
            n_pts += 1
            first_wt = ev.wall_time if first_wt is None else min(first_wt, ev.wall_time)
            last_wt = max(last_wt, ev.wall_time)
            last_step = max(last_step, int(ev.step))

    sid = f"s{int((first_wt or 0) * 1000)}"
    conn.execute(
        "INSERT OR IGNORE INTO runs (name, workspace_dir, source, preset_id, status, start_time, "
        "last_wall_time, last_step, max_steps, active_session_id, hparams_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (run_name, run_dir, "tfevents", None, "completed",
         first_wt or 0.0, last_wt, last_step, last_step, sid,
         json.dumps({"imported_from": "tensorboard"})),
    )
    conn.execute(
        "UPDATE runs SET last_wall_time=?, last_step=?, max_steps=?, status='completed' WHERE name=? AND source='tfevents'",
        (last_wt, last_step, last_step, run_name),
    )
    conn.commit()
    print(f"[board-import] run '{run_name}': {len(tags)} tags, {n_pts} points, last_step {last_step}")
    print(f"[board-import] tags: {tags}")


if __name__ == "__main__":
    main()
