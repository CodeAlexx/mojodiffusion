"""squareq-plan.json / build-state.json schemas + atomic IO.

The plan records what the builder decided per layer (format, rank, quality,
bytes) and is the manifest the Mojo loader validates against. The build state
makes conversions restartable: it is rewritten (tmp+rename) after every
completed unit of work, so a killed build resumes instead of restarting.
"""

import json
import os
import tempfile

PLAN_VERSION = 1
STATE_VERSION = 1


def atomic_write_json(path: str, obj: dict) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".sq_tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def new_plan(model: str, source: str, format_tag: str, hblock: int, group: int) -> dict:
    return {
        "plan_version": PLAN_VERSION,
        "format": format_tag,
        "model": model,
        "source": source,
        "hblock": hblock,
        "group": group,
        "layers": {},  # key -> {out, in, rank, format, cos_w, rel_l2, bytes_q, bytes_bf16}
        "passthrough": [],  # keys stored verbatim
        "totals": {},
    }


def finalize_plan(plan: dict) -> dict:
    bq = sum(v["bytes_q"] for v in plan["layers"].values())
    bb = sum(v["bytes_bf16"] for v in plan["layers"].values())
    cs = [v["cos_w"] for v in plan["layers"].values()]
    plan["totals"] = {
        "quantized_layers": len(plan["layers"]),
        "passthrough_tensors": len(plan["passthrough"]),
        "bytes_quantized": bq,
        "bytes_bf16_equiv": bb,
        "compression": (bq / bb) if bb else None,
        "cos_w_min": min(cs) if cs else None,
        "cos_w_mean": (sum(cs) / len(cs)) if cs else None,
    }
    return plan


def new_state(source: str, source_bytes: int, source_mtime: int) -> dict:
    """Resume identity = source path+size+mtime; a changed source invalidates."""
    return {
        "state_version": STATE_VERSION,
        "source": source,
        "source_bytes": source_bytes,
        "source_mtime": source_mtime,
        "completed_shards": [],
    }


def load_state(path: str, source: str, source_bytes: int, source_mtime: int):
    """Return a valid resume state or None (missing/stale)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            st = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if (
        st.get("state_version") != STATE_VERSION
        or st.get("source") != source
        or st.get("source_bytes") != source_bytes
        or st.get("source_mtime") != source_mtime
    ):
        return None
    return st
