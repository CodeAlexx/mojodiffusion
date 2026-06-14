#!/usr/bin/env bash
# e2e_workflow.sh — end-to-end harness for the SERVER-INTEGRATION seam: the Rust
# control plane lowering a full Comfy/Swarm `workflow` graph before dispatching to
# the UNCHANGED Mojo `output/bin/serenity_worker_stub` (ZERO GPU).
#
# This proves the POST /v1/generate handler's NEW behavior: when the request body
# carries a `workflow` key, serenity-server calls serenity_graph::lower_request to
# flatten the graph into params, THEN runs the job — and when the graph contains an
# unknown active node it FAILS LOUD with a 501/422 instead of silently degrading to
# a plain txt2img.
#
# What it does:
#   1. `cargo build --workspace` (safe; cargo only — NEVER builds Mojo).
#   2. Launch  target/debug/serenity-server  --worker <stub> --out-dir <tmp> --port <free>.
#   3. Poll GET /v1/health until the server answers.
#   4. POST /v1/generate with a REAL workflow graph (the `workflow` object lifted from
#      tests/refs/serenityflow__zimage_t2i.request.json), wrapped as
#        {"model":"stub","prompt":"wf e2e","workflow":<that graph>}
#      -> assert HTTP 200 + a job_id (i.e. the server LOWERED it; it did NOT 501).
#   5. Open WS /v1/progress?job=<id> and read frames until a terminal `done` event.
#   6. Assert the produced PNG (<tmp>/<job_id>.png) exists, has the PNG signature,
#      AND embeds the `serenity.genparams.v1` tEXt keyword.
#   7. POST /v1/generate with the SAME graph but one active node's class_type swapped
#      to an unknown type -> assert HTTP 501 (or 422): fail-loud, NOT a job_id, NOT a
#      silent txt2img.
#   8. SIGKILL the server, clean the tempdir, print PASS/FAIL, exit non-zero on fail.
#
# HARD RULE: this script NEVER builds Mojo. The stub worker is pre-built.
#
# Usage:  tests/e2e_workflow.sh
# Env overrides (optional):
#   WORKER_BIN   path to serenity_worker_stub (default: <repo>/output/bin/serenity_worker_stub)
#   PORT         fixed port (default: an OS-assigned free port)

set -euo pipefail

# ── locations ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"        # .../serenity-server
REPO_ROOT="$(cd "${SERVER_DIR}/.." && pwd)"          # .../mojodiffusion
WORKER_BIN="${WORKER_BIN:-${REPO_ROOT}/output/bin/serenity_worker_stub}"
SERVER_BIN="${SERVER_DIR}/target/debug/serenity-server"
ZIMAGE_REQ="${SERVER_DIR}/tests/refs/serenityflow__zimage_t2i.request.json"

# ── state for cleanup ────────────────────────────────────────────────────────
SRV_PID=""
TMPDIR_E2E=""

fail() {
    echo
    echo "########################################"
    echo "# E2E RESULT: FAIL"
    echo "# reason: $*"
    echo "########################################"
    exit 1
}

cleanup() {
    # Best-effort: kill the server (SIGKILL) and reap, remove the tempdir.
    if [[ -n "${SRV_PID}" ]] && kill -0 "${SRV_PID}" 2>/dev/null; then
        kill -9 "${SRV_PID}" 2>/dev/null || true
        wait "${SRV_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TMPDIR_E2E}" && -d "${TMPDIR_E2E}" ]]; then
        rm -rf "${TMPDIR_E2E}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'fail "interrupted (signal)"' INT TERM

# ── preflight ────────────────────────────────────────────────────────────────
command -v curl    >/dev/null 2>&1 || fail "curl not found (required)"
command -v python3 >/dev/null 2>&1 || fail "python3 not found (required to build the workflow body)"
[[ -x "${WORKER_BIN}" ]] || fail "stub worker binary not executable at ${WORKER_BIN} (do NOT build Mojo here)"
[[ -f "${ZIMAGE_REQ}" ]] || fail "zimage reference request not found: ${ZIMAGE_REQ}"

echo "[e2e] repo_root  = ${REPO_ROOT}"
echo "[e2e] server_dir = ${SERVER_DIR}"
echo "[e2e] worker_bin = ${WORKER_BIN}"
echo "[e2e] zimage_req = ${ZIMAGE_REQ}"

# ── 1. build the workspace (safe; cargo only) ────────────────────────────────
echo "[e2e] cargo build --workspace ..."
( cd "${SERVER_DIR}" && cargo build --workspace ) || fail "cargo build --workspace failed"
[[ -x "${SERVER_BIN}" ]] || fail "server binary missing after build: ${SERVER_BIN}"

# ── 2. pick a port ───────────────────────────────────────────────────────────
pick_free_port() {
    python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}
PORT="${PORT:-$(pick_free_port)}"
echo "[e2e] port       = ${PORT}"

# ── 3. tempdir for outputs ───────────────────────────────────────────────────
TMPDIR_E2E="$(mktemp -d "${TMPDIR:-/tmp}/serenity_wf_e2e.XXXXXX")"
echo "[e2e] out_dir    = ${TMPDIR_E2E}"
SRV_LOG="${TMPDIR_E2E}/server.log"

# ── 3a. build the two request bodies from the zimage reference graph ──────────
# good body: {"model":"stub","prompt":"wf e2e","workflow":<that graph>}
# bad  body: same graph but node 8 (KSampler, an ACTIVE node feeding SaveImage) has
#            its class_type swapped to an unknown type -> must 501/422.
GOOD_BODY="${TMPDIR_E2E}/gen_good.json"
BAD_BODY="${TMPDIR_E2E}/gen_bad.json"
python3 - "${ZIMAGE_REQ}" "${GOOD_BODY}" "${BAD_BODY}" <<'PY'
import json, sys
ref_path, good_path, bad_path = sys.argv[1], sys.argv[2], sys.argv[3]
ref = json.load(open(ref_path))
graph = ref["workflow"]

good = {"model": "stub", "prompt": "wf e2e", "workflow": graph}
json.dump(good, open(good_path, "w"))

# Deep-copy via JSON round-trip, then poison one active output node's class_type.
bad_graph = json.loads(json.dumps(graph))
bad_graph["8"]["class_type"] = "TotallyUnknownActiveNode"
bad = {"model": "stub", "prompt": "wf e2e bad", "workflow": bad_graph}
json.dump(bad, open(bad_path, "w"))
print("[e2e] built good/bad workflow bodies", file=sys.stderr)
PY
echo "[e2e] good body  = ${GOOD_BODY}"
echo "[e2e] bad  body  = ${BAD_BODY}"

# ── 4. launch the server in the background ───────────────────────────────────
echo "[e2e] launching serenity-server ..."
"${SERVER_BIN}" \
    --worker "${WORKER_BIN}" \
    --out-dir "${TMPDIR_E2E}" \
    --port "${PORT}" \
    >"${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "[e2e] server pid = ${SRV_PID}"

# ── 5. wait for /v1/health ───────────────────────────────────────────────────
HEALTH_URL="http://127.0.0.1:${PORT}/v1/health"
echo "[e2e] waiting for health at ${HEALTH_URL} ..."
HEALTHY=0
for _ in $(seq 1 100); do          # up to ~20s
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
        echo "----- server.log -----"; cat "${SRV_LOG}" || true; echo "----------------------"
        fail "server process exited before becoming healthy"
    fi
    if curl -fsS "${HEALTH_URL}" -o "${TMPDIR_E2E}/health.json" 2>/dev/null; then
        HEALTHY=1
        break
    fi
    sleep 0.2
done
[[ "${HEALTHY}" == "1" ]] || { echo "----- server.log -----"; cat "${SRV_LOG}" || true; fail "health never came up"; }
echo "[e2e] health: $(cat "${TMPDIR_E2E}/health.json")"

GEN_URL="http://127.0.0.1:${PORT}/v1/generate"

# ── 6. POST the GOOD workflow body -> expect 200 + job_id (lowered, NOT a 501) ─
echo "[e2e] POST ${GEN_URL}  (good workflow body)"
GEN_OUT="${TMPDIR_E2E}/generate.json"
GOOD_CODE="$(curl -sS -o "${GEN_OUT}" -w '%{http_code}' \
    -X POST "${GEN_URL}" \
    -H 'content-type: application/json' \
    --data-binary "@${GOOD_BODY}" )" \
    || fail "curl POST (good) failed to execute"
echo "[e2e] good response: HTTP ${GOOD_CODE} body=$(cat "${GEN_OUT}")"
if [[ "${GOOD_CODE}" == "501" || "${GOOD_CODE}" == "422" ]]; then
    echo "----- server.log -----"; cat "${SRV_LOG}" || true; echo "----------------------"
    fail "server REJECTED a valid workflow graph with HTTP ${GOOD_CODE} (lowering broken)"
fi
[[ "${GOOD_CODE}" == "200" ]] || { echo "----- server.log -----"; cat "${SRV_LOG}" || true; fail "good workflow POST returned HTTP ${GOOD_CODE}, expected 200"; }

# Extract job_id (prefer jq; fall back to portable grep/sed).
JOB_ID=""
if command -v jq >/dev/null 2>&1; then
    JOB_ID="$(jq -r '.job_id // empty' "${GEN_OUT}")"
fi
if [[ -z "${JOB_ID}" ]]; then
    JOB_ID="$(sed -n 's/.*"job_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${GEN_OUT}")"
fi
[[ -n "${JOB_ID}" ]] || fail "could not parse job_id from good-workflow response (server did not create a job)"
echo "[e2e] job_id (workflow-lowered) = ${JOB_ID}"

# ── 7. read WS /v1/progress until a terminal event ───────────────────────────
WS_URL="ws://127.0.0.1:${PORT}/v1/progress?job=${JOB_ID}"
WS_OUT="${TMPDIR_E2E}/ws_frames.txt"
echo "[e2e] opening WS ${WS_URL}"

read_progress_ws() {
    # Reads WS frames into ${WS_OUT}; returns 0 iff a 'done' terminal frame is seen.
    # Prefers websocat; else inline python3 websocket-client.
    if command -v websocat >/dev/null 2>&1; then
        timeout 40 websocat -t "${WS_URL}" >"${WS_OUT}" 2>/dev/null || true
        if grep -q '"ev"[[:space:]]*:[[:space:]]*"done"' "${WS_OUT}"; then return 0; fi
        return 1
    fi

    python3 - "${PORT}" "${JOB_ID}" "${WS_OUT}" <<'PY'
import sys, json
try:
    import websocket  # websocket-client
except Exception as e:
    sys.stderr.write(f"python websocket-client import failed: {e}\n")
    sys.exit(3)

port, job, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"ws://127.0.0.1:{port}/v1/progress?job={job}"
rc = 1
try:
    ws = websocket.create_connection(url, timeout=40)
except Exception as e:
    sys.stderr.write(f"ws connect failed: {e}\n")
    sys.exit(2)
try:
    ws.settimeout(40)
    with open(out_path, "w") as f:
        while True:
            try:
                msg = ws.recv()
            except Exception:
                break
            if msg is None or msg == "":
                break
            f.write(msg + "\n")
            f.flush()
            try:
                ev = json.loads(msg)
            except Exception:
                continue
            kind = ev.get("ev")
            if kind == "done":
                rc = 0
                break
            if kind in ("failed", "cancelled"):
                rc = 4
                break
finally:
    try:
        ws.close()
    except Exception:
        pass
sys.exit(rc)
PY
}

if read_progress_ws; then
    echo "[e2e] WS reached terminal 'done'. frames:"
else
    echo "----- ws frames -----"; cat "${WS_OUT}" 2>/dev/null || true; echo "---------------------"
    echo "----- server.log -----"; cat "${SRV_LOG}" || true; echo "----------------------"
    fail "WS did not reach a 'done' event for the workflow-lowered job (see frames/log above)"
fi
sed 's/^/[e2e][ws] /' "${WS_OUT}" 2>/dev/null || true

# ── 8. assert the PNG artifact ───────────────────────────────────────────────
PNG="${TMPDIR_E2E}/${JOB_ID}.png"
echo "[e2e] expecting PNG at ${PNG}"
for _ in $(seq 1 25); do
    [[ -f "${PNG}" ]] && break
    sleep 0.1
done
[[ -f "${PNG}" ]] || { echo "out_dir contents:"; ls -la "${TMPDIR_E2E}"; fail "output PNG not found: ${PNG}"; }

PNG_SIZE="$(wc -c < "${PNG}")"
[[ "${PNG_SIZE}" -gt 0 ]] || fail "output PNG is empty: ${PNG}"
echo "[e2e] PNG exists, ${PNG_SIZE} bytes"

# 8a. PNG signature: 89 50 4E 47 0D 0A 1A 0A.
SIG_HEX="$(od -An -tx1 -N8 "${PNG}" | tr -d ' \n')"
EXPECT_SIG="89504e470d0a1a0a"
[[ "${SIG_HEX}" == "${EXPECT_SIG}" ]] \
    || fail "PNG signature mismatch: got '${SIG_HEX}', expected '${EXPECT_SIG}'"
echo "[e2e] PNG signature OK (${SIG_HEX})"

# 8b. embedded genparams tEXt keyword.
if LC_ALL=C grep -a -q 'serenity.genparams.v1' "${PNG}"; then
    echo "[e2e] genparams tEXt keyword 'serenity.genparams.v1' present"
else
    fail "PNG missing 'serenity.genparams.v1' tEXt keyword"
fi

# ── 9. POST the BAD workflow body -> expect 501 (or 422): fail-loud, no job ───
echo "[e2e] POST ${GEN_URL}  (bad workflow body: unknown active node)"
BAD_OUT="${TMPDIR_E2E}/generate_bad.json"
BAD_CODE="$(curl -sS -o "${BAD_OUT}" -w '%{http_code}' \
    -X POST "${GEN_URL}" \
    -H 'content-type: application/json' \
    --data-binary "@${BAD_BODY}" )" \
    || fail "curl POST (bad) failed to execute"
echo "[e2e] bad response: HTTP ${BAD_CODE} body=$(cat "${BAD_OUT}")"

if [[ "${BAD_CODE}" != "501" && "${BAD_CODE}" != "422" ]]; then
    echo "----- server.log -----"; cat "${SRV_LOG}" || true; echo "----------------------"
    fail "unknown-active-node workflow returned HTTP ${BAD_CODE}; expected 501/422 (server must fail loud, NOT silently txt2img)"
fi
# Body must carry the lowering message, NOT a job_id (no silent degrade).
if grep -q '"job_id"' "${BAD_OUT}"; then
    fail "bad workflow produced a job_id (silent fallback to txt2img); expected a fail-loud ${BAD_CODE}"
fi
echo "[e2e] bad workflow correctly rejected with HTTP ${BAD_CODE} (fail-loud)"

# ── PASS ─────────────────────────────────────────────────────────────────────
echo
echo "########################################"
echo "# E2E RESULT: PASS"
echo "#   workflow job_id : ${JOB_ID}"
echo "#   good POST       : HTTP ${GOOD_CODE} (lowered + ran)"
echo "#   png             : ${PNG} (${PNG_SIZE} bytes)"
echo "#   signature       : ${SIG_HEX}"
echo "#   genparams       : serenity.genparams.v1 present"
echo "#   bad  POST       : HTTP ${BAD_CODE} (fail-loud, no job_id)"
echo "########################################"
exit 0
