#!/usr/bin/env bash
# e2e_stub.sh — Phase-A end-to-end harness for the Serenity Rust control plane.
#
# Proves the full Rust->Mojo seam with ZERO GPU, against the ALREADY-BUILT CPU stub
# worker (output/bin/serenity_worker_stub). The orchestrator runs this script.
#
# What it does:
#   1. `cargo build` the serenity-server workspace (safe; never OOMs).
#   2. Launch  target/debug/serenity-server  --worker <stub> --out-dir <tmp> --port <free>
#      in the background.
#   3. Poll GET /v1/health until the server answers.
#   4. POST /v1/generate {model:"stub",steps:6,width:64,height:64} -> capture job_id.
#   5. Open WS /v1/progress?job=<id> (websocat if present, else an inline python3
#      websocket client) and read frames until a terminal `done` event.
#   6. Assert the produced PNG (<tmp>/<job_id>.png) exists, has the 8-byte PNG
#      signature, AND embeds the `serenity.genparams.v1` tEXt keyword.
#   7. SIGKILL the server, clean the tempdir, print PASS/FAIL, exit non-zero on fail.
#
# HARD RULE: this script NEVER builds Mojo. The stub worker is pre-built.
#
# Usage:  tests/e2e_stub.sh
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
command -v curl  >/dev/null 2>&1 || fail "curl not found (required)"
[[ -x "${WORKER_BIN}" ]] || fail "stub worker binary not executable at ${WORKER_BIN} (Phase-A precondition; do NOT build Mojo here)"

echo "[e2e] repo_root  = ${REPO_ROOT}"
echo "[e2e] server_dir = ${SERVER_DIR}"
echo "[e2e] worker_bin = ${WORKER_BIN}"

# ── 1. build the workspace (safe; cargo only) ────────────────────────────────
echo "[e2e] cargo build (workspace) ..."
( cd "${SERVER_DIR}" && cargo build ) || fail "cargo build failed"
[[ -x "${SERVER_BIN}" ]] || fail "server binary missing after build: ${SERVER_BIN}"

# ── 2. pick a port ───────────────────────────────────────────────────────────
pick_free_port() {
    # Ask the OS for an ephemeral free TCP port via python3; fall back to a fixed
    # high port if python3 is unavailable.
    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
    else
        echo "7811"
    fi
}
PORT="${PORT:-$(pick_free_port)}"
echo "[e2e] port       = ${PORT}"

# ── 3. tempdir for outputs ───────────────────────────────────────────────────
TMPDIR_E2E="$(mktemp -d "${TMPDIR:-/tmp}/serenity_e2e.XXXXXX")"
echo "[e2e] out_dir    = ${TMPDIR_E2E}"
SRV_LOG="${TMPDIR_E2E}/server.log"

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

# ── 6. POST /v1/generate -> job_id ───────────────────────────────────────────
GEN_URL="http://127.0.0.1:${PORT}/v1/generate"
GEN_BODY='{"model":"stub","prompt":"e2e stub smoke","steps":6,"width":64,"height":64}'
echo "[e2e] POST ${GEN_URL}  ${GEN_BODY}"
curl -fsS -X POST "${GEN_URL}" \
    -H 'content-type: application/json' \
    -d "${GEN_BODY}" \
    -o "${TMPDIR_E2E}/generate.json" \
    || fail "POST /v1/generate failed"
echo "[e2e] generate response: $(cat "${TMPDIR_E2E}/generate.json")"

# Extract job_id (prefer jq; fall back to a portable grep/sed).
JOB_ID=""
if command -v jq >/dev/null 2>&1; then
    JOB_ID="$(jq -r '.job_id // empty' "${TMPDIR_E2E}/generate.json")"
fi
if [[ -z "${JOB_ID}" ]]; then
    JOB_ID="$(sed -n 's/.*"job_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${TMPDIR_E2E}/generate.json")"
fi
[[ -n "${JOB_ID}" ]] || fail "could not parse job_id from generate response"
echo "[e2e] job_id     = ${JOB_ID}"

# ── 7. read WS /v1/progress until a terminal event ───────────────────────────
WS_URL="ws://127.0.0.1:${PORT}/v1/progress?job=${JOB_ID}"
WS_OUT="${TMPDIR_E2E}/ws_frames.txt"
echo "[e2e] opening WS ${WS_URL}"

read_progress_ws() {
    # Reads WS frames into ${WS_OUT}; returns 0 iff a 'done' terminal frame is seen,
    # non-zero on failed/cancelled/timeout. Prefers websocat; else inline python3.
    if command -v websocat >/dev/null 2>&1; then
        # -n1 not relied upon; we read until the server closes after terminal.
        timeout 40 websocat -t "${WS_URL}" >"${WS_OUT}" 2>/dev/null || true
        if grep -q '"ev"[[:space:]]*:[[:space:]]*"done"' "${WS_OUT}"; then return 0; fi
        return 1
    fi

    command -v python3 >/dev/null 2>&1 || fail "neither websocat nor python3 available for WS"
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
    fail "WS did not reach a 'done' event (see frames/log above)"
fi
sed 's/^/[e2e][ws] /' "${WS_OUT}" 2>/dev/null || true

# ── 8. assert the PNG artifact ───────────────────────────────────────────────
PNG="${TMPDIR_E2E}/${JOB_ID}.png"
echo "[e2e] expecting PNG at ${PNG}"

# Give the FS a brief moment after the done event (the worker writes then emits done,
# so it should already be present; loop a few times for robustness).
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

# 8b. embedded genparams tEXt keyword (grep the raw bytes; -a treats binary as text).
if LC_ALL=C grep -a -q 'serenity.genparams.v1' "${PNG}"; then
    echo "[e2e] genparams tEXt keyword 'serenity.genparams.v1' present"
else
    fail "PNG missing 'serenity.genparams.v1' tEXt keyword"
fi

# ── PASS ─────────────────────────────────────────────────────────────────────
echo
echo "########################################"
echo "# E2E RESULT: PASS"
echo "#   job_id     : ${JOB_ID}"
echo "#   png        : ${PNG} (${PNG_SIZE} bytes)"
echo "#   signature  : ${SIG_HEX}"
echo "#   genparams  : serenity.genparams.v1 present"
echo "########################################"
exit 0
