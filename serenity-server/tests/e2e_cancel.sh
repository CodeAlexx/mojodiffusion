#!/usr/bin/env bash
# e2e_cancel.sh — Phase-A hardening harness for the Serenity Rust control plane.
#
# Exercises the THREE server-hardening fixes, end-to-end, against the ALREADY-BUILT
# CPU stub worker (output/bin/serenity_worker_stub). ZERO GPU. cargo + bash only.
# The orchestrator runs this script alongside e2e_stub.sh.
#
# Tests:
#   A) WS SUBSCRIBE-RACE / REPLAY (the HIGH fix): submit a FAST job (steps=3, ~300ms),
#      WAIT past its completion, THEN open the WS. The client must STILL receive every
#      frame (the intermediate `progress` frames AND the terminal `done`) — proving the
#      per-job event HISTORY is replayed to a late subscriber (the entry is retained for
#      a grace window past terminal).
#
#   B) CANCEL (POST /v1/cancel): submit a LONGER job (steps=40, ~4s), connect a WS,
#      POST /v1/cancel {"job":<id>} while it's in flight, and assert:
#        - the cancel endpoint returns HTTP 200 with {"accepted":true},
#        - the WS receives a terminal `cancelled` event,
#        - a cancel for a bogus/again-finished job returns HTTP 404.
#
# HARD RULE: this script NEVER builds Mojo. The stub worker is pre-built.
#
# Usage:  tests/e2e_cancel.sh
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
    echo "# E2E(cancel) RESULT: FAIL"
    echo "# reason: $*"
    echo "########################################"
    exit 1
}

cleanup() {
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
command -v python3 >/dev/null 2>&1 || fail "python3 not found (required for WS client)"
python3 -c "import websocket" 2>/dev/null || fail "python module 'websocket' (websocket-client) not importable"
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
    if command -v python3 >/dev/null 2>&1; then
        python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
    else
        echo "7812"
    fi
}
PORT="${PORT:-$(pick_free_port)}"
echo "[e2e] port       = ${PORT}"

# ── 3. tempdir + launch server ───────────────────────────────────────────────
TMPDIR_E2E="$(mktemp -d "${TMPDIR:-/tmp}/serenity_e2e_cancel.XXXXXX")"
echo "[e2e] out_dir    = ${TMPDIR_E2E}"
SRV_LOG="${TMPDIR_E2E}/server.log"

echo "[e2e] launching serenity-server ..."
"${SERVER_BIN}" \
    --worker "${WORKER_BIN}" \
    --out-dir "${TMPDIR_E2E}" \
    --port "${PORT}" \
    >"${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "[e2e] server pid = ${SRV_PID}"

# ── 4. wait for /v1/health ───────────────────────────────────────────────────
HEALTH_URL="http://127.0.0.1:${PORT}/v1/health"
echo "[e2e] waiting for health at ${HEALTH_URL} ..."
HEALTHY=0
for _ in $(seq 1 100); do
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
        echo "----- server.log -----"; cat "${SRV_LOG}" || true; echo "----------------------"
        fail "server process exited before becoming healthy"
    fi
    if curl -fsS "${HEALTH_URL}" -o "${TMPDIR_E2E}/health.json" 2>/dev/null; then
        HEALTHY=1; break
    fi
    sleep 0.2
done
[[ "${HEALTHY}" == "1" ]] || { echo "----- server.log -----"; cat "${SRV_LOG}" || true; fail "health never came up"; }
echo "[e2e] health: $(cat "${TMPDIR_E2E}/health.json")"

GEN_URL="http://127.0.0.1:${PORT}/v1/generate"
CANCEL_URL="http://127.0.0.1:${PORT}/v1/cancel"

post_generate() {  # $1 = JSON body -> echoes job_id
    local body="$1" resp jid
    resp="$(curl -fsS -X POST "${GEN_URL}" -H 'content-type: application/json' -d "${body}")" \
        || fail "POST /v1/generate failed"
    jid="$(printf '%s' "${resp}" | jq -r '.job_id // empty' 2>/dev/null)"
    [[ -z "${jid}" ]] && jid="$(printf '%s' "${resp}" | sed -n 's/.*"job_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    [[ -n "${jid}" ]] || fail "could not parse job_id from: ${resp}"
    printf '%s' "${jid}"
}

# read_ws <job_id> <out_file> <terminal_kinds_csv> : exit 0 if any terminal kind seen.
# terminal_kinds_csv e.g. "done" or "cancelled" or "done,cancelled".
read_ws() {
    local job="$1" out="$2" kinds="$3"
    python3 - "${PORT}" "${job}" "${out}" "${kinds}" <<'PY'
import sys, json
import websocket  # websocket-client (preflight-checked)
port, job, out_path, kinds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
want = set(k.strip() for k in kinds.split(",") if k.strip())
url = f"ws://127.0.0.1:{port}/v1/progress?job={job}"
rc = 1
try:
    ws = websocket.create_connection(url, timeout=40)
except Exception as e:
    sys.stderr.write(f"ws connect failed: {e}\n"); sys.exit(2)
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
            f.write(msg + "\n"); f.flush()
            try:
                ev = json.loads(msg)
            except Exception:
                continue
            if ev.get("ev") in want:
                rc = 0; break
finally:
    try: ws.close()
    except Exception: pass
sys.exit(rc)
PY
}

# ─────────────────────────────────────────────────────────────────────────────
# TEST A — WS subscribe-race / replay: a FAST job whose events were all emitted
# BEFORE we connect must still be replayed in full (incl. terminal done).
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "[e2e] === TEST A: fast-job WS replay (subscribe-race fix) ==="
JOB_A="$(post_generate '{"model":"stub","prompt":"fast replay","steps":3,"width":32,"height":32}')"
echo "[e2e] fast job_id = ${JOB_A}"

# The stub sleeps ~100ms/step; steps=3 => 2 progress frames + done in ~0.3s. Wait
# well past completion so EVERY event (progress + done) is already in the per-job
# history before we connect — a pure exercise of the replay path. (steps=1 would
# emit ONLY a done with no intermediate progress, so we use 3 to prove the full
# history — not just the terminal frame — is buffered and replayed.)
echo "[e2e] sleeping 1.5s so the fast job finishes BEFORE we open the WS ..."
sleep 1.5

WS_A_OUT="${TMPDIR_E2E}/ws_a.txt"
echo "[e2e] opening WS for ALREADY-FINISHED job ${JOB_A} (expect replayed frames+done)"
if read_ws "${JOB_A}" "${WS_A_OUT}" "done"; then
    echo "[e2e] TEST A: WS replayed to terminal 'done' AFTER job finished. frames:"
    sed 's/^/[e2e][wsA] /' "${WS_A_OUT}" 2>/dev/null || true
else
    echo "----- ws A frames -----"; cat "${WS_A_OUT}" 2>/dev/null || true; echo "-----------------------"
    echo "----- server.log -----";  cat "${SRV_LOG}" || true; echo "----------------------"
    fail "TEST A: late WS did NOT receive replayed frames+done (subscribe-race fix broken)"
fi
# Sanity: at least one progress frame AND the done frame are present in the replay.
grep -q '"ev"[[:space:]]*:[[:space:]]*"progress"' "${WS_A_OUT}" \
    || fail "TEST A: replay missing progress frame(s) (history not fully buffered)"
grep -q '"ev"[[:space:]]*:[[:space:]]*"done"' "${WS_A_OUT}" \
    || fail "TEST A: replay missing terminal done frame"
# And the artifact really was produced (the job genuinely ran).
[[ -f "${TMPDIR_E2E}/${JOB_A}.png" ]] || fail "TEST A: fast job PNG missing: ${JOB_A}.png"
echo "[e2e] TEST A PASS (replayed progress+done; PNG present)"

# ─────────────────────────────────────────────────────────────────────────────
# TEST B — cancel: a longer job is cancelled mid-flight; 200 accepted, WS sees
# 'cancelled'; a cancel for a non-in-flight job is 404.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "[e2e] === TEST B: POST /v1/cancel (in-flight 200 + cancelled; bogus 404) ==="

# B0. Cancel a job id that was never submitted -> 404.
echo "[e2e] B0: cancel a bogus job id -> expect HTTP 404"
BOGUS_CODE="$(curl -s -o "${TMPDIR_E2E}/cancel_bogus.json" -w '%{http_code}' \
    -X POST "${CANCEL_URL}" -H 'content-type: application/json' \
    -d '{"job":"job_does_not_exist"}')"
echo "[e2e] B0: bogus cancel http=${BOGUS_CODE} body=$(cat "${TMPDIR_E2E}/cancel_bogus.json")"
[[ "${BOGUS_CODE}" == "404" ]] || fail "B0: bogus cancel expected 404, got ${BOGUS_CODE}"

# B1. Submit a LONG job (40 steps * ~0.1s ≈ 4s) so we have time to cancel it.
JOB_B="$(post_generate '{"model":"stub","prompt":"cancel me","steps":40,"width":32,"height":32}')"
echo "[e2e] long job_id = ${JOB_B}"

# B2. Start a WS reader in the background that succeeds on a 'cancelled' terminal.
WS_B_OUT="${TMPDIR_E2E}/ws_b.txt"
read_ws "${JOB_B}" "${WS_B_OUT}" "cancelled,done,failed" &
WS_B_PID=$!

# B3. Give the job a moment to be genuinely in flight, then cancel -> expect 200.
sleep 0.6
echo "[e2e] B3: POST /v1/cancel for in-flight ${JOB_B} -> expect HTTP 200 accepted:true"
CANCEL_CODE="$(curl -s -o "${TMPDIR_E2E}/cancel_ok.json" -w '%{http_code}' \
    -X POST "${CANCEL_URL}" -H 'content-type: application/json' \
    -d "{\"job\":\"${JOB_B}\"}")"
echo "[e2e] B3: cancel http=${CANCEL_CODE} body=$(cat "${TMPDIR_E2E}/cancel_ok.json")"
[[ "${CANCEL_CODE}" == "200" ]] || { wait "${WS_B_PID}" 2>/dev/null || true; fail "B3: in-flight cancel expected 200, got ${CANCEL_CODE}"; }
if command -v jq >/dev/null 2>&1; then
    [[ "$(jq -r '.accepted' "${TMPDIR_E2E}/cancel_ok.json")" == "true" ]] \
        || fail "B3: cancel response missing accepted:true"
fi

# B4. The WS reader must terminate having seen 'cancelled'.
if wait "${WS_B_PID}"; then
    echo "[e2e] B4: WS reached a terminal event after cancel. frames (tail):"
    tail -n 5 "${WS_B_OUT}" 2>/dev/null | sed 's/^/[e2e][wsB] /' || true
else
    echo "----- ws B frames -----"; cat "${WS_B_OUT}" 2>/dev/null || true; echo "-----------------------"
    echo "----- server.log -----";  cat "${SRV_LOG}" || true; echo "----------------------"
    fail "B4: WS did not reach a terminal event after cancel"
fi
grep -q '"ev"[[:space:]]*:[[:space:]]*"cancelled"' "${WS_B_OUT}" \
    || { echo "----- ws B frames -----"; cat "${WS_B_OUT}" 2>/dev/null || true; echo "-----------------------"; \
         fail "B4: WS never saw a 'cancelled' event (cancel did not propagate to the worker)"; }
echo "[e2e] B4: 'cancelled' event observed"

# B5. A second cancel of the now-finished job -> 404 (no longer in flight).
echo "[e2e] B5: re-cancel the finished job -> expect HTTP 404"
AGAIN_CODE="$(curl -s -o "${TMPDIR_E2E}/cancel_again.json" -w '%{http_code}' \
    -X POST "${CANCEL_URL}" -H 'content-type: application/json' \
    -d "{\"job\":\"${JOB_B}\"}")"
echo "[e2e] B5: re-cancel http=${AGAIN_CODE} body=$(cat "${TMPDIR_E2E}/cancel_again.json")"
[[ "${AGAIN_CODE}" == "404" ]] || fail "B5: re-cancel of finished job expected 404, got ${AGAIN_CODE}"
echo "[e2e] TEST B PASS (404 bogus; 200 accepted; cancelled observed; 404 re-cancel)"

# ─────────────────────────────────────────────────────────────────────────────
# TEST C — the server SURVIVED both tests and still serves a fresh job (proves the
# worker was reused / driver stayed up across cancel).
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "[e2e] === TEST C: server still serves a fresh job after cancel ==="
JOB_C="$(post_generate '{"model":"stub","prompt":"post-cancel","steps":2,"width":32,"height":32}')"
WS_C_OUT="${TMPDIR_E2E}/ws_c.txt"
if read_ws "${JOB_C}" "${WS_C_OUT}" "done"; then
    echo "[e2e] TEST C PASS (fresh job ${JOB_C} ran to done after a cancel)"
else
    echo "----- ws C frames -----"; cat "${WS_C_OUT}" 2>/dev/null || true; echo "-----------------------"
    echo "----- server.log -----";  cat "${SRV_LOG}" || true; echo "----------------------"
    fail "TEST C: server failed to serve a fresh job after cancel (driver torn down?)"
fi
[[ -f "${TMPDIR_E2E}/${JOB_C}.png" ]] || fail "TEST C: post-cancel job PNG missing"

# ── PASS ─────────────────────────────────────────────────────────────────────
echo
echo "########################################"
echo "# E2E(cancel) RESULT: PASS"
echo "#   A replay job  : ${JOB_A} (late WS replayed progress+done)"
echo "#   B cancel job  : ${JOB_B} (200 accepted; WS saw cancelled; 404 re-cancel)"
echo "#   C post-cancel : ${JOB_C} (server still serving)"
echo "########################################"
exit 0
