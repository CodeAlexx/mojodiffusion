#!/usr/bin/env bash
# capture_lowering_refs.sh — generate the Rust-executor parity oracle corpus.
#
# Given a BUILT lower-only oracle at output/bin/serenity_lower (the orchestrator
# builds serenity_lower_cli.mojo capped to that path), this script walks a corpus
# of real workflow graphs, wraps each into a minimal /v1/generate request body,
# runs the oracle, and saves the lowered (flat-genparams) output. Those lowered
# JSON files become the reference the Rust workflow executor is diffed against:
# Rust lowering must match Mojo lowering byte-for-byte.
#
# ── WRAPPING CONTRACT ─────────────────────────────────────────────────────────
# Each corpus graph G is wrapped as a /v1/generate request body:
#
#     { "model":  <placeholder>,
#       "prompt": <placeholder>,
#       "width":  <placeholder>,
#       "height": <placeholder>,
#       "workflow": G }
#
# apply_workflow_params() reads obj["workflow"] and writes the flat backend
# params onto obj IN PLACE using _set_if_missing / _copy_field_if_missing
# semantics: a value the GRAPH supplies (model via UNETLoader/CheckpointLoader,
# size via Empty*LatentImage, prompt via CLIPTextEncode, sampler/scheduler/cfg
# via the sampler nodes) WINS over the placeholder, because the graph's value is
# written first and the placeholder only fills a still-missing field. The
# placeholders therefore (a) satisfy adapters that demand a top-level prompt
# (e.g. the Ideogram4 Comfy export's prompt-builder subgraph) and (b) make the
# wrapping uniform/deterministic. They never override real graph values, so the
# captured lowering reflects the GRAPH, not the wrapper.
#
# Two corpora are scanned:
#   1. /home/alex/serenityflow-v2/serenityflow/workflows/*.json
#        — bare Comfy-API prompt graphs (node-id -> {class_type, inputs}); these
#          are the substantive corpus. Wrapped as workflow:G.
#   2. /home/alex/mojodiffusion/output/checks/*.json
#        — scanned for any file that IS a workflow graph (has nodes+edges, OR is
#          a bare Comfy-API prompt graph, OR has a top-level "workflow" key).
#          Most check files are readiness REPORTS, not graphs, and are skipped.
#          This keeps the script forward-compatible if graph fixtures land there.
#
# A graph file already shaped as a full request (has a top-level "workflow" key)
# is passed through unwrapped; otherwise the file's whole JSON is the graph G.
#
# Output: /home/alex/mojodiffusion/serenity-server/tests/refs/<name>.lowered.json
#   <name> = "<corpus_tag>__<basename-without-.json>" to avoid collisions across
#   the two corpora.
#
# HARD RULE: this script NEVER builds Mojo. It consumes a prebuilt oracle.
#
# Usage:  scripts/capture_lowering_refs.sh
# Env overrides (optional):
#   LOWER_BIN   path to serenity_lower (default: <repo>/output/bin/serenity_lower)
#   REFS_DIR    output dir (default: <repo>/serenity-server/tests/refs)

set -euo pipefail

REPO="/home/alex/mojodiffusion"
LOWER_BIN="${LOWER_BIN:-${REPO}/output/bin/serenity_lower}"
REFS_DIR="${REFS_DIR:-${REPO}/serenity-server/tests/refs}"

SERENITYFLOW_DIR="/home/alex/serenityflow-v2/serenityflow/workflows"
CHECKS_DIR="${REPO}/output/checks"

# Placeholder request fields (only fill graph-missing values; see wrapping note).
PH_MODEL="oracle-placeholder-model"
PH_PROMPT="oracle placeholder prompt"
PH_WIDTH=1024
PH_HEIGHT=1024

if [[ ! -x "${LOWER_BIN}" ]]; then
  echo "FAIL: oracle not found or not executable: ${LOWER_BIN}" >&2
  echo "      (the orchestrator builds serenity_lower_cli.mojo -> ${LOWER_BIN})" >&2
  exit 1
fi

mkdir -p "${REFS_DIR}"

TMP_REQ="$(mktemp -t serenity_lower_req.XXXXXX.json)"
TMP_OUT="$(mktemp -t serenity_lower_out.XXXXXX.json)"
cleanup() { rm -f "${TMP_REQ}" "${TMP_OUT}"; }
trap cleanup EXIT

captured=0
skipped=0
failed=0

# is_workflow_graph <file> -> prints "graph" | "request" | "" (and exits 0)
# "graph"   : file JSON is itself a workflow graph -> wrap as workflow:G
# "request" : file already has a top-level "workflow" key -> pass through
# ""        : not a workflow graph -> skip
classify_graph() {
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(""); sys.exit(0)
if isinstance(d, dict) and "workflow" in d and isinstance(d["workflow"], (dict, list)):
    print("request"); sys.exit(0)
if isinstance(d, dict) and "nodes" in d and "edges" in d:
    print("graph"); sys.exit(0)
# bare Comfy-API prompt graph: every key an int, every node has class_type+inputs
if isinstance(d, dict) and d:
    ks = list(d.keys())
    if all(k.lstrip("-").isdigit() for k in ks) and all(
        isinstance(d[k], dict) and "class_type" in d[k] and "inputs" in d[k]
        for k in ks
    ):
        print("graph"); sys.exit(0)
print("")
PY
}

# build_request <file> <kind> -> writes wrapped request to TMP_REQ
build_request() {
  python3 - "$1" "$2" "${TMP_REQ}" \
    "${PH_MODEL}" "${PH_PROMPT}" "${PH_WIDTH}" "${PH_HEIGHT}" <<'PY'
import json, sys
src, kind, out_path, ph_model, ph_prompt, ph_w, ph_h = sys.argv[1:8]
d = json.load(open(src))
if kind == "request":
    body = d  # already a full /v1/generate body with a workflow key
else:
    body = {
        "model": ph_model,
        "prompt": ph_prompt,
        "width": int(ph_w),
        "height": int(ph_h),
        "workflow": d,
    }
with open(out_path, "w") as f:
    json.dump(body, f)
PY
}

run_corpus() {
  local tag="$1" dir="$2"
  [[ -d "${dir}" ]] || { echo "  (corpus dir absent, skipped: ${dir})"; return 0; }
  shopt -s nullglob
  local f
  for f in "${dir}"/*.json; do
    local base kind name
    base="$(basename "${f}" .json)"
    kind="$(classify_graph "${f}")"
    if [[ -z "${kind}" ]]; then
      skipped=$((skipped + 1))
      continue
    fi
    name="${tag}__${base}"
    if ! build_request "${f}" "${kind}"; then
      echo "  WRAP-FAIL ${name}" >&2
      failed=$((failed + 1))
      continue
    fi
    if "${LOWER_BIN}" "${TMP_REQ}" >"${TMP_OUT}" 2>/dev/null; then
      mv "${TMP_OUT}" "${REFS_DIR}/${name}.lowered.json"
      # Persist the exact input request next to the ref so the Rust parity
      # harness can re-lower deterministically from the same bytes the oracle
      # consumed (the oracle reads TMP_REQ; we copy it as <name>.request.json).
      cp "${TMP_REQ}" "${REFS_DIR}/${name}.request.json"
      captured=$((captured + 1))
      echo "  ok  ${name}"
    else
      # Lowering raised (e.g. a 501 unsupported-node graph). Record nothing; the
      # Rust executor must raise on the same input, verified out-of-band.
      echo "  LOWER-FAIL ${name} (oracle rejected; not a parity ref)" >&2
      failed=$((failed + 1))
    fi
  done
  shopt -u nullglob
}

echo "Capturing workflow-lowering parity refs -> ${REFS_DIR}"
echo "Oracle: ${LOWER_BIN}"
echo "[corpus] serenityflow workflows: ${SERENITYFLOW_DIR}"
run_corpus "serenityflow" "${SERENITYFLOW_DIR}"
echo "[corpus] output/checks graphs:   ${CHECKS_DIR}"
run_corpus "checks" "${CHECKS_DIR}"

echo "----------------------------------------------------------------"
echo "Captured: ${captured}   Skipped(non-graph): ${skipped}   Oracle-rejected: ${failed}"
echo "Refs written under: ${REFS_DIR}"
if [[ "${captured}" -eq 0 ]]; then
  echo "WARNING: 0 refs captured." >&2
  exit 1
fi
