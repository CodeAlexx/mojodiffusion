#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

forbidden_brand=$(printf '\157\156\145\164\162\141\151\156\145\162')
forbidden_brand_regex="\\b${forbidden_brand:0:3}[_. -]?${forbidden_brand:3}\\b"

if rg -a -ni -P --hidden --glob '!.git/**' --glob '!output/**' \
  --glob '!artifacts/**' --glob '!evidence/**' "$forbidden_brand_regex" .; then
  echo "error: competitor-branded text is forbidden in the clean repository" >&2
  exit 1
fi

if git ls-files --cached --others --exclude-standard | rg -i -P "$forbidden_brand_regex"; then
  echo "error: competitor-branded filename is forbidden in the clean repository" >&2
  exit 1
fi

legacy_lower=$(printf '\157\164')
legacy_upper=$(printf '\117\124')
legacy_abbreviation_regex="(?<![A-Za-z0-9])(?:${legacy_lower}_|${legacy_upper}_)|(?:_${legacy_lower}|_${legacy_upper})[0-9_]*\\b|\\b${legacy_upper}\\b|\\b${legacy_lower}\\b"
owned_text_paths=(
  serenitymojo trainer scripts docs benchmarks probes tests
  serenity-server/Cargo.toml serenity-server/Cargo.lock
  serenity-server/crates serenity-server/scripts serenity-server/tests
  README.md pixi.toml
)
owned_text_globs=(
  --glob '!webui/board/vendor/**' --glob '!trainer/webui/board/vendor/**' \
  --glob '!**/target/**'
)
if rg -I -ni -P "$legacy_abbreviation_regex" \
  "${owned_text_paths[@]}" "${owned_text_globs[@]}"; then
  echo "error: legacy competitor abbreviation remains in repository-owned text" >&2
  exit 1
fi

legacy_attached_regex="(?-i:\\b${legacy_upper}preset\\b|_${legacy_lower}keys\\b|W${legacy_upper}Device)"
if rg -I -n -P "$legacy_attached_regex" \
  "${owned_text_paths[@]}" "${owned_text_globs[@]}"; then
  echo "error: attached legacy competitor abbreviation remains in repository-owned text" >&2
  exit 1
fi


legacy_filename_regex="(^|[/_.-])(?:${legacy_lower}|${legacy_upper})[0-9_]*([/_.-]|$)"
if git ls-files --cached --others --exclude-standard | rg -P "$legacy_filename_regex"; then
  echo "error: legacy competitor abbreviation remains in a repository filename" >&2
  exit 1
fi

internal_doc_regex='(^|/)[^/]*(PLAN|AUDIT|STATUS|CAMPAIGN|HANDOFF|ROADMAP|RECOVERY|VERIFICATION|LEDGER|REPORT|notes)[^/]*\.md$'
if git ls-files | rg -i -P "$internal_doc_regex" | rg -v '^vendor/'; then
  echo "error: internal planning or evidence document is tracked" >&2
  exit 1
fi

if git ls-files | rg -P '^(artifacts|evidence)/'; then
  echo "error: generated artifacts or internal evidence are tracked" >&2
  exit 1
fi

if ! rg -Fxq '*.md' .gitignore; then
  echo "error: Markdown is not default-denied in .gitignore" >&2
  exit 1
fi

while IFS= read -r doc; do
  if git check-ignore --no-index -q "$doc"; then
    echo "error: tracked Markdown is not explicitly whitelisted: $doc" >&2
    exit 1
  fi
done < <(git ls-files '*.md')

required_source_counts=(
  'serenitymojo:2300'
  'serenity-server:180'
  'trainer/src:650'
)
for entry in "${required_source_counts[@]}"; do
  source_root=${entry%%:*}
  minimum=${entry##*:}
  actual=$(find "$source_root" -type f | wc -l)
  if ((actual < minimum)); then
    echo "error: source inventory loss under $source_root: $actual < $minimum" >&2
    exit 1
  fi
done

map_count=$(find serenity-server/canvas -type f -name '*.map' | wc -l)
if ((map_count != 35)); then
  echo "error: expected 35 canvas source maps, found $map_count" >&2
  exit 1
fi

critical_docs=(
  docs/architecture/RECOMMENDED_TRAINER_STRUCTURE.md
  docs/maps/T5_ZIMAGE_TRAINING_MAP.md
  serenitymojo/MAP.md
  serenity-server/canvas/CONTRACT.md
  docs/MOJO_AUTOGRAD_INTERNALS.md
  docs/MOJO_CONVENTIONS.md
  docs/MOJO_DIAGNOSTICS.md
  docs/MOJO_DIFFUSION_NUMERIC_API.md
  docs/MOJO_GPU_DEVICE_MATH_GOTCHAS.md
  docs/MOJO_KERNELS.md
  docs/MOJO_MODULES.md
  docs/MOJO_REUSABLE_INFERENCE_COMPONENTS.md
  docs/TRAINER_PRODUCT_CONTRACT.md
  trainer/docs/TRAINER_DRIVER_SEPARATION.md
  trainer/docs/UI_MAP_2026-07-05.md
)
for doc in "${critical_docs[@]}"; do
  if [[ ! -f "$doc" ]]; then
    echo "error: critical documentation missing: $doc" >&2
    exit 1
  fi
done

portable_first_slice_files=(
  pixi.toml
  scripts/setup.sh
  scripts/fix_mojo_runpath.sh
  serenitymojo/pipeline/krea2_paths.mojo
  serenitymojo/pipeline/zimage_stage.mojo
  serenitymojo/models/krea2/krea2_prepare_cache.mojo
  serenitymojo/models/anima/anima_prepare_cache.mojo
  serenitymojo/models/chroma/chroma_prepare_cache.mojo
  serenitymojo/models/krea2/train_krea2.mojo
  serenitymojo/io/env.mojo
  serenitymojo/training/train_ltx2_av.mojo
  serenitymojo/pipeline/ltx2_t2v_av_hq.mojo
  scripts/run_ltx2_hq121.sh
  serenitymojo/serve/model_scan.mojo
  serenitymojo/serve/sample_cli_backend.mojo
  serenitymojo/serve/chroma_encode_subprocess.mojo
  serenitymojo/serve/serenity_daemon.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_block_load_decomp_probe.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_ckpt_equiv_probe.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_resident_budget_probe.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_stack_bwd_parity.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_stack_parity.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_stack_smoke.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_step_budget_probe.mojo
  serenitymojo/models/ltx2/parity/ltx2_av_tape_vs_recompute_probe.mojo
  configs/ltx2_av_smoke.json
  trainer/src/serenity_trainer/trainer/train_anima_real.mojo
  trainer/src/serenity_trainer/trainer/train_chroma_real.mojo
  serenitymojo/configs/anima.json
  serenitymojo/configs/chroma.json
  serenitymojo/configs/krea2.json
  serenitymojo/ops/cshim/build.sh
  trainer/webui/src/main.rs
  trainer/webui/src/bin/trainer_runner.rs
  trainer/webui/src/bin/config_smoke.rs
  trainer/webui/src/board.rs
  trainer/webui/src/captioner.rs
  trainer/webui/src/config_merge.rs
  trainer/webui/presets.json
  trainer/webui/static/index.html
  serenity-server/crates/server/src/trainer.rs
  serenity-server/crates/server/src/caption.rs
  serenity-server/crates/server/src/magic.rs
  serenity-server/crates/server/src/models.rs
  serenity-server/crates/server/src/settings.rs
  serenity-server/crates/server/src/video.rs
  serenity-server/canvas/js/workflow-builder.js
  serenity-server/crates/ipc/tests/stub_seam.rs
)
if rg -n '/home/[A-Za-z0-9._-]+/|/mnt/disk[0-9]+/' "${portable_first_slice_files[@]}"; then
  echo "error: developer-machine absolute path found in an active product file" >&2
  exit 1
fi

if rg -n '/home/[A-Za-z0-9._-]+/serenity-trainer|\.\./serenity-trainer' \
  trainer/docs trainer/parity/lens trainer/smoke \
  trainer/src/serenity_trainer/smoke; then
  echo "error: trainer smoke/parity source resolves through a stale sibling checkout" >&2
  exit 1
fi

if ! rg -Fq 'env!("CARGO_MANIFEST_DIR")' \
  serenity-server/crates/server/src/main.rs \
  || ! rg -Fq '"/../../../output/bin/serenity_worker_stub"' \
  serenity-server/crates/server/src/main.rs; then
  echo "error: inference worker default is not repository-rooted" >&2
  exit 1
fi

if ! rg -Fq 'fix_mojo_runpath.sh' pixi.toml scripts/verify_renamed_entrypoints.sh \
  || ! rg -Fq 'patchelf' pixi.toml scripts/fix_mojo_runpath.sh; then
  echo "error: Mojo product builds do not enforce portable ELF RUNPATHs" >&2
  exit 1
fi

required_contract_terms=(
  'output/<run_id>/'
  'trainer creates, validates, reuses, and invalidates its cache'
  '`logs/train.log` exists before validation begins'
  'does not submit paths for generated configuration or cache artifacts'
  'No production stage requires Python, a shell operator, or an AI assistant'
)

for term in "${required_contract_terms[@]}"; do
  if ! rg -Fq "$term" docs/TRAINER_PRODUCT_CONTRACT.md; then
    echo "error: trainer contract is missing required term: $term" >&2
    exit 1
  fi
done

if git remote -v | rg -i -P "$forbidden_brand_regex"; then
  echo "error: competitor-branded Git remote is forbidden in the clean repository" >&2
  exit 1
fi

echo "repository contract: PASS"
