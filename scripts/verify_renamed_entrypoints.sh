#!/usr/bin/env bash
set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
mkdir -p output/verification/bin output/verification/logs

common=(
  --optimization-level 2
  -I .
  -I trainer/src
  -I vendor/mojo-libs
  -Xlinker -L"$CONDA_PREFIX/lib"
  -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib"
  -Xlinker -L"$CONDA_PREFIX/targets/x86_64-linux/lib/stubs"
  -Xlinker -lcuda
  -Xlinker -lcublas
  -Xlinker -lm
  -Xlinker -ldl
  -Xlinker -lpng16
  -Xlinker -lturbojpeg
  -Xlinker -lsqlite3
  -Xlinker -Lserenitymojo/ops/cshim/lib
  -Xlinker -lserenity_cudnn_sdpa
)

targets=(
  trainer-ideogram4:trainer/src/serenity_trainer/trainer/Ideogram4LiveTrainer.mojo
  trainer-anima:trainer/src/serenity_trainer/trainer/train_anima_real.mojo
  trainer-chroma:trainer/src/serenity_trainer/trainer/train_chroma_real.mojo
  trainer-ernie:trainer/src/serenity_trainer/trainer/train_ernie_real.mojo
  trainer-flux:trainer/src/serenity_trainer/trainer/train_flux_real.mojo
  trainer-hidream:trainer/src/serenity_trainer/trainer/train_hidream_o1_real.mojo
  trainer-klein:trainer/src/serenity_trainer/trainer/train_klein_real.mojo
  trainer-l2p:trainer/src/serenity_trainer/trainer/train_l2p_real.mojo
  trainer-ltx2:trainer/src/serenity_trainer/trainer/train_ltx2_real.mojo
  trainer-qwenimage:trainer/src/serenity_trainer/trainer/train_qwenimage_real.mojo
  trainer-sd35:trainer/src/serenity_trainer/trainer/train_sd35_real.mojo
  trainer-sdxl:trainer/src/serenity_trainer/trainer/train_sdxl_real.mojo
  trainer-wan22:trainer/src/serenity_trainer/trainer/train_wan22_real.mojo
  trainer-zimage:trainer/src/serenity_trainer/trainer/train_zimage_real.mojo
  inference-stub:serenitymojo/serve/serenity_worker_stub.mojo
  inference-anima:serenitymojo/serve/serenity_worker_anima.mojo
  inference-chroma:serenitymojo/serve/serenity_worker_chroma.mojo
  inference-flux:serenitymojo/serve/serenity_worker_flux.mojo
  inference-ideogram4:serenitymojo/serve/serenity_worker_ideogram4.mojo
  inference-klein:serenitymojo/serve/serenity_worker_klein.mojo
  inference-krea2:serenitymojo/serve/serenity_worker_krea2.mojo
  inference-lens:serenitymojo/serve/serenity_worker_lens.mojo
  inference-qwenimage:serenitymojo/serve/serenity_worker_qwenimage.mojo
  inference-sd3:serenitymojo/serve/serenity_worker_sd3.mojo
  inference-sdxl:serenitymojo/serve/serenity_worker_sdxl.mojo
  inference-sensenova:serenitymojo/serve/serenity_worker_sensenova.mojo
  inference-zimage:serenitymojo/serve/serenity_worker_zimage.mojo
)

failures=0
requested=${1:-}
production_inference=false
if [[ "$requested" == production-inference ]]; then
  production_inference=true
fi
printf '%-24s %s\n' target result
for item in "${targets[@]}"; do
  name=${item%%:*}
  source=${item#*:}
  if [[ "$production_inference" == true && "$name" != inference-* ]]; then
    continue
  fi
  if [[ "$production_inference" == false && -n "$requested" && "$requested" != "$name" ]]; then
    continue
  fi
  if [[ "$name" == trainer-* ]]; then
    output="output/bin/serenity_${name#trainer-}_live_trainer"
  elif [[ "$production_inference" == true ]]; then
    output="output/bin/serenity_worker_${name#inference-}"
  else
    output="output/verification/bin/${name}"
  fi
  if [[ "$output" == output/bin/* ]]; then
    runtime_rpath='$ORIGIN/../../.pixi/envs/default/lib:$ORIGIN/../../serenitymojo/ops/cshim/lib'
  else
    runtime_rpath='$ORIGIN/../../../.pixi/envs/default/lib:$ORIGIN/../../../serenitymojo/ops/cshim/lib'
  fi
  log="output/verification/logs/${name}.log"
  if mojo build "${common[@]}" -Xlinker -rpath -Xlinker "$runtime_rpath" \
    "$source" -o "$output" >"$log" 2>&1 \
    && bash scripts/fix_mojo_runpath.sh "$output" >>"$log" 2>&1; then
    printf '%-24s PASS\n' "$name"
  else
    printf '%-24s FAIL (%s)\n' "$name" "$log"
    failures=$((failures + 1))
  fi
done

if ((failures > 0)); then
  echo "entrypoint verification: FAIL ($failures targets)" >&2
  exit 1
fi
echo "entrypoint verification: PASS"
