#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
cd "$repo_root"

: "${LTX2_HQ_BIN:=${TMPDIR:-/tmp}/ltx2_t2v_av_hq}"
: "${LTX2_PIXI_ENV:=${CONDA_PREFIX:-$repo_root/.pixi/envs/default}}"
: "${LTX2_CUDNN_LIB:=$LTX2_PIXI_ENV/lib}"

if [[ ! -d "$LTX2_CUDNN_LIB" ]]; then
  echo "missing cuDNN runtime directory: $LTX2_CUDNN_LIB" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$LTX2_CUDNN_LIB:${LD_LIBRARY_PATH:-}"

pixi run mojo build -I . -I vendor/mojo-libs \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -Lserenitymojo/ops/cshim/lib \
  -Xlinker -lserenity_cudnn_sdpa \
  -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
  -Xlinker -lcudnn \
  -Xlinker -rpath -Xlinker "$repo_root/serenitymojo/ops/cshim/lib" \
  -Xlinker -rpath -Xlinker "$LTX2_CUDNN_LIB" \
  serenitymojo/pipeline/ltx2_t2v_av_hq.mojo \
  -o "$LTX2_HQ_BIN"

exec "$LTX2_HQ_BIN" staged lora stream audio nonag "$@"
