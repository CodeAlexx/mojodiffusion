#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${LTX2_HQ_BIN:=/tmp/ltx2_t2v_av_hq}"
: "${LTX2_CUDNN_LIB:=/home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib}"

if [[ ! -d "$LTX2_CUDNN_LIB" ]]; then
  echo "missing cuDNN runtime directory: $LTX2_CUDNN_LIB" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$LTX2_CUDNN_LIB:${LD_LIBRARY_PATH:-}"

pixi run mojo build -I . -I /home/alex/MOJO-libs \
  -Xlinker -lm -Xlinker -lcuda \
  -Xlinker -Lserenitymojo/ops/cshim/lib \
  -Xlinker -lserenity_cudnn_sdpa \
  -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
  -Xlinker -lcudnn \
  -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
  -Xlinker -rpath -Xlinker "$LTX2_CUDNN_LIB" \
  serenitymojo/pipeline/ltx2_t2v_av_hq.mojo \
  -o "$LTX2_HQ_BIN"

exec "$LTX2_HQ_BIN" staged lora stream audio nonag "$@"
