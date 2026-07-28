#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
registry="$repo_root/serenitymojo/configs/ltx2_request_profiles.json"
output_dir="$repo_root/output/bin"
source_file="$repo_root/serenitymojo/sampling/ltx2_request_cli.mojo"
sampling_file="$repo_root/serenitymojo/sampling/ltx2_sampling.mojo"
pipeline_file="$repo_root/serenitymojo/pipeline/ltx2_t2v_av_hq.mojo"
tiled_decode_file="$repo_root/serenitymojo/models/vae/ltx2_tiled_decode.mojo"
vae_encoder_file="$repo_root/serenitymojo/models/vae/ltx2_vae_encoder.mojo"
conv3d_file="$repo_root/serenitymojo/models/vae/conv3d.mojo"
png_file="$repo_root/serenitymojo/image/png.mojo"
image_io_file="$repo_root/serenitymojo/serve/image_io.mojo"
image_decode_file="$repo_root/serenitymojo/image/decode.mojo"

if [[ ! -f "$registry" ]]; then
    echo "missing LTX2 profile registry: $registry" >&2
    exit 2
fi
if ! jq -e '.schema == "serenity.ltx2.request_profiles.v1"' "$registry" >/dev/null; then
    echo "invalid LTX2 profile registry schema" >&2
    exit 2
fi

mkdir -p "$output_dir"
cd "$repo_root"

profile_filter="${1:-all}"
build_one() {
    local group_id="$1"
    local width="$2"
    local height="$3"
    local frames="$4"
    local fps="$5"
    local runner="$output_dir/ltx2_serenity_${width}x${height}_${frames}f_${fps}fps"

    if [[ "$profile_filter" != "all" && "$profile_filter" != "$group_id" && "$profile_filter" != "${width}x${height}_${frames}f_${fps}fps" ]]; then
        return
    fi
    if [[ "${LTX2_FORCE_REBUILD:-0}" != "1" && -x "$runner" \
        && "$runner" -nt "$registry" \
        && "$runner" -nt "$source_file" \
        && "$runner" -nt "$sampling_file" \
        && "$runner" -nt "$pipeline_file" \
        && "$runner" -nt "$tiled_decode_file" \
        && "$runner" -nt "$vae_encoder_file" \
        && "$runner" -nt "$conv3d_file" \
        && "$runner" -nt "$png_file" \
        && "$runner" -nt "$image_io_file" \
        && "$runner" -nt "$image_decode_file" ]]; then
        echo "LTX2 profile up to date: ${width}x${height}, ${frames} frames at ${fps} FPS"
        return
    fi

    echo "building LTX2 profile $group_id: ${width}x${height}, ${frames} frames at ${fps} FPS"
    pixi run mojo build --optimization-level 2 -j 1 \
        -D "LTX2_REQUEST_WIDTH=$width" \
        -D "LTX2_REQUEST_HEIGHT=$height" \
        -D "LTX2_REQUEST_FRAMES=$frames" \
        -D "LTX2_REQUEST_FPS=$fps" \
        -I . -I vendor/mojo-libs \
        -Xlinker -lm -Xlinker -lcuda \
        -Xlinker -Lserenitymojo/ops/cshim/lib \
        -Xlinker -lserenity_cudnn_sdpa \
        -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
        -Xlinker -lcudnn \
        -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
        -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
        "$source_file" -o "$runner"
}

while IFS=$'\t' read -r group_id width height fps frames_csv; do
    IFS=',' read -ra frame_values <<<"$frames_csv"
    for frames in "${frame_values[@]}"; do
        build_one "$group_id" "$width" "$height" "$frames" "$fps"
    done
done < <(
    jq -r '.profile_groups[] | [
        .id,
        (.width | tostring),
        (.height | tostring),
        (.fps | tostring),
        (.frames | map(tostring) | join(","))
    ] | @tsv' "$registry"
)

echo "LTX2 request profile build complete"
