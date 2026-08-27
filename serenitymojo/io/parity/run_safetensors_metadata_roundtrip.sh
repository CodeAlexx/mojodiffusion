#!/usr/bin/env bash
set -euo pipefail

repo=/home/alex/mojodiffusion
source_file="$repo/serenitymojo/io/parity/safetensors_metadata_roundtrip.mojo"
binary=/tmp/safetensors_metadata_roundtrip

cleanup() {
  rm -f "$binary" \
    /tmp/serenity_safetensors_metadata_roundtrip.safetensors \
    /tmp/serenity_safetensors_metadata_roundtrip.safetensors.tmp \
    /tmp/serenity_safetensors_metadata_dup_key.safetensors \
    /tmp/serenity_safetensors_metadata_dup_top.safetensors \
    /tmp/serenity_safetensors_metadata_bad_value.safetensors \
    /tmp/serenity_safetensors_metadata_bad_control.safetensors
}
trap cleanup EXIT

cd "$repo"
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$source_file" -o "$binary"
"$binary"
