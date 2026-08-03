#!/usr/bin/env bash
# Evict cold weight files from the NVMe to the USB backup disk, leaving a
# symlink behind so anything that resolves the original path still works.
#
# Copy is O_DIRECT on both sides: 27+ GiB through the page cache would be
# charged to this cgroup and push the desktop into reclaim.
#
# Verification is byte-size equality before the original is removed. Nothing is
# deleted until its copy is confirmed present at the right size.
#
# Usage: scripts/evict_to_backup.sh <dest-subdir> <file> [file ...]
set -euo pipefail

DEST_ROOT="/media/alex/backup/serenity-models-cold"
SUBDIR="${1:?usage: evict_to_backup.sh <dest-subdir> <file>...}"
shift

DEST="$DEST_ROOT/$SUBDIR"
mkdir -p "$DEST"

for src in "$@"; do
    if [ ! -f "$src" ]; then
        echo "SKIP (not a regular file): $src"
        continue
    fi
    if [ -L "$src" ]; then
        echo "SKIP (already a symlink): $src"
        continue
    fi
    name="$(basename "$src")"
    dst="$DEST/$name"
    want="$(stat -c %s "$src")"

    echo "copy  $src"
    echo "  ->  $dst  ($(numfmt --to=iec "$want"))"
    if ! dd if="$src" of="$dst" bs=16M iflag=direct oflag=direct status=none; then
        echo "  O_DIRECT copy failed, retrying buffered"
        dd if="$src" of="$dst" bs=16M status=none
    fi
    sync

    got="$(stat -c %s "$dst")"
    if [ "$got" != "$want" ]; then
        echo "  SIZE MISMATCH: $got != $want — leaving the original alone"
        exit 1
    fi
    echo "  verified $got bytes"

    rm -f "$src"
    ln -s "$dst" "$src"
    echo "  symlinked $src -> $dst"
done

echo
df -h / /media/alex/backup | grep -v ^Filesystem
