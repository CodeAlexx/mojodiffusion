#!/usr/bin/env bash
# Archive a HuggingFace cache repo to the USB backup disk, then remove it from
# the NVMe. Nothing is deleted until the archive verifies against the source.
#
# Why tar rather than a directory copy: an HF cache repo is
# `snapshots/<rev>/name -> ../../blobs/<sha>` relative symlinks over a blob
# store, and the backup disk is ntfs3, which cannot represent a symlink. A tar
# carries the symlinks inside the archive, so the repo restores byte-for-byte
# and structure-for-structure. `cp -aL` would instead dereference every link and
# store each blob twice.
#
# The trade-off, stated plainly: an archived repo is NOT loadable in place.
# Anything that resolves a path inside it must be restored first. Only archive
# repos nothing resolves a path into — use a per-file eviction with a symlink
# for repos that are still referenced.
#
# Usage: scripts/archive_hf_repo.sh models--org--name [more...]
set -euo pipefail

HUB="/home/alex/.cache/huggingface/hub"
DEST="/media/alex/backup/hf-cache-cold"
mkdir -p "$DEST"

for name in "$@"; do
    src="$HUB/$name"
    tarball="$DEST/$name.tar"

    if [ ! -d "$src" ]; then
        echo "SKIP (not a directory): $src"
        continue
    fi
    if [ -e "$tarball" ]; then
        echo "SKIP (archive already exists): $tarball"
        continue
    fi

    bytes="$(du -sb "$src" | cut -f1)"
    echo "archive $name  ($(numfmt --to=iec "$bytes"))"
    echo "  -> $tarball"

    tar -cf "$tarball" -C "$HUB" "$name"
    sync

    echo "  verifying archive against source"
    if ! tar -df "$tarball" -C "$HUB" 2>&1 | tee /tmp/tar_diff_$$.log | head -20; then
        echo "  VERIFY FAILED — source left untouched"
        exit 1
    fi
    if [ -s /tmp/tar_diff_$$.log ]; then
        echo "  VERIFY REPORTED DIFFERENCES — source left untouched"
        cat /tmp/tar_diff_$$.log
        rm -f /tmp/tar_diff_$$.log
        exit 1
    fi
    rm -f /tmp/tar_diff_$$.log
    echo "  verified clean"

    rm -rf "$src"
    echo "  removed $src"
    printf '%s\t%s\t%s\n' "$name" "$bytes" "$(date -Iseconds)" >> "$DEST/MANIFEST.tsv"
done

echo
df -h / /media/alex/backup | grep -v ^Filesystem
