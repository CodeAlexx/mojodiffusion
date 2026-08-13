#!/usr/bin/env bash
# MiniMax-H3 keyframe (I2VA / FL2VA / L2VA) end-to-end smoke.
#
# PENDING-GPU as of 2026-08-03: written, shell-checked, NOT run. The overnight
# chain owns the card.
#
# WHAT IT PROVES (and what it does not): that a keyframe request runs the whole
# chain — ingest, EXIF, PIL-exact canvas prepare, VAE encode, seeded posterior
# sample, 0.999 mix, anchored layout, denoise with the condition rows pinned,
# condition-row drop, video + audio decode — and lands real artifacts of the
# right size. It is NOT a parity check: the per-stage gates
# (pipeline/parity/minimax_h3_keyframe_*_probe.mojo) are.
#
# THE ONE THING TO ACTUALLY LOOK AT afterwards: frame_00000.png against the
# input keyframe for I2VA/FL2VA (they should be near-identical), and the LAST
# frame against it for L2VA/FL2VA's second keyframe. That is the only check that
# distinguishes "the anchor is wired" from "the anchor is wired to the wrong
# end", which every numeric gate here would pass either way.
#
# usage: scripts/minimax_h3_i2va_smoke.sh <mode> <keyframe.png> [<last.png>]
set -euo pipefail

REPO=/home/alex/mojodiffusion
SCRATCH="${H3_SMOKE_SCRATCH:-/tmp/h3_i2va_smoke}"
VENV=/home/alex/OneTrainer/venv/bin/python
CSHIM="$REPO/serenitymojo/ops/cshim/lib"

MODE="${1:?usage: $0 <i2va|l2va|fl2va> <keyframe.png> [<last.png>]}"
KEYFRAME="${2:?missing keyframe}"
LAST="${3:-}"

case "$MODE" in
  i2va|l2va) KEYFRAMES=1 ;;
  fl2va)     KEYFRAMES=2 ;;
  *) echo "unknown mode '$MODE'" >&2; exit 2 ;;
esac
if [ "$KEYFRAMES" = 2 ] && [ -z "$LAST" ]; then
  echo "fl2va needs a second keyframe" >&2; exit 2
fi

# ── GUARD: never start while another H3 job holds the card ─────────────────
# The failure mode this prevents is not a crash — it is two jobs fitting, both
# slowing down, and the overnight batch's per-step timings silently becoming
# meaningless.
if pgrep -f h3_night_chain >/dev/null || pgrep -f h3_t2va >/dev/null; then
  echo "REFUSING: an H3 job is running (h3_night_chain / h3_t2va)." >&2
  echo "  wait for it, or set H3_SMOKE_FORCE=1 to override." >&2
  [ "${H3_SMOKE_FORCE:-0}" = 1 ] || exit 3
fi
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free: ${FREE_MIB} MiB"
if [ "${FREE_MIB:-0}" -lt 20000 ] && [ "${H3_SMOKE_FORCE:-0}" != 1 ]; then
  echo "REFUSING: needs ~20 GiB free (transformer streaming + VAE)." >&2
  exit 3
fi

mkdir -p "$SCRATCH"
OUT="$SCRATCH/out_$MODE"
rm -rf "$OUT"; mkdir -p "$OUT"

# ── 1. Resolve the canvas the keyframe implies, and prepare it ─────────────
# The canvas belongs to the FIRST keyframe (before_encoder.py:172-173), and the
# binary's geometry is comptime, so it has to be resolved before the build.
read -r CANVAS_H CANVAS_W < <("$VENV" - "$KEYFRAME" <<'PY'
import sys
from PIL import Image, ImageOps
SHORT, MAXPX, MULT = 768, 768 * 1344, 32
with Image.open(sys.argv[1]) as im:
    w, h = ImageOps.exif_transpose(im).convert("RGB").size
ratio = w / h
assert 0.25 <= ratio <= 4.0, f"aspect ratio {ratio:g} outside MiniMax-H3's 1:4..4:1"
if ratio >= 1.0:
    width, height = SHORT * ratio, float(SHORT)
else:
    width, height = float(SHORT), SHORT / ratio
area = width * height
if area > MAXPX:
    s = (MAXPX / area) ** 0.5
    width, height = width * s, height * s
print(max(MULT, round(height / MULT) * MULT), max(MULT, round(width / MULT) * MULT))
PY
)
echo "canvas resolved from $KEYFRAME: ${CANVAS_W}x${CANVAS_H}"

# ── 2. Token budget. The prompt length is comptime too, so it has to be known
# before the build. Counted with H3's own tokenizer through the existing
# count_tokens CLI rather than guessed.
FRAMES="${H3_SMOKE_FRAMES:-124}"          # 124 frames = 5.17 s, the shortest legal render
BODY="${H3_SMOKE_PROMPT:-[Shot 1] Live-action, cinematic, the scene shown in <Picture 1> continues as the camera holds a static shot.

overall_soundscape: Quiet room tone with faint outdoor traffic.

non_diegetic_music: N/A}"

echo "building (canvas ${CANVAS_W}x${CANVAS_H}, ${FRAMES} frames, ${KEYFRAMES} keyframe(s))"
echo "NOTE: H3_TEXT_TOKENS must match the tokenized prompt; the binary prints the"
echo "      correct value in its error message if it is wrong — rebuild with it."
TEXT_TOKENS="${H3_SMOKE_TEXT_TOKENS:-256}"

cd "$REPO"
BIN="$SCRATCH/h3_i2va_${CANVAS_W}x${CANVAS_H}_${FRAMES}_${KEYFRAMES}kf"
pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
  -D "H3_HEIGHT=$CANVAS_H" -D "H3_WIDTH=$CANVAS_W" -D "H3_FRAMES=$FRAMES" \
  -D "H3_TEXT_TOKENS=$TEXT_TOKENS" -D "H3_KEYFRAMES=$KEYFRAMES" \
  -D H3_KF_NO_VISION=1 \
  serenitymojo/pipeline/minimax_h3_i2va.mojo -o "$BIN" \
  -Xlinker -L"$CSHIM" -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm

# H3_KF_NO_VISION=1 is REQUIRED until the Qwen3-VL vision tower is wired: the
# conditioner cannot encode a `<Picture i>` vision block yet. The keyframe still
# anchors the render through its VAE condition rows. See the pipeline header.

# ── 3. Run ─────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$CSHIM:${LD_LIBRARY_PATH:-}"
STEPS="${H3_SMOKE_STEPS:-30}"
SEED="${H3_SMOKE_SEED:-0}"
LOG="$OUT/smoke.log"
echo "running -> $LOG"
if [ "$KEYFRAMES" = 2 ]; then
  "$BIN" "$MODE" "$BODY" "$KEYFRAME" "$LAST" "$OUT" "$STEPS" "$SEED" 2>&1 | tee "$LOG"
else
  "$BIN" "$MODE" "$BODY" "$KEYFRAME" "$OUT" "$STEPS" "$SEED" 2>&1 | tee "$LOG"
fi

# ── 4. Artifact checks ─────────────────────────────────────────────────────
fail=0
# Frames are ONE raw RGB24 stream (frames.rgb) since the PNG-writer removal;
# the count check is exact byte math, the anchor frames are extracted with
# ffmpeg below.
RGB_BYTES=$(stat -c %s "$OUT/frames.rgb" 2>/dev/null || echo 0)
WANT_BYTES=$((FRAMES * CANVAS_W * CANVAS_H * 3))
NFRAMES=$((RGB_BYTES / (CANVAS_W * CANVAS_H * 3)))
echo "frames written: $NFRAMES (want $FRAMES; $RGB_BYTES bytes)"
[ "$RGB_BYTES" = "$WANT_BYTES" ] || { echo "FAIL: frames.rgb size"; fail=1; }
[ -s "$OUT/audio.wav" ] || { echo "FAIL: no audio.wav"; fail=1; }
[ -s "$OUT/result.json" ] || { echo "FAIL: no result.json"; fail=1; }
[ -s "$OUT/latents.safetensors" ] || { echo "FAIL: no latents.safetensors"; fail=1; }

# The frame that should look like the keyframe, per mode — extracted from the
# raw stream on demand.
extract_frame() {  # <index> <out.png>
  ffmpeg -v error -y -f rawvideo -pixel_format rgb24 \
    -video_size "${CANVAS_W}x${CANVAS_H}" \
    -skip_initial_bytes $(($1 * CANVAS_W * CANVAS_H * 3)) \
    -i "$OUT/frames.rgb" -frames:v 1 "$2"
}
case "$MODE" in
  i2va|fl2va) ANCHOR_FRAME="$OUT/anchor_first.png"; extract_frame 0 "$ANCHOR_FRAME" ;;
  l2va)       ANCHOR_FRAME="$OUT/anchor_last.png"; extract_frame $((FRAMES - 1)) "$ANCHOR_FRAME" ;;
esac
echo ""
echo "LOOK AT THIS: $ANCHOR_FRAME should closely match $KEYFRAME"
if [ "$MODE" = fl2va ]; then
  extract_frame $((FRAMES - 1)) "$OUT/anchor_last.png"
  echo "         and: $OUT/anchor_last.png should match $LAST"
fi
"$VENV" - "$ANCHOR_FRAME" "$KEYFRAME" <<'PY' || true
import sys
import numpy as np
from PIL import Image
try:
    a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64)
    b = Image.open(sys.argv[2]).convert("RGB").resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS)
    b = np.asarray(b, dtype=np.float64)
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"anchor frame vs keyframe: PSNR {psnr:.2f} dB (a wired anchor is well above a random pair's ~8 dB)")
except Exception as exc:                    # the render may have failed earlier
    print("could not compare:", exc)
PY

echo ""
[ "$fail" = 0 ] && echo "SMOKE PASS: $OUT" || { echo "SMOKE FAIL"; exit 1; }
