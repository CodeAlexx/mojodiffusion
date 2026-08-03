# serenitymojo/models/vae/parity/minimax_h3_ref_encode_gate.mojo
#
# ███ PENDING-GPU ███  MiniMax-H3 ref2va reference-ENCODE parity gate.
#
# WRITTEN BUT NOT YET RUNNABLE. It is checked in now so the oracle contract is
# fixed before the encode is implemented, not negotiated afterwards. It needs
# two things that do not exist yet:
#
#   1. the oracle dump (a GPU run of the vendor's own encode), and
#   2. the encode itself — `minimax_h3_encode_reference_visual_seam` in
#      `models/vae/minimax_h3_ref_encode.mojo` still raises.
#
# Until both land this binary prints a PENDING banner and RAISES. It never
# reports a pass, so it cannot be mistaken for a green gate in a sweep.
#
# ── WHY THIS IS A GPU GATE AND THE OTHER ONE IS NOT ──────────────────────────
# The CPU half — pixel normalize, fp16 round, latent normalize, patchify, audio
# row packing, the 0.999 mix — is ALREADY gated bit-exact, host-side, with no
# GPU: `minimax_h3_ref_encode_probe.mojo` (12 checks). What is left needs a
# device: the tiled VAE encoder itself and the posterior draw.
#
# ── ORACLE CONTRACT (write the producer to match this, not the reverse) ──────
# `scripts/minimax_h3_ref_encode_oracle.py`, run against the REAL Ref2VA
# checkpoint on the GPU, dumping one safetensors with these keys:
#
#   in.frames            uint8  [T, H, W, 3]   the prepared reference frames,
#                                              channels-LAST, at the reference's
#                                              own canvas, already 24 fps and
#                                              already trimmed to 17n+5
#   in.is_image          int32  []             1 -> `_encode_clip` path
#                                              0 -> `_encode` path
#   out.pixels           f32    [3, T, H, W]   after (x/255 - mean)/std
#   out.moments          f32    [1, 2C, T', H', W']  the VAE's raw moments
#   out.sample           f32    [1, C, T', H', W']   posterior.sample(gen=42),
#                                              BEFORE the fp16 round
#   out.rows             f32    [N, C*pt*ph*pw]  final condition rows, after
#                                              fp16 -> normalize -> patchify
#   out.noise            f32    [N, C*pt*ph*pw]  keyframe_condition_noise draw
#   out.rows_mixed       f32    [N, C*pt*ph*pw]  scale_noise(rows, 0.999, noise)
#
# Dumping `out.pixels`, `out.moments` and `out.sample` separately is the point:
# it lands a failure on the normalization, the convolutions, or the draw, rather
# than on "the encode".
#
# ── BARS, AND THE ONE THAT CANNOT BE MET ─────────────────────────────────────
#   out.pixels       BIT-EXACT. Pure elementwise f32; already proven host-side.
#   out.moments      cos >= 0.9999 vs the F32 noise floor. The encoder is
#                    convolutional; this is the same bar the video-encoder gate
#                    already holds.
#   out.sample       *** CANNOT BE MET AS A VALUE COMPARISON ***  The draw is
#                    `torch.Generator().manual_seed(42)` — torch's own CPU RNG
#                    stream. `ops/random.mojo`'s `randn` is distributionally
#                    correct but NOT torch-bit-exact (that file says so itself).
#                    So compare DISTRIBUTION here (mean/std of
#                    (sample - mean_of_moments) / std_of_moments against N(0,1)),
#                    and gate the VALUE path by FEEDING `out.sample` back in as
#                    an input. See the next line.
#   out.rows         BIT-EXACT *given* `out.sample` as input. This is the check
#                    that matters and it is fully in reach: it isolates the
#                    fp16-round/normalize/patchify chain from the RNG.
#   out.rows_mixed   BIT-EXACT *given* `out.rows` and `out.noise` as inputs,
#                    for exactly the same reason.
#
# THE STANDING RISK, stated plainly: a seeded ref2va run will NOT reproduce the
# vendor's conditioning byte for byte until either a torch-compatible normal
# draw exists or the vendor's noise is imported. Everything else can be exact.
# That is a property of the RNG, not of this port, and no arrangement of this
# gate makes it go away.
#
# Run (once the oracle exists AND the seam is implemented):
#   python3 scripts/minimax_h3_ref_encode_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/vae/parity/minimax_h3_ref_encode_gate.mojo \
#     -o output/checks/minimax_h3_ref_encode_gate \
#   && output/checks/minimax_h3_ref_encode_gate

from std.collections import List
from std.sys import argv

from serenitymojo.io.safetensors import SafeTensors


comptime ORACLE = (
    "/home/alex/mojodiffusion/output/minimax_h3_ref2va/ref_encode_ref.safetensors"
)


def _oracle_keys() -> List[String]:
    return [
        String("in.frames"),
        String("in.is_image"),
        String("out.pixels"),
        String("out.moments"),
        String("out.sample"),
        String("out.rows"),
        String("out.noise"),
        String("out.rows_mixed"),
    ]


def main() raises:
    var args = argv()
    var oracle_path = String(ORACLE)
    if len(args) >= 2:
        oracle_path = String(args[1])

    print("MiniMax-H3 ref2va reference-ENCODE parity gate")
    print("")
    print("  ###################################################################")
    print("  # PENDING-GPU. This gate is written, not runnable.                #")
    print("  # Blocked on BOTH:                                                #")
    print("  #   1. the oracle dump (needs a GPU run of the vendor's encode)   #")
    print("  #   2. minimax_h3_encode_reference_visual_seam, which still raises#")
    print("  # The CPU half of this stage IS gated, host-side, bit-exact:      #")
    print("  #   models/vae/parity/minimax_h3_ref_encode_probe.mojo (12 checks)#")
    print("  ###################################################################")
    print("")
    print("  expected oracle:", oracle_path)

    var have_oracle = True
    try:
        var st = SafeTensors.open(oracle_path)
        var names = st.names()
        print("  oracle present:", len(names), "tensors")
        var missing = List[String]()
        var wanted = _oracle_keys()
        for i in range(len(wanted)):
            if wanted[i] not in names:
                missing.append(wanted[i])
        if len(missing) > 0:
            print("  oracle is INCOMPLETE, missing:")
            for i in range(len(missing)):
                print("    ", missing[i])
            have_oracle = False
    except:
        print("  oracle ABSENT")
        have_oracle = False

    print("")
    if not have_oracle:
        raise Error(
            "minimax_h3_ref_encode_gate: PENDING-GPU — no usable oracle at "
            + oracle_path
            + ". Produce it with scripts/minimax_h3_ref_encode_oracle.py on the"
            " GPU (see this file's ORACLE CONTRACT header for the exact key"
            " set and the per-key bars), and implement"
            " minimax_h3_encode_reference_visual_seam. Note the standing RNG"
            " caveat: `out.sample` cannot be value-matched because the draw"
            " uses torch's own generator; feed it back as an INPUT and gate"
            " `out.rows` bit-exactly instead."
        )

    raise Error(
        "minimax_h3_ref_encode_gate: PENDING-GPU — the oracle is present but"
        " minimax_h3_encode_reference_visual_seam is not implemented, so there"
        " is nothing to compare against it yet. Implement the seam"
        " (models/vae/minimax_h3_ref_encode.mojo), then replace this raise with"
        " the comparisons the header specifies."
    )
