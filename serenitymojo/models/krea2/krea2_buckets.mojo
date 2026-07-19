# krea2_buckets.mojo — Krea-2 aspect-ratio bucket ladder (comptime dispatch table).
#
# The Krea-2 trainer + cache-prepare are comptime-monomorphized on the latent
# canvas (LH, LW): every op that turns the latent into image tokens
# (krea2_patchify[LH,LW], krea2_build_pos[LH,LW]) and the VAE encoder
# (QwenImageVaeEncoder[IH,IW]) needs the shape at COMPILE time. Aspect-ratio
# bucketing means those shapes vary per sample, so — exactly like the Z-Image
# trainer (train_zimage_real / zimage_prepare) — we dispatch each sample to a
# COMPTIME-instantiated bucket arm chosen at runtime by the sample's cached
# (LH, LW). This module is that shared comptime table: ONE source of truth the
# stage-B prepare (VAE encode) and the trainer (patchify/pos/step) both import,
# so a cache and the trainer that reads it can never disagree on the bucket set.
#
# SEMANTICS: SimpleTuner closest-aspect + pixel-area + center-crop, reusing the
# shared derivation in training/aspect_buckets.mojo (the runtime Float64 generator
# generate_aspect_buckets is the STAGE-time truth; aspect_lat_{h,w}_units is its
# integer, comptime-evaluable twin — the krea2 ladder gate proves they agree).
#
# EDGE UNITS e = E/align (align 64): the area budget. 512px → E=512, e=8; 1024px
# → E=1024, e=16. The latent square-snap dim is 8*e (= E/8): 64 @512px, 128 @1024px
# (== the trainer's LH=64/128). The trainer passes its KREA2_EDGE_UNITS (derived
# from KREA2_RES_512) so the ladder matches the training resolution.
#
# LADDER: the SimpleTuner default 7-aspect set (== aspect_buckets.default_aspect_ladder),
# x100 keys in the SAME order — 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3.
#
# Mojo 1.0.0b1 (host + comptime only; no GPU, no Tensor).

from serenitymojo.training.aspect_buckets import (
    DEFAULT_ASPECT_LADDER_LEN, DEFAULT_ASPECT_LADDER_X100,
    aspect_lat_h_units, aspect_lat_w_units,
)


comptime KREA2_LADDER_LEN = DEFAULT_ASPECT_LADDER_LEN
comptime KREA2_LADDER_X100 = DEFAULT_ASPECT_LADDER_X100


def krea2_lat_w(aspect_x100: Int, e_units: Int) -> Int:
    """Latent W of the krea2 bucket for aspect aspect_x100/100 under edge units
    e_units. Thin wrapper over the shared SimpleTuner-parity integer derivation;
    comptime-evaluable so it can pin QwenImageVaeEncoder[IH,IW] / krea2_patchify."""
    return aspect_lat_w_units(aspect_x100, e_units)


def krea2_lat_h(aspect_x100: Int, e_units: Int) -> Int:
    """Latent H of the krea2 bucket for aspect aspect_x100/100 under edge units
    e_units. See krea2_lat_w."""
    return aspect_lat_h_units(aspect_x100, e_units)
