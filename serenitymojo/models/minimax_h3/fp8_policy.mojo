# serenitymojo/models/minimax_h3/fp8_policy.mojo
#
# MiniMax-H3 unit 14: WHICH tensors become fp8, what that costs, and does the
# result fit one 24 GiB 3090 Ti.
#
# Alex, 2026-08-01: "i want to get running at least local, if possible fp8
# first, then bp16 on this 3090". This module is the arithmetic behind that
# sentence. It is deliberately separate from the encode kernels in
# serenitymojo/ops/fp8_quant.mojo: those know how to turn a row into bytes,
# this knows which rows are ALLOWED to become bytes and what is left over.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE POLICY  (four classes, applied to every one of the 638 planned keys)
#
#   F32_KEEP   the 12 tensors the checkpoint itself stores in fp32. Five
#              prefixes, no exceptions: the two patch projections, the timestep
#              embedder, and the two output heads. 17.2 M params — 0.06 GiB.
#              These are the numerically load-bearing ends of the network and
#              the vendor already decided they do not survive bf16; quantizing
#              them to 1 byte to save 0.05 GiB would be a bad trade at any
#              price.
#
#   BF16_KEEP  norms, biases, small vectors — 212 keys, 0.0006 B params, under
#              1 MiB in total. Per-row fp8 on a [5376] norm gain would save
#              5 KiB and cost a scale lookup in the hot path.
#
#   FP8_ROW    every 2-D projection of >= 1 M elements: qkv, out_proj, fc1,
#              fc2, condition_proj, the refiner's copies. 314 keys, 20.09 B
#              params. This is the class that pays: 37.43 GiB bf16 becomes
#              18.71 GiB of E4M3 bytes plus 12.1 MiB of f32 per-output-row
#              scales.
#
#   ADALN      the 100 adaptive-layernorm projections, 13.01 B params — 39% of
#              the whole model, 24.23 GiB bf16. NOT quantized: EVICTED.
#              See below; this is the entire reason H3 fits.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY adaLN IS EVICTED RATHER THAN QUANTIZED
#
# `adaln_proj.linear` is [96768, 2688] per block and its INPUT is the timestep
# embedding — a function of the step index alone, identical for every token in
# the sequence. A diffusion run visits a fixed, known-in-advance set of
# timesteps. So the whole 13.01 B of adaLN weight can be consumed in ONE
# streamed pass before sampling starts:
#
#     modulation[block][:, step] = adaln_w[block] @ t_emb[:, step] + adaln_b
#
# a [96768, 2688] x [2688, steps] GEMM per block, after which the weights are
# never read again. What stays resident is the RESULT — 96768 x steps x 2 bytes
# per block, 461 MiB for 50 blocks at 50 steps, 231 MiB at 25.
#
# 24.23 GiB of weights collapses to 0.45 GiB of cache. This is the same
# observation the ComfyUI port makes about split modulation; the vendor does
# not ship it, and it is worth more here than any quantization decision,
# because 18.78 + 0.45 fits in 24 and 18.78 + 24.23 does not.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE VERDICT  (from the real 638-key plan, transformer_key_plan.txt)
#
#     bf16, everything resident          61.70 GiB   does not fit, must stream
#     fp8 + adaLN resident               43.01 GiB   does not fit
#     fp8 + adaLN evicted, 50 steps      19.24 GiB   FITS, ~4.7 GiB to spare
#     fp8 + adaLN evicted, 25 steps      19.01 GiB   FITS
#
# The margin is for activations and the packed-sequence attention, which are
# NOT accounted here — this module counts weights and the modulation cache,
# nothing else. Do not read the spare 4.7 GiB as proven headroom; read it as
# the budget the runtime has to live inside.
#
# ─────────────────────────────────────────────────────────────────────────────
# ERROR
#
# The encoder here is the host-side twin of ops/fp8_quant.mojo's GPU kernel:
# same per-output-row symmetric absmax, same E4M3-fn rounding, same saturation.
# It exists because conversion happens while the 61.7 GiB download streams past,
# on the host, one shard at a time — there is no room to stage it on the GPU.
# Bit-exactness against torch.float8_e4m3fn is gated, not assumed
# (minimax_h3_fp8_policy_parity.mojo), including the cases that actually catch
# bugs: ties-to-even at the subnormal boundary, the 8->promote carry, negative
# zero, and saturation past 448.
#
# Mojo 1.0.0b1.

from std.collections import List
from std.math import ldexp, floor


comptime H3_FP8_F32_KEEP = 0
comptime H3_FP8_BF16_KEEP = 1
comptime H3_FP8_ROW = 2
comptime H3_FP8_ADALN = 3

comptime H3_E4M3_MAX = Float32(448.0)

# A 2-D tensor smaller than this stays bf16: the saving is under 0.5 MiB and it
# would put a scale indirection on a cold path for nothing.
comptime H3_FP8_MIN_ELEMENTS = 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Classification.
#
# NOTE ON NAMESPACES: the shapes and dtypes in the converter's plan are listed
# against DIFFUSERS keys, so the prefixes below are the diffusers spelling. The
# original checkpoint calls the same five tensors video_patch_proj.,
# audio_patch_proj., time_embedder., final_layer.video_out. and
# final_layer.audio_out. — loader.mojo bridges the two. The count is the
# invariant that binds them: exactly 12 tensors, and the gate checks it.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_fp8_is_f32_key(key: String) -> Bool:
    return (
        key.startswith("proj_in.")
        or key.startswith("audio_proj_in.")
        or key.startswith("time_embedder.")
        or key.startswith("proj_out.")
        or key.startswith("audio_proj_out.")
    )


def minimax_h3_fp8_class(key: String, rows: Int, cols: Int) -> Int:
    """`cols == 0` means a 1-D tensor.

    Order matters: the fp32 test comes first because time_embedder.linear_2 is
    a [2688, 5376] matrix of 14.5 M elements and would otherwise be swept into
    FP8_ROW by size alone."""
    if minimax_h3_fp8_is_f32_key(key):
        return H3_FP8_F32_KEEP
    if key.find("adaln") >= 0:
        return H3_FP8_ADALN
    if cols > 0 and rows * cols >= H3_FP8_MIN_ELEMENTS:
        return H3_FP8_ROW
    return H3_FP8_BF16_KEEP


# ─────────────────────────────────────────────────────────────────────────────
# Byte accounting.
#
# FP8_ROW carries a companion f32 scale per output row — 4 bytes per row, the
# `quantized_loading.py` convention ops/fp8.mojo already decodes. It is small
# (12.1 MiB over the whole model) but it is not zero and pretending otherwise
# is how a fit estimate turns into an OOM.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_fp8_bytes(cls: Int, rows: Int, cols: Int) -> Int:
    var n = rows * cols if cols > 0 else rows
    if cls == H3_FP8_F32_KEEP:
        return n * 4
    if cls == H3_FP8_ROW:
        return n + rows * 4
    return n * 2


def minimax_h3_adaln_distinct_timesteps(
    steps: Int, has_video_condition: Bool, has_audio_condition: Bool
) -> Int:
    """How many rows the adaLN modulation cache actually needs.

    NOT `steps`. MEASURED from the reference's `build_row_timesteps`, which
    returns `torch.unique` over FOUR values per forward:

        video_timestep                      varies per step
        audio_timestep                      varies per step, its OWN schedule
        max(video_timestep, NOISE_AUG)      keyframe conditioning rows
        1.0                                 audio reference rows, pinned

    H3's dual schedule is the whole point here: video runs shift 12.0 and audio
    runs shift 3.0, so the two are distinct at every step and the cache needs
    2 * steps rows, not `steps`. The two conditioning values are pinned for the
    whole run and contribute at most one row each — `max(t, NOISE_AUG)` collapses
    onto the video row whenever t >= NOISE_AUG, and 1.0 appears only for ref2va.

    This is the correction to the first version of unit 14, which assumed one
    row per step and understated the cache by 2.04x — 0.451 GiB against the real
    0.919 GiB at 50 steps. The fit verdict survives it; the spare does not,
    dropping from 4.76 GiB to 4.29 GiB."""
    var distinct = 2 * steps
    if has_video_condition:
        distinct += 1
    if has_audio_condition:
        distinct += 1
    return distinct


@fieldwise_init
struct MiniMaxH3Fp8Budget(Copyable, Movable):
    """Weight bytes by class, plus the adaLN modulation cache.

    `adaln_out_rows` accumulates the output width of every adaln_proj weight
    matrix; the cache is that many bf16 values per sampling step."""

    var keys: List[Int]
    var params: List[Int]
    var bytes: List[Int]
    var adaln_out_rows: Int

    def __init__(out self):
        self.keys = [0, 0, 0, 0]
        self.params = [0, 0, 0, 0]
        self.bytes = [0, 0, 0, 0]
        self.adaln_out_rows = 0

    def add(mut self, key: String, rows: Int, cols: Int) raises:
        var cls = minimax_h3_fp8_class(key, rows, cols)
        var n = rows * cols if cols > 0 else rows
        self.keys[cls] += 1
        self.params[cls] += n
        self.bytes[cls] += minimax_h3_fp8_bytes(cls, rows, cols)
        if cls == H3_FP8_ADALN and cols > 0:
            self.adaln_out_rows += rows

    def total_params(self) raises -> Int:
        var t = 0
        for i in range(4):
            t += self.params[i]
        return t

    def bf16_all_bytes(self) raises -> Int:
        """What the checkpoint costs with no policy at all — the 61.7 GiB
        baseline. fp32 tensors still cost 4 bytes; they are stored that way."""
        var t = 0
        for i in range(4):
            t += self.params[i] * 4 if i == H3_FP8_F32_KEEP else self.params[i] * 2
        return t

    def resident_bytes(self, distinct_timesteps: Int) raises -> Int:
        """Weights that stay on the GPU under the policy, adaLN evicted to its
        modulation cache.

        Takes DISTINCT TIMESTEPS, not sampling steps — see
        `minimax_h3_adaln_distinct_timesteps`. An earlier version of this took
        `steps` and multiplied by one row per step, which understated the cache
        by 2.04x."""
        var t = self.adaln_out_rows * distinct_timesteps * 2
        for i in range(4):
            if i != H3_FP8_ADALN:
                t += self.bytes[i]
        return t

    def adaln_resident_bytes(self) raises -> Int:
        """The other option: keep adaLN as weights. Kept as a named quantity so
        the comparison in the verdict is computed, not asserted."""
        var t = self.bytes[H3_FP8_ADALN]
        for i in range(4):
            if i != H3_FP8_ADALN:
                t += self.bytes[i]
        return t


# ─────────────────────────────────────────────────────────────────────────────
# Host-side E4M3-fn encode / decode.
#
# Line-for-line the same arithmetic as the GPU kernel in ops/fp8_quant.mojo;
# the parity gate encodes the same values through both paths and through torch
# and requires all three to agree BYTE FOR BYTE. A host encoder that is merely
# "close" to the device one would produce a checkpoint that silently differs
# from every other run.
# ─────────────────────────────────────────────────────────────────────────────
@always_inline
def _rne_int(x: Float32) -> Int:
    """Round-to-nearest-even on a non-negative Float32. Callers have already
    divided by a power of two, so the exact-half comparison is representable
    and `frac == 0.5` is a real test rather than a near-miss."""
    var fl = floor(x)
    var frac = x - fl
    var n = Int(fl)
    if frac > Float32(0.5):
        n += 1
    elif frac == Float32(0.5) and (n & 1) == 1:
        n += 1
    return n


def minimax_h3_e4m3_encode(v: Float32) -> UInt8:
    # Sign comes from the BIT, not from `v < 0`: negative zero is not less than
    # zero, so an ordering test silently encodes -0.0 as 0x00 where torch gives
    # 0x80. Both decode to a zero and no arithmetic here would notice, which is
    # exactly why it survives until something compares bytes — the gate caught
    # it on the first strict run. The same ordering test is still in
    # ops/fp8_quant.mojo's GPU kernel; changing it there would alter bytes in
    # already-converted checkpoints for other models, so that is Alex's call,
    # not a silent edit from here.
    var sign = Int((v.to_bits() >> 31) & 1) << 7
    var a = v
    if a < 0:
        a = -a
    if a != a:
        return UInt8(sign)  # NaN -> +-0; weights are finite, never emit NaN
    if a > H3_E4M3_MAX:
        a = H3_E4M3_MAX
    var byte: Int = 0
    if a < ldexp(Float32(1.0), Int32(-6)):
        # Subnormal: quantum 2^-9, m in [0..8]; m == 8 promotes to exp=1.
        var m = _rne_int(a * Float32(512.0))
        if m >= 8:
            byte = 0x08
        else:
            byte = m
    else:
        var e = -6
        while e < 8 and a >= ldexp(Float32(1.0), Int32(e + 1)):
            e += 1
        var m = _rne_int(ldexp(a, Int32(3 - e)))
        if m == 16:
            m = 8
            e += 1
        if e > 8:
            byte = (15 << 3) | 6  # 448, the largest finite E4M3-fn value
        else:
            byte = ((e + 7) << 3) | (m - 8)
    return UInt8(byte | sign)


def minimax_h3_e4m3_decode(byte: UInt8) -> Float32:
    var b = Int(byte)
    var exp = (b >> 3) & 0xF
    var mant = b & 0x7
    var value: Float32 = 0.0
    if exp == 0:
        value = ldexp(Float32(mant) / Float32(8.0), Int32(-6))
    else:
        value = ldexp(Float32(1.0) + Float32(mant) / Float32(8.0), Int32(exp - 7))
    if (b & 0x80) != 0:
        value = -value
    return value


def minimax_h3_e4m3_row_scale(row: List[Float32]) raises -> Float32:
    """Symmetric absmax over one output row. An all-zero row gets scale 1.0,
    not 0.0 — the row decodes to zeros either way and a zero scale would make
    the encode divide by zero."""
    var m = Float32(0.0)
    for i in range(len(row)):
        var a = row[i]
        if a < 0:
            a = -a
        if a > m:
            m = a
    return m / H3_E4M3_MAX if m > Float32(0.0) else Float32(1.0)


def minimax_h3_e4m3_encode_row(row: List[Float32], scale: Float32) raises -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(len(row)):
        out.append(minimax_h3_e4m3_encode(row[i] / scale))
    return out^
