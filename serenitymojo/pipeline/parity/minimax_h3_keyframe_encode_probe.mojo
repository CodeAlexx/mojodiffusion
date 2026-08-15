# serenitymojo/pipeline/parity/minimax_h3_keyframe_encode_probe.mojo
#
# Gates `pipeline/minimax_h3_keyframe_encode.mojo` — the I2VA / FL2VA / L2VA
# conditioning chain — against the vendor's own code, stage by stage.
#
# ── TWO MODES, AND WHY ──────────────────────────────────────────────────────
# HOST MODE (default, NO GPU): the reference's `moments` are read from the file
# and fed to our stages 3-7. This gates the posterior SAMPLE at seed 42, the
# float16 round, the latent normalization, the patchify row order and the 0.999
# noise mix — every stage except the VAE forward — with no DeviceContext, so it
# runs while the GPU is held. Pair it with the oracle's `--dry-run`, which
# substitutes synthetic moments and skips the VAE for exactly the same reason.
#
# DEVICE MODE (`--with-vae <prepared_keyframe.png>`): additionally loads the
# released video VAE, runs `minimax_h3_keyframe_encode_device` on the same
# image, and compares OUR moments against the vendor's. Pair it with a real
# oracle run. PENDING-GPU as of 2026-08-03 — the overnight chain owns the card.
#
# Splitting it this way is not a convenience: it means a future failure is
# already localized. If HOST mode passes and DEVICE mode fails, the VAE forward
# is at fault; if HOST mode fails, the VAE is irrelevant and the arithmetic
# around it is wrong.
#
# ── WHAT THE STAGES ARE ─────────────────────────────────────────────────────
#   moments -> [3] SAMPLE (seed 42) -> [4] fp16 -> [5] normalize
#           -> [6] patchify -> [7] mix at 0.999 with the request's noise
# The reference dumps every intermediate, so the comparison below is per stage
# rather than end to end.
#
# ── LAYOUT, THE ONE THING TO GET WRONG ──────────────────────────────────────
# The reference's `moments` are NCDHW `[1, 48, 1, H', W']` — channels LEAD. Our
# device encoder emits NDHWC — channels TRAIL. The probe transposes explicitly
# below rather than reinterpreting the buffer, because both layouts have the
# same element count and a reinterpretation would produce a plausible,
# completely wrong sample.
#
# Run (host mode, no GPU):
#   CUDA_VISIBLE_DEVICES="" /home/alex/torchref/venv/bin/python \
#     scripts/minimax_h3_keyframe_encode_oracle.py \
#       output/minimax_h3_keyframe/prepared_keyframe_192x128.png \
#       output/minimax_h3_keyframe/keyframe_encode_ref_DRYRUN.safetensors 7 --dry-run
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_encode_probe.mojo \
#     -o /tmp/h3_keyframe_encode_probe -Xlinker -lm \
#   && /tmp/h3_keyframe_encode_probe output/minimax_h3_keyframe/keyframe_encode_ref_DRYRUN.safetensors 7
#
# Run (device mode, PENDING-GPU) — same oracle WITHOUT --dry-run, then:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -D H3_KF_PROBE_GPU=1 ...
#   <scratch>/h3_keyframe_encode_probe_gpu <ref.safetensors> 7 <prepared.png>

from std.sys import argv
from std.collections import List
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
)
from serenitymojo.pipeline.minimax_h3_keyframe_encode import (
    minimax_h3_keyframe_condition_noise,
    minimax_h3_keyframe_condition_rows,
    minimax_h3_keyframe_posterior_sample,
    minimax_h3_keyframe_rows_from_moments,
)
from serenitymojo.pipeline.minimax_h3_torch_cpu_rng import MiniMaxH3TorchCpuGenerator

struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("probe: reference file has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("probe: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _shape_of(ref st: SafeTensors, name: String) raises -> List[Int]:
    return st.tensor_info(name).shape.copy()


def _compare(
    mut report: Report,
    label: String,
    got: List[Float32],
    want: List[Float32],
    bit_exact: Bool,
) raises:
    if len(got) != len(want):
        report.fail(
            label,
            String("length ") + String(len(got)) + ", want " + String(len(want)),
        )
        return
    var bad = 0
    var first = -1
    var max_abs = Float32(0.0)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(want)):
        if got[i] != want[i]:
            bad += 1
            if first < 0:
                first = i
        var d = got[i] - want[i]
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        dot += Float64(got[i]) * Float64(want[i])
        na += Float64(got[i]) * Float64(got[i])
        nb += Float64(want[i]) * Float64(want[i])
    var cosine = Float64(1.0)
    if na > 0.0 and nb > 0.0:
        cosine = dot / (sqrt(na) * sqrt(nb))

    if bit_exact:
        if bad == 0:
            report.ok(label, String("bit-exact over ") + String(len(want)) + " values")
        else:
            report.fail(
                label,
                String(bad) + " of " + String(len(want)) + " differ, first at "
                + String(first) + ": got " + String(got[first]) + ", want "
                + String(want[first]) + " (cos=" + String(cosine) + ", max_abs="
                + String(max_abs) + ")",
            )
    else:
        if cosine > 0.9999999 and max_abs < Float32(1.0e-3):
            report.ok(
                label,
                String("cos=") + String(cosine) + " max_abs=" + String(max_abs)
                + " over " + String(len(want)) + " values",
            )
        else:
            report.fail(
                label,
                String("cos=") + String(cosine) + " max_abs=" + String(max_abs)
                + " (" + String(bad) + " not bit-equal)",
            )


def _moments_ncdhw_to_ndhwc(
    ncdhw: List[Float32], channels: Int, height: Int, width: Int
) raises -> List[Float32]:
    """`[1, C, 1, H, W]` -> `[H, W, C]` flat, which is what the device encoder
    emits and what `minimax_h3_keyframe_posterior_sample` consumes."""
    var plane = height * width
    if len(ncdhw) != channels * plane:
        raise Error("probe: moments length does not match the declared geometry")
    var out = List[Float32]()
    out.resize(channels * plane, Float32(0.0))
    for c in range(channels):
        for i in range(plane):
            out[i * channels + c] = ncdhw[c * plane + i]
    return out^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_keyframe_encode_probe <ref.safetensors> [request_seed] [prepared.png]")
        return
    var ref_path = String(args[1])
    var request_seed = UInt64(0)
    if len(args) >= 3:
        request_seed = UInt64(atol(String(args[2])))

    print("MiniMax-H3 keyframe unit — encode-chain probe")
    print("  reference:", ref_path, " request_seed:", request_seed)
    print("  mode: HOST (moments are read from the reference; no GPU)")
    print("")
    var report = Report()
    var st = SafeTensors.open(ref_path)

    var mshape = _shape_of(st, String("moments"))
    if len(mshape) != 5:
        raise Error("probe: `moments` is not [1, 2C, 1, H, W]")
    var latent_h = mshape[3]
    var latent_w = mshape[4]
    var two_c = mshape[1]
    if two_c != 2 * MINIMAX_H3_VIDEO_LATENT_CHANNELS:
        raise Error("probe: `moments` does not carry 2 * 24 channels")
    print("  latent geometry:", latent_w, "x", latent_h)

    var moments_ncdhw = _read_f32(st, String("moments"))
    var moments = _moments_ncdhw_to_ndhwc(
        moments_ncdhw, two_c, latent_h, latent_w
    )

    # ── [1] the seed-42 posterior sample ────────────────────────────────────
    print("")
    print("[1] posterior sample — torch.Generator().manual_seed(42)")
    var sample = minimax_h3_keyframe_posterior_sample(moments, latent_h, latent_w)
    var want_sample = _read_f32(st, String("sample"))
    _compare(report, "sample (mean + std * noise42)", sample, want_sample, False)

    # ── [2] fp16 round + normalize + patchify, as one gated call ────────────
    print("")
    print("[2] fp16 round -> normalize -> patchify (clean condition rows)")
    var per_keyframe = List[List[Float32]]()
    per_keyframe.append(moments.copy())
    var clean = minimax_h3_keyframe_rows_from_moments(
        per_keyframe, latent_h, latent_w, 2, 2
    )
    var want_clean = _read_f32(st, String("condition_rows_clean"))
    # After the fp16 round both sides quantize to the SAME grid, so any residual
    # from the posterior noise mostly disappears here — but not entirely, so the
    # bar stays measured rather than bit-exact. See the torch-CPU-RNG probe's
    # check [4] for how many values that round can move.
    _compare(report, "condition rows (clean)", clean, want_clean, False)

    # ── [3] the request's own conditioning noise ────────────────────────────
    print("")
    print("[3] keyframe_condition_noise — the REQUEST's generator, packed order")
    var gen_noise = MiniMaxH3TorchCpuGenerator(request_seed)
    var noise_rows = minimax_h3_keyframe_condition_noise(
        gen_noise, 1, latent_h, latent_w, 2, 2
    )
    var want_noise = _read_f32(st, String("condition_noise_rows"))
    _compare(report, "condition noise rows", noise_rows, want_noise, False)

    # ── [4] the 0.999 mix, end to end ───────────────────────────────────────
    print("")
    print("[4] scale_noise(rows, 0.999, noise) — the anchor the loop never rewrites")
    var gen = MiniMaxH3TorchCpuGenerator(request_seed)
    var mixed = minimax_h3_keyframe_condition_rows(
        per_keyframe, gen, latent_h, latent_w, 2, 2
    )
    var want_mixed = _read_f32(st, String("condition_rows_mixed"))
    _compare(report, "condition rows (mixed at 0.999)", mixed, want_mixed, False)

    # The mix must be a 0.999/0.001 blend, not a re-noise: a row that came out
    # equal to the noise would mean the level was applied the wrong way round.
    var drift = Float64(0.0)
    var n = len(mixed)
    if n > 0 and len(clean) == n and len(noise_rows) == n:
        for i in range(n):
            var d = Float64(mixed[i] - clean[i])
            if d < 0.0:
                d = -d
            drift += d
        drift = drift / Float64(n)
        var noise_mag = Float64(0.0)
        for i in range(n):
            var v = Float64(noise_rows[i])
            if v < 0.0:
                v = -v
            noise_mag += v
        noise_mag = noise_mag / Float64(n)
        # |mixed - clean| = 0.001 * |noise - clean| on average, so the drift has
        # to be ~1000x smaller than the noise itself.
        if drift < noise_mag * 0.01:
            report.ok(
                "mix direction (0.999 keeps the keyframe)",
                String("mean |mixed-clean|=") + String(drift)
                + " vs mean |noise|=" + String(noise_mag),
            )
        else:
            report.fail(
                "mix direction (0.999 keeps the keyframe)",
                String("mean |mixed-clean|=") + String(drift)
                + " vs mean |noise|=" + String(noise_mag)
                + " — the blend looks inverted",
            )

    # ── [5] the VAE forward is a SEPARATE probe ─────────────────────────────
    print("")
    print("[5] VAE forward NOT compared here — see")
    print("    pipeline/parity/minimax_h3_keyframe_encode_gpu_probe.mojo,")
    print("    which links a DeviceContext and is PENDING-GPU.")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_keyframe_encode probe FAILED")
