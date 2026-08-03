# serenitymojo/models/minimax_h3/parity/minimax_h3_vae_geometry_parity.mojo
#
# MiniMax-H3 video VAE geometry parity gate.
#
# Proves serenitymojo/models/minimax_h3/vae_geometry.mojo reproduces the
# reference's tile layout, blend and temporal chunking exactly. Reference:
# diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_vae_geometry_oracle.py with the released spatial/temporal
# factors and stubbed convolutions, so the reference's own control flow decides
# every boundary.
#
# The 2-latent-frame case is absent from the sweeps on purpose: the reference
# raises there (see DECODE FLOOR in vae_geometry.mojo). This gate asserts our
# port raises too, rather than quietly returning a number the reference never
# produced.
#
# Oracle: python3 scripts/minimax_h3_vae_geometry_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_vae_geometry_parity.mojo \
#     -o output/checks/minimax_h3_vae_geometry_parity \
#   && output/checks/minimax_h3_vae_geometry_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.vae_geometry import (
    minimax_h3_blend,
    minimax_h3_decode_num_frames,
    minimax_h3_encode_latent_frames,
    minimax_h3_split_tiles,
    minimax_h3_vae_geometry,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_vae_geometry/vae_geometry_ref.safetensors"


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact_int(mut self, label: String, got: List[Int], want: List[Int]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )

    def exact_f32(mut self, label: String, got: List[Float32], want: List[Float32]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
        if bad == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print("  FAIL", label, bad, "of", len(got), "differ")

    def truth(mut self, label: String, condition: Bool):
        self.checks += 1
        if condition:
            print("  ok  ", label)
        else:
            self.failures += 1
            print("  FAIL", label)


def _sweep_tiles(
    mut report: Report,
    ref st: SafeTensors,
    prefix: String,
    lengths: List[Int],
    tile_size: Int,
    min_overlap: Int,
) raises:
    var counts = List[Int]()
    var starts = List[Int]()
    var sizes = List[Int]()
    var overlaps = List[Int]()
    for i in range(len(lengths)):
        var layout = minimax_h3_split_tiles(lengths[i], tile_size, min_overlap, 16)
        counts.append(len(layout.starts))
        for j in range(len(layout.starts)):
            starts.append(layout.starts[j])
            sizes.append(layout.sizes[j])
        for j in range(len(layout.overlaps)):
            overlaps.append(layout.overlaps[j])
    report.exact_int(prefix + ".num_tiles", counts, _load_i64(st, prefix + ".num_tiles"))
    report.exact_int(prefix + ".starts", starts, _load_i64(st, prefix + ".starts"))
    report.exact_int(prefix + ".sizes", sizes, _load_i64(st, prefix + ".sizes"))
    report.exact_int(prefix + ".overlaps", overlaps, _load_i64(st, prefix + ".overlaps"))


def main() raises:
    print("MiniMax-H3 video VAE geometry parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()
    var geometry = minimax_h3_vae_geometry()

    print("[1] derived geometry")
    var derived = List[Int]()
    derived.append(geometry.spatial_ratio)
    derived.append(geometry.temporal_ratio)
    derived.append(geometry.frame_pre_padding)
    derived.append(geometry.tokens_chunk_size)
    derived.append(geometry.token_overlap)
    derived.append(geometry.frame_overlap)
    report.exact_int("derived", derived, _load_i64(st, "derived"))

    print("[2] tile layout, released 256/64 geometry")
    var lengths = _load_i64(st, "split.lengths")
    _sweep_tiles(report, st, String("split"), lengths, 256, 64)

    print("[3] tile layout, 384/96 — proves the port is not fitted to 256/64")
    _sweep_tiles(report, st, String("split384"), lengths, 384, 96)

    print("[4] linear blend")
    var cases = _load_i64(st, "blend.cases")
    var blend_out = List[Float32]()
    for i in range(len(cases) // 3):
        var a_len = cases[3 * i]
        var b_len = cases[3 * i + 1]
        var extent = cases[3 * i + 2]
        var a = List[Float32]()
        for k in range(a_len):
            a.append(Float32(k))
        var b = List[Float32]()
        for k in range(b_len):
            b.append(Float32(k) + Float32(100.0))
        var out = minimax_h3_blend(a, b, extent)
        for k in range(len(out)):
            blend_out.append(out[k])
    report.exact_f32("blend.out", blend_out, _load_f32(st, "blend.out"))

    print("[5] encode / decode temporal plan")
    var frames = _load_i64(st, "temporal.frames")
    var latents_want = _load_i64(st, "temporal.latent_frames")
    var decoded_want = _load_i64(st, "temporal.decoded_frames")
    var latents_got = List[Int]()
    for i in range(len(frames)):
        latents_got.append(minimax_h3_encode_latent_frames(frames[i], geometry))
    report.exact_int("temporal.latent_frames", latents_got, latents_want)

    var decoded_got = List[Int]()
    for i in range(len(latents_want)):
        if decoded_want[i] < 0:
            # The oracle recorded -1 where the reference raises.
            decoded_got.append(-1)
            continue
        decoded_got.append(minimax_h3_decode_num_frames(latents_want[i], geometry))
    report.exact_int("temporal.decoded_frames", decoded_got, decoded_want)

    print("[6] decode from arbitrary latent lengths")
    var latent_lengths = _load_i64(st, "decode.latent_lengths")
    var decode_got = List[Int]()
    for i in range(len(latent_lengths)):
        decode_got.append(minimax_h3_decode_num_frames(latent_lengths[i], geometry))
    report.exact_int("decode.frames", decode_got, _load_i64(st, "decode.frames"))

    print("[7] decode floor: 2 latent frames must raise, 3 must not")
    var raised = False
    try:
        _ = minimax_h3_decode_num_frames(2, geometry)
    except:
        raised = True
    report.truth("decode(2) raises, matching the reference's empty-cat", raised)
    var three_ok = True
    try:
        _ = minimax_h3_decode_num_frames(3, geometry)
    except:
        three_ok = False
    report.truth("decode(3) succeeds", three_ok)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 VAE geometry parity gate failed")
