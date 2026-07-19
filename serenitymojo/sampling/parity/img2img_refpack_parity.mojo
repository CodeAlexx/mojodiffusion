# sampling/parity/img2img_refpack_parity.mojo — ref-pack token + position-id
# layout parity vs a hand oracle, on synthetic latents (weight-free).
#
# Checks:
#   (1) noise_img_ids exact: row r*LW+c == [0, row, col, 0].
#   (2) prepare_reference_ids exact: row == [t_offset, row, col, 0].
#   (3) combined_img_ids exact: [noise_ids ; ref_ids] along dim 0, shape
#       [2*N,4], and cos >= 0.999 vs flattened oracle.
#   (4) pack_latent_tokens: [1,C,LH,LW] -> [1,LH*LW,C] matches the NCHW->NHWC
#       reshape oracle exactly; cos >= 0.999.
#   (5) prepare_combined_tokens: [noise ; ref] along seq, shape [1,2N,C], exact.
#
# PARITY-BITROT GUARD: `--bitrot` swaps row/col in the *oracle* id layout, so
# the (correct) module output disagrees → assertion fires → exit 1.
#
# Build / run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/sampling/parity/img2img_refpack_parity.mojo
#   (append `--bitrot` for the deliberate-wrong exit-1 demo)

from std.collections import List
from std.sys import argv
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.img2img_refpack import (
    build_noise_img_ids,
    prepare_reference_ids,
    prepare_combined_img_ids,
    pack_latent_tokens,
    prepare_combined_tokens,
)


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def _cos(a: List[Float32], b: List[Float32]) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        return 1.0
    return dot / denom


def _check_eq(name: String, got: Float32, expected: Float32) raises:
    if _abs(got - expected) > 1.0e-5:
        raise Error(
            name + " got=" + String(got) + " expected=" + String(expected)
        )


def _check_shape(name: String, got: List[Int], e0: Int, e1: Int) raises:
    if len(got) != 2 or got[0] != e0 or got[1] != e1:
        raise Error(name + " shape mismatch")


def main() raises:
    var args = argv()
    var bitrot = False
    for i in range(len(args)):
        if args[i] == "--bitrot":
            bitrot = True

    var ctx = DeviceContext()
    print("=== img2img ref-pack token/id layout parity ===" + (" [BITROT]" if bitrot else ""))

    var LH = 3
    var LW = 4
    var N = LH * LW
    var C = 2  # small latent channel count for the token-pack oracle
    var t_offset: Float32 = 10.0

    # ---- (1) noise_img_ids ----
    var noise_ids = build_noise_img_ids(LH, LW, ctx)
    var noise_ids_h = noise_ids.to_host(ctx)
    _check_shape("noise_ids", noise_ids.shape(), N, 4)
    for row in range(LH):
        for col in range(LW):
            var base = (row * LW + col) * 4
            var orow = Float32(col) if bitrot else Float32(row)
            var ocol = Float32(row) if bitrot else Float32(col)
            _check_eq("noise t", noise_ids_h[base + 0], 0.0)
            _check_eq("noise row", noise_ids_h[base + 1], orow)
            _check_eq("noise col", noise_ids_h[base + 2], ocol)
            _check_eq("noise l", noise_ids_h[base + 3], 0.0)
    print("  noise_img_ids exact [0,row,col,0]  OK")

    # ---- (2) prepare_reference_ids ----
    var ref_ids = prepare_reference_ids(LH, LW, t_offset, ctx)
    var ref_ids_h = ref_ids.to_host(ctx)
    _check_shape("ref_ids", ref_ids.shape(), N, 4)
    for row in range(LH):
        for col in range(LW):
            var base = (row * LW + col) * 4
            _check_eq("ref t", ref_ids_h[base + 0], t_offset)
            _check_eq("ref row", ref_ids_h[base + 1], Float32(row))
            _check_eq("ref col", ref_ids_h[base + 2], Float32(col))
            _check_eq("ref l", ref_ids_h[base + 3], 0.0)
    print("  prepare_reference_ids exact [t_offset,row,col,0]  OK")

    # ---- (3) combined_img_ids ----
    var comb_ids = prepare_combined_img_ids(LH, LW, t_offset, ctx)
    var comb_ids_h = comb_ids.to_host(ctx)
    _check_shape("comb_ids", comb_ids.shape(), 2 * N, 4)
    # Oracle: [noise rows ; ref rows].
    var comb_oracle = List[Float32]()
    for row in range(LH):
        for col in range(LW):
            comb_oracle.append(0.0)
            comb_oracle.append(Float32(row))
            comb_oracle.append(Float32(col))
            comb_oracle.append(0.0)
    for row in range(LH):
        for col in range(LW):
            comb_oracle.append(t_offset)
            comb_oracle.append(Float32(row))
            comb_oracle.append(Float32(col))
            comb_oracle.append(0.0)
    for i in range(len(comb_oracle)):
        _check_eq("comb id" + String(i), comb_ids_h[i], comb_oracle[i])
    var cos_ids = _cos(comb_ids_h, comb_oracle)
    print("  combined_img_ids cos vs oracle = " + String(cos_ids))
    if cos_ids < 0.999:
        raise Error("combined_img_ids cos < 0.999")
    # First noise row T=0, first ref row T=t_offset.
    _check_eq("comb noise[0].t", comb_ids_h[0], 0.0)
    _check_eq("comb ref[0].t", comb_ids_h[N * 4 + 0], t_offset)
    print("  combined_img_ids = [noise ; ref] along dim0  OK")

    # ---- (4) pack_latent_tokens ----
    # Build [1, C, LH, LW] with a distinct value per (c, h, w): v = c*100 + h*10 + w.
    var lat_v = List[Float32]()
    for c in range(C):
        for h in range(LH):
            for w in range(LW):
                lat_v.append(Float32(c * 100 + h * 10 + w))
    var lat_shape = List[Int]()
    lat_shape.append(1); lat_shape.append(C); lat_shape.append(LH); lat_shape.append(LW)
    var latent = Tensor.from_host(lat_v, lat_shape^, STDtype.F32, ctx)

    var packed = pack_latent_tokens(latent, ctx)
    var packed_h = packed.to_host(ctx)
    var psh = packed.shape()
    if len(psh) != 3 or psh[0] != 1 or psh[1] != N or psh[2] != C:
        raise Error("pack_latent_tokens shape mismatch")
    # Oracle: token (h*LW+w), channel c == lat[c,h,w].
    var pack_oracle = List[Float32]()
    for h in range(LH):
        for w in range(LW):
            for c in range(C):
                pack_oracle.append(Float32(c * 100 + h * 10 + w))
    for i in range(len(pack_oracle)):
        _check_eq("pack" + String(i), packed_h[i], pack_oracle[i])
    var cos_pack = _cos(packed_h, pack_oracle)
    print("  pack_latent_tokens cos vs oracle = " + String(cos_pack))
    if cos_pack < 0.999:
        raise Error("pack_latent_tokens cos < 0.999")
    print("  pack_latent_tokens [1,C,LH,LW]->[1,N,C] exact  OK")

    # ---- (5) prepare_combined_tokens ----
    # Reference tokens: a second packed latent (offset values so we can tell apart).
    var ref_lat_v = List[Float32]()
    for c in range(C):
        for h in range(LH):
            for w in range(LW):
                ref_lat_v.append(Float32(c * 100 + h * 10 + w) + 1000.0)
    var ref_lat_shape = List[Int]()
    ref_lat_shape.append(1); ref_lat_shape.append(C); ref_lat_shape.append(LH); ref_lat_shape.append(LW)
    var ref_latent = Tensor.from_host(ref_lat_v, ref_lat_shape^, STDtype.F32, ctx)
    var ref_tokens = pack_latent_tokens(ref_latent, ctx)

    var comb_tokens = prepare_combined_tokens(packed, ref_tokens, ctx)
    var comb_tokens_h = comb_tokens.to_host(ctx)
    var ctsh = comb_tokens.shape()
    if len(ctsh) != 3 or ctsh[0] != 1 or ctsh[1] != 2 * N or ctsh[2] != C:
        raise Error("prepare_combined_tokens shape mismatch")
    # First N*C are noise tokens, next N*C are ref tokens (+1000).
    for i in range(N * C):
        _check_eq("ctok noise" + String(i), comb_tokens_h[i], pack_oracle[i])
        _check_eq("ctok ref" + String(i), comb_tokens_h[N * C + i], pack_oracle[i] + 1000.0)
    print("  prepare_combined_tokens [noise ; ref] along seq exact  OK")

    print("PASS: img2img ref-pack token/id layout parity")
