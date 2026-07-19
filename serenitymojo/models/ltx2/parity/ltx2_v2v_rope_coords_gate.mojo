# serenitymojo/models/ltx2/parity/ltx2_v2v_rope_coords_gate.mojo
#
# LTX-2 IC-LoRA / V2V two-grid RoPE COORDS gate (P5 unit 2). HOST-ONLY -- no
# DeviceContext, no GPU: exercises _build_v2v_coords (the pre-rope combined pixel
# coords the trainer's v2v head feeds _compute_rope) and asserts the musubi
# two-grid geometry (ltx2_train_network.py:3386-3424): ref grid PREPENDED, ref
# H/W coords *reference_downscale so both grids CO-LOCATE from origin 0 under the
# fixed BASE_HW normalization, target grid does NOT continue from ref.
#
# The rope-from-coords math (_compute_rope) is the already-gated single-grid
# spine path (musubi-exact bf16, cos 0.9999943) and is exercised on-device by the
# S=320 block parity gate; this gate isolates the NEW two-grid coord geometry.
#
# Run (no GPU needed):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_v2v_rope_coords_gate.mojo \
#       -o /tmp/ltx2_v2v_rope_coords_gate && /tmp/ltx2_v2v_rope_coords_gate

from serenitymojo.models.ltx2.ltx2_video_stack import _build_v2v_coords

comptime FR = Float64(25.0)


# image512 v2v preset (matches the oracle default): ref 1x8x8, tgt 1x16x16, ds=2
comptime REF_NF = 1
comptime REF_NH = 8
comptime REF_NW = 8
comptime TGT_NF = 1
comptime TGT_NH = 16
comptime TGT_NW = 16
comptime DS = 2
comptime S_REF = REF_NF * REF_NH * REF_NW      # 64
comptime S_TGT = TGT_NF * TGT_NH * TGT_NW      # 256
comptime P = S_REF + S_TGT                      # 320


def _c(coords: List[Float32], d: Int, tok: Int, bound: Int) -> Float32:
    return coords[(d * P + tok) * 2 + bound]


def main() raises:
    print("=== LTX-2 v2v two-grid RoPE coords gate (host-only, image512) ===")
    print("  ref", REF_NF, "x", REF_NH, "x", REF_NW, "=", S_REF,
          " tgt", TGT_NF, "x", TGT_NH, "x", TGT_NW, "=", S_TGT, " ds", DS, " P", P)
    var coords = _build_v2v_coords(
        REF_NF, REF_NH, REF_NW, TGT_NF, TGT_NH, TGT_NW, DS, FR)

    var fails = 0

    # 0) length
    if len(coords) != 3 * P * 2:
        fails += 1
        print("  [FAIL] coords length", len(coords), "!=", 3 * P * 2)
    else:
        print("  [PASS] coords length 3*P*2 =", 3 * P * 2)

    # 1) origin: ref token 0 (h=0,w=0) H/W start == 0
    if _c(coords, 1, 0, 0) == Float32(0.0) and _c(coords, 2, 0, 0) == Float32(0.0):
        print("  [PASS] ref token 0 at origin (H,W start = 0)")
    else:
        fails += 1
        print("  [FAIL] ref token 0 not at origin")

    # 2) ref H/W coords multiplied by DS: ref token (h=1,w=0) -> tok=REF_NW
    #    H start = 1*32*DS
    var ref_tok_h1 = 1 * REF_NW + 0
    var exp_h = Float32(1.0) * Float32(32.0) * Float32(DS)     # 64
    if _c(coords, 1, ref_tok_h1, 0) == exp_h:
        print("  [PASS] ref H coord *DS (h=1 ->", exp_h, ")")
    else:
        fails += 1
        print("  [FAIL] ref H coord *DS: got", _c(coords, 1, ref_tok_h1, 0), "exp", exp_h)

    # 3) target H coords NOT scaled: target token (h=2,w=0) -> tok=S_REF+2*TGT_NW
    #    H start = 2*32
    var tgt_tok_h2 = S_REF + 2 * TGT_NW + 0
    var exp_th = Float32(2.0) * Float32(32.0)                  # 64
    if _c(coords, 1, tgt_tok_h2, 0) == exp_th:
        print("  [PASS] target H coord unscaled (h=2 ->", exp_th, ")")
    else:
        fails += 1
        print("  [FAIL] target H coord: got", _c(coords, 1, tgt_tok_h2, 0), "exp", exp_th)

    # 4) CO-LOCATION: ref token (h=1) H-start == target token (h=1*DS) H-start
    if _c(coords, 1, ref_tok_h1, 0) == _c(coords, 1, tgt_tok_h2, 0):
        print("  [PASS] co-location: ref(h=1)*DS == target(h=2) H-start =",
              _c(coords, 1, ref_tok_h1, 0))
    else:
        fails += 1
        print("  [FAIL] co-location broken")

    # 5) target starts from ORIGIN (does NOT continue from ref): first target
    #    token (S_REF) H/W start == 0
    if _c(coords, 1, S_REF, 0) == Float32(0.0) and _c(coords, 2, S_REF, 0) == Float32(0.0):
        print("  [PASS] target grid starts from origin (no ref offset)")
    else:
        fails += 1
        print("  [FAIL] target grid does not start from origin")

    # 6) single-frame temporal (nf=1 both grids): START clamps to 0 (fsc=1-8<0),
    #    END = bf16(1/frame_rate) (fec=1); the phase is IDENTICAL for every token
    #    (all f=0) so temporal never breaks ref/target co-location.
    var t_start0 = _c(coords, 0, 0, 0)
    var t_end0 = _c(coords, 0, 0, 1)
    var temporal_varies = 0
    for tok in range(P):
        if _c(coords, 0, tok, 0) != t_start0 or _c(coords, 0, tok, 1) != t_end0:
            temporal_varies += 1
    if t_start0 == Float32(0.0) and temporal_varies == 0:
        print("  [PASS] single-frame temporal: start=0, constant across all tokens (end=",
              t_end0, ")")
    else:
        fails += 1
        print("  [FAIL] temporal start", t_start0, "varies-across-tokens", temporal_varies)

    # 7) ref W coords *DS too: ref token (h=0,w=3) -> tok=3, W start = 3*32*DS
    var ref_tok_w3 = 0 * REF_NW + 3
    var exp_w = Float32(3.0) * Float32(32.0) * Float32(DS)     # 192
    if _c(coords, 2, ref_tok_w3, 0) == exp_w:
        print("  [PASS] ref W coord *DS (w=3 ->", exp_w, ")")
    else:
        fails += 1
        print("  [FAIL] ref W coord *DS: got", _c(coords, 2, ref_tok_w3, 0), "exp", exp_w)

    if fails > 0:
        raise Error(String("LTX-2 V2V ROPE COORDS GATE FAIL: ") + String(fails) + " check(s)")
    print("LTX-2 V2V ROPE COORDS GATE PASS (co-location + prepend + origin + temporal)")
