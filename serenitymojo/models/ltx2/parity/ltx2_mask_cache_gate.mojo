# serenitymojo/models/ltx2/parity/ltx2_mask_cache_gate.mojo
#
# LTX2 inpaint mask-cache loader gate (P5.5 unit 2). HOST-ONLY, no GPU: writes a
# synthetic mask safetensors (save_safetensors_host) and reads it back through
# load_mask_tokens, asserting the >0.5 threshold + frame->h->w token order +
# fail-louds (missing key, numel mismatch). Grid 2x3x4 (seq=24).
#
# Run (no GPU):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_mask_cache_gate.mojo \
#       -o /tmp/ltx2_mask_cache_gate && /tmp/ltx2_mask_cache_gate

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors_host, HostTensorDesc
from serenitymojo.io.ffi import sys_mkdirs
from serenitymojo.training.ltx2.mask_cache import load_mask_tokens, mask_cache_path

comptime TMP = "/tmp/ltx2_mask_cache_gate"
comptime NF = 2
comptime NH = 3
comptime NW = 4
comptime SEQ = NF * NH * NW      # 24


def _write_mask_f32(path: String, vals: List[Float32], shape: List[Int]) raises:
    var bytes = List[UInt8]()
    for i in range(len(vals)):
        var bits = vals[i].to_bits()
        bytes.append(UInt8(bits & 0xFF))
        bytes.append(UInt8((bits >> 8) & 0xFF))
        bytes.append(UInt8((bits >> 16) & 0xFF))
        bytes.append(UInt8((bits >> 24) & 0xFF))
    var names = List[String]()
    names.append(String("mask"))
    var descs = List[HostTensorDesc]()
    descs.append(HostTensorDesc(STDtype.F32, shape.copy(), bytes^))
    save_safetensors_host(names, descs, path)


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def main() raises:
    print("=== LTX2 inpaint mask-cache loader gate (host-only, 2x3x4) ===")
    _ = sys_mkdirs(String(TMP))
    var fails = 0

    # mask [NF,NH,NW]: frame 0 all 0.0; frame 1 all 1.0; plus threshold probes:
    # frame0 token (h0,w0) = 0.7 (->True), frame0 (h0,w1)=0.3 (->False).
    var vals = List[Float32]()
    for f in range(NF):
        for h in range(NH):
            for w in range(NW):
                var v = Float32(1.0) if f == 1 else Float32(0.0)
                if f == 0 and h == 0 and w == 0:
                    v = Float32(0.7)
                if f == 0 and h == 0 and w == 1:
                    v = Float32(0.3)
                vals.append(v)
    var p = String(TMP) + "/sample0_ltx2.safetensors"
    _write_mask_f32(p, vals, _sh3(NF, NH, NW))

    var m = load_mask_tokens(p, NF, NH, NW)
    if len(m) != SEQ:
        fails += 1
        print("  [FAIL] len", len(m), "!=", SEQ)
    else:
        print("  [PASS] len == seq 24")
    # frame 1 (tokens 12..23) all True
    var f1_ok = True
    for t in range(NH * NW, SEQ):
        if not m[t]:
            f1_ok = False
    if f1_ok:
        print("  [PASS] frame 1 all True (token order frame->h->w)")
    else:
        fails += 1
        print("  [FAIL] frame 1 not all True")
    # threshold probes: token 0 (0.7)->True, token 1 (0.3)->False
    if m[0] and (not m[1]):
        print("  [PASS] threshold >0.5 (0.7->T, 0.3->F)")
    else:
        fails += 1
        print("  [FAIL] threshold wrong: m[0]=", m[0], " m[1]=", m[1])
    # frame 0 rest (tokens 2..11) False
    var f0_ok = True
    for t in range(2, NH * NW):
        if m[t]:
            f0_ok = False
    if f0_ok:
        print("  [PASS] frame 0 (non-probe) all False")
    else:
        fails += 1
        print("  [FAIL] frame 0 leaked True")

    # path route
    if mask_cache_path(String("/a/b/cache/sample0_ltx2.safetensors"), String("/a/b/maskc")) \
       == String("/a/b/maskc/sample0_ltx2.safetensors"):
        print("  [PASS] mask_cache_path same-basename route")
    else:
        fails += 1
        print("  [FAIL] mask_cache_path")

    # fail-loud: numel mismatch (ask for a different grid)
    var raised_numel = False
    try:
        _ = load_mask_tokens(p, NF, NH, NW + 1)
    except e:
        raised_numel = True
    if raised_numel:
        print("  [PASS] numel mismatch -> raised")
    else:
        fails += 1
        print("  [FAIL] numel mismatch did not raise")

    # fail-loud: missing 'mask' key
    var pbad = String(TMP) + "/nomask_ltx2.safetensors"
    var names2 = List[String](); names2.append(String("latents_2x3x4_bfloat16"))
    var descs2 = List[HostTensorDesc]()
    var by = List[UInt8]()
    for _ in range(SEQ * 2):
        by.append(UInt8(0))
    descs2.append(HostTensorDesc(STDtype.BF16, _sh3(NF, NH, NW), by^))
    save_safetensors_host(names2, descs2, pbad)
    var raised_key = False
    try:
        _ = load_mask_tokens(pbad, NF, NH, NW)
    except e:
        raised_key = True
    if raised_key:
        print("  [PASS] missing 'mask' key -> raised")
    else:
        fails += 1
        print("  [FAIL] missing 'mask' key did not raise")

    if fails > 0:
        raise Error(String("LTX2 MASK CACHE GATE FAIL: ") + String(fails))
    print("LTX2 MASK CACHE LOADER GATE PASS (threshold + token order + path + fail-louds)")
