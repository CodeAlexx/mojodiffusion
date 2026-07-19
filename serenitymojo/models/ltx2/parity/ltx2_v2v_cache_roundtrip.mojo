# serenitymojo/models/ltx2/parity/ltx2_v2v_cache_roundtrip.mojo
#
# LTX-2 IC-LoRA / V2V reference-cache pairing CPU round-trip gate (P5 unit 1,
# deliverable C). HOST-ONLY -- no DeviceContext, no GPU: writes a synthetic
# paired cache entry with save_safetensors_host, reads the reference latent back
# through the v2v pairing/discovery/validation path
# (serenitymojo/training/ltx2/v2v_cache.mojo), and byte-compares via
# ShardedSafeTensors.tensor_bytes (a raw host span). Also checks the fail-loud
# negatives: missing ref file, channel mismatch, ambiguous latents_* key.
#
# Mirrors musubi's training route: reference latent = a SEPARATE file in a
# reference_cache_directory under the SAME basename as the target, keyed
# `latents_{F}x{H}x{W}_bfloat16` on its own downscaled grid.
#
# Run (no GPU needed):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_v2v_cache_roundtrip.mojo \
#       -o /tmp/ltx2_v2v_cache_roundtrip && /tmp/ltx2_v2v_cache_roundtrip

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors_host, HostTensorDesc
from serenitymojo.io.ffi import sys_mkdirs
from serenitymojo.training.ltx2.v2v_cache import (
    reference_cache_path,
    discover_ref_latent_key,
    assert_reference_latent,
    pair_reference,
)

comptime TMP = "/tmp/ltx2_v2v_cache_gate"
comptime CACHE = TMP + "/cache"
comptime REFCACHE = TMP + "/ref_cache"
comptime CHANNELS = 128       # matches trainer comptime C


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


# Deterministic bf16 payload bytes (content is irrelevant to the gate; we only
# byte-round-trip). 2 bytes/element.
def _make_bytes(numel: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(numel * 2):
        out.append(UInt8((i * 7 + 3) & 0xFF))
    return out^


def _write_latent(path: String, key: String, shape: List[Int]) raises -> List[UInt8]:
    var n = _numel(shape)
    var payload = _make_bytes(n)
    var names = List[String]()
    names.append(key)
    var descs = List[HostTensorDesc]()
    descs.append(HostTensorDesc(STDtype.BF16, shape.copy(), payload.copy()))
    save_safetensors_host(names, descs, path)
    return payload^


def _sh(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def main() raises:
    print("=== LTX-2 v2v reference-cache pairing round-trip (host-only) ===")
    _ = sys_mkdirs(String(CACHE))
    _ = sys_mkdirs(String(REFCACHE))

    var fails = 0

    # target cache: cache/sample0_ltx2.safetensors, key latents_4x9x16_bfloat16
    var tgt_path = String(CACHE) + "/sample0_ltx2.safetensors"
    var tgt_key = String("latents_4x9x16_bfloat16")
    _ = _write_latent(tgt_path, tgt_key, _sh(CHANNELS, 4, 9, 16))

    # reference cache: ref_cache/sample0_ltx2.safetensors (SAME basename), key
    # latents_4x4x8_bfloat16 (ds=2: 9->4, 16->8) shape [128,4,4,8].
    var ref_path = String(REFCACHE) + "/sample0_ltx2.safetensors"
    var ref_key = String("latents_4x4x8_bfloat16")
    var ref_shape = _sh(CHANNELS, 4, 4, 8)
    var ref_bytes = _write_latent(ref_path, ref_key, ref_shape)

    # 1) reference_cache_path == musubi basename route
    var derived = reference_cache_path(tgt_path, String(REFCACHE))
    print("  [path] derived:", derived)
    if derived != ref_path:
        fails += 1
        print("    ^^ FAIL: expected", ref_path)

    # 2) pair_reference (open + discover + validate) returns the ref path
    var paired = pair_reference(tgt_path, String(REFCACHE), CHANNELS)
    print("  [pair] paired:", paired)
    if paired != ref_path:
        fails += 1
        print("    ^^ FAIL: paired path mismatch")

    # 3) key discovery
    var st = ShardedSafeTensors.open(ref_path)
    var disc = discover_ref_latent_key(st)
    print("  [key ] discovered:", disc)
    if disc != ref_key:
        fails += 1
        print("    ^^ FAIL: expected", ref_key)

    # 4) shape/channel validation passes for the good file
    var val_ok = True
    try:
        assert_reference_latent(st, ref_key, CHANNELS)
    except e:
        val_ok = False
        print("    validate raised:", String(e))
    if not val_ok:
        fails += 1
        print("    ^^ FAIL: validation of a good ref latent raised")
    else:
        print("  [val ] rank/channel OK (C=", CHANNELS, ")")

    # 5) BYTE round-trip: tensor_bytes(ref_key) == the bytes we wrote
    var got = st.tensor_bytes(ref_key)
    var exp_len = len(ref_bytes)
    if len(got) != exp_len:
        fails += 1
        print("    ^^ FAIL: ref bytes length", len(got), "!=", exp_len)
    else:
        var mismatch = 0
        for i in range(exp_len):
            if got[i] != ref_bytes[i]:
                mismatch += 1
        print("  [byte] round-trip", exp_len, "bytes, mismatches:", mismatch)
        if mismatch != 0:
            fails += 1
            print("    ^^ FAIL: ref latent bytes not byte-identical")

    # ── fail-loud negatives ──────────────────────────────────────────────────
    # 6) missing reference file -> pair_reference must raise
    var raised_missing = False
    try:
        _ = pair_reference(String(CACHE) + "/does_not_exist_ltx2.safetensors",
                           String(REFCACHE), CHANNELS)
    except e:
        raised_missing = True
    if raised_missing:
        print("  [neg ] missing ref -> raised (OK)")
    else:
        fails += 1
        print("  [neg ] missing ref -> DID NOT RAISE (FAIL)")

    # 7) channel mismatch -> assert_reference_latent must raise
    var raised_chan = False
    try:
        assert_reference_latent(st, ref_key, 64)   # wrong channels
    except e:
        raised_chan = True
    if raised_chan:
        print("  [neg ] channel mismatch -> raised (OK)")
    else:
        fails += 1
        print("  [neg ] channel mismatch -> DID NOT RAISE (FAIL)")

    # 8) ambiguous latents_* keys -> discover_ref_latent_key must raise
    var amb_path = String(REFCACHE) + "/ambiguous_ltx2.safetensors"
    var names2 = List[String]()
    names2.append(String("latents_4x4x8_bfloat16"))
    names2.append(String("latents_2x2x4_bfloat16"))
    var descs2 = List[HostTensorDesc]()
    descs2.append(HostTensorDesc(STDtype.BF16, _sh(CHANNELS, 4, 4, 8),
                                 _make_bytes(_numel(_sh(CHANNELS, 4, 4, 8)))))
    descs2.append(HostTensorDesc(STDtype.BF16, _sh(CHANNELS, 2, 2, 4),
                                 _make_bytes(_numel(_sh(CHANNELS, 2, 2, 4)))))
    save_safetensors_host(names2, descs2, amb_path)
    var st2 = ShardedSafeTensors.open(amb_path)
    var raised_amb = False
    try:
        _ = discover_ref_latent_key(st2)
    except e:
        raised_amb = True
    if raised_amb:
        print("  [neg ] ambiguous latents_* -> raised (OK)")
    else:
        fails += 1
        print("  [neg ] ambiguous latents_* -> DID NOT RAISE (FAIL)")

    if fails > 0:
        raise Error(String("LTX-2 V2V CACHE ROUND-TRIP FAIL: ") + String(fails) + " gate(s)")
    print("LTX-2 V2V CACHE ROUND-TRIP PASS (pairing + key + validate + byte + 3 negatives)")
