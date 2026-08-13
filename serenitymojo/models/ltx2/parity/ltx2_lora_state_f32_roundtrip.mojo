# ltx2_lora_state_f32_roundtrip.mojo — BIT-EXACT round-trip gate for the LTX-2
# F32-exact LoRA trainer state (save_lora_train_state_f32 /
# load_lora_train_state_f32). Proves the resume boundary carries the F32 masters
# and AdamW moments with ZERO loss (the MJ-1108-class bf16 rounding is confined to
# the bf16 `.weight` compat copies, never the `.master` payload).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_lora_state_f32_roundtrip.mojo \
#     -o /tmp/ltx2_f32_gate
#   /tmp/ltx2_f32_gate
#
# Also asserts lora_train_state_has_f32_masters() distinguishes an F32-state file
# (masters present -> True) from an OLD-era save_lora_train_state file (bf16 A/B +
# moments, no masters -> False).

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sin, cos

from serenitymojo.training.lora_save import (
    F32LoraState, NamedLora,
    save_lora_train_state_f32, save_lora_train_state,
    load_lora_train_state_f32, load_lora_train_state_meta,
    lora_train_state_has_f32_masters,
)
from serenitymojo.training.train_step import LoraAdapter


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


# NON-DEGENERATE sinusoidal fill (never a modular pattern — bf16 must NOT be able
# to alias distinct F32 elements to the same value and slip a mismatch past us).
def _sinfill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var fi = Float32(i)
        out.append(sin(Float32(0.7) * fi + phase) + Float32(0.3) * cos(Float32(1.3) * fi))
    return out^


# Fraction of elements NOT exactly representable in bf16 — i.e. carrying
# mantissa bits a bf16 hop would destroy. The bit-compare below is only a
# meaningful bf16-hop tripwire if this fraction is high on the SEEDS (test
# data validity) and stays high on the LOADED masters (no hop happened).
def _frac_sub_bf16(vals: List[Float32]) -> Float32:
    var cnt = 0
    for i in range(len(vals)):
        var rt = Float32(BFloat16(vals[i]))
        if rt != vals[i]:
            cnt += 1
    return Float32(cnt) / Float32(len(vals))


def _bitcmp(orig: List[Float32], got: List[Float32], tag: String, k: Int) raises:
    if len(orig) != len(got):
        raise Error(String("len mismatch ") + tag + " state " + String(k)
                    + ": " + String(len(orig)) + " vs " + String(len(got)))
    for i in range(len(orig)):
        # BIT-EXACT: masters/moments are F32 all the way through, so plain
        # Float32 equality must hold for EVERY element.
        if orig[i] != got[i]:
            raise Error(String("BIT MISMATCH ") + tag + " state " + String(k)
                        + " elem " + String(i))


def main() raises:
    var ctx = DeviceContext()
    var path = String("/tmp/ltx2_f32_state_gate.safetensors")
    var old_path = String("/tmp/ltx2_f32_state_gate_old.safetensors")
    comptime R = 4
    comptime IN = 16
    comptime OUT = 16
    var na = R * IN     # 64
    var nb = OUT * R    # 64

    var prefixes = List[String]()
    prefixes.append("diffusion_model.transformer_blocks.0.attn1.to_q")
    prefixes.append("diffusion_model.transformer_blocks.0.attn1.to_k")
    prefixes.append("diffusion_model.transformer_blocks.1.attn2.to_v")

    # originals — kept intact so the read-back compare is against the source F32.
    var oa = List[List[Float32]]()
    var ob = List[List[Float32]]()
    var oma = List[List[Float32]]()
    var ova = List[List[Float32]]()
    var omb = List[List[Float32]]()
    var ovb = List[List[Float32]]()
    for k in range(3):
        var fk = Float32(k) * Float32(0.37)
        oa.append(_sinfill(na, fk + Float32(0.10)))
        ob.append(_sinfill(nb, fk + Float32(0.50)))
        oma.append(_sinfill(na, fk + Float32(0.90)))
        ova.append(_sinfill(na, fk + Float32(1.30)))
        omb.append(_sinfill(nb, fk + Float32(1.70)))
        ovb.append(_sinfill(nb, fk + Float32(2.10)))

    var states = List[F32LoraState]()
    for k in range(3):
        states.append(F32LoraState(
            prefixes[k].copy(),
            oa[k].copy(), ob[k].copy(),
            oma[k].copy(), ova[k].copy(), omb[k].copy(), ovb[k].copy(),
            R, IN, OUT,
        ))

    # SUB-BF16 SEED GUARD: >90% of every seeded list must be bf16-UNrepresentable,
    # or the bit-compare could pass even through a bf16 hop (the flaw that makes
    # the older moment-fidelity smoke blind to exactly this bug class).
    for k in range(3):
        _require(_frac_sub_bf16(oa[k]) > Float32(0.9),
                 String("seed A not sub-bf16 enough, state ") + String(k))
        _require(_frac_sub_bf16(ob[k]) > Float32(0.9),
                 String("seed B not sub-bf16 enough, state ") + String(k))
        _require(_frac_sub_bf16(oma[k]) > Float32(0.9),
                 String("seed ma not sub-bf16 enough, state ") + String(k))

    var meta = List[Float32]()
    meta.append(Float32(7.0))
    meta.append(Float32(42.0))
    var n = save_lora_train_state_f32(states, path, ctx, meta^)
    _require(n == 3, String("expected 3 states written, got ") + String(n))

    # read back + verify meta
    var loaded = load_lora_train_state_f32(prefixes, path, ctx)
    _require(len(loaded) == 3, String("expected 3 states loaded, got ") + String(len(loaded)))
    var meta_r = load_lora_train_state_meta(path, ctx)
    _require(len(meta_r) == 2, String("expected 2 meta values, got ") + String(len(meta_r)))
    _require(meta_r[0] == Float32(7.0), "meta[0] (step) != 7.0")
    _require(meta_r[1] == Float32(42.0), "meta[1] (seed) != 42.0")

    # BIT-EXACT every element of all six lists + rank/in/out
    for k in range(3):
        ref st = loaded[k]
        _require(st.rank == R, String("rank mismatch state ") + String(k))
        _require(st.in_f == IN, String("in mismatch state ") + String(k))
        _require(st.out_f == OUT, String("out mismatch state ") + String(k))
        _bitcmp(oa[k], st.a, String("A"), k)
        _bitcmp(ob[k], st.b, String("B"), k)
        _bitcmp(oma[k], st.ma, String("ma"), k)
        _bitcmp(ova[k], st.va, String("va"), k)
        _bitcmp(omb[k], st.mb, String("mb"), k)
        _bitcmp(ovb[k], st.vb, String("vb"), k)
        # LOADED-DATA GUARD: the masters that came BACK are still sub-bf16 —
        # a bf16 hop anywhere in write/read would collapse this to 0.
        _require(_frac_sub_bf16(st.a) > Float32(0.9),
                 String("loaded A master lost sub-bf16 bits, state ") + String(k))
        _require(_frac_sub_bf16(st.b) > Float32(0.9),
                 String("loaded B master lost sub-bf16 bits, state ") + String(k))

    # has_f32_masters: TRUE on the F32-state file we just wrote.
    _require(
        lora_train_state_has_f32_masters(path, prefixes[0]),
        "expected f32 masters detected on the f32-state file",
    )

    # Write an OLD-era state (bf16 A/B + F32 moments, NO masters) and confirm the
    # probe returns FALSE — the resume path must warm-fall-back for these.
    var named = List[NamedLora]()
    for k in range(3):
        named.append(NamedLora(
            prefixes[k].copy(),
            LoraAdapter(
                oa[k].copy(), ob[k].copy(), R, IN, OUT, Float32(1.0),
                oma[k].copy(), ova[k].copy(), omb[k].copy(), ovb[k].copy(),
            ),
        ))
    var meta2 = List[Float32]()
    meta2.append(Float32(3.0))
    meta2.append(Float32(42.0))
    var n_old = save_lora_train_state(named, old_path, ctx, meta2^)
    _require(n_old == 3, String("expected 3 old-era states written, got ") + String(n_old))
    _require(
        not lora_train_state_has_f32_masters(old_path, prefixes[0]),
        "old-era state must NOT report f32 masters",
    )

    print("ALL PASS")
