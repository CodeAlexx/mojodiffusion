# wan22_resume_moment_fidelity_smoke.mojo — save/resume FULL-moment fidelity gate.
#
# The MJ-1077 / ERI-0217 class: a resume that reloads LoRA A/B but silently ZEROES
# the AdamW moments restarts momentum and corrupts the run.
#
# CONTRACT NOTE (GAP): wan22 currently exposes ONLY save_wan22_lora (a PEFT
# weights-only save) — there is NO save_wan22_lora_state / load_wan22_lora_state
# that persists AdamW moments, so a resume today would restart momentum at zero.
# This gate exercises the SHARED training/lora_save.mojo moment-state plumbing
# (save_lora_train_state / load_lora_train_state) with wan22's real adapter SHAPES
# and block prefix naming, proving the round-trip wan22 WOULD use once a native
# wan22 state save is wired. See FINAL NOTE below.
#
# What it proves:
#   1. build_wan22_lora_set builds the flat 8×num_blocks adapter carrier.
#   2. Seed KNOWN NONZERO A/B (BF16) + ALL FOUR moments (F32) on EVERY adapter.
#   3. save_lora_train_state(named) -> load_lora_train_state(prefixes) restores A/B
#      + m/v with ZERO element mismatches; a __meta__ (step,seed) vector round-trips.
#   4. lora_train_state_has_moments detects the .state as moment-carrying.
#   5. Missing-file probe: loading a nonexistent .state RAISES (does NOT fabricate
#      zeroed moments), and has_moments returns False for it.
#
# Bar: exact — every restored A/B (as BF16), every moment, and the meta vector must
# equal the seeded value (0 mismatches). Loud-fail otherwise.
#
# Build (compile-only proof):
#   cd /home/alex/mojodiffusion && pixi run mojo build --optimization-level 2 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/wan22/parity/wan22_resume_moment_fidelity_smoke.mojo \
#     -o /tmp/gate_wan22

from std.builtin.dtype import DType
from max.gpu.host import DeviceContext

from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_train_state,
    load_lora_train_state,
    load_lora_train_state_meta,
    lora_train_state_has_moments,
)
from serenitymojo.models.wan22.wan22_stack_lora import (
    Wan22LoraSet,
    build_wan22_lora_set,
)


comptime STATE_OUT = "/tmp/wan22_resume_fidelity.state"
comptime BAD_PATH = "/tmp/wan22_resume_fidelity_missing.state"


@fieldwise_init
struct Mism(Copyable, Movable):
    var w: Int
    var m: Int


def _bf(v: BFloat16) -> Float32:
    return v.cast[DType.float32]()


def _val(i: Int, j: Int, salt: Int) -> Float32:
    var k = ((i * 31 + j * 7 + salt) % 17) + 1
    return Float32(k) * Float32(0.0625)


# Reconstruct _wan22_lora_prefixes (8 per block, flat order sa_{q,k,v,o} then
# ca_{q,k,v,o}). Order matches Wan22LoraSet.ad.
def _wan22_prefixes(num_blocks: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_blocks):
        var b = String("blocks.") + String(bi) + "."
        out.append(b + "self_attn.q")
        out.append(b + "self_attn.k")
        out.append(b + "self_attn.v")
        out.append(b + "self_attn.o")
        out.append(b + "cross_attn.q")
        out.append(b + "cross_attn.k")
        out.append(b + "cross_attn.v")
        out.append(b + "cross_attn.o")
        out.append(b + "ffn.0")
        out.append(b + "ffn.2")
    return out^


def _seed(mut set: Wan22LoraSet):
    for i in range(len(set.ad)):
        for j in range(len(set.ad[i].a)):
            set.ad[i].a[j] = BFloat16(_val(i, j, 1))
        for j in range(len(set.ad[i].b)):
            set.ad[i].b[j] = BFloat16(_val(i, j, 2))
        for j in range(len(set.ad[i].ma)):
            set.ad[i].ma[j] = _val(i, j, 3)
            set.ad[i].va[j] = _val(i, j, 4)
        for j in range(len(set.ad[i].mb)):
            set.ad[i].mb[j] = _val(i, j, 5)
            set.ad[i].vb[j] = _val(i, j, 6)


def _check(loaded: List[NamedLora]) -> Mism:
    var w_mism = 0
    var m_mism = 0
    for i in range(len(loaded)):
        ref ad = loaded[i].adapter
        for j in range(len(ad.a)):
            if _bf(ad.a[j]) != _bf(BFloat16(_val(i, j, 1))):
                w_mism += 1
        for j in range(len(ad.b)):
            if _bf(ad.b[j]) != _bf(BFloat16(_val(i, j, 2))):
                w_mism += 1
        for j in range(len(ad.ma)):
            if ad.ma[j] != _val(i, j, 3):
                m_mism += 1
            if ad.va[j] != _val(i, j, 4):
                m_mism += 1
        for j in range(len(ad.mb)):
            if ad.mb[j] != _val(i, j, 5):
                m_mism += 1
            if ad.vb[j] != _val(i, j, 6):
                m_mism += 1
    return Mism(w_mism, m_mism)


def main() raises:
    print("=== wan22 resume moment-fidelity gate (shared-state plumbing) ===")
    var ctx = DeviceContext()

    # Small synthetic stack: 1 block, dim=8, ffn=16, rank=2 -> 10 adapters.
    var num_blocks = 1
    var lora = build_wan22_lora_set(num_blocks, 8, 16, 2, Float32(4.0))
    _seed(lora)
    print("seeded", len(lora.ad), "adapters with nonzero A/B + m/v")

    var prefixes = _wan22_prefixes(num_blocks)
    if len(prefixes) != len(lora.ad):
        raise Error("FAIL: prefix count != adapter count")

    var named = List[NamedLora]()
    for i in range(len(lora.ad)):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))

    # meta guard vector [step, seed].
    var meta = List[Float32]()
    meta.append(Float32(9.0))
    meta.append(Float32(4321.0))

    # (1) FULL state round-trip: moments MUST survive.
    _ = save_lora_train_state(named, String(STATE_OUT), ctx, meta.copy())
    var loaded = load_lora_train_state(prefixes, Float32(2.0), String(STATE_OUT), ctx)
    if len(loaded) != len(lora.ad):
        raise Error("FAIL: restored state adapter count mismatch")
    var s = _check(loaded)
    print("state round-trip: A/B mismatches=", s.w, " moment mismatches=", s.m)
    if s.w != 0:
        raise Error("FAIL: state A/B not element-exact")
    if s.m != 0:
        raise Error("FAIL: state AdamW moments not element-exact (momentum would restart)")

    # (2) meta round-trip.
    var rmeta = load_lora_train_state_meta(String(STATE_OUT), ctx)
    if len(rmeta) != 2 or rmeta[0] != meta[0] or rmeta[1] != meta[1]:
        raise Error("FAIL: __meta__ (step,seed) did not round-trip")
    print("meta round-trip OK: step=", rmeta[0], " seed=", rmeta[1])

    # (3) auto-probe primitive: the .state carries moments.
    if not lora_train_state_has_moments(String(STATE_OUT), prefixes[0]):
        raise Error("FAIL: .state not detected as moment-carrying")
    print("probe OK: .state has moments")

    # (4) missing-file probe: MUST raise, MUST NOT fabricate zeroed moments.
    if lora_train_state_has_moments(String(BAD_PATH), prefixes[0]):
        raise Error("FAIL: has_moments true for a nonexistent .state")
    var raised = False
    try:
        _ = load_lora_train_state(prefixes, Float32(2.0), String(BAD_PATH), ctx)
    except:
        raised = True
    if not raised:
        raise Error("FAIL: loading a missing .state silently succeeded (would restart at zero)")
    print("missing-file probe OK: loader raised, has_moments=False")

    print("=== PASS: wan22 shared-state moment plumbing is element-exact ===")
    print("  NOTE: wan22 has no native save_wan22_lora_state yet; wire the shared")
    print("        save_lora_train_state into the wan22 trainer to close the gap.")
