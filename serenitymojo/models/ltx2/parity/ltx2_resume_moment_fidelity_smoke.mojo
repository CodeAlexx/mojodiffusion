# ltx2_resume_moment_fidelity_smoke.mojo — save/resume FULL-moment fidelity gate.
#
# The MJ-1077 / ERI-0217 class: a resume that reloads LoRA A/B but silently ZEROES
# the AdamW moments restarts momentum and corrupts the run.
#
# CONTRACT NOTE (GAP): ltx2 currently exposes ONLY save_ltx2_lora (a PEFT
# weights-only save) — there is NO save_ltx2_lora_state / load_ltx2_lora_state that
# persists AdamW moments, so a resume today would restart momentum at zero. This
# gate exercises the SHARED training/lora_save.mojo moment-state plumbing
# (save_lora_train_state / load_lora_train_state) with ltx2's real adapter SHAPES
# and musubi prefix naming, proving the round-trip ltx2 WOULD use once a native
# ltx2 state save is wired. See FINAL NOTE below.
#
# What it proves:
#   1. build_ltx2_lora_set builds the flat 4×num_layers adapter carrier.
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
#     serenitymojo/models/ltx2/parity/ltx2_resume_moment_fidelity_smoke.mojo \
#     -o /tmp/gate_ltx2

from std.builtin.dtype import DType
from max.gpu.host import DeviceContext

from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_train_state,
    load_lora_train_state,
    load_lora_train_state_meta,
    lora_train_state_has_moments,
)
from serenitymojo.models.ltx2.ltx2_stack_lora import (
    LTX2LoraSet,
    build_ltx2_lora_set,
)


comptime STATE_OUT = "/tmp/ltx2_resume_fidelity.state"
comptime BAD_PATH = "/tmp/ltx2_resume_fidelity_missing.state"


@fieldwise_init
struct Mism(Copyable, Movable):
    var w: Int
    var m: Int


def _bf(v: BFloat16) -> Float32:
    return v.cast[DType.float32]()


def _val(i: Int, j: Int, salt: Int) -> Float32:
    var k = ((i * 31 + j * 7 + salt) % 17) + 1
    return Float32(k) * Float32(0.0625)


# Reconstruct save_ltx2_lora's musubi module prefixes (4 per block, flat order).
def _ltx2_prefixes(num_layers: Int) -> List[String]:
    var out = List[String]()
    for bi in range(num_layers):
        var b = String("transformer_blocks.") + String(bi) + "."
        out.append(b + "attn1.to_q")
        out.append(b + "attn1.to_k")
        out.append(b + "attn1.to_v")
        out.append(b + "attn1.to_out.0")
    return out^


def _seed(mut set: LTX2LoraSet):
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
    print("=== ltx2 resume moment-fidelity gate (shared-state plumbing) ===")
    var ctx = DeviceContext()

    # Small synthetic stack: 2 layers, D=8, rank=2 -> 8 adapters (q,k,v,o per block).
    var num_layers = 2
    var lora = build_ltx2_lora_set(num_layers, 8, 2, Float32(4.0))
    _seed(lora)
    print("seeded", len(lora.ad), "adapters with nonzero A/B + m/v")

    var prefixes = _ltx2_prefixes(num_layers)
    if len(prefixes) != len(lora.ad):
        raise Error("FAIL: prefix count != adapter count")

    var named = List[NamedLora]()
    for i in range(len(lora.ad)):
        named.append(NamedLora(prefixes[i], lora.ad[i].copy()))

    # meta guard vector [step, seed].
    var meta = List[Float32]()
    meta.append(Float32(7.0))
    meta.append(Float32(1234.0))

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

    print("=== PASS: ltx2 shared-state moment plumbing is element-exact ===")
    print("  NOTE: ltx2 has no native save_ltx2_lora_state yet; wire the shared")
    print("        save_lora_train_state into the ltx2 trainer to close the gap.")
