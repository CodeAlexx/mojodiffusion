# anima_resume_moment_fidelity_smoke.mojo — save/resume FULL-moment fidelity gate.
#
# The MJ-1077 / ERI-0217 class: a resume that reloads LoRA A/B but silently ZEROES
# the AdamW moments restarts momentum and corrupts the run. This gate proves —
# WITHOUT the base model, a cache, or a denoise step — that anima's save/load LoRA
# STATE path carries every moment and param element-for-element.
#
# What it proves:
#   1. build_anima_lora_set builds the flat 10×num_blocks adapter carrier.
#   2. Seed KNOWN NONZERO A/B (BF16) + ALL FOUR moments (F32) on EVERY adapter with
#      a distinct per-(adapter,element) pattern — a zeroed-moment restart or a
#      cross-wired moment (mb->ma etc.) would show up immediately.
#   3. save_anima_lora_state -> load_anima_lora_state restores A/B + m/v with ZERO
#      element mismatches across the whole set.
#   4. lora_train_state_has_moments detects the .state as moment-carrying.
#   5. The RAW SerenityTrainer resume (save_anima_lora -> load_anima_lora_resume)
#      correctly ZEROES moments (weights-only by contract).
#   6. Missing-file probe: loading a nonexistent .state RAISES (does NOT fabricate
#      zeroed moments), and has_moments returns False for it.
#
# Bar: exact — every restored A/B (as BF16) and every restored moment must equal
# the seeded value (0 mismatches). Loud-fail otherwise.
#
# Build (compile-only proof):
#   cd /home/alex/mojodiffusion && pixi run mojo build --optimization-level 2 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/anima/parity/anima_resume_moment_fidelity_smoke.mojo \
#     -o /tmp/gate_anima

from std.builtin.dtype import DType
from max.gpu.host import DeviceContext

from serenitymojo.training.lora_save import lora_train_state_has_moments
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet,
    build_anima_lora_set,
    save_anima_lora,
    save_anima_lora_state,
    load_anima_lora_resume,
    load_anima_lora_state,
    anima_lora_prefixes,
)


comptime STATE_OUT = "/tmp/anima_resume_fidelity.state"
comptime RAW_OUT = "/tmp/anima_resume_fidelity.safetensors"
comptime BAD_PATH = "/tmp/anima_resume_fidelity_missing.state"


@fieldwise_init
struct Mism(Copyable, Movable):
    var w: Int
    var m: Int


def _bf(v: BFloat16) -> Float32:
    return v.cast[DType.float32]()


def _val(i: Int, j: Int, salt: Int) -> Float32:
    var k = ((i * 31 + j * 7 + salt) % 17) + 1
    return Float32(k) * Float32(0.0625)


def _seed(mut set: AnimaLoraSet):
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


def _check(set: AnimaLoraSet, expect_mv: Bool) -> Mism:
    var w_mism = 0
    var m_mism = 0
    for i in range(len(set.ad)):
        for j in range(len(set.ad[i].a)):
            if _bf(set.ad[i].a[j]) != _bf(BFloat16(_val(i, j, 1))):
                w_mism += 1
        for j in range(len(set.ad[i].b)):
            if _bf(set.ad[i].b[j]) != _bf(BFloat16(_val(i, j, 2))):
                w_mism += 1
        for j in range(len(set.ad[i].ma)):
            var e_ma = _val(i, j, 3) if expect_mv else Float32(0.0)
            var e_va = _val(i, j, 4) if expect_mv else Float32(0.0)
            if set.ad[i].ma[j] != e_ma:
                m_mism += 1
            if set.ad[i].va[j] != e_va:
                m_mism += 1
        for j in range(len(set.ad[i].mb)):
            var e_mb = _val(i, j, 5) if expect_mv else Float32(0.0)
            var e_vb = _val(i, j, 6) if expect_mv else Float32(0.0)
            if set.ad[i].mb[j] != e_mb:
                m_mism += 1
            if set.ad[i].vb[j] != e_vb:
                m_mism += 1
    return Mism(w_mism, m_mism)


def main() raises:
    print("=== anima resume moment-fidelity gate ===")
    var ctx = DeviceContext()

    # Small synthetic stack: 2 blocks, D=8, JOINT=4, F=16, rank=2 -> 20 adapters.
    var num_blocks = 2
    var lora = build_anima_lora_set(num_blocks, 8, 4, 16, 2, Float32(4.0))
    _seed(lora)
    print("seeded", len(lora.ad), "adapters with nonzero A/B + m/v")

    # (1) FULL state round-trip: moments MUST survive.
    _ = save_anima_lora_state(lora, String(STATE_OUT), ctx)
    var st = load_anima_lora_state(num_blocks, 2, Float32(4.0), String(STATE_OUT), ctx)
    if len(st.ad) != len(lora.ad):
        raise Error("FAIL: restored state adapter count mismatch")
    var s = _check(st, True)
    print("state round-trip: A/B mismatches=", s.w, " moment mismatches=", s.m)
    if s.w != 0:
        raise Error("FAIL: state A/B not element-exact")
    if s.m != 0:
        raise Error("FAIL: state AdamW moments not element-exact (momentum would restart)")

    # (2) auto-probe primitive: the .state carries moments.
    var probes = anima_lora_prefixes(num_blocks)
    if not lora_train_state_has_moments(String(STATE_OUT), probes[0]):
        raise Error("FAIL: .state not detected as moment-carrying")
    print("probe OK: .state has moments")

    # (3) RAW SerenityTrainer resume path zeroes moments (weights-only by contract).
    _ = save_anima_lora(lora, String(RAW_OUT), ctx)
    var raw = load_anima_lora_resume(num_blocks, 2, Float32(4.0), String(RAW_OUT), ctx)
    var r = _check(raw, False)
    print("raw resume: A/B mismatches=", r.w, " (moments expected zero, mismatches=", r.m, ")")
    if r.w != 0:
        raise Error("FAIL: raw resume A/B not element-exact")
    if r.m != 0:
        raise Error("FAIL: raw resume did not zero moments as its contract requires")

    # (4) missing-file probe: MUST raise, MUST NOT fabricate zeroed moments.
    if lora_train_state_has_moments(String(BAD_PATH), probes[0]):
        raise Error("FAIL: has_moments true for a nonexistent .state")
    var raised = False
    try:
        _ = load_anima_lora_state(num_blocks, 2, Float32(4.0), String(BAD_PATH), ctx)
    except:
        raised = True
    if not raised:
        raise Error("FAIL: loading a missing .state silently succeeded (would restart at zero)")
    print("missing-file probe OK: loader raised, has_moments=False")

    print("=== PASS: anima FULL-moment resume is element-exact ===")
