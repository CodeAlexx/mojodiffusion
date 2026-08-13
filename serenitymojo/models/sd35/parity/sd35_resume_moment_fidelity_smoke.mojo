# sd35_resume_moment_fidelity_smoke.mojo — save/resume FULL-moment fidelity gate.
#
# The MJ-1077 / ERI-0217 class: a resume that reloads LoRA A/B but silently ZEROES
# the AdamW moments restarts momentum and corrupts the run. This gate proves —
# WITHOUT the base model, a cache, or a denoise step — that sd35's LoRA STATE save
# carries every moment and param element-for-element across a save/reload.
#
# CONTRACT NOTE: sd35 exposes save_sd35_lora_state (which writes A/B + AdamW m/v via
# the shared save_lora_train_state) but has NO model-specific load_sd35_lora_state.
# The resume load therefore goes through the SHARED load_lora_train_state with the
# exact sd35 module prefixes (reconstructed here to match _sd35_named_loras). This
# is the same round-trip ernie/anima wrap in their own load_*_lora_state.
#
# What it proves:
#   1. build_sd35_lora_set builds the flat 8×depth adapter carrier.
#   2. Seed KNOWN NONZERO A/B (BF16) + ALL FOUR moments (F32) on EVERY adapter with
#      a distinct per-(adapter,element) pattern.
#   3. save_sd35_lora_state -> load_lora_train_state(sd35_prefixes) restores A/B +
#      m/v with ZERO element mismatches across the whole set.
#   4. lora_train_state_has_moments detects the .state as moment-carrying.
#   5. Missing-file probe: loading a nonexistent .state RAISES (does NOT fabricate
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
#     serenitymojo/models/sd35/parity/sd35_resume_moment_fidelity_smoke.mojo \
#     -o /tmp/gate_sd35

from std.builtin.dtype import DType
from max.gpu.host import DeviceContext

from serenitymojo.training.lora_save import (
    NamedLora,
    load_lora_train_state,
    lora_train_state_has_moments,
)
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35LoraSet,
    build_sd35_lora_set,
    save_sd35_lora_state,
)


comptime STATE_OUT = "/tmp/sd35_resume_fidelity.state"
comptime BAD_PATH = "/tmp/sd35_resume_fidelity_missing.state"


@fieldwise_init
struct Mism(Copyable, Movable):
    var w: Int
    var m: Int


def _bf(v: BFloat16) -> Float32:
    return v.cast[DType.float32]()


def _val(i: Int, j: Int, salt: Int) -> Float32:
    var k = ((i * 31 + j * 7 + salt) % 17) + 1
    return Float32(k) * Float32(0.0625)


# Reconstruct the EXACT flat prefix list that _sd35_named_loras emits (8 per joint
# block, ctx-then-x, attn qkv/proj then mlp fc1/fc2). Order matches SD35LoraSet.ad.
def _sd35_prefixes(depth: Int) -> List[String]:
    var out = List[String]()
    for bi in range(depth):
        var bp = String("transformer.joint_blocks.") + String(bi)
        out.append(bp + ".context_block.attn.qkv")
        out.append(bp + ".context_block.attn.proj")
        out.append(bp + ".context_block.mlp.fc1")
        out.append(bp + ".context_block.mlp.fc2")
        out.append(bp + ".x_block.attn.qkv")
        out.append(bp + ".x_block.attn.proj")
        out.append(bp + ".x_block.mlp.fc1")
        out.append(bp + ".x_block.mlp.fc2")
    return out^


def _seed(mut set: SD35LoraSet):
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


# Loaded set comes back as List[NamedLora] from the shared loader.
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
    print("=== sd35 resume moment-fidelity gate ===")
    var ctx = DeviceContext()

    # Small synthetic stack: depth=2, D=8, MLP=16, rank=2 -> 16 adapters.
    var depth = 2
    var lora = build_sd35_lora_set(depth, 8, 16, 2, Float32(4.0))
    _seed(lora)
    print("seeded", len(lora.ad), "adapters with nonzero A/B + m/v")

    var prefixes = _sd35_prefixes(depth)
    if len(prefixes) != len(lora.ad):
        raise Error(
            String("FAIL: reconstructed prefix count ") + String(len(prefixes))
            + " != adapter count " + String(len(lora.ad))
        )

    # (1) FULL state round-trip: moments MUST survive.
    _ = save_sd35_lora_state(lora, String(STATE_OUT), ctx)
    var loaded = load_lora_train_state(prefixes, Float32(2.0), String(STATE_OUT), ctx)
    if len(loaded) != len(lora.ad):
        raise Error("FAIL: restored state adapter count mismatch")
    var s = _check(loaded)
    print("state round-trip: A/B mismatches=", s.w, " moment mismatches=", s.m)
    if s.w != 0:
        raise Error("FAIL: state A/B not element-exact")
    if s.m != 0:
        raise Error("FAIL: state AdamW moments not element-exact (momentum would restart)")

    # (2) auto-probe primitive: the .state carries moments.
    if not lora_train_state_has_moments(String(STATE_OUT), prefixes[0]):
        raise Error("FAIL: .state not detected as moment-carrying")
    print("probe OK: .state has moments")

    # (3) missing-file probe: MUST raise, MUST NOT fabricate zeroed moments.
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

    print("=== PASS: sd35 FULL-moment resume is element-exact ===")
