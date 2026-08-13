# l2p_resume_moment_fidelity_smoke.mojo — save/resume FULL-moment fidelity gate.
#
# The MJ-1077 / ERI-0217 class: a resume that reloads LoRA A/B but silently ZEROES
# the AdamW moments restarts momentum and corrupts the run.
#
# CONTRACT NOTE (l2p specifics + GAP): the L2P vertical's local decoder is FROZEN
# (models/l2p/local_decoder_train.mojo: decoder weights excluded from LoRA, no
# weight grads) — so there is NO decoder state to resume. The trainable + resumable
# state for L2P is the Z-Image DiT LoRA it drives. l2p has NO l2p-specific
# save/load LoRA state of its own; a native L2P resume would go through the same
# shared moment-state plumbing the zimage LoRA uses. This gate exercises the SHARED
# training/lora_save.mojo path (save_lora_train_state / load_lora_train_state) with
# Z-Image DiT projection SHAPES (q/k/v/o + gate/up/down), proving the round-trip.
# See FINAL NOTE below.
#
# What it proves:
#   1. Build a synthetic Z-Image DiT LoRA set (7 projections × N blocks).
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
#     serenitymojo/models/l2p/parity/l2p_resume_moment_fidelity_smoke.mojo \
#     -o /tmp/gate_l2p

from std.builtin.dtype import DType
from max.gpu.host import DeviceContext

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_train_state,
    load_lora_train_state,
    load_lora_train_state_meta,
    lora_train_state_has_moments,
)


comptime STATE_OUT = "/tmp/l2p_resume_fidelity.state"
comptime BAD_PATH = "/tmp/l2p_resume_fidelity_missing.state"


@fieldwise_init
struct Mism(Copyable, Movable):
    var w: Int
    var m: Int


def _bf(v: BFloat16) -> Float32:
    return v.cast[DType.float32]()


def _val(i: Int, j: Int, salt: Int) -> Float32:
    var k = ((i * 31 + j * 7 + salt) % 17) + 1
    return Float32(k) * Float32(0.0625)


def _mk(rank: Int, in_f: Int, out_f: Int, idx: Int) -> LoraAdapter:
    var a = List[Float32]()
    var ma = List[Float32]()
    var va = List[Float32]()
    for j in range(rank * in_f):
        a.append(_val(idx, j, 1))
        ma.append(_val(idx, j, 3))
        va.append(_val(idx, j, 4))
    var b = List[Float32]()
    var mb = List[Float32]()
    var vb = List[Float32]()
    for j in range(out_f * rank):
        b.append(_val(idx, j, 2))
        mb.append(_val(idx, j, 5))
        vb.append(_val(idx, j, 6))
    return LoraAdapter(a^, b^, rank, in_f, out_f, Float32(2.0), ma^, va^, mb^, vb^)


def _add(
    mut named: List[NamedLora], mut prefixes: List[String], mut idx: Int,
    prefix: String, rank: Int, in_f: Int, out_f: Int,
):
    named.append(NamedLora(prefix, _mk(rank, in_f, out_f, idx)))
    prefixes.append(prefix)
    idx += 1


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
    print("=== l2p resume moment-fidelity gate (Z-Image DiT LoRA; decoder frozen) ===")
    var ctx = DeviceContext()

    # Synthetic Z-Image DiT LoRA set: 2 blocks × 7 projections. D=8, F=16, rank=2.
    var num_blocks = 2
    var rank = 2
    var D = 8
    var F = 16
    var named = List[NamedLora]()
    var prefixes = List[String]()
    var idx = 0
    for bi in range(num_blocks):
        var b = String("blocks.") + String(bi) + "."
        _add(named, prefixes, idx, b + "attn.to_q", rank, D, D)
        _add(named, prefixes, idx, b + "attn.to_k", rank, D, D)
        _add(named, prefixes, idx, b + "attn.to_v", rank, D, D)
        _add(named, prefixes, idx, b + "attn.to_out", rank, D, D)
        _add(named, prefixes, idx, b + "mlp.gate", rank, D, F)
        _add(named, prefixes, idx, b + "mlp.up", rank, D, F)
        _add(named, prefixes, idx, b + "mlp.down", rank, F, D)
    print("built + seeded", len(named), "adapters with nonzero A/B + m/v")

    # meta guard vector [step, seed].
    var meta = List[Float32]()
    meta.append(Float32(3.0))
    meta.append(Float32(1618.0))

    # (1) FULL state round-trip: moments MUST survive.
    _ = save_lora_train_state(named, String(STATE_OUT), ctx, meta.copy())
    var loaded = load_lora_train_state(prefixes, Float32(2.0), String(STATE_OUT), ctx)
    if len(loaded) != len(named):
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

    print("=== PASS: l2p (Z-Image DiT LoRA) shared-state moment plumbing is element-exact ===")
    print("  NOTE: L2P local decoder is FROZEN (no decoder state to resume); the")
    print("        resumable state is the DiT LoRA. Wire save_lora_train_state into")
    print("        the L2P trainer to persist moments across a resume.")
