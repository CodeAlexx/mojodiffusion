# krea2_resume_moment_fidelity_smoke.mojo — mechanism gate for the krea2 fast-arm
# FULL-moment resume (save/resume audit loss #1 + MJ-1077).
#
# Proves, WITHOUT the 25GB base model or a real cache, the exact plumbing the
# audit's gate A depends on:
#   1. save_lora_train_state writes A/B + adam_m/adam_v (+ optional __meta__).
#   2. lora_train_state_has_moments distinguishes a FULL .state (has .adam_m) from
#      a plain PEFT LoRA (no moments) — the auto-probe primitive.
#   3. load_lora_train_state restores A/B + m/v VALUE-EXACT.
#   4. load_lora_train_state_meta round-trips the __meta__ guard vector.
#   5. lora_adamw_plain_device_state_init SEEDS the resident device dev_m/dev_v
#      from the restored adapters (offload_to_host mirrors dev_m/dev_v back and
#      they equal the restored host moments) — the fast arm continues on the saved
#      moments, NOT zeros. THIS is the fix for the "moments restart" gap.
#
# Bar: value-exact (max_abs == 0) for A/B/m/v across save->load->device-seed, and
# the meta vector reproduces exactly. Loud-fail on any mismatch.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/krea2/parity/krea2_resume_moment_fidelity_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_peft,
    save_lora_train_state,
    load_lora_train_state,
    load_lora_train_state_meta,
    lora_train_state_has_moments,
)
from serenitymojo.training.lora_adamw_plain_fused import (
    lora_adamw_plain_device_state_init,
    lora_adamw_plain_device_state_offload_to_host,
)


def _mk_adapter(rank: Int, in_f: Int, out_f: Int, seed: Int) -> LoraAdapter:
    """Known-value adapter: A/B and ALL FOUR moments filled with distinct nonzero
    patterns so a zeroed-moment restart would show up immediately."""
    var a = List[Float32]()
    var ma = List[Float32]()
    var va = List[Float32]()
    for i in range(rank * in_f):
        a.append(Float32(seed) * 0.01 + Float32(i) * 0.001)
        ma.append(Float32(seed) * 0.1 + Float32(i) * 0.002 + 0.5)     # nonzero
        va.append(Float32(seed) * 0.2 + Float32(i) * 0.003 + 0.25)    # nonzero
    var b = List[Float32]()
    var mb = List[Float32]()
    var vb = List[Float32]()
    for i in range(out_f * rank):
        b.append(Float32(seed) * 0.02 + Float32(i) * 0.0011)
        mb.append(Float32(seed) * 0.11 + Float32(i) * 0.0021 + 0.7)   # nonzero
        vb.append(Float32(seed) * 0.22 + Float32(i) * 0.0031 + 0.35)  # nonzero
    return LoraAdapter(a^, b^, rank, in_f, out_f, 1.0, ma^, va^, mb^, vb^)


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error(String("len mismatch ") + String(len(a)) + " vs " + String(len(b)))
    var m: Float32 = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


# BF16 storage round-trips exactly against the BF16-rounded originals; compare the
# adapters' own stored BF16 (a.a/a.b are BFloat16 lists) as F32.
def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def main() raises:
    print("=== krea2 resume moment-fidelity mechanism gate ===")
    var ctx = DeviceContext()

    var prefixes = List[String]()
    prefixes.append(String("diffusion_model.blocks.0.attn.wq"))
    prefixes.append(String("diffusion_model.blocks.0.attn.wk"))

    var orig = List[LoraAdapter]()
    orig.append(_mk_adapter(4, 8, 6, 11))
    orig.append(_mk_adapter(4, 6, 8, 23))

    var named = List[NamedLora]()
    for i in range(len(prefixes)):
        named.append(NamedLora(prefixes[i], orig[i].copy()))

    var state_path = String("/tmp/krea2_resume_fidelity.state")
    var peft_path = String("/tmp/krea2_resume_fidelity.safetensors")

    # meta guard vector [step, seed, n_samples, ltmax, cache_hash24].
    var meta = List[Float32]()
    meta.append(3.0)
    meta.append(42.0)
    meta.append(70.0)
    meta.append(384.0)
    meta.append(1234.0)

    # 1) write the FULL .state (with meta) AND a plain PEFT file.
    var named2 = List[NamedLora]()
    for i in range(len(prefixes)):
        named2.append(NamedLora(prefixes[i], orig[i].copy()))
    _ = save_lora_train_state(named, state_path, ctx, meta.copy())
    _ = save_lora_peft(named2, peft_path, ctx)
    print("wrote .state ->", state_path, " and PEFT ->", peft_path)

    # 2) auto-probe primitive: .state HAS moments, PEFT does NOT.
    var probe = prefixes[0]
    if not lora_train_state_has_moments(state_path, probe):
        raise Error("FAIL: .state not detected as a train_state (missing .adam_m probe)")
    if lora_train_state_has_moments(peft_path, probe):
        raise Error("FAIL: plain PEFT file wrongly detected as a train_state")
    print("probe OK: .state=has-moments, PEFT=no-moments")

    # 3) restore A/B + m/v and value-check vs originals.
    var loaded = load_lora_train_state(prefixes, 1.0, state_path, ctx)
    if len(loaded) != len(orig):
        raise Error("FAIL: restored adapter count mismatch")
    for i in range(len(orig)):
        var lo = loaded[i].adapter.copy()
        var og = orig[i].copy()
        var mad_a = _max_abs(_bf16_to_f32(lo.a), _bf16_to_f32(og.a))
        var mad_b = _max_abs(_bf16_to_f32(lo.b), _bf16_to_f32(og.b))
        var mad_ma = _max_abs(lo.ma, og.ma)
        var mad_va = _max_abs(lo.va, og.va)
        var mad_mb = _max_abs(lo.mb, og.mb)
        var mad_vb = _max_abs(lo.vb, og.vb)
        print("  adapter", i, " max_abs A=", mad_a, " B=", mad_b,
              " ma=", mad_ma, " va=", mad_va, " mb=", mad_mb, " vb=", mad_vb)
        if mad_a != 0.0 or mad_b != 0.0:
            raise Error(String("FAIL: A/B not exact for adapter ") + String(i))
        if mad_ma != 0.0 or mad_va != 0.0 or mad_mb != 0.0 or mad_vb != 0.0:
            raise Error(String("FAIL: AdamW moment not exact for adapter ") + String(i))
    print("restore OK: A/B + adam_m/adam_v value-exact")

    # 4) meta round-trip.
    var rmeta = load_lora_train_state_meta(state_path, ctx)
    if len(rmeta) != len(meta):
        raise Error(String("FAIL: meta len ") + String(len(rmeta)) + " != " + String(len(meta)))
    var mad_meta = _max_abs(rmeta, meta)
    if mad_meta != 0.0:
        raise Error(String("FAIL: meta not exact, max_abs=") + String(mad_meta))
    print("meta round-trip OK:", rmeta[0], rmeta[1], rmeta[2], rmeta[3], rmeta[4])
    # PEFT file has no meta -> empty.
    var pmeta = load_lora_train_state_meta(peft_path, ctx)
    if len(pmeta) != 0:
        raise Error("FAIL: PEFT file unexpectedly carried a __meta__ vector")

    # 5) THE FIX: device AdamW init seeds dev_m/dev_v from the RESTORED moments.
    var loaded_adapters = List[LoraAdapter]()
    for i in range(len(loaded)):
        loaded_adapters.append(loaded[i].adapter.copy())
    var state = lora_adamw_plain_device_state_init(loaded_adapters, 0, len(loaded_adapters), ctx)
    var snap = lora_adamw_plain_device_state_offload_to_host(state, ctx)

    # snap.m / snap.v are F32, flat [a0.ma, a0.mb, a1.ma, a1.mb, ...].
    var mp = snap.m.unsafe_ptr().bitcast[Float32]()
    var vp = snap.v.unsafe_ptr().bitcast[Float32]()
    var off = 0
    var worst_m: Float32 = 0.0
    var worst_v: Float32 = 0.0
    for i in range(len(loaded_adapters)):
        var lo = loaded_adapters[i].copy()
        var na = len(lo.ma)
        var nb = len(lo.mb)
        for j in range(na):
            var dm = mp[off + j] - lo.ma[j]; dm = -dm if dm < 0 else dm
            var dv = vp[off + j] - lo.va[j]; dv = -dv if dv < 0 else dv
            if dm > worst_m: worst_m = dm
            if dv > worst_v: worst_v = dv
        off += na
        for j in range(nb):
            var dm = mp[off + j] - lo.mb[j]; dm = -dm if dm < 0 else dm
            var dv = vp[off + j] - lo.vb[j]; dv = -dv if dv < 0 else dv
            if dm > worst_m: worst_m = dm
            if dv > worst_v: worst_v = dv
        off += nb
    print("device seed check: worst dev_m diff=", worst_m, " worst dev_v diff=", worst_v)
    if worst_m != 0.0 or worst_v != 0.0:
        raise Error("FAIL: device AdamW state NOT seeded with restored moments (would restart at zero)")

    print("=== PASS: FULL-moment resume plumbing is byte/value-exact ===")
    print("  (.state carries m/v; load restores exact; device init seeds dev_m/dev_v")
    print("   from the restored moments — the fast arm continues, not restarts)")
