# lens_saveload_smoke.mojo — Lens LoRA save→load PEFT-key + round-trip parity gate.
#
# PROVES: a saved Lens LoRA checkpoint carries the EXACT Serenity PEFT key set +
# shapes (parity/lens/lora_keys_ref.json, generated from Serenity pr-1510
# LoRAModule.py:563 naming) and round-trips A/B/alpha values bit-exactly through the
# real loader.
#
# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END SAVER UNDER TEST (now wired to the real model):
#   This gate drives the REAL saver — modelSaver/lens/LensLoRASaver.save_lens_lora
#   (→ build_lens_lora_state_dict) — against a REAL LensLoraSet
#   (model/lens/lens_backward.mojo) built from REAL LoraAdapter values
#   (module/LensLoRAModule.mojo: a/b host BF16 lists, scale=alpha/rank). The saver
#   re-derives alpha = scale*rank and emits, for each prefix in
#   lens_lora_target_prefixes (480, block-major/slot-minor, attn-mlp preset):
#       <prefix>.lora_down.weight = A [rank, in]   (BF16)
#       <prefix>.lora_up.weight   = B [out, rank]  (BF16)
#       <prefix>.alpha            = scalar (0-dim)  (= alpha = 16.0)
#   using the same serenitymojo.io.safetensors_writer.save_safetensors backend.
#   The READ side is the REAL loader (load_lens_lora). The gate validates:
#     (1) the PEFT key set + shapes vs the Serenity ref (lora_keys_ref.json), and
#     (2) the real save→load round-trip (A/B bit-exact, alpha preserved).
#
# DTYPE: BF16 storage (the trained boundary). A/B are F32-generated, stored BF16 in
# the adapter, written verbatim by the saver; the round-trip "truth" is the
# BF16-rounded host values so the loaded-vs-saved comparison is bit-exact (|Δ| == 0).
#
# Reference policy: Serenity ONLY (lora_keys_ref.json ← LoRAModule.py:563). No Rust.

from std.memory import ArcPointer
from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.json_header import (
    _Cursor, _parse_string, _parse_int, _parse_int_array, _skip_value,
)
from serenity_trainer.util.config.TrainConfigReader import _read_file_bytes

from serenity_trainer.modelSetup.lensLoraTargets import (
    lens_lora_target_prefixes,
    LORA_IMG_QKV, LORA_TXT_QKV, LORA_TO_OUT, LORA_TO_ADD_OUT,
    LORA_IMG_MLP_W1, LORA_IMG_MLP_W2, LORA_IMG_MLP_W3,
    LORA_TXT_MLP_W1, LORA_TXT_MLP_W2, LORA_TXT_MLP_W3,
    LORA_SLOTS_PER_BLOCK, LENS_N_BLOCKS,
)
from serenity_trainer.module.LensLoRAModule import LoraAdapter
from serenity_trainer.model.lens.lens_backward import LensLoraSet
from serenity_trainer.modelSaver.lens.LensLoRASaver import save_lens_lora
from serenity_trainer.modelLoader.LensModelLoader import load_lens_lora, LensLoraReload


comptime TArc = ArcPointer[Tensor]

comptime DIM        = 1536                       # inner_dim
comptime FF         = 4096                        # int(dim/3*8)
comptime RANK       = 16
comptime ALPHA      = Float32(16.0)
comptime N_ADAPTERS = LENS_N_BLOCKS * LORA_SLOTS_PER_BLOCK   # 480
comptime N_KEYS     = N_ADAPTERS * 3              # down+up+alpha = 1440
comptime REF_PATH   = "trainer/parity/lens/lora_keys_ref.json"
comptime OUT_PATH   = "/tmp/lens_lora_test.safetensors"


# (in_features, out_features) for a per-block LoRA slot — mirrors lens_backward
# _block_slot_dims (attn-mlp preset).  A = [rank, in]; B = [out, rank].
def _slot_dims(slot: Int) raises -> Tuple[Int, Int]:
    if slot == LORA_IMG_QKV or slot == LORA_TXT_QKV:       return (DIM, 3 * DIM)
    if slot == LORA_TO_OUT or slot == LORA_TO_ADD_OUT:     return (DIM, DIM)
    if slot == LORA_IMG_MLP_W1 or slot == LORA_TXT_MLP_W1: return (DIM, FF)
    if slot == LORA_IMG_MLP_W3 or slot == LORA_TXT_MLP_W3: return (DIM, FF)
    if slot == LORA_IMG_MLP_W2 or slot == LORA_TXT_MLP_W2: return (FF, DIM)
    raise Error(String("_slot_dims: bad slot ") + String(slot))


# PCG-style nonzero-distinct draw in roughly [-amp, amp], never exactly 0.
fn _draw(mut s: UInt64, amp: Float32) -> Float32:
    s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    var u = Float32((s >> UInt64(40)) & UInt64(0xFFFF)) / Float32(65536.0)
    var v = (u - Float32(0.5)) * Float32(2.0) * amp
    if v == Float32(0.0):
        v = amp * Float32(0.01)        # force nonzero
    return v


def _gen(n: Int, mut s: UInt64, amp: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(_draw(s, amp))
    return out^


def _shape_eq(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _shape_str(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += String(",")
        out += String(s[i])
    out += String("]")
    return out^


# Parse parity/lens/lora_keys_ref.json → (keys→shape, rank, n_keys).
struct RefKeys(Movable):
    var shapes: Dict[String, List[Int]]
    var rank: Int
    var n_keys: Int

    def __init__(out self, var shapes: Dict[String, List[Int]], rank: Int, n_keys: Int):
        self.shapes = shapes^
        self.rank = rank
        self.n_keys = n_keys


def _load_ref_keys(path: String) raises -> RefKeys:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var shapes = Dict[String, List[Int]]()
    var rank = -1
    var n_keys = -1
    cur.expect(0x7B)              # top-level '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return RefKeys(shapes^, rank, n_keys)
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)         # ':'
        if key == "keys":
            cur.skip_ws()
            cur.expect(0x7B)     # '{'
            cur.skip_ws()
            if cur.peek() == 0x7D:
                cur.advance()
            else:
                while True:
                    var kname = _parse_string(cur)
                    cur.expect(0x3A)
                    var sh = _parse_int_array(cur)
                    shapes[kname] = sh^
                    cur.skip_ws()
                    var ch = cur.peek()
                    if ch == 0x2C:
                        cur.advance()
                        continue
                    if ch == 0x7D:
                        cur.advance()
                        break
                    raise Error(String("ref keys: expected ',' or '}' at byte ")
                                + String(cur.pos))
        elif key == "rank":
            rank = _parse_int(cur)
        elif key == "n_keys":
            n_keys = _parse_int(cur)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:
            cur.advance()
            continue
        if c == 0x7D:
            cur.advance()
            break
        raise Error(String("ref top: expected ',' or '}' at byte ") + String(cur.pos))
    return RefKeys(shapes^, rank, n_keys)


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens LoRA save→load PEFT-key + round-trip parity smoke ===")
    print("  saver under test (real): modelSaver/lens/LensLoRASaver.save_lens_lora")
    print("        end-to-end: REAL LensLoraSet → REAL save_lens_lora → re-read")
    print("        header → REAL load_lens_lora → round-trip.")

    var prefixes = lens_lora_target_prefixes(LENS_N_BLOCKS)
    print("[setup] adapters =", len(prefixes), " expect", N_ADAPTERS)
    if len(prefixes) != N_ADAPTERS:
        raise Error("prefix count mismatch")

    # ── 1) build a REAL LensLoraSet with NONZERO, distinct A AND B ─────────────
    #     (real LoraAdapter: host BF16 a/b, scale = alpha/rank). The saver
    #     re-derives alpha = scale*rank and writes the device tensors itself.
    var block = List[LoraAdapter]()
    var truth_a = List[List[Float32]]()    # BF16-rounded host A per adapter
    var truth_b = List[List[Float32]]()    # BF16-rounded host B per adapter
    var seed = UInt64(0x9E3779B97F4A7C15)
    for i in range(len(prefixes)):
        var slot = i % LORA_SLOTS_PER_BLOCK
        var dims = _slot_dims(slot)
        var in_f = dims[0]
        var out_f = dims[1]
        # A [rank, in]  (B is deliberately NONZERO so round-trip can distinguish)
        var a_f = _gen(RANK * in_f, seed, Float32(0.05))
        var b_f = _gen(out_f * RANK, seed, Float32(0.03))
        var scale = ALPHA / Float32(RANK)
        var za = List[Float32]()
        for _ in range(RANK * in_f): za.append(Float32(0.0))
        var zb = List[Float32]()
        for _ in range(out_f * RANK): zb.append(Float32(0.0))
        var ad = LoraAdapter(
            a_f^, b_f^, RANK, in_f, out_f, scale,
            za.copy(), za^, zb.copy(), zb^,
        )
        # BF16-rounded truth = the adapter's stored bytes (what the saver writes).
        var ta = List[Float32]()
        for j in range(len(ad.a)): ta.append(ad.a[j].cast[DType.float32]())
        var tb = List[Float32]()
        for j in range(len(ad.b)): tb.append(ad.b[j].cast[DType.float32]())
        truth_a.append(ta^)
        truth_b.append(tb^)
        block.append(ad^)
    var lora_set = LensLoraSet(block^, RANK)

    # ── 2) SAVE via the REAL saver (save_lens_lora end-to-end) ─────────────────
    save_lens_lora(lora_set, String(OUT_PATH), ctx)
    print("[save]  save_lens_lora →", String(OUT_PATH))

    # ── 3) re-read the saved file HEADER: enumerate keys + shapes ──────────────
    var st = ShardedSafeTensors.open(String(OUT_PATH))
    var saved = st.names()
    var saved_shapes = Dict[String, List[Int]]()
    for ref nm in saved:
        saved_shapes[nm] = st.tensor_info(nm).shape.copy()
    print("[hdr]   keys in saved file =", len(saved), " expect", N_KEYS)

    # ── 4) load the Serenity PEFT ref key set ────────────────────────────────
    var refk = _load_ref_keys(String(REF_PATH))
    print("[ref]   ref keys =", len(refk.shapes), " refk.rank =", refk.rank,
          " refk.n_keys =", refk.n_keys)

    # ── 5) KEY-SET + SHAPE comparison ──────────────────────────────────────────
    var missing = List[String]()    # in ref, not saved
    var extra = List[String]()      # in saved, not ref
    var shape_mismatch = 0
    var shape_mismatch_examples = List[String]()
    for ref e in refk.shapes.items():
        var k = e.key
        if k not in saved_shapes:
            missing.append(k)
        else:
            if not _shape_eq(saved_shapes[k], refk.shapes[k]):
                shape_mismatch += 1
                if len(shape_mismatch_examples) < 5:
                    shape_mismatch_examples.append(
                        k + String(" saved=") + _shape_str(saved_shapes[k])
                        + String(" ref=") + _shape_str(refk.shapes[k]))
    for ref nm in saved:
        if nm not in refk.shapes:
            extra.append(nm)

    var keyset_ok = (len(missing) == 0) and (len(extra) == 0)
    print("")
    print("── KEY SET ─────────────────────────────────────────────")
    print("  #keys saved        =", len(saved), " (expect", N_KEYS, ")")
    print("  #keys ref          =", len(refk.shapes))
    print("  key-set match      =", "YES" if keyset_ok else "NO")
    print("  #missing (ref\\sav) =", len(missing))
    for i in range(min(5, len(missing))):
        print("      missing:", missing[i])
    print("  #extra   (sav\\ref) =", len(extra))
    for i in range(min(5, len(extra))):
        print("      extra:  ", extra[i])
    print("  #shape mismatches  =", shape_mismatch)
    for i in range(len(shape_mismatch_examples)):
        print("      shape!=:", shape_mismatch_examples[i])

    # ── 6) ROUND-TRIP via the REAL loader (load_lens_lora) ─────────────────────
    var reload = load_lens_lora(String(OUT_PATH), prefixes, ctx)
    print("")
    print("── ROUND-TRIP (real load_lens_lora) ────────────────────")
    print("  loaded adapters    =", len(reload.a), " rank =", reload.rank)

    var max_abs = Float32(0.0)
    var alpha_ok = True
    var alpha_max_err = Float32(0.0)
    for i in range(len(prefixes)):
        var la = reload.a[i][].to_host(ctx)
        var lb = reload.b[i][].to_host(ctx)
        ref ta = truth_a[i]
        ref tb = truth_b[i]
        if len(la) != len(ta) or len(lb) != len(tb):
            raise Error(String("round-trip length mismatch at adapter ") + String(i))
        for j in range(len(la)):
            var d = la[j] - ta[j]
            if d < 0: d = -d
            if d > max_abs: max_abs = d
        for j in range(len(lb)):
            var d = lb[j] - tb[j]
            if d < 0: d = -d
            if d > max_abs: max_abs = d
        var ae = reload.alpha[i] - ALPHA
        if ae < 0: ae = -ae
        if ae > alpha_max_err: alpha_max_err = ae
        if ae > Float32(1.0e-6):
            alpha_ok = False

    print("  max round-trip |Δ| =", max_abs, " (expect 0 for BF16 exact)")
    print("  alpha preserved    =", "YES" if alpha_ok else "NO",
          " (max |Δalpha| =", alpha_max_err, ", expect", ALPHA, ")")

    # ── 7) GATE ────────────────────────────────────────────────────────────────
    var rt_ok = (max_abs == Float32(0.0))
    var count_ok = (len(saved) == N_KEYS) and (len(reload.a) == N_ADAPTERS)
    var gate = keyset_ok and (shape_mismatch == 0) and rt_ok and alpha_ok and count_ok
    print("")
    print("──────────────────────────────────────────────────────────")
    print("  keyset_ok=", keyset_ok, " shapes_ok=", shape_mismatch == 0,
          " roundtrip_ok=", rt_ok, " alpha_ok=", alpha_ok, " count_ok=", count_ok)
    print("  GATE:", "OK" if gate else "FAIL")
    if not gate:
        raise Error("lens_saveload_smoke GATE FAIL")
