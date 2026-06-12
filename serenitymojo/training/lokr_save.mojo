# training/lokr_save.mojo — save / reopen TRAINED LoKr adapters in the
# UPSTREAM LyCORIS key convention (pip lycoris_lora 3.4.0, the T2.F oracle).
#
# MEASURED upstream schema (lycoris/modules/lokr.py weight_list /
# custom_state_dict — the factors are bare nn.Parameters, so NO `.weight`
# suffix on any LoKr key):
#
#       "<prefix>.lokr_w1"     BF16 [out_l, in_m]       (W1 full)        OR
#       "<prefix>.lokr_w1_a"   BF16 [out_l, rank]       (W1 factored)    +
#       "<prefix>.lokr_w1_b"   BF16 [rank, in_m]
#       "<prefix>.lokr_w2"     BF16 [out_k, in_n]       (W2 full)        OR
#       "<prefix>.lokr_w2_a"   BF16 [out_k, rank]       (W2 factored)    +
#       "<prefix>.lokr_w2_b"   BF16 [rank, in_n]
#       "<prefix>.alpha"       F32 [1]
#
# 2026-06-11 T2.F-2 skeptic FIX: keys previously carried a `.weight` suffix
# (citing the now-deleted EDv2 lycoris.rs) — upstream uses the bare spellings
# above, so the old files were ecosystem-unloadable (same bug class as the
# LoHa/OFT T2.F fixes). Gate: lycoris_family_load_check.py loads a Mojo-saved
# file through LokrModule.make_module_from_state_dict BIT-EXACT.
#
# Both W1-full and W1-factored (decompose_both) paths now ship. The conv-only
# `.lokr_t2` Tucker key lives in tucker_save.mojo (lora_mid), not here — this
# is the LINEAR LoKr path.
#
# Like LoHa (and unlike plain LoRA, which omits `.alpha`), the LyCORIS LoKr
# convention DOES carry a per-module `.alpha` scalar (scale=alpha/rank).
# MEASURED upstream quirk (lokr.py:209-211): when BOTH W1 and W2 are full,
# upstream forces scale=1 regardless of alpha — the saved alpha round-trips
# but upstream ignores it in that configuration (lokr_adapter.mojo mirrors it).
#
# Mojo 0.26.x: `def` not `fn`; move-only Tensor → ArcPointer; STDtype.F32 a value.

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.lokr_adapter import LoKrAdapter


# A trained LoKr adapter paired with the base-weight prefix it adapts.
@fieldwise_init
struct NamedLoKr(Copyable, Movable):
    var prefix: String
    var adapter: LoKrAdapter


def _bf16_2d(var values: List[BFloat16], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host_bf16(values^, sh^, ctx)


def _f32_scalar(value: Float32, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    v.append(value)
    var sh = List[Int]()
    sh.append(1)
    return Tensor.from_host(v^, sh^, STDtype.F32, ctx)


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each LoKr adapter's W1 + (W2 full OR W2_a/W2_b) + alpha. Returns
# the number of ADAPTERS written.
# ─────────────────────────────────────────────────────────────────────────────
def save_lokr_peft(
    adapters: List[NamedLoKr], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_lokr_peft: refusing to write an empty LoKr file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nk in adapters:
        var a = nk.adapter.copy()
        var OL = a.out_l; var OK = a.out_k; var IM = a.in_m; var INn = a.in_n; var R = a.rank

        if a.w1_factored:
            if len(a.w1a) != OL * R:
                raise Error(String("save_lokr_peft: w1a numel mismatch for '") + nk.prefix + "'")
            if len(a.w1b) != R * IM:
                raise Error(String("save_lokr_peft: w1b numel mismatch for '") + nk.prefix + "'")
            names.append(nk.prefix + ".lokr_w1_a")
            tensors.append(ArcPointer(_bf16_2d(a.w1a.copy(), OL, R, ctx)))
            names.append(nk.prefix + ".lokr_w1_b")
            tensors.append(ArcPointer(_bf16_2d(a.w1b.copy(), R, IM, ctx)))
        else:
            if len(a.w1) != OL * IM:
                raise Error(String("save_lokr_peft: w1 numel ") + String(len(a.w1)) + " != out_l*in_m for '" + nk.prefix + "'")
            names.append(nk.prefix + ".lokr_w1")
            tensors.append(ArcPointer(_bf16_2d(a.w1.copy(), OL, IM, ctx)))

        if a.w2_factored:
            if len(a.w2a) != OK * R:
                raise Error(String("save_lokr_peft: w2a numel mismatch for '") + nk.prefix + "'")
            if len(a.w2b) != R * INn:
                raise Error(String("save_lokr_peft: w2b numel mismatch for '") + nk.prefix + "'")
            names.append(nk.prefix + ".lokr_w2_a")
            tensors.append(ArcPointer(_bf16_2d(a.w2a.copy(), OK, R, ctx)))
            names.append(nk.prefix + ".lokr_w2_b")
            tensors.append(ArcPointer(_bf16_2d(a.w2b.copy(), R, INn, ctx)))
        else:
            if len(a.w2) != OK * INn:
                raise Error(String("save_lokr_peft: w2 numel mismatch for '") + nk.prefix + "'")
            names.append(nk.prefix + ".lokr_w2")
            tensors.append(ArcPointer(_bf16_2d(a.w2.copy(), OK, INn, ctx)))

        names.append(nk.prefix + ".alpha")
        tensors.append(ArcPointer(_f32_scalar(a.alpha, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


# ── REOPEN: read one tensor by name to a host F32 list ───────────────────────
def _read_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    if info.dtype == STDtype.F32:
        if info.size % 4 != 0:
            raise Error(String("_read_f32: bad F32 byte size for ") + name)
        var fp = bytes.unsafe_ptr().bitcast[Float32]()
        var out = List[Float32]()
        for i in range(info.size // 4):
            out.append(fp[i])
        return out^
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def _read_bf16(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[BFloat16]:
    return _f32_to_bf16_list(_read_f32(st, name, ctx))


def _has_key(st: SafeTensors, name: String) -> Bool:
    try:
        var _info = st.tensor_info(name)
        return True
    except:
        return False


# A read-back of one LoKr module: W1 + (W2 full OR W2_a/W2_b) + shapes + alpha.
@fieldwise_init
struct LoKrReadback(Copyable, Movable):
    var w1: List[BFloat16]
    var w1a: List[BFloat16]
    var w1b: List[BFloat16]
    var w1_factored: Bool
    var w2: List[BFloat16]
    var w2a: List[BFloat16]
    var w2b: List[BFloat16]
    var w2_factored: Bool
    var out_l: Int
    var out_k: Int
    var in_m: Int
    var in_n: Int
    var rank: Int
    var alpha: Float32


# Reopen one module's LoKr keys. W1 is factored iff `.lokr_w1_a` present
# (then out_l,in_m,rank from w1a/w1b shapes); else full `.lokr_w1`. W2 is
# factored iff `.lokr_w2_a` present; else full `.lokr_w2`. Asserts
# the present keys + alpha and cross-checks shapes. When BOTH are factored their
# ranks must agree (single LoKr rank).
def read_lokr_module(prefix: String, path: String, ctx: DeviceContext) raises -> LoKrReadback:
    var st = SafeTensors.open(path)

    # ── W1 ──
    var w1 = List[BFloat16]()
    var w1a = List[BFloat16]()
    var w1b = List[BFloat16]()
    var w1_factored = _has_key(st, prefix + ".lokr_w1_a")
    var OL: Int
    var IM: Int
    var R1: Int = 0
    if w1_factored:
        var i1a = st.tensor_info(prefix + ".lokr_w1_a")
        var i1b = st.tensor_info(prefix + ".lokr_w1_b")
        OL = i1a.shape[0]
        R1 = i1a.shape[1]
        IM = i1b.shape[1]
        if i1b.shape[0] != R1:
            raise Error("read_lokr_module: w1b rows != rank")
        w1a = _read_bf16(st, prefix + ".lokr_w1_a", ctx)
        w1b = _read_bf16(st, prefix + ".lokr_w1_b", ctx)
    else:
        var i1 = st.tensor_info(prefix + ".lokr_w1")
        OL = i1.shape[0]
        IM = i1.shape[1]
        w1 = _read_bf16(st, prefix + ".lokr_w1", ctx)

    # ── W2 ──
    var w2 = List[BFloat16]()
    var w2a = List[BFloat16]()
    var w2b = List[BFloat16]()
    var factored = _has_key(st, prefix + ".lokr_w2_a")
    var OK: Int
    var INn: Int
    var R: Int
    if factored:
        var ia = st.tensor_info(prefix + ".lokr_w2_a")
        var ib = st.tensor_info(prefix + ".lokr_w2_b")
        OK = ia.shape[0]
        R = ia.shape[1]
        INn = ib.shape[1]
        if ib.shape[0] != R:
            raise Error("read_lokr_module: w2b rows != rank")
        w2a = _read_bf16(st, prefix + ".lokr_w2_a", ctx)
        w2b = _read_bf16(st, prefix + ".lokr_w2_b", ctx)
    else:
        var i2 = st.tensor_info(prefix + ".lokr_w2")
        OK = i2.shape[0]
        INn = i2.shape[1]
        R = 0
        w2 = _read_bf16(st, prefix + ".lokr_w2", ctx)

    # Single LoKr rank: when both factored, the ranks must match.
    if w1_factored and factored and R1 != R:
        raise Error("read_lokr_module: W1 rank != W2 rank (single LoKr rank expected)")
    # Carry a meaningful rank even when only W1 is factored.
    if not factored and w1_factored:
        R = R1

    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_lokr_module: .alpha must be a 1-element tensor")

    return LoKrReadback(
        w1^, w1a^, w1b^, w1_factored,
        w2^, w2a^, w2b^, factored,
        OL, OK, IM, INn, R, alpha_h[0],
    )
