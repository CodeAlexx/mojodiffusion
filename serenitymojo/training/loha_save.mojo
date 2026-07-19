# training/loha_save.mojo — save / reopen TRAINED LoHa adapters in the UPSTREAM
# LyCORIS key convention (pip lycoris_lora 3.4.0, the T2.F campaign oracle).
#
# MEASURED upstream schema (lycoris/modules/loha.py LohaModule.state_dict, Linear):
#
#       "<prefix>.hada_w1_a"   BF16 [out, rank]    (NO `.weight` suffix)
#       "<prefix>.hada_w1_b"   BF16 [rank, in]
#       "<prefix>.hada_w2_a"   BF16 [out, rank]
#       "<prefix>.hada_w2_b"   BF16 [rank, in]
#       "<prefix>.alpha"       F32 [1]             (upstream stores 0-D; loaders
#                                                   read it via float(), [1] loads)
#
# 2026-06-11 T2.F skeptic FIX: the previous format (`.hada_w1_a.weight` keys with
# Flame-orientation [in,rank]/[rank,out] factors, citing a now-deleted EDv2
# lycoris.rs) is NOT what the lycoris/comfy ecosystem reads. Upstream computes
#   DW_up = (w1a @ w1b) (.) (w2a @ w2b) * scale        [out, in]
# while the Mojo adapter computes
#   DW_mojo = (w1a @ w1b) (.) (w2a @ w2b) * scale      [in, out]  (x@DW form)
# so DW_up = DW_mojo^T  =>  on-disk  hada_w1_a := mojo w1b^T  [out,rank]
#                                    hada_w1_b := mojo w1a^T  [rank,in]
#                                    (pair 2 identically). Transposes of bf16
# values are exact, so the Mojo round-trip stays byte-exact. Gate:
# training/tests/lycoris_family_parity.mojo + lycoris_family_load_check.py
# (loads the Mojo-saved file into upstream LohaModule.make_module_from_state_dict).
#
# Unlike plain LoRA (lora_save.mojo deliberately OMITS `.alpha` and lets the
# loader default alpha=rank), the LyCORIS LoHa convention DOES carry a per-module
# `.alpha` scalar (scale = alpha/rank). We write it as a 1-element F32 tensor.
#
# (Optional Tucker cores hada_t1/hada_t2 are conv-only and NOT part of this
# wave's Linear LoHa; flagged in the builder report.)
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor → ArcPointer in collections;
# STDtype.F32 is a value; from_host(values, shape, dtype, ctx).

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.loha_adapter import LoHaAdapter


# A trained LoHa adapter paired with the base-weight prefix it adapts.
@fieldwise_init
struct NamedLoHa(Copyable, Movable):
    var prefix: String
    var adapter: LoHaAdapter


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


# Transpose a flat bf16 [r,c] matrix to [c,r]. BF16 values are moved, not
# recomputed — the transpose is exact.
def _transpose_bf16(a: List[BFloat16], r: Int, c: Int) -> List[BFloat16]:
    var out = List[BFloat16]()
    for _ in range(r * c):
        out.append(BFloat16(0.0))
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each LoHa adapter's 4 factors + alpha into a single safetensors via
# the proven byte-exact writer. Keys are the LyCORIS/diffusers convention.
# Returns the number of ADAPTERS written.
# ─────────────────────────────────────────────────────────────────────────────
def save_loha_peft(
    adapters: List[NamedLoHa], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_loha_peft: refusing to write an empty LoHa file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nh in adapters:
        var a = nh.adapter.copy()
        var IN = a.in_f
        var OUT = a.out_f
        var R = a.rank
        # Shape sanity for all 4 factors.
        if len(a.w1a) != IN * R:
            raise Error(String("save_loha_peft: w1a numel ") + String(len(a.w1a)) + " != in*rank for '" + nh.prefix + "'")
        if len(a.w1b) != R * OUT:
            raise Error(String("save_loha_peft: w1b numel ") + String(len(a.w1b)) + " != rank*out for '" + nh.prefix + "'")
        if len(a.w2a) != IN * R:
            raise Error(String("save_loha_peft: w2a numel ") + String(len(a.w2a)) + " != in*rank for '" + nh.prefix + "'")
        if len(a.w2b) != R * OUT:
            raise Error(String("save_loha_peft: w2b numel ") + String(len(a.w2b)) + " != rank*out for '" + nh.prefix + "'")

        # Upstream orientation: hada_w1_a = mojo w1b^T [out,rank],
        #                       hada_w1_b = mojo w1a^T [rank,in]  (pair 2 same).
        names.append(nh.prefix + ".hada_w1_a")
        tensors.append(ArcPointer(_bf16_2d(_transpose_bf16(a.w1b, R, OUT), OUT, R, ctx)))
        names.append(nh.prefix + ".hada_w1_b")
        tensors.append(ArcPointer(_bf16_2d(_transpose_bf16(a.w1a, IN, R), R, IN, ctx)))
        names.append(nh.prefix + ".hada_w2_a")
        tensors.append(ArcPointer(_bf16_2d(_transpose_bf16(a.w2b, R, OUT), OUT, R, ctx)))
        names.append(nh.prefix + ".hada_w2_b")
        tensors.append(ArcPointer(_bf16_2d(_transpose_bf16(a.w2a, IN, R), R, IN, ctx)))
        names.append(nh.prefix + ".alpha")
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


# A read-back of one LoHa module: the 4 factors (with shapes) + alpha. Used by
# the SAVE gate to assert keys/shapes/alpha round-trip.
@fieldwise_init
struct LoHaReadback(Copyable, Movable):
    var w1a: List[BFloat16]
    var w1b: List[BFloat16]
    var w2a: List[BFloat16]
    var w2b: List[BFloat16]
    var in_f: Int
    var out_f: Int
    var rank: Int
    var alpha: Float32


# Reopen one module's LoHa keys from `path` (upstream schema). On-disk shapes:
# hada_w1_a [out,rank] so out = shape[0], rank = shape[1]; hada_w1_b [rank,in]
# so in = shape[1]. The Flame-orientation factors are recovered by the inverse
# mapping: mojo w1a = hada_w1_b^T, mojo w1b = hada_w1_a^T (pair 2 identically;
# bf16 transposes are exact, so the round-trip is byte-exact).
def read_loha_module(prefix: String, path: String, ctx: DeviceContext) raises -> LoHaReadback:
    var st = SafeTensors.open(path)

    var i1a = st.tensor_info(prefix + ".hada_w1_a")
    var i1b = st.tensor_info(prefix + ".hada_w1_b")
    var i2a = st.tensor_info(prefix + ".hada_w2_a")
    var i2b = st.tensor_info(prefix + ".hada_w2_b")

    var OUT = i1a.shape[0]
    var R = i1a.shape[1]
    var IN = i1b.shape[1]

    # Shape consistency across all 4 factors.
    if i1b.shape[0] != R:
        raise Error("read_loha_module: hada_w1_b rows != rank")
    if i2a.shape[0] != OUT or i2a.shape[1] != R:
        raise Error("read_loha_module: hada_w2_a shape mismatch")
    if i2b.shape[0] != R or i2b.shape[1] != IN:
        raise Error("read_loha_module: hada_w2_b shape mismatch")

    var d1a = _read_bf16(st, prefix + ".hada_w1_a", ctx)   # [out,rank] on disk
    var d1b = _read_bf16(st, prefix + ".hada_w1_b", ctx)   # [rank,in]
    var d2a = _read_bf16(st, prefix + ".hada_w2_a", ctx)
    var d2b = _read_bf16(st, prefix + ".hada_w2_b", ctx)
    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_loha_module: .alpha must be a 1-element tensor")

    # Inverse of the save mapping: mojo w1a [in,rank] = (hada_w1_b [rank,in])^T,
    # mojo w1b [rank,out] = (hada_w1_a [out,rank])^T.
    var w1a = _transpose_bf16(d1b, R, IN)
    var w1b = _transpose_bf16(d1a, OUT, R)
    var w2a = _transpose_bf16(d2b, R, IN)
    var w2b = _transpose_bf16(d2a, OUT, R)

    return LoHaReadback(w1a^, w1b^, w2a^, w2b^, IN, OUT, R, alpha_h[0])
