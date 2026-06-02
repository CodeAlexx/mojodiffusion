# training/loha_save.mojo — save / reopen TRAINED LoHa adapters in the LyCORIS
# (diffusers PEFT) key convention.
#
# Mirrors EDv2 crates/eridiffusion-core/src/lycoris.rs collect_adapter_tensors
# LoHa arm (lycoris.rs:864-874) and upstream lycoris/modules/loha.py
# custom_state_dict (loha.py:272-275):
#
#       "<prefix>.hada_w1_a.weight"   F32 [in, rank]
#       "<prefix>.hada_w1_b.weight"   F32 [rank, out]
#       "<prefix>.hada_w2_a.weight"   F32 [in, rank]
#       "<prefix>.hada_w2_b.weight"   F32 [rank, out]
#       "<prefix>.alpha"              F32 [1]
#
# Unlike plain LoRA (lora_save.mojo deliberately OMITS `.alpha` and lets the
# loader default alpha=rank), the LyCORIS LoHa convention DOES carry a per-module
# `.alpha` scalar — upstream `custom_state_dict` always writes it and the
# diffusers/LyCORIS loader reads it to recover `scale = alpha/rank`. We follow
# upstream and write the alpha as a 1-element F32 tensor.
#
# (Optional Tucker cores hada_t1/hada_t2 — lycoris.rs:870-873 — are conv-only and
# NOT part of this wave's Linear LoHa; flagged in the builder report.)
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


def _f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def _f32_scalar(value: Float32, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    v.append(value)
    var sh = List[Int]()
    sh.append(1)
    return Tensor.from_host(v^, sh^, STDtype.F32, ctx)


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

        names.append(nh.prefix + ".hada_w1_a.weight")
        tensors.append(ArcPointer(_f32_2d(a.w1a.copy(), IN, R, ctx)))
        names.append(nh.prefix + ".hada_w1_b.weight")
        tensors.append(ArcPointer(_f32_2d(a.w1b.copy(), R, OUT, ctx)))
        names.append(nh.prefix + ".hada_w2_a.weight")
        tensors.append(ArcPointer(_f32_2d(a.w2a.copy(), IN, R, ctx)))
        names.append(nh.prefix + ".hada_w2_b.weight")
        tensors.append(ArcPointer(_f32_2d(a.w2b.copy(), R, OUT, ctx)))
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


# A read-back of one LoHa module: the 4 factors (with shapes) + alpha. Used by
# the SAVE gate to assert keys/shapes/alpha round-trip.
@fieldwise_init
struct LoHaReadback(Copyable, Movable):
    var w1a: List[Float32]
    var w1b: List[Float32]
    var w2a: List[Float32]
    var w2b: List[Float32]
    var in_f: Int
    var out_f: Int
    var rank: Int
    var alpha: Float32


# Reopen one module's LoHa keys from `path`. Shapes are read from the header:
# w1a is [in,rank] so in = w1a.shape[0], rank = w1a.shape[1]; w1b is [rank,out]
# so out = w1b.shape[1]. Asserts all 4 keys + alpha are present.
def read_loha_module(prefix: String, path: String, ctx: DeviceContext) raises -> LoHaReadback:
    var st = SafeTensors.open(path)

    var i1a = st.tensor_info(prefix + ".hada_w1_a.weight")
    var i1b = st.tensor_info(prefix + ".hada_w1_b.weight")
    var i2a = st.tensor_info(prefix + ".hada_w2_a.weight")
    var i2b = st.tensor_info(prefix + ".hada_w2_b.weight")

    var IN = i1a.shape[0]
    var R = i1a.shape[1]
    var OUT = i1b.shape[1]

    # Shape consistency across all 4 factors.
    if i1b.shape[0] != R:
        raise Error("read_loha_module: w1b rows != rank")
    if i2a.shape[0] != IN or i2a.shape[1] != R:
        raise Error("read_loha_module: w2a shape mismatch")
    if i2b.shape[0] != R or i2b.shape[1] != OUT:
        raise Error("read_loha_module: w2b shape mismatch")

    var w1a = _read_f32(st, prefix + ".hada_w1_a.weight", ctx)
    var w1b = _read_f32(st, prefix + ".hada_w1_b.weight", ctx)
    var w2a = _read_f32(st, prefix + ".hada_w2_a.weight", ctx)
    var w2b = _read_f32(st, prefix + ".hada_w2_b.weight", ctx)
    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_loha_module: .alpha must be a 1-element tensor")

    return LoHaReadback(w1a^, w1b^, w2a^, w2b^, IN, OUT, R, alpha_h[0])
