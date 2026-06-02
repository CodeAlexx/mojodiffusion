# training/dora_save.mojo — save / reopen TRAINED DoRA adapters in the LyCORIS
# (diffusers PEFT) key convention.
#
# Mirrors EDv2 crates/eridiffusion-core/src/lycoris.rs DoRA save
# (lycoris.rs:787 `out.insert(format!("{full}.dora_scale"), m.clone())` — the
# `.dora_scale` magnitude appended NEXT TO the algo's own tensors) and dora.rs
# header §"Save format" (lycoris.rs:1131 also accepts `.magnitude_vector` as a
# load alias). The base low-rank legs use the project PEFT LoRA convention
# (lora_save.mojo:24-25):
#
#       "<prefix>.lora_A.weight"   F32 [rank, in]    (== DoRAAdapter.a / lora_down)
#       "<prefix>.lora_B.weight"   F32 [out, rank]   (== DoRAAdapter.b / lora_up)
#       "<prefix>.dora_scale"      F32 [out, 1]      (== DoRAAdapter.m magnitude)
#       "<prefix>.alpha"           F32 [1]
#
# ── AGENT-DEFAULT (flagged for review) ────────────────────────────────────────
# - Key spellings: lora_A/lora_B (PEFT, matching this port's plain-LoRA save) +
#   `.dora_scale` (lycoris.rs:787). PEFT's `.magnitude_vector` is accepted on
#   read as an alias (dora.rs:64-65 / lycoris.rs:1131) but NOT written.
# - magnitude saved as [out,1] (wd_on_out=true convention, dora.rs:35).
# - Plain LoRA (lora_save.mojo) deliberately OMITS `.alpha`; DoRA follows the
#   LyCORIS convention and DOES carry `.alpha` (so scale=alpha/rank reconstructs).
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
from serenitymojo.training.dora_adapter import DoRAAdapter


# A trained DoRA adapter paired with the base-weight prefix it adapts.
@fieldwise_init
struct NamedDoRA(Copyable, Movable):
    var prefix: String
    var adapter: DoRAAdapter


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
# SAVE: pack each DoRA adapter's lora_A/lora_B + dora_scale + alpha. Returns the
# number of ADAPTERS written.
# ─────────────────────────────────────────────────────────────────────────────
def save_dora_peft(
    adapters: List[NamedDoRA], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_dora_peft: refusing to write an empty DoRA file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nd in adapters:
        var a = nd.adapter.copy()
        var IN = a.in_f
        var OUT = a.out_f
        var R = a.rank
        if len(a.a) != R * IN:
            raise Error(String("save_dora_peft: lora_A numel ") + String(len(a.a)) + " != rank*in for '" + nd.prefix + "'")
        if len(a.b) != OUT * R:
            raise Error(String("save_dora_peft: lora_B numel ") + String(len(a.b)) + " != out*rank for '" + nd.prefix + "'")
        if len(a.m) != OUT:
            raise Error(String("save_dora_peft: magnitude numel ") + String(len(a.m)) + " != out for '" + nd.prefix + "'")

        names.append(nd.prefix + ".lora_A.weight")
        tensors.append(ArcPointer(_f32_2d(a.a.copy(), R, IN, ctx)))
        names.append(nd.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_f32_2d(a.b.copy(), OUT, R, ctx)))
        names.append(nd.prefix + ".dora_scale")
        tensors.append(ArcPointer(_f32_2d(a.m.copy(), OUT, 1, ctx)))   # [out,1]
        names.append(nd.prefix + ".alpha")
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


# True iff `name` is present in the file.
def _has_key(st: SafeTensors, name: String) -> Bool:
    try:
        var _info = st.tensor_info(name)
        return True
    except:
        return False


# A read-back of one DoRA module: A, B, magnitude (+ shapes) + alpha.
@fieldwise_init
struct DoRAReadback(Copyable, Movable):
    var a: List[Float32]
    var b: List[Float32]
    var m: List[Float32]
    var in_f: Int
    var out_f: Int
    var rank: Int
    var alpha: Float32


# Reopen one module's DoRA keys from `path`. Shapes from the header: lora_A is
# [rank,in] so rank = a.shape[0], in = a.shape[1]; lora_B is [out,rank] so
# out = b.shape[0]. The magnitude is read from `.dora_scale`, or `.magnitude_vector`
# as a fallback alias (dora.rs:64-65 / lycoris.rs:1131). Asserts all keys present.
def read_dora_module(prefix: String, path: String, ctx: DeviceContext) raises -> DoRAReadback:
    var st = SafeTensors.open(path)

    var ia = st.tensor_info(prefix + ".lora_A.weight")
    var ib = st.tensor_info(prefix + ".lora_B.weight")

    var R = ia.shape[0]
    var IN = ia.shape[1]
    var OUT = ib.shape[0]

    if ib.shape[1] != R:
        raise Error("read_dora_module: lora_B cols != rank")

    # magnitude key: prefer .dora_scale, fall back to .magnitude_vector alias.
    var mag_key = prefix + ".dora_scale"
    if not _has_key(st, mag_key):
        var alias_key = prefix + ".magnitude_vector"
        if _has_key(st, alias_key):
            mag_key = alias_key
        else:
            raise Error("read_dora_module: neither .dora_scale nor .magnitude_vector present for '" + prefix + "'")
    var im = st.tensor_info(mag_key)
    # magnitude is [out,1] (or [out]); first dim must be OUT.
    if im.shape[0] != OUT:
        raise Error("read_dora_module: magnitude first dim != out")

    var a = _read_f32(st, prefix + ".lora_A.weight", ctx)
    var b = _read_f32(st, prefix + ".lora_B.weight", ctx)
    var m = _read_f32(st, mag_key, ctx)
    if len(m) != OUT:
        raise Error("read_dora_module: magnitude numel != out")
    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_dora_module: .alpha must be a 1-element tensor")

    return DoRAReadback(a^, b^, m^, IN, OUT, R, alpha_h[0])
