# training/dora_save.mojo — save / reopen TRAINED DoRA adapters.
#
# Two on-disk shape conventions use the same key names:
# - upstream LyCORIS/Kohya DoRA: output-axis `dora_scale` [out,1].
# - OneTrainer default DoRA: input-axis `dora_scale` [1,in] when
#   lora_decompose_output_axis=false.
#
# MEASURED upstream schema (lycoris/modules/locon.py LoConModule with
# weight_decompose=True — upstream lycoris DoRA IS LoCon(wd=True); weight_list
# + custom_state_dict):
#
#       "<prefix>.lora_down.weight"  BF16 [rank, in]   (== DoRAAdapter.a)
#       "<prefix>.lora_up.weight"    BF16 [out, rank]  (== DoRAAdapter.b)
#       "<prefix>.dora_scale"        F32 [out, 1]      (== DoRAAdapter.m magnitude;
#                                     upstream keeps dora_scale float32 even in
#                                     bf16 models — locon.py `.float()`)
#       "<prefix>.alpha"             F32 [1]
#
# 2026-06-11 T2.F-2 skeptic FIX: the low-rank legs were previously keyed
# `.lora_A.weight`/`.lora_B.weight` (PEFT spellings, citing the now-deleted EDv2
# lycoris.rs) — a HYBRID no upstream loader consumes whole (pip lycoris wants
# lora_down/lora_up; PEFT-DoRA wants lora_magnitude_vector, not dora_scale).
# Same ecosystem-unloadable bug class as the T2.F LoHa/OFT key fixes. DOCUMENTED
# CHOICE: the upstream-lycoris LoCon(wd=True) schema above — pip lycoris is the
# only live upstream loader on this box (campaign oracle), and lora_up/lora_down
# + dora_scale is also ComfyUI's kohya-DoRA path. Gate:
# lycoris_family_load_check.py loads a Mojo-saved file through
# LoConModule.make_module_from_state_dict and reproduces the forward BIT-EXACT.
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# - magnitude saved as [out,1] (wd_on_out=true convention; upstream dora_scale
#   shape). `.magnitude_vector` is accepted on read as a legacy alias but NOT
#   written.
# - `save_dora_onetrainer` writes [out,1] for wd_on_out=true and [1,in] for the
#   OneTrainer default wd_on_out=false.
# - Plain LoRA (lora_save.mojo) deliberately OMITS `.alpha`; DoRA follows the
#   LyCORIS convention and DOES carry `.alpha` (so scale=alpha/rank reconstructs).
#
# Mojo 0.26.x: `def` not `fn`; move-only Tensor → ArcPointer.

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


def _bf16_2d(var values: List[BFloat16], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host_bf16(values^, sh^, ctx)


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


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each DoRA adapter's lora_down/lora_up + dora_scale + alpha. Returns
# the number of ADAPTERS written.
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
            raise Error(String("save_dora_peft: lora_down numel ") + String(len(a.a)) + " != rank*in for '" + nd.prefix + "'")
        if len(a.b) != OUT * R:
            raise Error(String("save_dora_peft: lora_up numel ") + String(len(a.b)) + " != out*rank for '" + nd.prefix + "'")
        if len(a.m) != OUT:
            raise Error(String("save_dora_peft: magnitude numel ") + String(len(a.m)) + " != out for '" + nd.prefix + "'")

        names.append(nd.prefix + ".lora_down.weight")
        tensors.append(ArcPointer(_bf16_2d(a.a.copy(), R, IN, ctx)))
        names.append(nd.prefix + ".lora_up.weight")
        tensors.append(ArcPointer(_bf16_2d(a.b.copy(), OUT, R, ctx)))
        names.append(nd.prefix + ".dora_scale")
        tensors.append(ArcPointer(_f32_2d(a.m.copy(), OUT, 1, ctx)))   # F32 [out,1]
        names.append(nd.prefix + ".alpha")
        tensors.append(ArcPointer(_f32_scalar(a.alpha, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


# OneTrainer uses the same key names as the LyCORIS save but preserves the DoRA
# decomposition axis in `dora_scale` shape. Linear per-input default is [1,in].
def save_dora_onetrainer(
    adapters: List[NamedDoRA], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_dora_onetrainer: refusing to write an empty DoRA file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nd in adapters:
        var a = nd.adapter.copy()
        var IN = a.in_f
        var OUT = a.out_f
        var R = a.rank
        var mlen = OUT if a.wd_on_out else IN
        if len(a.a) != R * IN:
            raise Error(String("save_dora_onetrainer: lora_down numel ") + String(len(a.a)) + " != rank*in for '" + nd.prefix + "'")
        if len(a.b) != OUT * R:
            raise Error(String("save_dora_onetrainer: lora_up numel ") + String(len(a.b)) + " != out*rank for '" + nd.prefix + "'")
        if len(a.m) != mlen:
            raise Error(String("save_dora_onetrainer: magnitude numel ") + String(len(a.m)) + " != decomposition axis for '" + nd.prefix + "'")

        names.append(nd.prefix + ".lora_down.weight")
        tensors.append(ArcPointer(_bf16_2d(a.a.copy(), R, IN, ctx)))
        names.append(nd.prefix + ".lora_up.weight")
        tensors.append(ArcPointer(_bf16_2d(a.b.copy(), OUT, R, ctx)))
        names.append(nd.prefix + ".dora_scale")
        if a.wd_on_out:
            tensors.append(ArcPointer(_f32_2d(a.m.copy(), OUT, 1, ctx)))
        else:
            tensors.append(ArcPointer(_f32_2d(a.m.copy(), 1, IN, ctx)))
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


def _read_bf16(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[BFloat16]:
    return _f32_to_bf16_list(_read_f32(st, name, ctx))


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
    var a: List[BFloat16]
    var b: List[BFloat16]
    var m: List[Float32]   # F32, like the adapter (upstream dora_scale is F32)
    var in_f: Int
    var out_f: Int
    var rank: Int
    var alpha: Float32
    var wd_on_out: Bool


# Reopen one module's DoRA keys from `path`. Shapes from the header: lora_down
# is [rank,in] so rank = a.shape[0], in = a.shape[1]; lora_up is [out,rank] so
# out = b.shape[0]. The magnitude is read from `.dora_scale`, or `.magnitude_vector`
# as a fallback legacy alias. `wd_on_out` is inferred from magnitude shape:
# [out,1]/[out] => output-axis, [1,in]/[in] => input-axis.
def read_dora_module(prefix: String, path: String, ctx: DeviceContext) raises -> DoRAReadback:
    var st = SafeTensors.open(path)

    var ia = st.tensor_info(prefix + ".lora_down.weight")
    var ib = st.tensor_info(prefix + ".lora_up.weight")

    var R = ia.shape[0]
    var IN = ia.shape[1]
    var OUT = ib.shape[0]

    if ib.shape[1] != R:
        raise Error("read_dora_module: lora_up cols != rank")

    # magnitude key: prefer .dora_scale, fall back to .magnitude_vector alias.
    var mag_key = prefix + ".dora_scale"
    if not _has_key(st, mag_key):
        var alias_key = prefix + ".magnitude_vector"
        if _has_key(st, alias_key):
            mag_key = alias_key
        else:
            raise Error("read_dora_module: neither .dora_scale nor .magnitude_vector present for '" + prefix + "'")
    var im = st.tensor_info(mag_key)
    var a = _read_bf16(st, prefix + ".lora_down.weight", ctx)
    var b = _read_bf16(st, prefix + ".lora_up.weight", ctx)
    var m = _read_f32(st, mag_key, ctx)
    var wd_on_out: Bool
    if len(m) == OUT and im.shape[0] == OUT:
        wd_on_out = True
    elif len(m) == IN:
        if len(im.shape) == 1:
            wd_on_out = False
        elif im.shape[0] == 1:
            wd_on_out = False
        else:
            raise Error("read_dora_module: input-axis magnitude shape must be [1,in] or [in]")
    else:
        raise Error("read_dora_module: magnitude numel does not match input or output axis")
    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_dora_module: .alpha must be a 1-element tensor")

    return DoRAReadback(a^, b^, m^, IN, OUT, R, alpha_h[0], wd_on_out)
