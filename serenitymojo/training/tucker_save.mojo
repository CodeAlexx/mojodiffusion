# training/tucker_save.mojo — save / reopen TRAINED Tucker conv adapters in the
# LyCORIS / Kohya Tucker-LoCon key convention.
#
# Mirrors EDv2 eri-lycoris/lycoris-rs/src/loader.rs build_locon WITH a mid factor
# (loader.rs:351,383-392) + upstream lycoris/modules/locon.py Tucker
# custom_state_dict + ops/tucker.rs rebuild_conv_tucker:
#
#   "<prefix>.lora_down.weight"  F32 [rank, Cin, 1, 1]    (Kohya OIHW, 1x1)
#   "<prefix>.lora_mid.weight"   F32 [rank, rank, Kh, Kw] (Kohya OIHW core)
#   "<prefix>.lora_up.weight"    F32 [Cout, rank, 1, 1]   (Kohya OIHW, 1x1)
#   "<prefix>.alpha"             F32 [1]
#
# ── INTERNAL vs ON-DISK LAYOUT ────────────────────────────────────────────────
# Our adapter stores Flame-RSCF:
#   down [1,1,Cin,rank], core [Kh,Kw,rank,rank] (= [Kh,Kw,Rin,Rout]), up [1,1,rank,Cout].
# loader.rs permutes Kohya OIHW [O,I,KH,KW] → Flame [2,3,1,0] = [KH,KW,I,O].
# We invert on SAVE: Flame [KH,KW,I,O] → Kohya [O,I,KH,KW] = permute [3,2,0,1].
# For the mid/core, Kohya [Rout,Rin,Kh,Kw] → Flame [Kh,Kw,Rin,Rout] (so O=Rout,
# I=Rin); we save Flame[Kh,Kw,Rin,Rout] → Kohya[Rout,Rin,Kh,Kw].
#
# ── AGENT-DEFAULT (flagged) ───────────────────────────────────────────────────
# Key spellings lora_down / lora_mid / lora_up + `.alpha`, the `.weight` suffix,
# and the OIHW on-disk shapes [rank,Cin,1,1] / [rank,rank,Kh,Kw] / [Cout,rank,1,1]
# (loader.rs build_locon + tucker.rs rebuild_conv_tucker). Carries a per-module
# `.alpha` (scale = alpha/rank), like LoCon/LoHa/LoKr.
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
from serenitymojo.training.tucker_conv_adapter import TuckerConvAdapter


@fieldwise_init
struct NamedTucker(Copyable, Movable):
    var prefix: String
    var adapter: TuckerConvAdapter


def _f32_4d(var values: List[Float32], d0: Int, d1: Int, d2: Int, d3: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(d0); sh.append(d1); sh.append(d2); sh.append(d3)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def _f32_scalar(value: Float32, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    v.append(value)
    var sh = List[Int]()
    sh.append(1)
    return Tensor.from_host(v^, sh^, STDtype.F32, ctx)


# Flame down [1,1,Cin,R] (= [Cin,R]) → Kohya [R,Cin,1,1] (= [R,Cin]).
def _down_to_oihw(d: List[Float32], Cin: Int, R: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(R * Cin):
        out.append(Float32(0.0))
    for ci in range(Cin):
        for r in range(R):
            out[r * Cin + ci] = d[ci * R + r]
    return out^


# Flame up [1,1,R,Cout] (= [R,Cout]) → Kohya [Cout,R,1,1] (= [Cout,R]).
def _up_to_oihw(u: List[Float32], R: Int, Cout: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(Cout * R):
        out.append(Float32(0.0))
    for r in range(R):
        for co in range(Cout):
            out[co * R + r] = u[r * Cout + co]
    return out^


# Flame core [Kh,Kw,Rin,Rout] → Kohya mid [Rout,Rin,Kh,Kw] (permute [3,2,0,1]).
def _core_to_oihw(c: List[Float32], Kh: Int, Kw: Int, Ri: Int, Ro: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(Ro * Ri * Kh * Kw):
        out.append(Float32(0.0))
    for kh in range(Kh):
        for kw in range(Kw):
            for ri in range(Ri):
                for ro in range(Ro):
                    var src = (((kh * Kw + kw) * Ri) + ri) * Ro + ro
                    var dst = (((ro * Ri + ri) * Kh) + kh) * Kw + kw
                    out[dst] = c[src]
    return out^


# Kohya down [R,Cin,1,1] → Flame [1,1,Cin,R] (= [Cin,R]).
def _down_from_oihw(d: List[Float32], R: Int, Cin: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(Cin * R):
        out.append(Float32(0.0))
    for r in range(R):
        for ci in range(Cin):
            out[ci * R + r] = d[r * Cin + ci]
    return out^


# Kohya up [Cout,R,1,1] → Flame [1,1,R,Cout] (= [R,Cout]).
def _up_from_oihw(u: List[Float32], Cout: Int, R: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(R * Cout):
        out.append(Float32(0.0))
    for co in range(Cout):
        for r in range(R):
            out[r * Cout + co] = u[co * R + r]
    return out^


# Kohya mid [Rout,Rin,Kh,Kw] → Flame core [Kh,Kw,Rin,Rout].
def _core_from_oihw(c: List[Float32], Ro: Int, Ri: Int, Kh: Int, Kw: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(Kh * Kw * Ri * Ro):
        out.append(Float32(0.0))
    for ro in range(Ro):
        for ri in range(Ri):
            for kh in range(Kh):
                for kw in range(Kw):
                    var src = (((ro * Ri + ri) * Kh) + kh) * Kw + kw
                    var dst = (((kh * Kw + kw) * Ri) + ri) * Ro + ro
                    out[dst] = c[src]
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack down + mid(core) + up + alpha (Kohya OIHW). Returns adapter count.
# ─────────────────────────────────────────────────────────────────────────────
def save_tucker_peft(
    adapters: List[NamedTucker], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_tucker_peft: refusing to write an empty Tucker file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nk in adapters:
        var a = nk.adapter.copy()
        var Kh = a.kh; var Kw = a.kw; var Cin = a.cin; var R = a.rank; var Cout = a.cout
        if len(a.down) != Cin * R:
            raise Error(String("save_tucker_peft: down numel mismatch for '") + nk.prefix + "'")
        if len(a.core) != Kh * Kw * R * R:
            raise Error(String("save_tucker_peft: core numel mismatch for '") + nk.prefix + "'")
        if len(a.up) != R * Cout:
            raise Error(String("save_tucker_peft: up numel mismatch for '") + nk.prefix + "'")

        var down_oihw = _down_to_oihw(a.down.copy(), Cin, R)
        var core_oihw = _core_to_oihw(a.core.copy(), Kh, Kw, R, R)
        var up_oihw = _up_to_oihw(a.up.copy(), R, Cout)

        names.append(nk.prefix + ".lora_down.weight")
        tensors.append(ArcPointer(_f32_4d(down_oihw^, R, Cin, 1, 1, ctx)))
        names.append(nk.prefix + ".lora_mid.weight")
        tensors.append(ArcPointer(_f32_4d(core_oihw^, R, R, Kh, Kw, ctx)))
        names.append(nk.prefix + ".lora_up.weight")
        tensors.append(ArcPointer(_f32_4d(up_oihw^, Cout, R, 1, 1, ctx)))
        names.append(nk.prefix + ".alpha")
        tensors.append(ArcPointer(_f32_scalar(a.alpha, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


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


@fieldwise_init
struct TuckerReadback(Copyable, Movable):
    var down: List[Float32]   # [1,1,Cin,R] (Flame)
    var core: List[Float32]   # [Kh,Kw,R,R] (Flame)
    var up: List[Float32]     # [1,1,R,Cout] (Flame)
    var cin: Int
    var cout: Int
    var kh: Int
    var kw: Int
    var rank: Int
    var alpha: Float32


# Reopen one module's Tucker keys. mid present is REQUIRED (Tucker). Shapes:
# down [R,Cin,1,1] → R,Cin; mid [R,R,Kh,Kw] → Kh,Kw (and asserts the two ranks
# match down's R); up [Cout,R,1,1] → Cout. Returns Flame-RSCF lists.
def read_tucker_module(prefix: String, path: String, ctx: DeviceContext) raises -> TuckerReadback:
    var st = SafeTensors.open(path)

    var id = st.tensor_info(prefix + ".lora_down.weight")
    if len(id.shape) != 4 or id.shape[2] != 1 or id.shape[3] != 1:
        raise Error("read_tucker_module: lora_down must be 4D [R,Cin,1,1]")
    var R = id.shape[0]
    var Cin = id.shape[1]
    var down_oihw = _read_f32(st, prefix + ".lora_down.weight", ctx)

    var im = st.tensor_info(prefix + ".lora_mid.weight")
    if len(im.shape) != 4:
        raise Error("read_tucker_module: lora_mid must be 4D [R,R,Kh,Kw]")
    if im.shape[0] != R or im.shape[1] != R:
        raise Error("read_tucker_module: lora_mid rank dims must match lora_down rank")
    var Kh = im.shape[2]
    var Kw = im.shape[3]
    var core_oihw = _read_f32(st, prefix + ".lora_mid.weight", ctx)

    var iu = st.tensor_info(prefix + ".lora_up.weight")
    if len(iu.shape) != 4 or iu.shape[2] != 1 or iu.shape[3] != 1:
        raise Error("read_tucker_module: lora_up must be 4D [Cout,R,1,1]")
    var Cout = iu.shape[0]
    if iu.shape[1] != R:
        raise Error("read_tucker_module: lora_up rank dim != lora_down rank")
    var up_oihw = _read_f32(st, prefix + ".lora_up.weight", ctx)

    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_tucker_module: .alpha must be a 1-element tensor")

    var down_rscf = _down_from_oihw(down_oihw, R, Cin)
    var core_rscf = _core_from_oihw(core_oihw, R, R, Kh, Kw)
    var up_rscf = _up_from_oihw(up_oihw, Cout, R)
    return TuckerReadback(down_rscf^, core_rscf^, up_rscf^, Cin, Cout, Kh, Kw, R, alpha_h[0])
