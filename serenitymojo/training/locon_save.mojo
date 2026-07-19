# training/locon_save.mojo — save / reopen TRAINED LoCon conv adapters in the
# LyCORIS / Kohya (diffusers PEFT) conv key convention.
#
# Mirrors EDv2 eri-lycoris/lycoris-rs/src/loader.rs build_locon (loader.rs:353-394)
# and upstream lycoris/modules/locon.py custom_state_dict:
#
#   "<prefix>.lora_down.weight"  F32 [rank, Cin, Kh, Kw]   (Kohya OIHW)
#   "<prefix>.lora_up.weight"    F32 [Cout, rank, 1, 1]    (Kohya OIHW, 1x1)
#   "<prefix>.alpha"             F32 [1]
#
# (The Tucker `.lora_mid.weight` key is part of the convention but LoCon-no-Tucker
# has no mid — that's Item B's tucker_conv_adapter. flagged below.)
#
# ── INTERNAL vs ON-DISK LAYOUT ────────────────────────────────────────────────
# Our adapter stores Flame-RSCF:  down [Kh,Kw,Cin,rank], up [1,1,rank,Cout].
# The on-disk Kohya convention is OIHW:  down [rank,Cin,Kh,Kw], up [Cout,rank,1,1].
# loader.rs:372-382 permutes Kohya→Flame via PyTorch[O,I,KH,KW]→[2,3,1,0]=[KH,KW,I,O].
# We invert that on SAVE: Flame[KH,KW,I,O] → Kohya[O,I,KH,KW] is permute [3,2,0,1].
# We open-code the index remap (no Tensor.permute on host F32 lists).
#
# ── AGENT-DEFAULT (flagged) ───────────────────────────────────────────────────
# Key spellings lora_down / lora_up + `.alpha`, the `.weight` suffix, and the
# OIHW on-disk shapes [rank,Cin,Kh,Kw] / [Cout,rank,1,1] (loader.rs build_locon).
# Like LoHa/LoKr (and unlike plain LoRA), LoCon DOES carry a per-module `.alpha`.
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
from serenitymojo.training.locon_conv_adapter import LoConConvAdapter


@fieldwise_init
struct NamedLoCon(Copyable, Movable):
    var prefix: String
    var adapter: LoConConvAdapter


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


# Flame RSCF down [Kh,Kw,Cin,R] → Kohya OIHW [R,Cin,Kh,Kw]  (permute [3,2,0,1]).
def _down_rscf_to_oihw(d: List[Float32], Kh: Int, Kw: Int, Cin: Int, R: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(R * Cin * Kh * Kw):
        out.append(Float32(0.0))
    for kh in range(Kh):
        for kw in range(Kw):
            for ci in range(Cin):
                for r in range(R):
                    var src = (((kh * Kw + kw) * Cin) + ci) * R + r
                    var dst = (((r * Cin + ci) * Kh) + kh) * Kw + kw
                    out[dst] = d[src]
    return out^


# Flame RSCF up [1,1,R,Cout] (= [R,Cout]) → Kohya OIHW [Cout,R,1,1] (= [Cout,R]).
def _up_rscf_to_oihw(u: List[Float32], R: Int, Cout: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(Cout * R):
        out.append(Float32(0.0))
    for r in range(R):
        for co in range(Cout):
            out[co * R + r] = u[r * Cout + co]
    return out^


# Kohya OIHW down [R,Cin,Kh,Kw] → Flame RSCF [Kh,Kw,Cin,R].
def _down_oihw_to_rscf(d: List[Float32], R: Int, Cin: Int, Kh: Int, Kw: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(Kh * Kw * Cin * R):
        out.append(Float32(0.0))
    for r in range(R):
        for ci in range(Cin):
            for kh in range(Kh):
                for kw in range(Kw):
                    var src = (((r * Cin + ci) * Kh) + kh) * Kw + kw
                    var dst = (((kh * Kw + kw) * Cin) + ci) * R + r
                    out[dst] = d[src]
    return out^


# Kohya OIHW up [Cout,R,1,1] → Flame RSCF [1,1,R,Cout] (= [R,Cout]).
def _up_oihw_to_rscf(u: List[Float32], Cout: Int, R: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(R * Cout):
        out.append(Float32(0.0))
    for co in range(Cout):
        for r in range(R):
            out[r * Cout + co] = u[co * R + r]
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each LoCon adapter's down + up + alpha (Kohya OIHW). Returns count.
# ─────────────────────────────────────────────────────────────────────────────
def save_locon_peft(
    adapters: List[NamedLoCon], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_locon_peft: refusing to write an empty LoCon file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nk in adapters:
        var a = nk.adapter.copy()
        var Kh = a.kh; var Kw = a.kw; var Cin = a.cin; var R = a.rank; var Cout = a.cout
        if len(a.down) != Kh * Kw * Cin * R:
            raise Error(String("save_locon_peft: down numel mismatch for '") + nk.prefix + "'")
        if len(a.up) != R * Cout:
            raise Error(String("save_locon_peft: up numel mismatch for '") + nk.prefix + "'")

        var down_oihw = _down_rscf_to_oihw(a.down.copy(), Kh, Kw, Cin, R)
        var up_oihw = _up_rscf_to_oihw(a.up.copy(), R, Cout)

        names.append(nk.prefix + ".lora_down.weight")
        tensors.append(ArcPointer(_f32_4d(down_oihw^, R, Cin, Kh, Kw, ctx)))
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


# A read-back of one LoCon module: Flame-RSCF down/up + shapes + alpha.
@fieldwise_init
struct LoConReadback(Copyable, Movable):
    var down: List[Float32]   # [Kh,Kw,Cin,R] (Flame RSCF)
    var up: List[Float32]     # [1,1,R,Cout]  (Flame RSCF)
    var cin: Int
    var cout: Int
    var kh: Int
    var kw: Int
    var rank: Int
    var alpha: Float32


# Reopen one module's LoCon keys. down on-disk [R,Cin,Kh,Kw] → R,Cin,Kh,Kw from
# its shape; up [Cout,R,1,1] → Cout,R. Returns Flame-RSCF lists.
def read_locon_module(prefix: String, path: String, ctx: DeviceContext) raises -> LoConReadback:
    var st = SafeTensors.open(path)

    var id = st.tensor_info(prefix + ".lora_down.weight")
    if len(id.shape) != 4:
        raise Error("read_locon_module: lora_down must be 4D [R,Cin,Kh,Kw]")
    var R = id.shape[0]
    var Cin = id.shape[1]
    var Kh = id.shape[2]
    var Kw = id.shape[3]
    var down_oihw = _read_f32(st, prefix + ".lora_down.weight", ctx)

    var iu = st.tensor_info(prefix + ".lora_up.weight")
    if len(iu.shape) != 4:
        raise Error("read_locon_module: lora_up must be 4D [Cout,R,1,1]")
    var Cout = iu.shape[0]
    if iu.shape[1] != R:
        raise Error("read_locon_module: lora_up rank dim != lora_down rank")
    if iu.shape[2] != 1 or iu.shape[3] != 1:
        raise Error("read_locon_module: lora_up must be a 1x1 kernel")
    var up_oihw = _read_f32(st, prefix + ".lora_up.weight", ctx)

    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_locon_module: .alpha must be a 1-element tensor")

    var down_rscf = _down_oihw_to_rscf(down_oihw, R, Cin, Kh, Kw)
    var up_rscf = _up_oihw_to_rscf(up_oihw, Cout, R)
    return LoConReadback(down_rscf^, up_rscf^, Cin, Cout, Kh, Kw, R, alpha_h[0])
