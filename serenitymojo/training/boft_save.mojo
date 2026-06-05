# training/boft_save.mojo — save / reopen TRAINED BOFT adapters in the LyCORIS
# (diffusers PEFT) key convention.
#
# Mirrors EDv2 crates/eridiffusion-core/src/lycoris.rs BOFT save
# (lycoris.rs:908-913 `out.insert(format!("{full}.oft_blocks.weight"), pt(&m.blocks))`):
#
#       "<prefix>.oft_blocks.weight"  BF16 [boft_m, num_blocks, block_size, block_size]
#       "<prefix>.alpha"              F32 [1]
#
# Same `oft_blocks` suffix as Diag-OFT — the 4D shape [boft_m, nb, b, b] is what
# disambiguates BOFT from OFT (3D [nb, b, b]) at load time (lycoris.rs:910-912,
# the upstream `boft.py` weight_list convention). The stored tensor is the
# per-stage per-block PRE-SKEW square S; a reopened S reconstructs Q, R, B, T
# bit-identically.
#
# ── AGENT-DEFAULT (flagged for review) ────────────────────────────────────────
# - Key spelling: `oft_blocks.weight` + `.alpha` (lycoris.rs:908-913). NOT
#   `oft_diag` (the upper-triangular vector form is not used — EDv2/upstream BOFT
#   store full blocks).
# - W_base (the frozen base weight) is NOT saved — model weight, not an adapter
#   parameter (matches the ref: BOFTModule owns only `blocks`). Reader returns
#   S + shapes + alpha; the caller re-supplies w_base on reconstruction.
# - OFT/BOFT carry a per-module `.alpha` scalar (unlike plain LoRA).
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
from serenitymojo.training.boft_adapter import BOFTAdapter


# A trained BOFT adapter paired with the base-weight prefix it adapts.
@fieldwise_init
struct NamedBOFT(Copyable, Movable):
    var prefix: String
    var adapter: BOFTAdapter


def _bf16_4d(var values: List[BFloat16], d0: Int, d1: Int, d2: Int, d3: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(d0)
    sh.append(d1)
    sh.append(d2)
    sh.append(d3)
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
# SAVE: pack each BOFT adapter's oft_blocks (4D S squares) + alpha. Returns the
# number of ADAPTERS written.
# ─────────────────────────────────────────────────────────────────────────────
def save_boft_peft(
    adapters: List[NamedBOFT], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_boft_peft: refusing to write an empty BOFT file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nk in adapters:
        var a = nk.adapter.copy()
        var MM = a.boft_m; var NB = a.num_blocks; var B = a.block_size
        if len(a.s) != MM * NB * B * B:
            raise Error(String("save_boft_peft: s numel ") + String(len(a.s)) + " != boft_m*num_blocks*b*b for '" + nk.prefix + "'")

        names.append(nk.prefix + ".oft_blocks.weight")
        tensors.append(ArcPointer(_bf16_4d(a.s.copy(), MM, NB, B, B, ctx)))

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


# A read-back of one BOFT module: S squares + shapes + alpha.
@fieldwise_init
struct BOFTReadback(Copyable, Movable):
    var s: List[BFloat16]
    var boft_m: Int
    var num_blocks: Int
    var block_size: Int
    var alpha: Float32


# Reopen one module's BOFT keys. oft_blocks [boft_m, nb, b, b] → shapes from its
# 4D shape (4D rank is the BOFT-vs-OFT discriminator). Asserts the keys + alpha.
def read_boft_module(prefix: String, path: String, ctx: DeviceContext) raises -> BOFTReadback:
    var st = SafeTensors.open(path)

    var info = st.tensor_info(prefix + ".oft_blocks.weight")
    if len(info.shape) != 4:
        raise Error(String("read_boft_module: oft_blocks must be 4D [boft_m,nb,b,b], got rank ") + String(len(info.shape)))
    var MM = info.shape[0]
    var NB = info.shape[1]
    var B = info.shape[2]
    if info.shape[3] != B:
        raise Error("read_boft_module: oft_blocks last two dims must be equal (square blocks)")
    var s = _read_bf16(st, prefix + ".oft_blocks.weight", ctx)

    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_boft_module: .alpha must be a 1-element tensor")

    return BOFTReadback(s^, MM, NB, B, alpha_h[0])
