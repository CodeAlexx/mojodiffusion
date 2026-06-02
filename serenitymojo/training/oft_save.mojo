# training/oft_save.mojo — save / reopen TRAINED Diag-OFT adapters in the
# LyCORIS (diffusers PEFT) key convention.
#
# Mirrors EDv2 crates/eridiffusion-core/src/lycoris.rs OFT save
# (lycoris.rs:905-906 `out.insert(format!("{full}.oft_blocks.weight"), pt(&m.blocks))`)
# and lycoris.rs:41 (`OFT | .oft_blocks.weight`):
#
#       "<prefix>.oft_blocks.weight"  F32 [num_blocks, block_size, block_size]
#       "<prefix>.alpha"              F32 [1]
#
# The stored tensor is the per-block PRE-SKEW square S (oft.rs stores `blocks`,
# the same pre-skew square; a reopened S reconstructs Q=0.5(S-Sᵀ), R, W_eff
# bit-identically). 3D shape [num_blocks, b, b] is what disambiguates OFT from
# BOFT (4D [boft_m, num_blocks, b, b]) at load time — lycoris.rs:910-912.
#
# ── AGENT-DEFAULT (flagged for review) ────────────────────────────────────────
# - Key spelling: `oft_blocks.weight` + `.alpha` (lycoris.rs:905-906, :41). The
#   task prompt mentioned `oft_diag` as an alternative; EDv2 + upstream BOFT
#   weight_list use only `oft_blocks` (the FULL-block form), NOT the
#   upper-triangular `oft_diag` vector form — we follow EDv2. The `oft_diag`
#   spelling is NOT written or read by this port.
# - W_base (the frozen base weight the rotation acts on) is NOT saved — it is the
#   model's own weight, not an adapter parameter (matches the ref: OFTModule
#   only owns `blocks`). The reader returns S + shapes + alpha; the caller
#   re-supplies w_base when reconstructing an OFTAdapter.
# - Like LoHa/DoRA/LoKr (and unlike plain LoRA, which omits `.alpha`), OFT
#   carries a per-module `.alpha` scalar.
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
from serenitymojo.training.oft_adapter import OFTAdapter


# A trained OFT adapter paired with the base-weight prefix it adapts.
@fieldwise_init
struct NamedOFT(Copyable, Movable):
    var prefix: String
    var adapter: OFTAdapter


def _f32_3d(var values: List[Float32], d0: Int, d1: Int, d2: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(d0)
    sh.append(d1)
    sh.append(d2)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


def _f32_scalar(value: Float32, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    v.append(value)
    var sh = List[Int]()
    sh.append(1)
    return Tensor.from_host(v^, sh^, STDtype.F32, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each OFT adapter's oft_blocks (S squares) + alpha. Returns the
# number of ADAPTERS written.
# ─────────────────────────────────────────────────────────────────────────────
def save_oft_peft(
    adapters: List[NamedOFT], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_oft_peft: refusing to write an empty OFT file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nk in adapters:
        var a = nk.adapter.copy()
        var NB = a.num_blocks; var B = a.block_size
        if len(a.s) != NB * B * B:
            raise Error(String("save_oft_peft: s numel ") + String(len(a.s)) + " != num_blocks*b*b for '" + nk.prefix + "'")

        names.append(nk.prefix + ".oft_blocks.weight")
        tensors.append(ArcPointer(_f32_3d(a.s.copy(), NB, B, B, ctx)))

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


# A read-back of one OFT module: S squares + shapes + alpha.
@fieldwise_init
struct OFTReadback(Copyable, Movable):
    var s: List[Float32]
    var num_blocks: Int
    var block_size: Int
    var alpha: Float32


# Reopen one module's OFT keys. oft_blocks [num_blocks, b, b] → shapes from its
# 3D shape (3D rank is the OFT-vs-BOFT discriminator). Asserts the keys + alpha.
def read_oft_module(prefix: String, path: String, ctx: DeviceContext) raises -> OFTReadback:
    var st = SafeTensors.open(path)

    var info = st.tensor_info(prefix + ".oft_blocks.weight")
    if len(info.shape) != 3:
        raise Error(String("read_oft_module: oft_blocks must be 3D [num_blocks,b,b], got rank ") + String(len(info.shape)))
    var NB = info.shape[0]
    var B = info.shape[1]
    if info.shape[2] != B:
        raise Error("read_oft_module: oft_blocks last two dims must be equal (square blocks)")
    var s = _read_f32(st, prefix + ".oft_blocks.weight", ctx)

    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_oft_module: .alpha must be a 1-element tensor")

    return OFTReadback(s^, NB, B, alpha_h[0])
