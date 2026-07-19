# training/oft_save.mojo — save / reopen TRAINED Diag-OFT adapters in the
# UPSTREAM LyCORIS key convention (pip lycoris_lora 3.4.0, the T2.F oracle).
#
# MEASURED upstream schema (lycoris/modules/diag_oft.py DiagOFTModule):
#
#       "<prefix>.oft_blocks"  BF16 [num_blocks, block_size, block_size]
#       "<prefix>.alpha"       F32 [1]   (upstream semantics: the CONSTRAINT,
#                                         0 = unconstrained — NOT alpha/rank)
#
# 2026-06-11 T2.F skeptic FIX (was `.oft_blocks.weight`, citing a now-deleted
# EDv2 lycoris.rs): upstream uses NO `.weight` suffix, and parametrizes the
# skew as Q = blocks - blocksᵀ while the Mojo adapter uses Q = 0.5*(S - Sᵀ).
# For the SAME Q (hence bit-same rotation) the on-disk tensor must therefore be
#       blocks := 0.5 * S      (reopen: S = 2 * blocks; *0.5/*2 are exact in bf16)
# Upstream applies r = (I+Q)(I-Q)⁻¹ via einsum("k n m, k n i -> k m i", r, W)
# = rᵀ @ W_block, and rᵀ = (I+Q)⁻¹(I-Q) = exactly the Mojo R — so given equal Q
# the effective weight matches upstream (gate: lycoris_family_load_check.py
# loads a Mojo-saved file through DiagOFTModule.make_module_from_state_dict).
#
# 3D shape [num_blocks, b, b] is what disambiguates OFT from BOFT (4D) at load
# time (upstream algo_check: oft_blocks.ndim == 3).
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# - W_base (the frozen base weight the rotation acts on) is NOT saved — it is the
#   model's own weight, not an adapter parameter. The reader returns S + shapes
#   + alpha; the caller re-supplies w_base when reconstructing an OFTAdapter.
# - `.alpha`: upstream OFT reads this as the norm CONSTRAINT (constraint*out_dim
#   clamp on Q), not a scale. The Mojo adapter has no constraint feature, so
#   write alpha=0.0 for upstream-faithful files; nonzero values round-trip but
#   are interpreted as a (usually inactive) clamp by upstream loaders.
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


def _bf16_3d(var values: List[BFloat16], d0: Int, d1: Int, d2: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(d0)
    sh.append(d1)
    sh.append(d2)
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

        # Upstream parametrization: blocks = 0.5 * S (Q = blocks - blocksᵀ
        # upstream == 0.5*(S - Sᵀ) here). *0.5 is an exponent shift — exact in
        # bf16 (values never reach the subnormal floor in practice).
        var blocks = List[BFloat16]()
        for i in range(len(a.s)):
            blocks.append(BFloat16(a.s[i].cast[DType.float32]() * Float32(0.5)))

        names.append(nk.prefix + ".oft_blocks")
        tensors.append(ArcPointer(_bf16_3d(blocks^, NB, B, B, ctx)))

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


# A read-back of one OFT module: S squares + shapes + alpha.
@fieldwise_init
struct OFTReadback(Copyable, Movable):
    var s: List[BFloat16]
    var num_blocks: Int
    var block_size: Int
    var alpha: Float32


# Reopen one module's OFT keys. oft_blocks [num_blocks, b, b] → shapes from its
# 3D shape (3D rank is the OFT-vs-BOFT discriminator). Asserts the keys + alpha.
# Inverts the save parametrization: S = 2 * blocks (exact in bf16).
def read_oft_module(prefix: String, path: String, ctx: DeviceContext) raises -> OFTReadback:
    var st = SafeTensors.open(path)

    var info = st.tensor_info(prefix + ".oft_blocks")
    if len(info.shape) != 3:
        raise Error(String("read_oft_module: oft_blocks must be 3D [num_blocks,b,b], got rank ") + String(len(info.shape)))
    var NB = info.shape[0]
    var B = info.shape[1]
    if info.shape[2] != B:
        raise Error("read_oft_module: oft_blocks last two dims must be equal (square blocks)")
    var blocks = _read_bf16(st, prefix + ".oft_blocks", ctx)
    var s = List[BFloat16]()
    for i in range(len(blocks)):
        s.append(BFloat16(blocks[i].cast[DType.float32]() * Float32(2.0)))

    var alpha_h = _read_f32(st, prefix + ".alpha", ctx)
    if len(alpha_h) != 1:
        raise Error("read_oft_module: .alpha must be a 1-element tensor")

    return OFTReadback(s^, NB, B, alpha_h[0])
