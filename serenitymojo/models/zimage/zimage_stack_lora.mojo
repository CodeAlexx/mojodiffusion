# serenitymojo/models/zimage/zimage_stack_lora.mojo
#
# Z-IMAGE (NextDiT) FULL DiT STACK *WITH LoRA* on every trained projection:
# forward (saving ckpt-inputs) + reduced/full-depth backward (training) using the
# parity-verified per-block LoRA variants (models/zimage/lora_block.mojo), COLLECTS
# every adapter's d_A/d_B, and supports an AdamW step + a PEFT/ai-toolkit save
# across all 7 × (num_nr + num_cr + num_main) adapters. This file COMPOSES; it
# rebuilds NOTHING. Mirrors models/ernie/ernie_stack_lora.mojo (the PROVEN pattern).
#
# WHAT IS ALREADY PROVEN (cos>=0.999 vs torch) AND ONLY REUSED HERE
#   * models/zimage/block.mojo : base block fwd+bwd (modulated 19/19 + refiner 15/15).
#   * models/zimage/zimage_stack.mojo : the BASE full-stack composition (VERDICT
#     PASS, all token/weight/mod grads cos>=0.999). THIS FILE is that file with the
#     base per-block calls swapped for the LoRA variants + LoRA-grad collection.
#   * models/zimage/lora_block.mojo : zimage_block_lora_forward/backward (modulated)
#     and zimage_refiner_lora_forward/backward (unmodulated context refiner); each
#     reduces to base when adapters absent; LoRA d_x summed into the proj-input grad.
#   * training/{lora_save, train_step, optim} : LoraAdapter, _lora_adamw, save_lora_peft.
#
# TARGET SET (OneTrainer baseline filter) — read line-by-line from OT source:
#   ZImageLoRASetup.py:57 -> LoRAModuleWrapper(model.transformer, "transformer",
#   config, config.layer_filter.split(",")). With an empty filter,
#   LoRAModule.py:638-656 (__create_modules) adapts EVERY nn.Linear/Conv2d child of
#   the transformer. The diffusers ZImageTransformer2DModel
#   (transformer_z_image.py:184-224, 359+) gives each block:
#       attention (diffusers Attention -> to_q, to_k, to_v, to_out.0)  [4 Linear]
#       feed_forward (FeedForward -> w1, w3, w2)                        [3 Linear]
#   in noise_refiner.<i>, context_refiner.<i>, and layers.<i>.
#
#   The active OneTrainer Z-Image baseline does NOT use the empty filter. It uses:
#     ^(?=.*attention)(?!.*refiner).*,^(?=.*feed_forward)(?!.*refiner).*
#   so only main `layers.<i>.{attention,feed_forward}` are trainable; noise/context
#   refiners are excluded. This file can carry refiner adapters for parity smokes,
#   but production train uses the `*_main_only` optimizer/save helpers below.
#
# KEY NAMING (round-trip with OT / inference-flame) — slot order Q,K,V,O,w1,w3,w2:
#   OT saves transformer_lora.state_dict() (ZImageLoRASaver.py:24-25), whose keys
#   are the diffusers submodule paths under prefix "transformer.":
#       transformer.<stream>.<i>.attention.{to_q,to_k,to_v,to_out.0}
#       transformer.<stream>.<i>.feed_forward.{w1,w3,w2}
#   with kohya lora_down/lora_up (PeftBase LoRAModule.py:143-144). We REUSE
#   save_lora_peft, which emits PEFT "<prefix>.lora_A.weight"/".lora_B.weight" — the
#   convention train_klein and the inference lora.mojo loader use (lora_save.mojo
#   header) — using the diffusers module path as <prefix> (the Ernie precedent:
#   omit the OT wrapper "transformer." prefix; lora.mojo detects DiffusionModel by
#   the ".lora_A.weight" suffix). A = lora_down [rank,in], B = lora_up [out,rank].
#   The inference-flame zimage_nextdit.rs fuses qkv -> attention.qkv.weight; for the
#   pure-Mojo path the un-fused diffusers names are canonical (zimage/weights.mojo
#   loads to_q/to_k/to_v/to_out.0/feed_forward.{w1,w3,w2} directly).
#
# CARRIER DESIGN (Tenet-2) — mirrors ErnieLoraSet but THREE flat segments:
#   ZImageLoraSet holds ONE flat List[LoraAdapter] laid out as
#       [ nr blocks ][ cr blocks ][ main blocks ]   (each block = 7 slots)
#   indexed by a deterministic scheme: flat = (segment_base + block)*ZIMAGE_SLOTS +
#   slot. The optimizer walks the flat list; the backward SCATTERS each per-block
#   7-slot d_A/d_B into the matching flat entry.
#
# SCOPE: LoRA-on-projection training. Base weights are FROZEN — their grads come
#   from the base path and are discarded for the optimizer; only d_A/d_B are
#   trained. Per-block RAW mod-vec grads are returned (each block backprops them into
#   its OWN adaLN_modulation.0 at step end; Z-Image mod is PER-BLOCK, NOT shared —
#   so NO summation across blocks, unlike Ernie).
#
# Mojo 1.0.0b1: def not fn; Tensor move-only; host List[Float32] carriers.

from std.gpu import global_idx
from std.gpu.host import DeviceContext, HostBuffer
from std.utils.index import IndexList
from std.collections import List, Optional
from std.memory import ArcPointer
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.vec_modulate import vec_modulate
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward, layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs, ZImageBlockGrads, ZImageRefinerGrads,
    zimage_block_forward, zimage_refiner_forward,
)
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockLora, ZImageBlockLoraDevice, ZImageBlockLoraGrads,
    ZImageModVecsDevice, zimage_modvecs_to_device,
    ZImageLoraAdapterDevice, zimage_lora_adapter_to_device, ZIMAGE_SLOTS,
    SLOT_Q, SLOT_K, SLOT_V, SLOT_O, SLOT_W1, SLOT_W3, SLOT_W2,
    zimage_block_lora_forward, zimage_block_lora_backward,
    zimage_block_lora_forward_device, zimage_block_lora_backward_device,
    zimage_block_lora_forward_device_tensor, zimage_block_lora_backward_device_tensors,
    zimage_block_lora_forward_device_tensor_batch,
    zimage_block_lora_backward_device_tensors_batch,
    zimage_modvecs_pack2_to_device,
    zimage_block_lora_predict_device_tensor,
    zimage_block_lora_predict_device_tensor_moddev,
    zimage_refiner_lora_forward, zimage_refiner_lora_backward,
    zimage_block_forward_device_moddev, zimage_refiner_forward_device,
)
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.models.zimage.zimage_stack import (
    ZImageStackForward, _zeros, _ones, _t, _add_lists,
    _concat_img_cap, _split_img_cap, _linear_wdev_bias, _saved_x_out,
)

from serenitymojo.autograd_v2.zimage_block_graph import (
    zimage_block_lora_graph_backward,
)
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_adamw_plain_fused import (
    fused_lora_adamw_plain_step,
    LoraAdamWPlainDeviceState, lora_adamw_plain_device_state_init,
    fused_lora_adamw_plain_step_resident,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.training.lora_save import (
    NamedLora, save_lora_peft, load_lora_for_resume,
    save_lora_train_state, load_lora_train_state, _read_f32,
)


comptime TArc = ArcPointer[Tensor]
comptime _ZIMG_DYN1 = Layout.row_major(-1)
comptime _ZIMG_BLOCK = 256


def _zimage_patch_rows_to_nchw_f32(
    patches: LayoutTensor[DType.float32, _ZIMG_DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _ZIMG_DYN1, MutAnyOrigin],
    channels: Int,
    height: Int,
    width: Int,
    patch: Int,
):
    var idx = Int(global_idx.x)
    var total = channels * height * width
    if idx < total:
        var iw = idx % width
        var rem = idx // width
        var ih = rem % height
        var c = rem // height
        var ht = ih // patch
        var wt = iw // patch
        var ph = ih % patch
        var pw = iw % patch
        var patch_w = width // patch
        var row = ht * patch_w + wt
        var out_ch = channels * patch * patch
        # Z-Image/diffusers stores patch features channel-minor:
        # [Ht,Wt,patch_h,patch_w,C], unlike the generic DiT C-major helper.
        var col = ((ph * patch) + pw) * channels + c
        dst[idx] = rebind[dst.element_type](
            rebind[Scalar[DType.float32]](patches[row * out_ch + col])
        )


def zimage_unpatchify_image_rows_channel_minor(
    patches: Tensor,
    channels: Int,
    height: Int,
    width: Int,
    patch: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Device-only inverse for Z-Image final patch rows.

    The input is the final projection [S, C*p*p]. Only the leading real image
    rows are consumed; caption and image-pad rows stay ignored.
    """
    if patches.dtype() != STDtype.F32:
        raise Error("zimage_unpatchify_image_rows_channel_minor: expected F32 patches")
    var sh = patches.shape()
    if len(sh) != 2:
        raise Error("zimage_unpatchify_image_rows_channel_minor: patches must be [S, C*p*p]")
    if patch <= 0 or height % patch != 0 or width % patch != 0:
        raise Error("zimage_unpatchify_image_rows_channel_minor: patch must divide height/width")
    var real_rows = (height // patch) * (width // patch)
    var out_ch = channels * patch * patch
    if sh[0] < real_rows or sh[1] != out_ch:
        raise Error("zimage_unpatchify_image_rows_channel_minor: patch tensor shape mismatch")

    var total = channels * height * width
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var src_rl = RuntimeLayout[_ZIMG_DYN1].row_major(IndexList[1](patches.numel()))
    var out_rl = RuntimeLayout[_ZIMG_DYN1].row_major(IndexList[1](total))
    var P = LayoutTensor[DType.float32, _ZIMG_DYN1, MutAnyOrigin](
        patches.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var O = LayoutTensor[DType.float32, _ZIMG_DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    var grid = (total + _ZIMG_BLOCK - 1) // _ZIMG_BLOCK
    ctx.enqueue_function[
        _zimage_patch_rows_to_nchw_f32, _zimage_patch_rows_to_nchw_f32
    ](
        P, O, channels, height, width, patch,
        grid_dim=grid, block_dim=_ZIMG_BLOCK,
    )
    return Tensor(out_buf^, [1, channels, height, width], STDtype.F32)


# ── adapter init (A small randn, B=0 — PEFT identity at step 0) ───────────────
# LCG randn byte-identical to ernie_stack_lora._randn / train_step._randn.
def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def make_lora_adapter(
    rank: Int, alpha: Float32, in_f: Int, out_f: Int, seed: UInt64
) -> LoraAdapter:
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        _randn(rank * in_f, seed, 0.01),   # A small randn
        _zeros(out_f * rank),              # B = 0 (adapter identity at init)
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),    # ma / va
        _zeros(out_f * rank), _zeros(out_f * rank),  # mb / vb
    )


# ── the LoRA carrier: every trained adapter, flat-indexed across 3 segments ───
struct ZImageLoraSet(Copyable, Movable):
    var ad: List[LoraAdapter]   # (num_nr + num_cr + num_main) * ZIMAGE_SLOTS
    var num_nr: Int
    var num_cr: Int
    var num_main: Int
    var rank: Int

    def __init__(
        out self, var ad: List[LoraAdapter],
        num_nr: Int, num_cr: Int, num_main: Int, rank: Int,
    ):
        self.ad = ad^
        self.num_nr = num_nr
        self.num_cr = num_cr
        self.num_main = num_main
        self.rank = rank

    # segment base (in BLOCKS) for the three streams.
    def nr_base(self) -> Int:
        return 0

    def cr_base(self) -> Int:
        return self.num_nr

    def main_base(self) -> Int:
        return self.num_nr + self.num_cr

    def num_blocks(self) -> Int:
        return self.num_nr + self.num_cr + self.num_main


struct ZImageLoraDeviceSet(Copyable, Movable):
    var ad: List[ZImageLoraAdapterDevice]
    var num_nr: Int
    var num_cr: Int
    var num_main: Int
    var rank: Int

    def __init__(
        out self, var ad: List[ZImageLoraAdapterDevice],
        num_nr: Int, num_cr: Int, num_main: Int, rank: Int,
    ):
        self.ad = ad^
        self.num_nr = num_nr
        self.num_cr = num_cr
        self.num_main = num_main
        self.rank = rank

    def nr_base(self) -> Int:
        return 0

    def cr_base(self) -> Int:
        return self.num_nr

    def main_base(self) -> Int:
        return self.num_nr + self.num_cr

    def num_blocks(self) -> Int:
        return self.num_nr + self.num_cr + self.num_main


def zimage_lora_set_to_device(
    set: ZImageLoraSet, ctx: DeviceContext
) raises -> ZImageLoraDeviceSet:
    var ad = List[ZImageLoraAdapterDevice]()
    var n = set.num_blocks() * ZIMAGE_SLOTS
    for i in range(n):
        ad.append(zimage_lora_adapter_to_device(set.ad[i], ctx))
    return ZImageLoraDeviceSet(ad^, set.num_nr, set.num_cr, set.num_main, set.rank)


def zimage_lora_set_to_device_resident(
    set: ZImageLoraSet,
    state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
) raises -> ZImageLoraDeviceSet:
    """v2 engine (resident-set): build the device LoRA set ONCE, with every
    MAIN adapter's a/b as zero-copy sub-buffer views into the persistent
    optimizer parameter buffer `state.dev_p` — the in-place AdamW update IS
    the next step's weights; the per-step `zimage_lora_set_to_device` upload
    (~420 syncing from_host calls, 0.057 s/step measured) disappears.
    NR/CR adapters (frozen in the OT baseline) upload once as before.
    `state` must outlive the returned set (views; scratch_ring discipline)."""
    var ad = List[ZImageLoraAdapterDevice]()
    var n = set.num_blocks() * ZIMAGE_SLOTS
    for i in range(n):
        if i < state.start or i >= state.end:
            ad.append(zimage_lora_adapter_to_device(set.ad[i], ctx))
        else:
            var n_a = len(set.ad[i].a)
            var n_b = len(set.ad[i].b)
            var a_off = state.elem_offset(i, False)
            var b_off = state.elem_offset(i, True)
            ad.append(ZImageLoraAdapterDevice(
                TArc(Tensor(
                    state.dev_p.create_sub_buffer[DType.uint8](a_off * 2, n_a * 2),
                    [set.ad[i].rank, set.ad[i].in_f], STDtype.BF16,
                )),
                TArc(Tensor(
                    state.dev_p.create_sub_buffer[DType.uint8](b_off * 2, n_b * 2),
                    [set.ad[i].out_f, set.ad[i].rank], STDtype.BF16,
                )),
                set.ad[i].rank, set.ad[i].in_f, set.ad[i].out_f, set.ad[i].scale,
            ))
    return ZImageLoraDeviceSet(ad^, set.num_nr, set.num_cr, set.num_main, set.rank)


def build_zimage_zero_lora_device_set(
    num_nr: Int, num_cr: Int, num_main: Int, ctx: DeviceContext
) raises -> ZImageLoraDeviceSet:
    var ad = List[ZImageLoraAdapterDevice]()
    var zero = TArc(Tensor.from_host([Float32(0.0)], [1, 1], STDtype.BF16, ctx))
    var n = (num_nr + num_cr + num_main) * ZIMAGE_SLOTS
    for _ in range(n):
        ad.append(ZImageLoraAdapterDevice(
            zero.copy(), zero.copy(), 1, 1, 1, Float32(0.0)
        ))
    return ZImageLoraDeviceSet(ad^, num_nr, num_cr, num_main, 1)


# ── per-block adapter slot shapes (in, out) for slot s, given D (hidden), F (ffn) ─
def _slot_in(s: Int, D: Int, F: Int) -> Int:
    if s == SLOT_W2:   # feed_forward.w2: in = F
        return F
    return D           # to_q/k/v/out: in=D ; w1/w3: in=D


def _slot_out(s: Int, D: Int, F: Int) -> Int:
    if s == SLOT_W1 or s == SLOT_W3:   # w1/w3: out = F
        return F
    return D           # to_q/k/v/out: out=D ; w2: out=D


# Append one block's 7 adapters to `ad` (advances `seed`).
def _append_block_adapters(
    mut ad: List[LoraAdapter], mut seed: UInt64, rank: Int, alpha: Float32, D: Int, F: Int
):
    for s in range(ZIMAGE_SLOTS):
        var in_f = _slot_in(s, D, F)
        var out_f = _slot_out(s, D, F)
        ad.append(make_lora_adapter(rank, alpha, in_f, out_f, seed))
        seed += 1


# ── build the full LoRA set for a Z-Image stack (nr | cr | main segments) ─────
def build_zimage_lora_set(
    num_nr: Int, num_cr: Int, num_main: Int, D: Int, F: Int, rank: Int, alpha: Float32
) -> ZImageLoraSet:
    var ad = List[LoraAdapter]()
    var seed = UInt64(3000)
    for _ in range(num_nr):
        _append_block_adapters(ad, seed, rank, alpha, D, F)
    for _ in range(num_cr):
        _append_block_adapters(ad, seed, rank, alpha, D, F)
    for _ in range(num_main):
        _append_block_adapters(ad, seed, rank, alpha, D, F)
    return ZImageLoraSet(ad^, num_nr, num_cr, num_main, rank)


# Build a transient ZImageBlockLora for block `block_idx` (in flat-block space).
def _block_lora_for(set: ZImageLoraSet, block_idx: Int) -> ZImageBlockLora:
    var base = block_idx * ZIMAGE_SLOTS
    return ZImageBlockLora(
        Optional[LoraAdapter](set.ad[base + SLOT_Q].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_K].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_V].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_O].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_W1].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_W3].copy()),
        Optional[LoraAdapter](set.ad[base + SLOT_W2].copy()),
    )


def _block_lora_dev_for(set: ZImageLoraDeviceSet, block_idx: Int) -> ZImageBlockLoraDevice:
    var base = block_idx * ZIMAGE_SLOTS
    return ZImageBlockLoraDevice(
        set.ad[base + SLOT_Q].copy(),
        set.ad[base + SLOT_K].copy(),
        set.ad[base + SLOT_V].copy(),
        set.ad[base + SLOT_O].copy(),
        set.ad[base + SLOT_W1].copy(),
        set.ad[base + SLOT_W3].copy(),
        set.ad[base + SLOT_W2].copy(),
    )


# ── collected LoRA grads (flat, parallel to ZImageLoraSet) ───────────────────
struct ZImageLoraGrads(Movable):
    var d_a: List[List[Float32]]   # num_blocks * ZIMAGE_SLOTS
    var d_b: List[List[Float32]]
    # load-bearing input-token grads (prove the full chain back to embedder outs).
    var d_x_seq: List[Float32]        # [N_IMG, D]
    var d_cap_seq: List[Float32]      # [N_TXT, D]
    # per-block RAW mod-vec grads (Z-Image mod is PER-BLOCK — not summed).
    var nr_mod: List[List[Float32]]   # num_nr   x [4D]
    var main_mod: List[List[Float32]] # num_main x [4D]
    var d_f_scale: List[Float32]      # [D]
    var d_final_lin: List[Float32]    # [out_ch, D] (base, discarded by AdamW)
    var nonfinite_lora_grads: Int

    def __init__(
        out self,
        var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_x_seq: List[Float32], var d_cap_seq: List[Float32],
        var nr_mod: List[List[Float32]], var main_mod: List[List[Float32]],
        var d_f_scale: List[Float32], var d_final_lin: List[Float32],
        nonfinite_lora_grads: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x_seq = d_x_seq^
        self.d_cap_seq = d_cap_seq^
        self.nr_mod = nr_mod^
        self.main_mod = main_mod^
        self.d_f_scale = d_f_scale^
        self.d_final_lin = d_final_lin^
        self.nonfinite_lora_grads = nonfinite_lora_grads


def _modvec4(g: ZImageBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_scale_msa)):
        o.append(g.d_scale_msa[i])
    for i in range(len(g.d_gate_msa)):
        o.append(g.d_gate_msa[i])
    for i in range(len(g.d_scale_mlp)):
        o.append(g.d_scale_mlp[i])
    for i in range(len(g.d_gate_mlp)):
        o.append(g.d_gate_mlp[i])
    return o^


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


struct _ZImageHostGradLists(Copyable, Movable):
    var d_a: List[List[Float32]]
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


def _host_grad_slice_to_list(
    host: HostBuffer[DType.uint8], offset: Int, numel: Int
) -> List[Float32]:
    var out = List[Float32]()
    var fp = (host.unsafe_ptr() + offset).bitcast[Float32]()
    for i in range(numel):
        out.append(fp[i])
    return out^


def _grad_arc_f32(t: TArc, ctx: DeviceContext) raises -> TArc:
    if t[].dtype() == STDtype.F32:
        return t.copy()
    # Host AdamW stores master params and moments as F32; device grads may be BF16.
    var t32 = cast_tensor(t[], STDtype.F32, ctx)
    return TArc(t32^)


def _zimage_tensor_grads_to_host(
    indices: List[Int], d_a_t: List[TArc], d_b_t: List[TArc],
    total_slots: Int, ctx: DeviceContext,
) raises -> _ZImageHostGradLists:
    var a_f32 = List[TArc]()
    var b_f32 = List[TArc]()
    for i in range(len(d_a_t)):
        a_f32.append(_grad_arc_f32(d_a_t[i], ctx))
    for i in range(len(d_b_t)):
        b_f32.append(_grad_arc_f32(d_b_t[i], ctx))

    var total_bytes = 0
    for i in range(len(a_f32)):
        total_bytes += a_f32[i][].nbytes()
    for i in range(len(b_f32)):
        total_bytes += b_f32[i][].nbytes()

    var host = ctx.enqueue_create_host_buffer[DType.uint8](total_bytes)
    var a_offsets = List[Int]()
    var a_numels = List[Int]()
    var b_offsets = List[Int]()
    var b_numels = List[Int]()
    var cursor = 0
    for i in range(len(a_f32)):
        a_offsets.append(cursor)
        a_numels.append(a_f32[i][].numel())
        var dst_a = host.create_sub_buffer[DType.uint8](cursor, a_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst_a, src_buf=a_f32[i][].buf)
        cursor += a_f32[i][].nbytes()
    for i in range(len(b_f32)):
        b_offsets.append(cursor)
        b_numels.append(b_f32[i][].numel())
        var dst_b = host.create_sub_buffer[DType.uint8](cursor, b_f32[i][].nbytes())
        ctx.enqueue_copy(dst_buf=dst_b, src_buf=b_f32[i][].buf)
        cursor += b_f32[i][].nbytes()
    ctx.synchronize()

    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(total_slots):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    for i in range(len(indices)):
        var flat = indices[i]
        d_a_flat[flat] = _host_grad_slice_to_list(host, a_offsets[i], a_numels[i])
        d_b_flat[flat] = _host_grad_slice_to_list(host, b_offsets[i], b_numels[i])
    return _ZImageHostGradLists(d_a_flat^, d_b_flat^)


# ─────────────────────────────────────────────────────────────────────────────
# RESIDENT LoRA stack (small depth, for the COMPOSITION parity gate). Mirrors
# zimage_stack_forward/backward exactly, swapping per-block calls for LoRA ones.
# Blocks are passed resident (the gate uses NR=1/CR=1/MAIN=2 so they fit 24 GB).
# ─────────────────────────────────────────────────────────────────────────────
def zimage_stack_lora_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    # ── noise refiner (MODULATED + LoRA) on x_seq [N_IMG,D] ──
    var nr_x_in = List[TArc]()
    var xs = x_seq.copy()
    for i in range(num_nr):
        nr_x_in.append(TArc(_t(xs.copy(), [N_IMG, D], ctx)))
        var bl = _block_lora_for(lora, lora.nr_base() + i)
        var fwd = zimage_block_lora_forward[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], nr_mod[i], bl, x_cos, x_sin, D, F, eps, ctx,
        )
        xs = fwd.out.copy()

    # ── context refiner (UNMODULATED + LoRA) on cap_seq [N_TXT,D] ──
    var cr_x_in = List[TArc]()
    var cs = cap_seq.copy()
    for i in range(num_cr):
        cr_x_in.append(TArc(_t(cs.copy(), [N_TXT, D], ctx)))
        var bl = _block_lora_for(lora, lora.cr_base() + i)
        var fwd = zimage_refiner_lora_forward[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], bl, cap_cos, cap_sin, D, F, eps, ctx,
        )
        cs = fwd.out.copy()

    # ── unified = concat([x, cap]) -> [S,D] ──
    var x = _concat_img_cap(xs, cs)

    # ── main layers (MODULATED + LoRA) ──
    var main_x_in = List[TArc]()
    for i in range(num_main):
        main_x_in.append(TArc(_t(x.copy(), [S, D], ctx)))
        var bl = _block_lora_for(lora, lora.main_base() + i)
        var fwd = zimage_block_lora_forward[H, Dh, S](
            x.copy(), main_blocks[i], main_mod[i], bl, uni_cos, uni_sin, D, F, eps, ctx,
        )
        x = fwd.out.copy()

    # ── final layer ──
    var ln_x = layer_norm(
        _t(x.copy(), [S, D], ctx),
        _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    ).to_host(ctx)
    var x_out = modulate(
        _t(ln_x.copy(), [S, D], ctx),
        _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    ).to_host(ctx)
    var patches = _linear_wdev_bias(x_out, final_lin_w, final_lin_b, S, D, ctx)
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()

    return ZImageStackForward(
        out^, x_seq.copy(), cap_seq.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        TArc(_t(x^, [S, D], ctx)), TArc(_t(ln_x^, [S, D], ctx)),
    )


def zimage_stack_lora_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraSet,
    f_scale: List[Float32],
    final_lin_w: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageLoraGrads:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)
    var num_blocks = num_nr + num_cr + num_main

    # ── final-layer backward (identical to base stack) ──
    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var x_out = _saved_x_out(saved, f_scale, D, S, ctx)
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var d_x_out = final_dx.to_host(ctx)
    var d_final_lin = List[Float32]()
    var mbf = modulate_backward(
        _t(d_x_out, [S, D], ctx), saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_ln_x = mbf.d_x.to_host(ctx)
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var lnbf = layer_norm_backward(
        _t(d_ln_x, [S, D], ctx), saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = lnbf.d_x.to_host(ctx)

    # flat LoRA grad slots.
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_blocks * ZIMAGE_SLOTS):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    var nonfinite = 0

    # ── main layers backward (REVERSE; per-block recompute) ──
    var main_mod_rev = List[List[Float32]]()
    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_for(lora, lora.main_base() + bi)
        var refwd = zimage_block_lora_forward[H, Dh, S](
            saved.main_x_in[bi][].to_host(ctx), main_blocks[bi], main_mod[bi], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward[H, Dh, S](
            d_x.copy(), main_blocks[bi], main_mod[bi], bl, refwd.saved,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.base.d_x.copy()
        main_mod_rev.append(_modvec4(bg.base))
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        bi -= 1
    var main_mod_grads = List[List[Float32]]()
    var jm = len(main_mod_rev) - 1
    while jm >= 0:
        main_mod_grads.append(main_mod_rev[jm].copy())
        jm -= 1

    # ── unified seam: split d_x [S,D] -> d_xs (first), d_cs (rest) ──
    var seam = _split_img_cap(d_x, N_IMG, N_TXT, D)
    var d_xs = seam[0].copy()
    var d_cs = seam[1].copy()

    # ── context refiner backward (UNMODULATED + LoRA; REVERSE) ──
    var ci = num_cr - 1
    while ci >= 0:
        var bl = _block_lora_for(lora, lora.cr_base() + ci)
        var refwd = zimage_refiner_lora_forward[H, Dh, N_TXT](
            saved.cr_x_in[ci][].to_host(ctx), cr_blocks[ci], bl, cap_cos, cap_sin, D, F, eps, ctx,
        )
        var bg = zimage_refiner_lora_backward[H, Dh, N_TXT](
            d_cs.copy(), cr_blocks[ci], bl, refwd.saved, cap_cos, cap_sin, D, F, eps, ctx,
        )
        d_cs = bg.base.d_x.copy()
        var base_idx = (lora.cr_base() + ci) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        ci -= 1

    # ── noise refiner backward (MODULATED + LoRA; REVERSE) ──
    var nr_mod_rev = List[List[Float32]]()
    var ni = num_nr - 1
    while ni >= 0:
        var bl = _block_lora_for(lora, lora.nr_base() + ni)
        var refwd = zimage_block_lora_forward[H, Dh, N_IMG](
            saved.nr_x_in[ni][].to_host(ctx), nr_blocks[ni], nr_mod[ni], bl, x_cos, x_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward[H, Dh, N_IMG](
            d_xs.copy(), nr_blocks[ni], nr_mod[ni], bl, refwd.saved, x_cos, x_sin, D, F, eps, ctx,
        )
        d_xs = bg.base.d_x.copy()
        nr_mod_rev.append(_modvec4(bg.base))
        var base_idx = (lora.nr_base() + ni) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            d_a_flat[base_idx + s] = bg.lora.d_a[s].copy()
            d_b_flat[base_idx + s] = bg.lora.d_b[s].copy()
            nonfinite += _nonfinite(bg.lora.d_a[s]) + _nonfinite(bg.lora.d_b[s])
        ni -= 1
    var nr_mod_grads = List[List[Float32]]()
    var jn = len(nr_mod_rev) - 1
    while jn >= 0:
        nr_mod_grads.append(nr_mod_rev[jn].copy())
        jn -= 1

    return ZImageLoraGrads(
        d_a_flat^, d_b_flat^,
        d_xs^, d_cs^,
        nr_mod_grads^, main_mod_grads^,
        d_f_scale^, d_final_lin^,
        nonfinite,
    )


def zimage_stack_lora_forward_main_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    var nr_x_in = List[TArc]()
    var xs = x_seq.copy()
    for i in range(num_nr):
        var fwd = zimage_block_forward[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], nr_mod[i], x_cos, x_sin, D, F, eps, ctx,
        )
        xs = fwd.out.copy()

    var cr_x_in = List[TArc]()
    var cs = cap_seq.copy()
    for i in range(num_cr):
        var fwd = zimage_refiner_forward[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], cap_cos, cap_sin, D, F, eps, ctx,
        )
        cs = fwd.out.copy()

    var x = _concat_img_cap(xs, cs)
    var x_arc = TArc(_t(x^, [S, D], ctx))

    var main_x_in = List[TArc]()
    for i in range(num_main):
        var bl = _block_lora_dev_for(lora, lora.main_base() + i)
        var fwd = zimage_block_lora_forward_device_tensor[H, Dh, S](
            x_arc.copy(), main_blocks[i], main_mod[i], bl, uni_cos, uni_sin, D, F, eps, ctx,
        )
        main_x_in.append(fwd.saved.x.copy())
        x_arc = fwd.out.copy()

    var ln_t = layer_norm(
        x_arc[], _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    )
    var x_out_t = vec_modulate(
        ln_t, _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    )
    var bias = Optional[Tensor](final_lin_b.clone(ctx))
    var patches = linear(x_out_t, final_lin_w, bias^, ctx).to_host(ctx)
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()

    return ZImageStackForward(
        out^, x_seq.copy(), cap_seq.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        x_arc.copy(), TArc(ln_t^),
    )


def zimage_stack_lora_forward_main_device_v2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    main_mod_dev: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    """v2-engine B=1 forward (mandate 2026-06-11): the gated batch engine at
    B=1 — device-resident mod-vecs (ONE packed upload per step via
    zimage_modvecs_all_to_device) instead of per-block syncing `_t()` uploads.
    Math is op-for-op the B=1 path; the batch block functions at B=1 run the
    identical kernels on identical shapes (b2dup gate proved the batch path
    reproduces the B1 trajectory exactly). Old path kept (gate-don't-delete):
    zimage_stack_lora_forward_main_device."""
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    var nr_x_in = List[TArc]()
    var xs = x_seq.copy()
    for i in range(num_nr):
        var fwd = zimage_block_forward[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], nr_mod[i], x_cos, x_sin, D, F, eps, ctx,
        )
        xs = fwd.out.copy()

    var cr_x_in = List[TArc]()
    var cs = cap_seq.copy()
    for i in range(num_cr):
        var fwd = zimage_refiner_forward[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], cap_cos, cap_sin, D, F, eps, ctx,
        )
        cs = fwd.out.copy()

    var x = _concat_img_cap(xs, cs)
    var x_arc = TArc(_t(x^, [S, D], ctx))

    var main_x_in = List[TArc]()
    for i in range(num_main):
        var bl = _block_lora_dev_for(lora, lora.main_base() + i)
        var fwd = zimage_block_lora_forward_device_tensor_batch[1, H, Dh, S](
            x_arc.copy(), main_blocks[i], main_mod_dev[i], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        main_x_in.append(fwd.saved.x.copy())
        x_arc = fwd.out.copy()

    var ln_t = layer_norm(
        x_arc[], _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    )
    var x_out_t = vec_modulate(
        ln_t, _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    )
    var bias = Optional[Tensor](final_lin_b.clone(ctx))
    var patches = linear(x_out_t, final_lin_w, bias^, ctx).to_host(ctx)
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()

    return ZImageStackForward(
        out^, x_seq.copy(), cap_seq.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        x_arc.copy(), TArc(ln_t^),
    )


# ── Phase D.3: final-layer constants as ONE packed device slab ────────────────
# Rows: [ ones | zeros | f_scale ] each [D] F32, one H2D upload (one sync) per
# step instead of three per-call `_t()` uploads in the final layer. Views are
# zero-copy sub-buffers; `slab` owns the allocation and MUST outlive the views
# (scratch_ring discipline — same as ZImageModVecsAllDevice).
struct ZImageFinalConstsDevice(Movable):
    var slab: Tensor
    var ones: TArc      # [D] = 1.0  (layer_norm unit gamma)
    var zeros: TArc     # [D] = 0.0  (layer_norm beta / vec_modulate shift)
    var f_scale: TArc   # [D] final adaLN scale (raw; vec_modulate applies 1+)

    def __init__(
        out self, var slab: Tensor,
        var ones: TArc, var zeros: TArc, var f_scale: TArc,
    ):
        self.slab = slab^
        self.ones = ones^
        self.zeros = zeros^
        self.f_scale = f_scale^


def zimage_final_consts_to_device(
    f_scale: List[Float32], D: Int, ctx: DeviceContext
) raises -> ZImageFinalConstsDevice:
    """Pack ones/zeros/f_scale into a single H2D upload. Values byte-identical
    to the per-call `_t(_ones(D))/_t(_zeros(D))/_t(f_scale)` uploads."""
    var vb = D * 4
    var nbytes = 3 * vb
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for c in range(D):
        fp[c] = Float32(1.0)
    for c in range(D):
        fp[D + c] = Float32(0.0)
    for c in range(D):
        fp[2 * D + c] = f_scale[c]
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()                   # the ONE final-consts upload sync
    var ones = TArc(Tensor(
        dev.create_sub_buffer[DType.uint8](0, vb), [D], STDtype.F32
    ))
    var zeros = TArc(Tensor(
        dev.create_sub_buffer[DType.uint8](vb, vb), [D], STDtype.F32
    ))
    var fs = TArc(Tensor(
        dev.create_sub_buffer[DType.uint8](2 * vb, vb), [D], STDtype.F32
    ))
    return ZImageFinalConstsDevice(
        Tensor(dev^, [3, D], STDtype.F32), ones^, zeros^, fs^
    )


def zimage_stack_lora_forward_main_device_v3[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights],
    nr_mod_dev: List[ZImageModVecsDevice],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    main_mod_dev: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    unit_ones: TArc, unit_zeros: TArc, f_scale_dev: TArc,
    final_bias: Optional[Tensor],
    final_lin_w: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    """Phase D.1+D.3 forward (MOJO_V2_ENGINE_PLAN.md): _v2 with the remaining
    host round trips removed — bit-exact (C14: identical ops in identical
    order on identical values; only WHERE tensors live / WHEN uploaded moved):
      * NR loop on zimage_block_forward_device_moddev (device in/out, packed
        per-NR-block device modvecs) — no per-block host List round trip.
      * CR loop on zimage_refiner_forward_device (device in/out).
      * x enters as ONE upload before NR; caption ONE upload before CR;
        img‖cap concat on DEVICE (ops.tensor_algebra.concat dim 0 = D2D byte
        concat — exact bytes of the host _concat_img_cap + upload).
      * final layer consumes prebuilt unit_ones/unit_zeros/f_scale_dev (one
        packed slab, zimage_final_consts_to_device) + a prebuilt final_bias
        Optional[Tensor] (cloned once per step by the caller, passed borrowed
        — `linear` only reads bias) instead of per-call _t()/clone.
    Returned ZImageStackForward identical to _v2 (out host list, main_x_in,
    x_final, ln_x; nr_x_in/cr_x_in stay empty). Old paths kept (C13).
    NOTE final_bias is Optional[Tensor] not TArc: `linear` takes
    Optional[Tensor] and Tensor is move-only, so a TArc arg would force a
    per-call clone back into an Optional — the borrowed Optional is the only
    zero-cost shape."""
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    var nr_x_in = List[TArc]()
    var xs_arc = TArc(_t(x_seq, [N_IMG, D], ctx))    # the ONE x upload
    for i in range(num_nr):
        xs_arc = zimage_block_forward_device_moddev[H, Dh, N_IMG](
            xs_arc.copy(), nr_blocks[i], nr_mod_dev[i],
            x_cos, x_sin, D, F, eps, ctx,
        )

    var cr_x_in = List[TArc]()
    var cs_arc = TArc(_t(cap_seq, [N_TXT, D], ctx))  # the ONE caption upload
    for i in range(num_cr):
        cs_arc = zimage_refiner_forward_device[H, Dh, N_TXT](
            cs_arc.copy(), cr_blocks[i], cap_cos, cap_sin, D, F, eps, ctx,
        )

    # device concat [N_IMG,D] ‖ [N_TXT,D] -> [S,D] (image first — _concat_img_cap)
    var x_arc = TArc(concat(0, ctx, xs_arc[], cs_arc[]))

    var main_x_in = List[TArc]()
    for i in range(num_main):
        var bl = _block_lora_dev_for(lora, lora.main_base() + i)
        var fwd = zimage_block_lora_forward_device_tensor_batch[1, H, Dh, S](
            x_arc.copy(), main_blocks[i], main_mod_dev[i], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        main_x_in.append(fwd.saved.x.copy())
        x_arc = fwd.out.copy()

    var ln_t = layer_norm(x_arc[], unit_ones[], unit_zeros[], final_eps, ctx)
    var x_out_t = vec_modulate(ln_t, f_scale_dev[], unit_zeros[], ctx)
    var patches = linear(x_out_t, final_lin_w, final_bias, ctx).to_host(ctx)
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    var out = parts[0].copy()

    return ZImageStackForward(
        out^, x_seq.copy(), cap_seq.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        x_arc.copy(), TArc(ln_t^),
    )


def zimage_stack_lora_backward_main_device_v2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    main_blocks: List[ZImageBlockWeights],
    main_mod_dev: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageLoraGrads:
    """v2-engine B=1 backward: batch engine at B=1 with device mod-vecs.
    Same recompute-then-backprop structure as backward_main_device; the
    batch backward additionally skips frozen-gate/base-norm param grads
    (gate_residual_backward_dxdy) — those grads were computed and DISCARDED
    by the old path (frozen-weight skip, flame v2 workstream #4)."""
    var num_main = len(main_blocks)
    var num_blocks = lora.num_blocks()

    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var d_final_lin = List[Float32]()
    var mbf = modulate_backward(
        final_dx, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_x_t = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = TArc(d_x_t^)

    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()

    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var refwd = zimage_block_lora_forward_device_tensor_batch[1, H, Dh, S](
            saved.main_x_in[bi].copy(), main_blocks[bi], main_mod_dev[bi], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward_device_tensors_batch[1, H, Dh, S](
            d_x[], main_blocks[bi], main_mod_dev[bi], bl, refwd.saved,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(bg.d_a[s].copy())
            d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    var host_grads = _zimage_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_blocks * ZIMAGE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    var empty_x = List[Float32]()
    var empty_cap = List[Float32]()
    var empty_nr = List[List[Float32]]()
    var empty_main = List[List[Float32]]()
    return ZImageLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        empty_x^, empty_cap^,
        empty_nr^, empty_main^,
        d_f_scale^, d_final_lin^,
        nonfinite,
    )


def zimage_stack_lora_backward_main_device_v3[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    main_blocks: List[ZImageBlockWeights],
    main_mod_dev: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageLoraGrads:
    """P3 graph backward (ZIMAGE_V2_GRAPH, AUTOGRAD_V2_MOJO_DESIGN.md P3):
    byte-copy of _v2 with the per-block recompute + hand-chain backward pair
    replaced by zimage_block_lora_graph_backward (forward re-recorded through
    the autograd_v2 ops_record wrappers, engine.execute drives the backward -
    recompute-style checkpoint, flame checkpoint.rs shape). The FINAL LAYER
    stays hand-chain in P3 (prologue + grads-to-host + return UNCHANGED);
    old paths untouched (C13)."""
    var num_main = len(main_blocks)
    var num_blocks = lora.num_blocks()

    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var d_final_lin = List[Float32]()
    var mbf = modulate_backward(
        final_dx, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_x_t = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = TArc(d_x_t^)

    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()

    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var bg = zimage_block_lora_graph_backward[H, Dh, S](
            d_x[], main_blocks[bi], main_mod_dev[bi], bl,
            saved.main_x_in[bi], uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(bg.d_a[s].copy())
            d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    var host_grads = _zimage_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_blocks * ZIMAGE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    var empty_x = List[Float32]()
    var empty_cap = List[Float32]()
    var empty_nr = List[List[Float32]]()
    var empty_main = List[List[Float32]]()
    return ZImageLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        empty_x^, empty_cap^,
        empty_nr^, empty_main^,
        d_f_scale^, d_final_lin^,
        nonfinite,
    )


def zimage_refine_x_seq[
    H: Int, Dh: Int, N_IMG: Int
](
    x_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    x_cos: Tensor, x_sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var xs = x_seq.copy()
    for i in range(len(nr_blocks)):
        var fwd = zimage_block_forward[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], nr_mod[i], x_cos, x_sin, D, F, eps, ctx,
        )
        xs = fwd.out.copy()
    return xs^


def zimage_stack_lora_predict_main_from_refined_device_tensor[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    xs: List[Float32], cap_seq: List[Float32],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var main_mod_dev = List[ZImageModVecsDevice]()
    for i in range(len(main_mod)):
        main_mod_dev.append(zimage_modvecs_to_device(main_mod[i], D, ctx))
    return zimage_stack_lora_predict_main_from_refined_moddev_tensor[H, Dh, N_IMG, N_TXT, S](
        xs, cap_seq, cr_blocks, main_blocks, main_mod_dev^, lora,
        f_scale, final_lin_w, final_lin_b,
        cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, out_ch, eps, final_eps, ctx,
    )


def zimage_stack_lora_predict_main_from_refined_moddev_tensor[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    xs: List[Float32], cap_seq: List[Float32],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    var cs = cap_seq.copy()
    for i in range(num_cr):
        var fwd = zimage_refiner_forward[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], cap_cos, cap_sin, D, F, eps, ctx,
        )
        cs = fwd.out.copy()

    var x = _concat_img_cap(xs, cs)
    var x_arc = TArc(_t(x^, [S, D], ctx))

    for i in range(num_main):
        var bl = _block_lora_dev_for(lora, lora.main_base() + i)
        x_arc = zimage_block_lora_predict_device_tensor_moddev[H, Dh, S](
            x_arc.copy(), main_blocks[i], main_mod[i], bl, uni_cos, uni_sin, D, F, eps, ctx,
        )

    var ln_t = layer_norm(
        x_arc[], _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    )
    var x_out_t = modulate(
        ln_t, _t(f_scale.copy(), [D], ctx), _t(_zeros(D), [D], ctx), ctx,
    )
    var bias = Optional[Tensor](final_lin_b.clone(ctx))
    return linear(x_out_t, final_lin_w, bias^, ctx)


def zimage_stack_lora_predict_main_device_tensor[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var xs = zimage_refine_x_seq[H, Dh, N_IMG](
        x_seq, nr_blocks, nr_mod, x_cos, x_sin, D, F, eps, ctx,
    )
    return zimage_stack_lora_predict_main_from_refined_device_tensor[H, Dh, N_IMG, N_TXT, S](
        xs^, cap_seq,
        cr_blocks, main_blocks, main_mod, lora,
        f_scale, final_lin_w, final_lin_b,
        cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, out_ch, eps, final_eps, ctx,
    )


def zimage_stack_lora_predict_main_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq: List[Float32], cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> List[Float32]:
    var patches_t = zimage_stack_lora_predict_main_device_tensor[H, Dh, N_IMG, N_TXT, S](
        x_seq, cap_seq,
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale, final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, out_ch, eps, final_eps, ctx,
    )
    var patches = patches_t.to_host(ctx)
    var parts = _split_img_cap(patches, N_IMG, N_TXT, out_ch)
    return parts[0].copy()


def zimage_stack_lora_backward_main_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    final_lin_w: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
    trace: Bool = False,
) raises -> ZImageLoraGrads:
    var num_main = len(main_blocks)
    var num_blocks = lora.num_blocks()

    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var d_final_lin = List[Float32]()
    var mbf = modulate_backward(
        final_dx, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_f_scale = mbf.d_scale.to_host(ctx)
    var d_x_t = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = TArc(d_x_t^)

    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()

    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var refwd = zimage_block_lora_forward_device_tensor[H, Dh, S](
            saved.main_x_in[bi].copy(), main_blocks[bi], main_mod[bi], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward_device_tensors[H, Dh, S](
            d_x[], main_blocks[bi], main_mod[bi], bl, refwd.saved,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(bg.d_a[s].copy())
            d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    var host_grads = _zimage_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_blocks * ZIMAGE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    var empty_x = List[Float32]()
    var empty_cap = List[Float32]()
    var empty_nr = List[List[Float32]]()
    var empty_main = List[List[Float32]]()
    return ZImageLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        empty_x^, empty_cap^,
        empty_nr^, empty_main^,
        d_f_scale^, d_final_lin^,
        nonfinite,
    )


# ── AdamW step on EVERY adapter (reuses the proven per-adapter _lora_adamw) ───
def zimage_lora_adamw_step(
    mut set: ZImageLoraSet, grads: ZImageLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var n = set.num_blocks() * ZIMAGE_SLOTS
    for i in range(n):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── AdamW step on OneTrainer baseline trainable adapters only: main layers. ──
# Hot path: ONE fused GPU launch with the IDENTICAL plain-AdamW math
# (training/lora_adamw_plain_fused.mojo; gate
# training/lora_adamw_plain_fused_parity.mojo — host vs device ±1-ulp class).
# Host loop kept below as the gate reference.
comptime ZIMAGE_FUSED_ADAMW = True


def zimage_lora_adamw_step_main_only(
    mut set: ZImageLoraSet, grads: ZImageLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var start = set.main_base() * ZIMAGE_SLOTS
    var end = set.num_blocks() * ZIMAGE_SLOTS
    comptime if ZIMAGE_FUSED_ADAMW:
        fused_lora_adamw_plain_step(
            set.ad, grads.d_a, grads.d_b, start, end,
            t, lr, beta1, beta2, eps, weight_decay, ctx,
        )
    else:
        for i in range(start, end):
            var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
            _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


def zimage_lora_adamw_step_main_only_unfused(
    mut set: ZImageLoraSet, grads: ZImageLoraGrads, t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    var start = set.main_base() * ZIMAGE_SLOTS
    var end = set.num_blocks() * ZIMAGE_SLOTS
    for i in range(start, end):
        var lg = LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())
        _lora_adamw(set.ad[i], lg, t, lr, ctx, beta1, beta2, eps, weight_decay)


# ── per-block PEFT/kohya prefix scheme (the INVERSE of the inference target map) ─
# slot -> diffusers module suffix (transformer_z_image.py + zimage/weights.mojo).
def _slot_suffix(slot: Int) -> String:
    if slot == SLOT_Q:
        return String(".attention.to_q")
    elif slot == SLOT_K:
        return String(".attention.to_k")
    elif slot == SLOT_V:
        return String(".attention.to_v")
    elif slot == SLOT_O:
        return String(".attention.to_out.0")
    elif slot == SLOT_W1:
        return String(".feed_forward.w1")
    elif slot == SLOT_W3:
        return String(".feed_forward.w3")
    return String(".feed_forward.w2")


# stream prefix for a flat block index (nr | cr | main). Matches inference-flame
# zimage_nextdit.rs: noise_refiner.{i} / context_refiner.{i} / layers.{i}.
def _stream_prefix(set: ZImageLoraSet, block_idx: Int) -> String:
    if block_idx < set.cr_base():
        return String("noise_refiner.") + String(block_idx - set.nr_base())
    elif block_idx < set.main_base():
        return String("context_refiner.") + String(block_idx - set.cr_base())
    return String("layers.") + String(block_idx - set.main_base())


def _zimage_lora_prefix(set: ZImageLoraSet, block_idx: Int, slot: Int) -> String:
    return _stream_prefix(set, block_idx) + _slot_suffix(slot)


def zimage_lora_prefixes(set: ZImageLoraSet) -> List[String]:
    var out = List[String]()
    for bi in range(set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            out.append(_zimage_lora_prefix(set, bi, s))
    return out^


# ── SAVE every adapter as a PEFT/ai-toolkit safetensors ──────────────────────
def save_zimage_lora(set: ZImageLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            named.append(NamedLora(
                _zimage_lora_prefix(set, bi, s),
                set.ad[bi * ZIMAGE_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


def save_zimage_lora_main_only(set: ZImageLoraSet, path: String, ctx: DeviceContext) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.main_base(), set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            named.append(NamedLora(
                _zimage_lora_prefix(set, bi, s),
                set.ad[bi * ZIMAGE_SLOTS + s].copy(),
            ))
    return save_lora_peft(named, path, ctx)


def save_zimage_lora_main_only_state(
    set: ZImageLoraSet, path: String, ctx: DeviceContext
) raises -> Int:
    var named = List[NamedLora]()
    for bi in range(set.main_base(), set.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            named.append(NamedLora(
                _zimage_lora_prefix(set, bi, s),
                set.ad[bi * ZIMAGE_SLOTS + s].copy(),
            ))
    return save_lora_train_state(named, path, ctx)


# ── RESUME: load the adapter A/B back from a save_zimage_lora file ───────────
# AdamW moments are ZEROED (resume them from a loop TrainState checkpoint). The
# returned set carries the SAME flat order build_zimage_lora_set produces.
def load_zimage_lora_resume(
    num_nr: Int, num_cr: Int, num_main: Int, rank: Int, alpha: Float32,
    path: String, ctx: DeviceContext,
) raises -> ZImageLoraSet:
    # build a transient set to derive the flat prefix order, then overwrite A/B.
    var template = build_zimage_lora_set(num_nr, num_cr, num_main, 1, 1, rank, alpha)
    var prefixes = zimage_lora_prefixes(template)
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)   # flat order
    var ad = List[LoraAdapter]()
    for i in range(template.num_blocks() * ZIMAGE_SLOTS):
        ad.append(named[i].adapter.copy())
    return ZImageLoraSet(ad^, num_nr, num_cr, num_main, rank)


def load_zimage_lora_main_only_resume(
    num_nr: Int, num_cr: Int, num_main: Int, rank: Int, alpha: Float32,
    D: Int, F: Int, path: String, ctx: DeviceContext,
) raises -> ZImageLoraSet:
    var template = build_zimage_lora_set(num_nr, num_cr, num_main, D, F, rank, alpha)
    var prefixes = List[String]()
    for bi in range(template.main_base(), template.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            prefixes.append(_zimage_lora_prefix(template, bi, s))
    var scale = alpha / Float32(rank)
    var named = load_lora_for_resume(prefixes, scale, path, ctx)
    var j = 0
    for bi in range(template.main_base(), template.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            template.ad[bi * ZIMAGE_SLOTS + s] = named[j].adapter.copy()
            j += 1
    return template^


def load_zimage_lora_main_only_state(
    num_nr: Int, num_cr: Int, num_main: Int, rank: Int, alpha: Float32,
    D: Int, F: Int, path: String, ctx: DeviceContext,
) raises -> ZImageLoraSet:
    var template = build_zimage_lora_set(num_nr, num_cr, num_main, D, F, rank, alpha)
    var prefixes = List[String]()
    for bi in range(template.main_base(), template.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            prefixes.append(_zimage_lora_prefix(template, bi, s))
    var scale = alpha / Float32(rank)
    var named = load_lora_train_state(prefixes, scale, path, ctx)
    var j = 0
    for bi in range(template.main_base(), template.num_blocks()):
        for s in range(ZIMAGE_SLOTS):
            template.ad[bi * ZIMAGE_SLOTS + s] = named[j].adapter.copy()
            j += 1
    return template^


# ── COMFY/KOHYA Z-IMAGE LoRA LOAD (musubi-tuner networks.lora_zimage) ────────
# Key shape (verified on eri2_zimage_lora_comfy.safetensors, 2026-06-10):
#   lora_unet_layers_{i}_attention_qkv.{lora_down.weight,lora_up.weight,alpha}
#   lora_unet_layers_{i}_attention_out.{...}
#   lora_unet_layers_{i}_feed_forward_w{1,2,3}.{...}
# MAIN layers only (no refiner modules exist in the format). attention qkv is
# trained FUSED against the single-file checkpoint's layers.{i}.attention.qkv
# [3D, D]; the un-fused pure-Mojo stack needs per-projection adapters, and the
# split is EXACT: with up=[3D, r], down=[r, D],
#   ΔW_qkv = up @ down  ⇒  ΔW_q = up[0:D] @ down (rows; k/v the next chunks)
# so to_q/to_k/to_v share A=down and take the matching B row-chunk (q,k,v
# order — the Z-Image qkv Linear is the standard q|k|v concat).
# Per-module scale = mult * alpha/rank (kohya: missing .alpha ⇒ scale = mult).

def zimage_lora_file_is_comfy(path: String) raises -> Bool:
    """True when the file uses the comfy/kohya Z-Image export naming
    (`lora_unet_layers_...` keys) rather than the trainer resume format."""
    var st = SafeTensors.open(path)
    for ref n in st.names():
        if n.startswith("lora_unet_"):
            return True
    return False


def _comfy_zero_adapter() -> LoraAdapter:
    """Scale-0 placeholder for slots the comfy file does not cover (nr/cr
    refiner blocks); zimage_lora_apply_device skips scale==0 adapters — the
    same shape/scale convention build_zimage_zero_lora_device_set uses."""
    return LoraAdapter(
        _zeros(1), _zeros(1), 1, 1, 1, Float32(0.0),
        _zeros(1), _zeros(1), _zeros(1), _zeros(1),
    )


def _comfy_scale(
    st: SafeTensors, key: String, rank: Int, mult: Float32, ctx: DeviceContext
) raises -> Float32:
    var key_alpha = key + ".alpha"
    if key_alpha not in st.tensors:
        return mult  # kohya convention: no .alpha ⇒ alpha = rank ⇒ scale 1·mult
    var alpha_h = _read_f32(st, key_alpha, ctx)
    if len(alpha_h) != 1:
        raise Error(String("zimage comfy lora: .alpha must be scalar for ") + key)
    return mult * alpha_h[0] / Float32(rank)


def _comfy_module_adapter(
    st: SafeTensors, key: String, mult: Float32, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> LoraAdapter:
    """One un-fused comfy module → LoraAdapter. A=lora_down [r,in],
    B=lora_up [out,r] (shape-checked against the base projection)."""
    var key_a = key + ".lora_down.weight"
    var key_b = key + ".lora_up.weight"
    if key_a not in st.tensors or key_b not in st.tensors:
        raise Error(
            String("zimage comfy lora: missing ") + key_a + " / " + key_b
        )
    var a_info = st.tensor_info(key_a)
    var b_info = st.tensor_info(key_b)
    if len(a_info.shape) != 2 or len(b_info.shape) != 2:
        raise Error(String("zimage comfy lora: A/B must be 2-D for ") + key)
    var rank = a_info.shape[0]
    if a_info.shape[1] != in_f or b_info.shape[0] != out_f or b_info.shape[1] != rank:
        raise Error(
            String("zimage comfy lora: shape mismatch for ") + key
            + ": A=[" + String(a_info.shape[0]) + "," + String(a_info.shape[1])
            + "] B=[" + String(b_info.shape[0]) + "," + String(b_info.shape[1])
            + "] vs expected in=" + String(in_f) + " out=" + String(out_f)
        )
    var scale = _comfy_scale(st, key, rank, mult, ctx)
    return LoraAdapter(
        _read_f32(st, key_a, ctx), _read_f32(st, key_b, ctx),
        rank, in_f, out_f, scale,
        _zeros(rank * in_f), _zeros(rank * in_f),
        _zeros(out_f * rank), _zeros(out_f * rank),
    )


def _comfy_qkv_adapters(
    st: SafeTensors, key: String, mult: Float32, D: Int, ctx: DeviceContext,
) raises -> List[LoraAdapter]:
    """Split the fused attention_qkv LoRA exactly into [to_q, to_k, to_v]
    adapters (shared A = down; B = the matching up row-chunk). Exact — no
    approximation (see the section comment)."""
    var key_a = key + ".lora_down.weight"
    var key_b = key + ".lora_up.weight"
    if key_a not in st.tensors or key_b not in st.tensors:
        raise Error(
            String("zimage comfy lora: missing ") + key_a + " / " + key_b
        )
    var a_info = st.tensor_info(key_a)
    var b_info = st.tensor_info(key_b)
    if len(a_info.shape) != 2 or len(b_info.shape) != 2:
        raise Error(String("zimage comfy lora: A/B must be 2-D for ") + key)
    var rank = a_info.shape[0]
    if a_info.shape[1] != D or b_info.shape[0] != 3 * D or b_info.shape[1] != rank:
        raise Error(
            String("zimage comfy lora: fused qkv shape mismatch for ") + key
            + ": A=[" + String(a_info.shape[0]) + "," + String(a_info.shape[1])
            + "] B=[" + String(b_info.shape[0]) + "," + String(b_info.shape[1])
            + "] vs expected A=[r," + String(D) + "] B=[" + String(3 * D) + ",r]"
        )
    var scale = _comfy_scale(st, key, rank, mult, ctx)
    var a_h = _read_f32(st, key_a, ctx)
    var b_h = _read_f32(st, key_b, ctx)
    var out = List[LoraAdapter]()
    var chunk = D * rank  # rows are [out, rank] row-major ⇒ contiguous chunks
    for c in range(3):
        var b_c = List[Float32]()
        for j in range(chunk):
            b_c.append(b_h[c * chunk + j])
        out.append(LoraAdapter(
            a_h.copy(), b_c^, rank, D, D, scale,
            _zeros(rank * D), _zeros(rank * D),
            _zeros(D * rank), _zeros(D * rank),
        ))
    return out^


def load_zimage_lora_main_only_comfy(
    num_nr: Int, num_cr: Int, num_main: Int, D: Int, F: Int,
    mult: Float32, path: String, ctx: DeviceContext,
) raises -> ZImageLoraSet:
    """Load a comfy/kohya Z-Image LoRA export into the flat ZImageLoraSet
    order (nr|cr|main × Q,K,V,O,W1,W3,W2). rank/alpha come from the FILE
    (per module); `mult` is the request's LoRA weight. nr/cr slots get
    scale-0 placeholders (the format carries main layers only)."""
    var st = SafeTensors.open(path)
    var ad = List[LoraAdapter]()
    for _ in range((num_nr + num_cr) * ZIMAGE_SLOTS):
        ad.append(_comfy_zero_adapter())
    var file_rank = 0
    for i in range(num_main):
        var base = String("lora_unet_layers_") + String(i)
        var qkv = _comfy_qkv_adapters(st, base + "_attention_qkv", mult, D, ctx)
        ad.append(qkv[0].copy())   # SLOT_Q
        ad.append(qkv[1].copy())   # SLOT_K
        ad.append(qkv[2].copy())   # SLOT_V
        ad.append(_comfy_module_adapter(st, base + "_attention_out", mult, D, D, ctx))    # SLOT_O
        ad.append(_comfy_module_adapter(st, base + "_feed_forward_w1", mult, D, F, ctx))  # SLOT_W1
        ad.append(_comfy_module_adapter(st, base + "_feed_forward_w3", mult, D, F, ctx))  # SLOT_W3
        ad.append(_comfy_module_adapter(st, base + "_feed_forward_w2", mult, F, D, ctx))  # SLOT_W2
        if file_rank == 0:
            file_rank = ad[len(ad) - 1].rank
    print("[zimage][lora] comfy load:", num_main, "main layers, rank",
          file_rank, "(fused qkv split exactly into q/k/v)")
    return ZImageLoraSet(ad^, num_nr, num_cr, num_main, file_rank)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH-2 stacked-rows drivers (2026-06-11, OT-parity batch lever).
# Two samples stacked along rows: x = [xs0|cs0|xs1|cs1] = [2S, D]. Per-sample
# adaLN via [2, D] modvec tensors (zimage_modvecs_pack2_to_device); per-sample
# uni rope tables are concatenated by building rope over the CONCATENATED
# position list (positions per row → concat positions == tiled tables).
# Refiners (frozen) run per sample through the existing host-path functions.
# GATE: training/zimage_batch2_parity.mojo — loss_B2{s0,s1} vs
# mean(loss_B1(s0), loss_B1(s1)) with identical draws + grad average match.
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageStackForwardB2(Movable):
    var out0: List[Float32]             # sample-0 [N_IMG, out_ch] host patches
    var out1: List[Float32]             # sample-1 [N_IMG, out_ch]
    var main_x_in: List[TArc]           # num_main x [2S, D] block inputs
    var x_final: TArc                   # [2S, D]
    var ln_x: TArc                      # [2S, D]

    def __init__(
        out self,
        var out0: List[Float32], var out1: List[Float32],
        var main_x_in: List[TArc], var x_final: TArc, var ln_x: TArc,
    ):
        self.out0 = out0^
        self.out1 = out1^
        self.main_x_in = main_x_in^
        self.x_final = x_final^
        self.ln_x = ln_x^


def zimage_stack_lora_forward_main_device_b2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    x_seq0: List[Float32], cap_seq0: List[Float32],
    x_seq1: List[Float32], cap_seq1: List[Float32],
    nr_blocks: List[ZImageBlockWeights],
    nr_mod0: List[ZImageModVecs], nr_mod1: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    main_mod_b2: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    f_scale2: List[Float32],            # [2*D] per-sample final modulation
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_cos: Tensor, x_sin: Tensor,
    cap_cos0: Tensor, cap_sin0: Tensor,
    cap_cos1: Tensor, cap_sin1: Tensor,
    uni_cos2: Tensor, uni_sin2: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForwardB2:
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    # frozen refiners, per sample (same math as the B=1 driver)
    var xs0 = x_seq0.copy()
    var xs1 = x_seq1.copy()
    for i in range(num_nr):
        var f0 = zimage_block_forward[H, Dh, N_IMG](
            xs0.copy(), nr_blocks[i], nr_mod0[i], x_cos, x_sin, D, F, eps, ctx,
        )
        xs0 = f0.out.copy()
        var f1 = zimage_block_forward[H, Dh, N_IMG](
            xs1.copy(), nr_blocks[i], nr_mod1[i], x_cos, x_sin, D, F, eps, ctx,
        )
        xs1 = f1.out.copy()
    var cs0 = cap_seq0.copy()
    var cs1 = cap_seq1.copy()
    for i in range(num_cr):
        var f0 = zimage_refiner_forward[H, Dh, N_TXT](
            cs0.copy(), cr_blocks[i], cap_cos0, cap_sin0, D, F, eps, ctx,
        )
        cs0 = f0.out.copy()
        var f1 = zimage_refiner_forward[H, Dh, N_TXT](
            cs1.copy(), cr_blocks[i], cap_cos1, cap_sin1, D, F, eps, ctx,
        )
        cs1 = f1.out.copy()

    # stacked unified sequence [2S, D]
    var x = _concat_img_cap(xs0, cs0)
    var x1 = _concat_img_cap(xs1, cs1)
    for i in range(len(x1)):
        x.append(x1[i])
    var x_arc = TArc(_t(x^, [2 * S, D], ctx))

    var main_x_in = List[TArc]()
    for i in range(num_main):
        var bl = _block_lora_dev_for(lora, lora.main_base() + i)
        var fwd = zimage_block_lora_forward_device_tensor_batch[2, H, Dh, S](
            x_arc.copy(), main_blocks[i], main_mod_b2[i], bl,
            uni_cos2, uni_sin2, D, F, eps, ctx,
        )
        main_x_in.append(fwd.saved.x.copy())
        x_arc = fwd.out.copy()

    var ln_t = layer_norm(
        x_arc[], _t(_ones(D), [D], ctx), _t(_zeros(D), [D], ctx), final_eps, ctx,
    )
    var x_out_t = modulate(
        ln_t, _t(f_scale2.copy(), [2, D], ctx), _t(_zeros(2 * D), [2, D], ctx), ctx,
    )
    var bias = Optional[Tensor](final_lin_b.clone(ctx))
    var patches = linear(x_out_t, final_lin_w, bias^, ctx).to_host(ctx)

    var out0 = List[Float32]()
    for i in range(N_IMG * out_ch):
        out0.append(patches[i])
    var out1 = List[Float32]()
    var base1 = S * out_ch
    for i in range(N_IMG * out_ch):
        out1.append(patches[base1 + i])

    return ZImageStackForwardB2(
        out0^, out1^, main_x_in^, x_arc.copy(), TArc(ln_t^),
    )


def zimage_stack_lora_backward_main_device_b2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out0: List[Float32], d_out1: List[Float32],
    main_blocks: List[ZImageBlockWeights],
    main_mod_b2: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    f_scale2: List[Float32],
    final_lin_w: Tensor,
    uni_cos2: Tensor, uni_sin2: Tensor,
    saved: ZImageStackForwardB2,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageLoraGrads:
    var num_main = len(main_blocks)
    var num_blocks = lora.num_blocks()

    # d_patches [2S, out_ch]: per-sample img grads, zeros on cap rows.
    var d_patches = d_out0.copy()
    for _i in range(N_TXT * out_ch):
        d_patches.append(Float32(0.0))
    for i in range(len(d_out1)):
        d_patches.append(d_out1[i])
    for _i in range(N_TXT * out_ch):
        d_patches.append(Float32(0.0))

    var final_dx = linear_backward_dx(
        _t(d_patches^, [2 * S, out_ch], ctx), final_lin_w, 2 * S, D, out_ch, ctx,
    )
    var mbf = modulate_backward(
        final_dx, saved.ln_x[], _t(f_scale2.copy(), [2, D], ctx), ctx,
        compute_param_grads=False,
    )
    var d_x_t = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = TArc(d_x_t^)

    var grad_indices = List[Int]()
    var d_a_t = List[TArc]()
    var d_b_t = List[TArc]()

    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var refwd = zimage_block_lora_forward_device_tensor_batch[2, H, Dh, S](
            saved.main_x_in[bi].copy(), main_blocks[bi], main_mod_b2[bi], bl,
            uni_cos2, uni_sin2, D, F, eps, ctx,
        )
        var bg = zimage_block_lora_backward_device_tensors_batch[2, H, Dh, S](
            d_x[], main_blocks[bi], main_mod_b2[bi], bl, refwd.saved,
            uni_cos2, uni_sin2, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(bg.d_a[s].copy())
            d_b_t.append(bg.d_b[s].copy())
        bi -= 1

    var host_grads = _zimage_tensor_grads_to_host(
        grad_indices, d_a_t, d_b_t, num_blocks * ZIMAGE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(grad_indices)):
        var idx = grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])

    var empty_x = List[Float32]()
    var empty_cap = List[Float32]()
    var empty_nr = List[List[Float32]]()
    var empty_main = List[List[Float32]]()
    return ZImageLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        empty_x^, empty_cap^,
        empty_nr^, empty_main^,
        _zeros(D), List[Float32](),
        nonfinite,
    )
