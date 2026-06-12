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
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.utils.index import IndexList
from std.collections import List, Optional
from std.memory import ArcPointer
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear, linear_slab
from serenitymojo.ops.norm import layer_norm, layer_norm_slab
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.vec_modulate import vec_modulate, vec_modulate_slab
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dx_slab,
)
from serenitymojo.ops.norm_backward import (
    layer_norm_backward, layer_norm_backward_dx, layer_norm_backward_dx_slab,
)
from serenitymojo.ops.elementwise_backward import (
    modulate_backward, modulate_backward_slab,
)

from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs, ZImageBlockGrads, ZImageRefinerGrads,
    zimage_block_forward, zimage_refiner_forward,
)
from serenitymojo.models.zimage.lora_block import (
    ZImageBlockFullFTBackward, zimage_block_backward_device_tensors_fullft,
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
    zimage_block_forward_device_moddev_slab, zimage_refiner_forward_device_slab,
    zimage_block_lora_forward_device_only_slab,
)
from serenitymojo.ops.tensor_algebra import concat, add
from serenitymojo.models.zimage.zimage_stack import (
    ZImageStackForward, _zeros, _ones, _t, _add_lists,
    _concat_img_cap, _split_img_cap, _linear_wdev_bias, _saved_x_out,
)

from serenitymojo.autograd_v2.zimage_block_graph import (
    zimage_block_lora_graph_backward,
    zimage_block_lora_graph_backward_slab,
)
from serenitymojo.autograd_v2.step_slab import StepSlab
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


def _copy_out_of_slab(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    """One d2d into a NON-slab buffer (to_host-free): results that must
    survive a StepSlab.rewind (the per-block d_x carrier + adapter d_a/d_b)
    leave the slab through here. Single-stream ordering makes this safe: the
    copy is enqueued BEFORE any post-rewind kernel that reuses the slab bytes
    (P4 instruction; RING_ALLOC_DESIGN.md invariant 4 — rewind never frees)."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(dst_buf=buf, src_buf=t.buf)
    return Tensor(buf^, t.shape(), t.dtype())


def zimage_stack_lora_backward_main_device_v4[
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
    mut slab: StepSlab,
) raises -> ZImageLoraGrads:
    """P4 slab backward (ZIMAGE_V2_SLAB, AUTOGRAD_V2_MOJO_DESIGN.md P4 /
    contract C8): copy of _v3 with the per-block graph backward routed
    through StepSlab — slab.mark() before each block, block results copied
    OUT of the slab (one d2d each, no host staging) and slab.rewind(mark)
    after, so every step replays the identical allocation sequence
    (deterministic offsets — the P5 capture precondition). The FINAL-LAYER
    prologue + grads-to-host + return stay the _v3 hand-chain (non-slab);
    _v2/_v3 untouched (C13)."""
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
        # Per-block slab region: mark -> recompute+backward in slab -> copy
        # results out -> rewind (steady-state reuse, identical offsets/step).
        var m = slab.mark()
        var bg = zimage_block_lora_graph_backward_slab[H, Dh, S](
            d_x[], main_blocks[bi], main_mod_dev[bi], bl,
            saved.main_x_in[bi], uni_cos, uni_sin, D, F, eps, ctx, slab,
        )
        d_x = TArc(_copy_out_of_slab(bg.d_x[], ctx))
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(TArc(_copy_out_of_slab(bg.d_a[s][], ctx)))
            d_b_t.append(TArc(_copy_out_of_slab(bg.d_b[s][], ctx)))
        slab.rewind(m)
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
    var xs_arc = TArc(_t(x_seq, [N_IMG, D], ctx))
    for i in range(len(nr_blocks)):
        var mv_dev = zimage_modvecs_to_device(nr_mod[i], D, ctx)
        xs_arc = zimage_block_forward_device_moddev[H, Dh, N_IMG](
            xs_arc.copy(), nr_blocks[i], mv_dev, x_cos, x_sin, D, F, eps, ctx,
        )
    return xs_arc[].to_host(ctx)


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

    var cs_arc = TArc(_t(cap_seq, [N_TXT, D], ctx))
    for i in range(num_cr):
        cs_arc = zimage_refiner_forward_device[H, Dh, N_TXT](
            cs_arc.copy(), cr_blocks[i], cap_cos, cap_sin, D, F, eps, ctx,
        )

    var xs_arc = TArc(_t(xs, [N_IMG, D], ctx))
    var x_arc = TArc(concat(0, ctx, xs_arc[], cs_arc[]))

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


def zimage_stack_lora_backward_main_device_b2_graph[
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
    mut slab: StepSlab,
) raises -> ZImageLoraGrads:
    """P7 graph backward for the BATCH-2 step (ZIMAGE_V2_GRAPH_B2,
    AUTOGRAD_V2_MOJO_DESIGN.md P7): byte-copy of
    zimage_stack_lora_backward_main_device_b2 (above) with the per-block
    recompute + hand-chain pair replaced by the SLAB graph backward at B=2
    (zimage_block_lora_graph_backward_slab — the _v4 per-block mark/copy-out/
    rewind discipline; the NON-slab graph path OOMs at B=2: per-slot fan-in
    buffers + the full recompute graph exceed the pool headroom, measured
    2026-06-11). Final-layer prologue, grads-to-host and the return are
    UNCHANGED; old path untouched (C13)."""
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
        # Per-block slab region (the _v4 discipline): mark -> recompute +
        # backward in slab -> copy results out -> rewind.
        var m = slab.mark()
        var bg = zimage_block_lora_graph_backward_slab[H, Dh, S, 2](
            d_x[], main_blocks[bi], main_mod_b2[bi], bl,
            saved.main_x_in[bi], uni_cos2, uni_sin2, D, F, eps, ctx, slab,
        )
        d_x = TArc(_copy_out_of_slab(bg.d_x[], ctx))
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            grad_indices.append(base_idx + s)
            d_a_t.append(TArc(_copy_out_of_slab(bg.d_a[s][], ctx)))
            d_b_t.append(TArc(_copy_out_of_slab(bg.d_b[s][], ctx)))
        slab.rewind(m)
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


# ══════════════════════════════════════════════════════════════════════════════
# Phase P5 (AUTOGRAD_V2_MOJO_DESIGN.md C9): CUDA-graph capture surface for the
# B=1 step. Everything a captured kernel touches lives at a FIXED address
# across steps:
#   * ZImageStepIO — per-bucket persistent device buffers for every per-step
#     INPUT (x_t, cap_seq, modvec slabs, final consts, rope tables, d_loss
#     root) written via H2D copy from staging BEFORE the graph launch, plus
#     the persistent OUTPUT carriers (d_f_scale, per-block d_x ping-pong, the
#     420 adapter grads) and the forward x-chain (block inputs/outputs).
#   * _v5 forward — the _v3 op chain routed through the _slab op variants on a
#     SECOND StepSlab (fwd_slab, reset per step): per-block mark/rewind with
#     the surviving x copied into the persistent chain. patches stay in a
#     fixed slab buffer; .to_host happens AFTER the graph launch.
#   * _v5 backward — the _v4 chain with the final-layer prologue slab-routed
#     (d_loss enters via io.d_patches; d_f_scale reduced in-graph into a fixed
#     buffer) and the per-block copy-outs landing in the persistent grads
#     region instead of fresh pool buffers.
# Same ops, same order, same values as _v3/_v4 (C14 bit-gates); only WHERE
# tensors live moved. Old paths untouched (C13).
# ══════════════════════════════════════════════════════════════════════════════

from serenitymojo.models.zimage.real_weights import ZImageRopeHost


struct ZImageStackForwardV5(Copyable, Movable):
    """_v5 forward result: device-resident, fixed-address. `patches` is the
    final projection [S, out_ch] in the fwd slab (deterministic address);
    main_x_in are views of the persistent x-chain; x_final == x_chain[last];
    ln_x is the fwd-slab layer_norm output the bwd prologue re-reads."""
    var patches: TArc
    var main_x_in: List[TArc]
    var x_final: TArc
    var ln_x: TArc

    def __init__(
        out self, var patches: TArc, var main_x_in: List[TArc],
        var x_final: TArc, var ln_x: TArc,
    ):
        self.patches = patches^
        self.main_x_in = main_x_in^
        self.x_final = x_final^
        self.ln_x = ln_x^


struct ZImageStepIO(Copyable, Movable):
    """Per-bucket fixed-address step I/O (contract C9 pointer stability).

    ONE device input slab holds every per-step input value; views are carved
    once at init. Per step the trainer packs host values into a pinned staging
    buffer and issues ONE H2D copy (values change, pointers don't) — value
    bytes identical to the _v3 builders (same pack layouts). Output carriers
    and the forward x-chain are plain persistent device buffers."""
    # dims
    var n_img: Int
    var n_txt: Int
    var s_total: Int
    var d_model: Int
    var out_ch: Int
    var num_nr: Int
    var num_main: Int
    var rope_half: Int
    var n_heads: Int
    # input slab + element offsets
    var in_slab: TArc           # [in_total] F32 (owner)
    var in_total: Int
    var off_x: Int
    var off_cap: Int
    var off_mv_main: Int
    var off_mv_nr: Int
    var off_fconsts: Int
    var off_rx: Int             # x cos | x sin (contiguous)
    var off_rc: Int             # cap cos | cap sin
    var off_ru: Int             # uni cos | uni sin
    # carved views
    var x_t: TArc               # [n_img, D] F32
    var cap: TArc               # [n_txt, D] F32
    var mv_main: List[ZImageModVecsDevice]
    var mv_nr: List[ZImageModVecsDevice]
    var ones: TArc              # [D]
    var zeros: TArc             # [D]
    var f_scale: TArc           # [D]
    var rope_x_cos: TArc
    var rope_x_sin: TArc
    var rope_cap_cos: TArc
    var rope_cap_sin: TArc
    var rope_uni_cos: TArc
    var rope_uni_sin: TArc
    # d_loss root (separate small buffer, written after the host loss)
    var d_patches: TArc         # [S, out_ch] F32 (cap rows stay zero)
    # output carriers
    var d_f_scale: TArc         # [D] F32
    var grad_a: List[TArc]      # lazy (warmup): bwd visit order
    var grad_b: List[TArc]
    var grad_indices: List[Int]
    var dx_carrier: List[TArc]  # lazy: 2 x [S, D] ping-pong
    # forward persistents
    var x_chain: List[TArc]     # (num_main+1) x [S, D] F32
    var nr_ping: List[TArc]     # 2 x [n_img, D] F32
    var cr_ping: List[TArc]     # 2 x [n_txt, D] F32
    var final_bias: TArc        # cloned once per bucket

    def __init__(
        out self,
        n_img: Int, n_txt: Int, s_total: Int, d_model: Int, out_ch: Int,
        num_nr: Int, num_main: Int, rope_half: Int, n_heads: Int,
        var in_slab: TArc, in_total: Int,
        off_x: Int, off_cap: Int, off_mv_main: Int, off_mv_nr: Int,
        off_fconsts: Int, off_rx: Int, off_rc: Int, off_ru: Int,
        var x_t: TArc, var cap: TArc,
        var mv_main: List[ZImageModVecsDevice],
        var mv_nr: List[ZImageModVecsDevice],
        var ones: TArc, var zeros: TArc, var f_scale: TArc,
        var rope_x_cos: TArc, var rope_x_sin: TArc,
        var rope_cap_cos: TArc, var rope_cap_sin: TArc,
        var rope_uni_cos: TArc, var rope_uni_sin: TArc,
        var d_patches: TArc, var d_f_scale: TArc,
        var x_chain: List[TArc], var nr_ping: List[TArc],
        var cr_ping: List[TArc], var final_bias: TArc,
    ):
        self.n_img = n_img
        self.n_txt = n_txt
        self.s_total = s_total
        self.d_model = d_model
        self.out_ch = out_ch
        self.num_nr = num_nr
        self.num_main = num_main
        self.rope_half = rope_half
        self.n_heads = n_heads
        self.in_slab = in_slab^
        self.in_total = in_total
        self.off_x = off_x
        self.off_cap = off_cap
        self.off_mv_main = off_mv_main
        self.off_mv_nr = off_mv_nr
        self.off_fconsts = off_fconsts
        self.off_rx = off_rx
        self.off_rc = off_rc
        self.off_ru = off_ru
        self.x_t = x_t^
        self.cap = cap^
        self.mv_main = mv_main^
        self.mv_nr = mv_nr^
        self.ones = ones^
        self.zeros = zeros^
        self.f_scale = f_scale^
        self.rope_x_cos = rope_x_cos^
        self.rope_x_sin = rope_x_sin^
        self.rope_cap_cos = rope_cap_cos^
        self.rope_cap_sin = rope_cap_sin^
        self.rope_uni_cos = rope_uni_cos^
        self.rope_uni_sin = rope_uni_sin^
        self.d_patches = d_patches^
        self.d_f_scale = d_f_scale^
        self.grad_a = List[TArc]()
        self.grad_b = List[TArc]()
        self.grad_indices = List[Int]()
        self.dx_carrier = List[TArc]()
        self.x_chain = x_chain^
        self.nr_ping = nr_ping^
        self.cr_ping = cr_ping^
        self.final_bias = final_bias^


def _io_view(
    dev: DeviceBuffer[DType.uint8], off_elems: Int, var shape: List[Int]
) raises -> TArc:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return TArc(Tensor(
        dev.create_sub_buffer[DType.uint8](off_elems * 4, n * 4),
        shape^, STDtype.F32,
    ))


def _io_fresh(var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    return TArc(Tensor(buf^, shape^, STDtype.F32))


def zimage_step_io_init(
    n_img: Int, n_txt: Int, d_model: Int, out_ch: Int,
    num_nr: Int, num_main: Int, n_heads: Int, dh: Int,
    final_lin_b: Tensor, ctx: DeviceContext,
) raises -> ZImageStepIO:
    var s_total = n_img + n_txt
    var half = dh // 2
    var d = d_model
    # element offsets in the input slab (F32)
    var off_x = 0
    var off_cap = off_x + n_img * d
    var off_mv_main = off_cap + n_txt * d
    var off_mv_nr = off_mv_main + (num_main * 4 + 1) * d
    var off_fconsts = off_mv_nr + (num_nr * 4 + 1) * d
    var off_rx = off_fconsts + 3 * d
    var off_rc = off_rx + 2 * (n_img * n_heads * half)
    var off_ru = off_rc + 2 * (n_txt * n_heads * half)
    var in_total = off_ru + 2 * (s_total * n_heads * half)

    var dev = ctx.enqueue_create_buffer[DType.uint8](in_total * 4)

    var x_t = _io_view(dev, off_x, [n_img, d])
    var cap = _io_view(dev, off_cap, [n_txt, d])

    # modvec views mirror zimage_modvecs_all_to_device's slab layout exactly:
    # block-major scale_msa|gate_msa|scale_mlp|gate_mlp + shared zeros tail.
    var mv_main = List[ZImageModVecsDevice]()
    var mv_main_zeros = _io_view(dev, off_mv_main + num_main * 4 * d, [d])
    for i in range(num_main):
        var base = off_mv_main + i * 4 * d
        mv_main.append(ZImageModVecsDevice(
            _io_view(dev, base, [d]),
            _io_view(dev, base + d, [d]),
            _io_view(dev, base + 2 * d, [d]),
            _io_view(dev, base + 3 * d, [d]),
            mv_main_zeros.copy(),
        ))
    var mv_nr = List[ZImageModVecsDevice]()
    var mv_nr_zeros = _io_view(dev, off_mv_nr + num_nr * 4 * d, [d])
    for i in range(num_nr):
        var base = off_mv_nr + i * 4 * d
        mv_nr.append(ZImageModVecsDevice(
            _io_view(dev, base, [d]),
            _io_view(dev, base + d, [d]),
            _io_view(dev, base + 2 * d, [d]),
            _io_view(dev, base + 3 * d, [d]),
            mv_nr_zeros.copy(),
        ))

    var ones = _io_view(dev, off_fconsts, [d])
    var zeros = _io_view(dev, off_fconsts + d, [d])
    var f_scale = _io_view(dev, off_fconsts + 2 * d, [d])

    var rx_n = n_img * n_heads * half
    var rc_n = n_txt * n_heads * half
    var ru_n = s_total * n_heads * half
    var rope_x_cos = _io_view(dev, off_rx, [n_img * n_heads, half])
    var rope_x_sin = _io_view(dev, off_rx + rx_n, [n_img * n_heads, half])
    var rope_cap_cos = _io_view(dev, off_rc, [n_txt * n_heads, half])
    var rope_cap_sin = _io_view(dev, off_rc + rc_n, [n_txt * n_heads, half])
    var rope_uni_cos = _io_view(dev, off_ru, [s_total * n_heads, half])
    var rope_uni_sin = _io_view(dev, off_ru + ru_n, [s_total * n_heads, half])

    var in_slab = TArc(Tensor(dev^, [in_total], STDtype.F32))

    var d_patches = _io_fresh([s_total, out_ch], ctx)
    # cap-row tail of d_patches is CONSTANT zero — write it once here (the
    # per-step write only refreshes the n_img image rows; byte-identical to
    # _concat_img_cap(d_out, zeros)).
    var dp_host = ctx.enqueue_create_host_buffer[DType.uint8](s_total * out_ch * 4)
    var dp = dp_host.unsafe_ptr().bitcast[Float32]()
    for i in range(s_total * out_ch):
        dp[i] = Float32(0.0)
    ctx.enqueue_copy(dst_buf=d_patches[].buf, src_buf=dp_host)
    ctx.synchronize()

    var d_f_scale = _io_fresh([d], ctx)

    var x_chain = List[TArc]()
    for _ in range(num_main + 1):
        x_chain.append(_io_fresh([s_total, d], ctx))
    var nr_ping = List[TArc]()
    for _ in range(2):
        nr_ping.append(_io_fresh([n_img, d], ctx))
    var cr_ping = List[TArc]()
    for _ in range(2):
        cr_ping.append(_io_fresh([n_txt, d], ctx))

    var final_bias = TArc(final_lin_b.clone(ctx))

    return ZImageStepIO(
        n_img, n_txt, s_total, d, out_ch, num_nr, num_main, half, n_heads,
        in_slab^, in_total,
        off_x, off_cap, off_mv_main, off_mv_nr, off_fconsts, off_rx, off_rc,
        off_ru,
        x_t^, cap^, mv_main^, mv_nr^, ones^, zeros^, f_scale^,
        rope_x_cos^, rope_x_sin^, rope_cap_cos^, rope_cap_sin^,
        rope_uni_cos^, rope_uni_sin^,
        d_patches^, d_f_scale^,
        x_chain^, nr_ping^, cr_ping^, final_bias^,
    )


def _io_pack_list(host: HostBuffer[DType.uint8], off: Int, vals: List[Float32]):
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(len(vals)):
        fp[off + i] = vals[i]


def zimage_step_io_write_inputs(
    io: ZImageStepIO,
    x_t: List[Float32], cap_seq: List[Float32],
    nr_mod: List[ZImageModVecs], main_mod: List[ZImageModVecs],
    f_scale: List[Float32],
    rx: ZImageRopeHost, rc: ZImageRopeHost, ru: ZImageRopeHost,
    ctx: DeviceContext,
) raises:
    """Pack every per-step input into ONE pinned staging buffer (layouts
    byte-identical to the _v3 builders) and issue ONE H2D copy into the fixed
    input slab. Sync before return (staging dies at scope end; outside the
    captured region, so the sync is legal)."""
    if len(x_t) != io.n_img * io.d_model or len(cap_seq) != io.n_txt * io.d_model:
        raise Error("zimage_step_io_write_inputs: x/cap length mismatch")
    if len(nr_mod) != io.num_nr or len(main_mod) != io.num_main:
        raise Error("zimage_step_io_write_inputs: modvec count mismatch")
    var d = io.d_model
    var host = ctx.enqueue_create_host_buffer[DType.uint8](io.in_total * 4)
    var fp = host.unsafe_ptr().bitcast[Float32]()
    _io_pack_list(host, io.off_x, x_t)
    _io_pack_list(host, io.off_cap, cap_seq)
    for i in range(io.num_main):
        var base = io.off_mv_main + i * 4 * d
        _io_pack_list(host, base, main_mod[i].scale_msa)
        _io_pack_list(host, base + d, main_mod[i].gate_msa)
        _io_pack_list(host, base + 2 * d, main_mod[i].scale_mlp)
        _io_pack_list(host, base + 3 * d, main_mod[i].gate_mlp)
    for c in range(d):
        fp[io.off_mv_main + io.num_main * 4 * d + c] = Float32(0.0)
    for i in range(io.num_nr):
        var base = io.off_mv_nr + i * 4 * d
        _io_pack_list(host, base, nr_mod[i].scale_msa)
        _io_pack_list(host, base + d, nr_mod[i].gate_msa)
        _io_pack_list(host, base + 2 * d, nr_mod[i].scale_mlp)
        _io_pack_list(host, base + 3 * d, nr_mod[i].gate_mlp)
    for c in range(d):
        fp[io.off_mv_nr + io.num_nr * 4 * d + c] = Float32(0.0)
    for c in range(d):
        fp[io.off_fconsts + c] = Float32(1.0)
    for c in range(d):
        fp[io.off_fconsts + d + c] = Float32(0.0)
    _io_pack_list(host, io.off_fconsts + 2 * d, f_scale)
    var rx_n = io.n_img * io.n_heads * io.rope_half
    var rc_n = io.n_txt * io.n_heads * io.rope_half
    var ru_n = io.s_total * io.n_heads * io.rope_half
    if len(rx.cos_vals) != rx_n or len(rc.cos_vals) != rc_n or len(ru.cos_vals) != ru_n:
        raise Error("zimage_step_io_write_inputs: rope length mismatch")
    _io_pack_list(host, io.off_rx, rx.cos_vals)
    _io_pack_list(host, io.off_rx + rx_n, rx.sin_vals)
    _io_pack_list(host, io.off_rc, rc.cos_vals)
    _io_pack_list(host, io.off_rc + rc_n, rc.sin_vals)
    _io_pack_list(host, io.off_ru, ru.cos_vals)
    _io_pack_list(host, io.off_ru + ru_n, ru.sin_vals)
    ctx.enqueue_copy(dst_buf=io.in_slab[].buf, src_buf=host)
    ctx.synchronize()


def zimage_step_io_write_d_patches(
    io: ZImageStepIO, d_loss: List[Float32], ctx: DeviceContext
) raises:
    """Write the image-row d_loss into the fixed d_patches root (cap rows
    were zeroed once at init — _concat_img_cap(d_out, zeros) bytes)."""
    var n = io.n_img * io.out_ch
    if len(d_loss) != n:
        raise Error("zimage_step_io_write_d_patches: d_loss length mismatch")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n * 4)
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(n):
        fp[i] = d_loss[i]
    var dst = io.d_patches[].buf.create_sub_buffer[DType.uint8](0, n * 4)
    ctx.enqueue_copy(dst_buf=dst, src_buf=host)
    ctx.synchronize()


def _io_copy_out_persistent(
    mut lst: List[TArc], pos: Int, src: Tensor, ctx: DeviceContext
) raises:
    """d2d copy `src` (slab-resident) into the persistent buffer at `pos`,
    lazily creating it on the warmup step (creation order is deterministic —
    the same fixed pointer serves capture + every replay)."""
    if pos == len(lst):
        var buf = ctx.enqueue_create_buffer[DType.uint8](src.nbytes())
        lst.append(TArc(Tensor(buf^, src.shape(), src.dtype())))
    elif pos > len(lst):
        raise Error("zimage v5: persistent copy-out out of order")
    ctx.enqueue_copy(dst_buf=lst[pos][].buf, src_buf=src.buf)


def zimage_stack_lora_forward_main_device_v5[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    lora: ZImageLoraDeviceSet,
    final_lin_w: Tensor,
    io: ZImageStepIO,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
    mut fslab: StepSlab,
) raises -> ZImageStackForwardV5:
    """P5 capture-compatible forward: the _v3 op chain on the fwd StepSlab
    with fixed-address inputs (io views) — NO from_host, NO to_host, NO sync
    anywhere inside (C9). Per-block mark/rewind; survivors (block in/out x)
    are d2d-copied into persistent buffers (values unchanged). The caller
    resets `fslab` before calling (or before the graph launch on replay)."""
    var num_nr = len(nr_blocks)
    var num_cr = len(cr_blocks)
    var num_main = len(main_blocks)

    # ── noise refiner (frozen, modulated) on the fixed x_t ──
    var xs = io.x_t.copy()
    for i in range(num_nr):
        var m = fslab.mark()
        var out = zimage_block_forward_device_moddev_slab[H, Dh, N_IMG](
            xs.copy(), nr_blocks[i], io.mv_nr[i],
            io.rope_x_cos[], io.rope_x_sin[], D, F, eps, ctx, fslab,
        )
        ctx.enqueue_copy(dst_buf=io.nr_ping[i % 2][].buf, src_buf=out[].buf)
        xs = io.nr_ping[i % 2].copy()
        fslab.rewind(m)

    # ── context refiner (frozen, unmodulated) on the fixed cap ──
    var cs = io.cap.copy()
    for i in range(num_cr):
        var m = fslab.mark()
        var out = zimage_refiner_forward_device_slab[H, Dh, N_TXT](
            cs.copy(), cr_blocks[i], io.rope_cap_cos[], io.rope_cap_sin[],
            D, F, eps, ctx, fslab,
        )
        ctx.enqueue_copy(dst_buf=io.cr_ping[i % 2][].buf, src_buf=out[].buf)
        cs = io.cr_ping[i % 2].copy()
        fslab.rewind(m)

    # ── device concat img‖cap into the persistent chain head (dim-0 byte
    # concat — the exact bytes ops.tensor_algebra.concat produces) ──
    var img_bytes = N_IMG * D * 4
    var cap_bytes = N_TXT * D * 4
    var dst_img = io.x_chain[0][].buf.create_sub_buffer[DType.uint8](0, img_bytes)
    ctx.enqueue_copy(dst_buf=dst_img, src_buf=xs[].buf)
    var dst_cap = io.x_chain[0][].buf.create_sub_buffer[DType.uint8](
        img_bytes, cap_bytes
    )
    ctx.enqueue_copy(dst_buf=dst_cap, src_buf=cs[].buf)

    # ── main layers (LoRA) — chain[i] in, chain[i+1] out ──
    var main_x_in = List[TArc]()
    for i in range(num_main):
        var bl = _block_lora_dev_for(lora, lora.main_base() + i)
        main_x_in.append(io.x_chain[i].copy())
        var m = fslab.mark()
        var out = zimage_block_lora_forward_device_only_slab[1, H, Dh, S](
            io.x_chain[i].copy(), main_blocks[i], io.mv_main[i], bl,
            io.rope_uni_cos[], io.rope_uni_sin[], D, F, eps, ctx, fslab,
        )
        ctx.enqueue_copy(dst_buf=io.x_chain[i + 1][].buf, src_buf=out[].buf)
        fslab.rewind(m)

    # ── final layer (slab; deterministic addresses after the last rewind) ──
    var ln_t = layer_norm_slab(
        io.x_chain[num_main][], io.ones[], io.zeros[], final_eps, ctx, fslab,
    )
    var x_out_t = vec_modulate_slab(ln_t, io.f_scale[], io.zeros[], ctx, fslab)
    var fb = Optional[Tensor](Tensor(
        io.final_bias[].buf.copy(), io.final_bias[].shape(),
        io.final_bias[].dtype(),
    ))
    var patches = linear_slab(x_out_t, final_lin_w, fb, ctx, fslab)

    return ZImageStackForwardV5(
        TArc(patches^), main_x_in^, io.x_chain[num_main].copy(), TArc(ln_t^),
    )


def zimage_stack_lora_backward_main_device_v5[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    main_blocks: List[ZImageBlockWeights],
    lora: ZImageLoraDeviceSet,
    final_lin_w: Tensor,
    saved: ZImageStackForwardV5,
    mut io: ZImageStepIO,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises:
    """P5 capture-compatible backward: the _v4 chain with (a) the final-layer
    prologue slab-routed reading the fixed io.d_patches root, (b) d_f_scale
    reduced in-graph into the fixed io.d_f_scale buffer (read back AFTER the
    launch with the grads batch), (c) the per-block d_x + 14 adapter-grad
    copy-outs landing in the persistent grads region (lazy-created on the
    warmup step). NO from_host/to_host/sync inside (C9). The caller resets
    `slab` before calling. Grads are read via zimage_step_io_read_grads."""
    var num_main = len(main_blocks)

    # ── final-layer prologue (slab; values identical to the _v4 prologue) ──
    var final_dx = linear_backward_dx_slab(
        io.d_patches[], final_lin_w, S, D, out_ch, ctx, slab,
    )
    var mbf = modulate_backward_slab(
        final_dx, saved.ln_x[], io.f_scale[], ctx, slab,
    )
    ctx.enqueue_copy(dst_buf=io.d_f_scale[].buf, src_buf=mbf.d_scale.buf)
    var d_x_t = layer_norm_backward_dx_slab(
        mbf.d_x, saved.x_final[], io.ones[], final_eps, ctx, slab,
    )
    var d_x = TArc(d_x_t^)   # slab-resident, BELOW the per-block marks

    var bi = num_main - 1
    var visit = 0
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var m = slab.mark()
        var bg = zimage_block_lora_graph_backward_slab[H, Dh, S](
            d_x[], main_blocks[bi], io.mv_main[bi], bl,
            saved.main_x_in[bi], io.rope_uni_cos[], io.rope_uni_sin[],
            D, F, eps, ctx, slab,
        )
        _io_copy_out_persistent(io.dx_carrier, visit % 2, bg.d_x[], ctx)
        d_x = io.dx_carrier[visit % 2].copy()
        var base_idx = (lora.main_base() + bi) * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            var pos = visit * ZIMAGE_SLOTS + s
            _io_copy_out_persistent(io.grad_a, pos, bg.d_a[s][], ctx)
            _io_copy_out_persistent(io.grad_b, pos, bg.d_b[s][], ctx)
            if pos == len(io.grad_indices):
                io.grad_indices.append(base_idx + s)
        slab.rewind(m)
        visit += 1
        bi -= 1


def zimage_step_io_read_grads(
    io: ZImageStepIO, num_blocks_total: Int, ctx: DeviceContext
) raises -> ZImageLoraGrads:
    """AFTER the bwd graph launch: batch the persistent grads region to host
    (the SAME _zimage_tensor_grads_to_host path _v4 uses — identical staging,
    casts and flat scatter) + read the fixed d_f_scale. Returns the _v4-shaped
    ZImageLoraGrads (clip/AdamW/save downstream unchanged)."""
    var host_grads = _zimage_tensor_grads_to_host(
        io.grad_indices, io.grad_a, io.grad_b,
        num_blocks_total * ZIMAGE_SLOTS, ctx,
    )
    var nonfinite = 0
    for i in range(len(io.grad_indices)):
        var idx = io.grad_indices[i]
        nonfinite += _nonfinite(host_grads.d_a[idx]) + _nonfinite(host_grads.d_b[idx])
    var d_f_scale = io.d_f_scale[].to_host(ctx)
    var empty_x = List[Float32]()
    var empty_cap = List[Float32]()
    var empty_nr = List[List[Float32]]()
    var empty_main = List[List[Float32]]()
    var d_final_lin = List[Float32]()
    return ZImageLoraGrads(
        host_grads.d_a.copy(), host_grads.d_b.copy(),
        empty_x^, empty_cap^,
        empty_nr^, empty_main^,
        d_f_scale^, d_final_lin^,
        nonfinite,
    )


# ══════════════════════════════════════════════════════════════════════════════
# T2.C FULL-RANK FINETUNE (2026-06-11) — stack backward producing base-weight
# grads for every MAIN-block slot projection (30 blocks x 7 slots = 5.31B
# params). Same recompute-then-backprop conductor as _v2; per block the LoRA
# backward is replaced by zimage_block_backward_device_tensors_fullft and the
# 7 bf16 d_W tensors are D2H'd into ONE pinned host buffer per block, then the
# device grads are freed before the next block (peak device overhead = one
# block's d_W set ~354 MB — the full 10.6 GB grad set never lives on device).
# The recompute forward runs with the ZERO LoRA device set (scale==0 -> the
# apply is skipped -> base forward). ADDITIVE (C13): no existing path touched.
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageFullFTGrads(Movable):
    """Host bf16 base-weight grads for the MAIN blocks. bufs_rev[j] is the
    pinned buffer for block (num_main-1-j) (the backward's reverse order);
    use buf_index(bi). Within a buffer the 7 slots are contiguous in slot
    order Q,K,V,O,W1,W3,W2 at slot_offsets (bytes) / slot_numels (elems)."""

    var bufs_rev: List[HostBuffer[DType.uint8]]
    var slot_offsets: List[Int]
    var slot_numels: List[Int]
    var num_main: Int

    def __init__(
        out self,
        var bufs_rev: List[HostBuffer[DType.uint8]],
        var slot_offsets: List[Int], var slot_numels: List[Int],
        num_main: Int,
    ):
        self.bufs_rev = bufs_rev^
        self.slot_offsets = slot_offsets^
        self.slot_numels = slot_numels^
        self.num_main = num_main

    def buf_index(self, block_idx: Int) -> Int:
        return self.num_main - 1 - block_idx


def zimage_fullft_slot_numels(D: Int, F: Int) -> List[Int]:
    """Element counts per slot, slot order Q,K,V,O,W1,W3,W2."""
    var out = List[Int]()
    out.append(D * D)   # Q
    out.append(D * D)   # K
    out.append(D * D)   # V
    out.append(D * D)   # O
    out.append(F * D)   # W1
    out.append(F * D)   # W3
    out.append(D * F)   # W2
    return out^


def zimage_stack_lora_backward_main_device_fullft[
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
) raises -> ZImageFullFTGrads:
    var num_main = len(main_blocks)

    # ── final-layer prologue: byte-identical to _v2 ──────────────────────────
    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var mbf = modulate_backward(
        final_dx, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_x_t = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = TArc(d_x_t^)

    # slot layout (bf16 bytes) shared by every block buffer.
    var slot_numels = zimage_fullft_slot_numels(D, F)
    var slot_offsets = List[Int]()
    var cursor = 0
    for s in range(ZIMAGE_SLOTS):
        slot_offsets.append(cursor)
        cursor += slot_numels[s] * 2
    var block_bytes = cursor

    var bufs_rev = List[HostBuffer[DType.uint8]]()
    var bi = num_main - 1
    while bi >= 0:
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var refwd = zimage_block_lora_forward_device_tensor_batch[1, H, Dh, S](
            saved.main_x_in[bi].copy(), main_blocks[bi], main_mod_dev[bi], bl,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        var bg = zimage_block_backward_device_tensors_fullft[H, Dh, S](
            d_x[], main_blocks[bi], main_mod_dev[bi], refwd.saved,
            uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()

        var host = ctx.enqueue_create_host_buffer[DType.uint8](block_bytes)
        for s in range(ZIMAGE_SLOTS):
            if bg.d_w[s][].dtype() != STDtype.BF16:
                raise Error("fullft backward: d_w dtype must be BF16")
            var nbytes = slot_numels[s] * 2
            if bg.d_w[s][].nbytes() != nbytes:
                raise Error("fullft backward: d_w slot byte-size mismatch")
            var dst = host.create_sub_buffer[DType.uint8](slot_offsets[s], nbytes)
            ctx.enqueue_copy(dst_buf=dst, src_buf=bg.d_w[s][].buf)
        # device grads (bg) must stay alive until the copies complete.
        ctx.synchronize()
        bufs_rev.append(host)
        bi -= 1

    return ZImageFullFTGrads(bufs_rev^, slot_offsets^, slot_numels^, num_main)


# ══════════════════════════════════════════════════════════════════════════════
# T2.E ControlNet arms (default-off; reached ONLY from the controlnet driver in
# train_zimage_real.mojo when controlnet_layers > 0 — C13: every function above
# is untouched).
#
# Reference semantics (diffusers transformer_z_image.py:1032):
#   unified = layer(unified)
#   if layer_idx in controlnet_block_samples: unified = unified + samples[idx]
# where samples[idx] = hint_idx * conditioning_scale (the scale is applied by
# the ZImageControlNetModel output dict; here the caller passes UNSCALED hints
# + the scale and the multiply happens at the injection site — same math).
#
# Backward: the injection is an ADD, so the grad arriving at the boundary
# ABOVE main layer `p` IS the grad of (layer_p_out + scale*hint): d_hint =
# scale * d_boundary, d_layer_out = d_boundary (the step smoke's hand math,
# zimage_controlnet_step_smoke.mojo:354-371).
# ══════════════════════════════════════════════════════════════════════════════


def zimage_stack_lora_forward_main_from_unified_cn[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    xs: List[Float32], cs: List[Float32],
    main_blocks: List[ZImageBlockWeights],
    main_mod_dev: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    hints: List[List[Float32]], places: List[Int], cond_scale: Float32,
    f_scale: List[Float32],
    final_lin_w: Tensor, final_lin_b: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageStackForward:
    """Main-layer loop + final layer of the _v2 forward, taking the ALREADY
    NR/CR-refined image (`xs`) and caption (`cs`) sequences (the controlnet
    driver computes those first because the control stack needs the unified
    INPUT), with post-layer hint injection at `places`. ZImageStackForward
    contract matches _v2/_v3: main_x_in saves each block's (post-injection)
    input — the recompute backward below replays the same chain."""
    if len(hints) != len(places):
        raise Error("cn forward: hints/places length mismatch")
    var num_main = len(main_blocks)
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
        for pi in range(len(places)):
            if places[pi] == i:
                if len(hints[pi]) != S * D:
                    raise Error("cn forward: hint size != S*D")
                var hs = List[Float32]()
                for j in range(S * D):
                    hs.append(cond_scale * hints[pi][j])
                x_arc = TArc(add(x_arc[], _t(hs^, [S, D], ctx), ctx))

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

    var nr_x_in = List[TArc]()
    var cr_x_in = List[TArc]()
    return ZImageStackForward(
        out^, xs.copy(), cs.copy(),
        nr_x_in^, cr_x_in^, main_x_in^,
        x_arc.copy(), TArc(ln_t^),
    )


def zimage_stack_lora_backward_main_device_cn[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],
    main_blocks: List[ZImageBlockWeights],
    main_mod_dev: List[ZImageModVecsDevice],
    lora: ZImageLoraDeviceSet,
    places: List[Int], cond_scale: Float32,
    f_scale: List[Float32],
    final_lin_w: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    saved: ZImageStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32, final_eps: Float32,
    ctx: DeviceContext,
) raises -> List[List[Float32]]:
    """ControlNet-training backward through the FROZEN main stack (the v2
    graph-engine per-block backward, the _v3 arm): the only outputs are the
    per-place d_hints (ALREADY cond_scale-scaled — the control-stack backward
    contract, controlnet_block.mojo:389). Base + LoRA grads are computed by
    the per-block engine and DISCARDED (base frozen; the controlnet driver
    runs the ZERO LoRA set so the forward is the base forward)."""
    var num_main = len(main_blocks)

    var d_patches = _concat_img_cap(d_out, _zeros(N_TXT * out_ch))
    var final_dx = linear_backward_dx(
        _t(d_patches, [S, out_ch], ctx), final_lin_w, S, D, out_ch, ctx,
    )
    var mbf = modulate_backward(
        final_dx, saved.ln_x[], _t(f_scale.copy(), [D], ctx), ctx,
    )
    var d_x_t = layer_norm_backward_dx(
        mbf.d_x, saved.x_final[], _t(_ones(D), [D], ctx), final_eps, ctx,
    )
    var d_x = TArc(d_x_t^)

    var d_hints = List[List[Float32]]()
    for _ in range(len(places)):
        d_hints.append(List[Float32]())

    var bi = num_main - 1
    while bi >= 0:
        # entering iteration bi, d_x = grad w.r.t. the boundary ABOVE layer bi
        # (= layer bi's post-injection output) — capture the hint grad here.
        for pi in range(len(places)):
            if places[pi] == bi:
                var dh = d_x[].to_host(ctx)
                for j in range(len(dh)):
                    dh[j] = cond_scale * dh[j]
                d_hints[pi] = dh^
        var bl = _block_lora_dev_for(lora, lora.main_base() + bi)
        var bg = zimage_block_lora_graph_backward[H, Dh, S](
            d_x[], main_blocks[bi], main_mod_dev[bi], bl,
            saved.main_x_in[bi], uni_cos, uni_sin, D, F, eps, ctx,
        )
        d_x = bg.d_x.copy()
        bi -= 1

    for pi in range(len(places)):
        if len(d_hints[pi]) != S * D:
            raise Error("cn backward: missing d_hint (place outside main depth?)")
    return d_hints^
