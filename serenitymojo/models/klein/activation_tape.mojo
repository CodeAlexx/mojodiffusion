# Dtype-preserving Klein LoRA activation tape offload.
#
# This is the narrow runtime bridge needed before a bounded
# CPU_OFFLOADED/checkpoint backward replay can exist. It offloads only the
# boundaries consumed by the current LoRA backward path:
#   dbl_img_in, dbl_txt_in, sgl_x_in, img_out, ln_img_out.
# The input-projection activations are intentionally not carried here because
# current LoRA backward does not consume them.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.tensor import Tensor
from serenitymojo.training.checkpoint import (
    HostOffload,
    offload_to_host,
    restore_to_device,
    offload_to_host_fast,
    restore_to_device_fast,
)

from serenitymojo.models.klein.double_block import DoubleBlockSaved, StreamSaved
from serenitymojo.models.klein.single_block import SingleBlockSaved
from serenitymojo.models.klein.klein_stack import KleinStackForward


comptime TArc = ArcPointer[Tensor]


# ── per-tensor offload helpers (memcpy fast path) ────────────────────────────
def _off(t: TArc, ctx: DeviceContext) raises -> HostOffload:
    return offload_to_host_fast(t[], ctx)


def _off_opt(t: Optional[TArc], ctx: DeviceContext) raises -> Optional[HostOffload]:
    if t:
        return Optional[HostOffload](offload_to_host_fast(t.value()[], ctx))
    return None


def _res(o: HostOffload, ctx: DeviceContext) raises -> TArc:
    return TArc(restore_to_device_fast(o, ctx))


def _res_opt(o: Optional[HostOffload], ctx: DeviceContext) raises -> Optional[TArc]:
    if o:
        return Optional[TArc](TArc(restore_to_device_fast(o.value(), ctx)))
    return None


def _bytes_of(o: HostOffload) -> Int:
    return len(o.host)


def _bytes_of_opt(o: Optional[HostOffload]) -> Int:
    if o:
        return len(o.value().host)
    return 0


# ── host-offloaded SingleBlockSaved (full activation set, one single block) ───
struct SingleBlockSavedOffload(Copyable, Movable):
    var x: HostOffload
    var ln: HostOffload
    var norm: HostOffload
    var q_pre: HostOffload
    var k_pre: HostOffload
    var q_rms: HostOffload
    var k_rms: HostOffload
    var v: HostOffload
    var q_rope: HostOffload
    var k_rope: HostOffload
    var att_flat: HostOffload
    var mlp_gate: HostOffload
    var mlp_up: HostOffload
    var mlp: HostOffload
    var out_in: HostOffload
    var flash_q: Optional[HostOffload]
    var flash_k: Optional[HostOffload]
    var flash_v: Optional[HostOffload]
    var flash_o: Optional[HostOffload]
    var flash_stats: Optional[HostOffload]

    def __init__(
        out self,
        var x: HostOffload, var ln: HostOffload, var norm: HostOffload,
        var q_pre: HostOffload, var k_pre: HostOffload,
        var q_rms: HostOffload, var k_rms: HostOffload, var v: HostOffload,
        var q_rope: HostOffload, var k_rope: HostOffload,
        var att_flat: HostOffload,
        var mlp_gate: HostOffload, var mlp_up: HostOffload, var mlp: HostOffload,
        var out_in: HostOffload,
        var flash_q: Optional[HostOffload], var flash_k: Optional[HostOffload],
        var flash_v: Optional[HostOffload], var flash_o: Optional[HostOffload],
        var flash_stats: Optional[HostOffload],
    ):
        self.x = x^
        self.ln = ln^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.mlp_gate = mlp_gate^
        self.mlp_up = mlp_up^
        self.mlp = mlp^
        self.out_in = out_in^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^

    def host_bytes(self) -> Int:
        var t = _bytes_of(self.x) + _bytes_of(self.ln) + _bytes_of(self.norm)
        t += _bytes_of(self.q_pre) + _bytes_of(self.k_pre)
        t += _bytes_of(self.q_rms) + _bytes_of(self.k_rms) + _bytes_of(self.v)
        t += _bytes_of(self.q_rope) + _bytes_of(self.k_rope)
        t += _bytes_of(self.att_flat)
        t += _bytes_of(self.mlp_gate) + _bytes_of(self.mlp_up) + _bytes_of(self.mlp)
        t += _bytes_of(self.out_in)
        t += _bytes_of_opt(self.flash_q) + _bytes_of_opt(self.flash_k)
        t += _bytes_of_opt(self.flash_v) + _bytes_of_opt(self.flash_o)
        t += _bytes_of_opt(self.flash_stats)
        return t


def offload_single_block_saved(
    s: SingleBlockSaved, ctx: DeviceContext
) raises -> SingleBlockSavedOffload:
    return SingleBlockSavedOffload(
        _off(s.x, ctx), _off(s.ln, ctx), _off(s.norm, ctx),
        _off(s.q_pre, ctx), _off(s.k_pre, ctx),
        _off(s.q_rms, ctx), _off(s.k_rms, ctx), _off(s.v, ctx),
        _off(s.q_rope, ctx), _off(s.k_rope, ctx),
        _off(s.att_flat, ctx),
        _off(s.mlp_gate, ctx), _off(s.mlp_up, ctx), _off(s.mlp, ctx),
        _off(s.out_in, ctx),
        _off_opt(s.flash_q, ctx), _off_opt(s.flash_k, ctx),
        _off_opt(s.flash_v, ctx), _off_opt(s.flash_o, ctx),
        _off_opt(s.flash_stats, ctx),
    )


def restore_single_block_saved(
    o: SingleBlockSavedOffload, ctx: DeviceContext
) raises -> SingleBlockSaved:
    return SingleBlockSaved(
        _res(o.x, ctx), _res(o.ln, ctx), _res(o.norm, ctx),
        _res(o.q_pre, ctx), _res(o.k_pre, ctx),
        _res(o.q_rms, ctx), _res(o.k_rms, ctx), _res(o.v, ctx),
        _res(o.q_rope, ctx), _res(o.k_rope, ctx),
        _res(o.att_flat, ctx),
        _res(o.mlp_gate, ctx), _res(o.mlp_up, ctx), _res(o.mlp, ctx),
        _res(o.out_in, ctx),
        _res_opt(o.flash_q, ctx), _res_opt(o.flash_k, ctx),
        _res_opt(o.flash_v, ctx), _res_opt(o.flash_o, ctx),
        _res_opt(o.flash_stats, ctx),
    )


# ── host-offloaded StreamSaved (one double-block stream) ─────────────────────
struct StreamSavedOffload(Copyable, Movable):
    var x: HostOffload
    var ln1: HostOffload
    var norm: HostOffload
    var q_pre: HostOffload
    var k_pre: HostOffload
    var q_rms: HostOffload
    var k_rms: HostOffload
    var v: HostOffload
    var att: HostOffload
    var attn_res: HostOffload
    var ln2: HostOffload
    var mlp_in: HostOffload
    var gu: HostOffload
    var gate: HostOffload
    var up: HostOffload
    var act: HostOffload

    def __init__(
        out self,
        var x: HostOffload, var ln1: HostOffload, var norm: HostOffload,
        var q_pre: HostOffload, var k_pre: HostOffload,
        var q_rms: HostOffload, var k_rms: HostOffload, var v: HostOffload,
        var att: HostOffload, var attn_res: HostOffload,
        var ln2: HostOffload, var mlp_in: HostOffload,
        var gu: HostOffload, var gate: HostOffload, var up: HostOffload,
        var act: HostOffload,
    ):
        self.x = x^
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.att = att^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.gu = gu^
        self.gate = gate^
        self.up = up^
        self.act = act^

    def host_bytes(self) -> Int:
        var t = _bytes_of(self.x) + _bytes_of(self.ln1) + _bytes_of(self.norm)
        t += _bytes_of(self.q_pre) + _bytes_of(self.k_pre)
        t += _bytes_of(self.q_rms) + _bytes_of(self.k_rms) + _bytes_of(self.v)
        t += _bytes_of(self.att) + _bytes_of(self.attn_res)
        t += _bytes_of(self.ln2) + _bytes_of(self.mlp_in)
        t += _bytes_of(self.gu) + _bytes_of(self.gate) + _bytes_of(self.up)
        t += _bytes_of(self.act)
        return t


def _offload_stream_saved(
    s: StreamSaved, ctx: DeviceContext
) raises -> StreamSavedOffload:
    return StreamSavedOffload(
        _off(s.x, ctx), _off(s.ln1, ctx), _off(s.norm, ctx),
        _off(s.q_pre, ctx), _off(s.k_pre, ctx),
        _off(s.q_rms, ctx), _off(s.k_rms, ctx), _off(s.v, ctx),
        _off(s.att, ctx), _off(s.attn_res, ctx),
        _off(s.ln2, ctx), _off(s.mlp_in, ctx),
        _off(s.gu, ctx), _off(s.gate, ctx), _off(s.up, ctx),
        _off(s.act, ctx),
    )


def _restore_stream_saved(
    o: StreamSavedOffload, ctx: DeviceContext
) raises -> StreamSaved:
    return StreamSaved(
        _res(o.x, ctx), _res(o.ln1, ctx), _res(o.norm, ctx),
        _res(o.q_pre, ctx), _res(o.k_pre, ctx),
        _res(o.q_rms, ctx), _res(o.k_rms, ctx), _res(o.v, ctx),
        _res(o.att, ctx), _res(o.attn_res, ctx),
        _res(o.ln2, ctx), _res(o.mlp_in, ctx),
        _res(o.gu, ctx), _res(o.gate, ctx), _res(o.up, ctx),
        _res(o.act, ctx),
    )


# ── host-offloaded DoubleBlockSaved (one double block) ───────────────────────
struct DoubleBlockSavedOffload(Copyable, Movable):
    var img: StreamSavedOffload
    var txt: StreamSavedOffload
    var q_rope: HostOffload
    var k_rope: HostOffload
    var v_joint: HostOffload
    var flash_q: Optional[HostOffload]
    var flash_k: Optional[HostOffload]
    var flash_v: Optional[HostOffload]
    var flash_o: Optional[HostOffload]
    var flash_stats: Optional[HostOffload]

    def __init__(
        out self,
        var img: StreamSavedOffload, var txt: StreamSavedOffload,
        var q_rope: HostOffload, var k_rope: HostOffload, var v_joint: HostOffload,
        var flash_q: Optional[HostOffload], var flash_k: Optional[HostOffload],
        var flash_v: Optional[HostOffload], var flash_o: Optional[HostOffload],
        var flash_stats: Optional[HostOffload],
    ):
        self.img = img^
        self.txt = txt^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.v_joint = v_joint^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^

    def host_bytes(self) -> Int:
        var t = self.img.host_bytes() + self.txt.host_bytes()
        t += _bytes_of(self.q_rope) + _bytes_of(self.k_rope) + _bytes_of(self.v_joint)
        t += _bytes_of_opt(self.flash_q) + _bytes_of_opt(self.flash_k)
        t += _bytes_of_opt(self.flash_v) + _bytes_of_opt(self.flash_o)
        t += _bytes_of_opt(self.flash_stats)
        return t


def offload_double_block_saved(
    s: DoubleBlockSaved, ctx: DeviceContext
) raises -> DoubleBlockSavedOffload:
    return DoubleBlockSavedOffload(
        _offload_stream_saved(s.img, ctx), _offload_stream_saved(s.txt, ctx),
        _off(s.q_rope, ctx), _off(s.k_rope, ctx), _off(s.v_joint, ctx),
        _off_opt(s.flash_q, ctx), _off_opt(s.flash_k, ctx),
        _off_opt(s.flash_v, ctx), _off_opt(s.flash_o, ctx),
        _off_opt(s.flash_stats, ctx),
    )


def restore_double_block_saved(
    o: DoubleBlockSavedOffload, ctx: DeviceContext
) raises -> DoubleBlockSaved:
    return DoubleBlockSaved(
        _restore_stream_saved(o.img, ctx), _restore_stream_saved(o.txt, ctx),
        _res(o.q_rope, ctx), _res(o.k_rope, ctx), _res(o.v_joint, ctx),
        _res_opt(o.flash_q, ctx), _res_opt(o.flash_k, ctx),
        _res_opt(o.flash_v, ctx), _res_opt(o.flash_o, ctx),
        _res_opt(o.flash_stats, ctx),
    )


struct KleinStackLoraOffloadedTape(Copyable, Movable):
    var out: List[Float32]
    var dbl_img_in: List[HostOffload]
    var dbl_txt_in: List[HostOffload]
    var sgl_x_in: List[HostOffload]
    var img_out: HostOffload
    var ln_img_out: HostOffload
    # KLEIN_SAVE_ACTIVATIONS: full per-block activation sets, host-offloaded in
    # the forward and reloaded one block at a time in the backward (no recompute).
    # Empty when the flag is OFF (the classic recompute path is unchanged).
    var dbl_saved: List[DoubleBlockSavedOffload]
    var sgl_saved: List[SingleBlockSavedOffload]
    # KLEIN_DEVICE_TAPE (2026-07-11): DEVICE-RESIDENT block-input tape. The
    # block inputs are only ~838MB at 512px/b2, so instead of the host
    # round-trip (D2H + sync + memcpy out, then memcpy + H2D + sync back per
    # block — the measured ~0.6s/step residual stall) the forward parks TArc
    # REFS (refcount copy, zero bytes moved) and the backward reads them
    # directly. Empty lists = host mode (byte-path above, unchanged).
    var dbl_img_in_dev: List[TArc]
    var dbl_txt_in_dev: List[TArc]
    var sgl_x_in_dev: List[TArc]
    var img_out_dev: List[TArc]      # 0 or 1 entries
    var ln_img_out_dev: List[TArc]   # 0 or 1 entries

    def __init__(
        out self,
        var out_values: List[Float32],
        var dbl_img_in: List[HostOffload],
        var dbl_txt_in: List[HostOffload],
        var sgl_x_in: List[HostOffload],
        var img_out: HostOffload,
        var ln_img_out: HostOffload,
        var dbl_saved: List[DoubleBlockSavedOffload] = List[DoubleBlockSavedOffload](),
        var sgl_saved: List[SingleBlockSavedOffload] = List[SingleBlockSavedOffload](),
        var dbl_img_in_dev: List[TArc] = List[TArc](),
        var dbl_txt_in_dev: List[TArc] = List[TArc](),
        var sgl_x_in_dev: List[TArc] = List[TArc](),
        var img_out_dev: List[TArc] = List[TArc](),
        var ln_img_out_dev: List[TArc] = List[TArc](),
    ):
        self.out = out_values^
        self.dbl_img_in = dbl_img_in^
        self.dbl_txt_in = dbl_txt_in^
        self.sgl_x_in = sgl_x_in^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^
        self.dbl_saved = dbl_saved^
        self.sgl_saved = sgl_saved^
        self.dbl_img_in_dev = dbl_img_in_dev^
        self.dbl_txt_in_dev = dbl_txt_in_dev^
        self.sgl_x_in_dev = sgl_x_in_dev^
        self.img_out_dev = img_out_dev^
        self.ln_img_out_dev = ln_img_out_dev^

    def device_tape(self) -> Bool:
        return len(self.sgl_x_in_dev) > 0

    def has_saved_activations(self) -> Bool:
        return len(self.sgl_saved) > 0 or len(self.dbl_saved) > 0

    def saved_host_bytes(self) -> Int:
        var t = 0
        for i in range(len(self.dbl_saved)):
            t += self.dbl_saved[i].host_bytes()
        for i in range(len(self.sgl_saved)):
            t += self.sgl_saved[i].host_bytes()
        return t

    def num_double(self) -> Int:
        if len(self.dbl_img_in_dev) > 0:
            return len(self.dbl_img_in_dev)
        return len(self.dbl_img_in)

    def num_single(self) -> Int:
        if len(self.sgl_x_in_dev) > 0:
            return len(self.sgl_x_in_dev)
        return len(self.sgl_x_in)

    def total_host_bytes(self) -> Int:
        var total = len(self.img_out.host) + len(self.ln_img_out.host)
        for i in range(len(self.dbl_img_in)):
            total += len(self.dbl_img_in[i].host)
        for i in range(len(self.dbl_txt_in)):
            total += len(self.dbl_txt_in[i].host)
        for i in range(len(self.sgl_x_in)):
            total += len(self.sgl_x_in[i].host)
        return total

    def all_storage_dtype(self, dtype: STDtype) -> Bool:
        if self.img_out.dtype != dtype or self.ln_img_out.dtype != dtype:
            return False
        for i in range(len(self.dbl_img_in)):
            if self.dbl_img_in[i].dtype != dtype:
                return False
        for i in range(len(self.dbl_txt_in)):
            if self.dbl_txt_in[i].dtype != dtype:
                return False
        for i in range(len(self.sgl_x_in)):
            if self.sgl_x_in[i].dtype != dtype:
                return False
        return True


def offload_klein_stack_lora_backward_tape(
    saved: KleinStackForward, ctx: DeviceContext
) raises -> KleinStackLoraOffloadedTape:
    var dbl_img_in = List[HostOffload]()
    var dbl_txt_in = List[HostOffload]()
    var sgl_x_in = List[HostOffload]()

    for i in range(len(saved.dbl_img_in)):
        var off_img = offload_to_host(saved.dbl_img_in[i][], ctx)
        dbl_img_in.append(off_img^)
        var off_txt = offload_to_host(saved.dbl_txt_in[i][], ctx)
        dbl_txt_in.append(off_txt^)

    for i in range(len(saved.sgl_x_in)):
        var off_x = offload_to_host(saved.sgl_x_in[i][], ctx)
        sgl_x_in.append(off_x^)

    var img_out = offload_to_host(saved.img_out[], ctx)
    var ln_img_out = offload_to_host(saved.ln_img_out[], ctx)

    return KleinStackLoraOffloadedTape(
        saved.out.copy(),
        dbl_img_in^,
        dbl_txt_in^,
        sgl_x_in^,
        img_out^,
        ln_img_out^,
    )


def _unused_activation_arc(dtype: STDtype, ctx: DeviceContext) raises -> TArc:
    var shape = List[Int]()
    shape.append(1)
    var t = zeros_device(shape^, dtype, ctx)
    return TArc(t^)


def restore_klein_stack_lora_backward_tape(
    tape: KleinStackLoraOffloadedTape, ctx: DeviceContext
) raises -> KleinStackForward:
    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var sgl_x_in = List[TArc]()

    for i in range(len(tape.dbl_img_in)):
        var img_t = restore_to_device(tape.dbl_img_in[i], ctx)
        dbl_img_in.append(TArc(img_t^))
        var txt_t = restore_to_device(tape.dbl_txt_in[i], ctx)
        dbl_txt_in.append(TArc(txt_t^))

    for i in range(len(tape.sgl_x_in)):
        var x_t = restore_to_device(tape.sgl_x_in[i], ctx)
        sgl_x_in.append(TArc(x_t^))

    var img_out_t = restore_to_device(tape.img_out, ctx)
    var ln_img_out_t = restore_to_device(tape.ln_img_out, ctx)

    var dbl_saved = List[DoubleBlockSaved]()
    var sgl_saved = List[SingleBlockSaved]()
    var dummy_img = _unused_activation_arc(tape.img_out.dtype, ctx)
    var dummy_txt = _unused_activation_arc(tape.ln_img_out.dtype, ctx)

    return KleinStackForward(
        tape.out.copy(),
        dummy_img^,
        dummy_txt^,
        dbl_img_in^,
        dbl_txt_in^,
        sgl_x_in^,
        dbl_saved^,
        sgl_saved^,
        TArc(img_out_t^),
        TArc(ln_img_out_t^),
    )
