# models/ltx2/ltx2_video_stack_graph.mojo — L2 stack conductor for the LTX-2.3
# VIDEO graph backward (autograd_v2 engine, Option A coarse). A NEW file (not in
# ltx2_video_stack.mojo) mirrors the ideogram4/klein precedent of keeping the
# stack-graph driver out of the block module to sidestep block->graph->block
# import concerns (AUTOGRAD_V2_MOJO_DESIGN.md P7 recipe).
#
# BYTE-for-byte the same conductor as ltx2_video_stack_lora_backward
# (ltx2_video_stack.mojo:390-447) — same _tail_backward seed, same reverse block
# loop, same src.get_block + _attach_block_lora, same LoRA-grad collection by
# names[s], same d_x = bg.d_hidden.clone chaining, same LTX2VideoStackGrads
# return — EXCEPT the per-block (recompute train-forward + block backward) pair is
# replaced by ONE ltx2_video_block_graph_backward call (the graph engine does the
# recompute + backward inside apply_ltx2v). C13: this rides behind LTX2_V2_GRAPH;
# the hand-chain stays compiled + reachable.
#
# save_acts>0 FAILS LOUD: the graph path is full-recompute only (the Klein
# saved-tail precedent — a surface the graph doesn't carry). LTX2_SAVE_ACTS with
# the engine must use the hand-chain arm (it's a measured-negative knob anyway).
#
# Mojo 1.0.0b1, NVIDIA.

from std.memory import ArcPointer
from std.collections import Optional
from std.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.tensor import Tensor
from serenitymojo.models.ltx2.ltx2_video_stack import (
    _tail_backward, video_lora_names, _attach_block_lora, _nonfinite,
    LTX2VideoTail, LTX2VideoBlockSource, LTX2VideoStackGrads,
)
from serenitymojo.models.ltx2.ltx2_video_backward import (
    LTX2VideoBlockGrads, LTX2VideoBlockGradsDev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import Ltx2LoraGradStore
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.ltx2_video_block_graph import (
    ltx2_video_block_graph_backward, ltx2_video_block_graph_backward_slab_dev,
)
from serenitymojo.autograd_v2.step_slab import StepSlab


def ltx2_video_stack_lora_backward_graph[S_V: Int, N_TXT: Int](
    d_pred: Tensor, saved_inputs: List[ArcPointer[Tensor]], x_last: Tensor,
    enc: Tensor, v_temb: Tensor, v_embedded: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    tail: LTX2VideoTail,
    src: LTX2VideoBlockSource,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    preset: Int = 0,
    save_acts_k: Int = 0,
) raises -> LTX2VideoStackGrads:
    if save_acts_k > 0:
        raise Error(
            "ltx2_video_stack_lora_backward_graph: save_acts>0 is not carried by the"
            " graph path (full-recompute only — Klein saved-tail precedent); use the"
            " hand-chain arm (unset LTX2_V2_GRAPH) for LTX2_SAVE_ACTS")
    var names = video_lora_names(preset)
    var n_slots = len(names)
    var n_ad = num_layers * n_slots
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_ad):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())

    var d_x = _tail_backward[S_V](d_pred, x_last, v_embedded, tail, eps, ctx)

    var bi = num_layers - 1
    while bi >= 0:
        var w = src.get_block(bi, ctx)
        _attach_block_lora(w, bi, names, lora_a, lora_b, lora_scale)
        # ONE graph call = the (recompute train-forward + block backward) pair.
        var bg = ltx2_video_block_graph_backward[S_V, N_TXT](
            TArc(Tensor(d_x.buf.copy(), d_x.shape(), d_x.dtype())),
            w, saved_inputs[bi].copy(),
            enc, v_temb, v_prompt_ts, v_cos, v_sin, eps, ctx,
        )
        var base = bi * n_slots
        for s in range(n_slots):
            var wkey = names[s]
            for ref g in bg.lora:
                if g.name == wkey:
                    d_a_flat[base + s] = g.d_a.copy()
                    d_b_flat[base + s] = g.d_b.copy()
        d_x = bg.d_hidden.clone(ctx)
        bi -= 1

    var d_input = d_x.to_host(ctx)
    var nbad = _nonfinite(d_input)
    for i in range(n_ad):
        nbad += _nonfinite(d_a_flat[i]) + _nonfinite(d_b_flat[i])
    return LTX2VideoStackGrads(d_a_flat^, d_b_flat^, d_input^, nbad)


def ltx2_video_stack_lora_backward_graph_slab[S_V: Int, N_TXT: Int](
    d_pred: Tensor, saved_inputs: List[ArcPointer[Tensor]], x_last: Tensor,
    enc: Tensor, v_temb: Tensor, v_embedded: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    tail: LTX2VideoTail,
    src: LTX2VideoBlockSource,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    preset: Int = 0,
    save_acts_k: Int = 0,
) raises -> LTX2VideoStackGrads:
    """StepSlab + device-grad variant of `ltx2_video_stack_lora_backward_graph`
    (this file) — LTX2_V2_SLAB. Identical conductor to the flag-OFF graph loop
    EXCEPT: (1) ONE stack-owned StepSlab (apply_ltx2v_slab_dev marks/rewinds per
    block, so only ONE block's peak is ever live); (2) the per-block backward is
    the device-grad engine path (ltx2_video_block_graph_backward_slab_dev) — LoRA
    d_A/d_B stay device-resident through the loop; (3) ONE boundary
    ltx2_lora_dev_readback after the loop fills d_a_flat/d_b_flat host F32 —
    byte-identical to the flag-OFF path (C12), so the optimizer input is
    unchanged. The C10 seam (src.get_block + _attach_block_lora block acquisition)
    stays OUTSIDE the per-block slab window.

    Slab size 2 GiB: covers the measured per-block v2v peak 1_518_931_456 B
    (~1.41 GiB, ltx2_block_slab_parity p1_v2v_ffn = 582 allocs) with headroom;
    per-block rewind means the loop never exceeds one block's footprint.

    Rung 3 STAGING (this rung): the per-step inputs (block_input/enc/v_temb/
    v_prompt_ts) are copied into FIXED buffers and d_x rides a 2-buffer PING-PONG,
    so the block graph reads/writes STABLE device addresses (C8, capture
    prerequisite). This RETIRES the d_x `.clone` chaining (both the stack copy and
    the apply copy-out) via d_hidden_dst. Buffers are per-step here (inert); unit
    (d) makes them persistent for cross-step capture + does the VRAM re-budget."""
    if save_acts_k > 0:
        raise Error(
            "ltx2_video_stack_lora_backward_graph_slab: save_acts>0 is not carried"
            " by the graph path (full-recompute only); use the hand-chain arm")
    var names = video_lora_names(preset)
    var n_slots = len(names)
    var n_ad = num_layers * n_slots
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_ad):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())

    var slab = StepSlab(ctx, 2 * 1024 * 1024 * 1024)   # stack-owned, one slab
    # RESIDENT grad store (rung 3): per (block,slot) PRE-ALLOCATED buffers; the
    # per-block backward enqueue_copy's d_A/d_B into store[bi] (NO alloc,
    # capture-safe — RETIRES the .clone in the stack path). ONE boundary readback
    # fills d_a_flat/d_b_flat host F32 (C12, byte-identical to flag-OFF).
    var store = Ltx2LoraGradStore.create(num_layers, names, lora_a, lora_b, ctx)

    # ── rung 3 STAGING (LTX2_V2_SLAB): per-step FIXED buffers so the coming
    #    captured block graph reads/writes STABLE addresses (C8: replay re-runs
    #    recorded kernels on recorded pointers). The block fwd/bwd input set is
    #    CLOSED (audited 2026-07-17: ltx2_video_backward.mojo has no noise/randn/
    #    global/external per-step device read): {block_input, enc, v_temb,
    #    v_prompt_ts, v_cos, v_sin, eps}. enc/v_temb/v_prompt_ts are identical
    #    across all 48 blocks ⇒ staged ONCE; block_input is copied per block;
    #    d_x rides a 2-buffer PING-PONG (block j reads pp[j%2], writes pp[(j+1)%2])
    #    so read and write never alias — this retires BOTH the stack d_x.clone AND
    #    the apply copy-out .clone (W4b). v_cos/v_sin/eps are geometry-fixed
    #    resident, passed as-is.
    #    NOTE: buffers are allocated PER-STEP here (numerically inert; a net alloc
    #    win vs the 96 per-block .clones). Unit (d) hoists them to a persistent
    #    stage for cross-step capture stability + the VRAM re-budget.
    var d_seed = _tail_backward[S_V](d_pred, x_last, v_embedded, tail, eps, ctx)
    var dh_nbytes = d_seed.nbytes()

    var enc_buf = ctx.enqueue_create_buffer[DType.uint8](enc.nbytes())
    var enc_dst = enc_buf.create_sub_buffer[DType.uint8](0, enc.nbytes())
    ctx.enqueue_copy(dst_buf=enc_dst, src_buf=enc.buf)
    var enc_st = Tensor(enc_buf.create_sub_buffer[DType.uint8](0, enc.nbytes()),
                        enc.shape(), enc.dtype())

    var vt_buf = ctx.enqueue_create_buffer[DType.uint8](v_temb.nbytes())
    var vt_dst = vt_buf.create_sub_buffer[DType.uint8](0, v_temb.nbytes())
    ctx.enqueue_copy(dst_buf=vt_dst, src_buf=v_temb.buf)
    var vt_st = Tensor(vt_buf.create_sub_buffer[DType.uint8](0, v_temb.nbytes()),
                       v_temb.shape(), v_temb.dtype())

    var vp_buf = ctx.enqueue_create_buffer[DType.uint8](v_prompt_ts.nbytes())
    var vp_dst = vp_buf.create_sub_buffer[DType.uint8](0, v_prompt_ts.nbytes())
    ctx.enqueue_copy(dst_buf=vp_dst, src_buf=v_prompt_ts.buf)
    var vp_st = Tensor(vp_buf.create_sub_buffer[DType.uint8](0, v_prompt_ts.nbytes()),
                       v_prompt_ts.shape(), v_prompt_ts.dtype())

    # block_input staging (rewritten per block; same shape as d_hidden).
    var bin_buf = ctx.enqueue_create_buffer[DType.uint8](dh_nbytes)

    # d_x ping-pong: two fixed buffers; seed pp0 with the tail-backward output.
    var pp0 = ctx.enqueue_create_buffer[DType.uint8](dh_nbytes)
    var pp1 = ctx.enqueue_create_buffer[DType.uint8](dh_nbytes)
    var seed_dst = pp0.create_sub_buffer[DType.uint8](0, dh_nbytes)
    ctx.enqueue_copy(dst_buf=seed_dst, src_buf=d_seed.buf)
    var d_x = Tensor(pp0.create_sub_buffer[DType.uint8](0, dh_nbytes),
                     d_seed.shape(), d_seed.dtype())

    # Steady-state VRAM headroom print (rung 3 diagnostic; the fail-loud init
    # re-budget lands in unit (d) with the weight stage + resident-block target).
    var stage_mib = (enc.nbytes() + v_temb.nbytes() + v_prompt_ts.nbytes()
                     + 3 * dh_nbytes) // (1024 * 1024)
    var mi = ctx.get_memory_info()
    print("  [ltx2 rung3] VRAM steady-state: free=",
          Float64(Int(mi[0])) / (1024.0 * 1024.0 * 1024.0), "GB / total=",
          Float64(Int(mi[1])) / (1024.0 * 1024.0 * 1024.0),
          "GB; per-step staging+pingpong=", stage_mib, "MiB")

    var bi = num_layers - 1
    var j = 0
    while bi >= 0:
        var w = src.get_block(bi, ctx)   # C10: block acquire OUTSIDE the slab window
        _attach_block_lora(w, bi, names, lora_a, lora_b, lora_scale)

        # block_input = saved_inputs[bi] copied into the FIXED staging buffer.
        var bin_nb = saved_inputs[bi][].nbytes()
        var bin_dst = bin_buf.create_sub_buffer[DType.uint8](0, bin_nb)
        ctx.enqueue_copy(dst_buf=bin_dst, src_buf=saved_inputs[bi][].buf)

        # d_hidden DEST = pp[(j+1)%2] (differs from the source pp[j%2] d_x reads ⇒
        # no read/write alias). d_x already views pp[j%2] (seed or prev write).
        var dh_dst = pp1.copy()
        if (j + 1) % 2 == 0:
            dh_dst = pp0.copy()

        var bg = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
            TArc(Tensor(d_x.buf.copy(), d_x.shape(), d_x.dtype())),
            w, TArc(Tensor(bin_buf.create_sub_buffer[DType.uint8](0, bin_nb),
                           saved_inputs[bi][].shape(), saved_inputs[bi][].dtype())),
            enc_st, vt_st, vp_st, v_cos, v_sin, eps, ctx, slab, store, bi,
            Optional[DeviceBuffer[DType.uint8]](dh_dst^),
        )
        _ = bg^   # d_hidden already written into pp[(j+1)%2] (a view); no clone.

        # d_x for the next block = the just-written ping-pong buffer pp[(j+1)%2].
        var nxt = pp1.create_sub_buffer[DType.uint8](0, dh_nbytes)
        if (j + 1) % 2 == 0:
            nxt = pp0.create_sub_buffer[DType.uint8](0, dh_nbytes)
        d_x = Tensor(nxt^, d_seed.shape(), d_seed.dtype())
        bi -= 1
        j += 1

    # ── BOUNDARY readback (C12): the resident store → host F32. readback_block is
    #    slot-order (= names order), so d_a_flat[base+s] = host[s] directly (no
    #    name match needed — the per-block backward wrote store[bi][slot_of(name)]). ─
    for block_bi in range(num_layers):
        var host = store.readback_block(block_bi, ctx)
        var base = block_bi * n_slots
        for s in range(n_slots):
            d_a_flat[base + s] = host[s].d_a.copy()
            d_b_flat[base + s] = host[s].d_b.copy()

    var d_input = d_x.to_host(ctx)
    var nbad = _nonfinite(d_input)
    for i in range(n_ad):
        nbad += _nonfinite(d_a_flat[i]) + _nonfinite(d_b_flat[i])
    return LTX2VideoStackGrads(d_a_flat^, d_b_flat^, d_input^, nbad)
