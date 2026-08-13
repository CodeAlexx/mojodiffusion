# serenitymojo/models/ltx2/ltx2_av_stack.mojo
#
# LTX-2.3 22B JOINT AUDIO-VIDEO training STACK forward (P6.1). Mirrors the video
# stack `ltx2_video_stack.mojo:327 ltx2_video_stack_lora_forward` with the audio
# stream added: HEAD (patchify + AdaLayerNormSingle modulation from ONE sigma via
# the 8 global adaln MLPs) -> num_layers x `ltx2_block_forward_av_train` (the
# gated AV block, saving LTX2AVBlockActs per block) -> TAIL (output_stage per
# stream). Full S_A, no attention padding mask (scout: audio padding is loss-only).
#
# Head/tail math is an op-for-op port of pipeline/ltx2_t2v_av_mvp.mojo
# (_timestep_embedding / _adaln_single / _output_stage) — the MVP is a main()
# module (can't import), so ported here; scripts/ltx2_av_stack_oracle.py is the
# numeric reference (gate parity/ltx2_av_stack_parity.mojo, cos>=0.999 F32).

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import reshape, add, mul, add_scalar, slice
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx

from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.offload.ltx2_int4_block_stream import LTX2Int4BlockStream
from serenitymojo.models.ltx2.ltx2_av_backward import (
    ltx2_block_forward_av_train,
    LTX2AVBlockActs,
    LTX2AVTrainForward,
    ltx2_block_backward_av,
    LTX2AVBlockGrads,
    LoraPairGrad,
)

comptime VD = 4096
comptime AD = 2048
comptime TS_MULT = Float32(1000.0)   # sigma -> timestep scale (MVP _build_mod)


# ── AV block source: streamed or fp8-RESIDENT block weights ──────────────────
# Mirrors the video arm's LTX2VideoBlockSource (ltx2_video_stack.mojo:107-152).
#
# WHY THE DEQUANT PATH, NOT `load_block_fp8_resident`/`from_fp8_resident`:
# those return RAW F8_E4M3 weights for the fused `linear_fp8` GEMM, and
# `LTX2AVBlockWeights.to_f32` REFUSES such a block outright (ltx2_dit.mojo:
# 1010-1017, "raw F8_E4M3 weights have no F32 cast path"). The AV training
# stack is an F32 class — every gate is F32 at cos>=0.999 — so it cannot use
# them. `enable_fp8_resident_range` + `load_block_bf16` parks the raw fp8 bytes
# in VRAM and dequants from the RESIDENT copy (ltx2_block_stream.mojo:360-366
# dispatches to _load_resident_block_bf16), which is what the video TRAINING arm
# does and which "changes NO math" (train_ltx2_av.mojo:1221). MEASURED: the fp8
# checkpoint and the dequant-bf16 checkpoint give BIT-IDENTICAL weights on this
# path, 86/86 keys on both a BF16 boundary block and an fp8 inner block
# (parity/ltx2_av_ckpt_equiv_probe.mojo), so residency costs zero accuracy.
struct LTX2AVBlockSource(Movable):
    var stream: LTX2BlockStream
    # When set, blocks come from a packed W4A16 slab instead of the fp8
    # checkpoint (LTX2Int4BlockStream reconstructs dense BF16 on use; it
    # auto-detects class-A svdquant vs squareq_w4 per linear by `.smooth`
    # key presence). Mirrors LTX2VideoBlockSource.int4.
    var quant: Optional[LTX2Int4BlockStream]
    var cfg: LTX2Config
    var run_f32: Bool

    def __init__(
        out self, var stream: LTX2BlockStream,
        var quant: Optional[LTX2Int4BlockStream], cfg: LTX2Config, run_f32: Bool
    ):
        self.stream = stream^
        self.quant = quant^
        self.cfg = cfg
        self.run_f32 = run_f32

    @staticmethod
    def open(ckpt: String, cfg: LTX2Config, run_f32: Bool = True) raises -> LTX2AVBlockSource:
        return LTX2AVBlockSource(
            LTX2BlockStream.open(ckpt),
            Optional[LTX2Int4BlockStream](None), cfg, run_f32)

    @staticmethod
    def open_squareq(
        ckpt: String, slab: String, cfg: LTX2Config, run_f32: Bool = True
    ) raises -> LTX2AVBlockSource:
        """SquareQ W4A16 residency arm: blocks reconstructed to dense BF16 from
        the packed slab (per quantized linear: qweight U8 [out,in/2] + wscales
        BF16 [in/group,out] + lora_down/lora_up BF16, NO smooth —
        ops/squareq.squareq_reconstruct_weight, dequant4 + inverse-H256 +
        low-rank). Downstream is UNCHANGED: same FP8Block key contract ->
        from_fp8_block -> same forward. The checkpoint is still opened (cheap
        mmap, no VRAM) for block_count parity; head/tail keep loading from it
        elsewhere. Mirrors LTX2VideoBlockSource.open_int4."""
        return LTX2AVBlockSource(
            LTX2BlockStream.open(ckpt),
            Optional[LTX2Int4BlockStream](LTX2Int4BlockStream.open(slab)),
            cfg, run_f32)

    def block_count(self) -> Int:
        return self.stream.block_count()

    def enable_resident(mut self, first: Int, last: Int, ctx: DeviceContext) raises:
        """Park blocks [first,last] in VRAM (raw fp8 where present, BF16 else).

        Fails loud if the CHECKPOINT carries no fp8 tensors AT ALL — i.e. someone
        pointed a resident run at the pre-dequantized bf16 file, where residency
        would balloon VRAM (bf16 is 2 B/param vs fp8's 1) for no dequant saving.

        The emptiness test is CHECKPOINT-scoped, deliberately NOT range-scoped:
        blocks 0-3 and 47 are BF16 boundary blocks carrying ZERO fp8 tensors even
        in the correct fp8 checkpoint, so a small-range run (the 2-block parity
        gates use blocks 0..1) is legitimately all-BF16. Range-scoping the test
        would reject the RIGHT file. The range is checked first as the fast path
        — a real 48-block run hits fp8 by block 4 — and only an empty range pays
        for the full sweep needed to tell "boundary blocks" from "wrong file"."""
        if self.quant:
            raise Error(
                "LTX2AVBlockSource.enable_resident: fp8 residency does not"
                " apply to the packed-slab (squareq/int4) arm — blocks"
                " reconstruct from the slab per visit. Set --resident_blocks 0"
                " / unset LTX2_RESIDENT_BLOCKS."
            )
        var n_fp8 = 0
        for b in range(first, last + 1):
            n_fp8 += self.stream.fp8_tensor_count(b)
            if n_fp8 > 0:
                break
        if n_fp8 == 0:
            for b in range(self.stream.block_count()):
                n_fp8 += self.stream.fp8_tensor_count(b)
                if n_fp8 > 0:
                    break
        if n_fp8 == 0:
            raise Error(
                String("LTX2AVBlockSource.enable_resident: checkpoint carries NO")
                + " fp8 tensors in any of its " + String(self.stream.block_count())
                + " blocks — this is the pre-dequantized bf16 file. Point"
                " --ltx2_checkpoint at ltx-2.3-22b-distilled-fp8.safetensors"
                " for a resident run, or set --resident_blocks 0."
            )
        self.stream.enable_fp8_resident_range(first, last, ctx)

    def resident_bytes(self) -> Int:
        return self.stream.resident_bytes()

    def get_block(self, i: Int, ctx: DeviceContext) raises -> LTX2AVBlockWeights:
        var w: LTX2AVBlockWeights
        if self.quant:
            var blkq = self.quant.value().load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blkq^, self.cfg, ctx)
        else:
            var blk = self.stream.load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk^, self.cfg, ctx)
        if self.run_f32:
            return w.to_f32(ctx)
        return w^


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


def _linear_b(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, Optional[Tensor](b.clone(ctx)), ctx)


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


# ── head: AdaLayerNormSingle (MVP _timestep_embedding / _adaln_single, F32) ────
def _timestep_embedding(ts: List[Float32], dim: Int, ctx: DeviceContext) raises -> Tensor:
    var n = len(ts)
    var half = dim // 2
    var out = List[Float32]()
    out.resize(n * dim, Float32(0.0))
    for r in range(n):
        var t = Float64(ts[r])
        for i in range(half):
            var freq = fexp(-Float64(i) * flog(Float64(10000.0)) / Float64(half))
            var arg = t * freq
            out[r * dim + i] = Float32(fcos(arg))          # cos first
            out[r * dim + half + i] = Float32(fsin(arg))   # then sin
    return Tensor.from_host(out, _sh2(n, dim), STDtype.F32, ctx)


# AdaLayerNormSingle: (mod [N, n*dim], embedded [N, dim]). Weights loaded from st.
def _adaln_single(
    st: ShardedSafeTensors, base: String, ts_vals: List[Float32], ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    var pre = String("model.diffusion_model.") + base
    var emb = _timestep_embedding(ts_vals, 256, ctx)
    var h = _linear_b(emb, _load(st, pre + ".emb.timestep_embedder.linear_1.weight", ctx),
                      _load(st, pre + ".emb.timestep_embedder.linear_1.bias", ctx), ctx)
    h = silu(h, ctx)
    var embedded = _linear_b(h, _load(st, pre + ".emb.timestep_embedder.linear_2.weight", ctx),
                             _load(st, pre + ".emb.timestep_embedder.linear_2.bias", ctx), ctx)
    var mod = _linear_b(silu(embedded, ctx), _load(st, pre + ".linear.weight", ctx),
                        _load(st, pre + ".linear.bias", ctx), ctx)
    return (mod^, embedded^)


struct LTX2AVHead(Movable):
    var v_temb: Tensor
    var a_temb: Tensor
    var v_embedded: Tensor
    var a_embedded: Tensor
    var v_ca_ss: Tensor
    var a_ca_ss: Tensor
    var v_ca_gate: Tensor
    var a_ca_gate: Tensor
    var v_prompt_ts: Tensor
    var a_prompt_ts: Tensor

    def __init__(
        out self, var v_temb: Tensor, var a_temb: Tensor,
        var v_embedded: Tensor, var a_embedded: Tensor,
        var v_ca_ss: Tensor, var a_ca_ss: Tensor,
        var v_ca_gate: Tensor, var a_ca_gate: Tensor,
        var v_prompt_ts: Tensor, var a_prompt_ts: Tensor,
    ):
        self.v_temb = v_temb^; self.a_temb = a_temb^
        self.v_embedded = v_embedded^; self.a_embedded = a_embedded^
        self.v_ca_ss = v_ca_ss^; self.a_ca_ss = a_ca_ss^
        self.v_ca_gate = v_ca_gate^; self.a_ca_gate = a_ca_gate^
        self.v_prompt_ts = v_prompt_ts^; self.a_prompt_ts = a_prompt_ts^


# Build all modulation tensors from ONE sigma (MVP _build_mod, 8 global adaln MLPs).
def ltx2_av_build_head[S_V: Int, S_A: Int, N_TXT: Int](
    st: ShardedSafeTensors, sigma: Float32, ctx: DeviceContext
) raises -> LTX2AVHead:
    var stv = sigma * TS_MULT
    var tv = List[Float32]()
    for _ in range(S_V): tv.append(stv)
    var vt = _adaln_single(st, String("adaln_single"), tv, ctx)
    var v_temb = reshape(vt[0], _sh3(1, S_V, 9 * VD), ctx)
    var v_embedded = reshape(vt[1], _sh3(1, S_V, VD), ctx)

    var ta = List[Float32]()
    for _ in range(S_A): ta.append(stv)
    var at = _adaln_single(st, String("audio_adaln_single"), ta, ctx)
    var a_temb = reshape(at[0], _sh3(1, S_A, 9 * AD), ctx)
    var a_embedded = reshape(at[1], _sh3(1, S_A, AD), ctx)

    var g1 = List[Float32](); g1.append(stv)
    var vcs = _adaln_single(st, String("av_ca_video_scale_shift_adaln_single"), g1, ctx)
    var v_ca_ss = reshape(vcs[0], _sh3(1, 1, 4 * VD), ctx)
    var acs = _adaln_single(st, String("av_ca_audio_scale_shift_adaln_single"), g1, ctx)
    var a_ca_ss = reshape(acs[0], _sh3(1, 1, 4 * AD), ctx)
    var vcg = _adaln_single(st, String("av_ca_a2v_gate_adaln_single"), g1, ctx)
    var v_ca_gate = reshape(vcg[0], _sh3(1, 1, VD), ctx)
    var acg = _adaln_single(st, String("av_ca_v2a_gate_adaln_single"), g1, ctx)
    var a_ca_gate = reshape(acg[0], _sh3(1, 1, AD), ctx)

    var tp = List[Float32]()
    for _ in range(N_TXT): tp.append(stv)
    var vpt = _adaln_single(st, String("prompt_adaln_single"), tp, ctx)
    var v_prompt_ts = reshape(vpt[0], _sh3(1, N_TXT, 2 * VD), ctx)
    var apt = _adaln_single(st, String("audio_prompt_adaln_single"), tp, ctx)
    var a_prompt_ts = reshape(apt[0], _sh3(1, N_TXT, 2 * AD), ctx)

    return LTX2AVHead(v_temb^, a_temb^, v_embedded^, a_embedded^, v_ca_ss^, a_ca_ss^,
                      v_ca_gate^, a_ca_gate^, v_prompt_ts^, a_prompt_ts^)


# ── patchify (MVP): latent [1,S,128] -> [1,S,D] via patchify_proj ─────────────
def _patchify(latent: Tensor, st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    var pre = String("model.diffusion_model.") + key
    return _linear_b(latent, _load(st, pre + ".weight", ctx), _load(st, pre + ".bias", ctx), ctx)


# ── tail (MVP _output_stage): layer_norm + (sst row0/1 + embedded) + proj ─────
def _output_stage(
    hs: Tensor, st: ShardedSafeTensors, sst_key: String, embedded: Tensor,
    proj_key: String, dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    var pre = String("model.diffusion_model.")
    var ones = List[Float32](); var zeros = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0)); zeros.append(Float32(0.0))
    var w_ln = Tensor.from_host(ones, _sh1(dim), hs.dtype(), ctx)
    var b_ln = Tensor.from_host(zeros, _sh1(dim), hs.dtype(), ctx)
    var normed = layer_norm(hs, w_ln, b_ln, Float32(1e-6), ctx)
    var sst = _load(st, pre + sst_key, ctx)   # [2, dim]
    var shift_row = reshape(slice(sst, 0, 0, 1, ctx), _sh3(1, 1, dim), ctx)
    var scale_row = reshape(slice(sst, 0, 1, 1, ctx), _sh3(1, 1, dim), ctx)
    var v_shift = add(shift_row, embedded, ctx)
    var v_scale = add(scale_row, embedded, ctx)
    var one_plus = add_scalar(v_scale, Float32(1.0), ctx)
    var out = add(mul(normed, one_plus, ctx), v_shift, ctx)
    return _linear_b(out, _load(st, pre + proj_key + ".weight", ctx),
                     _load(st, pre + proj_key + ".bias", ctx), ctx)


# the 24 AV LoRA target modules (musubi LTX2_INCLUDE_PATTERNS_T2V), slot order.
def _av_lora_slots() -> List[String]:
    var mods = List[String]()
    mods.append(String("attn1")); mods.append(String("attn2"))
    mods.append(String("audio_attn1")); mods.append(String("audio_attn2"))
    mods.append(String("audio_to_video_attn")); mods.append(String("video_to_audio_attn"))
    var projs = List[String]()
    projs.append(String("to_q")); projs.append(String("to_k"))
    projs.append(String("to_v")); projs.append(String("to_out.0"))
    var out = List[String]()
    for ref m in mods:
        for ref p in projs:
            out.append(m + "." + p)
    # P6.2 (a) TRUE-672 + rider: 4 FFN pairs (audio_ff for the 672 preset + video
    # ff for the 1344 v2v-full reconcile), appended AFTER the 24 attention pairs.
    # Order MUST match scripts/ltx2_av_stack_oracle.py FFN_MODULES.
    out.append(String("audio_ff.net.0.proj"))
    out.append(String("audio_ff.net.2"))
    out.append(String("ff.net.0.proj"))
    out.append(String("ff.net.2"))
    return out^


struct LTX2AVStackForward(Movable):
    var video_vel: Tensor
    var audio_vel: Tensor
    var saved: List[ArcPointer[LTX2AVTrainForward]]   # retained acts, blocks [0,save_acts_k)
    # RECOMPUTE conductor (mirrors ltx2_video_stack.mojo:300-318 saved_inputs/x_last).
    # Per-block INPUTS for BOTH streams — the only thing the backward needs to
    # rebuild a block's acts. At the trainer geometry (S_V=576/S_A=16, F32):
    # video 576*4096*4 = 9.00 MiB + audio 16*2048*4 = 0.125 MiB = 9.125 MiB/block,
    # i.e. 438 MiB for 48 blocks, versus the 17.59 GiB the all-blocks-retained arm
    # parked on device (GATE 0: recompute 35.2ms/block beats a host activation
    # tape's 80.1ms/block by 2.3x, so we recompute rather than offload).
    var saved_v_in: List[ArcPointer[Tensor]]
    var saved_a_in: List[ArcPointer[Tensor]]
    # Last block's outputs = the tail's inputs. REQUIRED as explicit fields: the
    # callers used to read these off saved[num_layers-1], which does not exist
    # once save_acts_k < num_layers.
    var v_last: Tensor
    var a_last: Tensor

    def __init__(
        out self, var video_vel: Tensor, var audio_vel: Tensor,
        var saved: List[ArcPointer[LTX2AVTrainForward]],
        var saved_v_in: List[ArcPointer[Tensor]],
        var saved_a_in: List[ArcPointer[Tensor]],
        var v_last: Tensor, var a_last: Tensor,
    ):
        self.video_vel = video_vel^
        self.audio_vel = audio_vel^
        self.saved = saved^
        self.saved_v_in = saved_v_in^
        self.saved_a_in = saved_a_in^
        self.v_last = v_last^
        self.a_last = a_last^


# ── STACK FORWARD: HEAD (caller) -> N blocks (save acts) -> TAIL ──────────────
# lora_a/lora_b are flat [num_layers*24] ArcPointer tensors (slot order
# _av_lora_slots); block i uses [i*24, i*24+24). lora empty -> base only.
def ltx2_av_stack_forward[S_V: Int, S_A: Int, N_TXT: Int](
    st: ShardedSafeTensors, src: LTX2AVBlockSource,
    hidden: Tensor, ahs: Tensor, enc: Tensor, aenc: Tensor,
    head: LTX2AVHead,
    v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    save_acts_k: Int = 0,
) raises -> LTX2AVStackForward:
    var slots = _av_lora_slots()
    var n_slot = len(slots)
    var have_lora = len(lora_a) >= num_layers * n_slot
    var x = hidden.clone(ctx)
    var xa = ahs.clone(ctx)
    var saved = List[ArcPointer[LTX2AVTrainForward]]()
    var saved_v_in = List[ArcPointer[Tensor]]()
    var saved_a_in = List[ArcPointer[Tensor]]()

    for i in range(num_layers):
        # save this block's INPUTS (both streams) — the recompute checkpoint.
        saved_v_in.append(ArcPointer[Tensor](x.clone(ctx)))
        saved_a_in.append(ArcPointer[Tensor](xa.clone(ctx)))
        var w = src.get_block(i, ctx)
        if have_lora:
            for s in range(n_slot):
                var idx = i * n_slot + s
                w.add_lora_factor(slots[s] + ".weight",
                                  lora_a[idx][].clone(ctx), lora_b[idx][].clone(ctx), lora_scale)
        var fwd = ltx2_block_forward_av_train[S_V, S_A, N_TXT](
            w, x, xa, enc, aenc,
            head.v_temb, head.a_temb, head.v_ca_ss, head.a_ca_ss,
            head.v_ca_gate, head.a_ca_gate, head.v_prompt_ts, head.a_prompt_ts,
            v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
            eps, ctx,
        )
        x = fwd.video_out.clone(ctx)
        xa = fwd.audio_out.clone(ctx)
        if i < save_acts_k:
            saved.append(ArcPointer[LTX2AVTrainForward](fwd^))
        # else: fwd (incl. its ~375 MiB of acts) drops at scope end — the backward
        # recomputes it from saved_v_in[i]/saved_a_in[i].

    var v_last = x.clone(ctx)
    var a_last = xa.clone(ctx)
    var v_vel = _output_stage(x, st, String("scale_shift_table"), head.v_embedded,
                              String("proj_out"), VD, ctx)
    var a_vel = _output_stage(xa, st, String("audio_scale_shift_table"), head.a_embedded,
                              String("audio_proj_out"), AD, ctx)
    return LTX2AVStackForward(v_vel^, a_vel^, saved^, saved_v_in^, saved_a_in^,
                              v_last^, a_last^)


# ── TAIL BACKWARD (reverse of _output_stage; d_vel -> d into block-N output) ──
def av_tail_backward[S: Int](
    d_vel: Tensor, x_last: Tensor, embedded: Tensor,
    st: ShardedSafeTensors, sst_key: String, proj_key: String, dim: Int,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var pre = String("model.diffusion_model.")
    var out_ch = d_vel.shape()[len(d_vel.shape()) - 1]
    var proj_w = _load(st, pre + proj_key + ".weight", ctx)   # [out_ch, dim]
    var sst = _load(st, pre + sst_key, ctx)                   # [2, dim]
    var d_vel2 = reshape(d_vel, _sh2(S, out_ch), ctx)
    var d_x_out = linear_backward_dx(d_vel2, proj_w, S, dim, out_ch, ctx)   # [S, dim]
    var d_x_mod = reshape(d_x_out, _sh3(1, S, dim), ctx)
    var sst1 = reshape(slice(sst, 0, 1, 1, ctx), _sh3(1, 1, dim), ctx)      # scale row
    var scale = add(sst1, embedded, ctx)
    var d_normed = mul(d_x_mod, add_scalar(scale, Float32(1.0), ctx), ctx)
    var ones = List[Float32]()
    for _ in range(dim):
        ones.append(Float32(1.0))
    var w_ln = Tensor.from_host(ones, _sh1(dim), x_last.dtype(), ctx)
    return layer_norm_backward_dx(d_normed, x_last, w_ln, eps, ctx)


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


struct LTX2AVStackGrads(Movable):
    var d_a: List[List[Float32]]   # [num_layers*24][rank*in]
    var d_b: List[List[Float32]]   # [num_layers*24][out*rank]
    var d_hidden: List[Float32]    # grad into block-0 video input (head frozen)
    var d_ahs: List[Float32]       # grad into block-0 audio input
    var nonfinite: Int

    def __init__(
        out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_hidden: List[Float32], var d_ahs: List[Float32], nonfinite: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_hidden = d_hidden^
        self.d_ahs = d_ahs^
        self.nonfinite = nonfinite


# ── STACK BACKWARD: tail bwd (both streams) -> reverse block loop (block backward
#    on the saved acts, mirroring ltx2_video_stack_lora_backward's conductor) ->
#    the 24 LoRA d_A/d_B per block + d into block-0 hidden/ahs. Head FROZEN, so the
#    chain stops at the block-0 input. ─────────────────────────────────────────
def ltx2_av_stack_backward[S_V: Int, S_A: Int, N_TXT: Int](
    d_video_vel: Tensor, d_audio_vel: Tensor,
    saved_v_in: List[ArcPointer[Tensor]], saved_a_in: List[ArcPointer[Tensor]],
    v_hidden_last: Tensor, a_hidden_last: Tensor,
    enc: Tensor, aenc: Tensor,
    head: LTX2AVHead,
    v_cos: Tensor, v_sin: Tensor, a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor, ca_a_cos: Tensor, ca_a_sin: Tensor,
    st: ShardedSafeTensors, src: LTX2AVBlockSource,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    save_acts_k: Int = 0,
    saved_fwds: List[ArcPointer[LTX2AVTrainForward]] = List[ArcPointer[LTX2AVTrainForward]](),
) raises -> LTX2AVStackGrads:
    # SAVE-ACTS vs RECOMPUTE (mirrors ltx2_video_stack.mojo:400-434): blocks
    # [0, save_acts_k) consume the forward's retained acts; blocks
    # [save_acts_k, num_layers) RECOMPUTE the train-forward from that block's
    # saved inputs. save_acts_k=0 (the default, and what the trainer now uses)
    # => recompute every block. The two arms are numerically IDENTICAL — the
    # recompute runs the same code on the same inputs with the same-step LoRA
    # already re-attached below, which is why the gates must not move.
    var slots = _av_lora_slots()
    var n_slot = len(slots)
    var have_lora = len(lora_a) >= num_layers * n_slot
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(num_layers * n_slot):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())

    var d_x = av_tail_backward[S_V](d_video_vel, v_hidden_last, head.v_embedded,
                                    st, String("scale_shift_table"), String("proj_out"), VD, eps, ctx)
    var d_xa = av_tail_backward[S_A](d_audio_vel, a_hidden_last, head.a_embedded,
                                     st, String("audio_scale_shift_table"), String("audio_proj_out"), AD, eps, ctx)

    var bi = num_layers - 1
    while bi >= 0:
        var w = src.get_block(bi, ctx)
        if have_lora:
            for s in range(n_slot):
                var idx = bi * n_slot + s
                w.add_lora_factor(slots[s] + ".weight",
                                  lora_a[idx][].clone(ctx), lora_b[idx][].clone(ctx), lora_scale)
        var bg: LTX2AVBlockGrads
        if bi < save_acts_k:
            # SAVE-ACTS: read the retained forward's acts (no recompute).
            bg = ltx2_block_backward_av[S_V, S_A, N_TXT](
                w, saved_fwds[bi][].acts, d_x, d_xa,
                head.v_temb, head.a_temb, head.v_ca_ss, head.a_ca_ss,
                head.v_ca_gate, head.a_ca_gate,
                v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                eps, ctx,
            )
        else:
            # RECOMPUTE the train-forward on this block's saved inputs (same `w`,
            # LoRA already attached above). `rf` — and its ~375 MiB of acts — dies
            # at the end of this iteration, so only ONE block's acts are ever
            # device-resident.
            var rf = ltx2_block_forward_av_train[S_V, S_A, N_TXT](
                w, saved_v_in[bi][], saved_a_in[bi][], enc, aenc,
                head.v_temb, head.a_temb, head.v_ca_ss, head.a_ca_ss,
                head.v_ca_gate, head.a_ca_gate, head.v_prompt_ts, head.a_prompt_ts,
                v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                eps, ctx,
            )
            bg = ltx2_block_backward_av[S_V, S_A, N_TXT](
                w, rf.acts, d_x, d_xa,
                head.v_temb, head.a_temb, head.v_ca_ss, head.a_ca_ss,
                head.v_ca_gate, head.a_ca_gate,
                v_cos, v_sin, a_cos, a_sin, ca_v_cos, ca_v_sin, ca_a_cos, ca_a_sin,
                eps, ctx,
            )
        var base = bi * n_slot
        for s in range(n_slot):
            var wkey = slots[s] + ".weight"
            for ref g in bg.lora:
                if g.name == wkey:
                    d_a_flat[base + s] = g.d_a.copy()
                    d_b_flat[base + s] = g.d_b.copy()
        d_x = bg.d_hidden.clone(ctx)
        d_xa = bg.d_ahs.clone(ctx)
        bi -= 1

    var d_hidden_host = d_x.to_host(ctx)
    var d_ahs_host = d_xa.to_host(ctx)
    var nbad = _nonfinite(d_hidden_host) + _nonfinite(d_ahs_host)
    for i in range(num_layers * n_slot):
        nbad += _nonfinite(d_a_flat[i]) + _nonfinite(d_b_flat[i])
    return LTX2AVStackGrads(d_a_flat^, d_b_flat^, d_hidden_host^, d_ahs_host^, nbad)
