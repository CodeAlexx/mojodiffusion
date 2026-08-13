# serenitymojo/models/ltx2/ltx2_video_stack.mojo
#
# LTX-2.3 22B VIDEO-MODE (audio=None) FULL TRAINING STACK: frozen head + N
# streamed transformer blocks (video-only train arm) + frozen musubi tail, with
# per-block factorized LoRA on the 8 video-mode targets and a hand-chained
# backward through the tail into the last block.
#
#   latent -> patchify -> patchify_proj -> [block 0 .. block N-1] -> tail -> pred
#
# COMPOSES (does NOT modify) the lead-gated per-block video train arm
#   serenitymojo/models/ltx2/ltx2_video_backward.mojo
#     ltx2_video_block_train_forward[S_V,N_TXT] / ltx2_video_block_backward[..]
# over LTX2AVBlockWeights (models/dit/ltx2_dit.mojo), exactly as the AV MVP spine
# (pipeline/ltx2_t2v_av_mvp.mojo) composes ltx2_block_forward_av. Streaming +
# per-block recompute-in-backward mirror models/ltx2/ltx2_stack_lora.mojo.
#
# HEAD (frozen, no backward — grad stops at block-0 input): patchify (musubi
# VideoLatentPatchifier, patch_size (1,1,1): `b c f h w -> b (f h w) c`) ->
# patchify_proj; adaln_single(sigma) -> (v_temb [1,S_V,9D], v_embedded
# [1,S_V,D]); prompt_adaln_single(sigma) -> v_prompt_ts [1,N_TXT,2D]; 3D video
# RoPE. All head math mirrors the proven AV MVP spine (ltx2_t2v_av_mvp.mojo:
# _timestep_embedding / _adaln_single / _compute_rope / _build_video_coords).
# The text context `enc` is the cached POST-connector video_prompt_embeds
# (musubi video_only_encoder runs the connector at cache time — it is NOT re-run
# here; the DiT consumes the encoder output directly via each block's
# prompt_scale_shift_table KV-modulation).
#
# TAIL (frozen): musubi _process_output "AdaLN Structural Fix" done in F32
#   (musubi model.py:797): x32 = layer_norm(x.f32, no-affine, eps);
#   shift/scale = scale_shift_table[0/1] + embedded_timestep (F32);
#   x = (x32*(1+scale)+shift).to(dtype); pred = proj_out(x). Backward through the
#   tail is required: d_pred -> proj_out dx -> F32 modulate -> F32 layer_norm dx
#   -> d(last block). Grads to scale_shift_table/proj_out/embedded are DISCARDED.
#
# Mojo current-beta: `def` not `fn`; `comptime` not `alias`; move-only Tensor
# (Movable result structs, `^`, ArcPointer for List storage).

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt, cos as fcos, sin as fsin, pow as fpow, log as flog, pi

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import (
    reshape, add, mul, mul_scalar, add_scalar, slice, permute, concat,
)
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx

from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.offload.ltx2_int4_block_stream import LTX2Int4BlockStream
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward, ltx2_video_block_backward,
    LTX2VideoTrainForward, LTX2VideoBlockGrads,
)

comptime VD = 4096
comptime VIDEO_LORA_SLOTS = 8      # {to_q,to_k,to_v,to_out.0} x {attn1, attn2}


# ── shape helpers ────────────────────────────────────────────────────────────
def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^

def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^

def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


# The video-mode LoRA weight keys, flat order (block base = bi*len(names)).
# preset 0 = t2v (8 attention slots — the default, VIDEO_LORA_SLOTS); preset 1 =
# v2v (+2 video FFN slots = 10). FFN is appended AFTER the 8 attention slots so
# t2v flat indices are unchanged. Callers drive their slot loops off len(names)
# / a runtime n_slots — the comptime VIDEO_LORA_SLOTS (=8) stays the t2v default.
def video_lora_names(preset: Int = 0) -> List[String]:
    var out = List[String]()
    var mods = List[String]()
    mods.append("attn1")
    mods.append("attn2")
    var projs = List[String]()
    projs.append("to_q")
    projs.append("to_k")
    projs.append("to_v")
    projs.append("to_out.0")
    for ref m in mods:
        for ref p in projs:
            out.append(m + "." + p + ".weight")
    if preset == 1:  # PRESET_V2V (video FFN)
        out.append(String("ff.net.0.proj.weight"))
        out.append(String("ff.net.2.weight"))
    return out^


# ── streamed block source (one path for the F32 gate + the BF16 smoke) ───────
# Both open LTX2BlockStream and materialize each block as a full AV block via
# from_fp8_block (per-tensor dispatch: raw BF16 boundary blocks read verbatim,
# fp8 inner blocks dequant-to-bf16). `run_f32` upcasts for the exact F32 gate.
struct LTX2VideoBlockSource(Movable):
    var stream: LTX2BlockStream
    var int4: Optional[LTX2Int4BlockStream]   # when set, blocks come from the int4 slab
    var cfg: LTX2Config
    var run_f32: Bool

    def __init__(
        out self, var stream: LTX2BlockStream,
        var int4: Optional[LTX2Int4BlockStream], cfg: LTX2Config, run_f32: Bool
    ):
        self.stream = stream^
        self.int4 = int4^
        self.cfg = cfg
        self.run_f32 = run_f32

    @staticmethod
    def open(ckpt: String, cfg: LTX2Config, run_f32: Bool) raises -> LTX2VideoBlockSource:
        return LTX2VideoBlockSource(
            LTX2BlockStream.open(ckpt), Optional[LTX2Int4BlockStream](None), cfg, run_f32)

    @staticmethod
    def open_int4(
        fp8_ckpt: String, slab: String, cfg: LTX2Config, run_f32: Bool
    ) raises -> LTX2VideoBlockSource:
        """INT4 arm: blocks reconstructed bf16 from the svdint4 slab. The fp8
        checkpoint header is still opened (cheap mmap, no VRAM) for block_count
        parity; get_block dispatches to the int4 stream."""
        return LTX2VideoBlockSource(
            LTX2BlockStream.open(fp8_ckpt),
            Optional[LTX2Int4BlockStream](LTX2Int4BlockStream.open(slab)),
            cfg, run_f32)

    def block_count(self) -> Int:
        return self.stream.block_count()

    def get_block(self, i: Int, ctx: DeviceContext) raises -> LTX2AVBlockWeights:
        var w: LTX2AVBlockWeights
        if self.int4:
            var blk4 = self.int4.value().load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk4^, self.cfg, ctx)
        else:
            var blk = self.stream.load_block_bf16(i, ctx)
            w = LTX2AVBlockWeights.from_fp8_block(blk^, self.cfg, ctx)
        if self.run_f32:
            return w.to_f32(ctx)
        return w^


# Attach the video-mode adapters to block `bi` (flat A/B lists, bi*n_slots +
# slot). n_slots = len(names) — preset-derived (8 t2v / 10 v2v), NOT the comptime
# VIDEO_LORA_SLOTS, so the same code serves both presets.
def _attach_block_lora(
    mut w: LTX2AVBlockWeights, bi: Int, names: List[String],
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    scale: Float32,
) raises:
    var n_slots = len(names)
    var base = bi * n_slots
    for s in range(n_slots):
        w.add_lora_factor_arc(names[s], lora_a[base + s].copy(), lora_b[base + s].copy(), scale)


# ── frozen tail (musubi _process_output, F32 interior) ───────────────────────
struct LTX2VideoTail(Movable):
    var sst: Tensor        # [2,VD]  F32 (bf16-roundtripped)
    var proj_w: Tensor     # [128,VD] stack dtype
    var proj_b: Tensor     # [128]    stack dtype
    var ones: Tensor       # [VD] F32 (no-affine LN affine)
    var zeros: Tensor      # [VD] F32

    def __init__(
        out self, var sst: Tensor, var proj_w: Tensor, var proj_b: Tensor,
        var ones: Tensor, var zeros: Tensor,
    ):
        self.sst = sst^
        self.proj_w = proj_w^
        self.proj_b = proj_b^
        self.ones = ones^
        self.zeros = zeros^

    @staticmethod
    def load(ckpt: String, run_f32: Bool, ctx: DeviceContext) raises -> LTX2VideoTail:
        var st = ShardedSafeTensors.open(ckpt)
        # scale_shift_table is F32 [2,VD] on disk; bf16-roundtrip then F32 to
        # match the block-weight loader discipline (both sides round identically).
        var sst = cast_tensor(
            _load_top_bf16(st, "scale_shift_table", ctx), STDtype.F32, ctx)
        var pw = _load_top_bf16(st, "proj_out.weight", ctx)
        var pb = _load_top_bf16(st, "proj_out.bias", ctx)
        var proj_dt = STDtype.F32 if run_f32 else STDtype.BF16
        var proj_w = cast_tensor(pw, proj_dt, ctx)
        var proj_b = cast_tensor(pb, proj_dt, ctx)
        var od = proj_w.shape()
        var pdim = od[1]                 # VD (proj_out is [128, VD])
        var onesl = List[Float32]()
        var zerol = List[Float32]()
        for _ in range(pdim):
            onesl.append(Float32(1.0)); zerol.append(Float32(0.0))
        var ones = Tensor.from_host(onesl, _sh1(pdim), STDtype.F32, ctx)
        var zeros = Tensor.from_host(zerol, _sh1(pdim), STDtype.F32, ctx)
        return LTX2VideoTail(sst^, proj_w^, proj_b^, ones^, zeros^)


def _load_top_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var key = String("model.diffusion_model.") + name
    if not _st_has(st, key):
        key = name
    return Tensor.from_view_as_bf16(st.tensor_view(key), ctx)


def _st_has(st: ShardedSafeTensors, name: String) -> Bool:
    for ref nm in st.names():
        if nm == name:
            return True
    return False


# The cached `video_prompt_embeds` is POST-connector: the musubi
# Embeddings1DConnector runs at TE-CACHE time inside the text encoder
# (text_encoders/gemma/encoders/video_only_encoder.py::_run_connector), NOT in
# the DiT forward — so the DiT (and this stack) consumes the cached embeds
# DIRECTLY as the block context, and there is NO connector load/forward here.
# The connector replaces pad positions with learnable registers and re-emits a
# fully-valid mask, so maskless cross-attn (attn2's sdpa_cross_nomask) is
# faithful ONLY when `prompt_attention_mask` is all-ones. FAIL LOUD otherwise so
# a future cache with different connector semantics can never silently train
# maskless. (Measured 2026-07-09: all-ones for every sample in ltx2_musubi_v3.)
def assert_prompt_mask_all_ones(
    te_path: String, mask_name: String = "prompt_attention_mask"
) raises:
    var st = ShardedSafeTensors.open(te_path)
    var tv = st.tensor_view(mask_name)
    if tv.dtype != STDtype.I64:
        raise Error(String("assert_prompt_mask_all_ones: ") + mask_name + " not I64")
    var n = tv.numel()
    var bad = 0
    for i in range(n):
        var base = i * 8
        var ok = Int(tv.data[base]) == 1     # int64 LE == 1 <=> [1,0,0,0,0,0,0,0]
        if ok:
            for k in range(1, 8):
                if Int(tv.data[base + k]) != 0:
                    ok = False
                    break
        if not ok:
            bad += 1
    if bad != 0:
        raise Error(
            String("prompt_attention_mask NOT all-ones (") + String(bad) + "/"
            + String(n) + " positions != 1): the connector-register mask"
            + " semantics changed for this cache — maskless attn2 is no longer"
            + " faithful; add real cross-attn masking before training on it")


# tail forward: pred [1,S_V,128]. x is the last block output (stack dtype).
def _tail_forward[S_V: Int](
    x: Tensor, v_embedded: Tensor, tail: LTX2VideoTail, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var x32 = cast_tensor(x, STDtype.F32, ctx)                     # [1,S_V,VD]
    var emb32 = cast_tensor(v_embedded, STDtype.F32, ctx)          # [1,S_V,VD]
    var sst0 = reshape(slice(tail.sst, 0, 0, 1, ctx), _sh3(1, 1, VD), ctx)
    var sst1 = reshape(slice(tail.sst, 0, 1, 1, ctx), _sh3(1, 1, VD), ctx)
    var shift = add(sst0, emb32, ctx)                              # [1,S_V,VD]
    var scale = add(sst1, emb32, ctx)
    var normed = layer_norm(x32, tail.ones, tail.zeros, eps, ctx)  # F32 no-affine
    var x_mod = add(mul(normed, add_scalar(scale, Float32(1.0), ctx), ctx), shift, ctx)
    var x_out = cast_tensor(x_mod, tail.proj_w.dtype(), ctx)       # back to stack dtype
    return linear(x_out, tail.proj_w, Optional[Tensor](tail.proj_b.clone(ctx)), ctx)


# tail backward: d(last block output) [1,S_V,VD], stack dtype. scale/shift/proj
# grads discarded (frozen).
def _tail_backward[S_V: Int](
    d_pred: Tensor, x_last: Tensor, v_embedded: Tensor, tail: LTX2VideoTail,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var out_ch = d_pred.shape()[len(d_pred.shape()) - 1]
    var d_pred2 = reshape(d_pred, _sh2(S_V, out_ch), ctx)
    var d_x_out = linear_backward_dx(d_pred2, tail.proj_w, S_V, VD, out_ch, ctx)  # [S_V,VD] F32
    var d_x_mod = reshape(d_x_out, _sh3(1, S_V, VD), ctx)          # F32
    var emb32 = cast_tensor(v_embedded, STDtype.F32, ctx)
    var sst1 = reshape(slice(tail.sst, 0, 1, 1, ctx), _sh3(1, 1, VD), ctx)
    var scale = add(sst1, emb32, ctx)                              # F32
    var d_normed = mul(d_x_mod, add_scalar(scale, Float32(1.0), ctx), ctx)  # F32
    var x32 = cast_tensor(x_last, STDtype.F32, ctx)
    var d_x32 = layer_norm_backward_dx(d_normed, x32, tail.ones, eps, ctx)  # F32
    return cast_tensor(d_x32, x_last.dtype(), ctx)                 # stack dtype


# ── forward tape ─────────────────────────────────────────────────────────────
struct LTX2VideoStackForward(Movable):
    var pred: Tensor                        # [1,S_V,128] velocity (patchified)
    var saved_inputs: List[ArcPointer[Tensor]]  # per-block input [1,S_V,VD]
    var x_last: Tensor                      # last block output (tail input)
    # SAVE-ACTS arm: retained whole per-block train-forwards for blocks [0, K)
    # when save_acts_k>0 (else empty). The backward reads their .acts instead of
    # recomputing the forward — pure memory-for-compute (the forward already
    # computes+discards these). ArcPointer-boxed (LTX2VideoTrainForward is
    # move-only; List needs Copyable). Whole-fwd box avoids a partial-move of the
    # acts field; the extra video_out held (~2MB/block bf16) is negligible.
    var saved_fwds: List[ArcPointer[LTX2VideoTrainForward]]

    def __init__(
        out self, var pred: Tensor,
        var saved_inputs: List[ArcPointer[Tensor]], var x_last: Tensor,
        var saved_fwds: List[ArcPointer[LTX2VideoTrainForward]],
    ):
        self.pred = pred^
        self.saved_inputs = saved_inputs^
        self.x_last = x_last^
        self.saved_fwds = saved_fwds^


# ═════════════════════════════════════════════════════════════════════════════
# STACK FORWARD: head-outputs -> N streamed blocks (save block inputs) -> tail.
#   `hidden` is the post-patchify-proj block-0 input (head output). enc is the
#   POST-connector context. lora_a/lora_b are flat device tensors (num_layers*8),
#   in the block dtype (F32 gate / BF16 smoke).
# ═════════════════════════════════════════════════════════════════════════════
def ltx2_video_stack_lora_forward[S_V: Int, N_TXT: Int](
    hidden: Tensor, enc: Tensor,
    v_temb: Tensor, v_embedded: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    tail: LTX2VideoTail,
    src: LTX2VideoBlockSource,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    preset: Int = 0, save_acts_k: Int = 0,
) raises -> LTX2VideoStackForward:
    var names = video_lora_names(preset)
    var saved = List[ArcPointer[Tensor]]()
    var saved_fwds = List[ArcPointer[LTX2VideoTrainForward]]()  # blocks [0,K) retained
    var x = hidden.clone(ctx)
    for i in range(num_layers):
        saved.append(ArcPointer[Tensor](x.clone(ctx)))       # save block input
        var w = src.get_block(i, ctx)
        _attach_block_lora(w, i, names, lora_a, lora_b, lora_scale)
        var fwd = ltx2_video_block_train_forward[S_V, N_TXT](
            w, x, enc, v_temb, v_prompt_ts, v_cos, v_sin, eps, ctx,
        )
        x = fwd.video_out.clone(ctx)                          # keep output (borrow)
        if i < save_acts_k:
            saved_fwds.append(ArcPointer[LTX2VideoTrainForward](fwd^))  # retain WHOLE fwd
        # else: fwd (incl acts) drops at scope end (recomputed in backward)
    var x_last = x.clone(ctx)
    var pred = _tail_forward[S_V](x, v_embedded, tail, eps, ctx)
    return LTX2VideoStackForward(pred^, saved^, x_last^, saved_fwds^)


# ── backward result ──────────────────────────────────────────────────────────
struct LTX2VideoStackGrads(Movable):
    var d_a: List[List[Float32]]   # [num_layers*8][rank*in]
    var d_b: List[List[Float32]]   # [num_layers*8][out*rank]
    var d_input: List[Float32]     # grad into block-0 hidden (head frozen)
    var nonfinite: Int

    def __init__(
        out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]],
        var d_input: List[Float32], nonfinite: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_input = d_input^
        self.nonfinite = nonfinite


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


# ═════════════════════════════════════════════════════════════════════════════
# STACK BACKWARD (reverse block stream; recompute-in-backward per block).
#   d_pred -> tail backward -> for each block (reverse): re-stream weights,
#   RECOMPUTE the train forward (saving acts) on the saved block input, run the
#   block backward, chain d, collect the 8 LoRA grads. Returns d_input (grad into
#   block-0 hidden; head frozen so the chain stops here) + all LoRA d_A/d_B.
# ═════════════════════════════════════════════════════════════════════════════
def ltx2_video_stack_lora_backward[S_V: Int, N_TXT: Int](
    d_pred: Tensor, saved_inputs: List[ArcPointer[Tensor]], x_last: Tensor,
    enc: Tensor, v_temb: Tensor, v_embedded: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    tail: LTX2VideoTail,
    src: LTX2VideoBlockSource,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    preset: Int = 0,
    save_acts_k: Int = 0,
    saved_fwds: List[ArcPointer[LTX2VideoTrainForward]] = List[ArcPointer[LTX2VideoTrainForward]](),
) raises -> LTX2VideoStackGrads:
    # SAVE-ACTS: blocks [0, save_acts_k) consume the forward's retained acts
    # (skip the recompute train-forward); blocks [save_acts_k, num_layers)
    # recompute as before. save_acts_k=0 (default) => recompute all (byte-
    # identical to the pre-save-acts path — the gates keep passing untouched).
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
        var bg: LTX2VideoBlockGrads
        if bi < save_acts_k:
            # SAVE-ACTS: read the retained forward's acts (no recompute)
            bg = ltx2_video_block_backward[S_V, N_TXT](
                w, saved_fwds[bi][].acts, d_x, v_temb, v_cos, v_sin, eps, ctx,
            )
        else:
            # recompute the train forward on the saved block input (acts for bwd)
            var fwd = ltx2_video_block_train_forward[S_V, N_TXT](
                w, saved_inputs[bi][], enc, v_temb, v_prompt_ts, v_cos, v_sin, eps, ctx,
            )
            bg = ltx2_video_block_backward[S_V, N_TXT](
                w, fwd.acts, d_x, v_temb, v_cos, v_sin, eps, ctx,
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


# ═════════════════════════════════════════════════════════════════════════════
# FROZEN HEAD (used by the real-depth smoke; the parity gate loads head-outputs
# from the fixture). Mirrors the AV MVP spine head (ltx2_t2v_av_mvp.mojo).
# ═════════════════════════════════════════════════════════════════════════════
struct LTX2VideoHeadOut(Movable):
    var hidden: Tensor         # [1,S_V,VD]
    var v_temb: Tensor         # [1,S_V,9*VD]
    var v_embedded: Tensor     # [1,S_V,VD]
    var v_prompt_ts: Tensor    # [1,N_TXT,2*VD]
    var v_cos: Tensor          # [S_V*32, 64]
    var v_sin: Tensor

    def __init__(
        out self, var hidden: Tensor, var v_temb: Tensor, var v_embedded: Tensor,
        var v_prompt_ts: Tensor, var v_cos: Tensor, var v_sin: Tensor,
    ):
        self.hidden = hidden^
        self.v_temb = v_temb^
        self.v_embedded = v_embedded^
        self.v_prompt_ts = v_prompt_ts^
        self.v_cos = v_cos^
        self.v_sin = v_sin^


struct LTX2VideoStackHead(Movable):
    var w: Dict[String, ArcPointer[Tensor]]  # head weights (dt)
    var dt: STDtype                          # compute dtype (BF16 prod / F32 gate)

    def __init__(out self, var w: Dict[String, ArcPointer[Tensor]], dt: STDtype):
        self.w = w^
        self.dt = dt

    @staticmethod
    def load(
        ckpt: String, ctx: DeviceContext, run_f32: Bool = False
    ) raises -> LTX2VideoStackHead:
        var st = ShardedSafeTensors.open(ckpt)
        var dt = STDtype.F32 if run_f32 else STDtype.BF16
        var w = Dict[String, ArcPointer[Tensor]]()
        var keys = List[String]()
        keys.append("patchify_proj.weight")
        keys.append("patchify_proj.bias")
        for base in ["adaln_single", "prompt_adaln_single"]:
            keys.append(base + ".emb.timestep_embedder.linear_1.weight")
            keys.append(base + ".emb.timestep_embedder.linear_1.bias")
            keys.append(base + ".emb.timestep_embedder.linear_2.weight")
            keys.append(base + ".emb.timestep_embedder.linear_2.bias")
            keys.append(base + ".linear.weight")
            keys.append(base + ".linear.bias")
        for ref k in keys:
            var t = _load_top_bf16(st, k, ctx)     # BF16 (bf16-roundtripped)
            if run_f32:
                t = cast_tensor(t, STDtype.F32, ctx)
            w[k] = ArcPointer[Tensor](t^)
        return LTX2VideoStackHead(w^, dt)

    def _lin(self, x: Tensor, wk: String, bk: String, ctx: DeviceContext) raises -> Tensor:
        return linear(x, self.w[wk][], Optional[Tensor](self.w[bk][].clone(ctx)), ctx)

    # AdaLayerNormSingle: embedded = linear_2(silu(linear_1(sinusoidal(ts))));
    # mod = linear(silu(embedded)). ts already scaled (sigma*1000).
    def _adaln(
        self, base: String, ts_vals: List[Float32], ctx: DeviceContext
    ) raises -> Tuple[Tensor, Tensor]:
        var n = len(ts_vals)
        var emb = _timestep_embedding(ts_vals, 256, self.dt, ctx)  # [n,256]
        var h = silu(self._lin(
            emb, base + ".emb.timestep_embedder.linear_1.weight",
            base + ".emb.timestep_embedder.linear_1.bias", ctx), ctx)
        var embedded = self._lin(
            h, base + ".emb.timestep_embedder.linear_2.weight",
            base + ".emb.timestep_embedder.linear_2.bias", ctx)          # [n,VD]
        var mod = self._lin(
            silu(embedded, ctx), base + ".linear.weight", base + ".linear.bias", ctx)
        return (mod^, embedded^)

    def forward[S_V: Int, N_TXT: Int](
        self, latent: Tensor, enc: Tensor, sigma: Float32,
        nf: Int, nh: Int, nw: Int, frame_rate: Float64, ctx: DeviceContext,
    ) raises -> LTX2VideoHeadOut:
        # patchify: [1,C,NF,NH,NW] -> [1,C,S_V] -> [1,S_V,C]  (musubi p=(1,1,1))
        var lsh = latent.shape()
        var C = lsh[1]
        var latent_c = cast_tensor(latent, self.dt, ctx)   # to head compute dtype
        var vflat = permute(reshape(latent_c, _sh3(1, C, S_V), ctx), _perm021(), ctx)
        var hidden = self._lin(vflat, "patchify_proj.weight", "patchify_proj.bias", ctx)

        var ts_v = List[Float32]()
        for _ in range(S_V):
            ts_v.append(sigma * Float32(1000.0))
        var vt = self._adaln("adaln_single", ts_v, ctx)
        var v_temb = reshape(vt[0], _sh3(1, S_V, 9 * VD), ctx)
        var v_embedded = reshape(vt[1], _sh3(1, S_V, VD), ctx)

        var ts_p = List[Float32]()
        for _ in range(N_TXT):
            ts_p.append(sigma * Float32(1000.0))
        var pt = self._adaln("prompt_adaln_single", ts_p, ctx)
        var v_prompt_ts = reshape(pt[0], _sh3(1, N_TXT, 2 * VD), ctx)

        var rope = _build_video_rope(nf, nh, nw, frame_rate, hidden.dtype(), ctx)
        return LTX2VideoHeadOut(
            hidden^, v_temb^, v_embedded^, v_prompt_ts^,
            rope[0].clone(ctx), rope[1].clone(ctx),
        )

    # IC-LoRA / v2v head: reference latent (CLEAN) PREPENDED to the target latent
    # (noisy) on the token axis (musubi ltx2_train_network.py:3354-3356). Same
    # patchify_proj + adaln weights; the ONLY differences are (a) two patchify
    # calls concatenated ref-first, (b) per-token AdaLN ts = [0 x S_REF,
    # sigma*1000 x S_TGT] (musubi torch.where(cond_mask,0,sigma) :3376-3377), and
    # (c) the two-grid rope _build_v2v_rope. Inlined patchify (NOT shared with
    # forward) so the committed t2v head stays byte-identical.
    def forward_v2v[S_COMB: Int, S_REF: Int, S_TGT: Int, N_TXT: Int](
        self, ref_latent: Tensor, target_latent: Tensor, enc: Tensor, sigma: Float32,
        tnf: Int, tnh: Int, tnw: Int, rnf: Int, rnh: Int, rnw: Int, ds: Int,
        tgt_mask: List[Bool],   # P5.5: per-target-token conditioning mask (t=0 where 1)
        frame_rate: Float64, ctx: DeviceContext,
    ) raises -> LTX2VideoHeadOut:
        var rsh = ref_latent.shape()
        var C = rsh[1]
        # patchify reference (clean) -> [1,S_REF,VD]
        var ref_c = cast_tensor(ref_latent, self.dt, ctx)
        var ref_flat = permute(reshape(ref_c, _sh3(1, C, S_REF), ctx), _perm021(), ctx)
        var ref_h = self._lin(ref_flat, "patchify_proj.weight", "patchify_proj.bias", ctx)
        # patchify target (noisy) -> [1,S_TGT,VD]
        var tgt_c = cast_tensor(target_latent, self.dt, ctx)
        var tgt_flat = permute(reshape(tgt_c, _sh3(1, C, S_TGT), ctx), _perm021(), ctx)
        var tgt_h = self._lin(tgt_flat, "patchify_proj.weight", "patchify_proj.bias", ctx)
        var hidden = concat(1, ctx, ref_h, tgt_h)          # [1,S_COMB,VD] ref-prepended

        # per-token AdaLN timesteps: ref/clean -> 0, target -> sigma; intrinsic-
        # conditioned target tokens (tgt_mask==1) are clean too -> t=0 (official
        # flexible.py:516 timesteps=(1-mask)*timesteps). tgt_mask is [S_TGT]
        # (all-False when nothing fired).
        var ts_v = List[Float32]()
        for _ in range(S_REF):
            ts_v.append(Float32(0.0))
        var have_mask = len(tgt_mask) == S_TGT
        for t in range(S_TGT):
            if have_mask and tgt_mask[t]:
                ts_v.append(Float32(0.0))
            else:
                ts_v.append(sigma * Float32(1000.0))
        var vt = self._adaln("adaln_single", ts_v, ctx)
        var v_temb = reshape(vt[0], _sh3(1, S_COMB, 9 * VD), ctx)
        var v_embedded = reshape(vt[1], _sh3(1, S_COMB, VD), ctx)

        var ts_p = List[Float32]()
        for _ in range(N_TXT):
            ts_p.append(sigma * Float32(1000.0))
        var pt = self._adaln("prompt_adaln_single", ts_p, ctx)
        var v_prompt_ts = reshape(pt[0], _sh3(1, N_TXT, 2 * VD), ctx)

        var rope = _build_v2v_rope(
            rnf, rnh, rnw, tnf, tnh, tnw, ds, frame_rate, hidden.dtype(), ctx)
        return LTX2VideoHeadOut(
            hidden^, v_temb^, v_embedded^, v_prompt_ts^,
            rope[0].clone(ctx), rope[1].clone(ctx),
        )


def _perm021() -> List[Int]:
    var p = List[Int](); p.append(0); p.append(2); p.append(1); return p^


# sinusoidal timestep embedding (diffusers get_timestep_embedding,
# flip_sin_to_cos=True, downscale_freq_shift=0) — cos-first, matches the MVP.
def _timestep_embedding(
    ts: List[Float32], dim: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = len(ts)
    var half = dim // 2
    var out = List[Float32]()
    out.resize(n * dim, Float32(0.0))
    for r in range(n):
        var t = Float64(ts[r])
        for i in range(half):
            var freq = fpow(
                Float64(2.718281828459045), -Float64(i) * flog(10000.0) / Float64(half))
            var arg = t * freq
            out[r * dim + i] = Float32(fcos(arg))
            out[r * dim + half + i] = Float32(fsin(arg))
    return Tensor.from_host(out, _sh2(n, dim), dtype, ctx)


# 3D video RoPE for the [NF,NH,NW] grid at frame_rate — MUSUBI-EXACT dtype
# pipeline (VAE sf 8/32/32, causal offset 1, pos_embed_max_pos 20, base_hw
# 2048, theta 10000, ckpt config: rope_type=split, frequencies_precision=
# float64, use_middle=true). Returns halfsplit cos/sin [S_V*32, 64] in
# (token,head) row order.
#
# DTYPE CONTRACT (measured 2026-07-16, ltx2 fwd-parity gate): musubi's
# LTX2Wrapper casts positions to BF16 (`.to(video_latents.dtype)`,
# lora_ltx2.py:349) and divides the temporal axis by frame_rate IN BF16; the
# mid/frac/scale chain then runs in bf16 inside generate_freqs, and the final
# cos/sin are cast to the model dtype (bf16). The temporal `scaled` therefore
# quantizes at bf16 ulp (-0.998 -> -0.99609375) — invisible at F=1 (constant
# temporal phase cancels in q.k) but a REAL per-frame phase error at F>=2.
# The old f64-everywhere builder produced video-arm pred cos 0.9958-0.9991 vs
# musubi (bar 0.999); replicating the bf16 rounding is required for parity.
# Freq grid stays f64 theta^(i/(n-1))*pi/2 rounded to F32 (musubi np64 path).
comptime _POS_MAX = Float64(20.0)
comptime _BASE_HW = Float64(2048.0)
comptime _VAE_T = Float64(8.0)
comptime _VAE_H = Float64(32.0)
comptime _VAE_W = Float64(32.0)
comptime _CAUSAL = Float64(1.0)
comptime _THETA = Float64(10000.0)


def _bf16_rne(x: Float32) -> Float32:
    # round-to-nearest-even bf16 quantization, kept as F32 (torch bf16-op
    # semantics: compute, then round the result to bf16 per op)
    return BFloat16(x).cast[DType.float32]()


# Fill the [3,P,2] coord slab for ONE grid at token offset `tok_off`, mirroring
# _build_video_rope's per-token math. `hw_scale` multiplies the H/W pixel coords
# (=1 for a plain grid; = reference_downscale for a v2v ref grid, so ref & target
# co-locate under the fixed BASE_HW normalization). Temporal axis stays
# bf16-rounded after /frame_rate; H/W stay F32 here (bf16-rounded later inside
# _compute_rope) — identical to the original spine.
def _fill_grid_coords(
    mut coords: List[Float32], P: Int, tok_off: Int,
    nf: Int, nh: Int, nw: Int, fr32: Float32, hw_scale: Float32,
):
    for f in range(nf):
        for h in range(nh):
            for wv in range(nw):
                var tok = tok_off + f * nh * nw + h * nw + wv
                var fs = Float64(f) * _VAE_T
                var fe = Float64(f + 1) * _VAE_T
                var fsc = fs + _CAUSAL - _VAE_T
                var fec = fe + _CAUSAL - _VAE_T
                if fsc < 0.0: fsc = 0.0
                if fec < 0.0: fec = 0.0
                coords[(0 * P + tok) * 2 + 0] = _bf16_rne(Float32(fsc) / fr32)
                coords[(0 * P + tok) * 2 + 1] = _bf16_rne(Float32(fec) / fr32)
                coords[(1 * P + tok) * 2 + 0] = Float32(Float64(h) * _VAE_H) * hw_scale
                coords[(1 * P + tok) * 2 + 1] = Float32(Float64(h + 1) * _VAE_H) * hw_scale
                coords[(2 * P + tok) * 2 + 0] = Float32(Float64(wv) * _VAE_W) * hw_scale
                coords[(2 * P + tok) * 2 + 1] = Float32(Float64(wv + 1) * _VAE_W) * hw_scale


def _build_video_rope(
    nf: Int, nh: Int, nw: Int, frame_rate: Float64, dtype: STDtype, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    var P = nf * nh * nw
    var num_heads = 32
    var dim = VD
    # coords [3, P, 2] — integer pixel coords (bf16-exact), temporal axis
    # divided by frame_rate IN BF16 (musubi wrapper semantics).
    var coords = List[Float32]()
    coords.resize(3 * P * 2, Float32(0.0))
    var fr32 = Float32(frame_rate)
    _fill_grid_coords(coords, P, 0, nf, nh, nw, fr32, Float32(1.0))
    var max_pos = List[Float32]()
    max_pos.append(Float32(_POS_MAX))
    max_pos.append(Float32(_BASE_HW))
    max_pos.append(Float32(_BASE_HW))
    return _compute_rope(coords, 3, P, dim, max_pos, _THETA, num_heads, dtype, ctx)


# ── IC-LoRA / v2v two-grid RoPE (P5) ─────────────────────────────────────────
# Reference grid PREPENDED to the target grid on the token axis, ref H/W pixel
# coords *reference_downscale so both grids co-locate from origin 0 under the
# fixed BASE_HW=2048 normalization (musubi ltx2_train_network.py:3386-3424:
# ref_positions[:,1/2,...] *= reference_downscale_factor; combined = cat dim=2).
# Same spine convention as _build_video_rope (causal offset 1, temporal
# /frame_rate in bf16, fixed max_pos [20,2048,2048]); ONE _compute_rope over the
# combined [S_ref+S_target] sequence. Co-location is EXACT: ref token at ref-grid
# h maps to (h*32*ds)/2048 == a target token at target-grid (h*ds): (h*ds*32)/2048.
def _build_v2v_coords(
    ref_nf: Int, ref_nh: Int, ref_nw: Int,
    tgt_nf: Int, tgt_nh: Int, tgt_nw: Int,
    ds: Int, frame_rate: Float64,
) -> List[Float32]:
    var s_ref = ref_nf * ref_nh * ref_nw
    var s_tgt = tgt_nf * tgt_nh * tgt_nw
    var P = s_ref + s_tgt
    var coords = List[Float32]()
    coords.resize(3 * P * 2, Float32(0.0))
    var fr32 = Float32(frame_rate)
    # ref FIRST (prepended), H/W *ds; target after, H/W *1
    _fill_grid_coords(coords, P, 0, ref_nf, ref_nh, ref_nw, fr32, Float32(ds))
    _fill_grid_coords(coords, P, s_ref, tgt_nf, tgt_nh, tgt_nw, fr32, Float32(1.0))
    return coords^


def _build_v2v_rope(
    ref_nf: Int, ref_nh: Int, ref_nw: Int,
    tgt_nf: Int, tgt_nh: Int, tgt_nw: Int,
    ds: Int, frame_rate: Float64, dtype: STDtype, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var P = ref_nf * ref_nh * ref_nw + tgt_nf * tgt_nh * tgt_nw
    var coords = _build_v2v_coords(
        ref_nf, ref_nh, ref_nw, tgt_nf, tgt_nh, tgt_nw, ds, frame_rate)
    var max_pos = List[Float32]()
    max_pos.append(Float32(_POS_MAX))
    max_pos.append(Float32(_BASE_HW))
    max_pos.append(Float32(_BASE_HW))
    return _compute_rope(coords, 3, P, VD, max_pos, _THETA, 32, dtype, ctx)


def _compute_rope(
    coords: List[Float32], num_pos_dims: Int, P: Int, dim: Int,
    max_pos: List[Float32], theta: Float64, num_heads: Int,
    dtype: STDtype, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var num_rope_elems = num_pos_dims * 2
    var freq_count = dim // num_rope_elems
    var half_dim = dim // 2
    var rope_freqs = freq_count * num_pos_dims
    var pad = half_dim - rope_freqs
    if pad < 0:
        raise Error("rope: rope_freqs > half_dim")
    var denom = Float64(freq_count - 1)
    if denom < 1.0:
        denom = 1.0
    # musubi np64 grid: f64 theta^(i/(n-1)) * pi/2, stored as an f32 tensor
    var freq = List[Float32]()
    for i in range(freq_count):
        freq.append(Float32(fpow(theta, Float64(i) / denom) * pi / 2.0))
    var half_head = half_dim // num_heads
    var cos_tok = List[Float32]()
    var sin_tok = List[Float32]()
    cos_tok.resize(P * half_dim, Float32(0.0))
    sin_tok.resize(P * half_dim, Float32(0.0))
    for p in range(P):
        # mid/frac/scale run in BF16 per-op (the positions tensor is bf16 on
        # the musubi side; every torch op rounds its result to bf16)
        var scaled = List[Float32]()
        for d in range(num_pos_dims):
            var s0 = _bf16_rne(coords[(d * P + p) * 2 + 0])
            var e0 = _bf16_rne(coords[(d * P + p) * 2 + 1])
            var mid = _bf16_rne(_bf16_rne(s0 + e0) / Float32(2.0))
            var frac = _bf16_rne(mid / max_pos[d])
            scaled.append(_bf16_rne(_bf16_rne(frac * Float32(2.0)) - Float32(1.0)))
        var b = p * half_dim
        for q in range(pad):
            cos_tok[b + q] = Float32(1.0)
            sin_tok[b + q] = Float32(0.0)
        var off = b + pad
        for i in range(freq_count):
            for d in range(num_pos_dims):
                # f32 angle (torch: f32 grid x bf16 scaled -> f32), f32
                # cos/sin, then the final .to(bf16) table cast
                var a = Float64(scaled[d] * freq[i])
                cos_tok[off] = _bf16_rne(Float32(fcos(a)))
                sin_tok[off] = _bf16_rne(Float32(fsin(a)))
                off += 1
    var cos_rows = List[Float32]()
    var sin_rows = List[Float32]()
    for p in range(P):
        for h in range(num_heads):
            var src = p * half_dim + h * half_head
            for j in range(half_head):
                cos_rows.append(cos_tok[src + j])
                sin_rows.append(sin_tok[src + j])
    var sh = _sh2(P * num_heads, half_head)
    var cos_t = Tensor.from_host(cos_rows, sh.copy(), dtype, ctx)
    var sin_t = Tensor.from_host(sin_rows, sh^, dtype, ctx)
    return (cos_t^, sin_t^)


# ── unpatchify velocity [1,S_V,128] -> latent [1,128,NF,NH,NW] ───────────────
def unpatchify_video[S_V: Int](
    pred: Tensor, nf: Int, nh: Int, nw: Int, ctx: DeviceContext
) raises -> Tensor:
    var C = pred.shape()[2]                       # 128
    var t = permute(pred, _perm021(), ctx)        # [1,128,S_V]
    var sh = List[Int]()
    sh.append(1); sh.append(C); sh.append(nf); sh.append(nh); sh.append(nw)
    return reshape(t, sh^, ctx)
