# autograd_v2/acestep_block_graph.mojo — ACE-Step-1.5 DiT per-block LoRA backward,
# driven by the graph engine (Tier-3 T3.C). Records ONE DiT layer's forward
# op-for-op from the saved block INPUT x (the ONLY tracked activation leaf) through
# the record_* wrappers — matching acestep_dit.acestep_block0_forward exactly —
# then engine.execute drives the backward from the block output seeded with d_out.
# The 8 LoRA A/B pairs (self q/k/v/o + cross q/k/v/o) are ENGINE LEAVES: their
# device grads land in the execute() Dict keyed by the a_id/b_id, exactly the
# zimage device-leaf pattern (NOT krea2's host-list out-of-band path).
#
# Layout (zimage discipline): the token dim runs FLAT [S, D] 2D; attention
# reshapes to [1,S,H,Dh] only inside the attn sections. This makes every proj/
# linear backward's [M, in] output shape-consistent with its fan-in siblings — no
# [1,S,D]-vs-[S,D] ambiguity in the leaf grad folds.
#
# What differs from the zimage/krea2 templates (and needs the two new op kinds):
#  * RoPE = Qwen3 HALFSPLIT (record_rope_halfsplit → OPK_ROPE_HALFSPLIT), NOT the
#    interleaved OPK_ROPE (the FLUX/Klein pairing). Wrong variant silently aliases
#    the wrong angle (rope_struct_backward docstring).
#  * The MLP (gate/up/down) is FROZEN (LoRA targets only q/k/v/o). record_linear_dx
#    (→ OPK_LINEAR_DX) routes d_x through it so x1's grad — hence every upstream
#    cross/self LoRA grad — is complete. Its base d_w is never materialized.
#  * Cross-attn: enc = FROZEN leaf (unregistered id → null edge → d_enc dropped);
#    q from x1, k/v from enc; NO rope; plain residual ADD (not gated).
#  * GQA (KVHEADS<HEADS) → record_repeat_kv on k/v, both self and cross.
#
# GATE path (this version): S==L (the block-0 oracle is SP=64/L=64) and S<=window
# so every layer's self-attn is sdpa_nomask (block-0 is "sliding" but S<=128 →
# mask is a no-op == full). The general L≠S padded-cross and SP>window masked-
# sliding backward are documented follow-ons (raise below).
#
# Frozen (untracked, null edges, C7): all base weights (q/k/v/o/mlp), the norm
# weights, the qk-norm weights, the 6 AdaLN mod chunks (sst+temb, computed OUTSIDE
# the graph), the rope cos/sin tables, and enc. The 8 LoRA A/B are the tracked
# leaves; x is the tracked block input (its accumulated grad = the returned d_x).
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.node import TArc, arc_view
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute
from serenitymojo.autograd_v2.ops_record import (
    record_proj_lora,
    record_rms_norm_dx,
    record_modulate,
    record_rope_halfsplit,
    record_sdpa,
    record_swiglu,
    record_residual_gate,
    record_reshape,
    record_add,
    record_linear_dx,
    record_repeat_kv,
)
from serenitymojo.models.zimage.lora_block import ZImageLoraAdapterDevice
from serenitymojo.models.dit.acestep_dit import (
    AceStepDiTConfig,
    _add as _ace_add,
    _mod_chunk,
    _tile_rows,
    _build_rope,
    _time_embed,
    _conv1d_patch,
    _conv_transpose1d_patch,
    _layer_bw,
    _w as _ace_w,
    acestep_block0_forward,
)
from serenitymojo.ops.linear import linear as _lin
from serenitymojo.ops.norm import rms_norm as _rms
from serenitymojo.ops.elementwise import modulate as _modulate
from serenitymojo.ops.tensor_algebra import (
    concat as _concat, reshape as _reshape, transpose as _transpose,
    sub as _sub, mul_scalar as _mul_scalar, mul as _ta_mul,
)
from serenitymojo.ops.reduce import reduce_mean_f32 as _reduce_mean_f32
from serenitymojo.ops.elementwise_backward import modulate_backward as _modulate_bwd
from serenitymojo.ops.norm_backward import rms_norm_backward_dx as _rms_bwd_dx
from serenitymojo.ops.linalg_backward import linear_backward_dx as _lin_bwd_dx


struct AcestepBlockLoraGrads(Movable):
    """One DiT layer's LoRA backward result. d_x = block-input grad; d_a/d_b are
    the 8 adapter grads in slot order [self_q, self_k, self_v, self_o, cross_q,
    cross_k, cross_v, cross_o] (matching the oracle's per-proj grad_* keys)."""

    var d_x: TArc
    var d_a: List[TArc]
    var d_b: List[TArc]

    def __init__(out self, var d_x: TArc, var d_a: List[TArc], var d_b: List[TArc]):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def _adapter(
    bw: Dict[String, ArcPointer[Tensor]], key: String, scale: Float32,
    ctx: DeviceContext,
) raises -> ZImageLoraAdapterDevice:
    """Build the device LoRA adapter for one projection from the per-layer bw
    dict (relative keys "<attn>.<proj>.lora_A"/".lora_B"). A=[rank,in], B=[out,
    rank] (the acestep_lora_fwd_gate storage, == zimage_lora_apply_device)."""
    var a = bw[key + ".lora_A"][].clone(ctx)
    var b = bw[key + ".lora_B"][].clone(ctx)
    var rank = a.shape()[0]
    var in_f = a.shape()[1]
    var out_f = b.shape()[0]
    return ZImageLoraAdapterDevice(TArc(a^), TArc(b^), rank, in_f, out_f, scale)


def acestep_block_lora_graph_backward[
    S: Int, L: Int, NH: Int
](
    d_out: Tensor,           # [1,S,H] or [S,H] — grad into the block output
    x_in: TArc,              # [1,S,H] — the block input (tracked leaf)
    temb: Tensor,            # [1,6,H] — timestep_proj
    enc: Tensor,             # [1,L,H] — condition_embedder output (frozen)
    rope_cos: Tensor,        # [S, Dh/2] — base half-width rope cos (frozen)
    rope_sin: Tensor,        # [S, Dh/2]
    bw: Dict[String, ArcPointer[Tensor]],   # per-layer weights + A/B (relative keys)
    cfg: AceStepDiTConfig,
    lora_scale: Float32,
    ctx: DeviceContext,
) raises -> AcestepBlockLoraGrads:
    """Graph-engine per-block LoRA backward for one ACE-Step DiT layer. Records
    the layer forward (acestep_block0_forward) op-for-op in flat [S,D] token
    space, executes the backward seeded with d_out, returns d_x + the 8 LoRA A/B
    device grads. GATE version: S==L, S<=sliding_window (self-attn = sdpa_nomask
    everywhere). General L≠S padded-cross is a follow-on.

    rope_cos/rope_sin are the BASE [S, Dh/2] half-width tables (tiled per-head
    inside); pass the REFERENCE's own rope (the oracle's captured cos/sin first
    half) — the in-tree `_build_rope` formula is only forward-parity (~0.065 off
    the reference table), which the roped self-q backward is sensitive to."""
    if S != L:
        raise Error(
            "acestep_block_lora_graph_backward: S != L (padded cross-attn) is a"
            " LATER phase; this arm is the S==L block-0 gate path (SP=64,L=64)"
        )
    var h = cfg.hidden_size
    var nh = cfg.num_heads
    var nkv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var nrep = nh // nkv
    var eps = cfg.rms_norm_eps
    comptime DH = 128                          # head_dim (matches the forward's sdpa[..,128])
    var scale = Float32(1.0) / sqrt(Float32(dh))
    var g = Graph()

    # ── frozen AdaLN mods (sst + temb → 6 chunks), computed OUTSIDE the graph ──
    var mod6 = _ace_add(bw["scale_shift_table"][], temb, ctx)   # [1,6,H]
    var shift_msa = TArc(_mod_chunk(mod6, 0, h, ctx))
    var scale_msa = TArc(_mod_chunk(mod6, 1, h, ctx))
    var gate_msa = TArc(_mod_chunk(mod6, 2, h, ctx))
    var c_shift = TArc(_mod_chunk(mod6, 3, h, ctx))
    var c_scale = TArc(_mod_chunk(mod6, 4, h, ctx))
    var c_gate = TArc(_mod_chunk(mod6, 5, h, ctx))

    # ── frozen rope tables (tiled per-head, like the forward's _self_attn) ────
    var cos_q = TArc(_tile_rows(rope_cos, S, nh, dh // 2, ctx))   # [S*nh, dh/2]
    var sin_q = TArc(_tile_rows(rope_sin, S, nh, dh // 2, ctx))
    var cos_k = TArc(_tile_rows(rope_cos, S, nkv, dh // 2, ctx))  # [S*nkv, dh/2]
    var sin_k = TArc(_tile_rows(rope_sin, S, nkv, dh // 2, ctx))

    # ── tracked block input x, flat [S,H] (its accumulated grad = returned d_x)
    var x_t = Tensor(x_in[].buf.copy(), [S, h], x_in[].dtype())
    x_t.set_id(g.fresh_tensor_id())
    var x_id = x_t.id
    _ = g.leaf(x_id)
    var x = TArc(x_t^)

    # frozen enc leaf, flat [L,H] (unregistered id → null edge → d_enc dropped).
    var enc_t = Tensor(enc.buf.copy(), [L, h], enc.dtype())
    enc_t.set_id(g.fresh_tensor_id())
    var enc_x = TArc(enc_t^)

    # 8 adapter leaves, slot order [sq,sk,sv,so,cq,ck,cv,co].
    var a_ids = List[Int]()
    var b_ids = List[Int]()
    for _ in range(8):
        a_ids.append(g.fresh_tensor_id())
        b_ids.append(g.fresh_tensor_id())

    var sq = _adapter(bw, "self_attn.q_proj", lora_scale, ctx)
    var sk = _adapter(bw, "self_attn.k_proj", lora_scale, ctx)
    var sv = _adapter(bw, "self_attn.v_proj", lora_scale, ctx)
    var so = _adapter(bw, "self_attn.o_proj", lora_scale, ctx)
    var cq = _adapter(bw, "cross_attn.q_proj", lora_scale, ctx)
    var ck = _adapter(bw, "cross_attn.k_proj", lora_scale, ctx)
    var cv = _adapter(bw, "cross_attn.v_proj", lora_scale, ctx)
    var co = _adapter(bw, "cross_attn.o_proj", lora_scale, ctx)

    # ── SELF-ATTENTION (AdaLN modulate → q/k/v LoRA → qk-norm → rope → GQA →
    #    sdpa → o LoRA → gated residual) ────────────────────────────────────────
    var x_norm = record_rms_norm_dx(g, x, bw["self_attn_norm.weight"].copy(), eps, ctx)
    var norm_hs = record_modulate(g, x_norm, scale_msa, shift_msa, 0, ctx)

    var q = record_proj_lora(g, norm_hs, bw["self_attn.q_proj.weight"].copy(), sq, a_ids[0], b_ids[0], S, h, nh * dh, ctx)
    var k = record_proj_lora(g, norm_hs, bw["self_attn.k_proj.weight"].copy(), sk, a_ids[1], b_ids[1], S, h, nkv * dh, ctx)
    var v = record_proj_lora(g, norm_hs, bw["self_attn.v_proj.weight"].copy(), sv, a_ids[2], b_ids[2], S, h, nkv * dh, ctx)

    var q4 = record_reshape(g, q, [1, S, nh, dh], ctx)
    var k4 = record_reshape(g, k, [1, S, nkv, dh], ctx)
    var v4 = record_reshape(g, v, [1, S, nkv, dh], ctx)

    var qn = record_rms_norm_dx(g, q4, bw["self_attn.q_norm.weight"].copy(), eps, ctx)
    var kn = record_rms_norm_dx(g, k4, bw["self_attn.k_norm.weight"].copy(), eps, ctx)
    var qr = record_rope_halfsplit(g, qn, cos_q, sin_q, ctx)
    var kr = record_rope_halfsplit(g, kn, cos_k, sin_k, ctx)

    var kf = record_repeat_kv(g,kr, S, nkv, nrep, dh, ctx)
    var vf = record_repeat_kv(g,v4, S, nkv, nrep, dh, ctx)
    var att = record_sdpa[1, S, NH, DH](g, qr, kf, vf, scale, ctx)
    var flat = record_reshape(g, att, [S, nh * dh], ctx)
    var ao = record_proj_lora(g, flat, bw["self_attn.o_proj.weight"].copy(), so, a_ids[3], b_ids[3], S, nh * dh, h, ctx)
    var x1 = record_residual_gate(g, x, gate_msa, ao, ctx)

    # ── CROSS-ATTENTION (q from x1; k/v from frozen enc; NO rope; plain add) ──
    var cnorm = record_rms_norm_dx(g, x1, bw["cross_attn_norm.weight"].copy(), eps, ctx)
    var cqp = record_proj_lora(g, cnorm, bw["cross_attn.q_proj.weight"].copy(), cq, a_ids[4], b_ids[4], S, h, nh * dh, ctx)
    var ckp = record_proj_lora(g, enc_x, bw["cross_attn.k_proj.weight"].copy(), ck, a_ids[5], b_ids[5], L, h, nkv * dh, ctx)
    var cvp = record_proj_lora(g, enc_x, bw["cross_attn.v_proj.weight"].copy(), cv, a_ids[6], b_ids[6], L, h, nkv * dh, ctx)

    var cq4 = record_reshape(g, cqp, [1, S, nh, dh], ctx)
    var ck4 = record_reshape(g, ckp, [1, L, nkv, dh], ctx)
    var cv4 = record_reshape(g, cvp, [1, L, nkv, dh], ctx)
    var cqn = record_rms_norm_dx(g, cq4, bw["cross_attn.q_norm.weight"].copy(), eps, ctx)
    var ckn = record_rms_norm_dx(g, ck4, bw["cross_attn.k_norm.weight"].copy(), eps, ctx)

    var ckf = record_repeat_kv(g,ckn, L, nkv, nrep, dh, ctx)
    var cvf = record_repeat_kv(g,cv4, L, nkv, nrep, dh, ctx)
    var catt = record_sdpa[1, S, NH, DH](g, cqn, ckf, cvf, scale, ctx)   # S==L
    var cflat = record_reshape(g, catt, [S, nh * dh], ctx)
    var cop = record_proj_lora(g, cflat, bw["cross_attn.o_proj.weight"].copy(), co, a_ids[7], b_ids[7], S, nh * dh, h, ctx)
    var x2 = record_add(g, x1, cop, ctx)

    # ── SwiGLU MLP (AdaLN modulate → frozen gate/up → swiglu → frozen down →
    #    gated residual) ──────────────────────────────────────────────────────
    var inter = cfg.intermediate
    var mnorm = record_rms_norm_dx(g, x2, bw["mlp_norm.weight"].copy(), eps, ctx)
    var mi = record_modulate(g, mnorm, c_scale, c_shift, 0, ctx)
    var g_pre = record_linear_dx(g, mi, bw["mlp.gate_proj.weight"].copy(), S, h, inter, ctx)
    var u = record_linear_dx(g, mi, bw["mlp.up_proj.weight"].copy(), S, h, inter, ctx)
    var act = record_swiglu(g, g_pre, u, ctx)
    var md = record_linear_dx(g, act, bw["mlp.down_proj.weight"].copy(), S, inter, h, ctx)
    var x3 = record_residual_gate(g, x2, c_gate, md, ctx)

    # ── engine backward from the block output, seeded with d_out ─────────────
    # cast d_out to the acts dtype (bf16) so the whole backward runs bf16 (the
    # training path); the oracle grads are F32-upcast bf16 values → cosine gate.
    var d_out_flat = Tensor(d_out.buf.copy(), [S, h], d_out.dtype())
    var root_idx = g.node_of_tensor[x3[].id]
    var grads = execute(g, root_idx, arc_view(d_out_flat), ctx)

    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for s in range(8):
        d_a.append(grads[a_ids[s]].copy())
        d_b.append(grads[b_ids[s]].copy())
    return AcestepBlockLoraGrads(grads[x_id].copy(), d_a^, d_b^)


# ══════════════════════════════════════════════════════════════════════════════
# FULL-STACK LoRA backward: forward conductor (saves per-block inputs) → MSE loss
# grad → proj_out ConvTranspose1d bwd → final-AdaLN bwd → 32× block bwd (routing
# d_x) → 512 LoRA A/B grads. Non-block parts (proj_in, condition_embedder,
# time_embed, proj_out, final-AdaLN) are FROZEN — the backward only routes d_x
# through them (proj_out/final-AdaLN via their frozen backward ops; proj_in/
# cond_emb/embeds contribute nothing — d_x into block 0's input is dropped).
# ══════════════════════════════════════════════════════════════════════════════


struct AcestepStackLoraGrads(Movable):
    """All 512 LoRA grads, layer-major: index = layer*8 + slot, slot order
    [self q,k,v,o, cross q,k,v,o] (matching the per-block backward). d_a[i]/d_b[i]
    are the A/B grads for that (layer, slot)."""

    var d_a: List[TArc]      # 32*8 = 256
    var d_b: List[TArc]      # 256
    var d_block0_out: TArc   # grad into block-0's OUTPUT (== oracle block0_d_out)
    var loss: Float32        # MSE(out0, flow) — the training-step scalar

    def __init__(out self, var d_a: List[TArc], var d_b: List[TArc], var d_block0_out: TArc, loss: Float32):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_block0_out = d_block0_out^
        self.loss = loss


def acestep_stack_lora_graph_backward[
    SP: Int, L: Int, NH: Int, LAYERS: Int
](
    xt: Tensor, context: Tensor, enc_in: Tensor,
    timestep: Float32, timestep_r: Float32,
    flow: Tensor,            # MSE target (x1 - x0), out0's shape [1,T,acoustic]
    full: Dict[String, ArcPointer[Tensor]],   # all decoder.* weights + 512 A/B
    cfg: AceStepDiTConfig,
    lora_scale: Float32,
    ctx: DeviceContext,
) raises -> AcestepStackLoraGrads:
    """Full-stack DiT LoRA backward. Runs the forward (saving each block's input
    hidden), seeds from d(MSE loss)/d(out0), backprops through proj_out +
    final-AdaLN (frozen), then loops the per-block backward 31→0 routing d_x, and
    returns the 512 LoRA A/B grads. Also returns d into block-0's output (==
    oracle block0_d_out) as an intermediate check. SP<=window (all sdpa_nomask)
    and SP*patch==T (no crop)."""
    var h = cfg.hidden_size
    var eps = cfg.rms_norm_eps
    var ps = cfg.patch_size
    var t = xt.shape()[1]                 # T (== SP*ps)
    var acoustic = cfg.acoustic_dim
    var window = cfg.sliding_window
    var dh = cfg.head_dim

    # ── timestep embeddings: temb [1,H] (final AdaLN), timestep_proj [1,6,H] (blocks)
    var d0 = Tensor.from_host([Float32(0.0)], [1], STDtype.BF16, ctx)
    var temb_t = d0.clone(ctx); var proj_t = d0.clone(ctx)
    var temb_r = d0.clone(ctx); var proj_r = d0.clone(ctx)
    _time_embed(timestep, "decoder.time_embed", full, cfg, ctx, temb_t, proj_t)
    _time_embed(timestep - timestep_r, "decoder.time_embed_r", full, cfg, ctx, temb_r, proj_r)
    var temb = _ace_add(temb_t, temb_r, ctx)              # [1,H]
    var timestep_proj = _ace_add(proj_t, proj_r, ctx)     # [1,6,H]

    # ── proj_in (frozen): cat(context, xt) → conv1d patch → [1,SP,H]
    var xin = _concat(2, ctx, context, xt)                # [1,T,in_channels]
    var x = _conv1d_patch(xin, _ace_w(full, "decoder.proj_in.1.weight", ctx),
                          _ace_w(full, "decoder.proj_in.1.bias", ctx),
                          t, cfg.in_channels, h, ps, ctx)  # [1,SP,H]

    # ── condition_embedder (frozen): enc [1,L,H]
    var enc = _lin(enc_in, _ace_w(full, "decoder.condition_embedder.weight", ctx),
                   Optional[Tensor](_ace_w(full, "decoder.condition_embedder.bias", ctx)), ctx)

    # ── rope base tables [SP, dh/2]
    var rope_cos = d0.clone(ctx); var rope_sin = d0.clone(ctx)
    _build_rope(SP, dh, cfg.rope_theta, ctx, rope_cos, rope_sin)

    # ── forward through the 32 blocks, SAVING each block's INPUT hidden ────────
    var block_inputs = List[TArc]()
    for li in range(LAYERS):
        block_inputs.append(TArc(x.clone(ctx)))           # input to block li
        var lbw = _layer_bw(full, li, ctx)
        var ltype = 1 if ((li + 1) % 2 == 1) else 0
        x = acestep_block0_forward[SP, L, NH](
            x, timestep_proj, enc, rope_cos, rope_sin, lbw, cfg, ltype, window,
            ctx, lora_scale,
        )
    # x = block-31 output (input to the final AdaLN)

    # ── final AdaLN (frozen): shift,scale = (scale_shift_table[1,2,H] + temb).chunk2
    var out_sst = _ace_w(full, "decoder.scale_shift_table", ctx)   # [1,2,H]
    var temb3 = _reshape(temb, [1, 1, h], ctx)
    var temb2 = _concat(1, ctx, temb3, temb3)                      # [1,2,H]
    var sst_t = _ace_add(out_sst, temb2, ctx)
    var f_shift = _mod_chunk(sst_t, 0, h, ctx)                     # [H]
    var f_scale = _mod_chunk(sst_t, 1, h, ctx)
    var xn = _rms(x, _ace_w(full, "decoder.norm_out.weight", ctx), eps, ctx)  # [1,SP,H]
    var xmod = _modulate(xn, f_scale, f_shift, ctx)               # [1,SP,H]

    # ── proj_out (frozen ConvTranspose1d): [1,SP,H] → [1,T,acoustic] = out0
    var out0 = _conv_transpose1d_patch(
        xmod, _ace_w(full, "decoder.proj_out.1.weight", ctx),
        _ace_w(full, "decoder.proj_out.1.bias", ctx), SP, h, acoustic, ps, ctx,
    )  # [1,T,acoustic] (SP*ps==T → no crop)

    # ── loss + grad: loss = mean((out0-flow)^2) → d_out0 = (2/N)(out0-flow) ────
    var diff = _sub(out0, flow, ctx)                       # [1,T,acoustic]
    var n_elem = Float32(t * acoustic)
    var d_out0 = _mul_scalar(diff, Float32(2.0) / n_elem, ctx)
    # loss scalar (F32-accumulated mean of the bf16 squared error, ~torch mse_loss).
    var sq = _ta_mul(diff, diff, ctx)
    var loss_t = _reduce_mean_f32(_reshape(sq, [t * acoustic], ctx), [0], False, ctx)
    var loss_val = loss_t.to_host(ctx)[0]

    # ── proj_out backward (frozen): rebuild w2d_t = [ps*acoustic, H], dx only ──
    var w_po = _ace_w(full, "decoder.proj_out.1.weight", ctx)     # [Cin=H, Cout=acoustic, ps]
    var wperm = _transpose(w_po, 1, 2, ctx)                       # [H, ps, acoustic]
    var w2d = _reshape(wperm, [h, ps * acoustic], ctx)           # [H, ps*acoustic]
    var w2d_t = _transpose(w2d, 0, 1, ctx)                       # [ps*acoustic, H] = [out,in]
    var d_out_lin = _reshape(d_out0, [SP, ps * acoustic], ctx)   # bias frozen; reshape T→SP
    var d_xmod = _lin_bwd_dx(d_out_lin, w2d_t, SP, h, ps * acoustic, ctx)  # [SP,H]

    # ── final-AdaLN backward (frozen scale): modulate then rms_norm ───────────
    var mb = _modulate_bwd(d_xmod, _reshape(xn, [SP, h], ctx), f_scale, ctx, compute_param_grads=False)
    var d_x_final = _rms_bwd_dx(mb.d_x, _reshape(x, [SP, h], ctx),
                               _ace_w(full, "decoder.norm_out.weight", ctx), eps, ctx)  # [SP,H]

    # ── loop the per-block backward 31→0, routing d_x, collecting 512 grads ────
    var d_a_all = List[TArc]()
    var d_b_all = List[TArc]()
    for _ in range(LAYERS * 8):
        d_a_all.append(TArc(d0.clone(ctx)))   # placeholders (filled by layer index)
        d_b_all.append(TArc(d0.clone(ctx)))
    var d = TArc(d_x_final^)                   # grad into block-31's OUTPUT
    var d_block0_out = TArc(d0.clone(ctx))
    for rev in range(LAYERS):
        var li = LAYERS - 1 - rev
        if li == 0:
            d_block0_out = d.copy()            # == oracle block0_d_out (intermediate check)
        var lbw = _layer_bw(full, li, ctx)
        var bg = acestep_block_lora_graph_backward[SP, L, NH](
            d[], block_inputs[li], timestep_proj, enc, rope_cos, rope_sin,
            lbw, cfg, lora_scale, ctx,
        )
        for s in range(8):
            d_a_all[li * 8 + s] = bg.d_a[s].copy()
            d_b_all[li * 8 + s] = bg.d_b[s].copy()
        d = bg.d_x.copy()                      # grad into block li's INPUT = block (li-1)'s OUTPUT

    return AcestepStackLoraGrads(d_a_all^, d_b_all^, d_block0_out^, loss_val)
