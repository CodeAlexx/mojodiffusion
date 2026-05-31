# pipeline/pid_net_failclosed_smoke.mojo — SKEPTIC fail-closed proof.
#
# Re-runs the assembled PidNet forward but deliberately CORRUPTS the LQ
# injection wiring: at each injection point it uses output-head (oidx+1)%7 and
# gate-module (oidx+1)%7 instead of oidx. Every op is identical to the verified
# net; only the LQ->block mapping is wrong. If the parity gate is genuinely
# fail-closed, the net velocity cos must DROP well below 0.999 vs the golden
# reference (which used the correct mapping). A gate that still "PASSES" here is
# not measuring what it claims to measure.
#
# Run: pixi run mojo run -I . serenitymojo/pipeline/pid_net_failclosed_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import add, reshape, slice
from serenitymojo.models.pid.pid_ops import patchify, unpatchify, timestep_conditioner
from serenitymojo.models.pid.pixeldit_block import (
    MMDiTBlockWeights, mmdit_block_forward_textrope,
)
from serenitymojo.models.pid.pit_block import PiTBlockWeights, pit_block_forward
from serenitymojo.models.pid.pid_net import (
    LQProj, GateWeights, _ld, _clone, _apply_gate, _nearest_upsample_nchw,
    _broadcast_t, _nchw_to_nhwc, _add_pixpos, _pixelize, _final_reorder,
    _load_mmdit_block, _load_pit_block, _scalar_f32,
)


comptime CKPT = "/home/alex/.serenity/models/pid/checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.safetensors"
comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/pid/parity/pid_net_ref.safetensors"


def _ldr(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx)


# Faulted forward: LQ output-head + gate-module index shifted by +1 (mod 7).
def _faulted_forward[
    B: Int, H: Int, W: Int, PH: Int, PW: Int, L: Int, LTXT: Int, ZH: Int, ZW: Int
](
    st: ShardedSafeTensors, x: Tensor, t: Tensor, y: Tensor, lq_latent: Tensor,
    sigma: Float32, pix_pos: Tensor, img_cos: Tensor, img_sin: Tensor,
    txt_cos: Tensor, txt_sin: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    comptime HID = 1536
    comptime GROUPS = 24
    comptime HEAD_DIM = 64
    comptime PATCH_DEPTH = 14
    comptime PIXEL_DEPTH = 2
    comptime PIXEL_GROUPS = 16
    comptime PIXEL_HEAD = 72
    comptime PIXEL_DIM = 16
    comptime ATTN_DIM = 1152
    comptime LQ_HID = 512
    comptime PS = 16
    comptime P2 = PS * PS
    comptime INTERVAL = 2
    comptime NUM_HEADS = 7

    var lq = LQProj(st, ctx)
    var lat_up = _nearest_upsample_nchw(lq_latent, B, 16, ZH, ZW, PH, PW, ctx)
    var lq_tokens = lq.tokens[B, PH, PW, LQ_HID](lat_up, ctx)

    var x_patches = patchify(x, PS, ctx)
    var semb_w = _ld(st, "s_embedder.proj.weight", ctx)
    var semb_b = _ld(st, "s_embedder.proj.bias", ctx)
    var s = linear(x_patches, semb_w, Optional[Tensor](_clone(semb_b, ctx)), ctx)

    var tm0_w = _ld(st, "t_embedder.mlp.0.weight", ctx)
    var tm0_b = _ld(st, "t_embedder.mlp.0.bias", ctx)
    var tm2_w = _ld(st, "t_embedder.mlp.2.weight", ctx)
    var tm2_b = _ld(st, "t_embedder.mlp.2.bias", ctx)
    var t_emb = timestep_conditioner(t, tm0_w, tm0_b, tm2_w, tm2_b, 256, ctx)
    var t_emb_3 = reshape(t_emb, [B, 1, HID], ctx)
    var condition = silu(t_emb_3, ctx)

    var ye_w = _ld(st, "y_embedder.proj.weight", ctx)
    var ye_b = _ld(st, "y_embedder.proj.bias", ctx)
    var ye_n = _ld(st, "y_embedder.norm.weight", ctx)
    var y_lin = linear(y, ye_w, Optional[Tensor](_clone(ye_b, ctx)), ctx)
    var y_normed = rms_norm(y_lin, ye_n, Float32(1e-6), ctx)
    var ypos_full = _ld(st, "y_pos_embedding", ctx)
    var ypos = slice(ypos_full, 1, 0, LTXT, ctx)
    var y_emb = add(y_normed, ypos, ctx)

    var s_cur = s^
    var y_cur = y_emb^
    for i in range(PATCH_DEPTH):
        if i % INTERVAL == 0:
            var oidx = i // INTERVAL
            var fidx = (oidx + 1) % NUM_HEADS   # ← FAULT: shift the LQ wiring
            var hp = String("lq_proj.output_heads.") + String(fidx) + "."
            var head_w = _ld(st, hp + "weight", ctx)
            var head_b = _ld(st, hp + "bias", ctx)
            var lq_feat = linear(lq_tokens, head_w, Optional[Tensor](_clone(head_b, ctx)), ctx)
            var gp = String("lq_proj.gate_modules.") + String(fidx) + "."
            var gw = GateWeights(
                _ld(st, gp + "content_proj.weight", ctx),
                _ld(st, gp + "content_proj.bias", ctx),
                _scalar_f32(st, gp + "log_alpha", ctx),
            )
            s_cur = _apply_gate(s_cur, lq_feat, gw, sigma, ctx)
        var bw = _load_mmdit_block(st, i, HID, ctx)
        var pair = mmdit_block_forward_textrope[GROUPS, HEAD_DIM, L, LTXT](
            s_cur, y_cur, condition, img_cos, img_sin, txt_cos, txt_sin, bw, HID, ctx
        )
        s_cur = _clone(pair.x, ctx)
        y_cur = _clone(pair.y, ctx)

    var t_emb_bcast = _broadcast_t(t_emb_3, B, L, HID, ctx)
    var s_sum = add(s_cur, t_emb_bcast, ctx)
    var s_act = silu(s_sum, ctx)
    var s_cond = reshape(s_act, [B * L, HID], ctx)

    var x_nhwc = _nchw_to_nhwc(x, ctx)
    var pe_w = _ld(st, "pixel_embedder.proj.weight", ctx)
    var pe_b = _ld(st, "pixel_embedder.proj.bias", ctx)
    var x_proj = linear(x_nhwc, pe_w, Optional[Tensor](_clone(pe_b, ctx)), ctx)
    var x_pos = _add_pixpos(x_proj, pix_pos, B, H, W, PIXEL_DIM, ctx)
    var x_pixels = _pixelize(x_pos, B, H, W, PIXEL_DIM, PS, ctx)

    for j in range(PIXEL_DEPTH):
        var pw = _load_pit_block(st, j, ctx)
        x_pixels = pit_block_forward[B, L, PIXEL_GROUPS, PIXEL_HEAD](
            x_pixels, s_cond, pw, PIXEL_DIM, HID, ATTN_DIM, P2, H, W, PS, 64, ctx
        )

    var fl_n = _ld(st, "final_layer.norm.weight", ctx)
    var fl_w = _ld(st, "final_layer.linear.weight", ctx)
    var fl_b = _ld(st, "final_layer.linear.bias", ctx)
    var xp_n = rms_norm(x_pixels, fl_n, Float32(1e-6), ctx)
    var xp_o = linear(xp_n, fl_w, Optional[Tensor](_clone(fl_b, ctx)), ctx)
    var toks = _final_reorder(xp_o, B, L, P2, 3, ctx)
    var out = unpatchify(toks, 3, H, W, PS, ctx)
    return out^


def main() raises:
    var ctx = DeviceContext()
    comptime B = 1
    comptime H = 64
    comptime W = 64
    comptime PH = 4
    comptime PW = 4
    comptime L = 16
    comptime LTXT = 8
    comptime ZH = 2
    comptime ZW = 2

    print("=== PiD FAIL-CLOSED proof — corrupt LQ wiring, expect cos DROP ===")
    var ckpt = ShardedSafeTensors.open(String(CKPT))
    var refs = ShardedSafeTensors.open(String(REF))

    var x = _ldr(refs, "x", ctx)
    var t = _ldr(refs, "t_scaled", ctx)
    var y = _ldr(refs, "y", ctx)
    var lq_latent = _ldr(refs, "lq_latent", ctx)
    var pix_pos = _ldr(refs, "pix_pos", ctx)
    var img_cos = _ldr(refs, "img_cos", ctx)
    var img_sin = _ldr(refs, "img_sin", ctx)
    var txt_cos = _ldr(refs, "txt_cos", ctx)
    var txt_sin = _ldr(refs, "txt_sin", ctx)

    var out = _faulted_forward[B, H, W, PH, PW, L, LTXT, ZH, ZW](
        ckpt, x, t, y, lq_latent, Float32(0.0),
        pix_pos, img_cos, img_sin, txt_cos, txt_sin, ctx
    )
    var golden = _ldr(refs, "net_out", ctx)
    var h = ParityHarness()
    var r = h.compare(out, golden.to_host(ctx), ctx)
    print("FAULTED net_out cos=", r.cos, " max_abs=", r.max_abs, " n=", r.n)
    if r.cos < 0.999:
        print("============================================================")
        print("FAIL-CLOSED CONFIRMED: corrupting LQ wiring drops cos below 0.999")
    else:
        print("============================================================")
        print("WARNING: gate did NOT drop — parity is NOT fail-closed!")
        raise Error("fail-closed check: corruption did not move cos")
