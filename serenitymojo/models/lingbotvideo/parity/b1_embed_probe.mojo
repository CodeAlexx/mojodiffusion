# models/lingbotvideo/parity/b1_embed_probe.mojo — CHUNK B1 gate.
#
# Loads oracle_b1.safetensors (all tensors stored F32) and gates the transformer
# PRE-BLOCK path: patchify -> patch_embed -> text_embed -> joint, the 3D RoPE
# BUILD (positions + per-axis complex freqs), and time -> t_emb -> temb6.
# CLOSES skeptic FRAGILE-2 (A1/A3 consumed a precomputed rope table; here we
# BUILD it — freqs_cos/freqs_sin + pos_ids — vs the oracle).
#
# DTYPE: the pre-block path is effectively FP32 (norms/time/rope/synthetic
# Linears all FP32). Everything is gated in the oracle's dtype (FP32).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/b1_embed_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.backbone import (
    lingbot_patchify,
    lingbot_patch_embed,
    lingbot_text_embed,
    lingbot_build_rope,
    lingbot_time_embed,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime GT = 1
comptime GH = 8
comptime GW = 8
comptime TEXT_LEN = 8
comptime N_VIDEO = GT * GH * GW  # 64
comptime S = N_VIDEO + TEXT_LEN  # 72
comptime H = 2048
comptime EPS = Float32(1e-6)
comptime THETA = Float32(256.0)


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _gate(name: String, mine: List[Float32], reference: List[Float32], thr: Float64) raises:
    var harness = ParityHarness(thr)
    var res = harness.compare_host(mine, reference)
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    for i in range(len(mine)):
        nm += Float64(mine[i]) * Float64(mine[i])
        nr += Float64(reference[i]) * Float64(reference[i])
    var mag = sqrt(nm) / sqrt(nr) if nr > 0.0 else Float64(0.0)
    var tag = "PASS" if res.cos >= thr else "FAIL"
    print(
        "[B1]", name, "cos =", res.cos, " max_abs =", res.max_abs,
        " |mine|/|ref| =", mag, " ", tag, "(thr", thr, ")",
    )


def main() raises:
    var ctx = DeviceContext()
    print("[B1] loading oracle_b1.safetensors  S =", S, " n_video =", N_VIDEO)
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_b1.safetensors")

    # ── inputs ───────────────────────────────────────────────────────────────
    var latent = _load_f32(st, "latent", ctx)          # [1,16,1,16,16]
    var timestep = _load_f32(st, "timestep", ctx)      # [1]
    var text_embeds = _load_f32(st, "text_embeds", ctx)  # [1,8,2560]

    # ── weights ──────────────────────────────────────────────────────────────
    var pe_w = _load_f32(st, "patch_embedder_w", ctx)  # [2048,64]
    var pe_b = _load_f32(st, "patch_embedder_b", ctx)  # [2048]
    var text_norm_w = _load_f32(st, "text_norm_w", ctx)
    var text_lin1_w = _load_f32(st, "text_lin1_w", ctx)
    var text_lin1_b = _load_f32(st, "text_lin1_b", ctx)
    var text_lin2_w = _load_f32(st, "text_lin2_w", ctx)
    var text_lin2_b = _load_f32(st, "text_lin2_b", ctx)
    var time_lin1_w = _load_f32(st, "time_lin1_w", ctx)
    var time_lin1_b = _load_f32(st, "time_lin1_b", ctx)
    var time_lin2_w = _load_f32(st, "time_lin2_w", ctx)
    var time_lin2_b = _load_f32(st, "time_lin2_b", ctx)
    var time_mod_w = _load_f32(st, "time_mod_w", ctx)
    var time_mod_b = _load_f32(st, "time_mod_b", ctx)

    # ── 1. patchify ──────────────────────────────────────────────────────────
    var patch_tokens = lingbot_patchify(latent, 1, 2, 2, ctx)  # [64,64]
    var pt_host = patch_tokens.to_host(ctx)
    _gate("patch_tokens", pt_host, _load_f32(st, "patch_tokens", ctx).to_host(ctx), 0.999)

    # ── 2. patch_embed / text_embed / joint ──────────────────────────────────
    var x = lingbot_patch_embed(patch_tokens, pe_w, pe_b, ctx)  # [64,2048]
    var x_host = x.to_host(ctx)
    _gate("x (patch_embed)", x_host, _load_f32(st, "x", ctx).to_host(ctx), 0.999)

    var text = lingbot_text_embed(
        text_embeds, text_norm_w, text_lin1_w, text_lin1_b,
        text_lin2_w, text_lin2_b, EPS, ctx,
    )  # [1,8,2048]
    var text_host = text.to_host(ctx)
    _gate("text (text_embed)", text_host, _load_f32(st, "text", ctx).to_host(ctx), 0.999)

    # joint = cat([x, text], dim=1) -> [72,2048] (host concat).
    var joint_host = List[Float32]()
    joint_host.resize(S * H, Float32(0.0))
    for i in range(N_VIDEO * H):
        joint_host[i] = x_host[i]
    for i in range(TEXT_LEN * H):
        joint_host[N_VIDEO * H + i] = text_host[i]
    _gate("joint", joint_host, _load_f32(st, "joint", ctx).to_host(ctx), 0.999)

    # ── 3. RoPE build (positions + freqs) ────────────────────────────────────
    var rope = lingbot_build_rope(GT, GH, GW, TEXT_LEN, THETA, ctx)
    var cos_host = rope[0].to_host(ctx)
    var sin_host = rope[1].to_host(ctx)
    var posids_host = rope[2].to_host(ctx)
    _gate("freqs_cos", cos_host, _load_f32(st, "freqs_cos", ctx).to_host(ctx), 0.9999)
    _gate("freqs_sin", sin_host, _load_f32(st, "freqs_sin", ctx).to_host(ctx), 0.9999)

    # pos_ids exact-match count out of S*3 = 216.
    var ref_posids = _load_f32(st, "pos_ids", ctx).to_host(ctx)
    var matched = 0
    for i in range(S * 3):
        if Int(posids_host[i]) == Int(ref_posids[i]):
            matched += 1
    print("[B1] pos_ids match:", matched, "/", S * 3,
          "(video-t offset = text_len+1 =", TEXT_LEN + 1, ")")

    # ── 4. time -> t_emb / temb6 ─────────────────────────────────────────────
    var te = lingbot_time_embed(
        timestep, time_lin1_w, time_lin1_b, time_lin2_w, time_lin2_b,
        time_mod_w, time_mod_b, S, ctx,
    )
    var t_emb_host = te[0].to_host(ctx)
    var temb6_host = te[1].to_host(ctx)
    _gate("t_emb", t_emb_host, _load_f32(st, "t_emb", ctx).to_host(ctx), 0.999)
    _gate("temb6", temb6_host, _load_f32(st, "temb6", ctx).to_host(ctx), 0.999)

    print("[B1] ===== B1 probe complete =====")
