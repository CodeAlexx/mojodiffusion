# mageflow_edit_pipeline.mojo — Mage-Flow-Edit-Turbo image-EDIT capstone (pure Mojo + MAX).
#
#   input image + instruction →
#     Qwen3-VL vision tower + text decoder (mage-flow-edit template, drop 64) → DiT
#       context [1,N_TXT,2560]                     (encode_mageflow_edit)
#     MageVAE encode the ref image → CLEAN ref latent tokens [1,Lr,128]  (mageflow_encode)
#     get target noise [1,Lt,128] (PURE noise, like T2I)
#     4-step turbo flow-match Euler denoise (MageFlow DiT), each step:
#       img = concat([target, ref], dim=1)  ([1,Lt+Lr,128], ref stays CLEAN)
#       vel = DiT(img, txt, sigma; frame=2, SH, SH)          → [1,Lt+Lr,128]
#       target += (sigma[i+1]-sigma[i]) * vel[:, :Lt, :]      (target tokens only)
#     MageVAE decode the target → edited PNG.
#
# ── HOW THE REF IMAGE FEEDS THE DiT (the finding, cited) ─────────────────────
# MageFlow-Edit's DiT in_channels stays 128 (NOT 256) — the ref is NOT channel-
# concatenated. It is VAE-encoded into CLEAN latent tokens and sequence-CONCAT'd
# to the target NOISE tokens along the token axis (pipeline.py:502-519, 551-565).
# The target denoises from pure noise; the refs are kept clean every step and only
# the target-token velocity steps the target (pipeline.py:556-559). RoPE: each
# shape in img_shapes=[(1,gh,gw) target, (1,rh,rw) ref] gets frame-axis position =
# its list index (target 0, ref 1); h/w restart per shape (mage_layers.py:187-208,
# MageFlowEmbedRope; frame comes from img_shapes[0], NOT img_ids which is vestigial
# — never passed to transformer, mage_flow.py:107,190-191). Since target & ref are
# encoded at the SAME (possibly NON-SQUARE) resolution, this is EXACTLY
# build_mageflow_rope_tables with frame=2, height=GH, width=GW (GH may differ from
# GW to preserve the input aspect ratio) — no new DiT math, the gated builder
# already handles non-square grids.
#
# Offload (16 GB): encode_mageflow_edit (vision+enc ~9 GB) → freed; VAE encode →
# freed; DiT (~8 GB) → freed; VAE decode. Only txt [1,N_TXT,2560] + latents survive.
# Latent kept F32 across the loop, cast to BF16 (RNE) to feed the DiT (NAVA lesson).
#
# Turbo edit = 4 steps / cfg 1.0 (README: no negative branch → one DiT forward/step).
#
# Run (real edit @ the size baked in main, cshim + cudnn sdpa):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   LD_LIBRARY_PATH="serenitymojo/ops/cshim/lib:$(pixi run python -c 'import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__)+"/lib")')" \
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda \
#     serenitymojo/pipeline/mageflow_edit_pipeline.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt, log, cos as _cos, floor, ceil

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.image.decode import decode_image
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_edit,
    MAGEFLOW_EDIT_DROP_IDX,
)
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import Qwen3VLVisionModel
from serenitymojo.models.dit.mageflow_dit import MageFlowDiT
from serenitymojo.models.vae.mageflow_vae import mageflow_encode
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import (
    _precompute, _resample_h, _resample_v,
)
from serenitymojo.sampling.flow_match import build_sigma_schedule
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.tensor_algebra import reshape, permute, add, mul_scalar, concat, slice
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.pipeline.mageflow_pipeline import mageflow_decode_latent


# ── mage-flow-edit template (utils.py PROMPT_TEMPLATE["mage-flow-edit"], drop 64) ──
comptime MFE_PRE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape,"
    " size, texture, objects, background), then explain how the user's text"
    " instruction should alter or modify the image. Generate a new image that meets"
    " the user's requirements while maintaining consistency with the original input"
    " where appropriate.<|im_end|>\n<|im_start|>user\n"
)
comptime MFE_POST = "<|im_end|>\n<|im_start|>assistant\n"
comptime VISION_START = "<|vision_start|>"
comptime IMAGE_PAD = "<|image_pad|>"
comptime VISION_END = "<|vision_end|>"

comptime MFE_DIR = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo"
comptime MFE_TE_DIR = MFE_DIR + "/text_encoder"
comptime MFE_TRANSFORMER = MFE_DIR + "/transformer"
comptime MFE_VAE = MFE_DIR + "/vae/diffusion_pytorch_model.safetensors"
comptime MFE_TOK_JSON = MFE_TE_DIR + "/tokenizer.json"

# Qwen3-VL image processor (preprocessor_config.json): patch 16, temporal 2,
# merge 2, mean/std 0.5, min 65536 / max 16777216, BICUBIC. VL cond long edge 384.
comptime VL_PATCH = 16
comptime VL_TEMPORAL = 2
comptime VL_MERGE = 2
comptime VL_FACTOR = 32          # PATCH * MERGE
comptime VL_MIN_PIXELS = 65536
comptime VL_MAX_PIXELS = 16777216
comptime VL_ROW_DIM = 1536       # 3 * TEMPORAL * PATCH * PATCH
comptime VL_LONG_EDGE = 384


# ── image edit prompt body: "Image 1: <vision_start><image_pad*nvis><vision_end>{instr}" ──
def mageflow_edit_body(instruction: String, nvis: Int) -> String:
    var pad = String("")
    for _ in range(nvis):
        pad += String(IMAGE_PAD)
    return (
        String("Image 1: ") + String(VISION_START) + pad + String(VISION_END)
        + instruction
    )


# ── tokenize edit prompt → full ids (image_pad already expanded to nvis) ──────
def mageflow_edit_tokenize(
    tok_json: String, instruction: String, nvis: Int
) raises -> List[Int]:
    var tok = Qwen3Tokenizer(tok_json)
    var body = mageflow_edit_body(instruction, nvis)
    var templated = String(MFE_PRE) + body + String(MFE_POST)
    return tok.encode(templated)


# ── Python round-half-to-even (banker's) for smart_resize ─────────────────────
def _py_round(x: Float64) -> Int:
    var fl = floor(x)
    var diff = x - fl
    var fli = Int(fl)
    if diff < 0.5:
        return fli
    if diff > 0.5:
        return fli + 1
    if fli % 2 == 0:
        return fli
    return fli + 1


def _round_by(x: Float64, f: Int) -> Int:
    return _py_round(x / Float64(f)) * f


def _floor_by(x: Float64, f: Int) -> Int:
    return Int(floor(x / Float64(f))) * f


def _ceil_by(x: Float64, f: Int) -> Int:
    return Int(ceil(x / Float64(f))) * f


def _smart_resize(h: Int, w: Int) -> Tuple[Int, Int]:
    var hb = _round_by(Float64(h), VL_FACTOR)
    var wb = _round_by(Float64(w), VL_FACTOR)
    if hb < VL_FACTOR:
        hb = VL_FACTOR
    if wb < VL_FACTOR:
        wb = VL_FACTOR
    if hb * wb > VL_MAX_PIXELS:
        var beta = sqrt(Float64(h) * Float64(w) / Float64(VL_MAX_PIXELS))
        hb = _floor_by(Float64(h) / beta, VL_FACTOR)
        wb = _floor_by(Float64(w) / beta, VL_FACTOR)
    elif hb * wb < VL_MIN_PIXELS:
        var beta = sqrt(Float64(VL_MIN_PIXELS) / (Float64(h) * Float64(w)))
        hb = _ceil_by(Float64(h) * beta, VL_FACTOR)
        wb = _ceil_by(Float64(w) * beta, VL_FACTOR)
    return (hb, wb)


# ── VL preprocess: decode → 384 long-edge cap (BICUBIC) → smart_resize (BICUBIC)
#    → normalize [-1,1] → patchify to [seq,1536] (Qwen3-VL merge-block row order).
# Returns (pixel_values, grid_h, grid_w, seq); grid_t == 1. Mirrors the reference
# _resize_long_edge(384) + Qwen2VLImageProcessorFast (pipeline.py:533 + processor).
struct VlPre(Movable):
    var pv: Tensor
    var grid_h: Int
    var grid_w: Int
    var seq: Int

    def __init__(out self, var pv: Tensor, grid_h: Int, grid_w: Int, seq: Int):
        self.pv = pv^
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.seq = seq


def mageflow_vl_preprocess(path: String, ctx: DeviceContext) raises -> VlPre:
    var img = decode_image(path, True)   # raw RGB, drop alpha (convert("RGB"))
    var iw = img.width
    var ih = img.height

    # uint8 HWC -> f64 HWC in [0,255]
    var srcf = List[Float64]()
    srcf.resize(ih * iw * 3, Float64(0.0))
    for i in range(ih * iw * 3):
        srcf[i] = Float64(img.rgb[i])

    # ── stage A: 384 long-edge cap (BICUBIC), only if long edge > 384 ──
    var cur = srcf^
    var ch_ = ih
    var cw_ = iw
    var long_edge = iw if iw > ih else ih
    if long_edge > VL_LONG_EDGE:
        var scale = Float64(VL_LONG_EDGE) / Float64(long_edge)
        var w2 = _py_round(Float64(iw) * scale)
        var h2 = _py_round(Float64(ih) * scale)
        if w2 < 1:
            w2 = 1
        if h2 < 1:
            h2 = 1
        var cah = _precompute(iw, w2)
        var cav = _precompute(ih, h2)
        var a_h = _resample_h(cur, ih, iw, w2, cah)      # [ih, w2, 3]
        var a_v = _resample_v(a_h, ih, w2, h2, cav)      # [h2, w2, 3]
        cur = a_v^
        ch_ = h2
        cw_ = w2

    # ── stage B: smart_resize the capped image (BICUBIC) → (h_bar, w_bar) ──
    var hw = _smart_resize(ch_, cw_)
    var h_bar = hw[0]
    var w_bar = hw[1]
    var grid_h = h_bar // VL_PATCH
    var grid_w = w_bar // VL_PATCH
    var seq = grid_h * grid_w
    var cbh = _precompute(cw_, w_bar)
    var cbv = _precompute(ch_, h_bar)
    var b_h = _resample_h(cur, ch_, cw_, w_bar, cbh)     # [ch_, w_bar, 3]
    var v_out = _resample_v(b_h, ch_, w_bar, h_bar, cbv) # [h_bar, w_bar, 3]

    # ── normalize (v/127.5 - 1) + patchify (merge-block row order, T dup) ──
    var gh_b = grid_h // VL_MERGE
    var gw_b = grid_w // VL_MERGE
    var vals = List[Float32]()
    vals.resize(seq * VL_ROW_DIM, Float32(0.0))
    for bh in range(gh_b):
        for bw in range(gw_b):
            for mh in range(VL_MERGE):
                for mw in range(VL_MERGE):
                    var gh = bh * VL_MERGE + mh
                    var gw = bw * VL_MERGE + mw
                    var row = ((bh * gw_b + bw) * VL_MERGE + mh) * VL_MERGE + mw
                    var rbase = row * VL_ROW_DIM
                    for c in range(3):
                        for tp in range(VL_TEMPORAL):
                            var cbase = rbase + (c * VL_TEMPORAL + tp) * VL_PATCH * VL_PATCH
                            for ph in range(VL_PATCH):
                                var py = gh * VL_PATCH + ph
                                for pw in range(VL_PATCH):
                                    var px = gw * VL_PATCH + pw
                                    var v = v_out[(py * w_bar + px) * 3 + c] / 127.5 - 1.0
                                    vals[cbase + ph * VL_PATCH + pw] = Float32(v)
    var pv = Tensor.from_host(vals, [seq, VL_ROW_DIM], STDtype.F32, ctx)
    return VlPre(pv^, grid_h, grid_w, seq)


# ── load an image, BICUBIC-resize to (IH,IW), normalize [-1,1] -> [1,3,IH,IW] ──
# Mirrors _preprocess_ref_image (pipeline.py:357-364): resize BICUBIC then
# TF.normalize(0.5,0.5) => v/127.5 - 1. NCHW for mageflow_encode.
def mageflow_load_image_nchw[IH: Int, IW: Int](
    path: String, ctx: DeviceContext
) raises -> Tensor:
    var img = decode_image(path, True)
    var ih = img.height
    var iw = img.width
    var srcf = List[Float64]()
    srcf.resize(ih * iw * 3, Float64(0.0))
    for i in range(ih * iw * 3):
        srcf[i] = Float64(img.rgb[i])
    var ch_ = _precompute(iw, IW)
    var cv_ = _precompute(ih, IH)
    var h_out = _resample_h(srcf, ih, iw, IW, ch_)       # [ih, IW, 3]
    var v_out = _resample_v(h_out, ih, IW, IH, cv_)      # [IH, IW, 3]
    # HWC [-1,1] -> NCHW
    var vals = List[Float32]()
    vals.resize(3 * IH * IW, Float32(0.0))
    for c in range(3):
        for y in range(IH):
            for x in range(IW):
                var v = v_out[(y * IW + x) * 3 + c] / 127.5 - 1.0
                vals[(c * IH + y) * IW + x] = Float32(v)
    return Tensor.from_host(vals, [1, 3, IH, IW], STDtype.F32, ctx)


# ── STAGE 1: edit conditioning (vision tower + text decoder; both freed on return) ──
# ids: exact edit token ids (image_pad expanded to nvis). pixel_values [seq,1536].
# Returns txt [1, L-64, 2560] BF16 — the DiT context. The ~9 GB VL encoder is
# destroyed when this def returns, freeing VRAM for the DiT.
def mageflow_edit_encode_text[S_VIS: Int](
    te_dir: String,
    ids: List[Int],
    grid_t: Int, grid_h: Int, grid_w: Int,
    pixel_values: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var vision = Qwen3VLVisionModel.load(te_dir, ctx)
    var enc = load_krea2_qwen3vl_4b(te_dir, ctx)
    return encode_mageflow_edit[S_VIS](
        vision, enc, ids, grid_t, grid_h, grid_w, pixel_values, ctx, False
    )


# ── STAGE 2a: MageVAE encode the ref image → CLEAN latent tokens [1, Lr, 128] ──
# img_nchw [1,3,IH,IW] in [-1,1]. mageflow_encode -> mean [1,128,SH,SW]; convert to
# token layout (b (h w) c) matching the reference rearrange (pipeline.py:509 form).
def mageflow_edit_vae_encode[IH: Int, IW: Int](
    vae_path: String, img_nchw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    comptime SH = IH // 16
    comptime SW = IW // 16
    var st = ShardedSafeTensors.open(vae_path)
    var lat = mageflow_encode[IH, IW](img_nchw, st, ctx)   # [1,128,SH,SW] NCHW
    var nhwc = permute(lat, [0, 2, 3, 1], ctx)             # [1,SH,SW,128]
    return reshape(nhwc, [1, SH * SW, 128], ctx)           # [1, SH*SW, 128] (h w) c


# ── STAGE 2b: 4-step edit denoise (DiT freed on return) ──────────────────────
# target_noise_f32 [1,Lt,128] pure noise; ref_tokens_f32 [1,Lr,128] CLEAN. The DiT
# runs over [txt, target, ref] (frame=2, SH,SH); only the target-token velocity
# steps the target. Ref stays clean every step. Returns final target [1,Lt,128] F32.
def mageflow_edit_denoise[
    Lt: Int, Lr: Int, N_TXT: Int, N_IMG: Int, S: Int, GH: Int, GW: Int
](
    transformer_dir: String,
    txt: Tensor,
    target_noise_f32: Tensor,
    ref_tokens_f32: Tensor,
    steps: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime assert N_IMG == Lt + Lr, "N_IMG must equal Lt + Lr"
    comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
    comptime assert Lt == GH * GW and Lr == GH * GW, "target/ref must be GH*GW (same res)"
    var dit = MageFlowDiT.load(transformer_dir, ctx)
    var sigmas = build_sigma_schedule(steps, Float32(6.0))  # steps+1 values
    var x_bf = torch_f32_to_bf16_rne(target_noise_f32, ctx)  # target, BF16
    var ref_bf = torch_f32_to_bf16_rne(ref_tokens_f32, ctx)  # ref (clean), BF16
    for i in range(steps):
        var img = concat(1, ctx, x_bf, ref_bf)               # [1, Lt+Lr, 128]
        var vel_full = dit.forward[N_IMG, N_TXT, S](
            img, txt, sigmas[i], 2, GH, GW, ctx
        )                                                    # [1, Lt+Lr, 128]
        var vel = slice(vel_full, 1, 0, Lt, ctx)             # target velocity [1,Lt,128]
        var dt = sigmas[i + 1] - sigmas[i]
        var delta = mul_scalar(vel, dt, ctx)                 # BF16 (dt*model_output)
        var x_f32 = add(
            cast_tensor(x_bf, STDtype.F32, ctx),
            cast_tensor(delta, STDtype.F32, ctx),
            ctx,
        )
        x_bf = torch_f32_to_bf16_rne(x_f32, ctx)             # round prev_sample -> BF16
        print("  [edit-denoise] step", i + 1, "/", steps, "sigma", sigmas[i], "->", sigmas[i + 1])
    return cast_tensor(x_bf, STDtype.F32, ctx)               # final target (F32); dit freed


# ── seeded Gaussian noise (Box-Muller) — same as t2i pipeline ────────────────
def _u01(mut state: UInt64) -> Float64:
    state = state + 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def gaussian_noise(n: Int, seed: UInt64) raises -> List[Float32]:
    comptime PI2 = 6.283185307179586
    var state = seed
    var out = List[Float32]()
    var i = 0
    while i < n:
        var u1 = _u01(state)
        var u2 = _u01(state)
        if u1 < 1e-12:
            u1 = 1e-12
        var r = sqrt(-2.0 * log(u1))
        out.append(Float32(r * _cos(PI2 * u2)))
        if i + 1 < n:
            out.append(Float32(r * _cos(PI2 * u2 + 1.5707963267948966)))
        i += 2
    return out^


# ── full edit: image + instruction → edited PNG ───────────────────────────────
# Comptime-specialized on the VL vision seq S_VIS, N_TXT, and the target latent
# grid (GH, GW) — SEPARATE so the target PRESERVES the input aspect ratio (the
# reference _compute_aspect_ratio_size / _edit_target_size rule, pipeline.py:96-118).
# The ref image is VAE-encoded at the SAME target size (IH=GH*16, IW=GW*16), so
# target-token grid == ref-token grid == GH×GW (the frame=2 concat + rope need
# equal grids). Lt=Lr=GH*GW.
def generate_mageflow_edit[
    S_VIS: Int, N_TXT: Int, GH: Int, GW: Int, NVIS: Int
](
    image_path: String,
    instruction: String,
    seed: UInt64,
    out_path: String,
    ctx: DeviceContext,
) raises:
    comptime IH = GH * 16
    comptime IW = GW * 16
    comptime Lt = GH * GW
    comptime Lr = GH * GW
    comptime N_IMG = Lt + Lr
    comptime S = N_IMG + N_TXT

    print("[mf-edit] VL preprocess (384-cap + smart_resize) ...")
    var vl = mageflow_vl_preprocess(image_path, ctx)
    print("[mf-edit]   grid =", vl.grid_h, "x", vl.grid_w, " seq =", vl.seq,
          " (expected S_VIS =", S_VIS, ")")
    if vl.seq != S_VIS:
        raise Error(
            String("generate_mageflow_edit: VL seq=") + String(vl.seq)
            + " != comptime S_VIS=" + String(S_VIS)
            + " — re-bake S_VIS for this image."
        )
    var nvis = (vl.grid_h // 2) * (vl.grid_w // 2)
    if nvis != NVIS:
        raise Error(
            String("generate_mageflow_edit: nvis=") + String(nvis)
            + " != comptime NVIS=" + String(NVIS)
        )

    print("[mf-edit] tokenize edit prompt (image_pad ->", nvis, "tokens) ...")
    var ids = mageflow_edit_tokenize(String(MFE_TOK_JSON), instruction, nvis)
    var keep = len(ids) - MAGEFLOW_EDIT_DROP_IDX
    if keep != N_TXT:
        raise Error(
            String("generate_mageflow_edit: keep(N_TXT)=") + String(keep)
            + " != comptime N_TXT=" + String(N_TXT)
            + " — re-bake N_TXT for this instruction."
        )

    print("[mf-edit] STAGE1 encode edit conditioning (Qwen3-VL, drop 64) ...")
    var txt = mageflow_edit_encode_text[S_VIS](
        String(MFE_TE_DIR), ids, 1, vl.grid_h, vl.grid_w, vl.pv, ctx
    )
    var ts = txt.shape()
    print("[mf-edit]   txt =", ts[0], "x", ts[1], "x", ts[2], "(VL encoder freed)")

    print("[mf-edit] STAGE2a MageVAE encode ref image ->", Lr, "clean tokens ...")
    var img_nchw = mageflow_load_image_nchw[IH, IW](image_path, ctx)
    var ref_tokens = mageflow_edit_vae_encode[IH, IW](String(MFE_VAE), img_nchw, ctx)

    print("[mf-edit] STAGE2b edit denoise (turbo 4-step, target+ref concat) ...")
    var noise_h = gaussian_noise(Lt * 128, seed)
    var tgt_noise = Tensor.from_host(noise_h, [1, Lt, 128], STDtype.F32, ctx)
    var latent = mageflow_edit_denoise[Lt, Lr, N_TXT, N_IMG, S, GH, GW](
        String(MFE_TRANSFORMER), txt, tgt_noise, ref_tokens, 4, ctx
    )
    print("[mf-edit]   final target latent computed (DiT freed)")

    print("[mf-edit] STAGE3 MageVAE decode -> RGB ...")
    var rgb = mageflow_decode_latent[Lt, GH, GW](String(MFE_VAE), latent, ctx)
    var rs = rgb.shape()
    print("[mf-edit]   image =", rs[0], rs[1], rs[2], rs[3])
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
    print("[mf-edit] saved:", out_path)


# ── demo edit: dog.jpg + "snowy forest", max_size 1024, ASPECT-RATIO PRESERVED ──
# The VL conditioning image is 384-long-edge capped regardless of the target
# output size, so S_VIS / N_TXT / NVIS are the SAME as the 256² gate (the VL grid
# depends only on the capped source, not the output resolution):
#   dog.jpg -> VL grid (1,24,12): S_VIS=288, NVIS=72; instruction keep N_TXT=92.
#
# TARGET SIZE (reference _edit_target_size / _compute_aspect_ratio_size, max_size
# 1024, pipeline.py:96-118): dog.jpg is 1024w × 2048h (PORTRAIT, h>=w), so
#   new_h = max_size = 1024,  new_w = round(w*max_size/h) = round(1024*1024/2048)
#         = 512,   both already /16  →  target 1024h × 512w.
# Latent grid: GH = 1024/16 = 64 (tall), GW = 512/16 = 32 (narrow) — NON-SQUARE.
# Lt = Lr = GH*GW = 2048, N_IMG = 4096. (The old code forced SH=64 square → the
# portrait dog came out stretched to 1024²; this preserves the input aspect ratio.)
comptime EDIT_IMAGE = "/home/alex/mojodiffusion/output/mageflow/edit_input.jpg"
comptime EDIT_INSTR = "Change the background to a snowy forest."
comptime EDIT_S_VIS = 288
comptime EDIT_N_TXT = 92
comptime EDIT_NVIS = 72
comptime EDIT_GH = 64        # 1024 / 16  (target height / 16)
comptime EDIT_GW = 32        # 512 / 16   (target width  / 16)
comptime EDIT_OUT = "/home/alex/mojodiffusion/output/mageflow/edit_result_v2.png"


def main() raises:
    var ctx = DeviceContext()
    print("=== Mage-Flow-Edit-Turbo image EDIT (pure Mojo, turbo 4-step) — 1024h×512w portrait ===")
    print("[mf-edit] image:", EDIT_IMAGE)
    print("[mf-edit] instruction:", EDIT_INSTR)
    generate_mageflow_edit[EDIT_S_VIS, EDIT_N_TXT, EDIT_GH, EDIT_GW, EDIT_NVIS](
        String(EDIT_IMAGE), String(EDIT_INSTR), UInt64(42), String(EDIT_OUT), ctx
    )
    print("[mf-edit] done. edited:", EDIT_OUT)
