# mageflow_pipeline.mojo — Mage-Flow-Turbo text→image capstone (pure Mojo + MAX).
#
#   tokenizer → Qwen3-VL text encoder (mage t2i template, drop 34) → 4-step
#   flow-match Euler denoise (MageFlow DiT, turbo CFG 1.0, no negative) →
#   MageVAE one-step decode → PNG.
#
# All three heavyweight modules (encoder 8.9 GB, DiT 8.2 GB, VAE) exceed a shared
# 16 GB card together, so they are loaded / freed SEQUENTIALLY: each stage is a
# `def` whose model struct is destroyed at return, freeing its VRAM before the
# next stage's model loads (the Z-Image offload pattern). Only the small txt
# context [1,N_TXT,2560] and the latent [1,N_IMG,128] survive between stages.
#
# Conventions taken VERBATIM from mage_flow/pipeline.py generate_images (cfg=1.0
# turbo → the `use_neg=False` single-forward branch):
#   • noise geometry (pipeline.py:299-316): MageVAE 16× downsample, 128 ch, NO
#     patch packing. gh=ceil(H/16), gw=ceil(W/16); img = [1, gh*gw, 128].
#   • scheduler (pipeline.py:37-50, 340-343): FlowMatchEulerDiscreteScheduler,
#     shift 6.0. Model timestep = RAW scheduler.sigmas[i] (NOT ×1000). Euler:
#     x += (sigma[i+1]-sigma[i]) * velocity.  build_sigma_schedule(4,6.0) matches
#     the diffusers sigma table exactly (verified vs the oracle).
#   • DiT context = encode_mageflow_text POST-norm last_hidden_state, rows [34:).
#   • decode (pipeline.py:121-127): vae.decode(unpack(latent)), image = [-1,1].
#
# The latent is kept F32 across the loop and cast to BF16 with RNE rounding to
# feed the DiT (the NAVA lesson: a plain bf16 cast round-trips accumulate in the
# sampling loop). Velocity is upcast to F32 for the Euler add.
#
# Run (2 user prompts @ 1024², turbo 4-step):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   LD_LIBRARY_PATH="serenitymojo/ops/cshim/lib:$(pixi run python -c 'import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__)+"/lib")')" \
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda \
#     serenitymojo/pipeline/mageflow_pipeline.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt, log, cos as _cos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_text,
    load_krea2_qwen3vl_4b,
    MAGEFLOW_T2I_DROP_IDX,
)
from serenitymojo.models.dit.mageflow_dit import MageFlowDiT
from serenitymojo.models.vae.mageflow_vae import mageflow_decode
from serenitymojo.sampling.flow_match import build_sigma_schedule
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.ops.tensor_algebra import reshape, permute, add, mul_scalar
from serenitymojo.image.png import save_png, ValueRange


# ── mage-flow t2i template (utils.py PROMPT_TEMPLATE["mage-flow"]) ────────────
comptime MF_PRE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size,"
    " texture, quantity, text, spatial relationships of the objects and"
    " background:<|im_end|>\n<|im_start|>user\n"
)
comptime MF_POST = "<|im_end|>\n<|im_start|>assistant\n"

comptime MF_TURBO = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Turbo"
comptime MF_TE_DIR = MF_TURBO + "/text_encoder"
comptime MF_TRANSFORMER = MF_TURBO + "/transformer"
comptime MF_VAE = MF_TURBO + "/vae/diffusion_pytorch_model.safetensors"
comptime MF_TOK_JSON = MF_TE_DIR + "/tokenizer.json"


# ── tokenize the mage-flow templated prompt → full token ids ─────────────────
def mageflow_tokenize(tok_json: String, prompt: String) raises -> List[Int]:
    var tok = Qwen3Tokenizer(tok_json)
    var templated = String(MF_PRE) + prompt + String(MF_POST)
    return tok.encode(templated)


# ── STAGE 1: text conditioning (encoder freed on return) ─────────────────────
# Returns txt [1, keep, 2560] (BF16) — the POST-norm last_hidden_state with the
# leading `drop_idx` system-prompt rows dropped. The 8.9 GB encoder is destroyed
# when this def returns, freeing VRAM for the DiT.
def mageflow_encode_text(
    te_dir: String, ids: List[Int], drop_idx: Int, ctx: DeviceContext
) raises -> Tensor:
    var enc = load_krea2_qwen3vl_4b(te_dir, ctx)
    return encode_mageflow_text(enc, ids, drop_idx, ctx)


# ── STAGE 2: 4-step flow-match denoise (DiT freed on return) ─────────────────
# init_f32: [1, N_IMG, 128] F32 initial packed noise. txt: [1, N_TXT, 2560].
# Returns the final latent [1, N_IMG, 128] F32 (pre-VAE). frame=1; SH is the
# latent side (N_IMG == SH*SH). CFG 1.0 turbo → one DiT forward per step.
def mageflow_denoise[
    N_IMG: Int, N_TXT: Int, S: Int, SH: Int
](
    transformer_dir: String,
    txt: Tensor,
    init_f32: Tensor,
    steps: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
    var dit = MageFlowDiT.load(transformer_dir, ctx)
    var sigmas = build_sigma_schedule(steps, Float32(6.0))  # steps+1 values
    # Match diffusers FlowMatchEulerDiscreteScheduler exactly: the running latent
    # is BF16 at each step boundary (get_noise is bf16; scheduler.step upcasts to
    # f32, does prev = sample + dt*model_output, then casts prev back to bf16).
    # dt*velocity is a bf16 tensor * python-float → stays bf16, exactly like the
    # reference; only the final add is f32. Feeding the bf16 latent (not an f32
    # one) to the DiT keeps the whole trajectory on the reference path.
    var x_bf = torch_f32_to_bf16_rne(init_f32, ctx)  # step-0 latent, BF16
    for i in range(steps):
        var vel = dit.forward[N_IMG, N_TXT, S](
            x_bf, txt, sigmas[i], 1, SH, SH, ctx
        )  # [1, N_IMG, 128] BF16
        var dt = sigmas[i + 1] - sigmas[i]
        var delta = mul_scalar(vel, dt, ctx)  # BF16 (dt*model_output, ref form)
        var x_f32 = add(
            cast_tensor(x_bf, STDtype.F32, ctx),
            cast_tensor(delta, STDtype.F32, ctx),
            ctx,
        )  # F32 add
        x_bf = torch_f32_to_bf16_rne(x_f32, ctx)  # round prev_sample → BF16
        print("  [denoise] step", i + 1, "/", steps, "sigma", sigmas[i], "->", sigmas[i + 1])
    return cast_tensor(x_bf, STDtype.F32, ctx)  # final latent (F32); dit freed on return


# ── STAGE 3: MageVAE one-step decode → RGB [1,3,SH*16,SH*16] ──────────────────
# latent_f32: [1, N_IMG, 128] F32. unpack "(h w) c -> c h w" then MageVAE decode.
def mageflow_decode_latent[
    N_IMG: Int, SH: Int
](vae_path: String, latent_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(vae_path)
    # unpack: [1, SH*SH, 128] (b (h w) c) → [1, SH, SH, 128] (b h w c) → NCHW
    var nhwc = reshape(latent_f32, [1, SH, SH, 128], ctx)
    var nchw = permute(nhwc, [0, 3, 1, 2], ctx)  # [1, 128, SH, SH]
    return mageflow_decode[SH](nchw, st, ctx)  # [1, 3, SH*16, SH*16]


# ── seeded Gaussian noise (Box-Muller) for fresh generation ──────────────────
# NOTE: this is NOT torch.randn (nor the Gaussian-Shading watermark encode_noise
# the reference uses); it is a valid N(0,1) init that produces a correct image.
# The parity gate feeds the ORACLE's exact init noise instead, so the DiT loop is
# tested against the true trajectory.
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


# ── full generate: prompt → PNG (comptime-specialized on N_TXT/N_IMG/SH) ─────
def generate_mageflow[
    N_IMG: Int, N_TXT: Int, S: Int, SH: Int
](prompt: String, seed: UInt64, out_path: String, ctx: DeviceContext) raises:
    print("[mageflow] tokenizing prompt (", N_TXT, "text tokens expected )")
    var ids = mageflow_tokenize(String(MF_TOK_JSON), prompt)
    var keep = len(ids) - MAGEFLOW_T2I_DROP_IDX
    if keep != N_TXT:
        raise Error(
            String("generate_mageflow: tokenized N_TXT=") + String(keep)
            + " != comptime N_TXT=" + String(N_TXT)
            + " — re-run mageflow_count_tokens.mojo and rebake."
        )

    print("[mageflow] STAGE1 encode text (Qwen3-VL, drop 34) ...")
    var txt = mageflow_encode_text(String(MF_TE_DIR), ids, MAGEFLOW_T2I_DROP_IDX, ctx)
    var ts = txt.shape()
    print("[mageflow]   txt =", ts[0], "x", ts[1], "x", ts[2], "(encoder freed)")

    print("[mageflow] STAGE2 denoise (turbo 4-step, MageFlow DiT) ...")
    var noise_h = gaussian_noise(N_IMG * 128, seed)
    var init = Tensor.from_host(noise_h, [1, N_IMG, 128], STDtype.F32, ctx)
    var latent = mageflow_denoise[N_IMG, N_TXT, S, SH](
        String(MF_TRANSFORMER), txt, init, 4, ctx
    )
    print("[mageflow]   final latent computed (DiT freed)")

    print("[mageflow] STAGE3 MageVAE decode → RGB ...")
    var rgb = mageflow_decode_latent[N_IMG, SH](String(MF_VAE), latent, ctx)
    var rs = rgb.shape()
    print("[mageflow]   image =", rs[0], rs[1], rs[2], rs[3])

    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
    print("[mageflow] saved:", out_path)


# ── the 2 user prompts (mageflow_test_prompts.txt), 1024², turbo 4-step ──────
# N_TXT baked from mageflow_count_tokens.mojo:
#   PROMPT1 keep=86 ; PROMPT2 keep=182 ; SH=64 (1024²) → N_IMG=4096.
comptime SH1024 = 64
comptime NIMG1024 = 4096  # 64*64
comptime P1_NTXT = 86
comptime P2_NTXT = 182

comptime PROMPT1 = (
    "A close-up of a face adorned with intricate black and blue patterns. The"
    " left side of the face is predominantly yellow, with symbols and doodles,"
    " while the right side is dark, featuring mechanical elements. The eye on the"
    " left is a striking shade of yellow, contrasting sharply with the surrounding"
    " patterns. The face is partially covered by a hooded garment, cinematic_1940s"
    " cinematic_octane"
)
comptime PROMPT2 = (
    "realistic, vibrant, detailed. a cinematic photograph classily depicts a"
    " prestigious beautiful woman dressed in an all white dress. The woman has no"
    " face. Her head is a sphere, a colorful gradient foreign planet, and not a"
    " human head. Her outfit is adorned by luxurious gold trim and other attire."
    " Her skin is tan with smooth rough textures. She has a very detailed"
    " planetary sphere for her head that emits natural weather phenomenon and neon"
    " ethereal ozonous layers, swirls, and hues. The image is sharp and"
    " professional with studio quality. The contrast and style is very sharp, but"
    " liquid and vivid. Gorgeous background and details with subtle hints of"
    " illumination. glass, ether, spirit, breath, light, energy. Particles, high"
    " density quality, vibrant colors and hues. The overall style of the image is"
    " highly stylish, dynamic and free-flowing elegance., modest, sfw"
)

comptime OUT1 = "/home/alex/mojodiffusion/output/mageflow/prompt1.png"
comptime OUT2 = "/home/alex/mojodiffusion/output/mageflow/prompt2.png"


def main() raises:
    var ctx = DeviceContext()
    print("=== Mage-Flow-Turbo T2I (pure Mojo, turbo 4-step) — 1024x1024 ===")

    print("\n--- PROMPT 1 ---")
    generate_mageflow[NIMG1024, P1_NTXT, NIMG1024 + P1_NTXT, SH1024](
        String(PROMPT1), UInt64(42), String(OUT1), ctx
    )

    print("\n--- PROMPT 2 ---")
    generate_mageflow[NIMG1024, P2_NTXT, NIMG1024 + P2_NTXT, SH1024](
        String(PROMPT2), UInt64(43), String(OUT2), ctx
    )

    print("\n[done] both images written.")
