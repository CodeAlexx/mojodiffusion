# pipeline/klein_edit_mojo.mojo — Klein-9B MULTI-REFERENCE edit CLI (task #26).
#
# Native image-conditioned editing/composition: 1 or 2 reference images are
# VAE-encoded, packed to latent token blocks, and appended after the target
# noise tokens with distinct RoPE T-coordinates (ref r -> ref_t_offset + r,
# default 10.0/11.0 — the Rust klein_edit_infer.rs REF_T_OFFSET=10.0 convention
# extended per-reference; ComfyUI `reference_latent_method: index` t=r+1 is
# reachable via argv ref_t=1.0). The Klein base model is a NATIVE edit model —
# zero training, pure inference wiring.
#
# Usage:
#   klein_edit_mojo [config.json] [prompts.json] [prompt_id] [out.png] \
#                   ref1.png [ref2.png|-] [denoise] [shift] [ref_t]
#   argv[1] model config          (default serenitymojo/configs/klein9b.json)
#   argv[2] sample prompts JSON   (serenity.sample_prompts.v1, caps precached)
#   argv[3] prompt id             ("" -> first prompt)
#   argv[4] output PNG
#   argv[5] reference image 1     (required; any png/jpeg/webp, staged to 512^2)
#   argv[6] reference image 2     (optional; "-"/"none" -> single-reference)
#   argv[7] denoise strength      (default 1.0 = full generation w/ conditioning)
#   argv[8] shift                 (default 2.02)
#   argv[9] reference t offset    (default 10.0)
#
# Bounded surface: Klein-9B (n_heads=32) at 512x512 (16GB RTX 5080 budget).
# 2 refs -> S = 512 + 3*1024 = 3584 joint tokens.
#
# Caps: prompt embeddings are precached by klein9b_precache_sample_prompts
# (Qwen3-8B layers 8/17/26 -> [1,512,12288]) so no text encoder co-resides.

from std.sys import argv
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.validation_sampler import load_caps
from serenitymojo.io.cap_cache import validate_klein_cap_cache_header
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.sampling.klein_sampler import (
    klein_sample_with_reference_latent,
    klein_sample_with_reference_latents2,
)
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.io.ffi import O_RDONLY, sys_close, sys_open
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from image.transform import resize_bilinear


comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"

comptime N_TXT = 512
comptime H_9B = 32
comptime Dh = 128
comptime LH_512 = 32
comptime LW_512 = 32
comptime N_IMG_512 = 1024
comptime N_EDIT1_IMG_512 = 2 * N_IMG_512      # target + 1 ref
comptime S_EDIT1_512 = N_EDIT1_IMG_512 + N_TXT  # 2560
comptime N_EDIT2_IMG_512 = 3 * N_IMG_512      # target + 2 refs
comptime S_EDIT2_512 = N_EDIT2_IMG_512 + N_TXT  # 3584


def _path_exists(path: String) -> Bool:
    if path == String(""):
        return False
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _require_file(label: String, path: String) raises:
    if not _path_exists(path):
        raise Error(String("klein_edit_mojo: missing ") + label + String(": ") + path)


def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if p.label == wanted:
            return p^
    raise Error(String("klein_edit_mojo: no prompt id ") + wanted)


def _empty_ref(path: String) -> Bool:
    return path == String("") or path == String("-") or path == String("none")


def _load_image_512(path: String, ctx: DeviceContext) raises -> Tensor:
    var img = decode_image_any(path)
    var resized = resize_bilinear(img, 512, 512)
    var host = image_to_signed_nchw(resized)
    print(
        "[klein-edit-mojo] reference", path,
        "(", img.width, "x", img.height, ") -> 512x512"
    )
    return Tensor.from_host(host, [1, 3, 512, 512], STDtype.F32, ctx)


# Encode 1 or 2 references with ONE encoder instance; the encoder weights free
# when this returns (before the DiT stack loads). Tensor is move-only ->
# List[ArcPointer[Tensor]] (house idiom).
def _encode_refs_512(
    ref1_path: String, ref2_path: String, cfg: TrainConfig, ctx: DeviceContext
) raises -> List[ArcPointer[Tensor]]:
    var enc = KleinVaeEncoder[512, 512].load(cfg.vae, ctx)
    var out = List[ArcPointer[Tensor]]()
    var img1 = _load_image_512(ref1_path, ctx)
    out.append(ArcPointer[Tensor](enc.encode(img1, ctx)))
    if not _empty_ref(ref2_path):
        var img2 = _load_image_512(ref2_path, ctx)
        out.append(ArcPointer[Tensor](enc.encode(img2, ctx)))
    return out^


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    if len(a) >= 2:
        cfg_path = String(a[1])
    var cfg = read_model_config(cfg_path)
    var prompt_file = cfg.validation_prompts_file.copy()
    if len(a) >= 3:
        prompt_file = String(a[2])
    if prompt_file == String(""):
        raise Error("klein_edit_mojo: config must provide validation_prompts_file or argv[2]")
    var prompt_id = String("")
    if len(a) >= 4:
        prompt_id = String(a[3])
    var sample_cfg = read_sample_prompt_config(prompt_file)
    var prompt = _select_prompt(sample_cfg, prompt_id)

    var out_png = String("/home/alex/mojodiffusion/output/klein_edit_") + prompt.label + String(".png")
    if len(a) >= 5:
        out_png = String(a[4])
    var ref1_path = String("")
    if len(a) >= 6:
        ref1_path = String(a[5])
    if _empty_ref(ref1_path):
        raise Error("klein_edit_mojo: argv[5] reference image 1 is required")
    var ref2_path = String("")
    if len(a) >= 7:
        ref2_path = String(a[6])
        if _empty_ref(ref2_path):
            ref2_path = String("")
    var denoise_strength = Float32(1.0)
    if len(a) >= 8:
        denoise_strength = Float32(Float64(String(a[7])))
    var edit_shift = Float32(2.02)
    if len(a) >= 9:
        edit_shift = Float32(Float64(String(a[8])))
    var reference_t_offset = Float32(10.0)
    if len(a) >= 10:
        reference_t_offset = Float32(Float64(String(a[9])))

    _require_file(String("checkpoint"), cfg.checkpoint)
    _require_file(String("VAE"), cfg.vae)
    _require_file(String("reference image 1"), ref1_path)
    if ref2_path != String(""):
        _require_file(String("reference image 2"), ref2_path)
    validate_klein_cap_cache_header(prompt.caps_pos, cfg.joint_attention_dim)
    validate_klein_cap_cache_header(prompt.caps_neg, cfg.joint_attention_dim)
    if cfg.n_heads != H_9B or cfg.head_dim != Dh:
        raise Error(
            String("klein_edit_mojo: bounded to Klein-9B (n_heads=32, head_dim=128); config has ")
            + String(cfg.n_heads) + String("/") + String(cfg.head_dim)
        )
    if not (prompt.width == 512 and prompt.height == 512):
        raise Error(
            String("klein_edit_mojo: bounded to 512x512 prompts; got ")
            + String(prompt.width) + String("x") + String(prompt.height)
        )

    var n_refs = 1
    if ref2_path != String(""):
        n_refs = 2
    print("=== Klein-9B multi-reference edit:", cfg.name, "@ 512^2,", n_refs, "reference(s) ===")
    print("  prompts:", prompt_file, " id:", prompt.label)
    print("  ref1:", ref1_path)
    if n_refs == 2:
        print("  ref2:", ref2_path)
    print(
        "  denoise:", denoise_strength, " shift:", edit_shift,
        " ref_t base:", reference_t_offset,
        " steps:", prompt.steps, " cfg:", prompt.cfg, " seed:", prompt.seed,
    )
    print("  output:", out_png)

    var ctx = DeviceContext()
    var caps = load_caps(prompt.caps_pos, prompt.caps_neg, ctx)
    var pos_txt = reshape(caps.pos, [N_TXT, cfg.joint_attention_dim], ctx)
    var neg_txt = reshape(caps.neg, [N_TXT, cfg.joint_attention_dim], ctx)

    # Stage 1: VAE-encode the reference(s); encoder frees on return.
    var refs = _encode_refs_512(ref1_path, ref2_path, cfg, ctx)

    # Stage 2+3: denoise with reference conditioning, then decode + save.
    if n_refs == 2:
        var ref_a = refs[0][].clone(ctx)
        var ref_b = refs[1][].clone(ctx)
        var _img2 = klein_sample_with_reference_latents2[
            N_IMG_512, N_EDIT2_IMG_512, N_TXT, S_EDIT2_512, LH_512, LW_512, H_9B, Dh
        ](
            cfg, String(""), pos_txt, neg_txt, prompt.cfg, prompt.steps,
            prompt.seed, ref_a^, ref_b^, out_png, ctx, denoise_strength,
            edit_shift, reference_t_offset,
        )
    else:
        var ref_a = refs[0][].clone(ctx)
        var _img1 = klein_sample_with_reference_latent[
            N_IMG_512, N_EDIT1_IMG_512, N_TXT, S_EDIT1_512, LH_512, LW_512, H_9B, Dh
        ](
            cfg, String(""), pos_txt, neg_txt, prompt.cfg, prompt.steps,
            prompt.seed, ref_a^, out_png, ctx, denoise_strength,
            edit_shift, reference_t_offset,
        )
    print("DONE klein multi-reference edit ->", out_png)
