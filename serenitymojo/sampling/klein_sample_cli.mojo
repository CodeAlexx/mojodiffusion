# sampling/klein_sample_cli.mojo — standalone Klein staged sampler entry
# (the OneTrainer klein_lora_infer / sample-stage analogue). Runs as its OWN
# process so NO training stack co-resides — one big model on the GPU at a time.
#
# Usage:
#   pixi run mojo run -I . serenitymojo/sampling/klein_sample_cli.mojo \
#       [config.json] [lora.safetensors] [sample_prompts.json] [prompt_id] [out.png]
#   argv[1] config (default klein9b.json)
#   argv[2] LoRA path (""/"base"/"-" -> base model)
#   argv[3] shared sample prompt JSON (default config.validation_prompts_file)
#   argv[4] prompt id/label (default first prompt)
#   argv[5] output PNG override
#   argv[6] optional post-patch/post-pack initial-noise tensor bin for parity replay
#
# Caps: positive + negative prompt embeddings are precomputed by the separate
# encoder process and loaded from raw cap-cache files. This matches the verified
# Klein inference smoke path without loading the text encoder in-process.
#
# Mojo constraint: the attention shape + resolution are COMPTIME. Keep the CLI
# as ONE file and dispatch runtime resolution choices to finite comptime
# specializations inside this file. Do not create one CLI file per resolution.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.tensor import Tensor
from serenitymojo.training.validation_sampler import load_caps
from serenitymojo.io.cap_cache import load_tensor_bin, validate_klein_cap_cache_header
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.sampling.klein_sampler import (
    klein_sample, klein_sample_with_initial_noise, klein_sample_with_reference_latent,
)
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
from serenitymojo.io.dtype import STDtype
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from image.transform import resize_bilinear
from std.collections import List


comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/alina_train"

comptime N_TXT = 512
comptime H_9B = 32
comptime H_4B = 24
comptime Dh = 128
comptime LH_512 = 32
comptime LW_512 = 32
comptime N_IMG_512 = 1024
comptime S_512 = N_IMG_512 + N_TXT
comptime N_EDIT_IMG_512 = 2 * N_IMG_512
comptime S_EDIT_512 = N_EDIT_IMG_512 + N_TXT
comptime LH_1024 = 64
comptime LW_1024 = 64
comptime N_IMG_1024 = 4096
comptime S_1024 = N_IMG_1024 + N_TXT
comptime N_EDIT_IMG_1024 = 2 * N_IMG_1024
comptime S_EDIT_1024 = N_EDIT_IMG_1024 + N_TXT


def _load_pos_txt(
    cfg: TrainConfig, prompt: SamplePrompt, ctx: DeviceContext
) raises -> Tensor:
    var caps = load_caps(prompt.caps_pos, prompt.caps_neg, ctx)
    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var pos_txt = reshape(caps.pos, txt_sh^, ctx)
    return pos_txt^


def _load_neg_txt(
    cfg: TrainConfig, prompt: SamplePrompt, ctx: DeviceContext
) raises -> Tensor:
    var caps = load_caps(prompt.caps_pos, prompt.caps_neg, ctx)
    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var neg_txt = reshape(caps.neg, txt_sh^, ctx)
    return neg_txt^


def _reference_image_empty(path: String) -> Bool:
    return path == String("") or path == String("-") or path == String("none")


def _encode_reference_512(path: String, cfg: TrainConfig, ctx: DeviceContext) raises -> Tensor:
    var img = decode_image_any(path)
    var resized = resize_bilinear(img, 512, 512)
    var host = image_to_signed_nchw(resized)
    var image_t = Tensor.from_host(host, [1, 3, 512, 512], STDtype.F32, ctx)
    print("[klein-edit] reference image", path, "(", img.width, "x", img.height, ") -> 512x512 VAE encode")
    var enc = KleinVaeEncoder[512, 512].load(cfg.vae, ctx)
    return enc.encode(image_t, ctx)


def _encode_reference_1024(path: String, cfg: TrainConfig, ctx: DeviceContext) raises -> Tensor:
    var img = decode_image_any(path)
    var resized = resize_bilinear(img, 1024, 1024)
    var host = image_to_signed_nchw(resized)
    var image_t = Tensor.from_host(host, [1, 3, 1024, 1024], STDtype.F32, ctx)
    print("[klein-edit] reference image", path, "(", img.width, "x", img.height, ") -> 1024x1024 VAE encode")
    var enc = KleinVaeEncoder[1024, 1024].load(cfg.vae, ctx)
    return enc.encode(image_t, ctx)


def _sample_512(
    cfg: TrainConfig,
    lora_path: String,
    prompt: SamplePrompt,
    out_png: String,
    initial_noise_path: String,
    reference_image_path: String,
    denoise_strength: Float32,
    edit_shift: Float32,
    reference_t_offset: Float32,
    ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_cli: unsupported head_dim ") + String(cfg.head_dim))
    var has_reference = not _reference_image_empty(reference_image_path)
    if has_reference and initial_noise_path != String(""):
        raise Error("klein_sample_cli: reference edit and initial-noise sidecar are mutually exclusive")
    if cfg.n_heads == H_9B:
        if has_reference:
            var ref9 = _encode_reference_512(reference_image_path, cfg, ctx)
            var _edit9 = klein_sample_with_reference_latent[
                N_IMG_512, N_EDIT_IMG_512, N_TXT, S_EDIT_512, LH_512, LW_512, H_9B, Dh
            ](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps,
                prompt.seed, ref9^, out_png, ctx, denoise_strength, edit_shift,
                reference_t_offset,
            )
        elif initial_noise_path != String(""):
            var noise9 = load_tensor_bin(initial_noise_path, ctx)
            var _img9p = klein_sample_with_initial_noise[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, noise9^, out_png, ctx,
            )
        else:
            var _img9 = klein_sample[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
            )
    elif cfg.n_heads == H_4B:
        if has_reference:
            var ref4 = _encode_reference_512(reference_image_path, cfg, ctx)
            var _edit4 = klein_sample_with_reference_latent[
                N_IMG_512, N_EDIT_IMG_512, N_TXT, S_EDIT_512, LH_512, LW_512, H_4B, Dh
            ](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps,
                prompt.seed, ref4^, out_png, ctx, denoise_strength, edit_shift,
                reference_t_offset,
            )
        elif initial_noise_path != String(""):
            var noise4 = load_tensor_bin(initial_noise_path, ctx)
            var _img4p = klein_sample_with_initial_noise[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_4B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, noise4^, out_png, ctx,
            )
        else:
            var _img4 = klein_sample[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_4B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
            )
    else:
        raise Error(String("klein_sample_cli: unsupported num_heads ") + String(cfg.n_heads))


def _sample_1024(
    cfg: TrainConfig,
    lora_path: String,
    prompt: SamplePrompt,
    out_png: String,
    initial_noise_path: String,
    reference_image_path: String,
    denoise_strength: Float32,
    edit_shift: Float32,
    reference_t_offset: Float32,
    ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_cli: unsupported head_dim ") + String(cfg.head_dim))
    var has_reference = not _reference_image_empty(reference_image_path)
    if has_reference and initial_noise_path != String(""):
        raise Error("klein_sample_cli: reference edit and initial-noise sidecar are mutually exclusive")
    if cfg.n_heads == H_9B:
        if has_reference:
            var ref9 = _encode_reference_1024(reference_image_path, cfg, ctx)
            var _edit9 = klein_sample_with_reference_latent[
                N_IMG_1024, N_EDIT_IMG_1024, N_TXT, S_EDIT_1024, LH_1024, LW_1024, H_9B, Dh
            ](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps,
                prompt.seed, ref9^, out_png, ctx, denoise_strength, edit_shift,
                reference_t_offset,
            )
        elif initial_noise_path != String(""):
            var noise9 = load_tensor_bin(initial_noise_path, ctx)
            var _img9p = klein_sample_with_initial_noise[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, noise9^, out_png, ctx,
            )
        else:
            var _img9 = klein_sample[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
            )
    elif cfg.n_heads == H_4B:
        if has_reference:
            var ref4 = _encode_reference_1024(reference_image_path, cfg, ctx)
            var _edit4 = klein_sample_with_reference_latent[
                N_IMG_1024, N_EDIT_IMG_1024, N_TXT, S_EDIT_1024, LH_1024, LW_1024, H_4B, Dh
            ](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps,
                prompt.seed, ref4^, out_png, ctx, denoise_strength, edit_shift,
                reference_t_offset,
            )
        elif initial_noise_path != String(""):
            var noise4 = load_tensor_bin(initial_noise_path, ctx)
            var _img4p = klein_sample_with_initial_noise[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_4B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, noise4^, out_png, ctx,
            )
        else:
            var _img4 = klein_sample[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_4B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
            )
    else:
        raise Error(String("klein_sample_cli: unsupported num_heads ") + String(cfg.n_heads))


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
        raise Error(String("klein_sample_cli: missing ") + label + String(": ") + path)


def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if p.label == wanted:
            return p^
    raise Error(String("klein_sample_cli: no prompt id ") + wanted)


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    if len(a) >= 2:
        cfg_path = String(a[1])
    var lora_path = String("")
    if len(a) >= 3:
        lora_path = String(a[2])
        if lora_path == String("base") or lora_path == String("-"):
            lora_path = String("")
    var cfg = read_model_config(cfg_path)
    var prompt_file = cfg.validation_prompts_file.copy()
    if len(a) >= 4:
        prompt_file = String(a[3])
    if prompt_file == String(""):
        raise Error("klein_sample_cli: config must provide validation_prompts_file or argv[3]")
    var prompt_id = String("")
    if len(a) >= 5:
        prompt_id = String(a[4])
    var sample_cfg = read_sample_prompt_config(prompt_file)
    var prompt = _select_prompt(sample_cfg, prompt_id)
    var resolution = String(prompt.width)

    var out_png = String(OUT_DIR) + String("/staged_sample_") + prompt.label + String("_") + resolution + String(".png")
    if len(a) >= 6:
        out_png = String(a[5])
    var initial_noise_path = String("")
    if len(a) >= 7:
        initial_noise_path = String(a[6])
        if initial_noise_path == String("-") or initial_noise_path == String("none"):
            initial_noise_path = String("")
    var reference_image_path = String("")
    if len(a) >= 8:
        reference_image_path = String(a[7])
        if _reference_image_empty(reference_image_path):
            reference_image_path = String("")
    var denoise_strength = Float32(1.0)
    if len(a) >= 9:
        denoise_strength = Float32(Float64(String(a[8])))
    var edit_shift = Float32(2.02)
    if len(a) >= 10:
        edit_shift = Float32(Float64(String(a[9])))
    var reference_t_offset = Float32(10.0)
    if len(a) >= 11:
        reference_t_offset = Float32(Float64(String(a[10])))

    _require_file(String("checkpoint"), cfg.checkpoint)
    _require_file(String("VAE"), cfg.vae)
    if lora_path != String(""):
        _require_file(String("LoRA"), lora_path)
    if initial_noise_path != String(""):
        _require_file(String("initial-noise sidecar"), initial_noise_path)
    if reference_image_path != String(""):
        _require_file(String("reference image"), reference_image_path)
    validate_klein_cap_cache_header(prompt.caps_pos, cfg.joint_attention_dim)
    validate_klein_cap_cache_header(prompt.caps_neg, cfg.joint_attention_dim)
    if not (
        (prompt.width == 512 and prompt.height == 512)
        or (prompt.width == 1024 and prompt.height == 1024)
    ):
        raise Error(
            String("klein_sample_cli: unsupported resolution ")
            + String(prompt.width) + String("x") + String(prompt.height)
            + String(" (supported: 512, 1024)")
        )

    print("=== Klein staged sampler:", cfg.name, "@", resolution, "^2 ===")
    if lora_path != String(""):
        print("  config:", cfg_path, " lora:", lora_path)
    else:
        print("  config:", cfg_path, " lora: (none, base model)")
    print("  prompts:", prompt_file, " id:", prompt.label)
    print("  output:", out_png)
    if initial_noise_path != String(""):
        print("  parity initial noise (post-patch/post-pack):", initial_noise_path)
    if reference_image_path != String(""):
        print(
            "  ReferenceLatent edit:", reference_image_path,
            " denoise:", denoise_strength,
            " shift:", edit_shift,
            " ref_t:", reference_t_offset,
        )

    var ctx = DeviceContext()
    if prompt.width == 512 and prompt.height == 512:
        _sample_512(
            cfg, lora_path, prompt, out_png, initial_noise_path,
            reference_image_path, denoise_strength, edit_shift, reference_t_offset,
            ctx,
        )
    elif prompt.width == 1024 and prompt.height == 1024:
        _sample_1024(
            cfg, lora_path, prompt, out_png, initial_noise_path,
            reference_image_path, denoise_strength, edit_shift, reference_t_offset,
            ctx,
        )
    print("DONE staged sample ->", out_png)
