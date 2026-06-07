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
from serenitymojo.sampling.klein_sampler import klein_sample, klein_sample_with_initial_noise
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY
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
comptime LH_1024 = 64
comptime LW_1024 = 64
comptime N_IMG_1024 = 4096
comptime S_1024 = N_IMG_1024 + N_TXT


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


def _sample_512(
    cfg: TrainConfig,
    lora_path: String,
    prompt: SamplePrompt,
    out_png: String,
    initial_noise_path: String,
    ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_cli: unsupported head_dim ") + String(cfg.head_dim))
    if cfg.n_heads == H_9B:
        if initial_noise_path != String(""):
            var noise9 = load_tensor_bin(initial_noise_path, ctx)
            var _img9p = klein_sample_with_initial_noise[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, noise9^, out_png, ctx,
            )
        else:
            var _img9 = klein_sample[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
            )
    elif cfg.n_heads == H_4B:
        if initial_noise_path != String(""):
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
    ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    if cfg.head_dim != Dh:
        raise Error(String("klein_sample_cli: unsupported head_dim ") + String(cfg.head_dim))
    if cfg.n_heads == H_9B:
        if initial_noise_path != String(""):
            var noise9 = load_tensor_bin(initial_noise_path, ctx)
            var _img9p = klein_sample_with_initial_noise[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, noise9^, out_png, ctx,
            )
        else:
            var _img9 = klein_sample[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_9B, Dh](
                cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
            )
    elif cfg.n_heads == H_4B:
        if initial_noise_path != String(""):
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

    _require_file(String("checkpoint"), cfg.checkpoint)
    _require_file(String("VAE"), cfg.vae)
    if lora_path != String(""):
        _require_file(String("LoRA"), lora_path)
    if initial_noise_path != String(""):
        _require_file(String("initial-noise sidecar"), initial_noise_path)
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

    var ctx = DeviceContext()
    if prompt.width == 512 and prompt.height == 512:
        _sample_512(cfg, lora_path, prompt, out_png, initial_noise_path, ctx)
    elif prompt.width == 1024 and prompt.height == 1024:
        _sample_1024(cfg, lora_path, prompt, out_png, initial_noise_path, ctx)
    print("DONE staged sample ->", out_png)
