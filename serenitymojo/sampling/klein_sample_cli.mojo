# sampling/klein_sample_cli.mojo — standalone Klein staged sampler entry
# (the OneTrainer klein_lora_infer / sample-stage analogue). Runs as its OWN
# process so NO training stack co-resides — one big model on the GPU at a time.
#
# Usage:
#   pixi run mojo run -I . serenitymojo/sampling/klein_sample_cli.mojo \
#       [config.json] [lora.safetensors] [sample_prompts.json] [prompt_id] [out.png]
#   argv[1] config (default klein9b.json)
#   argv[2] LoRA path ("" -> base model)
#   argv[3] shared sample prompt JSON (default config.validation_prompts_file)
#   argv[4] prompt id/label (default first prompt)
#   argv[5] output PNG override
#
# Caps: positive + negative prompt embeddings are precomputed by the separate
# encoder process and loaded from raw cap-cache files. This matches the verified
# Klein inference smoke path without loading the text encoder in-process.
#
# Mojo constraint: the attention shape + resolution are COMPTIME. Keep the CLI
# as ONE file and dispatch runtime resolution choices to finite comptime
# specializations inside this file. Do not create one CLI file per resolution.

from sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.tensor import Tensor
from serenitymojo.training.validation_sampler import load_caps
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.sampling.klein_sampler import klein_sample
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.ops.cast import cast_tensor_if_needed
from serenitymojo.io.dtype import STDtype
from std.collections import List


comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/alina_train"

comptime N_TXT = 512
comptime H = 32
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
    var pos_txt = cast_tensor_if_needed(
        reshape(caps.pos, txt_sh^, ctx), STDtype.F32, ctx
    )
    return pos_txt^


def _load_neg_txt(
    cfg: TrainConfig, prompt: SamplePrompt, ctx: DeviceContext
) raises -> Tensor:
    var caps = load_caps(prompt.caps_pos, prompt.caps_neg, ctx)
    var txt_sh = List[Int]()
    txt_sh.append(N_TXT)
    txt_sh.append(cfg.joint_attention_dim)
    var neg_txt = cast_tensor_if_needed(
        reshape(caps.neg, txt_sh^, ctx), STDtype.F32, ctx
    )
    return neg_txt^


def _sample_512(
    cfg: TrainConfig, lora_path: String, prompt: SamplePrompt, out_png: String, ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    var _img = klein_sample[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H, Dh](
        cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
    )


def _sample_1024(
    cfg: TrainConfig, lora_path: String, prompt: SamplePrompt, out_png: String, ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    var _img = klein_sample[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H, Dh](
        cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, prompt.seed, out_png, ctx,
    )


def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if p.label == wanted:
            return p^
    raise Error(String("klein_sample_cli: no prompt id ") + wanted)


def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    if len(a) >= 2:
        cfg_path = String(a[1])
    var lora_path = String("")
    if len(a) >= 3:
        lora_path = String(a[2])
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

    print("=== Klein staged sampler:", cfg.name, "@", resolution, "^2 ===")
    if lora_path != String(""):
        print("  config:", cfg_path, " lora:", lora_path)
    else:
        print("  config:", cfg_path, " lora: (none, base model)")
    print("  prompts:", prompt_file, " id:", prompt.label)
    print("  output:", out_png)

    if prompt.width == 512 and prompt.height == 512:
        _sample_512(cfg, lora_path, prompt, out_png, ctx)
    elif prompt.width == 1024 and prompt.height == 1024:
        _sample_1024(cfg, lora_path, prompt, out_png, ctx)
    else:
        raise Error(
            String("klein_sample_cli: unsupported resolution ")
            + String(prompt.width) + String("x") + String(prompt.height)
            + String(" (supported: 512, 1024)")
        )
    print("DONE staged sample ->", out_png)
