# sampling/klein_sampler_parity_dump_cli.mojo -- CLI for Mojo-side Klein sampler
# parity artifacts.
#
# Usage:
#   pixi run mojo run -I . serenitymojo/sampling/klein_sampler_parity_dump_cli.mojo \
#       [config.json] [lora.safetensors|base|-] [sample_prompts.json] [prompt_id] \
#       [initial_noise_post_pack.bin] [artifact_dir] [manifest.json]
#
# The initial-noise sidecar is required. It must be the OneTrainer-equivalent
# post-patch/post-pack tensor accepted by klein_sample_with_initial_noise:
# [1,128,LH,LW] or [N_IMG,128], in the raw tensor-bin format.

from std.collections import List
from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.io.cap_cache import load_tensor_bin, validate_klein_cap_cache_header
from serenitymojo.io.ffi import O_RDONLY, sys_close, sys_open
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.tensor import Tensor
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt,
    SamplePromptConfig,
    read_sample_prompt_config,
)
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.validation_sampler import load_caps
from serenitymojo.sampling.klein_sampler_parity_dump import (
    dump_klein_sampler_parity_artifacts,
)


comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"
comptime DEFAULT_OUT_DIR = "/tmp/klein_sampler_mojo_artifacts"

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


def _dump_512(
    cfg: TrainConfig,
    lora_path: String,
    prompt: SamplePrompt,
    initial_noise_path: String,
    out_dir: String,
    manifest_path: String,
    ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    if cfg.head_dim != Dh:
        raise Error(String("klein_sampler_parity_dump_cli: unsupported head_dim ") + String(cfg.head_dim))
    if cfg.n_heads == H_9B:
        var noise9 = load_tensor_bin(initial_noise_path, ctx)
        var res9 = dump_klein_sampler_parity_artifacts[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_9B, Dh](
            cfg, lora_path, prompt, pos_txt, neg_txt, noise9^, initial_noise_path,
            out_dir, manifest_path, ctx,
        )
        print("[klein-parity-dump-cli] wrote:", res9.manifest_path)
    elif cfg.n_heads == H_4B:
        var noise4 = load_tensor_bin(initial_noise_path, ctx)
        var res4 = dump_klein_sampler_parity_artifacts[N_IMG_512, N_TXT, S_512, LH_512, LW_512, H_4B, Dh](
            cfg, lora_path, prompt, pos_txt, neg_txt, noise4^, initial_noise_path,
            out_dir, manifest_path, ctx,
        )
        print("[klein-parity-dump-cli] wrote:", res4.manifest_path)
    else:
        raise Error(String("klein_sampler_parity_dump_cli: unsupported num_heads ") + String(cfg.n_heads))


def _dump_1024(
    cfg: TrainConfig,
    lora_path: String,
    prompt: SamplePrompt,
    initial_noise_path: String,
    out_dir: String,
    manifest_path: String,
    ctx: DeviceContext,
) raises:
    var pos_txt = _load_pos_txt(cfg, prompt, ctx)
    var neg_txt = _load_neg_txt(cfg, prompt, ctx)
    if cfg.head_dim != Dh:
        raise Error(String("klein_sampler_parity_dump_cli: unsupported head_dim ") + String(cfg.head_dim))
    if cfg.n_heads == H_9B:
        var noise9 = load_tensor_bin(initial_noise_path, ctx)
        var res9 = dump_klein_sampler_parity_artifacts[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_9B, Dh](
            cfg, lora_path, prompt, pos_txt, neg_txt, noise9^, initial_noise_path,
            out_dir, manifest_path, ctx,
        )
        print("[klein-parity-dump-cli] wrote:", res9.manifest_path)
    elif cfg.n_heads == H_4B:
        var noise4 = load_tensor_bin(initial_noise_path, ctx)
        var res4 = dump_klein_sampler_parity_artifacts[N_IMG_1024, N_TXT, S_1024, LH_1024, LW_1024, H_4B, Dh](
            cfg, lora_path, prompt, pos_txt, neg_txt, noise4^, initial_noise_path,
            out_dir, manifest_path, ctx,
        )
        print("[klein-parity-dump-cli] wrote:", res4.manifest_path)
    else:
        raise Error(String("klein_sampler_parity_dump_cli: unsupported num_heads ") + String(cfg.n_heads))


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
        raise Error(String("klein_sampler_parity_dump_cli: missing ") + label + String(": ") + path)


def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        if p.label == wanted:
            return p^
    raise Error(String("klein_sampler_parity_dump_cli: no prompt id ") + wanted)


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
        raise Error("klein_sampler_parity_dump_cli: config must provide validation_prompts_file or argv[3]")

    var prompt_id = String("")
    if len(a) >= 5:
        prompt_id = String(a[4])
    var initial_noise_path = String("")
    if len(a) >= 6:
        initial_noise_path = String(a[5])
    if initial_noise_path == String(""):
        raise Error("klein_sampler_parity_dump_cli: argv[5] initial-noise sidecar is required")

    var out_dir = String(DEFAULT_OUT_DIR)
    if len(a) >= 7:
        out_dir = String(a[6])
    var manifest_path = String("")
    if len(a) >= 8:
        manifest_path = String(a[7])

    var sample_cfg = read_sample_prompt_config(prompt_file)
    var prompt = _select_prompt(sample_cfg, prompt_id)
    if not (
        (prompt.width == 512 and prompt.height == 512)
        or (prompt.width == 1024 and prompt.height == 1024)
    ):
        raise Error(
            String("klein_sampler_parity_dump_cli: unsupported resolution ")
            + String(prompt.width) + String("x") + String(prompt.height)
            + String(" (supported: 512, 1024)")
        )

    _require_file(String("checkpoint"), cfg.checkpoint)
    _require_file(String("VAE"), cfg.vae)
    if lora_path != String(""):
        _require_file(String("LoRA"), lora_path)
    _require_file(String("initial-noise sidecar"), initial_noise_path)
    validate_klein_cap_cache_header(prompt.caps_pos, cfg.joint_attention_dim)
    validate_klein_cap_cache_header(prompt.caps_neg, cfg.joint_attention_dim)

    print("=== Klein sampler parity dump:", cfg.name, "@", String(prompt.width), "x", String(prompt.height), "===")
    print("  config:", cfg_path)
    print("  lora:", lora_path if lora_path != String("") else String("(none, base model)"))
    print("  prompts:", prompt_file, " id:", prompt.label)
    print("  initial noise:", initial_noise_path)
    print("  artifacts:", out_dir)
    if manifest_path != String(""):
        print("  manifest:", manifest_path)

    var ctx = DeviceContext()
    if prompt.width == 512 and prompt.height == 512:
        _dump_512(cfg, lora_path, prompt, initial_noise_path, out_dir, manifest_path, ctx)
    elif prompt.width == 1024 and prompt.height == 1024:
        _dump_1024(cfg, lora_path, prompt, initial_noise_path, out_dir, manifest_path, ctx)
    print("DONE Klein sampler parity dump ->", out_dir)
