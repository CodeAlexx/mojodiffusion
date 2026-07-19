#!/usr/bin/env python
# ---------------------------------------------------------------------------
# nldiffusion/generate.py  —  custom-prompt T2I CLI for nvidia/NL-Diffusion-Image
# on a VRAM-constrained card (built + run on a 16GB RTX 5080).
#
# Drives the NVIDIA creator-source model (the parity authority for the pure-Mojo
# port in this dir). The pure-Mojo runtime is gated to a fixed prompt (no in-Mojo
# tokenizer yet), so arbitrary prompts run through this reference harness.
#
# Two things make 1024^2 fit 16GB where the stock demo OOMs:
#   1. MEMORY-LEAN CFG: the stock loop batches cond+uncond -> [2,4096,131072]
#      logits (+ a full softmax). We run the two forwards SEQUENTIALLY at batch 1
#      and combine, so peak logits is [1,4096,131072], not [2,...].
#   2. NO SOFTMAX on the deterministic path: confidence_policy=stratified selects
#      by position, not confidence, so we skip probs/x0_p entirely (argmax only).
#
# Two stages, separate processes (decode reloads the 1.8GB VQ into freed VRAM):
#   stage 1 (this proc)   : sample the 64x64 token grid, save to --tokens
#   stage 2 (--_decode)   : Emu3 VQ-decode the grid -> PNG
# main() runs stage 1 then subprocess-invokes stage 2 unless --no-decode.
#
# Examples:
#   python generate.py -p "a red apple on a wooden table" -o apple.png
#   python generate.py -p "..." -o out.png --gs 3.0         # memory-lean CFG
#   python generate.py -p "..." -o out.png --policy mmada --sample multinomial --seed 7
# ---------------------------------------------------------------------------
import os, sys, argparse, math, time, warnings, subprocess, tempfile
warnings.filterwarnings("ignore")

DIR_DEFAULT = "/mnt/disk1/models/NL-Diffusion-Image"
GEN_SHAPE_MAP = {1024: (64, 64), 512: (32, 32), 256: (16, 16)}


def _env_prelude():
    os.environ.setdefault("DEBUG_FIX_PADDING", "1")
    os.environ.setdefault("NOT_ALWASY_DO_2DPOOL", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    ap = argparse.ArgumentParser(description="NL-Diffusion-Image custom-prompt T2I (VRAM-lean)")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-o", "--out", default="nldiff_out.png", help="output PNG path")
    ap.add_argument("--model-dir", default=DIR_DEFAULT)
    ap.add_argument("--res", type=int, default=1024, choices=[1024],
                    help="release supports 1024^2 only (downsample reserve-slot convention)")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gs", type=float, default=0.0,
                    help="CFG scale; >0 uses memory-lean sequential CFG")
    ap.add_argument("--cfg-lo", type=float, default=0.0, help="cfg_interval start (frac of steps)")
    ap.add_argument("--cfg-hi", type=float, default=1.0, help="cfg_interval end (frac of steps)")
    ap.add_argument("--sample", default="argmax", choices=["argmax", "multinomial"])
    ap.add_argument("--policy", default="stratified", choices=["stratified", "mmada", "mask_git"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.86)
    ap.add_argument("--alg-temp", type=float, default=1.0)
    ap.add_argument("--min-temp", type=float, default=0.01)
    ap.add_argument("--gpu-cap", default="6GiB", help="device_map GPU budget for the backbone")
    ap.add_argument("--tokens", default=None, help="token-grid safetensors path (default: temp)")
    ap.add_argument("--no-decode", action="store_true", help="stop after the token grid")
    ap.add_argument("--keep-tokens", action="store_true")
    # hidden stage entries (run as separate processes so GPU frees between them)
    ap.add_argument("--_gen", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_decode", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args()


# ------------------------------- stage 1 -----------------------------------
def run_generate(a):
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
    from safetensors.torch import save_file

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = AutoConfig.from_pretrained(a.model_dir, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    print(f"[load] backbone bf16/sdpa, GPU cap {a.gpu_cap} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory={0: a.gpu_cap, "cpu": "120GiB"}, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    model.eval()
    model.config.dlm_paradigm = "bidirectional"       # forces use_cache=False (no KV branch)
    tok = PreTrainedTokenizerFast.from_pretrained(a.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id; tok.pad_token = tok.eos_token

    m = sys.modules[type(model).__module__]            # creator-source module internals
    NC = m._NC
    dev = model.get_model().device
    gm = model.get_model()
    img_mask_id = 131073
    gen_shape = GEN_SHAPE_MAP[a.res]
    n_tokens = gen_shape[0] * gen_shape[1]
    n_tokens_txt = 1024 if a.res == 1024 else n_tokens

    # bidirectional attention mode on every layer (matches text_to_image setup).
    # get_model() returns model.encoder (a Ministral3Model); its decoder layers
    # are at .layers directly (text_to_image reaches them via model.encoder.layers).
    for layer in gm.layers:
        layer.self_attn.mode = "bidirectional"
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = True

    # ---- build the prompt / gen-token scaffold exactly like text_to_image ----
    micro = "ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5"
    prompt_full = f"{a.prompt} {micro}"
    question = "Generate an image with the caption:\n <prompt>".replace("<prompt>", prompt_full)
    conv = m._MinistralConv()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1],
                        f"Sure {NC.gen_im_start_token}{NC.reserve_id_token * n_tokens_txt}{NC.gen_im_end_token}")
    prompt_question = conv.get_prompt()

    input_ids = m._tokenizer_image_token(prompt_question, tok, return_tensors="pt").unsqueeze(0).to(dev)
    is_gen = input_ids == NC.reserve_id
    is_eot = torch.where(input_ids == NC.eos_id)[1]
    assert len(is_eot) == 3, f"expected 3 EOT, got {len(is_eot)}"
    prompt_cutoff = is_eot[1]
    is_prompt = torch.zeros_like(input_ids, dtype=torch.bool)
    is_prompt[:, :prompt_cutoff + 1] = True

    inputs_embeds = gm.embed_tokens(input_ids)
    inputs_embeds_uncond = inputs_embeds.clone()
    noise_embed = gm.embed_tokens(torch.tensor([NC.mask_id], device=dev))
    inputs_embeds_uncond[is_prompt] = noise_embed

    # ---- schedules ----
    xt = torch.full((1, n_tokens), img_mask_id, dtype=torch.long, device=dev)
    num_transfer_tokens = m._get_num_transfer_tokens(
        xt == img_mask_id, a.steps, schedule="shift", shift=a.shift)
    sch_t = np.linspace(0, 1, a.steps)
    sch_temps = (1.0 - sch_t) * (1.0 - a.min_temp) + a.min_temp
    sch_temps = torch.tensor(sch_temps, device=dev, dtype=torch.float32)
    cfg_start = int(a.cfg_lo * a.steps); cfg_end = int(a.cfg_hi * a.steps)

    unmask_order = None
    if a.policy == "stratified":
        unmask_order = m._stratified_random(n=gen_shape[0], seed=a.seed, shuffle_blocks=True)

    need_probs = (a.policy in ("mmada", "mask_git")) or (a.sample == "multinomial")
    if a.gs > 0 and need_probs:
        print("[warn] gs>0 with a stochastic/confidence policy keeps a softmax alloc; "
              "argmax+stratified is the fully VRAM-lean path.", flush=True)

    def forward_logits(embeds_curr, tmask, timesteps_in):
        all_emb, ntm = m._t2i_wte(gm, None, gen_shape=gen_shape, x_gen=xt,
                                  inputs_embeds_curr=embeds_curr, new_token_mask=tmask)
        return m._t2i_get_logits(gm, all_emb, ntm, past_key_values=None, gen_shape=gen_shape,
                                 input_modality_indices=ntm, timesteps=timesteps_in)

    # ---- the diffusion loop (bidirectional, no KV cache) ----
    from tqdm import tqdm
    t0 = time.time()
    temp_idx = 0; x0 = xt
    print(f"[gen] '{a.prompt[:70]}...'  steps={a.steps} gs={a.gs} "
          f"{a.sample}+{a.policy} res={a.res}^2", flush=True)
    with torch.no_grad(), torch.inference_mode():
        for _, num_transfer in tqdm(enumerate(num_transfer_tokens[0]),
                                    total=num_transfer_tokens.shape[1]):
            local_temp = sch_temps[temp_idx]; temp_idx += 1
            mask_idx = xt == img_mask_id
            n_mask = mask_idx.sum()
            timesteps = (n_mask / mask_idx.numel()).view(1)
            do_cfg = a.gs > 0 and cfg_start <= temp_idx <= cfg_end

            if do_cfg:
                # MEMORY-LEAN: two sequential batch-1 forwards, not one batch-2
                logits_un = forward_logits(inputs_embeds_uncond.clone(), is_gen.clone(), timesteps)
                logits = forward_logits(inputs_embeds.clone(), is_gen.clone(), timesteps)
                ninf = logits == -np.inf
                logits = (1.0 + a.gs) * logits - a.gs * logits_un
                logits[ninf] = -np.inf
                del logits_un
            else:
                logits = forward_logits(inputs_embeds.clone(), is_gen.clone(), timesteps)

            if a.sample == "argmax":
                x0 = logits.argmax(-1)
            else:  # multinomial
                import torch.distributions as dists
                x0 = dists.Categorical(logits=logits / a.temperature).sample()

            x0_p = None
            if need_probs:
                probs = logits.softmax(dim=-1)
                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
                del probs
            del logits

            x0 = torch.where(mask_idx, x0, xt)

            if a.policy == "stratified":
                start = n_tokens - n_mask
                select_index = torch.tensor(unmask_order[start:start + num_transfer],
                                            device=dev, dtype=torch.long)
            elif a.policy == "mmada":
                _at = a.alg_temp
                conf = torch.log(x0_p.clamp(1e-20)) + _at * m._gumbel_noise(x0_p)
                conf = torch.where(mask_idx, conf, torch.tensor(-np.inf, device=dev))
                _, select_index = torch.topk(conf[0], k=int(num_transfer))
            else:  # mask_git
                _at = a.alg_temp
                conf = torch.where(mask_idx, x0_p / _at, torch.tensor(-np.inf, device=dev))
                conf = torch.softmax(conf, dim=-1)
                select_index = torch.multinomial(conf, num_samples=int(num_transfer))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            transfer_index[0, select_index] = True
            xt[transfer_index] = x0[transfer_index]

    xt2 = x0.clone()
    xt2[xt == img_mask_id] = x0[xt == img_mask_id]
    x0_img = xt2
    dt = time.time() - t0
    n_unique = int(x0_img.unique().numel())
    print(f"[gen] done {dt:.0f}s ({dt/a.steps:.2f}s/step), {n_unique} unique tokens", flush=True)

    tokens_path = a.tokens or tempfile.mktemp(prefix="nldiff_tok_", suffix=".safetensors")
    save_file({"x0_img": x0_img.detach().cpu().float()}, tokens_path)
    return tokens_path, n_unique


# ------------------------------- stage 2 -----------------------------------
def run_decode(a):
    import json, types, importlib.util
    from pathlib import Path
    import numpy as np, torch
    from safetensors.torch import load_file
    from PIL import Image

    vq_path = Path(a.model_dir) / "emu3_vqvae"
    pkg = f"_emu3_vqvae_{vq_path.name}"
    mm = types.ModuleType(pkg); mm.__path__ = [str(vq_path)]; mm.__package__ = pkg
    sys.modules[pkg] = mm

    def L(n, f):
        s = importlib.util.spec_from_file_location(
            f"{pkg}.{n}", vq_path / f, submodule_search_locations=[str(vq_path)])
        md = importlib.util.module_from_spec(s); md.__package__ = pkg
        sys.modules[f"{pkg}.{n}"] = md; s.loader.exec_module(md); return md

    cfgm = L("configuration_emu3p5visionvq", "configuration_emu3p5visionvq.py")
    mdl = L("modeling_emu3p5visionvq", "modeling_emu3p5visionvq.py")
    cfgd = json.load(open(vq_path / "config.json"))
    v = mdl.Emu3p5VisionVQModel(cfgm.Emu3p5VisionVQConfig(**cfgd))
    v.load_state_dict(load_file(str(vq_path / "model.safetensors")))
    v = v.to("cuda").float().eval()
    for pr in v.parameters():
        pr.requires_grad_(False)

    from einops import rearrange
    ids = load_file(a.tokens)["x0_img"].long().to("cuda")
    hw = int(math.sqrt(ids.shape[-1]))
    with torch.no_grad():
        cb = v.quantize.get_codebook_entry(ids)
        z = rearrange(cb, "b (h w) d -> b d h w", h=hw, w=hw)
        img = v.decode(z).float().clamp(-1, 1)
    arr = ((img + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    Image.fromarray(arr).save(a.out)
    print(f"[decode] SAVED {a.out}  {arr.shape}  range[{arr.min()},{arr.max()}]", flush=True)


def _gen_cmd(a, tokens_path):
    return [sys.executable, os.path.abspath(__file__), "--_gen",
            "-p", a.prompt, "--tokens", tokens_path, "--model-dir", a.model_dir,
            "--res", str(a.res), "--steps", str(a.steps), "--gs", str(a.gs),
            "--cfg-lo", str(a.cfg_lo), "--cfg-hi", str(a.cfg_hi),
            "--sample", a.sample, "--policy", a.policy, "--seed", str(a.seed),
            "--shift", str(a.shift), "--temperature", str(a.temperature),
            "--alg-temp", str(a.alg_temp), "--min-temp", str(a.min_temp),
            "--gpu-cap", a.gpu_cap]


def main():
    _env_prelude()
    a = parse_args()
    if a._decode:
        run_decode(a)
        return
    if a._gen:
        run_generate(a)                 # a.tokens set by the orchestrator
        return

    # orchestrator: gen and decode each run as their OWN process, so the ~11GB
    # backbone CUDA context is fully released before the VQ-VAE loads for decode.
    tokens_path = a.tokens or tempfile.mktemp(prefix="nldiff_tok_", suffix=".safetensors")
    subprocess.run(_gen_cmd(a, tokens_path), check=True)
    if a.no_decode:
        print(f"[gen] token grid at {tokens_path} (decode skipped)")
        return
    print("[decode] launching VQ decode subprocess (fresh VRAM) ...", flush=True)
    subprocess.run([sys.executable, os.path.abspath(__file__), "--_decode",
                    "--model-dir", a.model_dir, "--tokens", tokens_path,
                    "-o", a.out, "-p", a.prompt], check=True)
    if not a.keep_tokens and not tokens_path.startswith(
            os.path.dirname(os.path.abspath(a.out))):
        try:
            os.remove(tokens_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
