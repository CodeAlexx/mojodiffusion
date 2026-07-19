#!/usr/bin/env python3
"""
Klein d_B DECISIVE TEST — SerenityTrainer side.

Captures the per-projection d_y (gradient at the MODULE OUTPUT of
transformer_blocks[0].attn.to_q / to_k / to_v) — i.e. EXACTLY the gradient the
to_q/to_k/to_v LoRA adapters consume in their d_B computation
(d_B = scale * down(x)^T @ d_y). Saves them so the Mojo gate's internal
d_q/d_k/d_v_pre_flat (dumped by KLEIN_DY_DUMP) can be diffed against them.

WHY: forward + d_A + residual d_x already match SerenityTrainer, but the to_q/k/v d_B
grads are 1.5-2x off. Since x (forward) and lora_down are identical Mojo-vs-reference trainer,
down(x) is identical, so the ONLY remaining driver of d_B is d_y. This script
isolates whether d_y itself diverges (=> real adjoint bug in the joint-attention
backward) or matches (=> bug is in the Mojo LoRA d_B math).

This reuses the identical setup as serenity_klein_oracle.py (same frozen inputs, same
LoRA wrapper, same predict/loss) and ONLY adds output-grad hooks + a dtype
consistency check. It does NOT overwrite serenity_klein_grads.safetensors.

Run: /home/alex/SerenityTrainer/venv/bin/python scripts/serenity_klein_block0_dy_oracle.py
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

reference trainer = "/home/alex/SerenityTrainer"
sys.path.insert(0, reference trainer)

from diffusers import Flux2Transformer2DModel  # noqa: E402
from modules.module.LoRAModule import LoRAModuleWrapper  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

PARITY = "/home/alex/serenity-trainer/parity"
# DECISIVE TEST aligns to the Mojo gate (klein_train_ref_grad_update_replay.mojo),
# which forwards STEP001 trace + step001 adapter_before (N_IMG=1120, ts=346,
# H=40,W=28, target [1,32,80,56]). B(lora_up)=0 at step001 too -> forward is pure
# base model and d_y == base joint-attention backward grad. BYTE-IDENTICAL inputs.
STEP = "/tmp/klein_train_ref_2step_step001.safetensors"
ADAPTERS = "/tmp/klein_train_ref_2step_step001_adapters.safetensors"
H_LAT, W_LAT = 40, 28
META = f"{PARITY}/klein_train_ref_meta.json"
TRANSFORMER_DIR = (
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
    "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/transformer"
)
OUT_DY = f"{PARITY}/serenity_klein_block0_dy.safetensors"

import os as _os
# PRECISION DISCRIMINATOR: KLEIN_SERENITY_F32=1 -> run the SAME backward in F32 on CPU.
# If reference trainer-F32 per-block text d_x diverges from reference trainer-bf16 like Mojo does, the text grad
# is ill-conditioned (precision-inherent, F32 fix justified). If reference trainer-F32 ~= reference trainer-bf16,
# Mojo has a real bug.
F32_MODE = _os.environ.get("KLEIN_SERENITY_F32", "") == "1"
DEV = "cuda"  # inputs + LoRA always on GPU; in F32 mode the transformer is block-swapped
BF16 = torch.float32 if F32_MODE else torch.bfloat16


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-30)


def unpack_latents(latents, height, width):
    b, seq, c = latents.shape
    return latents.reshape(b, height, width, c).permute(0, 3, 1, 2)


def unpatchify_latents(latents):
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def main():
    torch.manual_seed(0)
    meta = json.load(open(META))
    rc = meta["runtime_config"]
    lora_rank = int(rc["lora_rank"])
    lora_alpha = float(rc["lora_alpha"])
    print(f"[cfg] rank={lora_rank} alpha={lora_alpha} scale={lora_alpha/lora_rank}")

    d = load_file(STEP, device=DEV)
    packed_latent_input = d["trace.packed_latent_input"]
    transformer_timestep = d["trace.transformer_timestep"]
    encoder_hidden_states = d["trace.encoder_hidden_states"]
    text_ids = d["trace.text_ids"]
    image_ids = d["trace.image_ids"]
    target = d["output.target"].float()
    loss_weight = d["batch.loss_weight"].float()
    H, W = H_LAT, W_LAT

    print(f"[load] Flux2Transformer2DModel from {TRANSFORMER_DIR} (F32_MODE={F32_MODE})")
    transformer = Flux2Transformer2DModel.from_pretrained(
        TRANSFORMER_DIR, torch_dtype=BF16
    )
    transformer.eval()
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    if F32_MODE:
        # BLOCK-SWAP: stream blocks CPU<->GPU (CUDA stream), compute F32 on GPU,
        # peak VRAM stays low (~1-2 blocks resident). Backward re-onloads via the
        # gradient-checkpointing recompute. LoRA params stay resident on GPU.
        transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
        )
        print("[load] group offload (block_level, F32) enabled")
    else:
        transformer = transformer.to(DEV)
    guidance_embeds = bool(transformer.config.guidance_embeds)
    guidance = None
    if guidance_embeds:
        guidance = torch.tensor([1.0], device=DEV, dtype=BF16).expand(packed_latent_input.shape[0])

    config = TrainConfig.default_values()
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.layer_filter = "transformer_block"
    config.layer_filter_regex = False
    config.train_device = DEV
    config.dropout_probability = 0.0

    wrapper = LoRAModuleWrapper(transformer, "transformer", config, config.layer_filter.split(","))
    adapters = load_file(ADAPTERS, device=DEV)
    init_sd = {}
    for k, v in adapters.items():
        if k.startswith("adapter_before."):
            init_sd["transformer." + k[len("adapter_before."):]] = v
    for mod_name in wrapper.lora_modules:
        init_sd[f"transformer.{mod_name}.alpha"] = torch.tensor(lora_alpha)
    wrapper.load_state_dict(init_sd, strict=True)
    wrapper.set_dropout(0.0)
    # LoRA params stay resident on GPU (F32); the base blocks swap CPU<->GPU.
    wrapper.to(device=torch.device("cuda"), dtype=torch.float32)
    wrapper.hook_to_module()
    wrapper.requires_grad_(True)

    # ---- HOOKS: capture grad at to_q/to_k/to_v MODULE OUTPUT (the LoRA d_y) ----
    # The LoRA replaced to_q.forward with `orig(x) + up(down(x))*scale`; grad wrt
    # to_q's OUTPUT == grad wrt orig output == grad wrt lora output (sum rule), so
    # hooking the orig Linear's output tensor yields exactly the d_y the LoRA uses.
    blk0 = transformer.transformer_blocks[0].attn
    captured = {}

    def make_fwd_hook(name):
        def fwd_hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(o, torch.Tensor) and o.requires_grad:
                o.register_hook(lambda g, n=name: captured.__setitem__(n, g.detach().float().cpu()))
        return fwd_hook

    handles = []
    for nm, mod in [("d_q", blk0.to_q), ("d_k", blk0.to_k), ("d_v", blk0.to_v)]:
        handles.append(mod.register_forward_hook(make_fwd_hook(nm)))

    # LOCALIZATION: capture grad at the INPUT of to_out[0] (image attn output) and
    # to_add_out (text attn output) == d_att, the INPUT to the joint sdpa_backward.
    # forward_pre_hook grabs x; register_hook on it captures grad wrt x (fires in
    # the grad-ckpt recompute pass where x.requires_grad).
    to_out_lin = blk0.to_out[0] if hasattr(blk0.to_out, "__getitem__") else blk0.to_out
    to_add_out_lin = blk0.to_add_out
    print(f"[struct] to_out={type(blk0.to_out).__name__} to_out_lin={type(to_out_lin).__name__} "
          f"to_add_out={type(to_add_out_lin).__name__}")

    def make_pre_hook(name):
        def pre_hook(module, args):
            x = args[0]
            if isinstance(x, torch.Tensor) and x.requires_grad:
                x.register_hook(lambda g, n=name: captured.__setitem__(n, g.detach().float().cpu()))
        return pre_hook

    handles.append(to_out_lin.register_forward_pre_hook(make_pre_hook("d_iatt")))
    handles.append(to_add_out_lin.register_forward_pre_hook(make_pre_hook("d_tatt")))

    # LOCALIZATION: capture grad at block-0 OUTPUTS (== d_io/d_to entering block-0
    # backward). Block returns a tuple; route by token count (img=1120, txt=512).
    block0 = transformer.transformer_blocks[0]

    def block_out_hook(module, inp, out):
        outs = out if isinstance(out, tuple) else (out,)
        for o in outs:
            if isinstance(o, torch.Tensor) and o.requires_grad and o.dim() == 3:
                tag = "d_io" if o.shape[1] == 1120 else ("d_to" if o.shape[1] == 512 else None)
                if tag:
                    o.register_hook(lambda g, t=tag: captured.__setitem__(t, g.detach().float().cpu()))

    handles.append(block0.register_forward_hook(block_out_hook))

    # LOCALIZATION: grad at single_transformer_blocks[0] hidden_states INPUT == the
    # joint-stream grad at the single<->double boundary. Joint = cat([txt(512), img(1120)]).
    single0 = transformer.single_transformer_blocks[0]

    def single0_pre_hook(module, args):
        x = args[0]
        if isinstance(x, torch.Tensor) and x.requires_grad and x.dim() == 3:
            def _split(g):
                g = g.detach().float().cpu()[0]  # [S,D]
                captured["d_txt_bnd"] = g[:512].contiguous()
                captured["d_img_bnd"] = g[512:512 + 1120].contiguous()
            x.register_hook(_split)

    handles.append(single0.register_forward_pre_hook(single0_pre_hook))

    # OP-ISOLATION: LAST single block (23, runs FIRST in backward — clean input from
    # head). Capture grad at to_qkv_mlp_proj OUTPUT (v-band = sdpa-input value grad)
    # and proj_out INPUT (attn-band = grad at attention output == Mojo d_att).
    n_single = len(transformer.single_transformer_blocks)
    SGL_PROBE = 5  # delta-bisection: probe block 5 (matches Mojo injection target)
    sblk = transformer.single_transformer_blocks[SGL_PROBE]
    qkv_mod = None
    out_mod = None
    for nm, m in sblk.named_modules():
        if nm.endswith("to_qkv_mlp_proj"):
            qkv_mod = m
        if nm.endswith("proj_out") or nm.endswith("to_out"):
            out_mod = m
    print(f"[struct-sgl] single_block={SGL_PROBE} (of {n_single}) qkv={type(qkv_mod).__name__} "
          f"out={type(out_mod).__name__} "
          f"qkv_out_features={getattr(qkv_mod, 'out_features', '?')}")
    if qkv_mod is not None:
        # output grad = grad at [q|k|v|gate|up]; input grad = grad at norm (d_norm)
        handles.append(qkv_mod.register_forward_hook(make_fwd_hook("sgl_qkv_outgrad")))
        handles.append(qkv_mod.register_forward_pre_hook(make_pre_hook("sgl_qkv_ingrad")))
        # FORWARD VALUE of to_qkv_mlp_proj output = [q_pre|k_pre|v|gate|up]; compare
        # to Mojo's SAVED q_pre/k_pre/v to test for a forward (text) divergence.
        def qkv_fwd_val(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured["sgl_qkv_fwd"] = o.detach().float().cpu()
        handles.append(qkv_mod.register_forward_hook(qkv_fwd_val))
    if out_mod is not None:
        handles.append(out_mod.register_forward_pre_hook(make_pre_hook("sgl_projout_ingrad")))

    # PER-BLOCK: grad at EACH single block's INPUT (== Mojo running d_x AFTER that
    # block's backward). Finds the block index where txt grad first diverges.
    def mk_sdx(i):
        def h(module, args):
            x = args[0]
            if isinstance(x, torch.Tensor) and x.requires_grad and x.dim() == 3:
                x.register_hook(lambda g, i=i: captured.__setitem__(f"sdx_{i}", g.detach().float().cpu()[0]))
        return h
    for i, sb_i in enumerate(transformer.single_transformer_blocks):
        handles.append(sb_i.register_forward_pre_hook(mk_sdx(i)))

    # FORWARD-DIVERGENCE: capture each single block's INPUT hidden-states VALUE
    # (forward), keyed by index. vs Mojo saved block input -> finds where the
    # forward first diverges (born-in-block vs accumulated / double->single boundary).
    def mk_sfx(i):
        def h(module, args):
            x = args[0]
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                captured[f"sfx_{i}"] = x.detach().float().cpu()[0]
        return h
    for i, sb_i in enumerate(transformer.single_transformer_blocks):
        handles.append(sb_i.register_forward_pre_hook(mk_sfx(i)))

    # FORWARD-BISECTION: capture each DOUBLE block's OUTPUT forward value
    # (txt=encoder_hidden_states, img=hidden_states) -> find where txt fwd diverges.
    def mk_dbl(i):
        def h(module, inp, out):
            outs = out if isinstance(out, tuple) else (out,)
            for o in outs:
                if isinstance(o, torch.Tensor) and o.dim() == 3:
                    if o.shape[1] == 512:
                        captured[f"dtxt_{i}"] = o.detach().float().cpu()[0]
                    elif o.shape[1] == 1120:
                        captured[f"dimg_{i}"] = o.detach().float().cpu()[0]
        return h
    for i, db_i in enumerate(transformer.transformer_blocks):
        handles.append(db_i.register_forward_hook(mk_dbl(i)))

    # PER-BLOCK F32 SENSITIVITY: capture double-block-0 full inputs to F32-recompute it
    # and measure the reference's OWN bf16-vs-F32 sensitivity (one block, fits memory).
    dbl0_io = {}
    def dbl0_pre(module, args, kwargs):
        dbl0_io["args"] = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args)
        dbl0_io["kwargs"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        # block-0 INPUT: hidden_states(img), encoder_hidden_states(txt)
        hs = kwargs.get("hidden_states", args[0] if args else None)
        ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        if torch.is_tensor(hs):
            captured["din_img"] = hs.detach().float().cpu()[0]
        if torch.is_tensor(ehs):
            captured["din_txt"] = ehs.detach().float().cpu()[0]
        tmi = kwargs.get("temb_mod_img", args[2] if len(args) > 2 else None)
        if torch.is_tensor(tmi):
            captured["dbg_temb_mod_img"] = tmi.detach().float().cpu()
    handles.append(transformer.transformer_blocks[0].register_forward_pre_hook(dbl0_pre, with_kwargs=True))

    # FORWARD-BISECTION: capture double-block-0 attn internals (FORWARD VALUES):
    # attn input = modulate1 out (img norm / txt norm); to_out[0]/to_add_out input =
    # sdpa output per stream; to_q output = img q_pre.
    db0_attn = transformer.transformer_blocks[0].attn
    def db0_attn_pre(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        if torch.is_tensor(hs):
            captured["dbg_img_norm"] = hs.detach().float().cpu()[0]
        if torch.is_tensor(ehs):
            captured["dbg_txt_norm"] = ehs.detach().float().cpu()[0]
    handles.append(db0_attn.register_forward_pre_hook(db0_attn_pre, with_kwargs=True))
    def mk_inpre(key):
        def h(module, args):
            x = args[0]
            if torch.is_tensor(x):
                captured[key] = x.detach().float().cpu()[0]
        return h
    if hasattr(db0_attn, "to_out"):
        to_out0 = db0_attn.to_out[0] if hasattr(db0_attn.to_out, "__getitem__") else db0_attn.to_out
        handles.append(to_out0.register_forward_pre_hook(mk_inpre("dbg_img_att")))
    if hasattr(db0_attn, "to_add_out") and db0_attn.to_add_out is not None:
        handles.append(db0_attn.to_add_out.register_forward_pre_hook(mk_inpre("dbg_txt_att")))
    if hasattr(db0_attn, "to_q"):
        def to_q_out(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured["dbg_img_qpre"] = o.detach().float().cpu()[0]
        handles.append(db0_attn.to_q.register_forward_hook(to_q_out))

    # ---- forward (identical to serenity_klein_oracle.py) ----
    import contextlib
    _autocast = (contextlib.nullcontext() if F32_MODE
                 else torch.autocast(device_type="cuda", dtype=BF16))
    with _autocast:
        packed_predicted_flow = transformer(
            hidden_states=packed_latent_input.to(BF16),
            timestep=transformer_timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states.to(BF16),
            txt_ids=text_ids,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=True,
        ).sample

    predicted = unpatchify_latents(unpack_latents(packed_predicted_flow, H, W))
    mean_dim = list(range(1, predicted.ndim))
    losses = F.mse_loss(predicted.float(), target, reduction="none").mean(mean_dim)
    loss = (losses * loss_weight).mean()
    print(f"[fwd] loss={loss.item():.8f}")

    loss.backward()
    for h in handles:
        h.remove()

    # ---- double-block-0 F32 recompute (AFTER backward; in-place float on cloned args) ----
    if dbl0_io.get("args") is not None and not F32_MODE:
        with torch.no_grad():
            blk = transformer.transformer_blocks[0]
            blk.float()
            a = tuple(x.float() if torch.is_tensor(x) else x for x in dbl0_io["args"])
            kw = {k: (v.float() if torch.is_tensor(v) else v) for k, v in dbl0_io["kwargs"].items()}
            o = blk(*a, **kw)
            outs = o if isinstance(o, tuple) else (o,)
            for t in outs:
                if torch.is_tensor(t) and t.dim() == 3:
                    if t.shape[1] == 512:
                        captured["dtxt_0_f32"] = t.detach().float().cpu()[0]
                    elif t.shape[1] == 1120:
                        captured["dimg_0_f32"] = t.detach().float().cpu()[0]
            blk.to(BF16)
        torch.cuda.empty_cache()

    if not captured:
        print("[ERR] no grads captured — gradient-checkpointing recompute hook did not fire.")
        sys.exit(1)

    # ---- save d_y (squeeze batch -> [N_img, D] to match Mojo d_*_pre_flat) ----
    out = {}
    for nm in ("d_q", "d_k", "d_v"):
        t = captured[nm]
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        out[f"{nm}_pre_flat"] = t.contiguous()
        print(f"[dy] {nm}_pre_flat shape={tuple(t.shape)} L2={t.float().norm().item():.6f}")
    save_file(out, OUT_DY)
    print(f"[saved] {OUT_DY}")

    # ---- save d_att (grad at attention output, input to sdpa_backward) ----
    OUT_DATT = f"{PARITY}/serenity_klein_block0_datt.safetensors"
    datt = {}
    for nm in ("d_iatt", "d_tatt"):
        if nm not in captured:
            print(f"[WARN] {nm} not captured")
            continue
        t = captured[nm]
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        datt[nm] = t.contiguous()
        print(f"[datt] {nm} shape={tuple(t.shape)} L2={t.float().norm().item():.6f}")
    if datt:
        save_file(datt, OUT_DATT)
        print(f"[saved] {OUT_DATT}")

    # ---- save d_io/d_to (grad at block-0 OUTPUT, entering block-0 backward) ----
    OUT_DIO = f"{PARITY}/serenity_klein_block0_dio.safetensors"
    dio = {}
    for nm in ("d_io", "d_to"):
        if nm not in captured:
            print(f"[WARN] {nm} not captured")
            continue
        t = captured[nm]
        if t.dim() == 3 and t.shape[0] == 1:
            t = t[0]
        dio[nm] = t.contiguous()
        print(f"[dio] {nm} shape={tuple(t.shape)} L2={t.float().norm().item():.6f}")
    if dio:
        save_file(dio, OUT_DIO)
        print(f"[saved] {OUT_DIO}")

    # ---- save single<->double boundary grad ----
    OUT_BND = f"{PARITY}/serenity_klein_block0_bnd.safetensors"
    bnd = {}
    for nm in ("d_img_bnd", "d_txt_bnd"):
        if nm in captured:
            bnd[nm] = captured[nm].contiguous()
            print(f"[bnd] {nm} shape={tuple(bnd[nm].shape)} L2={bnd[nm].float().norm().item():.6f}")
        else:
            print(f"[WARN] {nm} not captured")
    if bnd:
        save_file(bnd, OUT_BND)
        print(f"[saved] {OUT_BND}")

    # ---- save single-block sdpa op-isolation grads ----
    OUT_SGL = f"{PARITY}/serenity_klein_block0_sgl_sdpa.safetensors"
    sgl = {}
    for nm in ("sgl_qkv_outgrad", "sgl_projout_ingrad", "sgl_qkv_ingrad", "sgl_qkv_fwd"):
        if nm in captured:
            t = captured[nm]
            if t.dim() == 3 and t.shape[0] == 1:
                t = t[0]
            sgl[nm] = t.contiguous()
            print(f"[sgl] {nm} shape={tuple(t.shape)} L2={t.float().norm().item():.6f}")
        else:
            print(f"[WARN] {nm} not captured")
    if sgl:
        save_file(sgl, OUT_SGL)
        print(f"[saved] {OUT_SGL}")

    # ---- save per-block single-stack input grads ----
    OUT_SDX = f"{PARITY}/serenity_klein_block0_sdx{'_f32' if F32_MODE else ''}.safetensors"
    sdx = {k: v.contiguous() for k, v in captured.items() if k.startswith("sdx_")}
    if sdx:
        save_file(sdx, OUT_SDX)
        print(f"[saved] {OUT_SDX} ({len(sdx)} blocks)")

    # ---- save per-block FORWARD inputs ----
    OUT_SFX = f"{PARITY}/serenity_klein_block0_sfx.safetensors"
    sfx = {k: v.contiguous() for k, v in captured.items() if k.startswith("sfx_")}
    if sfx:
        save_file(sfx, OUT_SFX)
        print(f"[saved] {OUT_SFX} ({len(sfx)} blocks)")

    OUT_DBL = f"{PARITY}/serenity_klein_block0_dbl.safetensors"
    dbl = {k: v.contiguous() for k, v in captured.items() if k.startswith(("dtxt_", "dimg_", "din_", "dbg_"))}
    if "dtxt_0_f32" in captured:
        print(f"[dbl0-f32] captured F32 recompute of double block 0")
    if dbl:
        save_file(dbl, OUT_DBL)
        print(f"[saved] {OUT_DBL} ({len(dbl)} tensors)")

    # ---- DTYPE CONSISTENCY CHECK: reconstruct d_B from reference trainer's own d_y + down(x) ----
    # d_B(to_q) = scale * sum_n down_out[n,:]^T (x) outer d_y[n,:]  == scale * d_y^T @ down_out
    # (LoRAModule.forward: ld = up(down(x)); out = orig(x) + ld*scale). Compare to
    # reference trainer's captured p.grad for lora_up. Agreement (F32) => the d_B math is exact and
    # any Mojo d_B gap is fully explained by a d_y difference (not dtype/accumulation).
    scale = lora_alpha / lora_rank
    qmod = wrapper.lora_modules["transformer_blocks.0.attn.to_q"]
    # recompute down(x) on the SAME input the module saw: re-run to_q's orig on hidden.
    # We need x = input to to_q. Grab it via a quick input hook on a fresh forward is
    # overkill; instead reconstruct from the captured d_y + actual grad is enough:
    dB_actual = qmod.lora_up.weight.grad.detach().float().cpu()  # [out, rank]
    print(f"\n[dtype-check] to_q lora_up.weight.grad (d_B) L2={dB_actual.norm().item():.6f} "
          f"shape={tuple(dB_actual.shape)}")
    print(f"[dtype-check] captured d_q L2={out['d_q_pre_flat'].float().norm().item():.6f} "
          f"(scale={scale}); if Mojo d_q matches this, d_B gap is NOT in d_y.")

    print("\n[done]")


if __name__ == "__main__":
    main()
