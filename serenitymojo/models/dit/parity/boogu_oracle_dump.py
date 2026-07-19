#!/usr/bin/env python
# boogu_oracle_dump.py — Boogu-Image-0.1-Base parity ORACLE (dev tool, NOT shipped).
#
# Captures byte-exact reference tensors for every pure-Mojo port chunk by
# registering forward hooks on the REAL bf16 checkpoint and running one small
# T2I call. Anchored to the canonical entrypoint (inference_simple.py):
#   os.environ["device"]="cuda:0"; BooguImagePipeline.from_pretrained(..., bf16, trust_remote_code)
#
# Why hooks (not arg reconstruction): the mojo-port skill's technique — dump the
# EXACT inputs/outputs the pipeline feeds each submodule via
# register_forward_pre_hook(with_kwargs=True), so we never guess freqs_cis /
# masks / seq layout.
#
# 24GB GPU note: the full pipeline is ~38.5GB; we enable_model_cpu_offload() so
# only one module is resident at a time. flash_attn must be ABSENT in this venv
# so the Boogu blocks use the torch-SDPA processor (matches Mojo math-mode SDPA);
# we assert that below.
#
# Run (oracle is a SEPARATE command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/dit/parity/boogu_oracle_dump.py \
#       --res 256 --steps 4 --no-cfg
#
# Dumps -> serenitymojo/models/dit/parity/boogu_dumps/<name>.npy + manifest.json
import argparse
import json
import os

os.environ.setdefault("device", "cuda:0")  # REQUIRED before importing boogu

import numpy as np
import torch

CKPT = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boogu_dumps")

# Fixed English instruction (text-only T2I -> Qwen3-VL vision tower unused).
INSTRUCTION = (
    "A street photograph of an elderly scavenger with a deeply weathered, "
    "wrinkled face in the center of the frame, a trash can and a traffic light "
    "in the background, Leica camera street aesthetic, cinematic lighting, "
    "photorealistic."
)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return None


class Capture:
    """Saves the FIRST occurrence of each named tensor (one clean forward)."""

    def __init__(self):
        self.saved = {}
        self.manifest = {}
        self.per_step = []  # (timestep, latent-token-stats) across all denoise calls

    def save(self, name, t):
        if name in self.saved or t is None:
            return
        arr = _to_np(t)
        if arr is None:
            return
        self.saved[name] = arr
        self.manifest[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "absmax": float(np.abs(arr).max()),
        }

    def flush(self):
        os.makedirs(OUT, exist_ok=True)
        for name, arr in self.saved.items():
            np.save(os.path.join(OUT, name + ".npy"), arr)
        with open(os.path.join(OUT, "manifest.json"), "w") as f:
            json.dump(
                {"tensors": self.manifest, "per_step": self.per_step,
                 "instruction": INSTRUCTION},
                f, indent=2,
            )
        print(f"[oracle] wrote {len(self.saved)} tensors -> {OUT}")
        for n, m in self.manifest.items():
            print(f"  {n:38s} {m['shape']} std={m['std']:.5f} absmax={m['absmax']:.4f}")


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) and x else x


def register(pipe, cap):
    tf = pipe.transformer

    # --- mllm: last hidden state = instruction_feats [B,L,4096] ---
    def mllm_hook(mod, inp, out):
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            cap.save("mllm_last_hidden", hs[-1])
    if getattr(pipe, "mllm", None) is not None:
        pipe.mllm.register_forward_hook(mllm_hook)

    # --- transformer top-level: capture FULL input kwargs on first call + output ---
    def tf_pre(mod, args, kwargs):
        # per-step log (fires every denoise call incl. CFG)
        ts = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        hs = kwargs.get("hidden_states", args[0] if args else None)
        h0 = _first(hs)
        cap.per_step.append({
            "timestep": _to_np(ts).tolist() if isinstance(ts, torch.Tensor) else ts,
            "latent_std": float(_to_np(h0).std()) if isinstance(h0, torch.Tensor) else None,
        })
        # first-call unit dumps
        cap.save("dit_in_latent", h0)
        cap.save("dit_in_timestep", ts if isinstance(ts, torch.Tensor) else torch.tensor(ts))
        cap.save("dit_in_instruction_feats", kwargs.get("instruction_hidden_states"))
        cap.save("dit_in_freqs_cis", kwargs.get("freqs_cis"))
        cap.save("dit_in_attn_mask", kwargs.get("instruction_attention_mask"))
    tf.register_forward_pre_hook(tf_pre, with_kwargs=True)

    def tf_hook(mod, inp, out):
        cap.save("dit_out_velocity", _first(out if not hasattr(out, "sample") else out.sample))
    tf.register_forward_hook(tf_hook)

    # --- embedders ---
    if hasattr(tf, "time_caption_embed"):
        def tce_hook(mod, inp, out):
            cap.save("tce_temb", out[0]); cap.save("tce_caption", out[1])
        tf.time_caption_embed.register_forward_hook(tce_hook)
    if hasattr(tf, "x_embedder"):
        tf.x_embedder.register_forward_hook(
            lambda m, i, o: (cap.save("xembed_in", _first(i)), cap.save("xembed_out", o)))
    if hasattr(tf, "rope_embedder"):
        def rope_hook(mod, inp, out):
            names = ["cap_freqs", "ref_img_freqs", "img_freqs", "joint_freqs",
                     "rope_capseqlen", "rope_seqlen", "combined_img_freqs", "combined_img_seqlen"]
            if isinstance(out, (tuple, list)):
                for n, t in zip(names, out):
                    if isinstance(t, torch.Tensor):
                        cap.save("rope_" + n, t)
        tf.rope_embedder.register_forward_hook(rope_hook)

    # --- one double-stream + one single-stream block ---
    if getattr(tf, "double_stream_layers", None) is not None and len(tf.double_stream_layers):
        blk = tf.double_stream_layers[0]
        def ds_pre(mod, args, kwargs):
            a = list(args)
            cap.save("ds0_in_img", a[0] if len(a) > 0 else kwargs.get("img_hidden_states"))
            cap.save("ds0_in_instruct", a[1] if len(a) > 1 else kwargs.get("instruct_hidden_states"))
        blk.register_forward_pre_hook(ds_pre, with_kwargs=True)
        blk.register_forward_hook(
            lambda m, i, o: (cap.save("ds0_out_img", o[0]), cap.save("ds0_out_instruct", o[1])))
    if getattr(tf, "single_stream_layers", None) is not None and len(tf.single_stream_layers):
        blk = tf.single_stream_layers[0]
        blk.register_forward_pre_hook(
            lambda m, a, k: cap.save("ss0_in", a[0] if a else k.get("hidden_states")), with_kwargs=True)
        blk.register_forward_hook(lambda m, i, o: cap.save("ss0_out", _first(o)))

    # --- output norm ---
    if hasattr(tf, "norm_out"):
        tf.norm_out.register_forward_pre_hook(
            lambda m, a, k: cap.save("normout_in", a[0] if a else k.get("x")), with_kwargs=True)
        tf.norm_out.register_forward_hook(lambda m, i, o: cap.save("normout_out", _first(o)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--no-cfg", action="store_true",
                    help="disable CFG -> single batch-1 forward (clean unit dumps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # match Mojo math-mode SDPA: flash_attn must be absent
    try:
        import flash_attn  # noqa
        print("[oracle] WARNING: flash_attn present — blocks may use flash, NOT SDPA. "
              "Uninstall it from this venv for math-mode parity.")
    except ImportError:
        print("[oracle] flash_attn absent -> Boogu blocks use torch-SDPA (matches Mojo).")

    from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
    print(f"[oracle] loading {CKPT} (bf16)…")
    pipe = BooguImagePipeline.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # offload so a single module is resident (24GB GPU vs 38.5GB ckpt)
    try:
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"[oracle] enable_model_cpu_offload failed ({e}); trying sequential")
        pipe.enable_sequential_cpu_offload()

    cap = Capture()
    register(pipe, cap)

    res = args.res
    print(f"[oracle] generating {res}x{res}, steps={args.steps}, cfg={'off' if args.no_cfg else 'on'}")
    with torch.no_grad():
        out = pipe(
            instruction=INSTRUCTION,
            negative_instruction="",
            height=res, width=res,
            max_input_image_pixels=res * res,
            max_input_image_side_length=2 * res,
            num_inference_steps=args.steps,
            text_guidance_scale=1.0 if args.no_cfg else 4.0,
            do_classifier_free_guidance=not args.no_cfg,
            generator=torch.Generator(os.environ["device"]).manual_seed(args.seed),
        )
    img = out.images[0]
    os.makedirs(OUT, exist_ok=True)
    img.save(os.path.join(OUT, f"oracle_t2i_{res}.png"))
    cap.flush()
    torch.cuda.empty_cache()
    print("[oracle] done.")


if __name__ == "__main__":
    main()
