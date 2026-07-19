#!/usr/bin/env python3
"""Folder captioner for the serenity web trainer.

Runs a Qwen3-VL vision-language model over every image in a folder and writes a
`<stem>.txt` caption sidecar next to each image. Emits one machine-readable
`CAPJSON {...}` line per event on stdout so the Rust supervisor can pump progress
into the UI exactly like it pumps training runs. Model-loading and generation
follow ai-toolkit's Qwen3VLCaptioner (extensions_built_in/captioner) so the
numerics match the captioner the user already trusts.

Invoked as:
  python -u captioner.py --folder DIR --model HF_ID --max-tokens N \
      [--prompt "..."] [--skip-existing] [--one-sentence]
"""
import argparse
import glob
import json
import os
import sys
import traceback


def emit(obj):
    """One event -> one CAPJSON line the supervisor parses."""
    sys.stdout.write("CAPJSON " + json.dumps(obj) + "\n")
    sys.stdout.flush()


def note(msg):
    """Human line to stderr (model download/load chatter, kept out of the event stream)."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def patch_qwen_vl_patch_embed(model):
    """Qwen-VL's vision patch_embed is a Conv3d whose kernel == stride, i.e. a plain
    linear projection of each flattened patch. bf16 Conv3d has no fast cuDNN kernel and
    falls back to a slow path. Swap it for the equivalent F.linear (a GEMM). Verbatim
    from ai-toolkit's Qwen3VLCaptioner."""
    import torch
    import torch.nn.functional as F
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(proj.stride):
            def fast_forward(hidden_states, _proj=proj):
                w = _proj.weight.reshape(_proj.weight.shape[0], -1)
                x = hidden_states.view(-1, w.shape[1]).to(w.dtype)
                return F.linear(x, w, _proj.bias)
            module.forward = fast_forward
            patched += 1
    return patched


def downscale(img, maxres):
    w, h = img.size
    if max(w, h) <= maxres:
        return img
    s = maxres / float(max(w, h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


DEFAULT_PROMPT = (
    "Write a detailed caption for this image to train an image-generation model. "
    "Describe the main subject, appearance, pose, clothing, expression, any action, "
    "the setting and background, lighting, colors, composition, and the art medium or "
    "style. Output only the caption itself with no preamble."
)
ONE_SENTENCE_PROMPT = (
    "Describe this image in one detailed sentence for an image-generation training "
    "caption. Output only the sentence, no preamble."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-res", type=int, default=1024)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--one-sentence", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        emit({"type": "fatal", "error": f"folder not found: {args.folder}"})
        sys.exit(2)

    exts = ["jpg", "jpeg", "png", "webp"]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.folder, "*." + e))
        files += glob.glob(os.path.join(args.folder, "*." + e.upper()))
    files = sorted(set(files))

    todo = []
    for f in files:
        txt = os.path.splitext(f)[0] + ".txt"
        if args.skip_existing and os.path.exists(txt) and os.path.getsize(txt) > 0:
            continue
        todo.append(f)

    emit({"type": "start", "total": len(todo), "found": len(files),
          "folder": args.folder, "model": args.model})
    if not todo:
        emit({"type": "done", "done": 0, "total": 0})
        return

    # resolve the prompt
    if args.prompt and args.prompt.strip():
        prompt = args.prompt.strip()
        if args.one_sentence:
            prompt += " Respond in a single sentence."
    else:
        prompt = ONE_SENTENCE_PROMPT if args.one_sentence else DEFAULT_PROMPT

    note(f"loading {args.model} ...")
    import torch
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        AutoProcessor,
        AutoConfig,
    )
    from PIL import Image

    # dense vs MoE by config architecture (robust; name heuristics misfire on
    # e.g. "8B-Abliterated" which contains "B-A")
    is_moe = False
    try:
        cfg = AutoConfig.from_pretrained(args.model)
        arch = " ".join(getattr(cfg, "architectures", []) or [])
        is_moe = "Moe" in arch
    except Exception:
        is_moe = "A3B" in args.model
    ModelClass = Qwen3VLMoeForConditionalGeneration if is_moe else Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelClass.from_pretrained(args.model, dtype=dtype, device_map="cpu")
    patch_qwen_vl_patch_embed(model)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    note(f"model ready on {device}")
    emit({"type": "loaded", "model": args.model, "device": device})

    done = 0
    for f in todo:
        base = os.path.basename(f)
        emit({"type": "file_start", "file": base, "done": done, "total": len(todo)})
        try:
            img = downscale(Image.open(f).convert("RGB"), args.max_res)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
            text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()

            txt = os.path.splitext(f)[0] + ".txt"
            with open(txt, "w") as fh:
                fh.write(text)
            done += 1
            emit({"type": "progress", "done": done, "total": len(todo),
                  "file": base, "caption": text, "sidecar": txt})
        except Exception as e:
            emit({"type": "error", "file": base, "error": str(e)})
            note(traceback.format_exc())

    emit({"type": "done", "done": done, "total": len(todo)})


if __name__ == "__main__":
    main()
