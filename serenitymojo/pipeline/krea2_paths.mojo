# pipeline/krea2_paths.mojo — shared Krea-2 pipeline paths + conditioning template.
#
# Single source of truth for the checkpoint/encoder/VAE/tokenizer paths, the
# Qwen3-VL conditioning template (text_encoder.py:26-31), the default prompts, and
# the context-cache file paths shared between krea2_encode_cli (the TE child) and
# krea2_pipeline (the DiT/VAE main). Keeps the two in lock-step.

# ── checkpoint / encoder / VAE / tokenizer ───────────────────────────────────
comptime KREA2_RAW = (
    "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/"
    "snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
)
comptime KREA2_TE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/"
    "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
comptime KREA2_TOK_JSON = KREA2_TE_DIR + "/tokenizer.json"
# Qwen-Image VAE (f8, 16ch) diffusers `vae` subfolder — same VAE krea2.py loads via
# AutoencoderKLQwenImage.from_pretrained(QWEN_IMAGE_VAE_PATH, subfolder="vae").
comptime KREA2_VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image/"
    "snapshots/75e0b4be04f60ec59a75f475837eced720f823b6/vae"
)
# The real checkpoint stores BARE torch keys (no "w." prefix); the parity oracle
# uses the default "w." prefix. krea2_forward(key_prefix=...) selects.
comptime KREA2_RAW_KEY_PREFIX = String("")

# ── conditioning template (text_encoder.py:26-31) ────────────────────────────
comptime KREA2_TPL_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
comptime KREA2_TPL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

# ── default prompts (256² validation config) ─────────────────────────────────
comptime KREA2_DEFAULT_PROMPT = (
    "A photorealistic portrait of an astronaut riding a horse on Mars."
)
comptime KREA2_DEFAULT_NEGATIVE = ""

# ── context-cache files (TE child writes; DiT main reads). ───────────────────
comptime KREA2_CTX_POS_BIN = "/home/alex/mojodiffusion/output/krea2_ctx_pos.bin"
comptime KREA2_CTX_NEG_BIN = "/home/alex/mojodiffusion/output/krea2_ctx_neg.bin"
