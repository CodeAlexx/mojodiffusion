# pipeline/krea2_encode_cli.mojo — Krea-2 conditioning ENCODE child process.
#
# WHY A SEPARATE PROCESS (re-measured 2026-06-24, EXTERNAL nvidia-smi):
#   - On a FREE 24 GB card the TE load+encode shows a ~22 GB nvidia-smi peak, but
#     this is MAX's allocator OPPORTUNISTICALLY reserving free headroom, NOT a real
#     requirement: the SAME load under memory pressure (14 GB held elsewhere, ~9.4
#     GB free) completes fine and the TE's GENUINE working set is only ~8.3 GB
#     (it OOMs only when <7.5 GB is free). So the 22 GB is a high-water mark MAX
#     backs off, not 22 GB of live data.
#   - load_krea2_qwen3vl_4b now STREAMS every tensor through one reusable pinned
#     host staging buffer (~778 MB), so the per-tensor pinned-staging accumulation
#     the original from_view path caused is gone; the ~8.3 GB is the irreducible
#     bf16 4B device weights + compute.
# Process separation is still the right design: MAX's opportunistic ~22 GB reserve
# collides with a co-resident DiT, and process death is the only thing that returns
# MAX's reserved pool to the OS cleanly. This short-lived child loads the TE, encodes
# BOTH the positive and negative contexts, dumps them via cap_cache.save_tensor_bin,
# and EXITS — freeing everything. The main krea2_pipeline then load_tensor_bin's the
# two tiny contexts and runs the DiT with zero encoder code/weights resident.
#
# Output contexts are [1, LT, 12, 2560] bf16 (LT = prompt natural length - 34); the
# pipeline's comptime LT_POS/LT_NEG must match (it shape-checks the loaded cache).
#
# Run (the pipeline shells out to this; or stand-alone):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/pipeline/krea2_encode_cli.mojo \
#       "<prompt>" "<negative>" <pos_out.bin> <neg_out.bin>
# With no argv it uses the built-in default astronaut prompt + empty negative and
# the default cache paths (matching krea2_pipeline's defaults).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from std.gpu.host import DeviceContext
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import (
    load_krea2_qwen3vl_4b,
    encode_krea2_stack,
)
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.pipeline.krea2_paths import (
    KREA2_TE_DIR,
    KREA2_TOK_JSON,
    KREA2_TPL_PREFIX,
    KREA2_TPL_SUFFIX,
    KREA2_DEFAULT_PROMPT,
    KREA2_DEFAULT_NEGATIVE,
    KREA2_CTX_POS_BIN,
    KREA2_CTX_NEG_BIN,
)


def _encode_one(
    enc: Qwen3Encoder,
    tok: Qwen3Tokenizer,
    prompt: String,
    name: String,
    out_path: String,
    ctx: DeviceContext,
) raises -> Int:
    """Tokenize PREFIX+prompt+SUFFIX, encode the 12-layer krea2 stack, dump it.
    Returns LT (= L_full - 34) so the caller can report it for the pipeline pins."""
    var ids = tok.encode(KREA2_TPL_PREFIX + prompt + KREA2_TPL_SUFFIX)
    var stack = encode_krea2_stack(enc, ids, ctx)   # [1, LT, 12, 2560] bf16
    var sh = stack.shape()
    var lt = sh[1]
    print("[krea2-encode] ", name, " L_full=", len(ids), " LT=", lt,
          " stack=[", sh[0], sh[1], sh[2], sh[3], "] -> ", out_path)
    save_tensor_bin(stack, out_path, ctx)
    return lt


def main() raises:
    var ctx = DeviceContext()
    var args = argv()
    # argv: [bin, prompt, negative, pos_out, neg_out] — all optional (defaults).
    var prompt = String(KREA2_DEFAULT_PROMPT)
    var negative = String(KREA2_DEFAULT_NEGATIVE)
    var pos_out = String(KREA2_CTX_POS_BIN)
    var neg_out = String(KREA2_CTX_NEG_BIN)
    if len(args) >= 2:
        prompt = String(args[1])
    if len(args) >= 3:
        negative = String(args[2])
    if len(args) >= 4:
        pos_out = String(args[3])
    if len(args) >= 5:
        neg_out = String(args[4])

    print("[krea2-encode] prompt:", prompt)
    var tok = Qwen3Tokenizer(String(KREA2_TOK_JSON))

    # Load TE once, encode both prompts, then exit so the OS releases the MAX pool.
    var enc = load_krea2_qwen3vl_4b(String(KREA2_TE_DIR), ctx)
    var lt_pos = _encode_one(enc, tok, prompt, String("POS"), pos_out, ctx)
    var lt_neg = _encode_one(enc, tok, negative, String("NEG"), neg_out, ctx)

    print("[krea2-encode] DONE. LT_POS=", lt_pos, " LT_NEG=", lt_neg,
          " (pipeline comptime LT_POS/LT_NEG must equal these).")
