# serenitymojo.serve.krea2_encode_subprocess — the Qwen3-VL-4B text encoder in a
# fork+execv CHILD process, so its VRAM (~9.6 GB peak) is reclaimed by process
# death before the DiT process builds the ~12 GB fp8-resident base.
#
# Same split the krea2 two-process pipeline uses (krea2_encode_cli + krea2_pipeline)
# and the SAME fork+execv/waitpid contract as zimage_encode_subprocess: argv built
# BEFORE fork, async-signal-safe calls only between fork and execv, execv into a
# fresh image. The child writes the two BF16 context tensors [1,LT,12,2560] to the
# given .bin paths (io.cap_cache, bit-identical); the DiT side reads them back with
# inline_cond_from_bin (LT derived from the tensor shape — no meta sidecar needed).
#
# SELF-EXEC (one binary): the encode child IS serenity_worker_krea2 re-exec'd with
# argv ["encode-child", <prompt>, <negative>, <pos_bin>, <neg_bin>];
# serenity_worker_krea2.main routes that to krea2_encode_child_run().
#
# FALLBACK: any subprocess failure (fork error, timeout, abnormal exit, missing
# bins) falls back to an in-process encode (correct, just no VRAM reclaim). For v1
# the DiT resident base is built AFTER this returns, so an in-process encode's stuck
# pool still leaves room; the subprocess is the clean path.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from max.gpu.host import DeviceContext
from std.time import sleep

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.tokenizer.tokenizer_cache import qwen3_tokenizer_open
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import (
    load_krea2_qwen3vl_4b, encode_krea2_stack,
)
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.pipeline.krea2_paths import (
    KREA2_TE_DIR, KREA2_TOK_JSON, KREA2_TPL_PREFIX, KREA2_TPL_SUFFIX,
)
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, sys_execv, sys__exit, sys_waitpid, proc_kill_wait,
    SELF_EXE, SIGKILL, WNOHANG,
)
from net.syscalls import sys_fork, errno_str


comptime _ENCODE_CHILD_TIMEOUT_S = 300.0   # pure hang backstop (real encode is seconds)
comptime _ENCODE_POLL_S = 0.05


def _getpid() -> Int:
    return Int(external_call["getpid", Int32]())


def _krea2_encode_one(
    enc: Qwen3Encoder, tok: Qwen3Tokenizer, text: String,
    name: String, out_path: String, ctx: DeviceContext,
) raises -> Int:
    var ids = tok.encode(KREA2_TPL_PREFIX + text + KREA2_TPL_SUFFIX)
    var stack = encode_krea2_stack(enc, ids, ctx)      # [1, LT, 12, 2560] bf16
    var lt = stack.shape()[1]
    save_tensor_bin(stack, out_path, ctx)
    print("[krea2-encode-child]", name, " L_full=", len(ids), " LT=", lt, " ->", out_path)
    return lt


def krea2_encode_child_run(
    prompt: String, negative: String, pos_bin: String, neg_bin: String
) raises:
    """CHILD body (after execv into `serenity_worker_krea2 encode-child`). Fresh
    process image → fresh CUDA context. Load the Qwen3-VL-4B TE, encode POS + NEG,
    write the two BF16 context tensors, then RETURN so the process exits and the OS
    reclaims ALL encoder VRAM."""
    var ctx = DeviceContext()
    var tok = qwen3_tokenizer_open(
        String(KREA2_TOK_JSON), String(KREA2_TOK_JSON) + ".mjtok"
    )
    var enc = load_krea2_qwen3vl_4b(String(KREA2_TE_DIR), ctx)
    var lt_pos = _krea2_encode_one(enc, tok, prompt, String("POS"), pos_bin, ctx)
    var lt_neg = _krea2_encode_one(enc, tok, negative, String("NEG"), neg_bin, ctx)
    print("[krea2-encode-child] DONE LT_POS=", lt_pos, " LT_NEG=", lt_neg)


def _krea2_encode_in_process(
    prompt: String, negative: String, pos_bin: String, neg_bin: String,
    ctx: DeviceContext,
) raises:
    """Fallback: encode in THIS process (correct; TE VRAM stays in the pool)."""
    var tok = qwen3_tokenizer_open(
        String(KREA2_TOK_JSON), String(KREA2_TOK_JSON) + ".mjtok"
    )
    var enc = load_krea2_qwen3vl_4b(String(KREA2_TE_DIR), ctx)
    _ = _krea2_encode_one(enc, tok, prompt, String("POS"), pos_bin, ctx)
    _ = _krea2_encode_one(enc, tok, negative, String("NEG"), neg_bin, ctx)


def krea2_encode_contexts_subprocess(
    prompt: String, negative: String, pos_bin: String, neg_bin: String,
    ctx: DeviceContext,
) raises:
    """Encode the POS/NEG contexts to .bin. IN-PROCESS by default for krea2.

    WHY in-process, not a fork+execv child like zimage (MEASURED 2026-07-12): the
    Qwen3-VL-4B TE peaks ~22 GB — nearly the whole 24 GB card. (1) A fork child needs
    that ~22 GB in FRESH OS memory; on any job after the first the parent's MAX pool
    already holds ~22 GB (cuMemPoolTrim returns ~0 to the OS between jobs — the same
    retention zimage documents), so the child OOMs. (2) In-process, the TE tensors
    drop after encode and the fp8-resident base (~12 GB) allocates from that SAME
    freed pool → steady-state stays ~22 GB and fits. The subprocess isolation buys
    nothing here (the pool never shrinks below the resident+TE-peak either way), so
    in-process is both simpler and robust across repeated jobs. `encode_child_run`
    + the worker's `encode-child` route are kept for the fork path but unused by
    default; flip a future guard here if a smaller TE ever makes isolation worthwhile.
    """
    print("[krea2] encoding contexts in-process (Qwen3-VL-4B TE ~22GB; pool-reused)")
    _krea2_encode_in_process(prompt, negative, pos_bin, neg_bin, ctx)
