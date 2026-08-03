# SKEPTIC PROBE (not part of the port) — exercises claims made by
# minimax_h3_loader_device.mojo that its own selfcheck/main do not cover:
#   * partially-written / truncated shard files (not just a missing directory)
#   * the qkv de-interleave + fc1 swap at REALISTIC heads/head_dim proportions
#     (56/128), not the tiny (2,3) fixture the shipped selfcheck uses
#
# Written by the skeptic pass, run manually, not wired into any gate.

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_deinterleave_qkv_bf16,
    minimax_h3_swap_fc1_bf16,
)
from serenitymojo.models.minimax_h3.loader import (
    minimax_h3_deinterleave_qkv,
    minimax_h3_swap_fc1,
)


def _try_open(label: String, dir: String):
    print("--", label, "--")
    try:
        var st = ShardedSafeTensors.open(dir)
        print("  OPENED (unexpected for a broken fixture):", st.num_shards(), "shard(s),", st.num_tensors(), "tensors")
        # try to actually touch the bytes too -- open() succeeding doesn't
        # mean tensor_bytes()/tensor_view() would.
        try:
            var names = st.names()
            for i in range(len(names)):
                var b = st.tensor_bytes(names[i])
                print("    read", names[i], "ok,", len(b), "bytes")
        except e2:
            print("    tensor_bytes/tensor_view raised (still no hang):", e2)
    except e:
        print("  raised (clean, no hang):", e)


def main() raises:
    comptime SB = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/h3_partial_fixture/"

    print("[A] control: well-formed tiny shard")
    _try_open("good", SB + "good")

    print("")
    print("[B] shard data segment truncated by 10 bytes (mid-download, direct-to-final-name write)")
    _try_open("trunc_data", SB + "trunc_data")

    print("")
    print("[C] shard truncated to 4 bytes (can't even read the 8-byte length prefix)")
    _try_open("trunc_tiny", SB + "trunc_tiny")

    print("")
    print("[D] shard truncated to 20 bytes (8-byte prefix present, header JSON incomplete)")
    _try_open("trunc_header", SB + "trunc_header")

    print("")
    print("[E] shard file exists but is zero bytes (downloader pre-created the destination)")
    _try_open("zero_byte", SB + "zero_byte")

    print("")
    print("[F] index.json present, shard file MISSING entirely")
    _try_open("no_shard", SB + "no_shard")

    print("")
    print("[G] directory exists but is completely empty (nothing downloaded yet)")
    _try_open("empty_dir", SB + "empty_dir")

    # ── realistic-proportion qkv de-interleave / fc1 swap ──────────────────
    print("")
    print("[H] qkv de-interleave at REAL heads=56, head_dim=128 (not the shipped 2/3 fixture)")
    var heads = 56
    var head_dim = 128
    var hidden = 3  # kept tiny only to bound element count; heads/head_dim are real
    var qkv_rows = heads * 3 * head_dim
    var fused_f32 = List[Float32]()
    var fused_bf16 = List[BFloat16]()
    for r in range(qkv_rows):
        for c in range(hidden):
            var v = Float32(r * hidden + c) * Float32(0.001) - Float32(5.0)
            fused_f32.append(v)
            fused_bf16.append(v.cast[DType.bfloat16]())
    var want = minimax_h3_deinterleave_qkv(fused_f32, heads, head_dim, hidden)
    var got = minimax_h3_deinterleave_qkv_bf16(fused_bf16, heads, head_dim, hidden)
    var qkv_bad = False
    if len(want) != len(got):
        print("  FAIL length mismatch", len(want), len(got))
        qkv_bad = True
    else:
        for i in range(len(want)):
            if got[i] != want[i].cast[DType.bfloat16]():
                print("  FAIL diverges at", i)
                qkv_bad = True
                break
    if not qkv_bad:
        print("  ok   qkv de-interleave matches at realistic heads/head_dim (", len(want), "values)")

    print("")
    print("[I] fc1 swap at REAL ffn=14336 scaled hidden=3 (heads/ffn ratio irrelevant to fc1, but use a large ffn)")
    var ffn = 512  # real is 14336; kept smaller only to bound runtime, still >> tiny fixture's 5
    var fc1_f32 = List[Float32]()
    var fc1_bf16 = List[BFloat16]()
    for r in range(2 * ffn):
        for c in range(hidden):
            var v = Float32(r * hidden + c) * Float32(0.002) + Float32(2.0)
            fc1_f32.append(v)
            fc1_bf16.append(v.cast[DType.bfloat16]())
    var want_fc1 = minimax_h3_swap_fc1(fc1_f32, ffn, hidden)
    var got_fc1 = minimax_h3_swap_fc1_bf16(fc1_bf16, ffn, hidden)
    var fc1_bad = False
    if len(want_fc1) != len(got_fc1):
        print("  FAIL length mismatch")
        fc1_bad = True
    else:
        for i in range(len(want_fc1)):
            if got_fc1[i] != want_fc1[i].cast[DType.bfloat16]():
                print("  FAIL diverges at", i)
                fc1_bad = True
                break
    if not fc1_bad:
        print("  ok   fc1 swap matches at realistic ffn (", len(want_fc1), "values)")

    # ── does the (2,3) tiny fixture actually distinguish h/d-swap bugs? ────
    print("")
    print("[J] does heads=2/head_dim=3 (shipped fixture) catch a h<->d swap bug?")
    var bug_heads = 2
    var bug_head_dim = 3
    var bug_hidden = 4
    var bug_rows = bug_heads * 3 * bug_head_dim
    var bf = List[Float32]()
    for r in range(bug_rows):
        for c in range(bug_hidden):
            bf.append(Float32(r * bug_hidden + c))
    var correct = minimax_h3_deinterleave_qkv(bf, bug_heads, bug_head_dim, bug_hidden)
    # Manufacture the swapped-multiplier bug directly (dest_row uses `heads`
    # instead of `head_dim` as the per-h stride -- a plausible transcription
    # slip since both symbols are one-letter-different in the source).
    var buggy = List[Float32]()
    for _ in range(len(bf)):
        buggy.append(Float32(0.0))
    for part in range(3):
        for h in range(bug_heads):
            for d in range(bug_head_dim):
                var source_row = h * 3 * bug_head_dim + part * bug_head_dim + d
                var dest_row_buggy = part * bug_heads * bug_head_dim + h * bug_heads + d  # BUG: h*heads not h*head_dim
                for i in range(bug_hidden):
                    buggy[dest_row_buggy * bug_hidden + i] = bf[source_row * bug_hidden + i]
    var caught = False
    for i in range(len(correct)):
        if correct[i] != buggy[i]:
            caught = True
            break
    print("  at (2,3): buggy h*heads-instead-of-h*head_dim variant differs from correct:", caught)

    _extra_degeneracy_check()

    print("")
    print("DONE")


def _extra_degeneracy_check() raises:
    """head_dim=3 in the shipped fixture coincides with the hardcoded qkv
    factor '3' in the formula (h * 3 * head_dim). Check whether a bug that
    confuses the literal 3 with head_dim is masked at (heads=2, head_dim=3)
    but caught at realistic proportions (heads=56, head_dim=128)."""
    print("")
    print("[K] does head_dim==3 (shipped fixture) mask a '3'-vs-head_dim mixup?")

    # -- at the shipped fixture size --
    var heads_s = 2
    var hd_s = 3
    var hidden_s = 4
    var rows_s = heads_s * 3 * hd_s
    var bf_s = List[Float32]()
    for r in range(rows_s):
        for c in range(hidden_s):
            bf_s.append(Float32(r * hidden_s + c))
    var correct_s = minimax_h3_deinterleave_qkv(bf_s, heads_s, hd_s, hidden_s)
    var buggy_s = List[Float32]()
    for _ in range(len(bf_s)):
        buggy_s.append(Float32(0.0))
    for part in range(3):
        for h in range(heads_s):
            for d in range(hd_s):
                var source_row = h * hd_s * hd_s + part * hd_s + d  # BUG: hd*hd, not 3*hd
                var dest_row = part * heads_s * hd_s + h * hd_s + d
                for i in range(hidden_s):
                    buggy_s[dest_row * hidden_s + i] = bf_s[source_row * hidden_s + i]
    var same_s = True
    for i in range(len(correct_s)):
        if correct_s[i] != buggy_s[i]:
            same_s = False
            break
    print("  at (heads=2, head_dim=3): '3*head_dim' vs 'head_dim*head_dim' bug is MASKED (identical output):", same_s)

    # -- at realistic proportions --
    # NOTE: the buggy formula's source_row can run well past the CORRECT
    # buffer size (max source_row = (heads-1)*hd*hd + 2*hd + hd-1, vs. the
    # correct formula's max of (heads-1)*3*hd + 2*hd + hd-1) -- at (56,128)
    # this is 901503 vs 21503, a huge overshoot. Size the SOURCE buffer to
    # the buggy formula's requirement so the buggy read itself doesn't trap;
    # `minimax_h3_deinterleave_qkv` (the correct path) only ever reads the
    # smaller, correct range regardless of how large the source buffer is.
    var heads_r = 56
    var hd_r = 128
    var hidden_r = 2
    var rows_r = heads_r * 3 * hd_r
    var buggy_max_source_row = (heads_r - 1) * hd_r * hd_r + 2 * hd_r + (hd_r - 1)
    var source_rows_r = buggy_max_source_row + 1

    # Exactly-sized buffer for the CORRECT (library) function, which asserts
    # len(fused) == heads*3*head_dim*in_features.
    var bf_r = List[Float32]()
    for r in range(rows_r):
        for c in range(hidden_r):
            bf_r.append(Float32(r * hidden_r + c))
    var correct_r = minimax_h3_deinterleave_qkv(bf_r, heads_r, hd_r, hidden_r)

    # Oversized buffer (same value formula, just more rows) for the manual
    # buggy computation, which needs the larger range to avoid trapping.
    var bf_r_big = List[Float32]()
    for r in range(source_rows_r):
        for c in range(hidden_r):
            bf_r_big.append(Float32(r * hidden_r + c))
    var buggy_r = List[Float32]()
    for _ in range(rows_r * hidden_r):
        buggy_r.append(Float32(0.0))
    for part in range(3):
        for h in range(heads_r):
            for d in range(hd_r):
                var source_row = h * hd_r * hd_r + part * hd_r + d  # same bug
                var dest_row = part * heads_r * hd_r + h * hd_r + d
                for i in range(hidden_r):
                    buggy_r[dest_row * hidden_r + i] = bf_r_big[source_row * hidden_r + i]
    var same_r = True
    for i in range(len(correct_r)):
        if correct_r[i] != buggy_r[i]:
            same_r = False
            break
    print("  at (heads=56, head_dim=128):        same bug MASKED (identical output):", same_r)
