# celoss_embed_bwd_parity.mojo — GPU verification of CrossEntropy / NLL / BCE /
# Embedding BACKWARD vs a PyTorch reference.
#
# Gate: grad-parity cos >= 0.999 vs celoss_embed_bwd_oracle.py
# (-> celoss_embed_bwd_ref.txt). Inputs are read from the SAME ref file so the
# Mojo kernels are fed byte-identical data; gradient tags are gated.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/celoss_embed_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/celoss_embed_bwd_parity.mojo
#
# _read_ref is the proven tagged-text reader copied verbatim from
# sdpa_bwd_parity.mojo.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/celoss_embed_bwd_ref.txt"
)


# ── read one tagged space-separated float line (verbatim from sdpa_bwd_parity) ─
def _read_ref(tag: String) raises -> List[Float32]:
    var fd = sys_open(String(REF_PATH), O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + String(REF_PATH))
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)

    var prefix = tag + " "
    var pl = prefix.byte_length()
    var out = List[Float32]()
    var i = 0
    while i < done:
        var le = i
        while le < done and Int(buf[le]) != 0x0A:
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(prefix[byte=j]):
                    is_match = False
                    break
        if is_match:
            var p = i + pl
            while p < le:
                var c = Int(buf[p])
                if c == 0x20:
                    p += 1
                    continue
                var ne = p
                while ne < le and Int(buf[ne]) != 0x20:
                    ne += 1
                var chars = List[UInt8]()
                for q in range(p, ne):
                    chars.append(buf[q])
                var s = String(from_utf8=chars)
                out.append(Float32(atof(s)))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


# read a tag as Int indices (round the float text back to Int)
def _read_idx(tag: String) raises -> List[Int]:
    var f = _read_ref(tag)
    var out = List[Int]()
    for i in range(len(f)):
        out.append(Int(f[i] + Float32(0.5)))  # exact small non-neg ints
    return out^


def main() raises:
    from serenitymojo.ops.celoss_embed_backward import (
        cross_entropy_backward,
        nll_backward,
        bce_backward,
        embedding_backward,
    )

    var ctx = DeviceContext()
    var h = ParityHarness()
    var h_bf16 = ParityHarness(0.99)
    var all_pass = True

    # ── CrossEntropy: [N=5, C=7] ──────────────────────────────────────────────
    var ce_N = 5
    var ce_C = 7
    var ce_logits = Tensor.from_host(
        _read_ref(String("ce_logits")), [ce_N, ce_C], STDtype.F32, ctx)
    var ce_target = _read_idx(String("ce_target"))
    var ce_dl = cross_entropy_backward(ce_logits, ce_target, ctx)
    var r_ce = h.compare_host(ce_dl.to_host(ctx), _read_ref(String("ce_dlogits")))
    print("ce_dlogits vs torch:", r_ce)
    all_pass = all_pass and r_ce.passed

    # ── NLLLoss: [N=4, C=6] ───────────────────────────────────────────────────
    var nl_N = 4
    var nl_C = 6
    var nl_lp = Tensor.from_host(
        _read_ref(String("nll_logprobs")), [nl_N, nl_C], STDtype.F32, ctx)
    var nl_target = _read_idx(String("nll_target"))
    var nl_d = nll_backward(nl_lp, nl_target, ctx)
    var r_nll = h.compare_host(nl_d.to_host(ctx), _read_ref(String("nll_dlp")))
    print("nll_dlp vs torch:", r_nll)
    all_pass = all_pass and r_nll.passed

    # ── BCELoss (PLAIN prob form): [12] ───────────────────────────────────────
    var b_N = 12
    var b_pred = Tensor.from_host(
        _read_ref(String("bce_pred")), [b_N], STDtype.F32, ctx)
    var b_target = Tensor.from_host(
        _read_ref(String("bce_target")), [b_N], STDtype.F32, ctx)
    var b_d = bce_backward(b_pred, b_target, ctx)
    var r_bce = h.compare_host(b_d.to_host(ctx), _read_ref(String("bce_dpred")))
    print("bce_dpred vs torch (PLAIN prob form):", r_bce)
    all_pass = all_pass and r_bce.passed

    # ── Embedding: table [10, 8], idx [6] (repeats accumulate) ────────────────
    var num_emb = 10
    var dim = 8
    var n_idx = 6
    var e_idx = _read_idx(String("embed_idx"))
    var e_go = Tensor.from_host(
        _read_ref(String("embed_gradout")), [n_idx, dim], STDtype.F32, ctx)
    var e_d = embedding_backward(e_go, e_idx, num_emb, ctx)
    var r_emb = h.compare_host(e_d.to_host(ctx), _read_ref(String("embed_dtable")))
    print("embed_dtable vs torch:", r_emb)
    all_pass = all_pass and r_emb.passed

    # ── BF16 storage smoke: same F32 refs, relaxed cos. These are not bit parity
    # gates; they prove the storage-preserving kernels run and return BF16.
    var ce_logits_b = Tensor.from_host(
        _read_ref(String("ce_logits")), [ce_N, ce_C], STDtype.BF16, ctx)
    var ce_dl_b = cross_entropy_backward(ce_logits_b, ce_target, ctx)
    var r_ce_b = h_bf16.compare_host(
        ce_dl_b.to_host(ctx), _read_ref(String("ce_dlogits")))
    var ce_b_ok = ce_dl_b.dtype() == STDtype.BF16 and r_ce_b.passed
    print("bf16 ce_dlogits storage/parity:", r_ce_b, " dtype=", ce_dl_b.dtype().name())
    all_pass = all_pass and ce_b_ok

    var nl_lp_b = Tensor.from_host(
        _read_ref(String("nll_logprobs")), [nl_N, nl_C], STDtype.BF16, ctx)
    var nl_d_b = nll_backward(nl_lp_b, nl_target, ctx)
    var r_nll_b = h_bf16.compare_host(
        nl_d_b.to_host(ctx), _read_ref(String("nll_dlp")))
    var nll_b_ok = nl_d_b.dtype() == STDtype.BF16 and r_nll_b.passed
    print("bf16 nll_dlp storage/parity:", r_nll_b, " dtype=", nl_d_b.dtype().name())
    all_pass = all_pass and nll_b_ok

    var b_pred_b = Tensor.from_host(
        _read_ref(String("bce_pred")), [b_N], STDtype.BF16, ctx)
    var b_target_b = Tensor.from_host(
        _read_ref(String("bce_target")), [b_N], STDtype.BF16, ctx)
    var b_d_b = bce_backward(b_pred_b, b_target_b, ctx)
    var r_bce_b = h_bf16.compare_host(
        b_d_b.to_host(ctx), _read_ref(String("bce_dpred")))
    var bce_b_ok = b_d_b.dtype() == STDtype.BF16 and r_bce_b.passed
    print("bf16 bce_dpred storage/parity:", r_bce_b, " dtype=", b_d_b.dtype().name())
    all_pass = all_pass and bce_b_ok

    var e_go_b = Tensor.from_host(
        _read_ref(String("embed_gradout")), [n_idx, dim], STDtype.BF16, ctx)
    var e_d_b = embedding_backward(e_go_b, e_idx, num_emb, ctx)
    var r_emb_b = h_bf16.compare_host(
        e_d_b.to_host(ctx), _read_ref(String("embed_dtable")))
    var emb_b_ok = e_d_b.dtype() == STDtype.BF16 and r_emb_b.passed
    print("bf16 embed_dtable storage/parity:", r_emb_b, " dtype=", e_d_b.dtype().name())
    all_pass = all_pass and emb_b_ok

    print("")
    if all_pass:
        print("ALL CELOSS/EMBED BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("CELOSS/EMBED BACKWARD PARITY FAILURE")
        raise Error("celoss_embed_bwd_parity gate failed")
