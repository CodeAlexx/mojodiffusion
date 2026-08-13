# mageflow_textcond_probe.mojo — Mage-Flow Qwen3-VL TEXT conditioning parity gate.
#
# Reads the oracle's EXACT input_ids (full templated sequence) + the reference
# txt [1, L-34, 2560] (POST model.norm last_hidden_state, system prefix dropped)
# and vec [2560] from mageflow_textcond_oracle.py, loads the REAL Mage-Flow
# Qwen3-VL text stack via `load_krea2_qwen3vl_4b` (Mage's `model.language_model.*`
# shard keys are a byte-for-byte fit for that loader's remap), runs
# `encode_mageflow_text` (drop_idx=34), and compares vs torch (cos >= 0.999).
# vec is host-recomputed as mean over the kept rows and compared too.
# FAIL-LOUD: a length mismatch or cos < 0.999 raises.
#
# Run (oracle FIRST — dumps byte-identical token ids + reference txt/vec):
#   cd <worktree>
#   pixi run python \
#     serenitymojo/models/text_encoder/parity/mageflow_textcond_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#     serenitymojo/models/text_encoder/parity/mageflow_textcond_probe.mojo
from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.models.text_encoder.krea2_qwen3vl_4b import load_krea2_qwen3vl_4b
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_text,
    MAGEFLOW_T2I_DROP_IDX,
)


comptime DUMP = (
    "serenitymojo/models/text_encoder/parity/mageflow_dumps/"
)
comptime TE_DIR = (
    "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/text_encoder"
)
comptime HIDDEN = 2560


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(
            String("cannot open (run mageflow_textcond_oracle.py first): ") + path
        )
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle dump: ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var cnt = done // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(cnt):
        out.append(fp[i])
    buf.free()
    return out^


def _read_ids(path: String) raises -> List[Int]:
    """Token ids dumped as raw little-endian F32 -> Int."""
    var f = _read_bin_f32(path)
    var ids = List[Int]()
    for i in range(len(f)):
        ids.append(Int(f[i]))
    return ids^


def _std(vals: List[Float32]) -> Float32:
    var n = len(vals)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += vals[i]
    mean /= Float32(n)
    var acc = Float32(0.0)
    for i in range(n):
        var d = vals[i] - mean
        acc += d * d
    acc /= Float32(n)
    return sqrt(acc)


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            String("cos: length mismatch ") + String(len(a)) + " vs " + String(len(b))
        )
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    var denom = sqrt(na) * sqrt(nb)
    return 1.0 if denom == 0.0 else dot / denom


def _mean_rows(host: List[Float32], keep: Int, hidden: Int) -> List[Float32]:
    """Mean over the row (sequence) axis of a [keep, hidden] row-major buffer."""
    var out = List[Float32]()
    for j in range(hidden):
        var acc = Float32(0.0)
        for i in range(keep):
            acc += host[i * hidden + j]
        out.append(acc / Float32(keep))
    return out^


def main() raises:
    var ctx = DeviceContext()

    var ids = _read_ids(DUMP + "mageflow_input_ids.bin")
    var L_full = len(ids)
    var keep = L_full - MAGEFLOW_T2I_DROP_IDX
    print(
        "[mageflow-te-probe] loaded", L_full, "token ids (full templated seq);",
        "first/last =", ids[0], "/", ids[L_full - 1],
        "| drop_idx =", MAGEFLOW_T2I_DROP_IDX, "-> keep =", keep,
    )

    print("[mageflow-te-probe] loading Mage-Flow Qwen3-VL text stack (bf16) ...")
    var enc = load_krea2_qwen3vl_4b(TE_DIR, ctx)
    print("[mageflow-te-probe] encoder loaded (reused krea2 loader as-is).")

    var txt = encode_mageflow_text(enc, ids, MAGEFLOW_T2I_DROP_IDX, ctx)
    var sh = txt.shape()
    print(
        "[mageflow-te-probe] txt shape = [", sh[0], ",", sh[1], ",", sh[2], "]",
    )

    # Shape gate: must be [1, keep, 2560].
    if sh[0] != 1 or sh[1] != keep or sh[2] != HIDDEN:
        raise Error(
            String("mageflow-te-probe: shape mismatch, got [")
            + String(sh[0]) + "," + String(sh[1]) + "," + String(sh[2])
            + "] expected [1," + String(keep) + ",2560]"
        )

    var ref_txt = _read_bin_f32(DUMP + "mageflow_txt.bin")
    var host = txt.to_host(ctx)
    print(
        "[mageflow-te-probe] mojo txt std =", _std(host),
        "| n_mojo =", len(host), "n_ref =", len(ref_txt),
    )

    var harness = ParityHarness(0.999)
    var res = harness.compare(txt, ref_txt, ctx)
    print("[mageflow-te-probe] TXT   ", res)

    # vec = mean over kept rows; recompute on host + compare to oracle vec.
    var ref_vec = _read_bin_f32(DUMP + "mageflow_vec.bin")
    var vec = _mean_rows(host, keep, HIDDEN)
    var vec_cos = _cos(vec, ref_vec)
    print(
        "[mageflow-te-probe] VEC    cos =", vec_cos,
        "| n_vec =", len(vec), "n_ref =", len(ref_vec),
        "| mojo std =", _std(vec),
    )

    if not res.passed:
        raise Error(
            String("mageflow-te-probe FAIL (txt): cos=") + String(res.cos)
            + " < 0.999 (max_abs=" + String(res.max_abs) + ")"
        )
    if vec_cos < 0.999:
        raise Error(
            String("mageflow-te-probe FAIL (vec): cos=") + String(vec_cos) + " < 0.999"
        )
    print("[mageflow-te-probe] PASS — Mage-Flow Qwen3-VL text conditioning matches torch.")
