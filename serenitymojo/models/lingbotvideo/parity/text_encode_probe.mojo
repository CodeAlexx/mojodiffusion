# models/lingbotvideo/parity/text_encode_probe.mojo — Qwen3-VL TEXT parity gate.
#
# Proves the pure-Mojo LingBot text encoder reproduces the torch Qwen3-VL apple
# prompt_embeds. Loads the captured full input_ids + crop_start
# (oracle_text_ids.safetensors), runs load_lingbot_qwen3vl + encode_lingbot_text,
# and compares vs oracle_e.safetensors `prompt_embeds` [1,457,2560] (post-crop,
# post-model.norm). Gate: cos >= 0.999.
#
# Run:
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . \
#       serenitymojo/models/lingbotvideo/parity/text_encode_probe.mojo
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.lingbot_qwen3vl import (
    load_lingbot_qwen3vl, encode_lingbot_text, LINGBOT_TE_DIR,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx).to_host(ctx)


def _read_i32(st: ShardedSafeTensors, name: String) raises -> List[Int]:
    """Read an I32 tensor's values straight off the mmap host span."""
    var tv = st.tensor_view(name)
    var n = tv.numel()
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int]()
    for i in range(n):
        out.append(Int(p[i]))
    return out^


def _cos_maxabs(mine: List[Float32], reference: List[Float32]) -> Tuple[Float64, Float64, Float64]:
    """Return (cos, max_abs_diff, |mine|/|ref|)."""
    var dot: Float64 = 0.0
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    var maxabs: Float64 = 0.0
    var n = len(mine) if len(mine) < len(reference) else len(reference)
    for i in range(n):
        var a = Float64(mine[i])
        var b = Float64(reference[i])
        dot += a * b
        nm += a * a
        nr += b * b
        var d = a - b
        if d < 0.0:
            d = -d
        if d > maxabs:
            maxabs = d
    var cos = dot / (sqrt(nm) * sqrt(nr)) if (nm > 0.0 and nr > 0.0) else 0.0
    var mag = sqrt(nm) / sqrt(nr) if nr > 0.0 else 0.0
    return (cos, maxabs, mag)


def main() raises:
    var ctx = DeviceContext()

    print("[T] reading captured ids:", String(PARITY_DIR) + "/oracle_text_ids.safetensors")
    var idst = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_text_ids.safetensors")
    var ids = _read_i32(idst, "input_ids")
    var crop_start = _read_i32(idst, "crop_start")[0]
    var true_len = _read_i32(idst, "true_len")[0]
    print("[T] full_len =", len(ids), " crop_start =", crop_start,
          " true_len =", true_len, " post-crop =", len(ids) - crop_start)

    print("[T] loading Qwen3-VL text encoder from", LINGBOT_TE_DIR)
    var enc = load_lingbot_qwen3vl(String(LINGBOT_TE_DIR), ctx)

    print("[T] encode_lingbot_text ...")
    var emb = encode_lingbot_text(enc, ids, crop_start, ctx)
    var es = emb.shape()
    print("[T] mine embeds shape = [", es[0], ",", es[1], ",", es[2], "]")
    var mine = emb.to_host(ctx)

    print("[T] loading oracle prompt_embeds [1,457,2560]")
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_e.safetensors")
    var reference = _load_f32(oracle, "prompt_embeds", ctx)

    var cm = _cos_maxabs(mine, reference)
    print("[T] ===== TEXT ENCODE PARITY =====")
    print("   n(mine) =", len(mine), " n(ref) =", len(reference))
    print("   cos       =", cm[0])
    print("   max_abs   =", cm[1])
    print("   |mine|/|ref| =", cm[2])

    if len(mine) != len(reference):
        print("[T] ===== SHAPE MISMATCH (mine != ref numel) — FAIL =====")
    elif cm[0] >= 0.999:
        print("[T] ===== TEXT GATE PASS (cos >= 0.999) =====")
    else:
        print("[T] ===== TEXT GATE FAIL (cos < 0.999) — inspect tap/crop/norm =====")
