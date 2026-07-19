# Parity gate: pure-Mojo ti2v prep (Qwen3-VL vision preprocess + image-template
# tokenization + text-only negative) vs prep_ti2v.py's ti2v_inputs.safetensors
# (the HF Qwen3-VL processor ground truth). Proves the ti2v conditioning path runs
# with NO Python.
#
# Gates:
#   input_ids  EXACT match (incl. 448-token image block + crop_start 140)
#   neg_ids    EXACT match (text-only negative)
#   pixel_values cos >= 0.999 vs processor (bicubic + normalize + patchify)
#
# Run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#       serenitymojo/models/lingbotvideo/parity/lingbot_ti2v_prep_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer, _read_utf8_file
from serenitymojo.models.lingbotvideo.lingbot_tokenize import (
    tokenize_lingbot_image_prompt,
    tokenize_lingbot_prompt,
    LINGBOT_DEFAULT_NEGATIVE_PROMPT,
)
from serenitymojo.models.lingbotvideo.lingbot_vision_preprocess import (
    lingbot_vision_preprocess,
)

comptime PARITY = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime TOK_JSON = "/mnt/disk1/models/lingbot-video-moe/text_encoder/tokenizer.json"
comptime IMG = "/home/alex/.claude/uploads/6fd4828e-ac2b-4c91-a10b-c346b1b61e18/8280dc5c-1000004560.webp"
comptime MERGE = 2


def _ref_i32(st: ShardedSafeTensors, name: String) raises -> List[Int]:
    var tv = st.tensor_view(name)
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _ref_f32(st: ShardedSafeTensors, name: String) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _cmp_ids(tag: String, got: List[Int], want: List[Int]) -> Bool:
    var m = len(got)
    if len(want) < m:
        m = len(want)
    var first = -1
    for i in range(m):
        if got[i] != want[i]:
            first = i
            break
    var exact = (len(got) == len(want)) and (first == -1)
    if exact:
        print("  ", tag, ": EXACT (", len(got), " ids)")
    else:
        print("  ", tag, ": MISMATCH  len got", len(got), " ref", len(want))
        if first >= 0:
            var lo = first - 3
            if lo < 0:
                lo = 0
            var hi = first + 4
            if hi > m:
                hi = m
            var gs = String("      got[")
            var rs = String("      ref[")
            for i in range(lo, hi):
                if i != lo:
                    gs += String(", ")
                    rs += String(", ")
                gs += String(got[i])
                rs += String(want[i])
            print(gs + String("]  (from idx ") + String(lo) + String(")"))
            print(rs + String("]"))
    return exact


def main() raises:
    var ctx = DeviceContext()
    print("Loading tokenizer + reference ti2v_inputs.safetensors ...")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var rst = ShardedSafeTensors.open(String(PARITY) + "/ti2v_inputs.safetensors")
    var ref_ids = _ref_i32(rst, String("input_ids"))
    var ref_neg = _ref_i32(rst, String("neg_ids"))
    var ref_cs = _ref_i32(rst, String("crop_start"))[0]
    var ref_pv = _ref_f32(rst, String("pixel_values"))

    # ── vision preprocess (image -> pixel_values [seq,1536] + grid) ──────────
    print("\n[1] vision preprocess", IMG)
    var vp = lingbot_vision_preprocess(String(IMG), ctx)
    var n_image = vp.seq // (MERGE * MERGE)
    print("    grid_thw =", vp.grid_t, vp.grid_h, vp.grid_w, " seq =", vp.seq,
          " n_image_tokens =", n_image)

    # ── image-template tokenization (prompt from i2v_prompt.txt, no Python) ──
    print("[2] image-template tokenize + text-only negative")
    var prompt = _read_utf8_file(String(PARITY) + "/i2v_prompt.txt")
    var res = tokenize_lingbot_image_prompt(tok, prompt, n_image)
    var ids = res[0].copy()
    var crop_start = res[1]
    var neg_res = tokenize_lingbot_prompt(tok, String(LINGBOT_DEFAULT_NEGATIVE_PROMPT))
    var neg_ids = neg_res[0].copy()

    # ── gates ────────────────────────────────────────────────────────────────
    print("\n=== ID PARITY ===")
    var ids_ok = _cmp_ids(String("input_ids"), ids, ref_ids)
    var neg_ok = _cmp_ids(String("neg_ids  "), neg_ids, ref_neg)
    var cs_ok = crop_start == ref_cs
    print("   crop_start =", crop_start, " (ref", ref_cs, ") ok=", cs_ok)

    print("\n=== PIXEL PARITY ===")
    var got_pv = vp.pixel_values.to_host(ctx)
    var pv_ok = False
    if len(got_pv) != len(ref_pv):
        print("   pixel_values LENGTH mismatch got", len(got_pv), " ref", len(ref_pv))
    else:
        var dot = Float64(0.0)
        var na = Float64(0.0)
        var nb = Float64(0.0)
        var maxad = Float32(0.0)
        for i in range(len(got_pv)):
            var a = Float64(got_pv[i])
            var b = Float64(ref_pv[i])
            dot += a * b
            na += a * a
            nb += b * b
            var d = got_pv[i] - ref_pv[i]
            if d < 0:
                d = -d
            if d > maxad:
                maxad = d
        var cos = dot / (sqrt(na) * sqrt(nb) + 1e-12)
        pv_ok = cos >= 0.999
        print("   pixel_values cos =", cos, "  max_abs_diff =", maxad,
              "  (", len(got_pv), " elems)")

    print("\n==================================================")
    if ids_ok and neg_ok and cs_ok and pv_ok:
        print("TI2V PREP GATE: PASS — pure-Mojo prep matches the HF processor")
    else:
        print("TI2V PREP GATE: FAIL")
        print("  input_ids ok =", ids_ok, "  neg_ids ok =", neg_ok,
              "  crop_start ok =", cs_ok, "  pixel_values ok =", pv_ok)
    print("==================================================")
    _ = rst^
