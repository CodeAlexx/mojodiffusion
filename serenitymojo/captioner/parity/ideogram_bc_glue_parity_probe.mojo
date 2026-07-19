# ideogram_bc_glue_parity_probe.mojo — byte-exact parity gate for the pure-Mojo
# port of the B+C captioner GLUE (ideogram_bc_glue.mojo).
#
# Loads the fixture set produced from ai-toolkit's OWN deterministic glue (no GPU,
# no LLM) and asserts every case is BYTE-IDENTICAL. 12 groups:
#   b_compute_aspect_ratio   in [w,h]               → out string
#   b_build_prompt           in {caption_prompt,ar} → out_sha256_len + out_head/out_tail
#   b_convert_bbox           in bbox|junk           → out [y1,x1,y2,x2] | null  (x/y SWAP)
#   b_extract_json           in raw string          → out dict | null
#   b_full_glue              in raw string          → out pretty(indent=2) | swapped-raw
#   c_build_prompt           in {ar,op,creative,ins}→ out_sha256_len + out_head/out_tail
#   c_sanitize_bbox          in bbox|junk           → out [y1,x1,y2,x2] | null  (NO swap)
#   c_extract_json           in raw string          → out dict | null
#   c_normalize_item         in str|dict + default  → out [idea,ar] | null
#   c_full_glue              in raw string          → out compact-default | null
#   b_template_anchors       {len,sha256,head,tail} → embedded B template matches
#   c_template_anchors       {len,sha256,head,tail} → embedded C template matches
#
# CPU-only string probe (no GPU). FAIL-LOUD: non-zero exit on ANY mismatch.
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . -I /home/alex/MOJO-libs \
#     serenitymojo/captioner/parity/ideogram_bc_glue_parity_probe.mojo

from std.memory import alloc

from pdf.sha256 import sha256_hex

from serenitymojo.io.ffi import (
    sys_open,
    sys_close,
    sys_pread,
    file_size,
    BytePtr,
    O_RDONLY,
)
from serenitymojo.captioner.ideogram_caption import (
    JValue,
    parse_json,
    to_model_string,
    _bytes_to_string,
)
from serenitymojo.captioner.ideogram_bc_glue import (
    compute_aspect_ratio,
    b_build_prompt,
    c_build_prompt,
    b_convert_bbox,
    c_sanitize_bbox,
    extract_json,
    b_full_glue,
    c_full_glue,
    normalize_item,
    load_b_template,
    load_c_template,
    BBoxResult,
)

comptime FIXTURES = "/home/alex/EriTrainer/trainer/parity/ideogram_bc_glue/fixtures.json"
comptime ADV_FIXTURES = "/home/alex/mojodiffusion/serenitymojo/captioner/parity/ideogram_bc_glue_adv_fixtures.json"


def _read_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error("cannot open fixtures: " + path)
    var sz = file_size(fd)
    var tmp = alloc[UInt8](sz)
    var bp = BytePtr(unsafe_from_address=Int(tmp))
    var done = 0
    while done < sz:
        var got = sys_pread(fd, bp + done, sz - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var buf = List[UInt8]()
    for i in range(done):
        buf.append(tmp[i])
    tmp.free()
    return _bytes_to_string(buf)


def _sha256_hex(s: String) raises -> String:
    var data = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        data.append(b[i])
    return sha256_hex(data)


def _bbox_to_str(r: BBoxResult) -> String:
    # Render a BBoxResult the way the fixture stores it: a 4-int list, or "null".
    if not r.ok:
        return String("null")
    return (
        "[" + String(r.y1) + "," + String(r.x1) + "," + String(r.y2) + "," + String(r.x2) + "]"
    )


def _jval_bbox_to_str(v: JValue) -> String:
    # The fixture's expected bbox out is a JSON list-of-ints or null. Serialize it
    # the same compact way for comparison.
    if v.is_null():
        return String("null")
    return to_model_string(v)


def main() raises:
    var raw = _read_file(FIXTURES)
    var root = parse_json(raw)

    var total = 0
    var passed = 0
    var fails = List[String]()

    # ── b_compute_aspect_ratio ──────────────────────────────────────────────────
    var bcar = root.get("b_compute_aspect_ratio")
    for i in range(len(bcar.arr)):
        var c = bcar.arr[i].copy()
        var pair = c.get("in")
        var w = pair.arr[0].copy()
        var h = pair.arr[1].copy()
        var got = compute_aspect_ratio(_jint(w), _jint(h))
        var exp = c.get("out").s
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_compute_aspect_ratio[" + String(i) + "] in=[" + String(_jint(w)) + "," + String(_jint(h)) + "] exp=" + exp + " got=" + got)

    # ── b_build_prompt (sha256+len + head/tail) ─────────────────────────────────
    var btpl = load_b_template()
    var bbp = root.get("b_build_prompt")
    for i in range(len(bbp.arr)):
        var c = bbp.arr[i].copy()
        var inp = c.get("in")
        var cp_v = inp.get("caption_prompt")
        var has_cp = inp.has_key("caption_prompt") and cp_v.is_string()
        var cp = cp_v.s if cp_v.is_string() else String("")
        var ar = inp.get("aspect_ratio").s
        var prompt = b_build_prompt(btpl, cp, has_cp, ar)
        _check_prompt("b_build_prompt", i, prompt, c, passed, fails)
        total += 1

    # ── b_convert_bbox ──────────────────────────────────────────────────────────
    var bcb = root.get("b_convert_bbox")
    for i in range(len(bcb.arr)):
        var c = bcb.arr[i].copy()
        var got = _bbox_to_str(b_convert_bbox(c.get("in")))
        var exp = _jval_bbox_to_str(c.get("out"))
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_convert_bbox[" + String(i) + "] exp=" + exp + " got=" + got)

    # ── b_extract_json (dict|null) ──────────────────────────────────────────────
    var bej = root.get("b_extract_json")
    for i in range(len(bej.arr)):
        var c = bej.arr[i].copy()
        var ex = extract_json(c.get("in").s)
        var got = String("null") if not ex.ok else to_model_string(ex.value)
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else to_model_string(out_v)
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_extract_json[" + String(i) + "]\n  in : " + _trunc(c.get("in").s) + "\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    # ── b_full_glue ─────────────────────────────────────────────────────────────
    var bfg = root.get("b_full_glue")
    for i in range(len(bfg.arr)):
        var c = bfg.arr[i].copy()
        var got = b_full_glue(c.get("in").s)
        var exp = c.get("out").s
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_full_glue[" + String(i) + "]\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    # ── c_build_prompt ──────────────────────────────────────────────────────────
    var ctpl = load_c_template()
    var cbp = root.get("c_build_prompt")
    for i in range(len(cbp.arr)):
        var c = cbp.arr[i].copy()
        var inp = c.get("in")
        var ar = inp.get("aspect_ratio").s
        var op = inp.get("original_prompt").s
        var creative = inp.get("creative").b
        var ins = inp.get("instructions").s
        var prompt = c_build_prompt(ctpl, ar, op, creative, ins)
        _check_prompt("c_build_prompt", i, prompt, c, passed, fails)
        total += 1

    # ── c_sanitize_bbox ─────────────────────────────────────────────────────────
    var csb = root.get("c_sanitize_bbox")
    for i in range(len(csb.arr)):
        var c = csb.arr[i].copy()
        var got = _bbox_to_str(c_sanitize_bbox(c.get("in")))
        var exp = _jval_bbox_to_str(c.get("out"))
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_sanitize_bbox[" + String(i) + "] exp=" + exp + " got=" + got)

    # ── c_extract_json ──────────────────────────────────────────────────────────
    var cej = root.get("c_extract_json")
    for i in range(len(cej.arr)):
        var c = cej.arr[i].copy()
        var ex = extract_json(c.get("in").s)
        var got = String("null") if not ex.ok else to_model_string(ex.value)
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else to_model_string(out_v)
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_extract_json[" + String(i) + "]\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    # ── c_normalize_item ([idea,ar]|null) ───────────────────────────────────────
    var cni = root.get("c_normalize_item")
    for i in range(len(cni.arr)):
        var c = cni.arr[i].copy()
        var default_ar = c.get("default_ar").s
        var r = normalize_item(c.get("in"), default_ar)
        var got: String
        if not r.ok:
            got = String("null")
        else:
            # Serialize [idea, ar] the compact module-A way for comparison.
            var arr = JValue.new_array()
            arr.append(JValue.from_string(r.idea))
            arr.append(JValue.from_string(r.aspect_ratio))
            got = to_model_string(arr)
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else to_model_string(out_v)
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_normalize_item[" + String(i) + "] exp=" + exp + " got=" + got)

    # ── c_full_glue (compact-default | null) ────────────────────────────────────
    var cfg = root.get("c_full_glue")
    for i in range(len(cfg.arr)):
        var c = cfg.arr[i].copy()
        var r = c_full_glue(c.get("in").s)
        var got = String("null") if not r.ok else r.value
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else out_v.s
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_full_glue[" + String(i) + "]\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    # ── b_template_anchors / c_template_anchors ─────────────────────────────────
    total += _check_anchor("b_template_anchors", btpl, root.get("b_template_anchors"), passed, fails)
    total += _check_anchor("c_template_anchors", ctpl, root.get("c_template_anchors"), passed, fails)

    # ── ADVERSARIAL: fence-load-bearing extract_json + full_glue ────────────────
    # The committed fixtures' fenced inputs happen NOT to distinguish the fence-strip
    # from the brace-span fallback (the {...} between fences is the same span the
    # brace logic finds). These extra cases (regenerated from the REAL ai-toolkit
    # glue) DO distinguish it: a closing fence followed by prose with a stray '}', so
    # without a correct lazy-DOTALL first-fence scanner the brace-span over-extends
    # and parses differently. This makes the hand-rolled fence scanner genuinely
    # gated. NaN/Infinity etc. are out of scope (handled by module A's strict parser).
    var adv = parse_json(_read_file(ADV_FIXTURES))
    var baej = adv.get("b_extract_json_adv")
    for i in range(len(baej.arr)):
        var c = baej.arr[i].copy()
        var ex = extract_json(c.get("in").s)
        var got = String("null") if not ex.ok else to_model_string(ex.value)
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else to_model_string(out_v)
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_extract_json_adv[" + String(i) + "]\n  in : " + _trunc(c.get("in").s) + "\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    var caej = adv.get("c_extract_json_adv")
    for i in range(len(caej.arr)):
        var c = caej.arr[i].copy()
        var ex = extract_json(c.get("in").s)
        var got = String("null") if not ex.ok else to_model_string(ex.value)
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else to_model_string(out_v)
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_extract_json_adv[" + String(i) + "]\n  in : " + _trunc(c.get("in").s) + "\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    var bfga = adv.get("b_full_glue_adv")
    for i in range(len(bfga.arr)):
        var c = bfga.arr[i].copy()
        var got = b_full_glue(c.get("in").s)
        var exp = c.get("out").s
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_full_glue_adv[" + String(i) + "]\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    var cfga = adv.get("c_full_glue_adv")
    for i in range(len(cfga.arr)):
        var c = cfga.arr[i].copy()
        var r = c_full_glue(c.get("in").s)
        var got = String("null") if not r.ok else r.value
        var out_v = c.get("out")
        var exp = String("null") if out_v.is_null() else out_v.s
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_full_glue_adv[" + String(i) + "]\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got))

    # ── ADVERSARIAL: bbox float() coercion (FRAGILE #1 fix) ─────────────────────
    # Python `[float(v) for v in bbox]` coerces numbers, bools (True→1.0/False→0.0)
    # and numeric strings ("100"→100.0); None/list/dict or a non-numeric string raise
    # → drop the box. Generated from the REAL ai-toolkit glue. B SWAPS x/y, C does not.
    var bcc = adv.get("b_convert_bbox_coerce")
    for i in range(len(bcc.arr)):
        var c = bcc.arr[i].copy()
        var got = _bbox_to_str(b_convert_bbox(c.get("in")))
        var exp = _jval_bbox_to_str(c.get("out"))
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("b_convert_bbox_coerce[" + String(i) + "] exp=" + exp + " got=" + got)

    var ccc = adv.get("c_sanitize_bbox_coerce")
    for i in range(len(ccc.arr)):
        var c = ccc.arr[i].copy()
        var got = _bbox_to_str(c_sanitize_bbox(c.get("in")))
        var exp = _jval_bbox_to_str(c.get("out"))
        total += 1
        if got == exp:
            passed += 1
        else:
            fails.append("c_sanitize_bbox_coerce[" + String(i) + "] exp=" + exp + " got=" + got)

    # ── report ──────────────────────────────────────────────────────────────────
    print("ideogram_bc_glue parity:", passed, "/", total, "PASS")
    if len(fails) > 0:
        print("")
        print("FAILURES (", len(fails), "):")
        for i in range(len(fails)):
            print("----------------------------------------")
            print(fails[i])
    if passed != total:
        raise Error("BC GLUE PARITY FAILED: " + String(passed) + "/" + String(total))
    print("ALL BC GLUE CASES BYTE-IDENTICAL")


def _jint(v: JValue) -> Int:
    # JSON int token → Int.
    if v.kind == 2 or v.kind == 3:
        try:
            return Int(atof(v.num_text))
        except:
            return 0
    return 0


def _check_prompt(group: String, idx: Int, prompt: String, c: JValue, mut passed: Int, mut fails: List[String]) raises:
    # Compare sha256(bytes)+codepoint-len + head[:120]/tail vs the fixture.
    var sl = c.get("out_sha256_len")
    var exp_sha = sl.arr[0].s
    var exp_len = _jint(sl.arr[1].copy())
    var got_sha = _sha256_hex(prompt)
    var got_len = prompt.count_codepoints()
    var exp_head = c.get("out_head").s
    var exp_tail = c.get("out_tail").s
    var ok = (got_sha == exp_sha) and (got_len == exp_len)
    if ok:
        passed += 1
    else:
        fails.append(
            group + "[" + String(idx) + "]\n  exp sha/len: " + exp_sha + " / " + String(exp_len)
            + "\n  got sha/len: " + got_sha + " / " + String(got_len)
            + "\n  exp_head: " + _trunc(exp_head) + "\n  exp_tail: " + _trunc(exp_tail)
        )


def _check_anchor(name: String, template: String, anchor: JValue, mut passed: Int, mut fails: List[String]) raises -> Int:
    # anchor = {len, sha256, head, tail}. Verify the EMBEDDED template matches.
    var exp_len = _jint(anchor.get("len"))
    var exp_sha = anchor.get("sha256").s
    var got_len = template.count_codepoints()
    var got_sha = _sha256_hex(template)
    if got_len == exp_len and got_sha == exp_sha:
        passed += 1
    else:
        fails.append(
            name + "\n  exp len/sha: " + String(exp_len) + " / " + exp_sha
            + "\n  got len/sha: " + String(got_len) + " / " + got_sha
        )
    return 1


def _trunc(s: String) -> String:
    var b = s.as_bytes()
    if len(b) <= 300:
        return s
    var out = List[UInt8]()
    for i in range(300):
        out.append(b[i])
    return _bytes_to_string(out) + " …(" + String(len(b)) + "b)"
