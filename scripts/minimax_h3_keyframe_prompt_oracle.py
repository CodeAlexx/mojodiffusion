# Oracle for the §2.1 alignment-instruction emitters in
# serenitymojo/pipeline/minimax_h3_ref_prompt.mojo.
#
# No torch, no GPU. The expected strings are EXTRACTED from the vendor's own
# guide — creator-MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md — not
# retyped here, so a template that drifts from the doc fails the gate rather
# than matching a copy of itself.
import json, os, re, sys

DOC = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md"
text = open(DOC, encoding="utf-8").read()
blocks = re.findall(r"```text\n(.*?)```", text, re.S)
print(f"{len(blocks)} ```text blocks in the guide")

def one_liner(pred, what):
    hits = [b.strip("\n") for b in blocks if pred(b)]
    assert hits, f"no block matched {what}"
    return hits

# §2.1 templates (with the N / S.SS placeholders still in them).
tmpl_i2va = one_liner(lambda b: b.startswith("For the target video,"), "I2VA template")[0]
align = one_liner(lambda b: b.startswith("How the reference pictures align"), "alignment templates")
tmpl_fl2va = [a for a in align if "Picture 2" in a][0]
tmpl_l2va = [a for a in align if "Picture 2" not in a and "<Picture 1>" in a][0]

# §5 worked cases — the same lines with concrete values, as the model saw them.
case_lines = []
for b in blocks:
    for line in b.split("\n"):
        if line.startswith("For the target video,") or line.startswith("How the reference pictures align"):
            case_lines.append(line)
case_i2va = [l for l in case_lines if l.startswith("For the target")]
case_fl2va = [l for l in case_lines if "Picture 2" in l and "8.00-second" in l]
case_l2va = [l for l in case_lines if l.startswith("How the") and "Picture 2" not in l and "6.00-second" in l]
assert case_i2va and case_fl2va and case_l2va, (len(case_i2va), len(case_fl2va), len(case_l2va))

# The same templates with N and S.SS substituted, to exercise a final shot != 1.
sub_fl2va = tmpl_fl2va.replace("Shot N", "Shot 3").replace("S.SS", "7.25")
sub_l2va = tmpl_l2va.replace("Shot N", "Shot 3").replace("S.SS", "7.25")

out = {
    "template_i2va": tmpl_i2va,
    "template_fl2va": tmpl_fl2va,
    "template_l2va": tmpl_l2va,
    # I2VA carries no N and no duration, so the template IS the case line.
    "case_i2va": case_i2va[0],
    "case_fl2va": case_fl2va[0],      # 8.00 s == 192 frames, final shot 1
    "case_l2va": case_l2va[0],        # 6.00 s == 144 frames, final shot 1
    "sub_fl2va": sub_fl2va,           # 7.25 s == 174 frames, final shot 3
    "sub_l2va": sub_l2va,
}

# Two-decimal duration formatting, for EVERY legal aligned frame count.
frames = [n for n in range(1, 400) if n % 17 == 5 and 5.0 <= n / 24 <= 15.0]
out["two_decimals"] = {str(n): f"{n / 24:.2f}" for n in frames}
print("legal aligned frame counts:", frames)
print("durations:", [out["two_decimals"][str(n)] for n in frames])

OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_keyframe"
os.makedirs(OUT_DIR, exist_ok=True)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, "keyframe_prompt_ref.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", path)
for k in ("case_i2va", "case_fl2va", "case_l2va", "sub_fl2va"):
    print(f"  {k}: {out[k][:100]}...")
