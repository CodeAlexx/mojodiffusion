#!/usr/bin/env python3
# check_ltx2_resume_continuation.py — byte-compare the artifacts of a continuous
# LTX2 training run vs a save→resume run at the same final step.
#
# Usage:
#   python3 scripts/check_ltx2_resume_continuation.py <continuous_dir> <resumed_dir> <final_step>
#
# PASS bar: every tensor in the final-step PEFT file AND the final-step .state
# file is BYTE-IDENTICAL between the two runs (the driver's sigma/noise/sample
# streams are (seed, step)-derived, masters are F32-exact in the state, and the
# stack is deterministic — so a lossless resume must reproduce the continuous
# run exactly). Comparison is on RAW header-addressed byte ranges (dtype-
# agnostic — numpy cannot materialize BF16). Any mismatch prints the first
# differing keys.
import json
import struct
import sys
from pathlib import Path


def load_st(path: Path):
    data = path.read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8 : 8 + n])
    hdr.pop("__metadata__", None)
    return hdr, data[8 + n :]


def compare(path_a: Path, path_b: Path) -> tuple[int, int, list[str]]:
    ha, da = load_st(path_a)
    hb, db = load_st(path_b)
    if set(ha) != set(hb):
        only_a = sorted(set(ha) - set(hb))[:5]
        only_b = sorted(set(hb) - set(ha))[:5]
        print(f"  KEYSET DIFF: only-in-A {only_a} only-in-B {only_b}")
        return (0, 1, ["<keyset>"])
    same = 0
    diffs: list[str] = []
    for k in sorted(ha):
        ea, eb = ha[k], hb[k]
        sa, ta = ea["data_offsets"]
        sb, tb = eb["data_offsets"]
        if (
            ea["dtype"] == eb["dtype"]
            and ea["shape"] == eb["shape"]
            and da[sa:ta] == db[sb:tb]
        ):
            same += 1
        else:
            diffs.append(k)
    return (same, len(diffs), diffs)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    cont_dir, res_dir, step = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    ok = True
    for suffix in (f"ltx2_video_lora_step{step}.safetensors",
                   f"ltx2_video_lora_step{step}.state.safetensors"):
        pa, pb = cont_dir / suffix, res_dir / suffix
        if not pa.exists() or not pb.exists():
            print(f"MISSING: {pa if not pa.exists() else pb}")
            ok = False
            continue
        same, ndiff, diffs = compare(pa, pb)
        verdict = "PASS bit-exact" if ndiff == 0 else "FAIL"
        print(f"{suffix}: {same} identical / {ndiff} mismatched -> {verdict}")
        if diffs:
            print(f"  first diffs: {diffs[:8]}")
            ok = False
    print("RESUME CONTINUATION GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
