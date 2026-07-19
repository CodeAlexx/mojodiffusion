#!/usr/bin/env python3
"""Step-1 eval-1 divergence compare: Mojo refhq pipeline vs ltx_core reference.

Inputs:
  ref : output/ltx2_hq_ref_golden/step1_ref_dump.safetensors
        (scratchpad/ref_step1_dump.py — torch hooks on the golden setup)
  mojo: output/ltx2_step1_mojo/step1_mojo_dump.safetensors
        (ltx2_t2v_av_hq.mojo `refhq1` or 16 GiB-safe `refhq1lite` mode)

Compares, in pipeline order, every pipeline-built input and per-block hidden
state. THE FIRST TENSOR THAT DISAGREES is the bug's address.
Layout adapters:
  - Mojo rope tables are [P*heads, hd] token-major; ref is [1, heads, P, hd].
  - Mojo mod tensors may be broadcast [1,1,D] vs ref per-token [1,S,D].
"""
import argparse
import sys

import torch
from safetensors.torch import load_file

REF = "/home/alex/mojodiffusion/output/ltx2_hq_ref_golden/step1_ref_dump.safetensors"
import os
MOJO = os.environ.get("MOJO_DUMP", "/home/alex/mojodiffusion/output/ltx2_step1_mojo/step1_mojo_dump.safetensors")
REF = os.environ.get("REF_DUMP", REF)


def stat(name, m, r, bar=0.999):
    if m is None or r is None:
        print(f"  {name:24s} MISSING ({'mojo' if m is None else 'ref'})")
        return False
    m = m.double().flatten()
    r = r.double().flatten()
    if m.numel() != r.numel():
        print(f"  {name:24s} SIZE mojo{m.numel()} != ref{r.numel()}")
        return False
    c = float(m @ r / (m.norm() * r.norm() + 1e-30))
    rl = float((m - r).norm() / (r.norm() + 1e-30))
    sr = float(m.std() / (r.std() + 1e-30))
    flag = "PASS" if c >= bar else "FAIL <<<<"
    print(f"  {name:24s} cos={c:.7f} relL2={rl:.6f} std_ratio={sr:.4f} {flag}")
    return c >= bar


def rope_m2r(t, heads):
    # [P*heads, hd] token-major -> [heads, P, hd]
    ph, hd = t.shape
    p = ph // heads
    return t.reshape(p, heads, hd).permute(1, 0, 2)


def bcast(m, r):
    """Broadcast mojo [1,1,D] to ref [1,S,D] if needed."""
    if m.shape == r.shape:
        return m
    if m.shape[1] == 1 and r.shape[1] > 1:
        return m.expand(-1, r.shape[1], -1)
    if r.shape[1] == 1 and m.shape[1] > 1:
        return m[:, :1]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dump", nargs="?", default=REF)
    ap.add_argument("mojo_dump", nargs="?", default=MOJO)
    ap.add_argument("--bar", type=float, default=0.999)
    args = ap.parse_args()
    ref_path = os.path.abspath(args.ref_dump)
    mojo_path = os.path.abspath(args.mojo_dump)
    print("reference:", ref_path)
    print("mojo:", mojo_path)
    ref = load_file(ref_path)
    mojo = load_file(mojo_path)
    results = []

    def check(name, m, r):
        results.append((name, stat(name, m, r, args.bar)))

    detailed = "enc" in mojo
    print("mode:", "detailed" if detailed else "lite")
    if detailed:
        print("== contexts (post-connector) ==")
        check("enc", mojo["enc"], ref["p0_v_context"])
        check("aenc", mojo["aenc"], ref["p0_a_context"])
        check("neg_enc", mojo["neg_enc"], ref["p1_v_context"])
        check("neg_aenc", mojo["neg_aenc"], ref["p1_a_context"])

    print("== init latents ==")
    check("init_v", mojo["init_v"], ref["p0_v_latent"])
    check("init_a", mojo["init_a"], ref["p0_a_latent"])

    if detailed:
        print("== rope tables ==")
        check("v_pe_cos", rope_m2r(mojo["v_pe_cos"], 32), ref["p0_v_pe_cos"][0])
        check("v_pe_sin", rope_m2r(mojo["v_pe_sin"], 32), ref["p0_v_pe_sin"][0])
        check("a_pe_cos", rope_m2r(mojo["a_pe_cos"], 32), ref["p0_a_pe_cos"][0])
        check("a_pe_sin", rope_m2r(mojo["a_pe_sin"], 32), ref["p0_a_pe_sin"][0])
        check("v_cpe_cos", rope_m2r(mojo["v_cpe_cos"], 32), ref["p0_v_cpe_cos"][0])
        check("v_cpe_sin", rope_m2r(mojo["v_cpe_sin"], 32), ref["p0_v_cpe_sin"][0])
        check("a_cpe_cos", rope_m2r(mojo["a_cpe_cos"], 32), ref["p0_a_cpe_cos"][0])
        check("a_cpe_sin", rope_m2r(mojo["a_cpe_sin"], 32), ref["p0_a_cpe_sin"][0])

        print("== modulation (adaln outputs) ==")
        for mk, rk in [("v_timesteps", "p0_v_timesteps"), ("a_timesteps", "p0_a_timesteps"),
                       ("v_embedded", "p0_v_embedded"), ("a_embedded", "p0_a_embedded"),
                       ("v_ca_ss", "p0_v_ca_ss"), ("a_ca_ss", "p0_a_ca_ss"),
                       ("v_ca_gate", "p0_v_ca_gate"), ("a_ca_gate", "p0_a_ca_gate"),
                       ("v_prompt_ts", "p0_v_prompt_ts"), ("a_prompt_ts", "p0_a_prompt_ts")]:
            check(mk, bcast(mojo[mk], ref[rk]), ref[rk])

        print("== post-patchify tokens ==")
        for p in range(3):
            check(f"p{p}_v_x", mojo[f"p{p}_v_x"], ref[f"p{p}_v_x"])
            check(f"p{p}_a_x", mojo[f"p{p}_a_x"], ref[f"p{p}_a_x"])

        print("== per-block hidden states ==")
        for p in range(3):
            for b in (0, 1, 2, 8, 24, 47):
                check(f"p{p}_blk{b:02d}_v", mojo[f"p{p}_blk{b:02d}_v"], ref[f"p{p}_blk{b:02d}_v"])
                check(f"p{p}_blk{b:02d}_a", mojo[f"p{p}_blk{b:02d}_a"], ref[f"p{p}_blk{b:02d}_a"])

    print("== velocities ==")
    for p in range(3):
        check(f"p{p}_vel_v", mojo[f"p{p}_vel_v"], ref[f"p{p}_vel_v"])
        check(f"p{p}_vel_a", mojo[f"p{p}_vel_a"], ref[f"p{p}_vel_a"])

    print("== x0 / guider ==")
    for k in ["x0_cond_v", "x0_uncond_v", "x0_mod_v", "x0_cond_a", "x0_uncond_a",
              "x0_mod_a", "guided_v", "guided_a"]:
        check(k, mojo[k], ref[k])

    failed = [name for name, passed in results if not passed]
    print(f"summary: {len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
