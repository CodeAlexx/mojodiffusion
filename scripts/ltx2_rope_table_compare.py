#!/usr/bin/env python
"""Compare the Mojo training-head video-RoPE tables vs musubi's OWN rope
pipeline (patchifier grid -> get_pixel_coords(causal) -> wrapper bf16 cast +
/frame_rate -> precompute_freqs_cis), for both arm geometries and both
rope_type variants + both freq-grid precisions, reporting which reference
variant the Mojo table matches and how closely.
"""
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/home/alex/musubi-tuner/src")
from musubi_tuner.ltx_2.components.patchifiers import (  # noqa: E402
    VideoLatentPatchifier, get_pixel_coords,
)
from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape  # noqa: E402
from musubi_tuner.ltx_2.model.transformer.rope import (  # noqa: E402
    precompute_freqs_cis, generate_freq_grid_np, generate_freq_grid_pytorch, LTXRopeType,
)

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/mojodiffusion/output/ltx2_parity_fwd/mojo_rope.safetensors"

mojo = {}
with safe_open(DUMP, framework="pt") as f:
    for k in f.keys():
        mojo[k] = f.get_tensor(k).float()

pat = VideoLatentPatchifier(patch_size=1)

def ref_tables(F, H, W, rope_type, grid_gen, cast_bf16=True):
    shape = VideoLatentShape(batch=1, channels=128, frames=F, height=H, width=W)
    latent_coords = pat.get_patch_grid_bounds(output_shape=shape, device=torch.device("cpu"))
    pos = get_pixel_coords(latent_coords, scale_factors=SpatioTemporalScaleFactors.default(),
                           causal_fix=True)
    if cast_bf16:
        pos = pos.to(torch.bfloat16)  # wrapper: .to(video_latents.dtype)
    pos = pos.clone()
    pos[:, 0, ...] = pos[:, 0, ...] / 25.0
    cos, sin = precompute_freqs_cis(
        pos.float() if not cast_bf16 else pos,  # model consumes as passed
        dim=4096, out_dtype=torch.float32, theta=10000.0,
        max_pos=[20, 2048, 2048], use_middle_indices_grid=True,
        num_attention_heads=32, rope_type=rope_type,
        freq_grid_generator=grid_gen,
    )
    return cos, sin

def flatten_split(cos):  # (B,H,T,D2) -> [T*H, D2] token-major, head-minor
    b, h, t, d2 = cos.shape
    return cos.permute(0, 2, 1, 3).reshape(t * h, d2)

for arm, geo in (("video", (4, 9, 16)), ("image", (1, 16, 16))):
    mc, ms = mojo[f"{arm}_cos"], mojo[f"{arm}_sin"]
    print(f"== {arm} {geo}: mojo table {tuple(mc.shape)}")
    for rt_name, rt in (("SPLIT", LTXRopeType.SPLIT), ("INTERLEAVED", LTXRopeType.INTERLEAVED)):
        for gg_name, gg in (("np64", generate_freq_grid_np), ("pt32", generate_freq_grid_pytorch)):
            for cast in (True, False):
                try:
                    cos, sin = ref_tables(*geo, rt, gg, cast_bf16=cast)
                    if rt == LTXRopeType.SPLIT:
                        rc, rs = flatten_split(cos), flatten_split(sin)
                    else:
                        # interleaved: (B,T,D)? — compare flattened per-token
                        rc = cos.reshape(cos.shape[1] if cos.dim() > 2 else cos.shape[0], -1)
                        rs = sin.reshape(sin.shape[1] if sin.dim() > 2 else sin.shape[0], -1)
                    if rc.shape != mc.shape:
                        print(f"  {rt_name}/{gg_name}/bf16cast={cast}: shape {tuple(rc.shape)} != mojo — skip")
                        continue
                    dc = (rc - mc).abs().max().item()
                    ds = (rs - ms).abs().max().item()
                    cosim = torch.nn.functional.cosine_similarity(
                        rc.flatten(), mc.flatten(), dim=0).item()
                    print(f"  {rt_name}/{gg_name}/bf16cast={cast}: max|dcos|={dc:.3e} "
                          f"max|dsin|={ds:.3e} cos-sim={cosim:.8f}")
                except Exception as e:
                    print(f"  {rt_name}/{gg_name}/bf16cast={cast}: ERROR {type(e).__name__}: {e}")
