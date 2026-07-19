#!/usr/bin/env python3
"""LTX-2 IC-LoRA / V2V block-0 FORWARD+BACKWARD parity oracle (P5 unit 1).

torch.autograd reference for the Mojo IC-LoRA / video-to-video (v2v) geometry.
It reproduces musubi-tuner's v2v forward MECHANICS EXACTLY, layered on top of
the SAME block math the gated video block oracle uses
(scripts/ltx2_video_block_bwd_oracle.py -> imported as `vid`, whose helper math
is scripts/ltx2_av_block_bwd_oracle.py -> `av`).  main() of both is guarded so
importing them is side-effect free.

WHAT "v2v" ADDS over the t2v block gate (all mirrored line-for-line from
/home/alex/musubi-tuner/src/musubi_tuner/ltx2_train_network.py, the
`ref_latents is not None` branch :3345-3466):

  (1) CONCAT — reference tokens are PREPENDED to the target tokens on the seq
      axis IN THE MODEL FORWARD (:3354-3356 combined = cat([ref, target], dim=1)).
      At the block level the tokens are post-patch-embed hidden states, so the
      concat is simply cat([ref_hs, tgt_hs], dim=1); the raw latent->patchify->
      embed is the head's job (gated separately).  ref_seq_len / target_seq_len
      exactly as :3359-3360.

  (2) TWO-GRID POSITIONS (:3386-3424).  ref grid coords and target grid coords
      are each built (get_patch_grid_bounds), mapped to pixel space with
      causal_fix (get_pixel_coords), temporal axis /frame_rate (:3401,:3422).
      The ref H/W coords are then MULTIPLIED by reference_downscale_factor
      (:3402-3405) so the ref and target grids CO-LOCATE from origin 0 (target
      does NOT continue after ref; no offset; no negative positions).  The two
      grids are concatenated on the token axis (:3424 cat dim=2) and one rope
      table is built over the combined sequence.

  (3) CONDITIONING MASK + PER-TOKEN TIMESTEPS (:3367-3377).  ref tokens get
      conditioning-mask True, target tokens False (+ optional first-frame block
      set True when first-frame conditioning fired, :3370-3373).  The per-token
      timestep vector is torch.where(mask, 0, sigma) (:3376-3377): ref/clean
      tokens get LITERAL 0, target tokens get the flow sigma.

  (4) SLICE + TARGET-ONLY MASKED LOSS (:3452-3459 + ltx2_train.py:178-225).
      pred_tokens[:, ref_seq_len:] slices the ref off; target_velocity =
      patchify(noise - latents); target_loss_mask = ~target_conditioning_mask;
      reduction is _masked_mse == sum(per_elem*mask)/sum(mask) (target-token
      normalized).

THE BOUNDARY (honest scope of unit 1): this is a BLOCK-level geometry gate.  The
per-token timestep -> 9*D AdaLN embedding uses the head's time-embed MLP whose
weights are NOT in block-0; the gated video oracle already ABSTRACTS that by
feeding a synthetic `v_timestep` [1,S,9*D].  This oracle keeps that contract:
it builds combined_v_timestep = cat([ref_v_timestep, tgt_v_timestep]) from two
distinct synthetic blocks (so ref-vs-target AdaLN genuinely differ, reflecting
t=0 vs t=sigma) and DUMPS it, so the Mojo side gates against byte-identical
input.  The literal sigma->embedding map is validated by the separate head/
time-embed gate, and the where(mask,0,sigma) VECTOR is dumped here as the
faithful geometry artifact.

DEVIATION NOTES (see report):
  * rope table for the combined sequence uses av.compute_rope with the TARGET
    grid extent as max_positions ([20, 32*NH_tgt, 32*NW_tgt]).  musubi's real
    model normalizes by the FIXED config max_pos [20,2048,2048]
    (model.py:174); the block-gate convention (av.compute_rope) derives it from
    the grid extent.  The Mojo v2v block CONSUMES this dumped cos/sin
    (ltx2_video_bwd_v2v_parity.mojo:_load_rope), so the gate is self-consistent;
    faithfulness of the POSITIONS (which are reproduced from musubi exactly and
    dumped) is what matters.
  * temporal causal offset: av.build_video_coords uses causal_offset=0
    (max(fs-8,0)/fr) whereas musubi get_pixel_coords uses +1 (max(fs-7,0)/fr).
    This is INHERITED from the already-gated block math (Mojo spine agrees with
    av at cos 0.9999943), not introduced here.

Non-degenerate data: sinusoidal/random synth fills, LoRA A AND B nonzero (v2v
10-slot surface incl. the two FFN linears), real head count (32x128).

Dump -> output/ltx2_ic_v2v/ic_v2v_block0_ref.safetensors (all F32).

Run (defer to a GPU/compute window granted by the lead -- loads block-0 from
the 22B dequant-bf16 export):
  /home/alex/serenityflow-v2/.venv/bin/python \
      serenitymojo/models/ltx2/parity/ltx2_ic_v2v_oracle.py
  # variants:  --first-frame  (exercise first-frame conditioning mask)
  #            --self         (forward determinism check, no dump)
"""

import math
import os
import sys

import torch
from safetensors.torch import save_file

# reuse the gated block oracles (main() guarded -> import is side-effect free)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/alex/mojodiffusion/scripts")
import ltx2_av_block_bwd_oracle as av      # noqa: E402  (shared helper math)
import ltx2_video_block_bwd_oracle as vid  # noqa: E402  (video block forward + v2v LoRA)

OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_ic_v2v"

DEV = av.DEV  # "cpu"

# ── v2v geometry (documented; both grids from origin; DS = spatial downscale) ──
# Two grid presets, selected by env LTX2_V2V_GRID or --grid <name>:
#   image512 (DEFAULT): target 1x16x16 (256 tok, the image512-class anchor every
#     campaign gate uses) + ref 1x8x8 (64 tok) at DS=2 -> combined S=320.
#   video: target 4x4x4 (64) + ref 4x2x2 (16) at DS=2 -> S=80 (video-geometry
#     exercise; kept for the temporal-axis coverage).
def _select_grid():
    name = os.environ.get("LTX2_V2V_GRID", "image512")
    for i, a in enumerate(sys.argv):
        if a == "--grid" and i + 1 < len(sys.argv):
            name = sys.argv[i + 1]
    if name == "video":
        return name, (4, 4, 4), (4, 2, 2), 2
    if name == "video512":
        # matches the trainer video_v2v arm: target 4x9x16 (576) + single-image
        # ref 1x4x8 (32, DS=2) -> S_COMB=608. NF>1 -> first-frame is live here.
        return name, (4, 9, 16), (1, 4, 8), 2
    if name == "image512":
        return name, (1, 16, 16), (1, 8, 8), 2
    raise SystemExit(f"unknown --grid {name} (image512|video512|video)")


GRID_NAME, (TGT_NF, TGT_NH, TGT_NW), (REF_NF, REF_NH, REF_NW), DS = _select_grid()
SIGMA = 0.7                                 # flow timestep on the target tokens

S_REF = REF_NF * REF_NH * REF_NW
S_TGT = TGT_NF * TGT_NH * TGT_NW
S_COMB = S_REF + S_TGT

# --first-frame gets its OWN dump file so it doesn't overwrite the default
# (matters for the video grid, where first-frame is active; image512 is a no-op).
FIRST_FRAME = "--first-frame" in sys.argv
# --spine-rope builds cos/sin from musubi's OWN precompute_freqs_cis over the
# two-grid coords (the TRAINER operating point) instead of av.compute_rope — a
# distinct dump so variant-B (Mojo _build_v2v_rope + block) gates against it.
SPINE_ROPE = "--spine-rope" in sys.argv
OUT = os.path.join(
    OUT_DIR,
    "ic_v2v_block0_ref_" + GRID_NAME
    + ("_ff" if FIRST_FRAME else "") + ("_spine" if SPINE_ROPE else "")
    + ".safetensors")

INNER_DIM = av.INNER_DIM
NUM_HEADS = av.NUM_HEADS
HEAD_DIM = av.HEAD_DIM
N_TXT = av.N_TXT
SEED = av.SEED


# ---------------------------------------------------------------------------
# musubi v2v position construction (ltx2_train_network.py:3386-3424)
# ---------------------------------------------------------------------------
def build_two_grid_coords(device):
    """Reference + target latent grids -> combined [1,3,S_comb,2] pixel coords.

    av.build_video_coords == get_patch_grid_bounds + get_pixel_coords(causal_fix)
    + temporal /frame_rate baked in (axis 0).  We then apply the v2v H/W *DS on
    the ref grid so ref & target co-locate from origin, and cat on the token
    axis.  (musubi ref_positions[:,1/2,...] *= reference_downscale_factor
    :3404-3405 ; combined = cat([ref, tgt], dim=2) :3424).
    """
    ref = av.build_video_coords(
        REF_NF, REF_NH, REF_NW, av.VAE_SF, av.CAUSAL_OFFSET, av.FRAME_RATE, device)
    if DS != 1:
        ref = ref.clone()
        ref[:, 1, ...] = ref[:, 1, ...] * float(DS)   # height  (:3404)
        ref[:, 2, ...] = ref[:, 2, ...] * float(DS)   # width   (:3405)
    tgt = av.build_video_coords(
        TGT_NF, TGT_NH, TGT_NW, av.VAE_SF, av.CAUSAL_OFFSET, av.FRAME_RATE, device)
    combined = torch.cat([ref, tgt], dim=2)           # [1,3,S_comb,2]
    return ref, tgt, combined


def build_conditioning(first_frame, device):
    """conditioning_mask + per-token timestep vector (ltx2_train_network.py
    :3367-3377).  ref tokens True, target tokens False (+ first-frame block when
    conditioning fired); timesteps = where(mask, 0, sigma)."""
    # musubi guards first-frame conditioning to num_frames>1 (ltx2_train_network
    # .py:3335): a single-frame target has no subsequent frames to condition
    # from, so --first-frame is a no-op for the image512 grid (TGT_NF==1).
    ff_active = first_frame and TGT_NF > 1
    ref_mask = torch.ones((1, S_REF), dtype=torch.bool, device=device)     # :3367
    tgt_mask = torch.zeros((1, S_TGT), dtype=torch.bool, device=device)    # :3369
    if ff_active:
        # first_frame_tokens = tgt_height * tgt_width (:3371-3373)
        ff = TGT_NH * TGT_NW
        tgt_mask[:, :ff] = True
    cond_mask = torch.cat([ref_mask, tgt_mask], dim=1)                     # :3374
    sigma_vec = torch.full((1, S_COMB), float(SIGMA),
                           dtype=torch.float32, device=device)            # sigma.expand :3376
    timesteps = torch.where(cond_mask, torch.zeros_like(sigma_vec), sigma_vec)  # :3377
    return ref_mask, tgt_mask, cond_mask, timesteps


def build_v2v_inputs(device):
    """Combined block inputs for the ref-prepended sequence.  hs/v_timestep are
    built as ref-segment ++ target-segment (distinct seeds) so the two segments
    genuinely differ (t=0 ref vs t=sigma target at the AdaLN level)."""
    inp = {}
    ref_hs = av.synth((1, S_REF, INNER_DIM), SEED + 700, device)
    tgt_hs = av.synth((1, S_TGT, INNER_DIM), SEED + 701, device)
    inp["hs"] = torch.cat([ref_hs, tgt_hs], dim=1)                        # ref PREPENDED
    inp["enc_hs"] = av.synth((1, N_TXT, INNER_DIM), SEED + 2, device)     # text ctx (:3431)
    ref_vts = av.synth((1, S_REF, 9 * INNER_DIM), SEED + 704, device)     # ref AdaLN (t=0 class)
    tgt_vts = av.synth((1, S_TGT, 9 * INNER_DIM), SEED + 705, device)     # tgt AdaLN (t=sigma class)
    inp["v_timestep"] = torch.cat([ref_vts, tgt_vts], dim=1)
    inp["video_prompt_ts"] = av.synth((1, N_TXT, 2 * INNER_DIM), SEED + 10, device)

    _, _, coords = build_two_grid_coords(device)
    if SPINE_ROPE:
        # TRAINER operating point: musubi's OWN precompute_freqs_cis over the
        # two-grid coords built the musubi way (get_pixel_coords per grid, ref
        # H/W ×DS, temporal /fr, ref-prepend concat). Matches _build_v2v_rope.
        v_cos, v_sin = compute_spine_rope(device)
    else:
        # block-gate-family convention (av.compute_rope, grid-extent max_pos).
        v_cos, v_sin = av.compute_rope(
            coords, INNER_DIM,
            [av.POS_EMBED_MAX_POS, av.VAE_SF[1] * TGT_NH, av.VAE_SF[2] * TGT_NW],
            av.ROPE_THETA, NUM_HEADS, device)
    inp["v_cos"], inp["v_sin"] = v_cos, v_sin
    inp["_coords"] = coords
    return inp


def compute_spine_rope(device):
    """Combined cos/sin from musubi's OWN rope runtime (precompute_freqs_cis),
    reference-native from coords through cos/sin — the trainer operating point.
    Mirrors ltx2_train_network.py:3386-3424 for the two-grid positions, then the
    model's SPLIT rope with fixed max_pos [20,2048,2048] over bf16 positions
    (validated vs _build_video_rope at cos 0.99998, scripts/ltx2_rope_table_compare.py).
    Returns [H, P, hrd] cos/sin (same layout as av.compute_rope; run_video_block's
    apply_rope == musubi apply_split_rotary_emb)."""
    import sys as _sys
    _sys.path.insert(0, "/home/alex/musubi-tuner/src")
    from musubi_tuner.ltx_2.components.patchifiers import (
        VideoLatentPatchifier, get_pixel_coords)
    from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape
    from musubi_tuner.ltx_2.model.transformer.rope import (
        precompute_freqs_cis, generate_freq_grid_np, LTXRopeType)

    pat = VideoLatentPatchifier(patch_size=1)
    fr = float(av.FRAME_RATE)

    def _pos(nf, nh, nw, scale_hw):
        shape = VideoLatentShape(batch=1, channels=INNER_DIM // 32, frames=nf, height=nh, width=nw)
        coords = pat.get_patch_grid_bounds(output_shape=shape, device=torch.device("cpu"))
        pos = get_pixel_coords(coords, scale_factors=SpatioTemporalScaleFactors.default(),
                               causal_fix=True).to(torch.bfloat16)   # network_dtype (:3400)
        pos = pos.clone()
        pos[:, 0, ...] = pos[:, 0, ...] / fr                          # temporal /frame_rate (:3401)
        if scale_hw != 1:
            pos[:, 1, ...] = pos[:, 1, ...] * scale_hw                # ref H ×DS (:3404)
            pos[:, 2, ...] = pos[:, 2, ...] * scale_hw                # ref W ×DS (:3405)
        return pos

    ref_pos = _pos(REF_NF, REF_NH, REF_NW, DS)
    tgt_pos = _pos(TGT_NF, TGT_NH, TGT_NW, 1)
    combined = torch.cat([ref_pos, tgt_pos], dim=2)                  # ref-prepend concat (:3424)
    cos, sin = precompute_freqs_cis(
        combined, dim=INNER_DIM, out_dtype=torch.float32, theta=float(av.ROPE_THETA),
        max_pos=[20, 2048, 2048], use_middle_indices_grid=True,
        num_attention_heads=NUM_HEADS, rope_type=LTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np)
    return cos[0].to(torch.float32).to(device), sin[0].to(torch.float32).to(device)   # [H,P,hrd]


def _masked_mse(pred, tgt, mask, *, weighting=None, dtype=torch.float32,
                loss_type="mse", huber_delta=1.0):
    """VERBATIM from musubi ltx2_train.py:178-225 (mse branch)."""
    if isinstance(tgt, torch.Tensor):
        pred = pred.to(device=tgt.device, dtype=dtype)
    else:
        pred = pred.to(dtype=dtype)
    if loss_type in ("mae", "l1"):
        per_elem = torch.nn.functional.l1_loss(pred, tgt, reduction="none")
    elif loss_type in ("huber", "smooth_l1"):
        per_elem = torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="none", beta=huber_delta)
    else:
        per_elem = torch.nn.functional.mse_loss(pred, tgt, reduction="none")
    if weighting is not None:
        w = weighting
        if isinstance(w, torch.Tensor) and w.dim() != per_elem.dim():
            while w.dim() > per_elem.dim() and w.shape[-1] == 1:
                w = w.squeeze(-1)
        per_elem = per_elem * w
    if mask is None:
        return per_elem.mean()
    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)
    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() == 0:
        return per_elem.mean()
    return (per_elem * mask_f).div(denom).mean()


def main():
    first_frame = "--first-frame" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[ic-v2v-oracle] loading block-0 from {os.path.basename(av.CKPT)} "
          f"(first_frame={first_frame})")
    w = av.load_block0(av.CKPT)
    print(f"[ic-v2v-oracle] {len(w)} block-0 tensors (bf16-roundtripped F32)")
    print(f"[ic-v2v-oracle] grid={GRID_NAME}: ref {REF_NF}x{REF_NH}x{REF_NW}={S_REF}  "
          f"tgt {TGT_NF}x{TGT_NH}x{TGT_NW}={S_TGT}  DS={DS}  sigma={SIGMA}  S_comb={S_COMB}")

    inp = build_v2v_inputs(DEV)
    lora = vid.build_video_lora(w, DEV, v2v=True)   # 10 slots (attn1/2 x4 + 2 FFN)
    keys = vid.lora_key_list(v2v=True)
    print(f"[ic-v2v-oracle] S_comb={S_COMB} N_TXT={N_TXT} rank={av.LORA_RANK} "
          f"scale={av.LORA_SCALE} adapters={len(lora)} (v2v)")

    ref_mask, tgt_mask, cond_mask, timesteps = build_conditioning(first_frame, DEV)
    target_loss_mask = ~tgt_mask                                    # :3454

    hs_leaf = inp["hs"].clone().requires_grad_(True)
    fwd_inp = dict(inp)
    fwd_inp["hs"] = hs_leaf

    # forward through the SAME video block math the gated oracle uses
    block_out = vid.run_video_block(w, fwd_inp, lora)               # [1,S_comb,D]
    print(f"[ic-v2v-oracle] block_out mean={block_out.mean():.5f} std={block_out.std():.5f}")

    # slice the ref off, target-only masked MSE (:3452-3459)
    target_pred = block_out[:, S_REF:, :]                          # [1,S_tgt,D]
    target_velocity = av.synth((1, S_TGT, INNER_DIM), SEED + 800, DEV)  # patchify(noise-latents) proxy
    loss = _masked_mse(target_pred, target_velocity, target_loss_mask)
    print(f"[ic-v2v-oracle] masked loss={loss.item():.6f} "
          f"(target tokens={int(target_loss_mask.sum().item())}/{S_TGT})")

    # block-output cotangent the Mojo block backward consumes: zeros on the ref
    # rows (sliced off -> no grad), d(loss)/d(target_pred) on the target rows.
    d_block_out = torch.autograd.grad(loss, block_out, retain_graph=True)[0]  # [1,S_comb,D]

    # d_hidden + LoRA grads from the real masked loss
    leaves = [hs_leaf]
    names = ["g_d_hidden"]
    for key in keys:
        A, B, _ = lora[key]
        leaves += [A, B]
        names += [f"g_dA.{key}", f"g_dB.{key}"]
    grads = torch.autograd.grad(loss, leaves)

    if "--self" in sys.argv:
        out2 = vid.run_video_block(w, fwd_inp, lora)
        print(f"[self] block cos={av.cos_sim(block_out, out2):.7f}")
        return

    # ── dump ────────────────────────────────────────────────────────────────
    def f32(t):
        return t.detach().to(torch.float32).cpu().contiguous().clone()

    out = {}
    # forward inputs (block-level, ref-prepended combined)
    out["hs"] = f32(inp["hs"])
    out["enc_hs"] = f32(inp["enc_hs"])
    out["v_timestep"] = f32(inp["v_timestep"])
    out["video_prompt_ts"] = f32(inp["video_prompt_ts"])
    out["v_cos"] = f32(inp["v_cos"])
    out["v_sin"] = f32(inp["v_sin"])
    for key in keys:
        A, B, _ = lora[key]
        out[f"lora.{key}.A"] = f32(A)
        out[f"lora.{key}.B"] = f32(B)
    # v2v geometry artifacts
    out["combined_positions"] = f32(inp["_coords"])                # [1,3,S_comb,2]
    out["conditioning_mask"] = cond_mask.to(torch.float32).cpu().contiguous()
    out["target_conditioning_mask"] = tgt_mask.to(torch.float32).cpu().contiguous()
    out["target_loss_mask"] = target_loss_mask.to(torch.float32).cpu().contiguous()
    out["combined_timesteps"] = f32(timesteps)                     # where(mask,0,sigma)
    out["sigma"] = torch.tensor([float(SIGMA)], dtype=torch.float32)
    out["ref_seq_len"] = torch.tensor([S_REF], dtype=torch.int32)
    out["target_seq_len"] = torch.tensor([S_TGT], dtype=torch.int32)
    # outputs
    out["block_out"] = f32(block_out)
    out["target_pred"] = f32(target_pred)
    out["target_velocity"] = f32(target_velocity)
    out["loss"] = torch.tensor([loss.item()], dtype=torch.float32)
    # cotangent + grads
    out["d_block_out"] = f32(d_block_out)
    for name, g in zip(names, grads):
        if float(g.norm()) == 0.0:
            raise RuntimeError(f"degenerate (zero) reference grad: {name}")
        out[name] = f32(g)
    print(f"[ic-v2v-oracle] d_hidden |g|={grads[0].norm():.5f} "
          f"d_block_out |g|={d_block_out.norm():.5f}")

    save_file(out, OUT)
    print(f"[ic-v2v-oracle] dumped {len(out)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
