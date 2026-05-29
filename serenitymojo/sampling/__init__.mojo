# serenitymojo.sampling — diffusion scheduler and sampler glue.
#
# Phase A/B foundation. Schedulers own the noise->latent schedule + per-step
# update around a DiT (the DiT predicts velocity; the scheduler does the rest).
#
# Modules:
#   flow_match  — Z-Image rectified-flow Euler scheduler and Qwen-Image helpers.
#   sdxl_euler  — SDXL EulerDiscreteScheduler scalar setup plus GPU CFG/update.
#   sd15_euler  — SD1.5 EulerDiscreteScheduler wrapper around the same schedule.
#   flux2_klein — FLUX.2/Klein dynamic-mu/fixed-shift schedules plus GPU
#                 CFG/update helpers.
#   flux1_dev   — FLUX.1-dev BFL time-shift schedule and packed-latent plan.
#   sd3_flow_match — SD3 shifted-flow schedule plus textbook CFG/update helpers.
#   lens_flowmatch — Microsoft Lens N-sigma dynamic FlowMatch scalar schedule.
#   lance_t2v   — Lance shifted-flow schedule, CFG, renorm, and Euler update.
#   ernie_sampling — ERNIE fixed-shift FlowMatch schedule plus CFG/update helpers.
#   anima_sampling — Anima linear FlowMatch schedule plus CFG/update helpers.
