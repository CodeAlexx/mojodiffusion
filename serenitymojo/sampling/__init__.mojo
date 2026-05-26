# serenitymojo.sampling — diffusion scheduler and sampler glue.
#
# Phase A/B foundation. Schedulers own the noise->latent schedule + per-step
# update around a DiT (the DiT predicts velocity; the scheduler does the rest).
#
# Modules:
#   flow_match  — Z-Image rectified-flow Euler scheduler and Qwen-Image helpers.
#   sdxl_euler  — SDXL EulerDiscreteScheduler scalar setup plus GPU CFG/update.
#   flux2_klein — FLUX.2/Klein dynamic-mu/fixed-shift schedules plus GPU
#                 CFG/update helpers.
