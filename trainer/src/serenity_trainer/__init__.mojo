# serenity-trainer — pure-Mojo port of Serenity's training core.
#
# Port SPEC = Serenity Python (/home/alex/Serenity). The mojodiffusion
# serenitymojo/training and serenitymojo/models trainers are NOT a source of
# truth (untested) and are NOT reused. The ONLY reuse from mojodiffusion is the
# numerical foundation: serenitymojo/{autograd, tensor, ops}.
#
# Dtype policy: BF16 in/out (storage), F32 compute (accumulate) — per Serenity
# (AdamW state = zeros_like(p) = bf16 + stochastic rounding; no F32 master).
# Pure Mojo, no Python runtime. No MGDS / data pipeline.

comptime VERSION = "0.0.1"
