"""Bounded MiniMax-H3 LoRA training contracts and trainer seams.

Device block math and the streamed 50-block reverse loop live with the H3 model
modules. This package owns policy, cache/schedule/loss contracts, exact LoRA
inventory/layout mapping, F32 private state, and the first optimizer bridge.

This is not a runnable released trainer: current local H3 transformer artifacts
are ConvRot INT8 and are rejected by the BF16 backward seam. Device dual-shift
noising/native targets, independent AV loss roots, and the ordinary final-head
backward seam are present; chunked final-head backward and end-to-end
released-checkpoint gates are still required.
"""
