#!/usr/bin/env python3
# minimax_h3_train_packing_oracle.py — dump the TRAINING packed layout from the
# pinned Musubi fork (kohya musubi-tuner, branch h3-temporal-stretch @ 8191ec1)
# for the AV training geometries, as bitcast-integer text so the Mojo twin dump
# can be compared byte-for-byte (no float formatting ambiguity).
#
# Cases: av_t2va (37x16x28 + 207 audio) and av_fl2va (+ first/last conditions).
# The one-frame IMAGE layout is deliberately NOT gated here: this branch packs
# 2 silence-audio rows into image sequences (audio_present=0 context) while the
# akane/torchref lineage — which the Mojo image arm is gated against — packs
# [text | video] only. Recorded as a lineage divergence, not a defect.
#
# Timesteps are compared in f32 (the Mojo builder takes Float32 model times);
# positions are compared in f64 (both sides compute f64 grids).
import os
import struct
import sys

sys.path.insert(0, "/home/alex/musubi-tuner/src")

import numpy as np
import torch

from musubi_tuner.minimax_h3.packing import (
    build_h3_layout, build_position_grid, build_timestep_rows,
)

OUT = "/home/alex/mojodiffusion/output/checks/h3_train_packing"
os.makedirs(OUT, exist_ok=True)

U = 0.4375
import numpy as _np
def _shift_f32(u, s):
    # stepwise f32, mirroring h3_shift_sigma / torch f32 tensor ops
    u = _np.float32(u); s = _np.float32(s)
    return _np.float32(s * u) / _np.float32(_np.float32(1.0) + _np.float32(s - _np.float32(1.0)) * u)
T_V = _np.float32(_np.float32(1.0) - _shift_f32(U, 12.0))
T_A = _np.float32(_np.float32(1.0) - _shift_f32(U, 3.0))


def b64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def b32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def dump(f, name: str, task: str, conditions):
    text_len = 87
    layout = build_h3_layout(
        task=task,
        text_length=text_len,
        target_video=(37, 16, 28),
        target_audio_frames=207,
        visual_conditions=conditions,
    )
    pos = build_position_grid(layout).numpy()
    tags = torch.ones(1, text_len, dtype=torch.long)
    ts = build_timestep_rows(
        layout, text_token_tags=tags,
        model_t_video=np.float32(T_V), model_t_audio=np.float32(T_A),
    )
    f.write(f"case {name}\n")
    f.write(f"S {layout.row_count}\n")
    for seg in layout.segments:
        f.write(f"seg {seg.role} {seg.start} {seg.stop}\n")
    f.write("pos\n")
    for v in pos.reshape(-1):
        f.write(f"{b64(v)}\n")
    f.write("rowts\n")
    for v in ts.row_timesteps.numpy().reshape(-1):
        f.write(f"{b32(v)}\n")
    f.write("tags\n")
    for v in ts.token_tags.numpy().reshape(-1):
        f.write(f"{int(v)}\n")
    f.write("adaln\n")
    for v in ts.block_adaln_indices.numpy().reshape(-1):
        f.write(f"{int(v)}\n")
    f.write("endcase\n")
    print(name, "S", layout.row_count,
          "segments", [(s.role, s.start, s.stop) for s in layout.segments])


with open(f"{OUT}/musubi_dump.txt", "w") as f:
    f.write(f"tv {b32(T_V)}\n")
    f.write(f"ta {b32(T_A)}\n")
    dump(f, "av_t2va", "t2va", [])
    dump(f, "av_fl2va", "fl2va", [(1, 16, 28), (1, 16, 28)])
print("wrote", f"{OUT}/musubi_dump.txt")
