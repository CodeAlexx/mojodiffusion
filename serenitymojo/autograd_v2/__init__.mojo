# serenitymojo.autograd_v2 - dependency-counted graph autograd engine (Phase P1
# of docs/AUTOGRAD_V2_MOJO_DESIGN.md). Port pattern: flame-core
# src/autograd_v2/ (node.rs / input_buffer.rs / accumulator.rs / engine.rs).
#
# P1 surface: core types (node/graph/input_buffer) + engine over
# OPK_{LEAF,ADD,MUL,MATMUL,SUM} + toy gates (tests/toy_gates.mojo).
# P2 surface: C15 slot-ordered fan-in (Edge.contrib_slot + slotted
# InputBuffer) + the zimage DiT op kinds OPK_{PROJ_LORA,RMS_NORM_DX,MODULATE,
# ROPE,SDPA,SWIGLU,RESIDUAL_GATE_DXDY,RESHAPE} with record_* wrappers
# (ops_record.mojo) + per-op bit-parity gates (tests/dit_op_parity.mojo).
# Supersedes-but-does-not-delete the T1 tape (serenitymojo/autograd.mojo, C13).
