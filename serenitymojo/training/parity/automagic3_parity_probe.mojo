# automagic3_parity_probe.mojo — parity gate for training/automagic3.mojo vs the
# REAL ai-toolkit Automagic3 (fused=False) oracle.
#
# Oracle: /home/alex/EriTrainer/trainer/parity/automagic3/oracle.safetensors
#   (gen_oracle.py: seed 1234, lr0=1e-4, beta2=0.999, eps=1e-30, H=8, N=14;
#    three params A[16,64] factored, B[64,16] factored other orient, C[32] 1D;
#    common-mode-biased grads so signs persist and the controller's lr RISES.)
# Tensors: init.{A,B,C}, grads.{A,B,C}[N,*shape], final.{A,B,C},
#          lr_traj[N], meta=[N, lr0, beta2, eps, H].
#
# Drives the ACTUAL levers-wired step functions (automagic3_step_2d /
# automagic3_step_1d + Automagic3Ctl): all three params are ONE group (A,B,C
# pooled into a single adaptive lr, exactly as levers_optimizer_step_host pools
# the A+B of every adapter). Per step: ctl.reset_accum() -> step A,B,C ->
# ctl.apply_vote(). The C[32] param exercises the 1D second-moment branch.
# Gates:
#   (a) per-step lr-trajectory rel-diff <= 2% vs lr_traj (the novel controller),
#   (b) final A/B/C cosine >= 0.9999 (the factored/1D update math).
# FAIL-LOUD: any miss raises -> exit != 0. BITROT GUARD: argv "FAIL" compares the
# final params against zeros and MUST fail.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/parity/automagic3_parity_probe.mojo

from sys import argv
from serenitymojo.parity import ParityHarness
from serenitymojo.training.automagic3 import (
    Automagic3State, Automagic3Ctl,
    automagic3_step_2d, automagic3_step_1d, automagic3_clamp_h,
    AUTOMAGIC3_DEFAULT_CLIP, AUTOMAGIC3_DEFAULT_WD,
)
from serenitymojo.io.safetensors import SafeTensors
from std.memory import alloc


comptime ORACLE_PATH = (
    "/home/alex/EriTrainer/trainer/parity/automagic3/oracle.safetensors"
)
comptime LR_REL_TOL = Float64(0.02)      # per-step lr rel-diff bound (the gate)
comptime FINAL_COS_TOL = Float64(0.9999)  # final-param cosine bound


def _read_f32(st: SafeTensors, name: String) raises -> List[Float32]:
    """Read a named f32 tensor from the oracle into a fresh aligned
    List[Float32]. The mmap'd span may not be 4-byte aligned, so copy the bytes
    into an alloc'd (aligned) buffer before bitcasting to Float32."""
    var span = st.tensor_bytes(name)
    var nbytes = len(span)
    if nbytes % 4 != 0:
        raise Error(String("tensor not f32-sized: ") + name)
    var nf = nbytes // 4
    var buf = alloc[UInt8](nbytes)
    for i in range(nbytes):
        buf[i] = span[i]
    var fp = buf.bitcast[Float32]()
    var out = List[Float32](capacity=nf)
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _grad_step(
    flat: List[Float32], step: Int, shape_numel: Int
) -> List[Float32]:
    """Slice grads.X[step] out of the flat [N, *shape] tensor (row-major)."""
    var out = List[Float32](capacity=shape_numel)
    var base = step * shape_numel
    for i in range(shape_numel):
        out.append(flat[base + i])
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def main() raises:
    var sabotage = False
    var av = argv()
    for i in range(len(av)):
        if av[i] == String("FAIL"):
            sabotage = True

    var st = SafeTensors.open(String(ORACLE_PATH))

    # meta = [N, lr0, beta2, eps, H]
    var meta = _read_f32(st, String("meta"))
    var n_steps = Int(meta[0])
    var lr0 = Float64(meta[1])
    var beta2 = Float64(meta[2])
    var eps = Float64(meta[3])
    var h = automagic3_clamp_h(Int(meta[4]))
    print(
        "oracle meta: N=", n_steps, " lr0=", lr0, " beta2=", beta2,
        " eps=", eps, " H=", h,
    )

    # init params (flat row-major) and shapes.
    var pA = _read_f32(st, String("init.A"))  # [16,64]
    var pB = _read_f32(st, String("init.B"))  # [64,16]
    var pC = _read_f32(st, String("init.C"))  # [32]
    var nA = 16 * 64
    var nB = 64 * 16
    var nC = 32
    if len(pA) != nA or len(pB) != nB or len(pC) != nC:
        raise Error("init shape mismatch")

    # all-step grads (flat [N, *shape]).
    var gA = _read_f32(st, String("grads.A"))
    var gB = _read_f32(st, String("grads.B"))
    var gC = _read_f32(st, String("grads.C"))

    # reference lr trajectory + final params.
    var lr_ref = _read_f32(st, String("lr_traj"))
    var fA = _read_f32(st, String("final.A"))
    var fB = _read_f32(st, String("final.B"))
    var fC = _read_f32(st, String("final.C"))

    # ONE shared ctl (the group), three per-param states (A,B factored; C 1D).
    var ctl = Automagic3Ctl()
    ctl.init_lr(lr0)
    var sA = Automagic3State(16, 64, h)   # factored [16,64]
    var sB = Automagic3State(64, 16, h)   # factored [64,16]
    var sC = Automagic3State(32, h)       # 1D length 32
    var clip = AUTOMAGIC3_DEFAULT_CLIP
    var wd = AUTOMAGIC3_DEFAULT_WD

    # Run N steps; capture the shared lr AFTER each step's apply_vote()
    # (matches the oracle's lr_traj.append(get_avg_learning_rate()) post-step).
    var lr_mojo = List[Float64]()
    for step in range(n_steps):
        var gAi = _grad_step(gA, step, nA)
        var gBi = _grad_step(gB, step, nB)
        var gCi = _grad_step(gC, step, nC)
        ctl.reset_accum()
        automagic3_step_2d(pA, gAi, sA, beta2, eps, clip, wd, ctl)
        automagic3_step_2d(pB, gBi, sB, beta2, eps, clip, wd, ctl)
        automagic3_step_1d(pC, gCi, sC, beta2, eps, clip, wd, ctl)
        ctl.apply_vote()
        lr_mojo.append(ctl.lr)

    # ---- Gate (a): lr trajectory, per-step rel-diff <= 2% ----
    print("")
    print("step |        ref lr |       mojo lr |   rel-diff")
    print("-----+---------------+---------------+-----------")
    var lr_pass = True
    for step in range(n_steps):
        var ref_lr = Float64(lr_ref[step])
        var got = lr_mojo[step]
        var denom = ref_lr
        if denom < 0.0:
            denom = -denom
        if denom < 1.0e-30:
            denom = 1.0e-30
        var rel = (got - ref_lr) / denom
        if rel < 0.0:
            rel = -rel
        var ok = rel <= LR_REL_TOL
        if not ok:
            lr_pass = False
        print(
            String(step), "  | ", ref_lr, " | ", got, " | ", rel,
            "  ", "OK" if ok else "MISS",
        )

    # ---- Gate (b): final params cosine >= 0.9999 ----
    var hh = ParityHarness(FINAL_COS_TOL)
    if sabotage:
        fA = _zeros(nA)
        fB = _zeros(nB)
        fC = _zeros(nC)
    var rA = hh.compare_host(pA, fA)
    var rB = hh.compare_host(pB, fB)
    var rC = hh.compare_host(pC, fC)
    print("")
    print("final.A:", rA)
    print("final.B:", rB)
    print("final.C:", rC)

    var final_pass = rA.passed and rB.passed and rC.passed
    var all_pass = lr_pass and final_pass

    print("")
    print("lr-traj gate (rel<=2%):", "PASS" if lr_pass else "FAIL")
    print("final-param gate (cos>=0.9999):", "PASS" if final_pass else "FAIL")
    if all_pass:
        print("AUTOMAGIC3 PARITY PASSED")
    else:
        print("AUTOMAGIC3 PARITY FAILURE")
        raise Error("automagic3_parity_probe gate failed")
