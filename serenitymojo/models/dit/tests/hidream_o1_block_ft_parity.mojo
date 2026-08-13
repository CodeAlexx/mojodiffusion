# models/dit/tests/hidream_o1_block_ft_parity.mojo — full-FT rollout P1 gate:
# hidream_o1_block_ft_backward_dev (models/dit/hidream_o1_block_ft.mojo) vs the
# torch-autograd BASE-ONLY oracle at the REAL head config.
#
# Oracle: models/dit/parity/hidream_o1_block_oracle.py in HIDREAM_FT_ORACLE=1
# mode -> /tmp/hidream_block_oracle_ft.safetensors (~1.6GB, OUT OF GIT). The
# LoRA-mode fixture is NOT valid for FT (the adapters shift the dX chain —
# krea2 trap) and carries no W.grad refs. Synthetic NON-DEGENERATE weights (no
# hidream checkpoint on this box — rollout-plan-sanctioned).
#
# Dims: REAL head config D=4096 H=32 HKV=8 Dh=128 F=12288 (hidream_o1.mojo
# config), small seq S=64 ar_len=16 for speed. F32 end-to-end (the established
# hidream block-gate dtype).
#
# Forward for the FT backward: hidream_o1_block_lora_forward with ALL-None
# adapters IS the base forward (no LoRA anywhere) — its saved tape is the base
# tape.
#
# Acceptance: out + d_hidden + ALL 7 dW cosine >= 0.999 (ParityHarness) +
# the 4 v2 full-surface rms-scale d_g arms (in_ln/q_norm/k_norm/post_ln,
# FULL_SURFACE_PLAN_2026-07-08 Phase B row 2) cosine >= 0.9999 vs the
# oracle's grad_in_ln/grad_q_norm/grad_k_norm/grad_post_ln refs.
#
# Run (oracle FIRST, as SEPARATE commands — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   HIDREAM_FT_ORACLE=1 /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/dit/parity/hidream_o1_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#       -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#       -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker $PWD/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#       serenitymojo/models/dit/tests/hidream_o1_block_ft_parity.mojo \
#       -o /tmp/hidream_block_ft_par
#   LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/hidream_block_ft_par

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.hidream_o1_train_block import (
    HiDreamO1BlockWeights,
    HiDreamO1BlockLora,
    hidream_o1_block_lora_forward,
)
from serenitymojo.models.dit.hidream_o1_block_ft import (
    HD_FT_SLOT_Q,
    HD_FT_SLOT_K,
    HD_FT_SLOT_V,
    HD_FT_SLOT_O,
    HD_FT_SLOT_GATE,
    HD_FT_SLOT_UP,
    HD_FT_SLOT_DOWN,
    HD_FT_SLOT_IN_LN,
    HD_FT_SLOT_Q_NORM,
    HD_FT_SLOT_K_NORM,
    HD_FT_SLOT_POST_LN,
    hidream_o1_block_ft_backward_dev,
)

comptime TArc = ArcPointer[Tensor]
# REAL hidream-o1 head config (hidream_o1.mojo: hidden 4096, GQA 32/8,
# head_dim 128, SwiGLU 12288); small seq for gate speed.
comptime D = 4096
comptime H = 32
comptime HKV = 8
comptime Dh = 128
comptime F = 12288
comptime S = 64
comptime EPS = Float32(1.0e-6)
comptime BAR = 0.999
comptime BAR_V2 = 0.9999   # the 4 new 1D rms-scale arms (Phase B campaign bar)
comptime ORACLE = "/tmp/hidream_block_oracle_ft.safetensors"


def _t(fx: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(fx.tensor_view(name), ctx)


def _ta(fx: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view(fx.tensor_view(name), ctx))


def _replicate_heads_host(
    half_tab: List[Float32], heads: Int
) raises -> List[Float32]:
    """[S, half] per-position -> [S*heads*half] rows (row = s*heads + h),
    matching x flattened [1,S,h,Dh] row order (same as the LoRA gate)."""
    comptime half = Dh // 2
    var out = List[Float32]()
    for s in range(S):
        for _h in range(heads):
            for c in range(half):
                out.append(half_tab[s * half + c])
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("=== HiDream-O1 BLOCK FULL-FT torch-oracle parity (cos >= 0.999) ===")
    print("D=", D, " H=", H, " HKV=", HKV, " Dh=", Dh, " F=", F, " S=", S,
          " (base-only oracle: HIDREAM_FT_ORACLE=1)")

    var fx = ShardedSafeTensors.open(String(ORACLE))

    var w = HiDreamO1BlockWeights(
        _ta(fx, "w_in_ln", ctx), _ta(fx, "w_qw", ctx), _ta(fx, "w_kw", ctx),
        _ta(fx, "w_vw", ctx), _ta(fx, "w_q_norm", ctx), _ta(fx, "w_k_norm", ctx),
        _ta(fx, "w_ow", ctx), _ta(fx, "w_post_ln", ctx), _ta(fx, "w_gw", ctx),
        _ta(fx, "w_uw", ctx), _ta(fx, "w_dw", ctx),
    )

    # ALL-None adapters: the LoRA forward with no adapters IS the base forward.
    var lora = HiDreamO1BlockLora()

    var cos_h_host = _t(fx, "cos_half", ctx).to_host(ctx)
    var sin_h_host = _t(fx, "sin_half", ctx).to_host(ctx)
    comptime half = Dh // 2
    var cq: List[Int] = [S * H * half]
    var ck: List[Int] = [S * HKV * half]
    var cos_q = Tensor.from_host(_replicate_heads_host(cos_h_host, H), cq.copy(), STDtype.F32, ctx)
    var sin_q = Tensor.from_host(_replicate_heads_host(sin_h_host, H), cq^, STDtype.F32, ctx)
    var cos_k = Tensor.from_host(_replicate_heads_host(cos_h_host, HKV), ck.copy(), STDtype.F32, ctx)
    var sin_k = Tensor.from_host(_replicate_heads_host(sin_h_host, HKV), ck^, STDtype.F32, ctx)

    var mask_hss = _t(fx, "mask_hss", ctx)                       # [H*S, S] F32
    var mask_h = mask_hss.to_host(ctx)
    var m4_sh: List[Int] = [1, H, S, S]
    var mask4 = Tensor.from_host(mask_h.copy(), m4_sh^, STDtype.F32, ctx)
    var mhs_sh: List[Int] = [H * S, S]
    var mask_f32 = Tensor.from_host(mask_h^, mhs_sh^, STDtype.F32, ctx)

    var hidden = _ta(fx, "hidden", ctx)
    var d_out = _t(fx, "d_out", ctx)

    var h = ParityHarness(BAR)
    var n_fail = 0

    print("[fwd] hidream_o1_block_lora_forward (all-None adapters = base) ...")
    var fwd = hidream_o1_block_lora_forward[S, H, HKV, Dh](
        hidden, w, lora, cos_q, sin_q, cos_k, sin_k, mask4,
        D, F, EPS, ctx,
    )
    ctx.synchronize()
    var r_out = h.compare(fwd.out[], _t(fx, "out", ctx).to_host(ctx), ctx)
    print("  out            ", r_out)
    if not r_out.passed:
        n_fail += 1

    print("[bwd] hidream_o1_block_ft_backward_dev ...")
    var g = hidream_o1_block_ft_backward_dev[S, H, HKV, Dh](
        d_out, w, fwd.saved, cos_q, sin_q, cos_k, sin_k, mask_f32,
        D, F, EPS, ctx,
    )
    ctx.synchronize()

    var r_dh = h.compare(g.d_hidden[], _t(fx, "d_hidden", ctx).to_host(ctx), ctx)
    print("  d_hidden       ", r_dh)
    if not r_dh.passed:
        n_fail += 1

    # 7 dW arms vs the torch W.grad refs, FIXED HD_FT_SLOT_* order.
    var ref_names: List[String] = [
        String("grad_qw"), String("grad_kw"), String("grad_vw"),
        String("grad_ow"), String("grad_gw"), String("grad_uw"),
        String("grad_dw"),
    ]
    var slot_idx: List[Int] = [
        HD_FT_SLOT_Q, HD_FT_SLOT_K, HD_FT_SLOT_V, HD_FT_SLOT_O,
        HD_FT_SLOT_GATE, HD_FT_SLOT_UP, HD_FT_SLOT_DOWN,
    ]
    for i in range(len(ref_names)):
        var r_dw = h.compare(
            g.dw[slot_idx[i]][], _t(fx, ref_names[i], ctx).to_host(ctx), ctx
        )
        print("  dW." + ref_names[i] + "  ", r_dw)
        if not r_dw.passed:
            n_fail += 1

    # ── v2 FULL-SURFACE arms: the 4 rms-norm scale d_g (1D, bar 0.9999) ──────
    var h2 = ParityHarness(BAR_V2)
    var v2_names: List[String] = [
        String("grad_in_ln"), String("grad_q_norm"),
        String("grad_k_norm"), String("grad_post_ln"),
    ]
    var v2_idx: List[Int] = [
        HD_FT_SLOT_IN_LN, HD_FT_SLOT_Q_NORM,
        HD_FT_SLOT_K_NORM, HD_FT_SLOT_POST_LN,
    ]
    for i in range(len(v2_names)):
        var r_dg = h2.compare(
            g.dw[v2_idx[i]][], _t(fx, v2_names[i], ctx).to_host(ctx), ctx
        )
        print("  dP." + v2_names[i] + "  ", r_dg)
        if not r_dg.passed:
            n_fail += 1

    print("------------------------------------------------------------")
    if n_fail == 0:
        print("HIDREAM-O1 BLOCK FULL-FT PARITY PASS (9 v1 captures cos >= 0.999")
        print("  + 4 v2 full-surface rms-scale arms cos >= 0.9999)")
    else:
        raise Error(
            String("HIDREAM-O1 BLOCK FULL-FT PARITY FAIL: ")
            + String(n_fail) + " capture(s) below bar"
        )
