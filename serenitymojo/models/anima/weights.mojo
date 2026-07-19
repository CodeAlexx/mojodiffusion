# models/anima/weights.mojo — real safetensors -> Anima per-block weight struct.
#
# Loads one Anima MiniTrainDIT block's weights from the real .safetensors into
# AnimaBlockWeights, the struct the (forthcoming) anima block fwd/bwd consumes.
# Mirrors serenitymojo/models/klein/weights.mojo's _load_tensor pattern (device
# tensor, kept in stored BF16 dtype; F32 master copies are made by the optimizer
# layer, not here).
#
# Per-block key layout (block bi), verified against the real checkpoint header
# (2026-06-01) — net.blocks.{bi}.* — 20 tensors, all BF16:
#   adaln_modulation_self_attn.1.weight   [256, 2048]    (AdaLN-LoRA down)
#   adaln_modulation_self_attn.2.weight   [6144, 256]    (AdaLN-LoRA up -> 3*2048)
#   adaln_modulation_cross_attn.1.weight  [256, 2048]
#   adaln_modulation_cross_attn.2.weight  [6144, 256]
#   adaln_modulation_mlp.1.weight         [256, 2048]
#   adaln_modulation_mlp.2.weight         [6144, 256]
#   self_attn.q_proj.weight    [2048, 2048]   self_attn.k_proj.weight  [2048, 2048]
#   self_attn.v_proj.weight    [2048, 2048]   self_attn.output_proj.weight [2048, 2048]
#   self_attn.q_norm.weight    [128]          self_attn.k_norm.weight  [128]
#   cross_attn.q_proj.weight   [2048, 2048]   cross_attn.k_proj.weight [2048, 1024]
#   cross_attn.v_proj.weight   [2048, 1024]   cross_attn.output_proj.weight [2048, 2048]
#   cross_attn.q_norm.weight   [128]          cross_attn.k_norm.weight [128]
#   mlp.layer1.weight          [8192, 2048]   mlp.layer2.weight        [2048, 8192]
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only -> AnimaBlockWeights is
# Movable-only (never store a Tensor in a collection).

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts


comptime TArc = ArcPointer[Tensor]


struct AnimaBlockWeights(Copyable, Movable):
    """All 20 weights of one Anima MiniTrainDIT block, device-resident.

    Fields are TArc (ArcPointer[Tensor]) handles so the struct is Copyable+Movable
    and can live in a List for the resident 28-block stack (mirrors Ernie's
    ErnieBlockWeights). The block fwd/bwd dereference each field once (`w.sa_q[]`)
    to borrow the device Tensor — refcount bump only, no buffer copy.
    """

    # AdaLN-LoRA modulation (down [256,2048], up [6144,256]) x3 sub-blocks
    var sa_mod1: TArc
    var sa_mod2: TArc
    var ca_mod1: TArc
    var ca_mod2: TArc
    var mlp_mod1: TArc
    var mlp_mod2: TArc
    # self-attention
    var sa_q: TArc
    var sa_k: TArc
    var sa_v: TArc
    var sa_out: TArc
    var sa_qn: TArc
    var sa_kn: TArc
    # cross-attention
    var ca_q: TArc
    var ca_k: TArc
    var ca_v: TArc
    var ca_out: TArc
    var ca_qn: TArc
    var ca_kn: TArc
    # GELU MLP
    var mlp1: TArc
    var mlp2: TArc

    def __init__(
        out self,
        var sa_mod1: Tensor, var sa_mod2: Tensor,
        var ca_mod1: Tensor, var ca_mod2: Tensor,
        var mlp_mod1: Tensor, var mlp_mod2: Tensor,
        var sa_q: Tensor, var sa_k: Tensor, var sa_v: Tensor, var sa_out: Tensor,
        var sa_qn: Tensor, var sa_kn: Tensor,
        var ca_q: Tensor, var ca_k: Tensor, var ca_v: Tensor, var ca_out: Tensor,
        var ca_qn: Tensor, var ca_kn: Tensor,
        var mlp1: Tensor, var mlp2: Tensor,
    ):
        self.sa_mod1 = TArc(sa_mod1^); self.sa_mod2 = TArc(sa_mod2^)
        self.ca_mod1 = TArc(ca_mod1^); self.ca_mod2 = TArc(ca_mod2^)
        self.mlp_mod1 = TArc(mlp_mod1^); self.mlp_mod2 = TArc(mlp_mod2^)
        self.sa_q = TArc(sa_q^); self.sa_k = TArc(sa_k^); self.sa_v = TArc(sa_v^); self.sa_out = TArc(sa_out^)
        self.sa_qn = TArc(sa_qn^); self.sa_kn = TArc(sa_kn^)
        self.ca_q = TArc(ca_q^); self.ca_k = TArc(ca_k^); self.ca_v = TArc(ca_v^); self.ca_out = TArc(ca_out^)
        self.ca_qn = TArc(ca_qn^); self.ca_kn = TArc(ca_kn^)
        self.mlp1 = TArc(mlp1^); self.mlp2 = TArc(mlp2^)

# Load a named tensor as a device Tensor in its stored dtype (Anima base = BF16).
# Mirrors klein/weights.mojo::_load_tensor.
def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# Dim 0 of a named tensor's stored shape (for shape verification).
def _dim0(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[0])


def _dim1(st: SafeTensors, name: String) raises -> Int:
    var info = st.tensor_info(name)
    return Int(info.shape[1])


# Load Anima block `block_idx`'s 20 real weights into AnimaBlockWeights.
def load_anima_block_weights(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> AnimaBlockWeights:
    var bp = String("net.blocks.") + String(block_idx) + String(".")
    return AnimaBlockWeights(
        _load_tensor(st, bp + String("adaln_modulation_self_attn.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_self_attn.2.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_cross_attn.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_cross_attn.2.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_mlp.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_mlp.2.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.q_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.k_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.v_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.output_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.q_norm.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.k_norm.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.q_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.k_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.v_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.output_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.q_norm.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.k_norm.weight"), ctx),
        _load_tensor(st, bp + String("mlp.layer1.weight"), ctx),
        _load_tensor(st, bp + String("mlp.layer2.weight"), ctx),
    )


# Legacy symbol retained for streamed-stack callers. It now follows the runtime
# dtype contract: all checkpoint tensors, including q/k norm scales, preserve
# stored dtype; rms_norm handles the tiny mixed-dtype compute boundary.
def load_anima_block_weights_f32(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> AnimaBlockWeights:
    var bp = String("net.blocks.") + String(block_idx) + String(".")
    return AnimaBlockWeights(
        _load_tensor(st, bp + String("adaln_modulation_self_attn.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_self_attn.2.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_cross_attn.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_cross_attn.2.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_mlp.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_mlp.2.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.q_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.k_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.v_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.output_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.q_norm.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.k_norm.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.q_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.k_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.v_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.output_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.q_norm.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.k_norm.weight"), ctx),
        _load_tensor(st, bp + String("mlp.layer1.weight"), ctx),
        _load_tensor(st, bp + String("mlp.layer2.weight"), ctx),
    )


# Load Anima block `block_idx` for the DEVICE-RESIDENT fast path. Projection and
# norm checkpoint tensors all preserve stored dtype; mixed activation/norm dtype
# is handled inside rms_norm.
def load_anima_block_weights_bf16_normf32(
    st: SafeTensors, block_idx: Int, ctx: DeviceContext
) raises -> AnimaBlockWeights:
    var bp = String("net.blocks.") + String(block_idx) + String(".")
    return AnimaBlockWeights(
        _load_tensor(st, bp + String("adaln_modulation_self_attn.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_self_attn.2.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_cross_attn.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_cross_attn.2.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_mlp.1.weight"), ctx),
        _load_tensor(st, bp + String("adaln_modulation_mlp.2.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.q_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.k_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.v_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.output_proj.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.q_norm.weight"), ctx),
        _load_tensor(st, bp + String("self_attn.k_norm.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.q_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.k_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.v_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.output_proj.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.q_norm.weight"), ctx),
        _load_tensor(st, bp + String("cross_attn.k_norm.weight"), ctx),
        _load_tensor(st, bp + String("mlp.layer1.weight"), ctx),
        _load_tensor(st, bp + String("mlp.layer2.weight"), ctx),
    )


# Legacy symbol retained for callers; block tensors preserve checkpoint dtype.
def load_anima_all_blocks_f32(
    st: SafeTensors, num_blocks: Int, ctx: DeviceContext
) raises -> List[AnimaBlockWeights]:
    var blocks = List[AnimaBlockWeights]()
    for i in range(num_blocks):
        blocks.append(load_anima_block_weights_f32(st, i, ctx))
    return blocks^


# Load ALL `num_blocks` blocks in stored dtype resident ONCE — the 24GB-safe
# device-resident inference stack (28 blocks ≈ 3.7 GiB). Used by the no-save
# sampler forward so there is NO per-block disk reload across denoise steps.
def load_anima_all_blocks_bf16_normf32(
    st: SafeTensors, num_blocks: Int, ctx: DeviceContext
) raises -> List[AnimaBlockWeights]:
    var blocks = List[AnimaBlockWeights]()
    for i in range(num_blocks):
        blocks.append(load_anima_block_weights_bf16_normf32(st, i, ctx))
    return blocks^


# ── Resident (shared) base weights: x_embedder / t_embedder / t_embedding_norm /
#    final_layer. The LLM adapter (6 frozen blocks) is NOT loaded here — for LoRA
#    training the adapter output (context) is a cached, frozen INPUT; its weights
#    need no grad and are not on the LoRA-DiT path. (Full-FT would add them.)
#    Resident weights preserve checkpoint dtype. rms_norm handles the small
#    mixed-dtype timestep norm compute boundary. Key layout VERIFIED
#    vs the checkpoint header:
#      net.x_embedder.proj.1.weight       [2048, 68]
#      net.t_embedder.1.linear_1.weight   [2048, 2048]
#      net.t_embedder.1.linear_2.weight   [6144, 2048]
#      net.t_embedding_norm.weight        [2048]
#      net.final_layer.adaln_modulation.1.weight [256, 2048]
#      net.final_layer.adaln_modulation.2.weight [4096, 256]
#      net.final_layer.linear.weight      [64, 2048]
struct AnimaStackBase(Movable):
    var x_embed: TArc        # [D, 68]        x_embedder.proj.1.weight (patch embed)
    var te_lin1: TArc        # [D, D]         t_embedder.1.linear_1.weight
    var te_lin2: TArc        # [6144, D]      t_embedder.1.linear_2.weight (base_adaln)
    var t_norm: TArc         # [D]            t_embedding_norm.weight (RMSNorm)
    var fl_mod1: TArc        # [256, D]       final_layer.adaln_modulation.1.weight
    var fl_mod2: TArc        # [4096, 256]    final_layer.adaln_modulation.2.weight
    var fl_lin: TArc         # [64, D]        final_layer.linear.weight

    def __init__(
        out self,
        var x_embed: TArc, var te_lin1: TArc, var te_lin2: TArc, var t_norm: TArc,
        var fl_mod1: TArc, var fl_mod2: TArc, var fl_lin: TArc,
    ):
        self.x_embed = x_embed^
        self.te_lin1 = te_lin1^
        self.te_lin2 = te_lin2^
        self.t_norm = t_norm^
        self.fl_mod1 = fl_mod1^
        self.fl_mod2 = fl_mod2^
        self.fl_lin = fl_lin^


def load_anima_stack_base(st: SafeTensors, ctx: DeviceContext) raises -> AnimaStackBase:
    return AnimaStackBase(
        TArc(_load_tensor(st, String("net.x_embedder.proj.1.weight"), ctx)),
        TArc(_load_tensor(st, String("net.t_embedder.1.linear_1.weight"), ctx)),
        TArc(_load_tensor(st, String("net.t_embedder.1.linear_2.weight"), ctx)),
        TArc(_load_tensor(st, String("net.t_embedding_norm.weight"), ctx)),
        TArc(_load_tensor(st, String("net.final_layer.adaln_modulation.1.weight"), ctx)),
        TArc(_load_tensor(st, String("net.final_layer.adaln_modulation.2.weight"), ctx)),
        TArc(_load_tensor(st, String("net.final_layer.linear.weight"), ctx)),
    )


# Assert one named tensor's stored header shape matches `want`; print a receipt.
def _expect_shape(st: SafeTensors, name: String, var want: List[Int]) raises:
    var info = st.tensor_info(name)
    var got = List[Int]()
    for i in range(len(info.shape)):
        got.append(Int(info.shape[i]))
    var ok = len(got) == len(want)
    if ok:
        for i in range(len(want)):
            if got[i] != want[i]:
                ok = False
    var s = String("[")
    for i in range(len(got)):
        if i > 0:
            s += String(", ")
        s += String(got[i])
    s += String("]")
    print("  OK" if ok else "  MISMATCH", name, s)
    if not ok:
        raise Error(String("anima shape mismatch: ") + name)


# Verify a representative sample of REAL tensors load at the expected shapes:
# base weights + block 0 (shallowest) + block num_blocks-1 (deepest). RC=0 on
# success (Tenet 4: measured header reads with shape asserts).
def verify_anima_stack_shapes(st: SafeTensors, num_blocks: Int) raises:
    print("---- base (resident) weight shapes ----")
    _expect_shape(st, String("net.x_embedder.proj.1.weight"), [2048, 68])
    _expect_shape(st, String("net.t_embedder.1.linear_1.weight"), [2048, 2048])
    _expect_shape(st, String("net.t_embedder.1.linear_2.weight"), [6144, 2048])
    _expect_shape(st, String("net.t_embedding_norm.weight"), [2048])
    _expect_shape(st, String("net.final_layer.adaln_modulation.1.weight"), [256, 2048])
    _expect_shape(st, String("net.final_layer.adaln_modulation.2.weight"), [4096, 256])
    _expect_shape(st, String("net.final_layer.linear.weight"), [64, 2048])
    var probe = List[Int]()
    probe.append(0)
    probe.append(num_blocks - 1)
    for pi in range(len(probe)):
        var bi = probe[pi]
        var bp = String("net.blocks.") + String(bi) + String(".")
        print("---- block", bi, "weight shapes ----")
        _expect_shape(st, bp + String("adaln_modulation_self_attn.1.weight"), [256, 2048])
        _expect_shape(st, bp + String("adaln_modulation_self_attn.2.weight"), [6144, 256])
        _expect_shape(st, bp + String("self_attn.q_proj.weight"), [2048, 2048])
        _expect_shape(st, bp + String("self_attn.q_norm.weight"), [128])
        _expect_shape(st, bp + String("cross_attn.k_proj.weight"), [2048, 1024])
        _expect_shape(st, bp + String("mlp.layer1.weight"), [8192, 2048])
        _expect_shape(st, bp + String("mlp.layer2.weight"), [2048, 8192])
