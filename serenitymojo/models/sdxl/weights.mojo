# models/sdxl/weights.mojo — LDM-format safetensors -> SDXL ResBlock weights.
#
# Loads ONE SDXL ResBlock's weights from a real .safetensors (LDM key layout),
# mirroring models/klein/weights.mojo's _load_host_f32 discipline. Conv weights
# are stored OIHW [Cout,Cin,Kh,Kw] on disk and remapped to RSCF [Kh,Kw,Cin,Cout]
# on the host (the foundation ops/conv.mojo + ops/conv2d_backward.mojo filter
# layout) via the SAME index remap proven in models/dit/sdxl_unet.mojo::_to_rscf
# and models/vae/decoder2d.mojo::_load_conv_weight_rscf.
#
# ── LDM ResBlock keys (verified vs inference-flame/src/models/sdxl_unet.rs
#    resblock(), lines 593-631) ──
#   {prefix}.in_layers.0.weight/bias    GroupNorm(32)  [Cin]
#   {prefix}.in_layers.2.weight/bias    Conv3x3        OIHW [Cout,Cin,3,3] / [Cout]
#   {prefix}.emb_layers.1.weight/bias   Linear         [Cout, time_embed_dim] / [Cout]
#   {prefix}.out_layers.0.weight/bias   GroupNorm(32)  [Cout]
#   {prefix}.out_layers.3.weight/bias   Conv3x3        OIHW [Cout,Cout,3,3] / [Cout]
#   {prefix}.skip_connection.weight/bias  Conv1x1 OIHW [Cout,Cin,1,1] (only Cin!=Cout)
#
# GroupNorm, embedding-linear, and conv tensors preserve checkpoint dtype. Conv
# weights are remapped OIHW -> RSCF on the host without widening BF16/F16 lists.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts


# Does the safetensors contain `name`? (SafeTensors has no .has(); scan names.)
def _has(st: SafeTensors, name: String) -> Bool:
    var ns = st.names()
    for ref n in ns:
        if n == name:
            return True
    return False


# Read one named tensor from the safetensors preserving checkpoint storage dtype.
def _load_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _rscf_shape(kh: Int, kw: Int, cin: Int, cout: Int) -> List[Int]:
    var rshape = List[Int]()
    rshape.append(kh); rshape.append(kw); rshape.append(cin); rshape.append(cout)
    return rshape^


def _load_conv_rscf_bf16(var w: Tensor, kh: Int, kw: Int, cin: Int, cout: Int, ctx: DeviceContext) raises -> Tensor:
    var host = w.to_host_bf16(ctx)
    var rscf = List[BFloat16]()
    var total = kh * kw * cin * cout
    for _ in range(total):
        rscf.append(BFloat16(Float32(0.0)))
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    return Tensor.from_host_bf16(rscf^, _rscf_shape(kh, kw, cin, cout), ctx)


def _load_conv_rscf_f16(var w: Tensor, kh: Int, kw: Int, cin: Int, cout: Int, ctx: DeviceContext) raises -> Tensor:
    var host = w.to_host_f16(ctx)
    var rscf = List[Float16]()
    var total = kh * kw * cin * cout
    for _ in range(total):
        rscf.append(Float16(0.0))
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    return Tensor.from_host_f16(rscf^, _rscf_shape(kh, kw, cin, cout), ctx)


def _load_conv_rscf_f32(var w: Tensor, kh: Int, kw: Int, cin: Int, cout: Int, ctx: DeviceContext) raises -> Tensor:
    var host = w.to_host(ctx)
    var rscf = List[Float32]()
    var total = kh * kw * cin * cout
    for _ in range(total):
        rscf.append(0.0)
    for o in range(cout):
        for ci in range(cin):
            for r in range(kh):
                for s in range(kw):
                    var src = ((o * cin + ci) * kh + r) * kw + s
                    var dst = ((r * kw + s) * cin + ci) * cout + o
                    rscf[dst] = host[src]
    return Tensor.from_host(rscf^, _rscf_shape(kh, kw, cin, cout), STDtype.F32, ctx)


# Load a PyTorch conv weight (OIHW [Cout,Cin,Kh,Kw]) and remap on host to RSCF
# [Kh,Kw,Cin,Cout]. Index remap proven in sdxl_unet.mojo::_to_rscf:
#   OIHW idx = ((o*Cin + ci)*Kh + r)*Kw + s
#   RSCF idx = ((r*Kw + s)*Cin + ci)*Cout + o
def load_conv_rscf_checkpoint_dtype(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var w = _load_tensor(st, name, ctx)
    var sh = w.shape()
    if len(sh) != 4:
        raise Error(String("conv weight ") + name + " not rank-4 OIHW")
    var cout = sh[0]
    var cin = sh[1]
    var kh = sh[2]
    var kw = sh[3]
    if w.dtype() == STDtype.BF16:
        return _load_conv_rscf_bf16(w^, kh, kw, cin, cout, ctx)
    if w.dtype() == STDtype.F16:
        return _load_conv_rscf_f16(w^, kh, kw, cin, cout, ctx)
    return _load_conv_rscf_f32(w^, kh, kw, cin, cout, ctx)


struct ResBlockWeights(Movable):
    """One SDXL ResBlock's weights, conv filters in RSCF.

    gn1_w/gn1_b: GroupNorm1 scale/shift   [Cin]
    conv1_w:     Conv1 filter RSCF        [3,3,Cin,Cout]
    conv1_b:     Conv1 bias               [Cout]
    emb_w/emb_b: time-emb Linear          [Cout, time_embed_dim] / [Cout]
    gn2_w/gn2_b: GroupNorm2 scale/shift   [Cout]
    conv2_w:     Conv2 filter RSCF        [3,3,Cout,Cout]
    conv2_b:     Conv2 bias               [Cout]
    has_skip:    True iff Cin != Cout (a 1x1 skip conv is present)
    skip_w:      skip Conv1x1 filter RSCF [1,1,Cin,Cout]  (valid iff has_skip)
    skip_b:      skip Conv1x1 bias        [Cout]          (valid iff has_skip)
    """

    var gn1_w: Tensor
    var gn1_b: Tensor
    var conv1_w: Tensor
    var conv1_b: Tensor
    var emb_w: Tensor
    var emb_b: Tensor
    var gn2_w: Tensor
    var gn2_b: Tensor
    var conv2_w: Tensor
    var conv2_b: Tensor
    var has_skip: Bool
    var skip_w: Tensor
    var skip_b: Tensor

    def __init__(
        out self,
        var gn1_w: Tensor, var gn1_b: Tensor,
        var conv1_w: Tensor, var conv1_b: Tensor,
        var emb_w: Tensor, var emb_b: Tensor,
        var gn2_w: Tensor, var gn2_b: Tensor,
        var conv2_w: Tensor, var conv2_b: Tensor,
        has_skip: Bool, var skip_w: Tensor, var skip_b: Tensor,
    ):
        self.gn1_w = gn1_w^; self.gn1_b = gn1_b^
        self.conv1_w = conv1_w^; self.conv1_b = conv1_b^
        self.emb_w = emb_w^; self.emb_b = emb_b^
        self.gn2_w = gn2_w^; self.gn2_b = gn2_b^
        self.conv2_w = conv2_w^; self.conv2_b = conv2_b^
        self.has_skip = has_skip
        self.skip_w = skip_w^; self.skip_b = skip_b^


# Load one ResBlock's weights from an LDM-format SDXL safetensors.
# `prefix` e.g. "input_blocks.4.0" or "middle_block.0".
def load_resblock_weights(
    st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> ResBlockWeights:
    var gn1_w = _load_tensor(st, prefix + String(".in_layers.0.weight"), ctx)
    var gn1_b = _load_tensor(st, prefix + String(".in_layers.0.bias"), ctx)
    var conv1_w = load_conv_rscf_checkpoint_dtype(st, prefix + String(".in_layers.2.weight"), ctx)
    var conv1_b = _load_tensor(st, prefix + String(".in_layers.2.bias"), ctx)
    var emb_w = _load_tensor(st, prefix + String(".emb_layers.1.weight"), ctx)
    var emb_b = _load_tensor(st, prefix + String(".emb_layers.1.bias"), ctx)
    var gn2_w = _load_tensor(st, prefix + String(".out_layers.0.weight"), ctx)
    var gn2_b = _load_tensor(st, prefix + String(".out_layers.0.bias"), ctx)
    var conv2_w = load_conv_rscf_checkpoint_dtype(st, prefix + String(".out_layers.3.weight"), ctx)
    var conv2_b = _load_tensor(st, prefix + String(".out_layers.3.bias"), ctx)

    var skip_key = prefix + String(".skip_connection.weight")
    var has_skip = _has(st, skip_key)
    if has_skip:
        var skip_w = load_conv_rscf_checkpoint_dtype(st, skip_key, ctx)
        var skip_b = _load_tensor(st, prefix + String(".skip_connection.bias"), ctx)
        return ResBlockWeights(
            gn1_w^, gn1_b^, conv1_w^, conv1_b^, emb_w^, emb_b^,
            gn2_w^, gn2_b^, conv2_w^, conv2_b^, True, skip_w^, skip_b^,
        )
    # No skip: stash 1-element placeholder tensors (never read when has_skip=False).
    var ph1 = _zeros1(ctx)
    var ph2 = _zeros1(ctx)
    return ResBlockWeights(
        gn1_w^, gn1_b^, conv1_w^, conv1_b^, emb_w^, emb_b^,
        gn2_w^, gn2_b^, conv2_w^, conv2_b^, False, ph1^, ph2^,
    )


def _zeros1(ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(0.0)
    var s = List[Int]()
    s.append(1)
    return Tensor.from_host(h, s^, STDtype.F32, ctx)
