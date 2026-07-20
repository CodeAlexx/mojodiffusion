# SCAIL-2 visual conditioning tower (OpenCLIP XLM-R ViT-H/14 visual arm).
#
# Oracle: zai-org/SCAIL-2, wan-scail2 commit
# 5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5, wan/modules/clip.py.
# `CLIPModel.visual` calls VisionTransformer(..., use_31_block=True), therefore
# the conditioning output is the pre-norm residual stream after blocks 0..30:
# [1, 257, 1280]. Block 31, post_norm and head are present in the official
# checkpoint schema but deliberately not executed.
#
# Production math is Mojo-native. The loader consumes the creator's exact
# visual-only safetensors key schema, validates all 393 tensors, and uploads
# only the 377 tensors used by the 31-block visual forward. Creator F32 weights
# are cast once to F16 at load, matching CLIPModel(dtype=torch.float16).

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.layout import patchify
from serenitymojo.ops.linear import linear, linear_bias
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.activations import gelu_exact
from serenitymojo.ops.tensor_algebra import add, concat, reshape, slice


comptime SCAIL2_CLIP_IMAGE = 224
comptime SCAIL2_CLIP_PATCH = 14
comptime SCAIL2_CLIP_SEQ = 257
comptime SCAIL2_CLIP_DIM = 1280
comptime SCAIL2_CLIP_HEADS = 16
comptime SCAIL2_CLIP_HEAD_DIM = 80
comptime SCAIL2_CLIP_FF = 5120
comptime SCAIL2_CLIP_USED_LAYERS = 31
comptime SCAIL2_CLIP_SCHEMA_LAYERS = 32
comptime SCAIL2_CLIP_SCHEMA_TENSORS = 393
comptime SCAIL2_CLIP_PATCH_DIM = 3 * SCAIL2_CLIP_PATCH * SCAIL2_CLIP_PATCH
comptime SCAIL2_CLIP_EPS = Float32(1.0e-5)


def _shape_matches(actual: List[Int], expected: List[Int]) -> Bool:
    if len(actual) != len(expected):
        return False
    for i in range(len(actual)):
        if actual[i] != expected[i]:
            return False
    return True


def _expect(
    st: ShardedSafeTensors, name: String, var expected: List[Int]
) raises:
    var tv = st.tensor_view(name)
    if tv.dtype != STDtype.F32 and tv.dtype != STDtype.F16:
        raise Error(
            String("SCAIL-2 CLIP tensor must be F32/F16: ") + name
            + String(" dtype=") + tv.dtype.name()
        )
    if not _shape_matches(tv.shape, expected):
        raise Error(String("SCAIL-2 CLIP tensor shape mismatch: ") + name)


def _validate_official_schema(st: ShardedSafeTensors) raises:
    """Fail closed on the exact 393-key creator visual-only state dict."""
    if len(st.names()) != SCAIL2_CLIP_SCHEMA_TENSORS:
        raise Error(
            String("SCAIL-2 CLIP schema tensor count: got ")
            + String(len(st.names())) + String(" expected 393")
        )
    _expect(st, String("log_scale"), [])
    _expect(st, String("visual.cls_embedding"), [1, 1, SCAIL2_CLIP_DIM])
    _expect(st, String("visual.pos_embedding"), [1, SCAIL2_CLIP_SEQ, SCAIL2_CLIP_DIM])
    _expect(st, String("visual.head"), [SCAIL2_CLIP_DIM, 1024])
    _expect(
        st, String("visual.patch_embedding.weight"),
        [SCAIL2_CLIP_DIM, 3, SCAIL2_CLIP_PATCH, SCAIL2_CLIP_PATCH],
    )
    _expect(st, String("visual.pre_norm.weight"), [SCAIL2_CLIP_DIM])
    _expect(st, String("visual.pre_norm.bias"), [SCAIL2_CLIP_DIM])
    _expect(st, String("visual.post_norm.weight"), [SCAIL2_CLIP_DIM])
    _expect(st, String("visual.post_norm.bias"), [SCAIL2_CLIP_DIM])
    for i in range(SCAIL2_CLIP_SCHEMA_LAYERS):
        var p = String("visual.transformer.") + String(i) + String(".")
        _expect(st, p + String("norm1.weight"), [SCAIL2_CLIP_DIM])
        _expect(st, p + String("norm1.bias"), [SCAIL2_CLIP_DIM])
        _expect(
            st, p + String("attn.to_qkv.weight"),
            [3 * SCAIL2_CLIP_DIM, SCAIL2_CLIP_DIM],
        )
        _expect(st, p + String("attn.to_qkv.bias"), [3 * SCAIL2_CLIP_DIM])
        _expect(
            st, p + String("attn.proj.weight"),
            [SCAIL2_CLIP_DIM, SCAIL2_CLIP_DIM],
        )
        _expect(st, p + String("attn.proj.bias"), [SCAIL2_CLIP_DIM])
        _expect(st, p + String("norm2.weight"), [SCAIL2_CLIP_DIM])
        _expect(st, p + String("norm2.bias"), [SCAIL2_CLIP_DIM])
        _expect(
            st, p + String("mlp.0.weight"),
            [SCAIL2_CLIP_FF, SCAIL2_CLIP_DIM],
        )
        _expect(st, p + String("mlp.0.bias"), [SCAIL2_CLIP_FF])
        _expect(
            st, p + String("mlp.2.weight"),
            [SCAIL2_CLIP_DIM, SCAIL2_CLIP_FF],
        )
        _expect(st, p + String("mlp.2.bias"), [SCAIL2_CLIP_DIM])


def _load_f16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var raw = Tensor.from_view(st.tensor_view(name), ctx)
    if raw.dtype() == STDtype.F16:
        return raw^
    return cast_tensor(raw, STDtype.F16, ctx)


def scail2_clip_block[
    S: Int, D: Int, H: Int, DH: Int, FF: Int
](
    hidden: Tensor,
    norm1_w: Tensor, norm1_b: Tensor,
    qkv_w: Tensor, qkv_b: Tensor,
    proj_w: Tensor, proj_b: Tensor,
    norm2_w: Tensor, norm2_b: Tensor,
    fc1_w: Tensor, fc1_b: Tensor,
    fc2_w: Tensor, fc2_b: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    """One creator-exact pre-norm ViT block; generic params enable tiny gates."""
    if D != H * DH:
        raise Error("SCAIL-2 CLIP block D != H*DH")
    var n1 = layer_norm(hidden, norm1_w, norm1_b, eps, ctx)
    var qkv = linear_bias(n1, qkv_w, qkv_b, ctx)
    var q = reshape(slice(qkv, 2, 0, D, ctx), [1, S, H, DH], ctx)
    var k = reshape(slice(qkv, 2, D, D, ctx), [1, S, H, DH], ctx)
    var v = reshape(slice(qkv, 2, 2 * D, D, ctx), [1, S, H, DH], ctx)
    var scale = Float32(1.0) / sqrt(Float32(DH))
    var attn = sdpa_nomask[1, S, H, DH](q, k, v, scale, ctx)
    attn = reshape(attn, [1, S, D], ctx)
    var projected = linear_bias(attn, proj_w, proj_b, ctx)
    var h1 = add(hidden, projected, ctx)

    var n2 = layer_norm(h1, norm2_w, norm2_b, eps, ctx)
    var mlp = linear_bias(n2, fc1_w, fc1_b, ctx)
    mlp = gelu_exact(mlp, ctx)
    mlp = linear_bias(mlp, fc2_w, fc2_b, ctx)
    return add(h1, mlp, ctx)


struct Scail2ClipVision(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing loaded SCAIL-2 CLIP tensor: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _add(mut self, name: String, var tensor: Tensor):
        self.name_to_idx[name] = len(self.weights)
        self.weights.append(ArcPointer(tensor^))

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> Scail2ClipVision:
        var st = ShardedSafeTensors.open(path)
        _validate_official_schema(st)
        var weights = List[ArcPointer[Tensor]]()
        var names = Dict[String, Int]()
        var model = Scail2ClipVision(weights^, names^)

        var roots = List[String]()
        roots.append(String("visual.cls_embedding"))
        roots.append(String("visual.pos_embedding"))
        roots.append(String("visual.patch_embedding.weight"))
        roots.append(String("visual.pre_norm.weight"))
        roots.append(String("visual.pre_norm.bias"))
        for ref name in roots:
            model._add(name, _load_f16(st, name, ctx))

        for i in range(SCAIL2_CLIP_USED_LAYERS):
            var p = String("visual.transformer.") + String(i) + String(".")
            var suffixes = List[String]()
            suffixes.append(String("norm1.weight"))
            suffixes.append(String("norm1.bias"))
            suffixes.append(String("attn.to_qkv.weight"))
            suffixes.append(String("attn.to_qkv.bias"))
            suffixes.append(String("attn.proj.weight"))
            suffixes.append(String("attn.proj.bias"))
            suffixes.append(String("norm2.weight"))
            suffixes.append(String("norm2.bias"))
            suffixes.append(String("mlp.0.weight"))
            suffixes.append(String("mlp.0.bias"))
            suffixes.append(String("mlp.2.weight"))
            suffixes.append(String("mlp.2.bias"))
            for ref suffix in suffixes:
                var name = p + suffix
                model._add(name, _load_f16(st, name, ctx))
        return model^

    def forward(self, preprocessed_nchw: Tensor, ctx: DeviceContext) raises -> Tensor:
        """Encode one exact preprocessed [1,3,224,224] image to [1,257,1280]."""
        var shape = preprocessed_nchw.shape()
        if (
            len(shape) != 4 or shape[0] != 1 or shape[1] != 3
            or shape[2] != SCAIL2_CLIP_IMAGE or shape[3] != SCAIL2_CLIP_IMAGE
        ):
            raise Error("SCAIL-2 CLIP input must be [1,3,224,224]")
        var image = cast_tensor(preprocessed_nchw, STDtype.F16, ctx)
        var patches = patchify(image, SCAIL2_CLIP_PATCH, ctx)
        ref patch_w = self._w(String("visual.patch_embedding.weight"))
        var patch_w_2d = reshape(
            patch_w, [SCAIL2_CLIP_DIM, SCAIL2_CLIP_PATCH_DIM], ctx
        )
        var hidden = linear(patches, patch_w_2d, None, ctx)
        ref cls = self._w(String("visual.cls_embedding"))
        hidden = concat(1, ctx, cls, hidden)
        ref pos = self._w(String("visual.pos_embedding"))
        hidden = add(hidden, pos, ctx)
        ref pre_w = self._w(String("visual.pre_norm.weight"))
        ref pre_b = self._w(String("visual.pre_norm.bias"))
        hidden = layer_norm(hidden, pre_w, pre_b, SCAIL2_CLIP_EPS, ctx)

        for i in range(SCAIL2_CLIP_USED_LAYERS):
            var p = String("visual.transformer.") + String(i) + String(".")
            hidden = scail2_clip_block[
                SCAIL2_CLIP_SEQ, SCAIL2_CLIP_DIM, SCAIL2_CLIP_HEADS,
                SCAIL2_CLIP_HEAD_DIM, SCAIL2_CLIP_FF,
            ](
                hidden,
                self._w(p + String("norm1.weight")),
                self._w(p + String("norm1.bias")),
                self._w(p + String("attn.to_qkv.weight")),
                self._w(p + String("attn.to_qkv.bias")),
                self._w(p + String("attn.proj.weight")),
                self._w(p + String("attn.proj.bias")),
                self._w(p + String("norm2.weight")),
                self._w(p + String("norm2.bias")),
                self._w(p + String("mlp.0.weight")),
                self._w(p + String("mlp.0.bias")),
                self._w(p + String("mlp.2.weight")),
                self._w(p + String("mlp.2.bias")),
                SCAIL2_CLIP_EPS, ctx,
            )
        return hidden^
