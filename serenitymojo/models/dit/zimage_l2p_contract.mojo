# models/dit/zimage_l2p_contract.mojo - Z-Image L2P metadata contract.
#
# Header/static-shape gate for Z-Image L2P, the VAE-less pixel-space
# Z-Image-Turbo variant. This file intentionally has no DeviceContext, no
# tensor allocations, no model math, and no H2D weight load.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.registry.checkpoints import path_exists


comptime ZIMAGE_L2P_DEFAULT_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/L2P/model-1k-merge.safetensors"
comptime ZIMAGE_L2P_DEFAULT_CONDITIONING = (
    "/home/alex/EriDiffusion/inference-flame/output/l2p_embeddings.safetensors"
)
comptime ZIMAGE_L2P_INFER_WORKDIR = "/home/alex/EriDiffusion/inference-flame"
comptime ZIMAGE_L2P_WIDTH = 1024
comptime ZIMAGE_L2P_HEIGHT = 1024
comptime ZIMAGE_L2P_PIXEL_CHANNELS = 3
comptime ZIMAGE_L2P_PATCH_SIZE = 16
comptime ZIMAGE_L2P_PATCH_GRID_H = ZIMAGE_L2P_HEIGHT // ZIMAGE_L2P_PATCH_SIZE
comptime ZIMAGE_L2P_PATCH_GRID_W = ZIMAGE_L2P_WIDTH // ZIMAGE_L2P_PATCH_SIZE
comptime ZIMAGE_L2P_IMAGE_TOKENS = ZIMAGE_L2P_PATCH_GRID_H * ZIMAGE_L2P_PATCH_GRID_W
comptime ZIMAGE_L2P_PATCH_VECTOR_DIM = (
    ZIMAGE_L2P_PIXEL_CHANNELS * ZIMAGE_L2P_PATCH_SIZE * ZIMAGE_L2P_PATCH_SIZE
)
comptime ZIMAGE_L2P_HIDDEN = 3840
comptime ZIMAGE_L2P_DEPTH = 30
comptime ZIMAGE_L2P_REFINER_BLOCKS = 4
comptime ZIMAGE_L2P_ATTENTION_BLOCKS = ZIMAGE_L2P_DEPTH + ZIMAGE_L2P_REFINER_BLOCKS
comptime ZIMAGE_L2P_NUM_HEADS = 30
comptime ZIMAGE_L2P_HEAD_DIM = 128
comptime ZIMAGE_L2P_CAP_FEAT_DIM = 2560
comptime ZIMAGE_L2P_MLP_HIDDEN = 10240
comptime ZIMAGE_L2P_TIMESTEP_DIM = 256
comptime ZIMAGE_L2P_TIMESTEP_HIDDEN = 1024
comptime ZIMAGE_L2P_ADALN_DIM = 15360
comptime ZIMAGE_L2P_PAD_MULTIPLE = 32
comptime ZIMAGE_L2P_CHECKPOINT_TENSORS = 545
comptime ZIMAGE_L2P_DEFAULT_STEPS = 30
comptime ZIMAGE_L2P_LOCAL_DECODER_TENSORS = 28
comptime ZIMAGE_L2P_LD_C1 = 64
comptime ZIMAGE_L2P_LD_C2 = 128
comptime ZIMAGE_L2P_LD_C3 = 256
comptime ZIMAGE_L2P_LD_C4 = 512


def zimage_l2p_default_checkpoint_path() -> String:
    return String(ZIMAGE_L2P_DEFAULT_CHECKPOINT)


def zimage_l2p_default_conditioning_path() -> String:
    return String(ZIMAGE_L2P_DEFAULT_CONDITIONING)


def zimage_l2p_infer_command(embeddings_path: String, output_path: String) -> String:
    var cmd = String("cd ")
    cmd += String(ZIMAGE_L2P_INFER_WORKDIR)
    cmd += String(" && cargo run --release --bin l2p_infer -- --model ")
    cmd += String(ZIMAGE_L2P_DEFAULT_CHECKPOINT)
    cmd += String(" --embeddings ")
    cmd += embeddings_path
    cmd += String(" --output ")
    cmd += output_path
    cmd += String(" --height 1024 --width 1024 --steps 30 --cfg 2.0 --shift 3.0 --seed 42")
    return cmd^


def zimage_l2p_default_cfg_scale() -> Float32:
    return 2.0


def zimage_l2p_default_shift() -> Float32:
    return 3.0


def zimage_l2p_sigma(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if num_steps <= 0:
        raise Error("zimage_l2p_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("zimage_l2p_sigma: index out of range")
    if index == num_steps:
        return 0.0
    var t = 1.0 - Float32(index) / Float32(num_steps)
    return shift * t / (1.0 + (shift - 1.0) * t)


def zimage_l2p_schedule_delta(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if index < 0 or index >= num_steps:
        raise Error("zimage_l2p_schedule_delta: index out of range")
    var sigma = zimage_l2p_sigma(index, num_steps, shift)
    var sigma_next = zimage_l2p_sigma(index + 1, num_steps, shift)
    return sigma_next - sigma


def build_zimage_l2p_sigma_schedule(num_steps: Int, shift: Float32) raises -> List[Float32]:
    if num_steps <= 0:
        raise Error("build_zimage_l2p_sigma_schedule: num_steps must be > 0")
    var result = List[Float32]()
    for i in range(num_steps + 1):
        result.append(zimage_l2p_sigma(i, num_steps, shift))
    return result^


def zimage_l2p_model_timestep(sigma: Float32) -> Float32:
    return (1.0 - sigma) * 1000.0


def zimage_l2p_patch_grid_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("zimage_l2p_patch_grid_dim: image_dim must be > 0")
    if image_dim % ZIMAGE_L2P_PATCH_SIZE != 0:
        raise Error("Z-Image L2P image dimension must divide by patch_size=16")
    return image_dim // ZIMAGE_L2P_PATCH_SIZE


@fieldwise_init
struct ZImageL2PTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var patch_size: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var image_tokens: Int
    var patch_vector_dim: Int
    var pixel_elements: Int

    def validate_1024_contract(self) raises:
        if self.width != ZIMAGE_L2P_WIDTH or self.height != ZIMAGE_L2P_HEIGHT:
            raise Error("Z-Image L2P contract currently targets 1024x1024")
        if self.patch_size != ZIMAGE_L2P_PATCH_SIZE:
            raise Error("Z-Image L2P patch size must be 16")
        if (
            self.patch_grid_h != ZIMAGE_L2P_PATCH_GRID_H
            or self.patch_grid_w != ZIMAGE_L2P_PATCH_GRID_W
        ):
            raise Error("Z-Image L2P patch grid must be 64x64")
        if self.image_tokens != ZIMAGE_L2P_IMAGE_TOKENS:
            raise Error("Z-Image L2P image token count must be 4096")
        if self.patch_vector_dim != ZIMAGE_L2P_PATCH_VECTOR_DIM:
            raise Error("Z-Image L2P patch vector dim must be 768")
        if self.pixel_elements != (
            ZIMAGE_L2P_PIXEL_CHANNELS * ZIMAGE_L2P_HEIGHT * ZIMAGE_L2P_WIDTH
        ):
            raise Error("Z-Image L2P pixel element count mismatch")


@fieldwise_init
struct ZImageL2PConditioningContract(Copyable, Movable, ImplicitlyCopyable):
    var batch: Int
    var cap_tokens: Int
    var cap_feat_dim: Int
    var has_uncond: Bool
    var uncond_tokens: Int
    var padded_cap_tokens: Int
    var padded_uncond_tokens: Int

    def validate(self) raises:
        if self.batch != 1:
            raise Error("Z-Image L2P conditioning currently expects batch=1")
        if self.cap_tokens <= 0:
            raise Error("Z-Image L2P conditioning needs cap_feats tokens")
        if self.cap_feat_dim != ZIMAGE_L2P_CAP_FEAT_DIM:
            raise Error("Z-Image L2P cap_feats dim mismatch")
        if self.has_uncond and self.uncond_tokens <= 0:
            raise Error("Z-Image L2P cap_feats_uncond token count must be > 0")
        if self.padded_cap_tokens % ZIMAGE_L2P_PAD_MULTIPLE != 0:
            raise Error("Z-Image L2P padded cap token count must align to 32")
        if self.has_uncond and (
            self.padded_uncond_tokens % ZIMAGE_L2P_PAD_MULTIPLE != 0
        ):
            raise Error("Z-Image L2P padded uncond token count must align to 32")


def build_zimage_l2p_token_plan(width: Int, height: Int) raises -> ZImageL2PTokenPlan:
    var gh = zimage_l2p_patch_grid_dim(height)
    var gw = zimage_l2p_patch_grid_dim(width)
    return ZImageL2PTokenPlan(
        width,
        height,
        ZIMAGE_L2P_PATCH_SIZE,
        gh,
        gw,
        gh * gw,
        ZIMAGE_L2P_PATCH_VECTOR_DIM,
        ZIMAGE_L2P_PIXEL_CHANNELS * height * width,
    )


def zimage_l2p_pad_tokens(tokens: Int) raises -> Int:
    if tokens <= 0:
        raise Error("zimage_l2p_pad_tokens: token count must be > 0")
    var rem = tokens % ZIMAGE_L2P_PAD_MULTIPLE
    if rem == 0:
        return tokens
    return tokens + (ZIMAGE_L2P_PAD_MULTIPLE - rem)


def _shape1(a: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    return result^


def _shape2(a: Int, b: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    result.append(b)
    return result^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    result.append(b)
    result.append(c)
    result.append(d)
    return result^


def _shape_string(shape: List[Int]) -> String:
    var s = String("[")
    for i in range(len(shape)):
        if i > 0:
            s += String(", ")
        s += String(shape[i])
    s += String("]")
    return s^


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = got - expected
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        raise Error(
            String("Z-Image L2P float mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_tensor_shape(
    ref st: SafeTensors, name: String, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("Z-Image L2P tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("Z-Image L2P tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Z-Image L2P tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def _check_tensor(
    ref st: SafeTensors, name: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("Z-Image L2P tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_tensor_shape(st, name, expected_shape)


def _has_tensor(ref st: SafeTensors, name: String) -> Bool:
    var names = st.names()
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def _check_conditioning_tensor(
    ref st: SafeTensors, name: String
) raises -> ZImageL2PConditioningContract:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16 and info.dtype != STDtype.F32:
        raise Error(
            String("Z-Image L2P conditioning dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=BF16 or F32")
        )
    if len(info.shape) != 3:
        raise Error(
            String("Z-Image L2P conditioning rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=[1, seq, 2560]")
        )
    if info.shape[0] != 1:
        raise Error("Z-Image L2P conditioning batch must be 1")
    if info.shape[1] <= 0:
        raise Error("Z-Image L2P conditioning sequence must be non-empty")
    if info.shape[2] != ZIMAGE_L2P_CAP_FEAT_DIM:
        raise Error("Z-Image L2P conditioning feature dim must be 2560")

    var expected_nbytes = _numel(info.shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Z-Image L2P conditioning byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )

    return ZImageL2PConditioningContract(
        info.shape[0],
        info.shape[1],
        info.shape[2],
        False,
        0,
        zimage_l2p_pad_tokens(info.shape[1]),
        0,
    )


def validate_zimage_l2p_static_contract() raises:
    var plan = build_zimage_l2p_token_plan(ZIMAGE_L2P_WIDTH, ZIMAGE_L2P_HEIGHT)
    plan.validate_1024_contract()
    var hidden = ZIMAGE_L2P_HIDDEN
    var heads = ZIMAGE_L2P_NUM_HEADS
    var head_dim = ZIMAGE_L2P_HEAD_DIM
    var image_tokens = ZIMAGE_L2P_IMAGE_TOKENS
    var pad_multiple = ZIMAGE_L2P_PAD_MULTIPLE
    if hidden != heads * head_dim:
        raise Error("Z-Image L2P hidden/head dims mismatch")
    if image_tokens % pad_multiple != 0:
        raise Error("Z-Image L2P 1024 image tokens should need no image padding")
    _check_close(
        String("sigma[0]"),
        zimage_l2p_sigma(0, ZIMAGE_L2P_DEFAULT_STEPS, zimage_l2p_default_shift()),
        1.0,
        0.000001,
    )
    _check_close(
        String("sigma[end]"),
        zimage_l2p_sigma(
            ZIMAGE_L2P_DEFAULT_STEPS,
            ZIMAGE_L2P_DEFAULT_STEPS,
            zimage_l2p_default_shift(),
        ),
        0.0,
        0.000001,
    )
    _check_close(
        String("model_timestep_sigma1"),
        zimage_l2p_model_timestep(1.0),
        0.0,
        0.000001,
    )
    _check_close(
        String("model_timestep_sigma0"),
        zimage_l2p_model_timestep(0.0),
        1000.0,
        0.000001,
    )


def validate_zimage_l2p_checkpoint_header(checkpoint_path: String) raises:
    if not path_exists(checkpoint_path):
        raise Error(String("Z-Image L2P checkpoint missing: ") + checkpoint_path)

    var st = SafeTensors.open(checkpoint_path)
    if st.count() != ZIMAGE_L2P_CHECKPOINT_TENSORS:
        raise Error(
            String("Z-Image L2P tensor count mismatch: actual=")
            + String(st.count())
            + String(" expected=")
            + String(ZIMAGE_L2P_CHECKPOINT_TENSORS)
        )

    _check_tensor(
        st,
        String("all_x_embedder.16-1.weight"),
        STDtype.BF16,
        _shape2(ZIMAGE_L2P_HIDDEN, ZIMAGE_L2P_PATCH_VECTOR_DIM),
    )
    _check_tensor(
        st,
        String("cap_embedder.1.weight"),
        STDtype.BF16,
        _shape2(ZIMAGE_L2P_HIDDEN, ZIMAGE_L2P_CAP_FEAT_DIM),
    )
    _check_tensor(
        st,
        String("t_embedder.mlp.0.weight"),
        STDtype.BF16,
        _shape2(ZIMAGE_L2P_TIMESTEP_HIDDEN, ZIMAGE_L2P_TIMESTEP_DIM),
    )
    _check_tensor(
        st,
        String("noise_refiner.0.attention.to_q.weight"),
        STDtype.BF16,
        _shape2(ZIMAGE_L2P_HIDDEN, ZIMAGE_L2P_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.10.attention.to_q.weight"),
        STDtype.F32,
        _shape2(ZIMAGE_L2P_HIDDEN, ZIMAGE_L2P_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.10.attention.norm_q.weight"),
        STDtype.F32,
        _shape1(ZIMAGE_L2P_HEAD_DIM),
    )
    _check_tensor(
        st,
        String("layers.29.adaLN_modulation.0.weight"),
        STDtype.BF16,
        _shape2(ZIMAGE_L2P_ADALN_DIM, ZIMAGE_L2P_TIMESTEP_DIM),
    )
    _check_tensor(
        st,
        String("local_decoder.enc1.0.weight"),
        STDtype.BF16,
        _shape4(64, ZIMAGE_L2P_PIXEL_CHANNELS, 3, 3),
    )
    _check_tensor(
        st,
        String("local_decoder.bottleneck.0.weight"),
        STDtype.BF16,
        _shape4(512, 512 + ZIMAGE_L2P_HIDDEN, 1, 1),
    )
    _check_tensor(
        st,
        String("local_decoder.dec4.0.weight"),
        STDtype.BF16,
        _shape4(256, 1024, 3, 3),
    )
    _check_tensor(
        st,
        String("local_decoder.out_conv.weight"),
        STDtype.BF16,
        _shape4(ZIMAGE_L2P_PIXEL_CHANNELS, 64, 1, 1),
    )
    _check_tensor(st, String("x_pad_token"), STDtype.BF16, _shape2(1, ZIMAGE_L2P_HIDDEN))
    _check_tensor(st, String("cap_pad_token"), STDtype.BF16, _shape2(1, ZIMAGE_L2P_HIDDEN))


def validate_zimage_l2p_local_decoder_header(checkpoint_path: String) raises:
    if not path_exists(checkpoint_path):
        raise Error(String("Z-Image L2P checkpoint missing: ") + checkpoint_path)

    var st = SafeTensors.open(checkpoint_path)
    var found = 0
    var names = st.names()
    for i in range(len(names)):
        if names[i].byte_length() >= String("local_decoder.").byte_length():
            var is_local = True
            for j in range(String("local_decoder.").byte_length()):
                if names[i][byte = j] != String("local_decoder.")[byte = j]:
                    is_local = False
                    break
            if is_local:
                found += 1
    if found != ZIMAGE_L2P_LOCAL_DECODER_TENSORS:
        raise Error(
            String("Z-Image L2P local_decoder tensor count mismatch: actual=")
            + String(found)
            + String(" expected=")
            + String(ZIMAGE_L2P_LOCAL_DECODER_TENSORS)
        )

    _check_tensor(
        st,
        String("local_decoder.enc1.0.weight"),
        STDtype.BF16,
        _shape4(ZIMAGE_L2P_LD_C1, ZIMAGE_L2P_PIXEL_CHANNELS, 3, 3),
    )
    _check_tensor(st, String("local_decoder.enc1.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C1))
    _check_tensor(st, String("local_decoder.enc2.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C2, ZIMAGE_L2P_LD_C1, 3, 3))
    _check_tensor(st, String("local_decoder.enc2.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C2))
    _check_tensor(st, String("local_decoder.enc3.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C3, ZIMAGE_L2P_LD_C2, 3, 3))
    _check_tensor(st, String("local_decoder.enc3.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C3))
    _check_tensor(st, String("local_decoder.enc4.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C4, ZIMAGE_L2P_LD_C3, 3, 3))
    _check_tensor(st, String("local_decoder.enc4.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C4))

    _check_tensor(
        st,
        String("local_decoder.bottleneck.0.weight"),
        STDtype.BF16,
        _shape4(ZIMAGE_L2P_LD_C4, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_HIDDEN, 1, 1),
    )
    _check_tensor(st, String("local_decoder.bottleneck.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C4))

    _check_tensor(st, String("local_decoder.up4.1.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C4, ZIMAGE_L2P_LD_C4, 3, 3))
    _check_tensor(st, String("local_decoder.up4.1.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C4))
    _check_tensor(st, String("local_decoder.up3.1.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C3, ZIMAGE_L2P_LD_C3, 3, 3))
    _check_tensor(st, String("local_decoder.up3.1.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C3))
    _check_tensor(st, String("local_decoder.up2.1.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C2, ZIMAGE_L2P_LD_C2, 3, 3))
    _check_tensor(st, String("local_decoder.up2.1.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C2))
    _check_tensor(st, String("local_decoder.up1.1.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C1, ZIMAGE_L2P_LD_C1, 3, 3))
    _check_tensor(st, String("local_decoder.up1.1.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C1))

    _check_tensor(st, String("local_decoder.dec4.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C3, ZIMAGE_L2P_LD_C4 + ZIMAGE_L2P_LD_C4, 3, 3))
    _check_tensor(st, String("local_decoder.dec4.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C3))
    _check_tensor(st, String("local_decoder.dec3.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C2, ZIMAGE_L2P_LD_C3 + ZIMAGE_L2P_LD_C3, 3, 3))
    _check_tensor(st, String("local_decoder.dec3.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C2))
    _check_tensor(st, String("local_decoder.dec2.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C1, ZIMAGE_L2P_LD_C2 + ZIMAGE_L2P_LD_C2, 3, 3))
    _check_tensor(st, String("local_decoder.dec2.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C1))
    _check_tensor(st, String("local_decoder.dec1.0.weight"), STDtype.BF16, _shape4(ZIMAGE_L2P_LD_C1, ZIMAGE_L2P_LD_C1 + ZIMAGE_L2P_LD_C1, 3, 3))
    _check_tensor(st, String("local_decoder.dec1.0.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_LD_C1))

    _check_tensor(
        st,
        String("local_decoder.out_conv.weight"),
        STDtype.BF16,
        _shape4(ZIMAGE_L2P_PIXEL_CHANNELS, ZIMAGE_L2P_LD_C1, 1, 1),
    )
    _check_tensor(st, String("local_decoder.out_conv.bias"), STDtype.BF16, _shape1(ZIMAGE_L2P_PIXEL_CHANNELS))


def validate_zimage_l2p_default_checkpoint_contract() raises:
    validate_zimage_l2p_static_contract()
    validate_zimage_l2p_checkpoint_header(zimage_l2p_default_checkpoint_path())
    validate_zimage_l2p_local_decoder_header(zimage_l2p_default_checkpoint_path())


def validate_zimage_l2p_conditioning_header(
    embeddings_path: String, require_uncond: Bool
) raises -> ZImageL2PConditioningContract:
    if not path_exists(embeddings_path):
        raise Error(String("Z-Image L2P conditioning missing: ") + embeddings_path)

    var st = SafeTensors.open(embeddings_path)
    if st.count() < 1 or st.count() > 2:
        raise Error(
            String("Z-Image L2P conditioning tensor count mismatch: actual=")
            + String(st.count())
            + String(" expected=1 or 2")
        )
    var names = st.names()
    for i in range(len(names)):
        if names[i] != String("cap_feats") and names[i] != String("cap_feats_uncond"):
            raise Error(
                String("Z-Image L2P conditioning unexpected tensor: ")
                + names[i]
                + String(" expected only cap_feats/cap_feats_uncond")
            )

    var cond = _check_conditioning_tensor(st, String("cap_feats"))
    var has_uncond = _has_tensor(st, String("cap_feats_uncond"))
    var uncond_tokens = 0
    var padded_uncond_tokens = 0
    if has_uncond:
        var uncond = _check_conditioning_tensor(st, String("cap_feats_uncond"))
        if uncond.batch != cond.batch or uncond.cap_feat_dim != cond.cap_feat_dim:
            raise Error("Z-Image L2P uncond conditioning shape mismatch")
        uncond_tokens = uncond.cap_tokens
        padded_uncond_tokens = uncond.padded_cap_tokens
    elif require_uncond:
        raise Error("Z-Image L2P conditioning missing cap_feats_uncond")

    var result = ZImageL2PConditioningContract(
        cond.batch,
        cond.cap_tokens,
        cond.cap_feat_dim,
        has_uncond,
        uncond_tokens,
        cond.padded_cap_tokens,
        padded_uncond_tokens,
    )
    result.validate()
    return result
