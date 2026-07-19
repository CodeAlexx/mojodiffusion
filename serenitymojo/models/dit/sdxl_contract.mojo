# models/dit/sdxl_contract.mojo - SDXL header/conditioning contract checks.
#
# This is a metadata-only gate for the cached-embedding SDXL path. It opens
# safetensors headers and validates the shape/dtype keys that the Mojo UNet,
# LDM VAE decoder, and scheduler smoke expect before any expensive H2D load or
# denoise step.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.model_manifest import ModelManifest, sdxl_default_manifest


comptime SDXL_CACHED_EMBEDDINGS_PATH = (
    "/home/alex/EriDiffusion/inference-flame/output/sdxl_embeddings.safetensors"
)
comptime SDXL_ENCODER_WORKDIR = "/home/alex/EriDiffusion/inference-flame"


def sdxl_default_cached_embeddings_path() -> String:
    return String(SDXL_CACHED_EMBEDDINGS_PATH)


def sdxl_cached_embedding_generator_command(output_path: String) -> String:
    var cmd = String("cd ")
    cmd += String(SDXL_ENCODER_WORKDIR)
    cmd += String(
        " && cargo run --release --bin sdxl_encode -- --prompt '<prompt>' --negative '' --output "
    )
    cmd += output_path
    return cmd^


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


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


def _require_path(label: String, path: String) raises:
    if not path_exists(path):
        raise Error(label + String(" missing: ") + path)


def _require_cached_embedding_path(path: String) raises:
    if not path_exists(path):
        raise Error(
            String("SDXL cached embeddings missing: ")
            + path
            + String("; generate with: ")
            + sdxl_cached_embedding_generator_command(path)
        )


def _check_tensor(
    ref st: ShardedSafeTensors,
    name: String,
    dtype: STDtype,
    expected_shape: List[Int],
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("SDXL tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("SDXL tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("SDXL tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("SDXL tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def validate_sdxl_manifest_contract(manifest: ModelManifest) raises:
    if manifest.model_id != "sdxl":
        raise Error(String("SDXL contract got manifest: ") + manifest.model_id)
    if manifest.default_width != 1024 or manifest.default_height != 1024:
        raise Error("SDXL contract currently targets the 1024x1024 static profile")
    if manifest.latent_channels != 4:
        raise Error("SDXL manifest latent channel mismatch")
    if manifest.latent_downsample_s != 8:
        raise Error("SDXL manifest latent downsample mismatch")
    if manifest.latent_height() != 128 or manifest.latent_width() != 128:
        raise Error("SDXL manifest latent spatial mismatch")
    if manifest.text_tokens != 77:
        raise Error("SDXL manifest text-token count mismatch")
    if manifest.image_tokens != 16384:
        raise Error("SDXL manifest image-token count mismatch")


def validate_sdxl_unet_header(unet_path: String) raises:
    _require_path(String("SDXL UNet"), unet_path)
    var unet = ShardedSafeTensors.open(unet_path)
    if unet.num_tensors() != 1680:
        raise Error(
            String("SDXL UNet tensor count mismatch: actual=")
            + String(unet.num_tensors())
            + String(" expected=1680")
        )

    _check_tensor(
        unet,
        String("time_embed.0.weight"),
        STDtype.BF16,
        _shape2(1280, 320),
    )
    _check_tensor(
        unet,
        String("time_embed.2.weight"),
        STDtype.BF16,
        _shape2(1280, 1280),
    )
    _check_tensor(
        unet,
        String("label_emb.0.0.weight"),
        STDtype.BF16,
        _shape2(1280, 2816),
    )
    _check_tensor(
        unet,
        String("input_blocks.0.0.weight"),
        STDtype.BF16,
        _shape4(320, 4, 3, 3),
    )
    _check_tensor(
        unet,
        String("input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"),
        STDtype.BF16,
        _shape2(640, 2048),
    )
    _check_tensor(
        unet,
        String("middle_block.1.transformer_blocks.9.attn2.to_v.weight"),
        STDtype.BF16,
        _shape2(1280, 2048),
    )
    _check_tensor(
        unet,
        String("output_blocks.2.2.conv.weight"),
        STDtype.BF16,
        _shape4(1280, 1280, 3, 3),
    )
    _check_tensor(unet, String("out.0.weight"), STDtype.BF16, _shape1(320))
    _check_tensor(unet, String("out.2.weight"), STDtype.BF16, _shape4(4, 320, 3, 3))


def validate_sdxl_vae_header(vae_path: String) raises:
    _require_path(String("SDXL VAE"), vae_path)
    var vae = ShardedSafeTensors.open(vae_path)
    if vae.num_tensors() != 250:
        raise Error(
            String("SDXL VAE tensor count mismatch: actual=")
            + String(vae.num_tensors())
            + String(" expected=250")
        )

    _check_tensor(
        vae,
        String("post_quant_conv.weight"),
        STDtype.F32,
        _shape4(4, 4, 1, 1),
    )
    _check_tensor(vae, String("post_quant_conv.bias"), STDtype.F32, _shape1(4))
    _check_tensor(
        vae,
        String("decoder.conv_in.weight"),
        STDtype.F32,
        _shape4(512, 4, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.mid.attn_1.q.weight"),
        STDtype.F32,
        _shape4(512, 512, 1, 1),
    )
    _check_tensor(
        vae,
        String("decoder.up.3.upsample.conv.weight"),
        STDtype.F32,
        _shape4(512, 512, 3, 3),
    )
    _check_tensor(vae, String("decoder.norm_out.weight"), STDtype.F32, _shape1(128))
    _check_tensor(
        vae,
        String("decoder.conv_out.weight"),
        STDtype.F32,
        _shape4(3, 128, 3, 3),
    )


def validate_sdxl_cached_embedding_header(emb_path: String) raises:
    _require_cached_embedding_path(emb_path)
    var emb = ShardedSafeTensors.open(emb_path)
    if emb.num_tensors() != 4:
        raise Error(
            String("SDXL embedding tensor count mismatch: actual=")
            + String(emb.num_tensors())
            + String(" expected=4")
        )

    _check_tensor(emb, String("context"), STDtype.BF16, _shape3(1, 77, 2048))
    _check_tensor(
        emb,
        String("context_uncond"),
        STDtype.BF16,
        _shape3(1, 77, 2048),
    )
    _check_tensor(emb, String("y"), STDtype.BF16, _shape2(1, 2816))
    _check_tensor(emb, String("y_uncond"), STDtype.BF16, _shape2(1, 2816))


def validate_sdxl_static_checkpoint_contract(unet_path: String, vae_path: String) raises:
    var manifest = sdxl_default_manifest()
    validate_sdxl_manifest_contract(manifest)
    if unet_path != manifest.denoiser_path:
        raise Error("SDXL UNet path does not match registered manifest")
    if vae_path != manifest.vae_path:
        raise Error("SDXL VAE path does not match registered manifest")
    validate_sdxl_unet_header(unet_path)
    validate_sdxl_vae_header(vae_path)


def validate_sdxl_pipeline_contract(
    unet_path: String, vae_path: String, emb_path: String
) raises:
    validate_sdxl_static_checkpoint_contract(unet_path, vae_path)
    validate_sdxl_cached_embedding_header(emb_path)


def sdxl_cached_embeddings_present(emb_path: String) -> Bool:
    return path_exists(emb_path)
