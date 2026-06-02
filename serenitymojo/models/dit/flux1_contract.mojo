# models/dit/flux1_contract.mojo - FLUX.1-dev manifest/checkpoint contract.
#
# Metadata-only gate for the FLUX.1-dev 1024 text-to-image path. It validates
# static manifest fields and safetensors headers needed by the Mojo pipeline
# before any DeviceContext, H2D weight load, or denoise math runs.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest
from serenitymojo.sampling.flux1_dev import build_flux1_packed_latent_plan


comptime FLUX1_DEFAULT_INPUTS_PATH = (
    "/home/alex/EriDiffusion/inference-flame/output/flux1_inputs.safetensors"
)


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


def _has_tensor(ref st: ShardedSafeTensors, name: String) -> Bool:
    var names = st.names()
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def _check_shape(ref st: ShardedSafeTensors, name: String, expected: List[Int]) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected):
        raise Error(
            String("FLUX.1 tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected)
        )
    for i in range(len(expected)):
        if info.shape[i] != expected[i]:
            raise Error(
                String("FLUX.1 tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected)
            )
    var expected_nbytes = _numel(expected) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("FLUX.1 tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def _check_tensor(
    ref st: ShardedSafeTensors,
    name: String,
    dtype: STDtype,
    expected: List[Int],
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("FLUX.1 tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape(st, name, expected)


def _check_floating_tensor(
    ref st: ShardedSafeTensors, name: String, expected: List[Int]
) raises:
    var info = st.tensor_info(name)
    if not (
        info.dtype == STDtype.BF16
        or info.dtype == STDtype.F16
        or info.dtype == STDtype.F32
    ):
        raise Error(
            String("FLUX.1 floating tensor expected for ")
            + name
            + String(", got ")
            + info.dtype.name()
        )
    _check_shape(st, name, expected)


def flux1_default_cached_inputs_path() -> String:
    return String(FLUX1_DEFAULT_INPUTS_PATH)


def validate_flux1_manifest_contract(manifest: ModelManifest) raises:
    if manifest.model_id != "flux1_dev":
        raise Error(String("FLUX.1 contract got manifest: ") + manifest.model_id)
    if manifest.family != ModelFamily.text_to_image():
        raise Error("FLUX.1 manifest family mismatch")
    if manifest.variant != "flux1-dev":
        raise Error(String("FLUX.1 manifest variant mismatch: ") + manifest.variant)
    if manifest.profile_name != "flux1_dev_1024":
        raise Error(String("FLUX.1 manifest profile mismatch: ") + manifest.profile_name)
    if manifest.default_width != 1024 or manifest.default_height != 1024:
        raise Error("FLUX.1 contract currently targets the 1024x1024 profile")
    if manifest.default_frames != 1:
        raise Error("FLUX.1 manifest must be single-frame text-to-image")
    var plan = build_flux1_packed_latent_plan(
        manifest.default_width, manifest.default_height, manifest.text_tokens
    )
    plan.validate_dev_1024_contract()
    if manifest.latent_channels != 16:
        raise Error("FLUX.1 manifest latent channel mismatch")
    if manifest.latent_downsample_s != 8:
        raise Error("FLUX.1 manifest VAE downsample mismatch")
    if manifest.latent_height() != plan.latent_h or manifest.latent_width() != plan.latent_w:
        raise Error("FLUX.1 manifest latent spatial mismatch")
    if manifest.patch_size != plan.patch_size:
        raise Error("FLUX.1 packed-token patch size mismatch")
    if manifest.image_tokens != plan.image_tokens:
        raise Error("FLUX.1 image token count mismatch")
    if manifest.text_tokens != plan.text_tokens:
        raise Error("FLUX.1 T5 token count mismatch")
    if manifest.total_sequence != plan.total_sequence:
        raise Error("FLUX.1 total sequence mismatch")
    if manifest.production_entry != "serenitymojo/pipeline/flux1_pipeline_smoke.mojo":
        raise Error("FLUX.1 production entry mismatch")


def validate_flux1_text_encoder_headers(text_encoder_root: String, tokenizer_path: String) raises:
    var clip_path = text_encoder_root + String("/clip_l.safetensors")
    var clip_tok = text_encoder_root + String("/clip_l.tokenizer.json")
    var t5_path = text_encoder_root + String("/t5xxl_fp16.safetensors")
    _require_path(String("FLUX.1 CLIP-L"), clip_path)
    _require_path(String("FLUX.1 CLIP tokenizer"), clip_tok)
    _require_path(String("FLUX.1 T5-XXL"), t5_path)
    _require_path(String("FLUX.1 T5 tokenizer"), tokenizer_path)

    var clip = ShardedSafeTensors.open(clip_path)
    _check_floating_tensor(
        clip,
        String("text_model.embeddings.token_embedding.weight"),
        _shape2(49408, 768),
    )
    _check_floating_tensor(
        clip,
        String("text_model.embeddings.position_embedding.weight"),
        _shape2(77, 768),
    )
    _check_floating_tensor(
        clip,
        String("text_model.encoder.layers.11.mlp.fc2.weight"),
        _shape2(768, 3072),
    )
    _check_floating_tensor(
        clip,
        String("text_model.final_layer_norm.weight"),
        _shape1(768),
    )

    var t5 = ShardedSafeTensors.open(t5_path)
    if _has_tensor(t5, String("shared.weight")):
        _check_floating_tensor(t5, String("shared.weight"), _shape2(32128, 4096))
    else:
        _check_floating_tensor(
            t5,
            String("encoder.embed_tokens.weight"),
            _shape2(32128, 4096),
        )
    _check_floating_tensor(
        t5,
        String("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
        _shape2(32, 64),
    )
    _check_floating_tensor(
        t5,
        String("encoder.block.23.layer.1.DenseReluDense.wo.weight"),
        _shape2(4096, 10240),
    )
    _check_floating_tensor(t5, String("encoder.final_layer_norm.weight"), _shape1(4096))


def validate_flux1_dit_header(dit_path: String) raises:
    _require_path(String("FLUX.1 DiT"), dit_path)
    var dit = ShardedSafeTensors.open(dit_path)
    if dit.num_tensors() != 780:
        raise Error(
            String("FLUX.1 DiT tensor count mismatch: actual=")
            + String(dit.num_tensors())
            + String(" expected=780")
        )

    _check_tensor(dit, String("img_in.weight"), STDtype.BF16, _shape2(3072, 64))
    _check_tensor(dit, String("img_in.bias"), STDtype.BF16, _shape1(3072))
    _check_tensor(dit, String("txt_in.weight"), STDtype.BF16, _shape2(3072, 4096))
    _check_tensor(
        dit,
        String("time_in.in_layer.weight"),
        STDtype.BF16,
        _shape2(3072, 256),
    )
    _check_tensor(
        dit,
        String("guidance_in.in_layer.weight"),
        STDtype.BF16,
        _shape2(3072, 256),
    )
    _check_tensor(
        dit,
        String("vector_in.in_layer.weight"),
        STDtype.BF16,
        _shape2(3072, 768),
    )
    _check_tensor(
        dit,
        String("final_layer.adaLN_modulation.1.weight"),
        STDtype.BF16,
        _shape2(6144, 3072),
    )
    _check_tensor(
        dit,
        String("final_layer.linear.weight"),
        STDtype.BF16,
        _shape2(64, 3072),
    )

    _check_tensor(
        dit,
        String("double_blocks.0.img_mod.lin.weight"),
        STDtype.BF16,
        _shape2(18432, 3072),
    )
    _check_tensor(
        dit,
        String("double_blocks.0.img_attn.qkv.weight"),
        STDtype.BF16,
        _shape2(9216, 3072),
    )
    _check_tensor(
        dit,
        String("double_blocks.18.txt_mlp.2.bias"),
        STDtype.BF16,
        _shape1(3072),
    )
    _check_tensor(
        dit,
        String("single_blocks.0.linear1.weight"),
        STDtype.BF16,
        _shape2(21504, 3072),
    )
    _check_tensor(
        dit,
        String("single_blocks.37.linear2.weight"),
        STDtype.BF16,
        _shape2(3072, 15360),
    )
    _check_tensor(
        dit,
        String("single_blocks.37.norm.key_norm.scale"),
        STDtype.BF16,
        _shape1(128),
    )


def validate_flux1_vae_header(vae_path: String) raises:
    _require_path(String("FLUX.1 LDM VAE"), vae_path)
    var vae = ShardedSafeTensors.open(vae_path)
    if vae.num_tensors() != 244:
        raise Error(
            String("FLUX.1 VAE tensor count mismatch: actual=")
            + String(vae.num_tensors())
            + String(" expected=244")
        )

    _check_tensor(
        vae,
        String("decoder.conv_in.weight"),
        STDtype.F32,
        _shape4(512, 16, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.mid.attn_1.q.weight"),
        STDtype.F32,
        _shape4(512, 512, 1, 1),
    )
    _check_tensor(
        vae,
        String("decoder.mid.block_2.conv2.weight"),
        STDtype.F32,
        _shape4(512, 512, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.up.3.block.0.conv1.weight"),
        STDtype.F32,
        _shape4(512, 512, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.up.1.block.0.nin_shortcut.weight"),
        STDtype.F32,
        _shape4(256, 512, 1, 1),
    )
    _check_tensor(
        vae,
        String("decoder.up.1.upsample.conv.weight"),
        STDtype.F32,
        _shape4(256, 256, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.up.0.block.0.conv1.weight"),
        STDtype.F32,
        _shape4(128, 256, 3, 3),
    )
    _check_tensor(vae, String("decoder.norm_out.weight"), STDtype.F32, _shape1(128))
    _check_tensor(
        vae,
        String("decoder.conv_out.weight"),
        STDtype.F32,
        _shape4(3, 128, 3, 3),
    )


def validate_flux1_cached_inputs_header(inputs_path: String) raises:
    _require_path(String("FLUX.1 cached inputs"), inputs_path)
    var st = ShardedSafeTensors.open(inputs_path)
    if st.num_tensors() != 6:
        raise Error(
            String("FLUX.1 cached input tensor count mismatch: actual=")
            + String(st.num_tensors())
            + String(" expected=6")
        )
    _check_tensor(st, String("noise_nchw"), STDtype.F32, _shape4(1, 16, 128, 128))
    _check_tensor(st, String("img_packed"), STDtype.F32, _shape3(1, 4096, 64))
    _check_tensor(st, String("img_ids"), STDtype.F32, _shape2(4096, 3))
    _check_tensor(st, String("txt_ids"), STDtype.F32, _shape2(512, 3))
    _check_tensor(st, String("t5_hidden"), STDtype.F32, _shape3(1, 512, 4096))
    _check_tensor(st, String("clip_pooled"), STDtype.F32, _shape2(1, 768))


def validate_flux1_pipeline_contract(manifest: ModelManifest) raises:
    validate_flux1_manifest_contract(manifest)
    validate_flux1_text_encoder_headers(manifest.text_encoder_root, manifest.tokenizer_path)
    validate_flux1_dit_header(manifest.denoiser_path)
    validate_flux1_vae_header(manifest.vae_path)
