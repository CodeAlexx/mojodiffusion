# models/zimage/full_finetune_checkpoint.mojo -- Z-Image full-finetune payload surface.
#
# Scope: model tensor payload collection/save plus tensor-name manifest, only.
# This does not implement a product full-finetune loop, optimizer sidecars, or
# model-struct rebind on load. The load helper below is explicitly payload-only.
#
# Dtype contract: this module only copies ArcPointer[Tensor] handles from the
# live runtime structs into FullFinetuneTensor entries. It does not call
# Tensor.to_host(), Tensor.from_host(), or introduce an F32 storage boundary.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.models.zimage.full_finetune_inventory import (
    ZIMAGE_FULL_FT_MAIN_DEPTH,
    ZIMAGE_FULL_FT_NUM_CR,
    ZIMAGE_FULL_FT_NUM_NR,
    zimage_full_finetune_checkpoint_key_manifest,
    zimage_full_finetune_inventory_expected_count,
)
from serenitymojo.models.zimage.real_weights import ZImageRealAux
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import (
    FullFinetuneTensor,
    assert_full_finetune_name_manifest_matches,
    load_full_finetune_model_tensors,
    save_full_finetune_model_tensors,
    save_full_finetune_name_manifest,
)


comptime TArc = ArcPointer[Tensor]


def _require_len(got: Int, expected: Int, label: String) raises:
    if got != expected:
        raise Error(
            label
            + String(" length mismatch: got ")
            + String(got)
            + String(" expected ")
            + String(expected)
        )


def _append_tensor(mut out: List[FullFinetuneTensor], name: String, tensor: TArc):
    out.append(FullFinetuneTensor(name, tensor.copy()))


def _append_zimage_aux_tensors(
    mut out: List[FullFinetuneTensor], aux: ZImageRealAux
):
    _append_tensor(out, String("t_embedder.mlp.0.weight"), aux.t_w0)
    _append_tensor(out, String("t_embedder.mlp.0.bias"), aux.t_b0)
    _append_tensor(out, String("t_embedder.mlp.2.weight"), aux.t_w2)
    _append_tensor(out, String("t_embedder.mlp.2.bias"), aux.t_b2)
    _append_tensor(out, String("cap_embedder.0.weight"), aux.cap_norm)
    _append_tensor(out, String("cap_embedder.1.weight"), aux.cap_lin_w)
    _append_tensor(out, String("cap_embedder.1.bias"), aux.cap_lin_b)
    _append_tensor(out, String("all_x_embedder.2-1.weight"), aux.x_w)
    _append_tensor(out, String("all_x_embedder.2-1.bias"), aux.x_b)
    _append_tensor(out, String("x_pad_token"), aux.x_pad_token)
    _append_tensor(out, String("cap_pad_token"), aux.cap_pad_token)
    _append_tensor(
        out,
        String("all_final_layer.2-1.adaLN_modulation.1.weight"),
        aux.final_mod_w,
    )
    _append_tensor(
        out,
        String("all_final_layer.2-1.adaLN_modulation.1.bias"),
        aux.final_mod_b,
    )
    _append_tensor(out, String("all_final_layer.2-1.linear.weight"), aux.final_lin_w)
    _append_tensor(out, String("all_final_layer.2-1.linear.bias"), aux.final_lin_b)


def _append_zimage_block_base_tensors(
    mut out: List[FullFinetuneTensor], prefix: String, w: ZImageBlockWeights
):
    var ap = prefix + String(".attention")
    var fp = prefix + String(".feed_forward")
    _append_tensor(out, prefix + String(".attention_norm1.weight"), w.n1)
    _append_tensor(out, ap + String(".to_q.weight"), w.wq)
    _append_tensor(out, ap + String(".to_k.weight"), w.wk)
    _append_tensor(out, ap + String(".to_v.weight"), w.wv)
    _append_tensor(out, ap + String(".to_out.0.weight"), w.wo)
    _append_tensor(out, ap + String(".norm_q.weight"), w.q_norm)
    _append_tensor(out, ap + String(".norm_k.weight"), w.k_norm)
    _append_tensor(out, prefix + String(".attention_norm2.weight"), w.n2)
    _append_tensor(out, prefix + String(".ffn_norm1.weight"), w.fn1)
    _append_tensor(out, fp + String(".w1.weight"), w.w1)
    _append_tensor(out, fp + String(".w3.weight"), w.w3)
    _append_tensor(out, fp + String(".w2.weight"), w.w2)
    _append_tensor(out, prefix + String(".ffn_norm2.weight"), w.fn2)


def _append_zimage_modulated_block_tensors(
    mut out: List[FullFinetuneTensor],
    prefix: String,
    w: ZImageBlockWeights,
    mod_w: TArc,
    mod_b: TArc,
):
    _append_zimage_block_base_tensors(out, prefix, w)
    _append_tensor(out, prefix + String(".adaLN_modulation.0.weight"), mod_w)
    _append_tensor(out, prefix + String(".adaLN_modulation.0.bias"), mod_b)


def _validate_zimage_full_finetune_runtime_shape(
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
) raises:
    _require_len(
        len(nr_blocks), ZIMAGE_FULL_FT_NUM_NR, String("noise_refiner blocks")
    )
    _require_len(
        len(cr_blocks), ZIMAGE_FULL_FT_NUM_CR, String("context_refiner blocks")
    )
    _require_len(len(main_blocks), ZIMAGE_FULL_FT_MAIN_DEPTH, String("main blocks"))
    _require_len(
        len(aux.nr_mod_w),
        ZIMAGE_FULL_FT_NUM_NR,
        String("noise_refiner adaLN weights"),
    )
    _require_len(
        len(aux.nr_mod_b),
        ZIMAGE_FULL_FT_NUM_NR,
        String("noise_refiner adaLN biases"),
    )
    _require_len(
        len(aux.main_mod_w),
        ZIMAGE_FULL_FT_MAIN_DEPTH,
        String("main adaLN weights"),
    )
    _require_len(
        len(aux.main_mod_b),
        ZIMAGE_FULL_FT_MAIN_DEPTH,
        String("main adaLN biases"),
    )


def _assert_collected_names_match_manifest(
    tensors: List[FullFinetuneTensor]
) raises:
    var names = zimage_full_finetune_checkpoint_key_manifest()
    _require_len(
        len(tensors), len(names), String("Z-Image full-finetune manifest names")
    )
    for i in range(len(names)):
        if tensors[i].name != names[i]:
            raise Error(
                String("Z-Image full-finetune key mismatch at index ")
                + String(i)
                + String(": collected ")
                + tensors[i].name
                + String(" expected ")
                + names[i]
            )


def zimage_full_finetune_model_tensors(
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
) raises -> List[FullFinetuneTensor]:
    """Collect current Z-Image transformer tensors in OneTrainer key order.

    Order matches zimage_full_finetune_tensor_names():
    aux tensors, modulated noise refiners, unmodulated context refiners, then
    modulated main layers. Context refiners deliberately omit
    context_refiner.*.adaLN_modulation keys because that runtime branch is
    unmodulated.
    """

    _validate_zimage_full_finetune_runtime_shape(
        aux, nr_blocks, cr_blocks, main_blocks
    )

    var out = List[FullFinetuneTensor]()
    _append_zimage_aux_tensors(out, aux)

    for i in range(len(nr_blocks)):
        _append_zimage_modulated_block_tensors(
            out,
            String("noise_refiner.") + String(i),
            nr_blocks[i],
            aux.nr_mod_w[i],
            aux.nr_mod_b[i],
        )

    for i in range(len(cr_blocks)):
        _append_zimage_block_base_tensors(
            out, String("context_refiner.") + String(i), cr_blocks[i]
        )

    for i in range(len(main_blocks)):
        _append_zimage_modulated_block_tensors(
            out,
            String("layers.") + String(i),
            main_blocks[i],
            aux.main_mod_w[i],
            aux.main_mod_b[i],
        )

    _require_len(
        len(out),
        zimage_full_finetune_inventory_expected_count(),
        String("Z-Image full-finetune collected tensors"),
    )
    _assert_collected_names_match_manifest(out)
    return out^


def save_zimage_full_finetune_checkpoint(
    aux: ZImageRealAux,
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    model_tensor_path: String,
    name_manifest_path: String,
    ctx: DeviceContext,
) raises -> Int:
    """Save Z-Image full-finetune model tensors and tensor-name manifest.

    This is a payload/manifest hook only. It does not save optimizer state and
    does not make full-finetune training or resume product-ready by itself.
    """

    var tensors = zimage_full_finetune_model_tensors(
        aux, nr_blocks, cr_blocks, main_blocks
    )
    var saved = save_full_finetune_model_tensors(tensors, model_tensor_path, ctx)
    var names = zimage_full_finetune_checkpoint_key_manifest()
    var manifest_saved = save_full_finetune_name_manifest(
        names, name_manifest_path, ctx
    )
    if manifest_saved != saved:
        raise Error(
            String("Z-Image full-finetune manifest count ")
            + String(manifest_saved)
            + String(" != saved tensor count ")
            + String(saved)
        )
    return saved


def load_zimage_full_finetune_payload_only(
    model_tensor_path: String,
    name_manifest_path: String,
    ctx: DeviceContext,
) raises -> List[FullFinetuneTensor]:
    """Load Z-Image full-finetune tensors as an ordered flat payload only.

    The manifest is checked against the Z-Image inventory first. The returned
    tensors are not rebound into ZImageRealAux/ZImageBlockWeights here.
    """

    var names = zimage_full_finetune_checkpoint_key_manifest()
    assert_full_finetune_name_manifest_matches(names, name_manifest_path)
    return load_full_finetune_model_tensors(names, model_tensor_path, ctx)
