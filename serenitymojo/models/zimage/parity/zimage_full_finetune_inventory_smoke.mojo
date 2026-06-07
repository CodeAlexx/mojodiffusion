from serenitymojo.models.zimage.full_finetune_inventory import (
    zimage_full_finetune_tensor_names,
    zimage_full_finetune_inventory_expected_count,
)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def main() raises:
    var names = zimage_full_finetune_tensor_names()
    var expected = zimage_full_finetune_inventory_expected_count()
    _require(len(names) == expected, String("Z-Image full-finetune inventory count mismatch"))
    _require(names[0] == String("t_embedder.mlp.0.weight"), String("Z-Image first key mismatch"))
    _require(
        names[len(names) - 1] == String("layers.29.adaLN_modulation.0.bias"),
        String("Z-Image last key mismatch"),
    )
    print("Z-Image full-finetune inventory smoke PASS count=", len(names))
