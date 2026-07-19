from serenitymojo.models.klein.full_finetune_inventory import (
    klein_full_finetune_tensor_names,
    klein_full_finetune_inventory_expected_count,
)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def main() raises:
    var names = klein_full_finetune_tensor_names()
    var expected = klein_full_finetune_inventory_expected_count()
    _require(len(names) == expected, String("Klein full-finetune inventory count mismatch"))
    _require(names[0] == String("img_in.weight"), String("Klein first key mismatch"))
    _require(
        names[len(names) - 1] == String("single_blocks.23.norm.key_norm.scale"),
        String("Klein last key mismatch"),
    )
    print("Klein full-finetune inventory smoke PASS count=", len(names))
