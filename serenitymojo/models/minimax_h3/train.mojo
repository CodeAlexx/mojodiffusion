# MiniMax-H3 native trainer entrypoint seam.
#
# This executable intentionally stops after complete host-side config, cache,
# artifact, and released-base validation.  It must not construct DeviceContext
# until the H3 backward/update/checkpoint loop is product-wired and gated.

from std.sys import argv

from serenitymojo.training.minimax_h3.cache_consumer import (
    MiniMaxH3CacheConsumer,
    open_minimax_h3_cache,
)
from serenitymojo.training.minimax_h3.config import (
    read_minimax_h3_product_config,
)


def prepare_minimax_h3_trainer(
    config_path: String,
) raises -> MiniMaxH3CacheConsumer:
    """Complete the fail-closed host preflight; never creates a GPU context."""
    var config = read_minimax_h3_product_config(config_path)
    return open_minimax_h3_cache(config)


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: minimax_h3/train.mojo CONFIG_JSON")
    var consumer = prepare_minimax_h3_trainer(String(args[1]))
    print(
        "MiniMax H3 host preflight PASS for eri_with_trigger; verified samples:",
        consumer.len(),
    )
    raise Error(
        "MiniMax H3 product trainer is not launch-enabled: released-base "
        "backward, optimizer integration, save/resume, and the training loop "
        "remain gated; refusing before DeviceContext"
    )
