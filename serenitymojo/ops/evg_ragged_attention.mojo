# serenitymojo/ops/evg_ragged_attention.mojo
#
# Mojo-native host planning for EVG's ragged 2-D mixed-precision attention.
#
# Source contract (audited at EVG commit faf55cce6095965378f3477a85fb6b6a4997e3d4):
#   evg/layers/attention/mpa/ragged_2d.py
#   evg/layers/attention/mpa/routing.py::_route_counts
#   examples/minimax-h3/mpa-ragged2d-mixed.yaml
#
# EVG is Apache-2.0. This is a source-faithful Mojo reimplementation, not a
# Python/Torch product dependency. The request-static planner covers every
# positive HxW raster with exactly ceil(H*W/capacity) connected blocks, never
# promotes padding to a logical token, and uses EVG's lexicographic objective:
# perimeter, bounding-box waste, spatial moment, square-aspect error, bands.
#
# The attention executor lives separately in `evg_attention_int8.mojo` so this
# request-static host policy retains a small, exact oracle-parity surface.

from std.collections import List
from std.math import ceildiv, log


@fieldwise_init
struct EVGBand(Copyable, Movable, ImplicitlyCopyable):
    var height: Int
    var blocks: Int


@fieldwise_init
struct EVGPartitionCost(Copyable, Movable, ImplicitlyCopyable):
    var perimeter: Float64
    var waste: Float64
    var moment: Float64
    var aspect_error: Float64
    var bands: Int


struct EVGRagged2DPartition(Movable):
    var height: Int
    var width: Int
    var capacity: Int
    var blocks: List[List[Int]]
    var token_to_block: List[Int]
    # Flattened row-major [block_count, block_count].
    var adjacency: List[Bool]

    def __init__(
        out self,
        height: Int,
        width: Int,
        capacity: Int,
        var blocks: List[List[Int]],
        var token_to_block: List[Int],
        var adjacency: List[Bool],
    ):
        self.height = height
        self.width = width
        self.capacity = capacity
        self.blocks = blocks^
        self.token_to_block = token_to_block^
        self.adjacency = adjacency^

    def block_count(self) -> Int:
        return len(self.blocks)

    def count(self, block: Int) -> Int:
        return len(self.blocks[block])

    def adjacent(self, source: Int, target: Int) -> Bool:
        return self.adjacency[source * self.block_count() + target]


@fieldwise_init
struct EVGRouteCounts(Copyable, Movable, ImplicitlyCopyable):
    var fp16: Int
    var int8: Int


@fieldwise_init
struct EVGH3SparsePolicy(Copyable, Movable, ImplicitlyCopyable):
    """Released EVG H3 schedule; the lower-precision phase is INT8 Q/K.

    EVG calls that phase FP8 because its SM89 executor stores V as E4M3. The
    Ampere path will keep the name `int8` honest until an SM86 FP8-V executor
    exists.
    """

    var sparsity_ratio: Float64
    var retained_int8_ratio: Float64
    var retained_fp16_ratio: Float64
    var dense_first_steps: Int
    var dense_first_layers: Int
    var layers_per_step: Int

    def validate(self) raises:
        if self.sparsity_ratio < 0.0 or self.sparsity_ratio >= 1.0:
            raise Error("EVG sparsity_ratio must be in [0,1)")
        if self.retained_int8_ratio <= 0.0 or self.retained_fp16_ratio <= 0.0:
            raise Error("EVG retained precision ratios must be positive")
        var ratio_sum = self.retained_int8_ratio + self.retained_fp16_ratio
        var ratio_diff = ratio_sum - 1.0
        if ratio_diff < 0.0:
            ratio_diff = -ratio_diff
        if ratio_diff > 1.0e-6:
            raise Error("EVG retained precision ratios must sum to one")
        if self.dense_first_steps < 0 or self.dense_first_layers < 0:
            raise Error("EVG dense guards must be nonnegative")
        if self.layers_per_step <= 0 \
                or self.dense_first_layers > self.layers_per_step:
            raise Error("EVG dense layer guard is outside the stack")

    def layer_sparsity(self, layer: Int) raises -> Float64:
        if layer < 0 or layer >= self.layers_per_step:
            raise Error("EVG layer is outside the configured stack")
        if layer >= 18 and layer < 34:
            return 0.82
        if layer >= 34 and layer < 50:
            return 0.58
        return self.sparsity_ratio

    def is_dense(self, step: Int, layer: Int) raises -> Bool:
        _ = self.layer_sparsity(layer)
        return step < self.dense_first_steps or layer < self.dense_first_layers


def evg_route_counts(
    items: Int,
    sparsity_ratio: Float64,
    int8_ratio: Float64,
    fp16_ratio: Float64,
) raises -> EVGRouteCounts:
    """EVG Hamilton allocation, returned as (FP16 rescue, INT8 main)."""
    if items <= 0:
        raise Error("EVG route items must be positive")
    if sparsity_ratio < 0.0 or sparsity_ratio >= 1.0:
        raise Error("EVG route sparsity must be in [0,1)")
    if int8_ratio <= 0.0 or fp16_ratio <= 0.0:
        raise Error("EVG route precision ratios must be positive")
    var ratio_diff = int8_ratio + fp16_ratio - 1.0
    if ratio_diff < 0.0:
        ratio_diff = -ratio_diff
    if ratio_diff > 1.0e-6:
        raise Error("EVG route precision ratios must sum to one")

    # Python: floor((1-sparsity)*items + 0.5), clamped to [1, items].
    var retained = Int((1.0 - sparsity_ratio) * Float64(items) + 0.5)
    if retained < 1:
        retained = 1
    if retained > items:
        retained = items
    var q16 = Float64(retained) * fp16_ratio
    var q8 = Float64(retained) * int8_ratio
    var n16 = Int(q16)
    var n8 = Int(q8)
    var remaining = retained - n16 - n8
    if remaining > 0:
        var rem16 = q16 - Float64(n16)
        var rem8 = q8 - Float64(n8)
        # Python sorts (-remainder, index), so FP16 wins an exact tie.
        if rem16 >= rem8:
            n16 += 1
        else:
            n8 += 1
    return EVGRouteCounts(n16, n8)


def _balanced_segment_sizes(tokens: Int, blocks: Int) -> List[Int]:
    var base = tokens // blocks
    var larger = tokens % blocks
    var sizes = List[Int](capacity=blocks)
    for i in range(blocks):
        sizes.append(base + (1 if i < larger else 0))
    return sizes^


def _band_serpentine_blocks(
    height: Int, width: Int, block_count: Int
) -> List[List[Int]]:
    var path = List[Int](capacity=height * width)
    for column in range(width):
        if column % 2 == 0:
            for row in range(height):
                path.append(row * width + column)
        else:
            for reverse_row in range(height):
                var row = height - 1 - reverse_row
                path.append(row * width + column)
    var sizes = _balanced_segment_sizes(height * width, block_count)
    var blocks = List[List[Int]](capacity=block_count)
    var offset = 0
    for block_id in range(block_count):
        var block = List[Int](capacity=sizes[block_id])
        for i in range(sizes[block_id]):
            block.append(path[offset + i])
        offset += sizes[block_id]
        blocks.append(block^)
    return blocks^


def _block_contains(block: List[Int], token: Int) -> Bool:
    for i in range(len(block)):
        if block[i] == token:
            return True
    return False


def _block_cost(block: List[Int], width: Int) -> EVGPartitionCost:
    var min_row = 0x7FFFFFFF
    var max_row = -1
    var min_col = 0x7FFFFFFF
    var max_col = -1
    var row_sum = Float64(0.0)
    var col_sum = Float64(0.0)
    for i in range(len(block)):
        var row = block[i] // width
        var col = block[i] % width
        if row < min_row:
            min_row = row
        if row > max_row:
            max_row = row
        if col < min_col:
            min_col = col
        if col > max_col:
            max_col = col
        row_sum += Float64(row)
        col_sum += Float64(col)
    var centroid_row = row_sum / Float64(len(block))
    var centroid_col = col_sum / Float64(len(block))
    var perimeter = 0
    var moment = Float64(0.0)
    for i in range(len(block)):
        var token = block[i]
        var row = token // width
        var col = token % width
        if row == 0 or not _block_contains(block, token - width):
            perimeter += 1
        if not _block_contains(block, token + width):
            perimeter += 1
        if col == 0 or not _block_contains(block, token - 1):
            perimeter += 1
        if col + 1 == width or not _block_contains(block, token + 1):
            perimeter += 1
        var dr = Float64(row) - centroid_row
        var dc = Float64(col) - centroid_col
        moment += dr * dr + dc * dc
    var box_height = max_row - min_row + 1
    var box_width = max_col - min_col + 1
    var aspect = log(Float64(box_width) / Float64(box_height))
    if aspect < 0.0:
        aspect = -aspect
    return EVGPartitionCost(
        Float64(perimeter),
        Float64(box_height * box_width - len(block)),
        moment,
        aspect,
        1,
    )


def _zero_cost() -> EVGPartitionCost:
    return EVGPartitionCost(0.0, 0.0, 0.0, 0.0, 0)


def _add_cost(a: EVGPartitionCost, b: EVGPartitionCost) -> EVGPartitionCost:
    return EVGPartitionCost(
        a.perimeter + b.perimeter,
        a.waste + b.waste,
        a.moment + b.moment,
        a.aspect_error + b.aspect_error,
        a.bands + b.bands,
    )


def _partition_cost(blocks: List[List[Int]], width: Int) -> EVGPartitionCost:
    var total = _zero_cost()
    for i in range(len(blocks)):
        total = _add_cost(total, _block_cost(blocks[i], width))
    return total


def _cost_less(a: EVGPartitionCost, b: EVGPartitionCost) -> Bool:
    if a.perimeter != b.perimeter:
        return a.perimeter < b.perimeter
    if a.waste != b.waste:
        return a.waste < b.waste
    if a.moment != b.moment:
        return a.moment < b.moment
    if a.aspect_error != b.aspect_error:
        return a.aspect_error < b.aspect_error
    return a.bands < b.bands


def _cost_equal(a: EVGPartitionCost, b: EVGPartitionCost) -> Bool:
    return (
        a.perimeter == b.perimeter
        and a.waste == b.waste
        and a.moment == b.moment
        and a.aspect_error == b.aspect_error
        and a.bands == b.bands
    )


def _bands_less(a: List[EVGBand], b: List[EVGBand]) -> Bool:
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        if a[i].height != b[i].height:
            return a[i].height < b[i].height
        if a[i].blocks != b[i].blocks:
            return a[i].blocks < b[i].blocks
    return len(a) < len(b)


def _candidate_less(
    cost: EVGPartitionCost,
    bands: List[EVGBand],
    incumbent_cost: EVGPartitionCost,
    incumbent_bands: List[EVGBand],
) -> Bool:
    if _cost_less(cost, incumbent_cost):
        return True
    return _cost_equal(cost, incumbent_cost) and _bands_less(bands, incumbent_bands)


struct _OrientationResult(Movable):
    var blocks: List[List[Int]]
    var cost: EVGPartitionCost

    def __init__(out self, var blocks: List[List[Int]], cost: EVGPartitionCost):
        self.blocks = blocks^
        self.cost = cost


def _orientation_candidate(
    height: Int,
    width: Int,
    capacity: Int,
    target_blocks: Int,
) raises -> _OrientationResult:
    var stride = target_blocks + 1
    var state_count = (height + 1) * stride
    var present = List[Bool](capacity=state_count)
    var costs = List[EVGPartitionCost](capacity=state_count)
    var histories = List[List[EVGBand]](capacity=state_count)
    for _ in range(state_count):
        present.append(False)
        costs.append(_zero_cost())
        histories.append(List[EVGBand]())
    present[0] = True

    # EVG caches each local (band_height, band_blocks) construction. Without
    # this request-static memo the 64x65 arbitrary-shape gate repeats the same
    # O(tokens*capacity) perimeter/moment work thousands of times.
    var cache_count = (height + 1) * stride
    var cache_present = List[Bool](capacity=cache_count)
    var cache_costs = List[EVGPartitionCost](capacity=cache_count)
    var cache_blocks = List[List[List[Int]]](capacity=cache_count)
    for _ in range(cache_count):
        cache_present.append(False)
        cache_costs.append(_zero_cost())
        cache_blocks.append(List[List[Int]]())

    for row_offset in range(height):
        for blocks_used in range(target_blocks + 1):
            var state_index = row_offset * stride + blocks_used
            if not present[state_index]:
                continue
            var old_cost = costs[state_index]
            var old_bands = histories[state_index].copy()
            for band_height in range(1, height - row_offset + 1):
                var min_band_blocks = ceildiv(band_height * width, capacity)
                var remaining_rows = height - row_offset - band_height
                var min_remaining_blocks = ceildiv(remaining_rows * width, capacity)
                var max_band_blocks = target_blocks - blocks_used - min_remaining_blocks
                if max_band_blocks > band_height * width:
                    max_band_blocks = band_height * width
                if max_band_blocks < min_band_blocks:
                    continue
                for band_blocks in range(min_band_blocks, max_band_blocks + 1):
                    var cache_key = band_height * stride + band_blocks
                    if not cache_present[cache_key]:
                        var local_blocks = _band_serpentine_blocks(
                            band_height, width, band_blocks
                        )
                        cache_costs[cache_key] = _partition_cost(
                            local_blocks, width
                        )
                        cache_blocks[cache_key] = local_blocks^
                        cache_present[cache_key] = True
                    var candidate_cost = _add_cost(
                        old_cost, cache_costs[cache_key]
                    )
                    var candidate_bands = old_bands.copy()
                    candidate_bands.append(EVGBand(band_height, band_blocks))
                    var key = (row_offset + band_height) * stride \
                        + blocks_used + band_blocks
                    if not present[key] or _candidate_less(
                        candidate_cost,
                        candidate_bands,
                        costs[key],
                        histories[key],
                    ):
                        present[key] = True
                        costs[key] = candidate_cost
                        histories[key] = candidate_bands^

    var final_index = height * stride + target_blocks
    if not present[final_index]:
        raise Error("EVG stripe DP found no capacity-minimum cover")
    var result_blocks = List[List[Int]](capacity=target_blocks)
    var row_offset = 0
    for band_index in range(len(histories[final_index])):
        var band = histories[final_index][band_index]
        var local = cache_blocks[band.height * stride + band.blocks].copy()
        for block_index in range(len(local)):
            var shifted = List[Int](capacity=len(local[block_index]))
            for token_index in range(len(local[block_index])):
                var token = local[block_index][token_index]
                shifted.append((token // width + row_offset) * width + token % width)
            result_blocks.append(shifted^)
        row_offset += band.height
    return _OrientationResult(result_blocks^, costs[final_index])


def _transpose_to_raster(
    blocks: List[List[Int]], height: Int, width: Int
) -> List[List[Int]]:
    var out = List[List[Int]](capacity=len(blocks))
    for block_index in range(len(blocks)):
        var block = List[Int](capacity=len(blocks[block_index]))
        for token_index in range(len(blocks[block_index])):
            var token = blocks[block_index][token_index]
            var transposed_row = token // height
            var transposed_column = token % height
            block.append(transposed_column * width + transposed_row)
        out.append(block^)
    return out^


def _sort_block(mut block: List[Int]):
    for i in range(1, len(block)):
        var value = block[i]
        var j = i
        while j > 0 and block[j - 1] > value:
            block[j] = block[j - 1]
            j -= 1
        block[j] = value


def _build_adjacency(
    height: Int,
    width: Int,
    token_to_block: List[Int],
    block_count: Int,
) -> List[Bool]:
    var adjacency = List[Bool](capacity=block_count * block_count)
    for _ in range(block_count * block_count):
        adjacency.append(False)
    for block_id in range(block_count):
        adjacency[block_id * block_count + block_id] = True
    for row in range(height):
        for column in range(width):
            var token = row * width + column
            var source = token_to_block[token]
            if column + 1 < width:
                var target = token_to_block[token + 1]
                adjacency[source * block_count + target] = True
                adjacency[target * block_count + source] = True
            if row + 1 < height:
                var target = token_to_block[token + width]
                adjacency[source * block_count + target] = True
                adjacency[target * block_count + source] = True
    return adjacency^


def make_evg_ragged_2d_partition(
    height: Int, width: Int, capacity: Int = 64
) raises -> EVGRagged2DPartition:
    if height <= 0 or width <= 0:
        raise Error("EVG ragged height/width must be positive")
    if capacity <= 0:
        raise Error("EVG ragged capacity must be positive")
    var target_blocks = ceildiv(height * width, capacity)
    var selected: List[List[Int]]
    if height == 1 or width == 1:
        selected = _band_serpentine_blocks(1, height * width, target_blocks)
    else:
        var horizontal = _orientation_candidate(
            height, width, capacity, target_blocks
        )
        var transposed = _orientation_candidate(
            width, height, capacity, target_blocks
        )
        var vertical_blocks = _transpose_to_raster(
            transposed.blocks, height, width
        )
        var vertical_cost = _partition_cost(vertical_blocks, width)
        # Python compares (cost, False) <= (cost, True), favoring horizontal
        # on an exact tie.
        if _cost_less(vertical_cost, horizontal.cost):
            selected = vertical_blocks.copy()
        else:
            selected = horizontal.blocks.copy()

    for block_index in range(len(selected)):
        _sort_block(selected[block_index])
    var token_to_block = List[Int](capacity=height * width)
    for _ in range(height * width):
        token_to_block.append(-1)
    for block_id in range(len(selected)):
        if len(selected[block_id]) == 0 or len(selected[block_id]) > capacity:
            raise Error("EVG ragged block violates physical capacity")
        for token_index in range(len(selected[block_id])):
            var token = selected[block_id][token_index]
            if token < 0 or token >= height * width:
                raise Error("EVG ragged block contains an invalid token")
            if token_to_block[token] != -1:
                raise Error("EVG ragged partition assigned a token twice")
            token_to_block[token] = block_id
    for token in range(height * width):
        if token_to_block[token] < 0:
            raise Error("EVG ragged partition did not cover the raster")
    if len(selected) != target_blocks:
        raise Error("EVG ragged partition missed the capacity lower bound")
    var adjacency = _build_adjacency(
        height, width, token_to_block, len(selected)
    )
    return EVGRagged2DPartition(
        height,
        width,
        capacity,
        selected^,
        token_to_block^,
        adjacency^,
    )
