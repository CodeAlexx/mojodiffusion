# MiniMax-H3 FP8-resident training synchronization policy.
#
# Blocks [resident, num_blocks) reuse one streamed mmap/device slab and must
# fence on every visit. Blocks [0, resident) own independent quantized storage
# and fence only periodically to bound allocator transients.


def h3_fp8_forward_should_fence(block: Int, resident: Int) -> Bool:
    # Fence the final resident block before entering the streamed tail, every
    # tail block, and each group of eight resident blocks.
    return block + 1 >= resident or (block % 8) == 7


def h3_fp8_backward_should_fence(block: Int, resident: Int) -> Bool:
    # Reverse traversal enters the streamed tail first. Every tail block must
    # settle before the shared slab is overwritten by the next stage().
    return block >= resident or (block % 8) == 0
