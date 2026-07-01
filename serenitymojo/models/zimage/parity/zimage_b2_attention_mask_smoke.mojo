# ZImage B2 attention mask construction smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_b2_attention_mask_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.models.zimage.zimage_stack_lora import zimage_key_tail_mask_f32


def _idx(B: Int, H: Int, S: Int, b: Int, h: Int, q: Int, k: Int) -> Int:
    return (((b * H + h) * S + q) * S + k)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("zimage_b2_attention_mask_smoke FAILED: ") + msg)


def main() raises:
    comptime B = 2
    comptime H = 3
    comptime S = 5
    var ctx = DeviceContext()
    var mask = zimage_key_tail_mask_f32[B, H, S](5, 3, ctx)
    var shape = mask.shape()
    _check(len(shape) == 4, "rank")
    _check(shape[0] == B and shape[1] == H and shape[2] == S and shape[3] == S, "shape")
    var host = mask.to_host(ctx)

    for h in range(H):
        for q in range(S):
            for k in range(S):
                _check(host[_idx(B, H, S, 0, h, q, k)] == 0.0, "sample0 unmasked")
                var v1 = host[_idx(B, H, S, 1, h, q, k)]
                if k >= 3:
                    _check(v1 < -9999.0, "sample1 tail masked")
                else:
                    _check(v1 == 0.0, "sample1 valid unmasked")

    comptime B1 = 1
    comptime S1 = 4
    var cap_mask = zimage_key_tail_mask_f32[B1, H, S1](2, 2, ctx)
    var cap_host = cap_mask.to_host(ctx)
    for h in range(H):
        for q in range(S1):
            _check(cap_host[_idx(B1, H, S1, 0, h, q, 0)] == 0.0, "cap valid key 0")
            _check(cap_host[_idx(B1, H, S1, 0, h, q, 1)] == 0.0, "cap valid key 1")
            _check(cap_host[_idx(B1, H, S1, 0, h, q, 2)] < -9999.0, "cap tail key 2")
            _check(cap_host[_idx(B1, H, S1, 0, h, q, 3)] < -9999.0, "cap tail key 3")

    print("PASS: zimage B2 key-tail attention masks")
